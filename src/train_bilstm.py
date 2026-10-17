"""BiLSTM + attention (upgrade of the v1 LSTM). Needs TensorFlow >= 2.16 / Keras 3."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import keras
import numpy as np
from keras import layers

from .data import load_splits
from .evaluate import evaluate
from .preprocess import clean_text


@keras.saving.register_keras_serializable(package="sniffer")
class AttentionPooling(layers.Layer):
    """Learned weighted average over time steps (mask-aware)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.w = self.add_weight(name="att_w", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_b", shape=(1,), initializer="zeros")

    def call(self, x, mask=None):
        scores = keras.ops.matmul(keras.ops.tanh(x), self.w) + self.b  # (B, T, 1)
        if mask is not None:
            m = keras.ops.expand_dims(keras.ops.cast(mask, "float32"), -1)
            scores = scores + (1.0 - m) * -1e9
        weights = keras.ops.softmax(scores, axis=1)
        return keras.ops.sum(x * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def prep(text: str) -> str:
    return re.sub(r"([!?.,])", r" \1 ", clean_text(text))


def load_glove(path: str, vocab: list[str], dim: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 0.05, (len(vocab), dim)).astype("float32")
    emb[0] = 0.0
    index = {w: i for i, w in enumerate(vocab)}
    hits = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == dim + 1 and parts[0] in index:
                emb[index[parts[0]]] = np.asarray(parts[1:], dtype="float32")
                hits += 1
    print(f"GloVe coverage: {hits}/{len(vocab)} words")
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--glove", default=None, help="path to glove.twitter.27B.100d.txt")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--max-vocab", type=int, default=30000)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="models/bilstm.keras")
    args = ap.parse_args()
    keras.utils.set_random_seed(42)

    d = load_splits(args.data_dir)
    X = {k: np.array([prep(t) for t in v["text"]], dtype=object)[:, None] for k, v in d.items()}
    y = {k: v["label"].values.astype("float32") for k, v in d.items()}

    vec = layers.TextVectorization(max_tokens=args.max_vocab, output_sequence_length=args.max_len,
                                   standardize="lower", split="whitespace")
    vec.adapt(X["train"][:, 0])
    vocab = vec.get_vocabulary()
    init = (keras.initializers.Constant(load_glove(args.glove, vocab, args.dim))
            if args.glove else "uniform")

    inp = keras.Input(shape=(1,), dtype="string")
    x = vec(inp)
    x = layers.Embedding(len(vocab), args.dim, embeddings_initializer=init, mask_zero=True)(x)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = AttentionPooling()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    model.fit(
        X["train"], y["train"], validation_data=(X["val"], y["val"]),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )

    prob = model.predict(X["test"], verbose=0).ravel()
    evaluate(y["test"], (prob >= 0.5).astype(int), prob, name="bilstm_test")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
