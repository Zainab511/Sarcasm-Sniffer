"""Fine-tune a transformer for sarcasm detection.

English tweets : --model cardiffnlp/twitter-roberta-base   (default)
Hinglish / multilingual : --model xlm-roberta-base
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from .data import load_splits
from .evaluate import evaluate
from .preprocess import light_clean


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.enc, self.labels = encodings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base")
    ap.add_argument("--out", default="models/transformer")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    d = load_splits(args.data_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def encode(df):
        enc = tok([light_clean(t) for t in df["text"]], truncation=True, max_length=args.max_len)
        return TextDataset(enc, df["label"].values)

    ds = {k: encode(v) for k, v in d.items()}

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(), save_total_limit=1, report_to="none", seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    out = trainer.predict(ds["test"])
    prob = softmax(out.predictions, axis=-1)[:, 1]
    evaluate(d["test"]["label"].values, (prob >= 0.5).astype(int), prob, name="transformer_test")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
