"""Unified inference wrapper used by the API and the explainers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

LABELS = {0: "not sarcastic", 1: "sarcastic"}


class SarcasmPredictor:
    def __init__(self, backend: str = "baseline", model_dir: str = "models"):
        self.backend = backend
        root = Path(model_dir)
        if backend == "baseline":
            import joblib

            path = root / "baseline.joblib"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run `python -m src.train_baseline` first.")
            self._model = joblib.load(path)
        elif backend == "transformer":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = root / "transformer"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing - run `python -m src.train_transformer` first.")
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'baseline' or 'transformer')")

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return array (n, 2): columns = [not sarcastic, sarcastic]."""
        texts = [str(t) for t in texts]
        if self.backend == "baseline":
            return self._model.predict_proba(texts)

        from .preprocess import light_clean

        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [light_clean(t) for t in texts[i:i + batch_size]]
                enc = self._tok(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
                logits = self._model(**enc).logits
                out.append(self._torch.softmax(logits, dim=-1).numpy())
        return np.vstack(out)

    def predict(self, texts: list[str], threshold: float = 0.5) -> list[dict]:
        probs = self.predict_proba(texts)[:, 1]
        results = []
        for p in probs:
            label = int(p >= threshold)
            results.append({
                "label": LABELS[label],
                "sarcasm_probability": round(float(p), 4),
                "confidence": round(float(p if label else 1 - p), 4),
            })
        return results
