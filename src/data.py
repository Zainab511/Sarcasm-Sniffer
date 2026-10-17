"""Tiny helpers for loading the processed splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_splits(data_dir: str = "data/processed") -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out = {}
    for name in ("train", "val", "test"):
        p = d / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python -m src.data_prep` first (see README)."
            )
        out[name] = pd.read_csv(p).dropna(subset=["text", "label"])
        out[name]["label"] = out[name]["label"].astype(int)
    return out
