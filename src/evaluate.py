"""One evaluation routine used by every model: P/R/F1, confusion matrix, JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)

LABELS = ["not sarcastic", "sarcastic"]


def evaluate(y_true, y_pred, y_prob=None, name: str = "model", out_dir: str = "reports") -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
