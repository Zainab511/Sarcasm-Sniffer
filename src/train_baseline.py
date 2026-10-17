"""Logistic Regression baseline: word + char TF-IDF + hand-crafted features."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import load_splits
from .evaluate import evaluate
from .features import HandFeatures
from .preprocess import clean_many


def build_pipeline(C: float = 2.0) -> Pipeline:
    features = FeatureUnion([
        ("word", Pipeline([
            ("clean", FunctionTransformer(clean_many)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ])),
        ("char", Pipeline([
            ("clean", FunctionTransformer(clean_many)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                      sublinear_tf=True, max_features=200_000)),
        ])),
        ("hand", Pipeline([("feats", HandFeatures()), ("scale", StandardScaler())])),
    ])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="models/baseline.joblib")
    ap.add_argument("--tune", action="store_true", help="grid-search C")
    ap.add_argument("--cv", type=int, default=5, help="CV folds on train (0 to skip)")
    args = ap.parse_args()

    d = load_splits(args.data_dir)
    X_tr, y_tr = d["train"]["text"].tolist(), d["train"]["label"].values

    pipe = build_pipeline()
    if args.tune:
        search = GridSearchCV(pipe, {"clf__C": [0.5, 1, 2, 4]}, scoring="f1", cv=3, n_jobs=-1)
        search.fit(X_tr, y_tr)
        print("best C:", search.best_params_["clf__C"])
        pipe = search.best_estimator_
    else:
        if args.cv:
            scores = cross_val_score(pipe, X_tr, y_tr, cv=args.cv, scoring="f1", n_jobs=-1)
            print(f"{args.cv}-fold CV F1: {scores.mean():.4f} +/- {scores.std():.4f}")
        pipe.fit(X_tr, y_tr)

    for split in ("val", "test"):
        X, y = d[split]["text"].tolist(), d[split]["label"].values
        prob = pipe.predict_proba(X)[:, 1]
        evaluate(y, (prob >= 0.5).astype(int), prob, name=f"baseline_{split}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
