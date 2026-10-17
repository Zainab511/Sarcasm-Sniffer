"""Merge, clean, de-duplicate and split sarcasm datasets.

Examples
--------
python -m src.data_prep \
    --source adyl:data/raw/Train.csv --source adyl:data/raw/Test.csv \
    --source scraped_pos:data/raw/Sarcastic_tweets.csv \
    --source scraped_neg:data/raw/Non_sarcastic_tweets.csv \
    --source headlines:data/raw/Sarcasm_Headlines_Dataset.json

Source kinds: adyl | generic | headlines | semeval | scraped_pos | scraped_neg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import clean_text, light_clean

TEXT_COLS = ["text", "tweet", "tweets", "content", "headline", "tweet text", "comment", "body"]
LABEL_COLS = ["label", "class", "is_sarcastic", "sarcastic", "target"]
POS = {"1", "true", "yes", "sarcasm", "sarcastic", "irony", "ironic", "figurative"}
NEG = {"0", "false", "no", "regular", "normal", "notsarcastic", "not_sarcastic",
       "not sarcastic", "non-sarcastic", "nonsarcastic", "not_sarcasm"}


def _find_col(df: pd.DataFrame, candidates: list[str], override: str | None) -> str:
    if override:
        return override
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"Could not find a column among {candidates}. Columns: {list(df.columns)}")


def _norm_label(v) -> int | None:
    s = str(v).strip().lower()
    try:
        s = str(int(float(s)))
    except ValueError:
        pass
    if s in POS:
        return 1
    if s in NEG:
        return 0
    return None


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".json":
        return pd.read_json(p, lines=True)
    if p.suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, encoding_errors="replace")


def load_source(kind: str, path: str, text_col=None, label_col=None) -> pd.DataFrame:
    df = _read_table(path)
    tcol = _find_col(df, TEXT_COLS, text_col)
    if kind in {"scraped_pos", "scraped_neg"}:
        labels = pd.Series([1 if kind == "scraped_pos" else 0] * len(df))
    else:
        lcol = _find_col(df, LABEL_COLS, label_col)
        mapped = df[lcol].map(_norm_label)
        unknown = sorted(set(df.loc[mapped.isna(), lcol].astype(str)))
        if unknown:
            raise ValueError(
                f"{path}: unknown label values {unknown[:10]}. "
                "Add them to POS/NEG in src/data_prep.py or pass --label-col."
            )
        labels = mapped
    out = pd.DataFrame({"text": df[tcol].astype(str), "label": labels.astype(int).values})
    out["source"] = Path(path).name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="kind:path (repeatable)")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--balance", action="store_true", help="down-sample to equal class sizes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = []
    for s in args.source:
        kind, _, path = s.partition(":")
        frames.append(load_source(kind, path, args.text_col, args.label_col))
        print(f"loaded {len(frames[-1]):>7} rows from {path} ({kind})")
    df = pd.concat(frames, ignore_index=True)

    df["text"] = df["text"].map(light_clean)
    df = df[df["text"].str.split().str.len() >= args.min_words]

    # De-duplicate on a normalised key; drop texts that appear with conflicting labels.
    df["key"] = df["text"].map(clean_text)
    conflicting = df.groupby("key")["label"].nunique()
    conflicting = conflicting[conflicting > 1].index
    df = df[~df["key"].isin(conflicting)].drop_duplicates("key")
    print(f"after cleaning/dedup: {len(df)} rows ({len(conflicting)} conflicting texts dropped)")

    if args.balance:
        n = df["label"].value_counts().min()
        df = df.groupby("label", group_keys=False).sample(n=n, random_state=args.seed)
        print(f"balanced to {n} per class")

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=args.seed)

    # Leakage guard: no text may appear in two splits.
    assert not (set(train.key) & set(test.key)) and not (set(train.key) & set(val.key))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part[["text", "label", "source"]].to_csv(out / f"{name}.csv", index=False)
        share = part["label"].mean()
        print(f"{name:>5}: {len(part):>7} rows | sarcastic share {share:.1%}")


if __name__ == "__main__":
    main()
