import joblib
import pandas as pd
import emoji
from sklearn.metrics import accuracy_score

tr = pd.read_csv("data/processed/train.csv")
print("\nLabel balance by source (mean = share sarcastic):")
print(tr.groupby("source")["label"].agg(["count", "mean"]))

tr["has_emoji"] = tr["text"].map(lambda t: emoji.emoji_count(str(t)) > 0)
print("\nSarcastic share with / without emoji:")
print(tr.groupby("has_emoji")["label"].agg(["count", "mean"]))

tr["words"] = tr["text"].str.split().str.len()
print("\nAverage length by label:")
print(tr.groupby("label")["words"].mean())

model = joblib.load("models/baseline.joblib")
te = pd.read_csv("data/processed/test.csv")
te["pred"] = model.predict(te["text"].astype(str).tolist())
print("\nTest accuracy by source:")
for src, g in te.groupby("source"):
    print(f"{src:35s} n={len(g):5d}  acc={accuracy_score(g.label, g.pred):.3f}")