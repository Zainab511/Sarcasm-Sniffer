"""Explain a prediction: fast word-occlusion (used by the API) or LIME (CLI)."""
from __future__ import annotations

import argparse

from .predictor import SarcasmPredictor


def explain_occlusion(predictor: SarcasmPredictor, text: str) -> dict:
    """Drop each word in turn; the fall in P(sarcastic) is that word's push toward sarcasm.

    Positive score = word pushes toward sarcastic, negative = toward not sarcastic.
    """
    words = text.split()
    if not words:
        return {"sarcasm_probability": 0.0, "tokens": []}
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    probs = predictor.predict_proba(variants)[:, 1]
    base = float(probs[0])
    return {
        "sarcasm_probability": base,
        "tokens": [{"word": w, "score": round(base - float(p), 4)} for w, p in zip(words, probs[1:])],
    }


def explain_lime(predictor: SarcasmPredictor, text: str, num_features: int = 8, num_samples: int = 500):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["not sarcastic", "sarcastic"], random_state=42)
    exp = explainer.explain_instance(text, predictor.predict_proba, num_features=num_features,
                                     num_samples=num_samples)
    return exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--backend", default="baseline", choices=["baseline", "transformer"])
    ap.add_argument("--method", default="lime", choices=["lime", "occlusion"])
    ap.add_argument("--html", default=None, help="save LIME explanation to this .html file")
    args = ap.parse_args()

    pred = SarcasmPredictor(args.backend)
    print(pred.predict([args.text])[0])
    if args.method == "occlusion":
        for t in sorted(explain_occlusion(pred, args.text)["tokens"], key=lambda t: -abs(t["score"])):
            print(f"{t['word']:>20}  {t['score']:+.3f}")
    else:
        exp = explain_lime(pred, args.text)
        for word, weight in exp.as_list():
            print(f"{word:>20}  {weight:+.3f}")
        if args.html:
            exp.save_to_file(args.html)
            print("saved", args.html)


if __name__ == "__main__":
    main()
