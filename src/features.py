"""Hand-crafted sarcasm features: punctuation, caps, emoji, sentiment contrast."""
from __future__ import annotations

import re

import emoji
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import HASHTAG_RE, MENTION_RE, URL_RE, strip_label_hashtags

_VS16 = {"\ufe0f"}
POS_EMOJI = set("😀😃😄😁😆😊🙂😍🥰😘😂🤣😎👍👏🎉❤💕🔥✨🙌💯") - _VS16
NEG_EMOJI = set("😢😭😡😠🤬😞😔😩😫👎💔😤😱🤢") - _VS16
SARC_EMOJI = set("🙄😒🙃😏🤡😬🤦") - _VS16

FEATURE_NAMES = [
    "n_exclaim", "n_question", "n_ellipsis", "n_caps_words", "caps_ratio",
    "n_quotes", "n_hashtags", "n_mentions", "n_urls", "n_chars", "n_words",
    "n_emoji", "emoji_score", "n_sarcasm_emoji",
    "text_sentiment", "sentiment_contrast", "text_emoji_mismatch",
]

_vader = None


def _get_vader():
    global _vader
    if _vader is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _vader = SentimentIntensityAnalyzer()
    return _vader


def emoji_counts(text: str) -> tuple[int, int, int, int]:
    """Return (positive, negative, sarcasm-prone, total) emoji counts."""
    pos = sum(ch in POS_EMOJI for ch in text)
    neg = sum(ch in NEG_EMOJI for ch in text)
    sarc = sum(ch in SARC_EMOJI for ch in text)
    return pos, neg, sarc, emoji.emoji_count(text)


def emoji_score(text: str) -> float:
    pos, neg, _, total = emoji_counts(text)
    return (pos - neg) / total if total else 0.0


def extract_features(text: str) -> list[float]:
    raw = strip_label_hashtags(str(text))
    words = re.findall(r"[A-Za-z']+", raw)
    caps = [w for w in words if len(w) >= 2 and w.isupper()]

    pos, neg, sarc, n_emo = emoji_counts(raw)
    e_score = emoji_score(raw)
    t_sent = _get_vader().polarity_scores(emoji.replace_emoji(raw, ""))["compound"]
    contrast = abs(t_sent - e_score) if n_emo else 0.0
    mismatch = float(n_emo > 0 and ((t_sent > 0.3 and e_score < 0) or (t_sent < -0.3 and e_score > 0)))

    return [
        raw.count("!"), raw.count("?"), raw.count("..."),
        len(caps), len(caps) / max(len(words), 1),
        raw.count('"'), len(HASHTAG_RE.findall(raw)), len(MENTION_RE.findall(raw)),
        len(URL_RE.findall(raw)), len(raw), len(words),
        n_emo, e_score, sarc,
        t_sent, contrast, mismatch,
    ]


class HandFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: list[str] -> (n, len(FEATURE_NAMES)) float array."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([extract_features(t) for t in X], dtype="float64")
