"""Text cleaning shared by every model in Sarcasm Sniffer."""
from __future__ import annotations

import re
from typing import Iterable

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"^\s*RT\s+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Hashtags that give the label away. If we keep them the model just learns
# to look for "#sarcasm" instead of learning sarcasm.
LABEL_HASHTAGS = {
    "sarcasm", "sarcastic", "sarcasmalert", "irony", "ironic",
    "not", "jk", "justkidding", "yeahright",
}


def extract_hashtags(text: str) -> list[str]:
    return [h.lower() for h in HASHTAG_RE.findall(str(text))]


def strip_label_hashtags(text: str) -> str:
    def _sub(m: re.Match) -> str:
        return "" if m.group(1).lower() in LABEL_HASHTAGS else m.group(0)

    return SPACE_RE.sub(" ", HASHTAG_RE.sub(_sub, str(text))).strip()


def light_clean(text: str) -> str:
    """Minimal cleaning for transformers (they understand emojis and case)."""
    text = str(text)
    text = RT_RE.sub("", text)
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    text = strip_label_hashtags(text)
    return SPACE_RE.sub(" ", text).strip()


def _split_hashtag(m: re.Match) -> str:
    return " " + CAMEL_RE.sub(" ", m.group(1)) + " "


def clean_text(text: str) -> str:
    """Heavier cleaning for classic models: emojis become words, lowercase."""
    text = light_clean(text)
    text = HASHTAG_RE.sub(_split_hashtag, text)
    text = emoji.demojize(text, delimiters=(" emoji_", " "))
    text = re.sub(r"@user\b", " usertag ", text)
    text = re.sub(r"\bhttp\b", " urltag ", text)
    return SPACE_RE.sub(" ", text).strip().lower()


def clean_many(texts: Iterable[str]) -> list[str]:
    return [clean_text(t) for t in texts]
