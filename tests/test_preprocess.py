from src.features import emoji_counts, emoji_score
from src.preprocess import clean_text, light_clean, strip_label_hashtags


def test_label_hashtags_removed_but_others_kept():
    assert strip_label_hashtags("Great job #sarcasm #Monday") == "Great job #Monday"


def test_light_clean_masks_users_and_urls():
    out = light_clean("RT @bob look https://x.com/a now")
    assert "@user" in out and "http" in out and "bob" not in out and "RT" not in out


def test_clean_text_turns_emoji_into_words_and_lowercases():
    out = clean_text("Wow GREAT 🙄 #NoThanks")
    assert "emoji_" in out and out == out.lower() and "no thanks" in out


def test_emoji_scoring():
    assert emoji_score("so happy 😂😂") > 0
    assert emoji_score("so sad 😭") < 0
    assert emoji_score("no emoji here") == 0
    assert emoji_counts("ok 🙄")[2] == 1
