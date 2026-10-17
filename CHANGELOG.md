# Changelog

## v2.0.0
- Added transformer fine-tuning (RoBERTa / XLM-R for Hinglish) in `src/train_transformer.py`
- Replaced the plain LSTM with a BiLSTM + attention model
- Logistic Regression now uses word + char TF-IDF and hand-crafted sarcasm features
- Proper evaluation: precision, recall, F1, ROC-AUC, confusion matrix, cross-validation
- New data pipeline: multi-dataset merge, de-duplication, leakage guard, label-hashtag stripping
- Explainability: word-occlusion (API) and LIME (CLI)
- FastAPI service with single, batch and CSV endpoints; new web demo
- Repo hygiene: requirements, tests, Dockerfile, LICENSE, .gitignore

## v1.0.0 (retroactive tag)
- Logistic Regression (accuracy 0.893) and LSTM (accuracy 0.754) on tweets with emojis and hashtags
