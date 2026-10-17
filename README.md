# Sarcasm Sniffer 🕵️

Sarcasm detection for tweets and short text, built at the intersection of NLP and sentiment analysis.
Normal sentiment tools read "Oh great, another Monday meeting 🙄" as *positive*. Sarcasm Sniffer is built to catch that.

**Current version: v2.0.0** · See [CHANGELOG.md](CHANGELOG.md)

---

## What's new in v2.0

| Area | v1.0 | v2.0 |
|---|---|---|
| Classic model | Logistic Regression | Logistic Regression + word/char TF-IDF + sarcasm features (emoji-vs-text sentiment contrast, caps, punctuation) |
| Deep model | LSTM | BiLSTM + attention **and** fine-tuned RoBERTa (XLM-R for Hinglish) |
| Metrics | Accuracy only | Precision, recall, F1, ROC-AUC, confusion matrix, cross-validation |
| Data | One dataset + scraped tweets | Multi-dataset merge, de-duplication, leakage guard, label-hashtag stripping |
| Explainability | None | Word highlighting (occlusion) in the app, LIME on the CLI |
| Serving | Static website | FastAPI (single / batch / CSV) + new web demo |
| Repo | Notebooks + data | `requirements`, tests, Docker, LICENSE, `.gitignore`, changelog |

## Project structure

```
Sarcasm-Sniffer/
├── src/
│   ├── preprocess.py         # cleaning, emoji -> words, label-hashtag stripping
│   ├── features.py           # hand-crafted sarcasm features
│   ├── data_prep.py          # merge, dedupe, split datasets
│   ├── data.py               # split loader
│   ├── evaluate.py           # shared metrics + confusion matrix
│   ├── train_baseline.py     # Logistic Regression
│   ├── train_bilstm.py       # BiLSTM + attention (TensorFlow/Keras)
│   ├── train_transformer.py  # RoBERTa / XLM-R fine-tuning (PyTorch)
│   ├── predictor.py          # unified inference wrapper
│   └── explain.py            # occlusion + LIME explanations
├── api/app.py                # FastAPI service
├── website/index.html        # demo UI
├── tests/                    # pytest
├── data/{raw,processed}/     # your datasets (git-ignored)
├── models/  reports/         # outputs (git-ignored)
├── requirements.txt          # core + API
├── requirements-dl.txt       # + PyTorch, transformers, TensorFlow
└── Dockerfile
```

## Quickstart

```bash
git clone https://github.com/Zainab511/Sarcasm-Sniffer.git
cd Sarcasm-Sniffer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # baseline + API
pip install -r requirements-dl.txt                    # only for BiLSTM / transformer
```

### 1. Prepare data

Put your CSVs in `data/raw/`, then merge them into clean train/val/test splits:

```bash
python -m src.data_prep \
  --source adyl:data/raw/Train.csv \
  --source adyl:data/raw/Test.csv \
  --source scraped_pos:data/raw/Sarcastic_tweets.csv \
  --source scraped_neg:data/raw/Non_sarcastic_tweets.csv
```

Source kinds: `adyl`, `generic`, `headlines` (News Headlines JSON), `semeval` (TSV), `scraped_pos`, `scraped_neg`.
Text and label columns are auto-detected; use `--text-col` / `--label-col` to override. If a label value is not recognised, the script stops and tells you which one.
Add `--balance` to down-sample to equal class sizes.

### 2. Train

```bash
python -m src.train_baseline                          # fast, CPU, minutes
python -m src.train_baseline --tune                   # grid-search C

python -m src.train_bilstm --glove glove.twitter.27B.100d.txt

python -m src.train_transformer                       # English tweets (cardiffnlp/twitter-roberta-base)
python -m src.train_transformer --model xlm-roberta-base   # Hinglish / multilingual
```

A GPU is strongly recommended for the transformer (Colab free tier works). Metrics, confusion matrices and JSON reports are written to `reports/`.

### 3. Explain a prediction

```bash
python -m src.explain "Oh great, another Monday meeting 🙄" --method lime --html lime.html
python -m src.explain "Oh great, another Monday meeting 🙄" --method occlusion --backend transformer
```

### 4. Run the app

```bash
uvicorn api.app:app --reload
# open http://127.0.0.1:8000  (web demo)   |   http://127.0.0.1:8000/docs  (API docs)
```

## API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "Sure, I love waiting 3 hours 🙄", "model": "baseline", "explain": true}'
```

```json
{
  "label": "sarcastic",
  "sarcasm_probability": 0.91,
  "confidence": 0.91,
  "model": "baseline",
  "explanation": [{"word": "love", "score": 0.12}, ...]
}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status and version |
| `POST /predict` | One text. Optional `explain`, `threshold`, `model` (`baseline` or `transformer`) |
| `POST /predict/batch` | Up to 500 texts as JSON |
| `POST /predict/csv` | Upload a CSV (column `text`, max 5000 rows) |

An explanation score above 0 means the word pushes toward *sarcastic*; below 0, toward *not sarcastic*.

## Results

**v1.0 (original, accuracy only)**

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.893 |
| LSTM | 0.754 |

**v2.0:** run the training scripts, then copy the numbers from `reports/*_metrics.json` into this table. (Do not compare v1 accuracy to v2 F1 directly. Report the same metrics on the same test split.)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression + features | | | | |
| BiLSTM + attention | | | | |
| RoBERTa | | | | |

## Datasets

- **Labelled tweets**: Sarcasm Detection Dataset by Muhammad Adyl (`Train.csv`, `Test.csv`)
- **Real-time tweets**: scraped with Scraper API (`Sarcastic_tweets.csv`, `Non_sarcastic_tweets.csv`)
- **Suggested additions**: News Headlines Sarcasm Dataset, SemEval-2018 Task 3, Reddit SARC, and a Hinglish sarcasm dataset for the multilingual model
- **Embeddings / lexicons**: [GloVe Twitter 100d](https://nlp.stanford.edu/projects/glove/) (BiLSTM only), VADER (sentiment features), SentiWordNet (v1.0)

`src/data_prep.py` re-splits everything itself, so train/test never share a text. When you merge datasets, check that their labels mean the same thing (for example, "irony" is counted as sarcastic by default).

## Deployment

- **Docker (baseline model):** `docker build -t sarcasm-sniffer . && docker run -p 8000:8000 sarcasm-sniffer` (train first so `models/baseline.joblib` exists, or mount `models/`).
- **Hugging Face Spaces / Render:** deploy the Docker image, or run `uvicorn api.app:app --host 0.0.0.0 --port $PORT`.
- If the site is hosted separately from the API, set `window.SNIFFER_API = "https://your-api"` before the script in `website/index.html`.

## Limitations

- Sarcasm depends on context, tone and shared knowledge. A short text alone is often ambiguous, even for humans.
- Models trained on Twitter data may not transfer to other domains (Reddit, reviews, speech).
- Datasets built from hashtags such as `#sarcasm` are noisy and can encode cultural bias. Treat outputs as probabilities, not facts. Don't use them to judge individuals.

## Roadmap

- Conversation context (parent tweet / thread) as extra input
- Hinglish training data and evaluation
- Model cards and calibration plots
- Hosted public demo

## Testing

```bash
pytest
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgements

Muhammad Adyl for the dataset, Stanford NLP for GloVe, the authors of SentiWordNet, VADER and LIME, and the Hugging Face and CardiffNLP teams.
