"""FastAPI service. Run from the repo root:  uvicorn api.app:app --reload"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __version__
from src.explain import explain_occlusion
from src.predictor import SarcasmPredictor

Backend = Literal["baseline", "transformer"]
MAX_CSV_ROWS = 5000

app = FastAPI(title="Sarcasm Sniffer API", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=2)
def get_predictor(backend: str) -> SarcasmPredictor:
    try:
        return SarcasmPredictor(backend)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Transformer backend needs extra packages: pip install -r requirements-dl.txt",
        )


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    model: Backend = "baseline"
    explain: bool = False
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=500)
    model: Backend = "baseline"
    threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/predict")
def predict(body: PredictIn):
    pred = get_predictor(body.model)
    result = pred.predict([body.text], body.threshold)[0]
    result["model"] = body.model
    if body.explain:
        result["explanation"] = explain_occlusion(pred, body.text)["tokens"]
    return result


@app.post("/predict/batch")
def predict_batch(body: BatchIn):
    results = get_predictor(body.model).predict(body.texts, body.threshold)
    return [{"text": t, **r} for t, r in zip(body.texts, results)]


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), model: Backend = "baseline"):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as CSV.")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")
    col = "text" if "text" in df.columns else df.columns[0]
    texts = df[col].dropna().astype(str).tolist()[:MAX_CSV_ROWS]
    results = get_predictor(model).predict(texts)
    return [{"text": t, **r} for t, r in zip(texts, results)]


_site = Path(__file__).resolve().parent.parent / "website"
if _site.exists():
    app.mount("/", StaticFiles(directory=_site, html=True), name="site")
