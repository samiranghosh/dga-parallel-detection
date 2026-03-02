"""
Real-Time DGA Detection API
==============================
Priority 6 Enhancement — Deployability Demo

A lightweight FastAPI endpoint that accepts domain names and returns
DGA/Benign classification with probabilities, extracted features,
and inference latency.

Usage:
    pip install fastapi uvicorn
    python api.py

    # Or with uvicorn directly:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /classify          — Classify a single domain
    POST /classify/batch    — Classify multiple domains
    GET  /health            — Health check with model info
"""

import os
import time
import pickle
import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import List, Optional
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies not installed. Run:\n"
        "  pip install fastapi uvicorn\n"
    )

from src.features import (
    calc_length, calc_numerical_ratio, calc_meaningful_word_ratio,
    calc_pronounceability, calc_lms_percentage,
    FEATURE_NAMES_5,
)
from src.shared_resources import initialize_shared_resources
from src.classifier import train_random_forest


# ── Pydantic Models ──

class DomainRequest(BaseModel):
    domain: str = Field(..., example="xjk29df", description="Domain name to classify")

class BatchRequest(BaseModel):
    domains: List[str] = Field(..., example=["google", "xjk29df", "facebook"])

class FeatureDetail(BaseModel):
    length: float
    numerical_ratio: float
    meaningful_word_ratio: float
    pronounceability: float
    lms_percentage: float

class ClassifyResponse(BaseModel):
    domain: str
    label: str
    probability: float
    features: FeatureDetail
    latency_ms: float

class BatchResponse(BaseModel):
    results: List[ClassifyResponse]
    total_latency_ms: float
    avg_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model: str
    n_features: int
    feature_names: List[str]
    n_training_samples: int
    accuracy: float


# ── Application ──

app = FastAPI(
    title="DGA Detection API",
    description="Real-time Domain Generation Algorithm malware detection "
                "using parallel linguistic feature extraction and Random Forest.",
    version="1.0.0",
)

# Global state — loaded on startup
_model = None
_dictionary = None
_ngram_table = None
_model_info = {}


@app.on_event("startup")
def load_model():
    """Load shared resources and train (or load cached) model on startup."""
    global _model, _dictionary, _ngram_table, _model_info

    data_path = os.environ.get("DGA_DATA_PATH", "data")
    model_cache = os.path.join(data_path, "rf_model_5feat.pkl")

    print("[API] Loading shared resources...")
    _dictionary, _ngram_table = initialize_shared_resources(data_path)

    if os.path.exists(model_cache):
        print(f"[API] Loading cached model from {model_cache}")
        with open(model_cache, 'rb') as f:
            cache = pickle.load(f)
            _model = cache['model']
            _model_info = cache['info']
    else:
        print("[API] No cached model found. Training from data...")
        import pandas as pd
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')

        if not os.path.exists(train_path):
            raise RuntimeError(
                f"Training data not found at {train_path}. "
                "Run `python main.py --mode preprocess` first."
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Extract features for training data (5-feature config)
        from src.features import extract_all_sequential
        print("[API] Extracting training features (5-feature config)...")
        X_train = extract_all_sequential(
            train_df['domain'].tolist(), _dictionary, _ngram_table,
            skip_levenshtein=True,
        )
        y_train = train_df['label'].values

        X_test = extract_all_sequential(
            test_df['domain'].tolist(), _dictionary, _ngram_table,
            skip_levenshtein=True,
        )
        y_test = test_df['label'].values

        print("[API] Training Random Forest (n_estimators=50)...")
        _model = train_random_forest(X_train, y_train, n_estimators=50)

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, _model.predict(X_test))

        _model_info = {
            'n_features': 5,
            'feature_names': FEATURE_NAMES_5,
            'n_training_samples': len(y_train),
            'accuracy': float(accuracy),
        }

        # Cache for next startup
        with open(model_cache, 'wb') as f:
            pickle.dump({'model': _model, 'info': _model_info}, f)
        print(f"[API] Model cached to {model_cache}")

    print(f"[API] Ready. Model accuracy: {_model_info.get('accuracy', 'N/A')}")


def _extract_5_features(domain: str) -> np.ndarray:
    """Extract 5 linguistic features for a single domain (no Levenshtein)."""
    return np.array([
        calc_length(domain),
        calc_numerical_ratio(domain),
        calc_meaningful_word_ratio(domain, _dictionary),
        calc_pronounceability(domain, _ngram_table),
        calc_lms_percentage(domain, _dictionary),
    ], dtype=np.float64)


@app.post("/classify", response_model=ClassifyResponse)
def classify_domain(request: DomainRequest):
    """Classify a single domain as DGA or Benign."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    domain = request.domain.strip().lower()
    if not domain:
        raise HTTPException(status_code=400, detail="Empty domain")

    t0 = time.perf_counter()
    features = _extract_5_features(domain)
    proba = _model.predict_proba(features.reshape(1, -1))[0]
    latency = (time.perf_counter() - t0) * 1000

    label_idx = int(np.argmax(proba))
    label = "DGA" if label_idx == 1 else "Benign"

    return ClassifyResponse(
        domain=domain,
        label=label,
        probability=float(proba[label_idx]),
        features=FeatureDetail(
            length=features[0],
            numerical_ratio=features[1],
            meaningful_word_ratio=features[2],
            pronounceability=features[3],
            lms_percentage=features[4],
        ),
        latency_ms=round(latency, 3),
    )


@app.post("/classify/batch", response_model=BatchResponse)
def classify_batch(request: BatchRequest):
    """Classify multiple domains in a single request."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.domains) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 domains per batch")

    t0 = time.perf_counter()
    results = []

    for domain in request.domains:
        domain = domain.strip().lower()
        if not domain:
            continue

        t_single = time.perf_counter()
        features = _extract_5_features(domain)
        proba = _model.predict_proba(features.reshape(1, -1))[0]
        latency = (time.perf_counter() - t_single) * 1000

        label_idx = int(np.argmax(proba))
        label = "DGA" if label_idx == 1 else "Benign"

        results.append(ClassifyResponse(
            domain=domain,
            label=label,
            probability=float(proba[label_idx]),
            features=FeatureDetail(
                length=features[0],
                numerical_ratio=features[1],
                meaningful_word_ratio=features[2],
                pronounceability=features[3],
                lms_percentage=features[4],
            ),
            latency_ms=round(latency, 3),
        ))

    total_latency = (time.perf_counter() - t0) * 1000

    return BatchResponse(
        results=results,
        total_latency_ms=round(total_latency, 3),
        avg_latency_ms=round(total_latency / max(len(results), 1), 3),
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Return model info and API health status."""
    return HealthResponse(
        status="healthy" if _model is not None else "loading",
        model="RandomForest (n_estimators=50)",
        n_features=_model_info.get('n_features', 0),
        feature_names=_model_info.get('feature_names', []),
        n_training_samples=_model_info.get('n_training_samples', 0),
        accuracy=_model_info.get('accuracy', 0.0),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
