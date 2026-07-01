"""
RSS-Breakdown Harness  (Batch 1.4 — Shared Primitive #2)
========================================================
Measures where *total resident memory* (RSS) of the serving process actually
goes, staged cumulatively in a single process:

    interpreter baseline
      -> + import numpy / sklearn
      -> + load English dictionary (set)
      -> + load trigram table
      -> + load model (pickle)
      -> + first inference        (working set / peak)

plus idle / loaded / peak summary values.

Why this primitive exists:
- RQ2's target is shrinking *total resident memory of the serving process*, NOT
  the model pickle. AGENTS.md is explicit that the NLTK dictionary + trigram table
  + Python/sklearn/numpy runtime dominate RSS while the model pickle is tiny.
  This harness proves that decomposition with numbers instead of asserting it.
- It is the before/after primitive for the Aho-Corasick/DAWG kernel (Batch 3,
  serves C1/C2/C4) and for the RQ2 pruning/ONNX variants (C4).

CRITICAL: heavy modules (numpy, sklearn, src.features, src.shared_resources) are
imported *lazily inside stages* so the interpreter-baseline RSS is measured before
they inflate it. Run this as its own process to keep the baseline clean:

    python -m src.rss_harness --data-path data/ --model results/baseline_model_5feat.pkl
"""

import os
import json
import pickle
import argparse
import logging

import psutil  # measurement tool; part of the baseline cost by design

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROC = psutil.Process(os.getpid())


def _rss_mb() -> float:
    """Current resident set size of this process, in MiB."""
    return _PROC.memory_info().rss / (1024 ** 2)


def run(data_path: str = "data/", model_path: str = None,
        sample_domain: str = "googleadservices") -> dict:
    """Run the staged RSS breakdown in-process and return the measurements.

    Args:
        data_path: dir with english_dictionary.txt and ngram_table.pkl.
        model_path: path to a model pickle. Required for an honest 'load model'
            delta (training in-process would pollute the measurement).
        sample_domain: one domain used to trigger the first inference.

    Returns:
        Dict with per-stage cumulative RSS (MiB), incremental deltas, and an
        idle/loaded/peak summary.
    """
    stages = []

    def mark(label: str):
        stages.append((label, _rss_mb()))

    # Stage 0 — interpreter baseline (only os/json/pickle/psutil imported so far).
    mark("interpreter")

    # Stage 1 — import numpy + sklearn runtime.
    import numpy  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    mark("import_numpy_sklearn")

    # Stage 2 — load English dictionary as a set (the baseline dictionary form).
    dict_path = os.path.join(data_path, "english_dictionary.txt")
    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = set(line.strip() for line in f if line.strip())
    mark("load_dictionary")

    # Stage 3 — load trigram probability table.
    with open(os.path.join(data_path, "ngram_table.pkl"), "rb") as f:
        ngram_table = pickle.load(f)
    mark("load_trigram_table")

    # Stage 4 — load model (pickle). Train only as a fallback, flagged in output.
    trained_fallback = False
    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        trained_fallback = True
        logger.warning("No model pickle at %r — training a small RF in-process "
                       "(inflates the load_model delta).", model_path)
        import pandas as pd
        from src.features import extract_all_sequential
        from src.classifier import train_random_forest
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        Xtr = extract_all_sequential(train_df["domain"].astype(str).tolist()[:20000],
                                     dictionary, ngram_table, skip_levenshtein=True)
        model = train_random_forest(Xtr, train_df["label"].values[:20000], n_estimators=50)
    mark("load_model")

    # Stage 5 — first inference (materialises numpy/predict working set = peak).
    from src.features import extract_features
    feats = extract_features(sample_domain, sample_domain, dictionary, ngram_table,
                             skip_levenshtein=True)
    model.predict(feats.reshape(1, -1))
    mark("first_inference")

    # Build cumulative + incremental report.
    cumulative = {label: round(rss, 2) for label, rss in stages}
    deltas = {}
    for i in range(1, len(stages)):
        prev_label, prev_rss = stages[i - 1]
        label, rss = stages[i]
        deltas[label] = round(rss - prev_rss, 2)

    idle = stages[0][1]
    loaded = cumulative["load_model"]
    peak = max(rss for _, rss in stages)

    result = {
        "config": {
            "data_path": data_path,
            "model_path": model_path,
            "trained_fallback": trained_fallback,
            "n_dictionary_words": len(dictionary),
            "n_trigrams": len(ngram_table),
        },
        "cumulative_rss_mb": cumulative,
        "incremental_delta_mb": deltas,
        "summary_mb": {
            "idle": round(idle, 2),
            "loaded": round(loaded, 2),
            "peak": round(peak, 2),
        },
    }
    logger.info("RSS idle=%.1f MiB  loaded=%.1f MiB  peak=%.1f MiB",
                idle, loaded, peak)
    logger.info("Dominant deltas: dict=%.1f  trigram=%.1f  model=%.1f  numpy/sklearn=%.1f",
                deltas.get("load_dictionary", 0), deltas.get("load_trigram_table", 0),
                deltas.get("load_model", 0), deltas.get("import_numpy_sklearn", 0))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RSS-breakdown harness (Batch 1.4)")
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--model", default=None, help="Path to a model pickle")
    ap.add_argument("--output", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    res = run(args.data_path, args.model)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        logger.info("Wrote %s", args.output)
    else:
        print(json.dumps(res, indent=2))
