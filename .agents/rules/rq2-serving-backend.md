---
trigger: glob
globs: api.py, classifier.py, **/predict*.py, **/serve*.py, **/inference*.py
description: Serving / model backend architecture for RQ2 (sklearn vs ONNX).
---

# RQ2 serving backend (sklearn vs ONNX)

## Core decision
Use a **pluggable predictor backend selected at startup by config/env** — both paths must exist for benchmarking, but the deployed edge artifact is **ONNX-only**. Do **not** do per-request runtime switching.

## Rules
- Define a `Predictor` interface with `SklearnPredictor` and `OnnxPredictor` implementations, chosen **once at startup** from a config/env var (e.g. `BACKEND=onnx`).
- **Use lazy imports.** The ONNX path must **never** import `sklearn`, `scipy`, or `joblib`. Dropping that import stack is the entire RQ2 memory win — if sklearn is importable "for flexibility," it loads and the win evaporates.
- **Benchmark harness** (dev box, memory irrelevant): may construct both backends for A/B comparison.
- **Edge container** (the RSS reported under cgroups): `BACKEND=onnx` only, and `sklearn`/`scipy` **must not be installed in that image** — so it can be *proven* absent from RSS.
- **Never keep both backends loaded in the same serving process** — that is the worst case for memory.

## Required tests / measurements
- **Accuracy-parity test:** assert ONNX predictions match sklearn predictions within tolerance on a fixed sample set. `skl2onnx` conversion can shift predictions slightly; quantify and report any accuracy delta (this is a near-certain viva question).
- **Measure `onnxruntime`'s own RSS.** The runtime is lighter than sklearn, not free — report it in the footprint breakdown.

## Do not
- Do not add a backend toggle that imports both stacks.
- Do not report an RQ2 memory number from an image that still has sklearn installed.
