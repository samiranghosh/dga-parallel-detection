---
description: Run the RQ1 adaptive-vs-static load benchmark and the project's measurement protocol the correct way (utilization sweep, not batch rate).
---

# /benchmark

When the user runs `/benchmark`, execute the following. Pause for the user to review and run each measurement step (review-driven); do not auto-continue through the whole sequence.

## 1. Reproducibility header
- Record host CPU model, physical core count, total RAM, OS (Windows / WSL2 / Docker), Python version, and pinned versions of numpy, scikit-learn, onnxruntime. Write this to the top of the results file.

## 2. Measure saturation `μ`
- Measure the engine's **streaming** saturation throughput on **this host** using per-item dispatch through the queue (NOT batch / `Pool.map` over a preloaded list). Print `μ` in domains/s.
- Do not reuse any throughput number from another machine. The ~78,800/s figure is batch throughput and must never be used as an arrival rate.

## 3. Uniform profile — utilization sweep
- For `ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.05}`, set arrival rate `λ = ρ · μ`.
- At each `ρ`, run **both** the adaptive engine **and** static K=8 on identical generated traffic.
- Capture per run: throughput, p50 / p95 / p99 single-request latency, **mean active worker count over time**, and monitoring + spawn/teardown overhead.

## 4. Other traffic profiles
- Repeat for **bursty-Poisson** and **ramp** arrival profiles at a representative `ρ` (e.g. 0.6).

## 5. Repetition & statistics
- Run each configuration **≥5 times**. Report **mean ± SD with confidence intervals**.
- Keep **single-request latency** and **amortised batch throughput** reported separately and clearly labelled.

## 6. Footprint (when the run targets RQ2/RQ3)
- Emit an **RSS breakdown** — dictionary / trigram table / runtime / model — **before vs after** the change.
- Measure **idle RSS before any data loads.**
- If running under Docker, measure RSS **inside** the container under the 2c/512 MB and 1c/256 MB cgroup profiles.

## 7. Output
- Save all raw results to a **timestamped file** (e.g. `results/bench_<date>.json`).
- Produce the `ρ`-vs-(mean worker count) and `ρ`-vs-(p95 latency) **curves**.
- **Do not** write conclusions or headline numbers into any report/markdown prose. Output the data only; the human interprets and records it.
- Never fabricate, interpolate, or "fill in" a missing data point — if a run failed, report the failure.
