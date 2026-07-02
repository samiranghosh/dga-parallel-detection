# CLAUDE.md — DGA Parallel Detection (Dissertation Fork)

Forks the Group-09 coursework baseline. Dissertation:
*"Adaptive Parallel Feature Extraction and Model Compression for Real-Time DGA Detection on Resource-Constrained Platforms."*
Three decoupled research questions extend a static batch parallel pipeline:
- **RQ1** Adaptive parallelism — queue-fed worker processes, min–max band with hysteresis + cooldown.
- **RQ2** Model compression — shrink *total resident memory* of the serving process (structural pruning, ONNX runtime), not just the pickle.
- **RQ3** Edge verification — Docker cgroup profiles (≤2 cores / ≤512 MB); real ARM validation is the open gap.

## Critical rules
- **YOU MUST** keep `tests/test_parallel.py` green. It asserts parallel feature extraction == sequential output. It is the correctness gate; never merge with it failing or weakened.
- **YOU MUST** preserve baseline accuracy (see Baseline). Accuracy is preserved, not chased — the field is saturated.
- **IMPORTANT** Do not claim novelty on rejected directions (see Rejected). They are settled.
- Contributions are independent — do not let RQ work cross-block. Branch per RQ.

## Layout (flat `src/`)
- `main.py` — CLI entry (`--mode preprocess` / `--mode benchmark`)
- `api.py` — FastAPI serving endpoint (port 8000, model from pickle)
- `src/parallel_engine.py` `chunker.py` `shared_resources.py` — parallel core (RQ1 lives here)
- `src/features.py` — 5 production features (RQ2/RQ3 latency bottleneck lives here)
- `src/classifier.py` `preprocess.py` `benchmark.py` `fault_handler.py`
- Instrumentation (Batch 1): `src/latency_harness.py` (single-request p50/p95/p99) · `src/rss_harness.py` (staged RSS) · `src/golden_snapshot.py` + `tests/golden_features.json` (A2 oracle, 20 domains — expand before Batch 5)
- RQ1 (Batch 2): `src/benchmark_adaptive.py` · `src/load_harness.py` + `src/load_profiles.py` (uniform/Poisson/ramp, ρ-sweep) · `tests/test_adaptive.py`
- RQ2 groundwork (pre-existing): `src/compact_dict.py` (marisa-trie DAWG — already a hard runtime dep via `preprocess.py`) · `src/onnx_model.py` + `tests/test_onnx.py` (failing, known issue)
- `tests/test_parallel.py` (gate) + `test_boundary.py` `test_features.py` `test_fault_handler.py` `test_compact_dict.py`
- `results/metrics.json` — raw baseline · `results/rq1/FINDINGS.md` — RQ1 C3 report · `P3_Experimental_Report.ipynb`

## Commands
```bash
pytest tests/ -v                                              # gate first
python main.py --mode preprocess                             # data prep
python main.py --mode benchmark --output-dir results/ --repetitions 3
uvicorn api:app --host 0.0.0.0 --port 8000                   # serve (needs fastapi/uvicorn)
bash reproduce.sh [--quick]                                  # full pipeline + report
```

## Environment
- Python **3.11.x**. Deps in `requirements.txt`.
- `fastapi==0.135.1` + `uvicorn==0.41.0` are pinned in `requirements.txt` for `api.py`.
- **Env locked & tagged `baseline-verified`** (commit `929f3a2`): `requirements.txt` pins **sklearn 1.8.0 / numpy 2.4.6 / pandas 2.2.3**, which reproduce `results/metrics.json` bit-exact. (metrics.json was *originally* produced under numpy 2.4.2 / pandas 3.0.1 per its `system_info`; the held-back numpy/pandas were verified not to change feature values or accuracy.)

## Features (5 production + 1 dropped)
1. `length` — O(1)
2. `numerical_ratio` — O(m)
3. `meaningful_word_ratio` — dictionary, **O(m²)**
4. `pronounceability` — trigram, O(m)
5. `lms_percentage` (longest meaningful substring) — dictionary, **O(m²)**
- Dropped: `levenshtein` — hurt accuracy in E7 ablation. `python-Levenshtein` is still in `requirements.txt`.
- **Bottleneck (MEASURED, baseline-verified — corrects the outline):** single-request latency and RSS are **model-dominated, not feature-dominated**. Single-request predict = 99% of ~4.6 ms (n_jobs=1) / ~17 ms (n_jobs=-1); feature extraction is ~145 µs (<1%). The sklearn *serving path* (validation + joblib), not tree compute, is the cost. Model = 473 of 657 MiB RSS; dictionary = 23 MiB.
  - **RQ3 single-request lever** = leave sklearn's predict path (ONNX/treelite + n_jobs=1) + pruning. AC/DAWG does **not** move this metric.
  - **RQ2 memory lever** = pruning + float32 + native/ONNX-only runtime. Dictionary/DAWG is minor (23 MiB).
  - **AC/DAWG (Batch 5, conditional)** helps **batch throughput only** (76 µs/dom, feature-bound) + ~23 MiB, with a clean provably-identical-output story.

## Baseline to preserve (tag `baseline-verified`; env sklearn 1.8.0)
- Accuracy **93.179%** (5-feature) / **92.602%** (6-feature), E7 bit-exact · Speedup **7.13×** @ 8 cores · Throughput **78,800 dom/s** · IPC **1.32%**.
- Measured serving (RF-100): single-request p50/p95/p99 = **17.4 / 27.3 / 29.5 ms** (n_jobs=-1); n_jobs=1 = **4.6 ms**. Peak RSS **656.8 MiB** (interpreter 22 · numpy+sklearn 133 · dict 23 · trigram 4 · **model 473**).
- **Peak RSS 657 MiB > every cgroup profile (≤512 MB).** Model compression is a **prerequisite for RQ3 to run at all**, not just to hit a number. 256 MB likely needs an ONNX-only / treelite image (sklearn floor alone ≈133 MiB).
- `n_estimators=50` sufficient (~92.6%, ~9× faster than 500 trees). Pronounceability is the top feature (+25.8%).
- **Env trap:** sklearn **1.8.0 required** for bit-exact reproduction — 1.6.x builds the RF differently on pronounceability and drops 5-feature accuracy to 92.24%. Never re-pin without re-running E7.
- Baseline profiled on throttled **x86 via cgroups** — never on real ARM. That gap is the primary viva risk.

## RQ1 result (Batch 2, MEASURED — `results/rq1/FINDINGS.md`)
- Streaming queue-fed engine is **IPC-bound**: sequential (~14.8k dom/s) > static-8 (~8.4k) > adaptive (~5.6k). 65 µs/domain feature work can't amortise per-batch pickle/Queue cost. The 78.8k dom/s figure is **batch `Pool.map`** — that remains the throughput story.
- Adaptive's demonstrated value: **per-worker efficiency** (~1,250 vs ~1,050 dom/s/worker), **graceful saturation** (no collapse at 2× peak, ratios 0.99–1.01), **zero oscillation** (hysteresis+cooldown verified), fault recovery.
- ≤2-core edge: adaptive < static-2 < sequential — **small/negative, valid bounded finding**.
- Claim RQ1 as a **resource-efficiency + stability** contribution, never a streaming-throughput one.

## Known issues
- `test_onnx` **FIXED** (B3 Step 2): root cause was an **onnxruntime ≥1.26.0 TreeEnsemble kernel regression** (binary-RF probas emitted as (−p₁,+p₁), corrupting in-graph labels). Shipped: zipmap=False + **ort pinned 1.25.1**. Never bump onnxruntime without re-running `test_onnx` + a real-model parity check; the pin is load-bearing (re-verify wheel+parity on aarch64 in Batch 6).
- **Windows spawn cost** (~0.3–0.5 s/worker + dict unpickle) inflates parallel startup in RQ1 numbers; re-check on Linux `fork` / ARM in Batch 6.
- A1-adaptive equality + A5 end-to-end (kill/overload) tests were gaps in Batch 2 — closed in Batch-3 Step 0; keep them green.

## Batch-3 spike result (MEASURED — `results/spike/FINDINGS.md`)
- Serving overhead, not tree compute: sklearn n_jobs=1 predict ≈5.3 ms vs traversal floor 408 µs vs **ONNX 18 µs** (parity exact). `n_jobs=1` locked at serving (30.3→5.5 ms, free).
- **ONNX on the unpruned RF is undeployable:** session init 15–32 min, session RSS 990 MiB. Runtime floor **48.8 MiB** → a 256 MB image is feasible only with a compressed model.
- **DT-12 ≥ baseline:** 93.47% holdout vs 93.18%, 35.5 µs, 227 KB pickle, 0.6 MiB RSS. LR floor 89.9%. 3-feature drop costs −2.3 pp → dictionary stays.
- **DECISIONS (SamG, 02 Jul):** (1) **distill-primary, conditional** — DT-12 promotes only through 4 Batch-4 gates (McNemar · FPR/FNR parity · per-family incl. PCFG · cross-dataset chrmor); pruned-RF+ONNX = documented fallback; C4 curve on {RF-100, RF-pruned, DT-12, LR} regardless. (2) **AC/DAWG survives → Batch 5 re-promoted** (under DT-12, feature extraction is the single-request bottleneck again). Profile C (1c/256 MB) flips feasible (~183 MiB sklearn-path / ~50 MiB ONNX-only), pending B6 measurement.

## Rejected — do not relitigate
- Shannon entropy as a novelty claim (foundational; off-the-shelf).
- QEMU for latency/throughput numbers (functional emulation, not cycle-accurate; OK only for functional/portability parity).
- Beating detection accuracy (saturated field).
- Multi-feature score fusion as novel (already the RF probability output).
- CFG-compliance as a *detector* (collapses into n-gram methods; CFGs are the attacker's tool). Probabilistic-CFG DGAs are a hard family to *test against*, not to build.

## Planning
- Live plan: `docs/IMPLEMENTATION_PLAN.md` (gated batches; read on demand — **not** imported, plans change too often to live in memory) + the thread tracker (T1–T7).
- Highest-value levers, **post-spike (decisions locked 02 Jul)**: (1) **DT-12 distill-primary** through the 4 Batch-4 promotion gates (pruned-RF+ONNX fallback); (2) **T1** real ARM validation (publishability); (3) **AC/DAWG re-promoted** — single-request latency *and* batch throughput (bottleneck migrates back to features under DT-12). 3-feature drop **rejected** (−2.3 pp, probe 3).

## Statistical rigor
- Accuracy: stratified CV + paired t-tests. Latency/throughput: mean ± SD with CIs over repetitions.
