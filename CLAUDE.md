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
- Instrumentation (Batch 1): `src/latency_harness.py` (single-request p50/p95/p99) · `src/rss_harness.py` (staged RSS) · `src/golden_snapshot.py` + `tests/golden_features.json` (v1, 20 domains) · **`tests/golden_features_v2.json` (B5 A2 oracle: 1,049 domains incl. A4 edges; generator `scripts/kernel/make_golden_v2.py`)**
- RQ1 (Batch 2): `src/benchmark_adaptive.py` · `src/load_harness.py` + `src/load_profiles.py` (uniform/Poisson/ramp, ρ-sweep) · `tests/test_adaptive.py`
- RQ2 groundwork (pre-existing): `src/compact_dict.py` (marisa-trie DAWG — already a hard runtime dep via `preprocess.py`) · `src/onnx_model.py` + `tests/test_onnx.py` (failing, known issue)
- `tests/test_parallel.py` (gate) + `test_boundary.py` `test_features.py` `test_fault_handler.py` `test_compact_dict.py`
- `results/metrics.json` — raw baseline · `results/rq1/FINDINGS.md` — RQ1 C3 report · `results/spike/` — Batch-3 probes · `results/rq2/` — Batch-4 gates/C4 (FINDINGS, INTERIM, gate JSONs, `models/manifest.json`) · `results/kernel/` — Batch-5 AC-kernel (FINDINGS + bench/A2/step5 JSONs) · `results/wsl2/` — E1 Linux-fork re-run · `scripts/{spike,rq2,kernel}/` — probe/gate/bench scripts · `P3_Experimental_Report.ipynb`
- Canonical single-request protocol: constants in `src/latency_harness.py` (2000-domain sample, 200 warm-up, 1000 reps, n_jobs=1) — **every latency claim cites it**

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
- `pyahocorasick==2.3.1` pinned (B5 feature kernel); `marisa-trie==1.4.1` is a hard runtime dep via `preprocess.py`.

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
  - **AC/DAWG (Batch 5) — SUPERSEDED framing:** under the DT-12 primary (model ~13–45 µs), **feature extraction (47–92 µs, O(m²)-dominated) is the single-request bottleneck again** → AC/DAWG serves latency AND batch throughput (+~23 MiB dict). SHAP top feature = `lms_percentage` (an O(m²) scan).

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
- `test_onnx` **FIXED** (B3 Step 2): root cause was an **onnxruntime ≥1.26.0 TreeEnsemble kernel regression** (binary-RF probas emitted as (−p₁,+p₁), corrupting in-graph labels). Shipped: zipmap=False + **ort pinned 1.25.1**. Never bump onnxruntime without re-running `test_onnx` + a real-model parity check (re-verify wheel+parity on aarch64 in Batch 6).
- **RF-100·ONNX cold-start**: session init >1,800 s regardless of graph-opt level (kernel construction) — DQ'd for serving; small models init instantly.
- **Windows spawn cost** (~0.3–0.5 s/worker + dict unpickle) inflates parallel startup in RQ1 numbers; re-check on Linux `fork` / ARM in Batch 6.
- A1-adaptive equality + A5 end-to-end (kill/overload) tests were gaps in Batch 2 — closed in Batch-3 Step 0; keep them green.

## RQ2 result (Batch 4, MEASURED — `results/rq2/FINDINGS.md` + `INTERIM.md`)
- **All four DT-12 promotion gates PASS** → **PRIMARY = DT-12** (label applied by SamG); **fallback = RF-pruned·ONNX** (depth-10/α=0: 93.32%, 6.7 MB — also passes all C4 targets).
  - G1 McNemar (n=199,986): p=1.9e-09 in DT's favour — licenses **"beats"**, not just "not worse". G2 matched-FPR bracket: DT FNR 0.0595 vs RF 0.0636. G3: 0/25 chrmor families drop >5 pp. G4 shift (ExtraHop→chrmor, no fine-tune): **inverted** — DT-12 82.99% degrades least, RF-100 79.38% most.
- Canonical numbers: DT-12·ONNX **64 µs total / 56 MiB** (≈**83 MiB end-to-end** with dict+trigram → Profile C 244 MiB has ~2.9× headroom); DT-12·sklearn 102 µs / ≈182 MiB; RF-pruned·ONNX 77 µs / 69 MiB. Every RQ3 profile fits with margin — **RQ3 memory risk closed pending B6 measurement**.
- Key measured facts: **pruning does not fix sklearn latency** (4,214 vs 4,398 µs — joblib overhead dominates); **RF-100·ONNX DQ on cold-start** (TreeEnsemble kernel build 1,932 s); **int8 N/A for trees** (no weight matrices; no int8 TreeEnsemble op) — investigated, closed; **5-feature set is on the Pareto front** (31-subset sweep).
- **Blind spot (all models, model-parallel):** dictionary-composing families — matsnu 0.6–3%, suppobox 0.9–4.4%, nymaim ~2%, gozi ≤17% recall. Limitation of the lexical feature family → T3 + viva narrative; no gate affected.
- chrmor: github.com/chrmor/DGA_domains_dataset @ 9dcc29e5, 674,814 rows, 25 families; cite Cucchiarelli et al. 2021 (ESWA 170).

## Feature-kernel result (Batch 5, MEASURED — `results/kernel/FINDINGS.md`)
- The two O(m²) dictionary features run on a **C Aho-Corasick kernel** (`FEATURE_KERNEL=fast`, default; `legacy` selectable for A/B). **Provably identical output**: bit-identical on the 1,049-domain golden-v2 oracle *and* the full 999,927-domain corpus (max|Δ|=0.0) → B4 gate verdicts carry over untouched.
- Canonical single-request (same-session before/after): DT-12·ONNX **71.5→45.8 µs p50**, DT-12·sklearn 136.6→74.2; feature p99 tails linearised (sklearn arm **371→91 µs**, ONNX arm 299→110). Batch `Pool.map` k=8: 78.9k (baseline reproduced) → **152.4k dom/s (1.93×)**.
- Cost: **+34.5 MiB** automaton RSS — DT-12·ONNX end-to-end **118.9 MiB** (Profile-C headroom 2.05×). marisa-mmap fallback measured (0.7 MiB, 1.6× kernel latency) for tighter profiles.
- **Spawn traps (measured):** workers read `FEATURE_KERNEL` from the **env** (`set_kernel_mode` is process-local); per-worker parallel automaton build beats shipping the 26.5 MiB blob through spawn initargs — single-copy attach lives on the shm path. Re-check both on Linux `fork`/ARM in Batch 6.

## Rejected — do not relitigate
- Shannon entropy as a novelty claim (foundational; off-the-shelf).
- QEMU for latency/throughput numbers (functional emulation, not cycle-accurate; OK only for functional/portability parity).
- Beating detection accuracy (saturated field).
- Multi-feature score fusion as novel (already the RF probability output).
- int8 quantization for the tree ensemble (B4: thresholds not weights; no int8 TreeEnsemble op; corrupts [0,1] split grids).
- CFG-compliance as a *detector* (collapses into n-gram methods; CFGs are the attacker's tool). Probabilistic-CFG DGAs are a hard family to *test against*, not to build.

## Planning
- Live plan: `docs/IMPLEMENTATION_PLAN.md` (gated batches; read on demand — **not** imported, plans change too often to live in memory) + the thread tracker (T1–T7).
- Highest-value levers, **post-B5**: (1) **T1 real ARM validation** (Batch 6 — publishability; re-verify ort-1.25.1 aarch64 parity, `fork` spawn economics, DT-12 image sizing); (2) Batch-7 robustness/reproducibility (T3 dictionary-family blind spot feeds the viva narrative). DT-12 distill (B4) and AC/DAWG kernel (B5) are **done and measured**; 3-feature drop stays rejected (−2.3 pp).

## Statistical rigor
- Accuracy: stratified CV + paired t-tests. Latency/throughput: mean ± SD with CIs over repetitions.
