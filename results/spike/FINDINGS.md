# Batch 3 — Serving-Path Spike (RQ2/RQ3): FINDINGS

**Setup.** Branch `rq2-serving-spike` off `baseline-verified` (929f3a2). Baseline
model = RF-100 (`n_estimators=100, random_state=42`) on the 5-feature matrix,
retrained from `data/train.csv` and verified **bit-exact**: test accuracy
**93.1790%** == the tagged baseline. Model: 491 MB pickle, 6,138,656 tree nodes.
Machine: 8c/16t Ryzen 7 7840HS, Windows 11, Python 3.11. Latency protocol:
single (1,5) float32 row, 200 warmup, 1000 timed reps, p50/p95/p99.
Artifacts: `step1_njobs.json`, `probe1_traversal_floor.json`,
`probe2_onnx_treelite.json`, `probe3_3feature.json`, `probe4_distillation.json`.

## Headline
**The RF-100 is over-provisioned for this 5-feature space.** A single depth-12
decision tree **exceeds** the baseline accuracy (holdout **93.47% vs 93.18%**,
+0.29 pp; CV 93.31 ± 0.15 vs 93.10 ± 0.03) at **35.5 µs** single-request,
**227 KB** pickle, **0.6 MiB** load RSS — versus the RF's 5.5 ms (sklearn
n_jobs=1), 491 MB pickle, 473 MiB RSS, and a pathological **>30-minute** ONNX
session init. Every RQ2/RQ3 target (<1 ms, ≤256 MB) is met by the distilled
tree with orders of magnitude to spare. *(Decision RF-primary vs distill-primary
is SamG's; Batch 4 should add McNemar + paired t-tests before locking.)*

## Step 1 — n_jobs=1 at serving (locked, commit 66ecdf4)
| serving config | total p50 / p95 / p99 (ms) | predict p50 |
|---|---|---|
| n_jobs=-1 (as trained) | 30.33 / 33.42 / 36.20 | 30.12 ms |
| **n_jobs=1 (locked)** | **5.48 / 7.49 / 8.94** | 5.33 ms |

5.5× drop, free. (Batch-1 recorded 17.4→4.6 ms for the same lever; the joblib
fan-out path is noisy run-to-run — direction and magnitude concordant.)
`api.py` sets it on the loaded model; `latency_harness.py` single-request path
defaults to it (`serving_n_jobs=None` measures as-is). Batch/training untouched.

## Step 2 — ONNX parity FIXED (commit d1de64f)
Route taken: **(a) zipmap=False was necessary but insufficient** — probability
tensor came back as **(−p₁, +p₁)** and the in-graph label (its argmax) is 1
whenever p₁>0 → 47/100 mock labels wrong. **(b) post-normalize inadmissible** —
its precondition (labels `array_equal`) fails by construction. **(c) pin-shift
isolated the bug to onnxruntime's TreeEnsembleClassifier kernel**: 1.24.4 and
1.25.1 PASS; 1.26.0 and 1.27.0 FAIL; skl2onnx not implicated (1.19.1 identical).
**Shipped: zipmap=False + onnxruntime pinned 1.27.0 → 1.25.1.**
`test_onnx` GREEN. Real-model parity (RF-100, 2000 rows): labels equal
(0 mismatches), max |Δproba| = 2.24e-7.

## (d) Probe 1 — traversal floor
| path | p50 / p95 / p99 (µs) |
|---|---|
| sklearn predict, n_jobs=1 | 5,789 / 8,174 / 9,071 |
| **raw `tree_.predict` ×100 + aggregate** | **408 / 734 / 975** |

Parity 2000/2000 vs `model.predict`. **~93% of sklearn's n_jobs=1 latency is
Python/joblib serving overhead, not tree compute** — and the 408 µs floor still
contains 100 Python-level calls; ONNX (18 µs) shows the fused-native ceiling is
~20× lower still. `<1 ms` single-request is comfortably physical.

## (e) Probe 2 — ONNX vs treelite + ONNX-only RSS
| runtime | p50 / p95 / p99 (µs) | parity vs sklearn |
|---|---|---|
| ONNX (ort 1.25.1) | **18** / — / — *(p50 from main-run log)* | labels equal; ≤2.2e-7 |
| treelite GTIL 4.7.0 | 72.8 / 122.6 / 224.9 | labels equal; ≤3.3e-16 |
| tl2cgen compiled lib | **BLOCKED on Windows** (libloader DLL path + would need MSVC) → **deferred to Batch 6** (Linux/Docker), per plan | — |

ONNX-only staged RSS (fresh process, sklearn provably not imported):
interpreter **20.1** → +numpy **31.9** → +onnxruntime **48.8** → +session
**990.4** → first inference **990.7 MiB**.
- **Runtime floor = 48.8 MiB** (vs sklearn import floor ≈133 MiB) — an
  ONNX-only image leaves ~207 MiB of a 256 MB budget for the model.
- **Unpruned RF-100 session = 990 MiB — WORSE than sklearn's 473 MiB** and
  **session init took 1,932 s (32.2 min)**; a DISABLE_ALL attempt exceeded 30 min
  too, so the cost is TreeEnsemble kernel construction, not graph optimization.
- **Consequence: compression is a prerequisite for the ONNX path's LOAD TIME
  and RSS** — the uncompressed model is undeployable on edge regardless of its
  18 µs steady-state latency (cold-start alone disqualifies it).

## (f) Probe 3 — 3-feature accuracy (drop dictionary features)
{length, numerical_ratio, pronounceability}, RF-100, stratified 5-fold CV on
the combined 1M set + E7-protocol holdout:
| config | CV acc (±SD) | holdout acc | Δ vs baseline |
|---|---|---|---|
| 5-feature reference | 93.10 ± 0.03% | 93.179% | — |
| **3-feature** | **90.83 ± 0.05%** | **90.93%** | **−2.24 pp holdout / −2.27 pp CV** |

**The dictionary features earn their memory**: dropping them costs ~2.3 pp —
far outside the <1% envelope. The dictionary subsystem (23 MiB) **survives**;
by extension the AC/DAWG batch-throughput lever retains its target. Note: with
a 36 µs model (probe 4), **feature extraction (~65–145 µs, dominated by the two
O(m²) dictionary scans) becomes the single-request bottleneck again**, which
strengthens — not weakens — the AC/DAWG case if the distill route is chosen.

## (g) Probe 4 — distillation floor (5 features)
| model | CV acc (±SD) | CV F1 | holdout acc / F1 | p50 (µs) | pickle | load RSS Δ |
|---|---|---|---|---|---|---|
| LogisticRegression (scaled) | 89.87 ± 0.03% | 0.899 | 90.00% / 0.900 | 121 | 1.2 KB | ~0 MiB |
| DT depth=6 | 90.84 ± 0.07% | 0.905 | — | — | — | — |
| DT depth=8 | 91.67 ± 0.16% | 0.914 | — | — | — | — |
| DT depth=10 | 92.70 ± 0.13% | 0.926 | — | — | — | — |
| **DT depth=12** | **93.31 ± 0.15%** | **0.933** | **93.47% / 0.934** | **35.5** | **226.6 KB** | **0.6 MiB** |
| *(RF-100 baseline)* | *93.10 ± 0.03%* | — | *93.18%* | *5,480 (sklearn) / 18 (ONNX)* | *491 MB* | *473 MiB* |

Accuracy still rising at depth 12 (trend 90.8→91.7→92.7→93.3) — depth 14/16
worth probing in Batch 4 alongside overfitting checks. DT-12's CV mean sits
0.21 pp above the RF's with SD 0.15 — **statistically confirm with McNemar +
paired t-test in Batch 4 before treating "beats the RF" as settled** (holdout
+0.29 pp concurs).

## Decision inputs (SamG resolves; spike does not act on these)
1. **RF-primary vs distill-primary:** DT-12 dominates the RF on every measured
   axis — accuracy (+0.29 pp holdout), latency (35.5 µs plain sklearn, no ONNX
   needed), memory (0.6 vs 473 MiB), load time (instant vs 32 min ONNX init),
   and deployment simplicity (no runtime pinning, no parity risk). If chosen,
   RQ2 reframes to "the RF is over-provisioned; a distilled tree meets every
   edge target," and the ONNX/treelite path becomes the *fallback* story.
2. **Does AC/DAWG survive?** Probe 3 says the dictionary features stay
   (−2.3 pp without them). Under distill-primary, feature extraction becomes
   the dominant single-request cost again → AC/DAWG is relevant to both batch
   throughput *and* single-request latency. Survives on this evidence.

## Caveats
- Windows-only measurements; tl2cgen compiled-lib deferred to Batch 6 (Linux).
- ONNX 18 µs is p50-only (full percentile set lost to the RSS-timeout crash of
  the first probe-2 run; parity + p50 are from its log — rerun under Batch 4 on
  the pruned/chosen model, where session init is no longer pathological).
- probe-2 `treelite_import_sec=0.6` is the warm remeasure; first-ever import in
  the crashed run took minutes (one-time numba/JIT-style overhead).
- DT/LR latencies measured through `model.predict` (sklearn Python path); a
  native DT export would go lower still — irrelevant at 36 µs.
