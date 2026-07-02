# RQ2 INTERIM — Batch 4 checkpoint (Mid-Sem, 10 Jul)

**Standalone status after Steps 0–6.** Branch `rq2-compression` (off `eef15d5`).
All numbers below are committed artifacts under `results/rq2/` (+`results/spike/`
for cited Batch-3 values). Fixed seeds (rs=42) throughout. Gates verdicts are
reported; the primary-model **label is applied by SamG**, not in-batch.

## Canonical single-request protocol (locked, `src/latency_harness.py`)
First 2,000 domains of `data/test.csv` (file order) · 200 warm-up + 1,000 timed
reps · `perf_counter_ns` · serving `n_jobs=1` · per-request = feature extraction
+ predict on the (1,5) row · total/feature/predict p50/p95/p99 reported
separately · idle box, AC power. This supersedes the Batch-1 (17.4 ms) and
spike (30.3 ms) ad-hoc runs — under the canonical protocol RF-100 n_jobs=-1
re-measures at **17.98 ms p50** (Batch-1 concordant; the spike's 30.3 ms was
machine state).

## Headline table (canonical protocol; ExtraHop holdout n=199,986)
| model | holdout acc | F1 | single-request p50 (sklearn) | p50 (ONNX) | RSS sklearn-path | RSS ONNX-only | pickle |
|---|---|---|---|---|---|---|---|
| RF-100 (baseline) | 93.179% | 0.9321 | 4,398 µs (n_jobs=1); 17,980 (-1) | 18 µs* | 631.7 MiB | 990.7 MiB* + **cold-start DQ** | 491 MB |
| RF-pruned (d10, α=0) | **93.315%** | 0.9318 | 4,214 µs | **77 µs** | 165.6 MiB | **69.2 MiB** | 6.7 MB |
| **DT-12** | **93.4675%** | **0.9342** | **102 µs** | **64 µs** | 155.3 MiB | **56.0 MiB** | 227 KB |
| LR (scaled) | 90.001% | 0.900 | 200 µs | 61 µs | 154.8 MiB | 54.8 MiB | 1.2 KB |

\* RF-100-ONNX steady-state from Batch 3; **disqualified on cold-start** (init
1,932 s ENABLE_ALL / >1,800 s DISABLE_ALL — both >30× the 60 s rubric; cost is
TreeEnsemble kernel construction, so a pre-optimized `.ort` cannot cure it).
ONNX totals include ~50–60 µs feature extraction; ONNX parity asserted per
model (labels `array_equal`; max |Δp| ≤ 1.5e-6).

## Gate verdicts (1–4)
| gate | verdict | key numbers |
|---|---|---|
| **1 — McNemar** (DT-12 vs RF-100, full 199,986 holdout) | **BEATS (significant, DT favour)** | discordant b=4,316 / c=4,893; exact two-sided p = **1.9e-09** |
| **2 — Operating point** | **PASS** | argmax: DT FPR 0.0581 / FNR 0.0726 vs RF 0.0732 / 0.0632. At **matched FPR** (=RF's 0.0732): DT FNR **0.0505** < RF 0.0632. DT has 598 distinct leaf-scores — coarse but sufficient; no hidden asymmetry. |
| **3 — Per-family** | **PASS** (run on chrmor — ExtraHop carries no family labels; folded into gate 4 per brief) | **0 of 25 families** where DT recall drops >5 pp vs RF-100 |
| **4 — Cross-dataset shift** (train ExtraHop → test chrmor, no fine-tuning) | **PASS — inverted** | all models degrade under shift, but **RF-100 degrades most** (79.38%) and **DT-12 least** (82.99%; RF-pruned 82.82, LR 81.37). DT FNR 0.266 < RF 0.325. The "single tree is brittle under shift" risk did not materialise — the forest was the brittle one. |

chrmor provenance: github.com/chrmor/DGA_domains_dataset @ `9dcc29e5` (674,814
rows after ExtraHop-identical cleaning; 25 families × ~13.5k; licence: free for
research, cite Cucchiarelli et al. 2021, ESWA 170, DOI 10.1016/j.eswa.2020.114551).
Train-set overlap 11,768 domains (1.74%) — metrics also computed overlap-excluded
(`gate34_chrmor.json`), verdicts unchanged.

## The RQ2 story (one paragraph)
The RF-100 baseline is **over-provisioned for this 5-feature space**: a single
depth-12 decision tree is statistically *better* on the ExtraHop holdout
(McNemar p=1.9e-09), no worse at a matched operating point, no worse on any of
25 DGA families, and *more* robust — not less — under cross-dataset shift,
while costing 227 KB instead of 491 MB, 102 µs instead of 4.4 ms (sklearn
path), and 56 MiB total process instead of 632 MiB. Structural pruning tells
the same story from the forest side (depth-10 RF: 93.32% at 6.7 MB), so the
conclusion is not tree-luck but feature-space saturation. Every RQ3 edge
target — <1 ms single-request, ≤256 MB (even ≤64 MiB) — is met with large
margins by the distilled tree; the pruned-RF+ONNX arm stands as the documented
fallback, and the full C4 trade-off curve across {RF-100, RF-pruned, DT-12,
LR} × {sklearn, ONNX} completes the deliverable (Steps 7–8).

## Status of remaining steps
- Step 5 ✅ (float32: RF-100 could halve 147 MB of tree arrays; float32 *input*
  is accuracy-neutral at 4 d.p.; int8 = not applicable to tree thresholds —
  documented). Step 6 ✅ (parity + latency + RSS above; RF-100-ONNX cold-start DQ).
- Step 7 (31-subset sweep + SHAP, DT-12) — running. Step 8 (C4 curve + FINDINGS) — next.
