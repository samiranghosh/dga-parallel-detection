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
ONNX totals include ~47–57 µs feature extraction. Parity: labels `array_equal`,
max |Δp| ≤ 1.5e-6 (RF-pruned/DT-12/LR re-asserted in-batch; RF-100 from B3;
2,000-row samples). RSS columns are model+runtime process; end-to-end serving
adds dict 23 + trigram 4 MiB → **DT-12·ONNX ≈ 83 MiB end-to-end**, still ~2.9×
Profile-C headroom.

## Gate verdicts (1–4)
| gate | verdict | key numbers |
|---|---|---|
| **1 — McNemar** (DT-12 vs RF-100, full 199,986 holdout) | **BEATS (significant, DT favour)** | discordant b=4,316 / c=4,893; exact two-sided p = **1.9e-09** |
| **2 — Operating point** | **PASS** (re-measured; two-sided bracket) | argmax: DT FPR 0.0581 / FNR 0.0726 vs RF 0.0732 / 0.0632. At DT's **conservative matched point** (achieved FPR 0.0729 ≤ RF's 0.0732): DT FNR **0.0595** vs RF **0.0636 at the same achieved FPR** → DT better. (DT's 598 leaf-scores can't hit the target exactly; the permissive bracket FPR 0.0850 / FNR 0.0505 is reported but not used for the verdict.) |
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
path), and ≈83 MiB end-to-end instead of ≈659 MiB. Structural pruning tells
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

---

## Batch-5 addendum (02 Jul — feature kernel; full report `results/kernel/FINDINGS.md`)
The two O(m²) dictionary features were reimplemented on a C Aho-Corasick
kernel (`pyahocorasick`, bench winner 9.7 vs marisa 15.3 vs legacy 47.0
µs/domain) with **provably identical output**: bit-identical on a
1,049-domain edge-case oracle (`tests/golden_features_v2.json`) and
max|Δ| = 0.0 on the full 999,927-domain corpus — model inputs are unchanged
bits, so the B4 gate verdicts carry over untouched. Effect (canonical
protocol, same-session before/after): **DT-12·ONNX 71.5 → 45.8 µs p50**,
DT-12·sklearn 136.6 → 74.2 µs; the O(m²) feature tail is linearised
(feature p99: sklearn arm 371 → 91 µs, ONNX arm 299 → 110 µs). Batch `Pool.map` k=8: 78.9k (baseline
reproduced) → **152.4k dom/s (1.93×)**. Cost: +34.5 MiB automaton RSS —
DT-12·ONNX end-to-end 84.4 → **118.9 MiB** (Profile-C headroom 2.9× →
2.05×; all RQ3 profiles still fit; marisa-mmap fallback measured at
0.7 MiB / 1.6× kernel latency for tighter profiles). Gates green
throughout (suite 54 passed); `FEATURE_KERNEL=legacy` retained for
A/B + rollback.

---

## Batch-6 addendum (03 Jul — RQ3 edge validation; full report `results/rq3/FINDINGS.md`)
**C5 MET on x86+cgroup, with 20–40× margin**: all 12 grid cells
({DT-12·ONNX, RF-pruned·ONNX} × {fast, fast_marisa} × {A 2c/512M,
B 1c/512M, C 1c/256M}) sit at **24.0–50.3 µs p50** (canonical protocol,
run in-container); worst p99 anywhere is 363 µs. Idle RSS fits every
profile **before data arrives** (68.3–159.4 MiB peak; DT-12·fast_marisa
is 68 MiB under Profile C). Kernel choice locked (SamG): **fast for
A/B; fast_marisa for C + batch under RAM pressure** — the 1M-row batch
harness OOMs at A/fast and C/*, and marisa is the batch survivor
(73.3k dom/s at A, 0.7 MiB/worker). **T6 aarch64 functional parity:
PASS under QEMU (correctness only)** — golden-v2 bit-identical for all
three kernel modes, DT-12/RF-pruned labels array_equal, probas
max|Δ| = 0.0 (`results/rq3/parity_qemu_arm64.json`); the full pytest
suite did not run under QEMU (arm64 full image never built — WSL2 NAT;
FINDINGS §7). **T1 real ARM: NOT RUN — blocked on cloud account**;
the complete procedure is committed (`scripts/rq3/ARM_RUNBOOK.md`).
Claim bounding (approved): every latency/RSS claim is stated as
**"x86 under cgroup"** until T1 lands; QEMU contributed correctness
verification only.
