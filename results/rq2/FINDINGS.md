# Batch 4 — Model Grid + DT-12 Promotion Gates (RQ2): FINDINGS

**Setup.** Branch `rq2-compression` off `eef15d5` (spike head; the brief's
`97f9059` predates the decisions commit). Fixed seeds (rs=42) everywhere.
Machine: 8c/16t Ryzen 7 7840HS, Windows 11, Python 3.11, sklearn 1.8.0,
onnxruntime 1.25.1 (pinned), skl2onnx 1.20.0. Gates verdicts reported;
**the primary-model label is applied by SamG.** Artifacts: `manifest.json`,
`models/` (DT-12, RF-pruned, LR pickles), `pruning_sweep.json`,
`gate1_mcnemar.json`, `gate2_operating_point.json`, `gate34_chrmor.json`,
`step0b_canonical_contrast.json`, `step5_compression.json`,
`step6_runtime.json`, `step6_rf100_coldstart.json`, `step7_subsets.json`,
`fig_c4_tradeoff.png`, `INTERIM.md`. Scripts: `scripts/rq2/`.

## Canonical single-request protocol (Step 0b, locked in `src/latency_harness.py`)
First 2,000 domains of `data/test.csv` (file order) · 200 warm-up + 1,000 timed
reps · `perf_counter_ns` · serving `n_jobs=1` · per-request = feature
extraction + predict on the (1,5) row · total/feature/predict p50/p95/p99 ·
idle box. Under it, RF-100 n_jobs=-1 re-measures at **17.98 ms p50**
(Batch-1's 17.4 concordant; the spike's 30.3 ms was machine state) and
n_jobs=1 at **4.31 ms**. Every number below cites this protocol.

## Model grid (Step 1; manifest with data hash `5a77ae5b0940e671`)
| model | config | holdout acc | F1 | pickle | nodes |
|---|---|---|---|---|---|
| RF-100 | n_estimators=100, rs=42 | 93.1790% | 0.9321 | 491 MB (not committed; manifest-reproducible) | 6,138,656 |
| RF-pruned | **depth=10, α=0** (sweep winner: smallest within 1 pp — in fact **+0.14 pp**) | 93.3150% | 0.9318 | 6.7 MB | 83,504 |
| DT-12 | max_depth=12, rs=42 | 93.4675% | 0.9342 | 227 KB | 2,887 |
| LR | scaler+LogReg, rs=42 | 90.0013% | 0.9000 | 1.2 KB | — |

## Gate verdicts
**Gate 1 — McNemar (n=199,986): BEATS, significant in DT's favour.**
Discordant pairs: b=4,316 (DT wrong/RF right), c=4,893 (DT right/RF wrong);
exact two-sided binomial **p = 1.9e-09**. Per the rubric this licenses
"beats", not merely "not meaningfully worse".

**Gate 2 — operating point: PASS** *(re-measured after verification caught a
threshold overshoot in v1)*. Argmax: DT FPR 0.0581 / FNR 0.0726 vs RF-100
0.0732 / 0.0632. DT's 598 distinct leaf-scores cannot hit RF's FPR exactly, so
the comparison uses a **two-sided bracket**: at DT's *conservative* point
(achieved FPR **0.0729** ≤ RF's 0.0732) DT FNR = **0.0595** vs RF FNR
**0.0636 at the same achieved FPR** → DT better at a genuinely matched point.
(DT's permissive bracket: FPR 0.0850 / FNR 0.0505 — laxer FPR, not
like-for-like.) RF-pruned matched-FNR 0.0559; LR 0.1223. No asymmetry hides
behind argmax. Full brackets + confusions in `gate2_operating_point.json`.

**Gate 3 — per-family: PASS (run on chrmor; ExtraHop has no family labels).**
**0 of 25 families** where DT-12 recall drops >5 pp vs RF-100.

**Gate 4 — cross-dataset shift (train ExtraHop → test chrmor, no fine-tune): PASS, inverted.**
| model | chrmor acc | F1 | FPR | FNR | acc excl. train-overlap |
|---|---|---|---|---|---|
| RF-100 | 79.38% | 0.766 | 0.087 | 0.325 | 79.12% |
| RF-pruned | 82.82% | 0.807 | 0.062 | 0.281 | 82.63% |
| **DT-12** | **82.99%** | **0.812** | 0.074 | **0.266** | **82.82%** |
| LR | 81.37% | 0.800 | 0.118 | 0.254 | 81.24% |

All models degrade under shift; **the forest degrades most and the single tree
least** — the "brittle single tree" risk inverted. Verdicts identical with the
1.74% train-overlap (11,768 domains) excluded.
chrmor provenance: github.com/chrmor/DGA_domains_dataset @ `9dcc29e5`;
674,814 rows after ExtraHop-identical cleaning (lowercase → tldextract
strip-suffix keeping subdomains → dedupe → sort); 25 families; licence "free
for research purposes", cite Cucchiarelli et al. 2021, ESWA 170,
DOI 10.1016/j.eswa.2020.114551.

## Compression arms (Step 5)
- **Structural pruning** (the C4 fallback axis): depth-10/α=0 wins the 20-cell
  sweep — 93.315% at 6.7 MB (73× smaller than RF-100, +0.14 pp). Full grid in
  `pruning_sweep.json`.
- **float32**: RF-100 carries 147.3 MB of float64 value+threshold arrays →
  ~73.7 MB savable at cast; float32 *input* is accuracy-neutral to 4 d.p. on
  both RFs. True in-place cast needs a custom serving struct — sklearn's Tree
  is fixed-dtype; ONNX (float32 end-to-end) is the deployable realization.
- **int8**: **not applicable as weight quantization** — trees store split
  thresholds and leaf distributions, not weights; 8-bit threshold grids
  corrupt decision boundaries on the [0,1] ratio features, and ai.onnx.ml
  TreeEnsemble has no int8 mode. Investigated, documented, not pursued.

## Runtime arms (Step 6; ONNX = ort 1.25.1, zipmap=False)
Parity: labels `array_equal`, max |Δp| ≤ 1.5e-6 — re-asserted in-batch for
RF-pruned/DT-12/LR on 2,000 holdout rows; RF-100 parity carried from B3 (same
2,000-row protocol, 2.24e-7).
Canonical single-request (total = extract+predict; feat ≈ 47–92 µs of it):
| variant | total p50 | predict p50 | RSS total (MiB) |
|---|---|---|---|
| RF-100 · sklearn | 4,398 µs | 4,279 µs | 631.7 |
| RF-100 · ONNX | *(steady 18 µs, B3)* | — | 990.7 + **cold-start DQ** |
| RF-pruned · sklearn | 4,214 µs | 4,092 µs | 165.6 |
| RF-pruned · ONNX | **77 µs** | 20 µs | **69.2** |
| DT-12 · sklearn | **102 µs** | 45 µs | 155.3 |
| DT-12 · ONNX | **64 µs** | 13 µs | **56.0** |
| LR · sklearn | 200 µs | 132 µs | 154.8 |
| LR · ONNX | 61 µs | 14 µs | 54.8 |

- **RF-100-ONNX: DISQUALIFIED ON COLD-START** (rubric >60 s): init 1,932 s
  (ENABLE_ALL) / >1,800 s (DISABLE_ALL) — both >30× the threshold; the cost is
  TreeEnsemble kernel construction, which a pre-optimized `.ort` cannot skip
  (not separately tested; recorded per do-not-sink-time). `step6_rf100_coldstart.json`.
- Striking sklearn-path fact: **pruning does not fix sklearn serving latency**
  (4,214 vs 4,398 µs — the Python/joblib overhead dominates, as probe 1
  predicted); it fixes memory. The latency fix is the runtime (ONNX) or the
  tiny model (DT-12: 102 µs even in sklearn).

## Feature rigor (Step 7, DT-12 = gate winner; 31-subset exhaustive + SHAP)
Best per subset size: 1→85.45% {lms} · 2→90.63% {pron,lms} · 3→92.57%
{len,pron,lms} · 4→93.40% {len,mwr,pron,lms} · **5→93.4675% (full set)** —
**the 5-feature set is on the Pareto front** (supersedes probe 3's single
point). SHAP mean-|v| ranking: **lms_percentage 0.226 > pronounceability
0.175 > length 0.072 > meaningful_word_ratio 0.065 > numerical_ratio 0.011**.
The top feature is a dictionary O(m²) scan — reinforcing Batch 5 (AC/DAWG).
Note: dropping `numerical_ratio` costs only 0.07 pp, but it is O(m)-cheap;
no case for removal.

## C4 adjudication (`fig_c4_tradeoff.png`)
Targets: ≥3× RSS↓ · ≥2× speedup · <1 ms single-request · <1 pp accuracy drop.
| variant | RSS↓ vs 631.7 | speedup vs 4,398 µs | <1 ms | Δacc | verdict |
|---|---|---|---|---|---|
| RF-pruned · sklearn | 3.8× | 1.04× | ✗ | +0.14 pp | partial (memory only) |
| RF-pruned · ONNX | **9.1×** | **57×** | ✓ | +0.14 pp | **all met** |
| DT-12 · sklearn | 4.1× | 43× | ✓ | +0.29 pp | **all met** |
| DT-12 · ONNX | **11.3×** | **69×** | ✓ | +0.29 pp | **all met** |
| LR · ONNX | 11.5× | 72× | ✓ | −3.18 pp | fails accuracy |
| RF-100 · ONNX | 0.64× (worse) | (244× steady) | ✓* | 0 | fails RSS + cold-start |

RSS cells are model+runtime process (dict/trigram excluded on BOTH sides, so
the ratios are like-for-like; baseline 631.7 + dict 23 + trigram 4 ≈ 658.7
reproduces Batch-1's 656.8 total). End-to-end serving adds the dictionary +
trigram: **DT-12·ONNX ≈ 83 MiB end-to-end** (Profile C 244 MiB → **~2.9×
headroom**); DT-12·sklearn ≈ 182 MiB — every RQ3 profile fits with margin.
Latency ratios use the Step-6 canonical run (RF-100 total 4,398 µs; the 0b run
of the same config measured 4,310 µs — ~2% run-to-run variance).

## The RQ2 story
The RF-100 baseline is **over-provisioned for this 5-feature space**. A single
depth-12 tree is statistically better on the holdout (McNemar p=1.9e-09),
better at a matched operating point, within 5 pp on all 25 families, and *more*
robust under cross-dataset shift — while meeting every edge target with
orders-of-magnitude margins (227 KB, 64–102 µs, ≈83 MiB end-to-end). Structural pruning
tells the same story from the forest side (93.32% @ 6.7 MB), so this is
feature-space saturation, not tree-luck. The pruned-RF+ONNX arm passes all
C4 targets too and stands as the documented fallback.

## Caveats (honest)
- **Dictionary-composing DGA families evade ALL models** (chrmor recalls:
  matsnu 0.6–3.0%, suppobox 0.9–4.4%, nymaim 1.4–2.4%, gozi 15.5–16.8%) — a limitation of
  the linguistic feature family itself, not of any model choice (DT-12 even
  detects symmi at 34% where RF-100 scores 0.0%). Feeds Batch-7 T3 and the
  viva narrative; no gate is affected (gaps are model-parallel).
- chrmor benigns are Alexa domains (different benign distribution than
  ExtraHop's); shift results measure the combined domain+label shift.
- DT-12 threshold granularity (598 scores) is adequate here but coarser
  operating-point control than the RF; relevant if a deployment needs FPR
  tuning below ~1e-3 resolution.
- Windows-only; ARM re-validation is Batch 6 (incl. ort-1.25.1 aarch64 wheel).
- RF-100 pickle not committed (491 MB); manifest records config+seed+data hash
  for exact reproduction (verified bit-exact twice in this batch).
