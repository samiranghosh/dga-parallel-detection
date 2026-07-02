# DGA Dissertation — Implementation Plan (Claude Code)

Gated batches. Execute one, report, then the next is issued. Files in `src/` unless noted.
**Status (02 Jul, post-spike):** B1 ✅ · B2 ✅ (RQ1 done, honest-negative streaming) · B3 ✅ (spike done; **both decisions RESOLVED** — distill-primary conditional, AC/DAWG survives) · B4 next (reshaped) · B5 confirmed+re-promoted · B6/B7 updated.
Acceptance IDs (A1–A5 / B1–B5 / C1–C5 / D / E) reference the Testing Approach doc.
**Revised after Batch-1 measurement:** single-request latency and RSS are **model-dominated, not feature-dominated** (predict = 99% of ~4.6 ms; features ~145 µs / <1%; model 473 of 657 MiB). Sequencing updated accordingly: **model compression is the shared lever for RQ2 memory *and* RQ3 latency**; AC/DAWG is demoted to a conditional batch-throughput lever. RQ1 stays front-loaded (independent, feeds Mid-Sem).

---

## Batch 1 — Phase 0: Baseline + shared instrumentation  ✅ DONE
Tag `baseline-verified` (sklearn 1.8.0); E7 bit-exact (93.179% / 92.602%); latency + RSS harnesses + golden snapshot committed. Finding: model-dominated bottleneck (see header). Peak RSS 657 MiB > every cgroup profile.

## Batch 2 — Phase 1: RQ1 Adaptive parallel engine  ✅ DONE — `results/rq1/FINDINGS.md`
**Result (honest-negative for streaming throughput):** queue-fed engine is IPC-bound — sequential ~14.8k > static-8 ~8.4k > adaptive ~5.6k dom/s; the 78.8k figure remains the *batch* `Pool.map` story. Adaptive's value = **per-worker efficiency** (~1,250 vs ~1,050 dom/s/worker) + **graceful 2×-peak saturation** (0.99–1.01, target HELD) + **zero oscillation** + fault recovery. ≤2-core: small/negative, valid bounded finding. Conformance verified in code (500 ms sampling · 0.5 s cooldown · kp=0.5 · hysteresis dead-band · min–max · shm attach · reap-and-replace). Feeds Mid-Sem as the RQ1 section.
**ZIP-audit gaps → closed in Batch-3 Step 0:** A1-adaptive equality test · A5 end-to-end kill/overload · rq1 figs/JSONs untracked · `docs/IMPLEMENTATION_PLAN.md` absent.
Original spec (executed):
**Prereq:** land the uncommitted June-29 `parallel_engine.py` on `rq1-adaptive` first (validate+eval, don't rewrite blind).
Files: `parallel_engine.py`, `chunker.py`, `shared_resources.py` (reuse `_init_worker_shm`), `fault_handler.py`, `tests/test_parallel.py` (gate), new load harness.
1. Confirm the queue-fed re-arch (controller + shared queue + workers); `Pool.map` can't resize.
2. Controller: sample queue depth + CPU every 500 ms; proportional scale; **min–max band + hysteresis + cooldown**.
3. New workers attach to shared-memory dictionary — no per-worker reload.
4. `test_parallel.py` green — adaptive == sequential — **A1**. 5. Worker-kill + queue-overload recovery — **A5**.
6. Adaptive vs static K=8 under **uniform / bursty-Poisson / ramp**; throughput + CPU + worker-count series; **cost–benefit curve**; target = no collapse at 2× peak — **C3, RQ1**.
**Exit:** gate green · cost–benefit curve.

## Batch 3 — Phase 2: Serving-path spike  ✅ DONE — `results/spike/FINDINGS.md`
**Result:** predict path was ~93% Python/joblib overhead (traversal floor 408 µs; ONNX **18 µs** with exact parity after zipmap=False + **ort pin 1.25.1** — ≥1.26.0 kernel regression emits (−p₁,+p₁)). ONNX on the *unpruned* RF is undeployable: session init 15–32 min, 990 MiB session RSS; runtime floor **48.8 MiB**. Probe 3: 3-feature −2.3 pp → dictionary stays. Probe 4: **DT-12 = 93.47% holdout / 35.5 µs / 227 KB / 0.6 MiB** ≥ RF-100 baseline. n_jobs=1 locked (30.3→5.5 ms). tl2cgen import-blocked on Windows → B6. Probe scripts: `scripts/spike/`.
Original spec (executed): branch `rq2-serving-spike`; deliverable FINDINGS + JSONs; report-only — decisions resolved by SamG below.
**Step 0 (on `rq1-adaptive`):** close Batch-2 gaps — A1-adaptive `allclose` equality test · A5 end-to-end worker-kill + queue-overload tests · commit rq1 figs/sweep JSONs (`.gitignore` carve-out) · land `docs/IMPLEMENTATION_PLAN.md` + fix CLAUDE.md pointer.
**Step 1:** lock `n_jobs=1` at serving (`api.py` + latency harness single-request path): 17.4 → ~4.6 ms, free.
**Step 2:** fix ONNX parity (`test_onnx`) — known root cause (skl2onnx 1.20.0 + ort 1.27.0 malformed binary-RF probas). Route: zipmap=False → documented post-normalize (iff labels `array_equal`) → pin-shift. Hard req: predictions `array_equal`, probas `allclose` rtol=1e-3.
Then measure:
**Step 3 — Probe 1 (traversal floor):** bypass sklearn's predict path (`est.tree_.predict`, float32, aggregate votes), ≥1000 warmed reps, p50/p95/p99. Ceiling for any native runtime (the 4.6 ms is Python+joblib overhead, not tree compute).
**Step 4 — Probe 2 (ONNX vs treelite):** single-request p50/p95/p99, same protocol as Batch 1. If tl2cgen native compile blocks on the Windows MSVC toolchain: bench treelite GTIL now, defer compiled-`.so` to Batch 6 (Linux/Docker). Plus staged RSS of an **ONNX-only process** (no sklearn) — the direct 256 MB feasibility number.
**Step 5 — Probe 3 (3-feature accuracy):** RF on {length, numerical_ratio, pronounceability} only (all O(1)/O(m)); stratified 5-fold CV, acc+F1 vs 93.179%. Decides whether the dictionary subsystem (23 MiB) + AC/DAWG survive.
**Step 6 — Probe 4 (distillation floor):** LogisticRegression + single DecisionTree (depth ∈ {6,8,10,12}) on 5 features; CV acc+F1 + single-request latency + model-load RSS. Decides RF-primary vs distill-primary.
**Exit:** floor · ONNX-vs-treelite winner + ONNX-only RSS stages · probe-3 delta · probe-4 table. **Report resolves the two decisions below before Batch 4.**

### `[DECISIONS RESOLVED — SamG, 02 Jul, post-spike]`
- **Decision 1 — Distill-primary (conditional): DT-12 is the deployment candidate.** Basis: dominates every measured axis (93.47% holdout ≥ RF 93.18% · 35.5 µs · 227 KB · 0.6 MiB). **Promotion gates, ALL in Batch 4:** (1) McNemar vs RF-100 on the full holdout → must be "not meaningfully worse"; claim "beats" only if significant; (2) FPR/FNR parity at the operating threshold (accuracy hides asymmetry; DT leaf-probas are coarse); (3) per-family breakdown incl. dictionary/PCFG — no catastrophic family regression (T3 subset pulled forward); (4) cross-dataset generalization on the chrmor set (B4 pulled forward from Batch 7 — single trees are brittle under shift; this is the real risk). **Fallback:** pruned-RF + ONNX remains the documented arm. The **C4 curve runs on {RF-100, RF-pruned, DT-12, LR} regardless** — the curve is the RQ2 deliverable; "primary" is a label applied after gates. RQ2 story if gates pass: *"the RF is over-provisioned for this 5-feature space; a depth-capped tree meets every edge target."*
- **Decision 2 — AC/DAWG survives; Batch 5 confirmed and RE-PROMOTED.** −2.3 pp without dictionary features → the 23 MiB subsystem stays. Under distill-primary the model costs ~35 µs → feature extraction (~65–145 µs, O(m²)) is the single-request bottleneck again → AC/DAWG returns to **latency** relevance, not just batch throughput. Viva narrative (keep): bottleneck migrates — RF serving path dominated → distillation/native fixed it → bottleneck returned to features → AC/DAWG fixes that.
- **Consequence — Profile C (1c/256 MB) flips feasible:** sklearn-path DT-12 ≈ 183 MiB (133 sklearn floor + 23 dict + 4 trigram + 22 interpreter + 0.6 model; concurs with probe-4's measured 154.6+0.6 + dict/trigram ≈ 182); ONNX-only DT-12 ≈ 50 MiB. RQ3's memory risk effectively closed **pending B6 measurement**.

## Batch 4 — Phase 3: Model grid + DT-12 promotion gates  `[RESHAPED post-spike · carries RQ2 AND RQ3]`
Files: `classifier.py` (pruning helpers already staged uncommitted), ONNX export/serve path, `api.py`. **Model grid = {RF-100, RF-pruned, DT-12, LR}.**
1. **Four promotion gates for DT-12** (Decision 1): McNemar vs RF-100 (full holdout) · FPR/FNR parity at the operating threshold · per-family breakdown incl. dictionary/PCFG (**T3 subset pulled forward**) · cross-dataset generalization on chrmor (**B4 pulled forward from Batch 7**). Pass → distill-primary locked; fail → pruned-RF+ONNX arm promotes.
2. **Structural pruning** (fallback arm + C4 axis) — depth caps / ccp_alpha; accuracy within ~1% — **A3**. float32 + int8 = investigate-only sub-results.
3. **Protocol re-run on the chosen model** (spike flag 1): full ONNX percentile set (the 18 µs is p50-only) + per-variant staged RSS — session init is no longer pathological once the model is small.
4. Rigor: **McNemar + paired t-test** — **B2/B3**; exhaustive **31-subset** sweep (extends E7; supersedes probe 3's single point); SHAP for narrative.
5. **C4 trade-off curve** (accuracy vs memory vs latency) across the 4-model grid — the RQ2 deliverable.
**Exit:** gates adjudicated · C4 curve produced · ≥3× RSS↓ / ≥2× speedup / <1 ms single-request demonstrated or honestly bounded.

## Batch 5 — Phase 2b: Aho-Corasick over DAWG  `[CONFIRMED + RE-PROMOTED (Decision 2) · single-request latency AND batch throughput]`
Decision 2: survives. Under DT-12 (~35 µs model) **feature extraction (~65–145 µs, O(m²)) is the single-request bottleneck again**, so AC/DAWG now serves the RQ3 latency story *and* batch throughput (76 µs/dom) + ~23 MiB dict RAM.
**Prereq:** expand the golden set first — current 20-domain smoke test → add A4 edge cases (empty, single-char, max-length, digit-only, internationalised) + a larger random sample; those are where an AC rewrite diverges.
Files: `features.py` (`calc_meaningful_word_ratio`, `calc_lms_percentage`), `shared_resources.py` (dict `set` → DAWG + AC automaton).
**Groundwork already exists:** `src/compact_dict.py` (marisa-trie DAWG, hard runtime dep via `preprocess.py`) + `test_compact_dict.py` — extend, don't rebuild.
1. Build DAWG + Aho-Corasick automaton; replace the two O(m²) scans with a single-pass match.
2. **A2**: feature values bit-identical to the (expanded) snapshot across the dataset.
3. Batch-throughput before/after — the honest, provably-identical-output contribution.
**Exit:** A2 passes · batch-throughput delta recorded.

## Batch 6 — Phase 4: RQ3 edge validation  `[incl. real ARM — T1]`
**Gate:** requires Batch 4 (peak RSS 657 MiB > 512 MB — container won't load uncompressed).
1. Dockerfiles: **A 2c/512 MB · B 1c/512 MB · C 1c/256 MB**.
2. Full suite under cgroups; percentile latency curves; **<1 ms median** — **C5**.
3. Tightest profile: **idle RSS first**; ONNX-only / treelite image is the headline if sklearn won't fit — **C5**.
4. ONNX-only container bit-identical on aarch64 (QEMU functional only) — **T6 / D**.
5. **T1 real ARM**: cloud ARM VM (Ampere A1 / Graviton t4g) cgroup-constrained; run `benchmark.py`; **bound the claim** (x86+cgroup *and* ARM+cgroup). Re-confirm the model-dominated bottleneck holds on ARM — **D**.
6. Re-measure RQ1 spawn costs under Linux `fork` (Windows spawn ~0.3–0.5 s/worker inflated Batch-2 startup); re-run tl2cgen compiled-`.so` bench (deferred from the spike — import-blocked on Windows).
7. **Post-spike additions:** **ort-1.25.1 aarch64 parity check** (pinned wheel must exist and pass `test_onnx` on ARM — the ≥1.26.0 regression makes the pin load-bearing) · **DT-12 image sizing** (expect ≈50 MiB ONNX-only / ≈183 MiB sklearn-path — the Profile-C feasibility measurement).
**Exit:** <1 ms met or honest fallback · ARM result with bounded claim.

## Batch 7 — Phase 5: Robustness, generalization, reproducibility  `[feeds Final report]`
*(T3 per-family subset + B4 cross-dataset generalization pulled forward into the Batch-4 gates; Batch 7 keeps the full versions.)*
1. **T3** compression vs dictionary / PCFG DGAs — per-type breakdown — **B5**.
2. **T5** entropy-stratified evaluation (measurement, not detection claim).
3. **B3** generalization: train one set, test a different family set.
4. **Reproducibility**: pinned deps, Dockerfiles, fixed seeds, ≥10 timed runs, one regenerate script — **E**.

---

## Efficiency-levers backlog (measured basis)
| lever | metric moved | status |
|---|---|---|
| n_jobs=1 serving | single-request | **DONE (B3): 30.3→5.5 ms measured, committed** |
| ONNX Runtime | single-request + batch | Batch 4 fallback arm (18 µs shown; **ort pinned 1.25.1** — ≥1.26 regression) |
| treelite native compile | single-request + **runtime floor** | import-blocked on Windows → **Batch 6** (GTIL 73 µs benched) |
| structural pruning | RSS (473 MiB) + **ONNX load time (15–32 min!)** | Batch 4 fallback arm + C4 axis |
| float32 model | RSS (~2× model) | Batch 4, investigate-only |
| **distill → DT-12** | single-request + RSS (collapses both) | **CHOSEN (Decision 1, conditional on 4 B4 gates)** |
| drop 2 dict features (31-sweep) | RSS (−23 MiB) + deletes AC/DAWG | **REJECTED (probe 3: −2.3 pp)**; 31-sweep still runs in B4 for rigor |
| model cascade (linear gate → RF) | median single-request | mooted by DT-12 (35 µs flat) |
| AC/DAWG | **single-request latency + batch throughput** + 23 MiB | **Batch 5, CONFIRMED (Decision 2)** |
| int8 quantisation | RSS | investigate-only sub-result |
| mmap model load | RSS | mooted by DT-12 (0.6 MiB) |
