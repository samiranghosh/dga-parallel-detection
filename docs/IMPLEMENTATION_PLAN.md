# DGA Dissertation — Implementation Plan (Claude Code)

Gated batches. Execute one, report, then the next is issued. Files in `src/` unless noted.
**Status (03 Jul):** B1 ✅ · B2 ✅ · B3 ✅ (decisions resolved) · **B4 ✅ (all 4 gates PASS → DT-12 PRIMARY locked, SamG; RF-pruned·ONNX fallback)** · B5 next (`feature-kernel`) · B6 partially pre-run (WSL2 E1) · B7 pending.
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

## Batch 3 — Phase 2: Serving-path spike  ⏳ ISSUED — `[gates Batches 4–6 · 4 cheap probes, each can delete a batch]`
Branch `rq2-serving-spike`. Deliverable: `results/spike/FINDINGS.md` + JSONs. Report-only — the two decisions below are resolved by SamG, not in-batch.
**Step 0 (on `rq1-adaptive`):** close Batch-2 gaps — A1-adaptive `allclose` equality test · A5 end-to-end worker-kill + queue-overload tests · commit rq1 figs/sweep JSONs (`.gitignore` carve-out) · land `docs/IMPLEMENTATION_PLAN.md` + fix CLAUDE.md pointer.
**Step 1:** lock `n_jobs=1` at serving (`api.py` + latency harness single-request path): 17.4 → ~4.6 ms, free.
**Step 2:** fix ONNX parity (`test_onnx`) — known root cause (skl2onnx 1.20.0 + ort 1.27.0 malformed binary-RF probas). Route: zipmap=False → documented post-normalize (iff labels `array_equal`) → pin-shift. Hard req: predictions `array_equal`, probas `allclose` rtol=1e-3.
Then measure:
**Step 3 — Probe 1 (traversal floor):** bypass sklearn's predict path (`est.tree_.predict`, float32, aggregate votes), ≥1000 warmed reps, p50/p95/p99. Ceiling for any native runtime (the 4.6 ms is Python+joblib overhead, not tree compute).
**Step 4 — Probe 2 (ONNX vs treelite):** single-request p50/p95/p99, same protocol as Batch 1. If tl2cgen native compile blocks on the Windows MSVC toolchain: bench treelite GTIL now, defer compiled-`.so` to Batch 6 (Linux/Docker). Plus staged RSS of an **ONNX-only process** (no sklearn) — the direct 256 MB feasibility number.
**Step 5 — Probe 3 (3-feature accuracy):** RF on {length, numerical_ratio, pronounceability} only (all O(1)/O(m)); stratified 5-fold CV, acc+F1 vs 93.179%. Decides whether the dictionary subsystem (23 MiB) + AC/DAWG survive.
**Step 6 — Probe 4 (distillation floor):** LogisticRegression + single DecisionTree (depth ∈ {6,8,10,12}) on 5 features; CV acc+F1 + single-request latency + model-load RSS. Decides RF-primary vs distill-primary.
**Exit:** floor · ONNX-vs-treelite winner + ONNX-only RSS stages · probe-3 delta · probe-4 table. **Report resolves the two decisions below before Batch 4.**

### `[DECISION PENDING — do not lock without spike data]`
- **RF-primary vs distill-primary.** If probe 4 ≥ ~90%, RQ2 reframes to *"the RF is over-provisioned for this feature space; a distilled model meets every edge target."* Stronger + simpler, but the RF is no longer what ships.
- **Does AC/DAWG (Batch 5) survive?** Mooted if probe 3 (3-feature) or probe 4 (distill) wins.

## Batch 4 — Phase 3: Model grid + promotion gates  ✅ DONE — `results/rq2/FINDINGS.md` + `INTERIM.md`
**Verdicts:** G1 McNemar p=1.9e-09 → DT-12 **beats** RF-100 · G2 matched-FPR PASS · G3 0/25 families PASS · G4 shift PASS-inverted (DT degrades least). **PRIMARY = DT-12; fallback = RF-pruned·ONNX (93.32%, 6.7 MB).** C4: DT-12·ONNX 64 µs / 56 MiB (≈83 MiB end-to-end) · DT-12·sklearn 102 µs · RF-pruned·ONNX 77 µs / 69 MiB — all targets met; LR fails accuracy; RF-100·ONNX DQ cold-start (1,932 s). Canonical protocol locked (RF-100 17.98 ms n_jobs=-1 / 4.31 ms n_jobs=1; spike 30.3 = machine state). int8 closed N/A. 5-feature set Pareto-front; SHAP top = lms (O(m²)) → B5 reinforced. Blind spot: dictionary-composing families evade all models (→ T3). chrmor provenance documented.
Original spec (executed):
Files: `classifier.py`, ONNX/treelite export+serve path, `api.py`. Model to compress = whichever the Batch-3 decision selects.
1. **Structural pruning** — fewer trees / depth caps / cost-complexity (50≈100 shown, E5); accuracy within ~1% — **A3**.
2. **float32 model arrays** — clean ~2× on the 473 MiB; safer than int8 for trees. **int8 = investigate-only** sub-result (trees store thresholds, not weights).
3. **Native runtime** — ONNX Runtime and/or **treelite `.so`** (per Batch-3 winner); predictions == source model — **A3**. treelite also drops the 133 MiB sklearn floor → the 256 MB enabler.
4. Rigor: F1; **McNemar + paired t-test** — **B2/B3**; exhaustive **31-subset** sweep (extends E7); SHAP for narrative.
5. Per-variant RSS + single-request latency — **C4**; **trade-off curve** (accuracy vs memory).
**Exit:** ≥3× RSS↓, ≥2× speedup, <1 ms single-request, <1% acc drop — demonstrated or honestly bounded. This now carries both RQ2 and RQ3.

## Batch 5 — Phase 2b: Aho-Corasick over DAWG  ✅ DONE — `results/kernel/FINDINGS.md`
**Verdicts:** backend = **pyahocorasick** (9.7 vs marisa 15.3 vs legacy 47.0 µs/domain; latency criterion) · **A2 PASS bit-identical** (golden v2 1,049 domains + full 999,927-domain corpus, max|Δ|=0.0) → B4 gates carry over · DT-12·ONNX **71.5→45.8 µs p50** (feature p99: sklearn arm 371→91, ONNX arm 299→110) · batch **78.9k→152.4k dom/s (1.93×)** · cost **+34.5 MiB** (end-to-end 118.9 MiB, Profile-C headroom 2.05×; marisa-mmap fallback 0.7 MiB @1.6× measured) · spawn traps documented (env-driven kernel mode; worker-build beats blob on initargs; single-copy attach on shm path) · suite 54 green.
Original spec (executed):
`[CONFIRMED + RE-PROMOTED (Decision 2 + B4) · single-request latency AND batch throughput]`
B4 measured: under DT-12 the model costs **13 µs (ONNX) / 45 µs (sklearn)** while **features cost 47–92 µs** of the 64–102 µs totals — extraction IS the single-request bottleneck; SHAP's top feature (`lms_percentage`) is itself an O(m²) scan. Value: RQ3 single-request latency + batch throughput (76 µs/dom) + ~23 MiB dict RAM. Branch: `feature-kernel` off `rq2-compression` head (needs canonical harness + DT-12 artifacts).
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
6. Re-measure RQ1 spawn costs under Linux `fork`; re-run tl2cgen compiled-`.so` bench (Windows-blocked). **Partially pre-run:** `results/wsl2/metrics.json` — E1 under WSL2: k=2 → 1.81×, IPC ~0.4% (vs 1.32% Windows). Full native-Linux/ARM still due.
**Exit:** <1 ms met or honest fallback · ARM result with bounded claim.

## Batch 7 — Phase 5: Robustness, generalization, reproducibility  `[feeds Final report]`
1. **T3** compression vs dictionary / PCFG DGAs — per-type breakdown — **B5**.
2. **T5** entropy-stratified evaluation (measurement, not detection claim).
3. **B3** generalization: train one set, test a different family set.
4. **Reproducibility**: pinned deps, Dockerfiles, fixed seeds, ≥10 timed runs, one regenerate script — **E**.

---

## Efficiency-levers backlog (measured basis)
| lever | metric moved | status |
|---|---|---|
| n_jobs=1 serving | single-request (17.4→4.6 ms) | lock in Batch 3 |
| ONNX Runtime | single-request + batch | Batch 4 |
| treelite native compile | single-request + **runtime floor (256 MB)** | Batch 3 bench → Batch 4 |
| structural pruning | RSS (473 MiB) | Batch 4 |
| float32 model | RSS (~2× model) | Batch 4 |
| distill → DT-12 | single-request + RSS | **PRIMARY (B4 gates passed)** |
| drop 2 dict features (31-sweep) | RSS (−23 MiB) + deletes AC/DAWG | probe 3 → decision |
| model cascade (linear gate → RF) | median single-request | optional hedge |
| AC/DAWG | **single-request features + batch throughput** (costs +34.5 MiB, absorbed) | **DONE (B5)** — 45.8 µs / 1.93× |
| int8 quantisation | — | **CLOSED N/A** (B4: no weight matrices / no int8 TreeEnsemble op) |
| mmap model load | RSS | marginal; note only |
