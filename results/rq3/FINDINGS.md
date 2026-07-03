# RQ3 Edge Validation — Batch 6 FINDINGS

Branch `rq3-edge` (off `feature-kernel` head). Evidence files referenced
throughout live in `results/rq3/` (x86 grid under `x86/`). Canonical
single-request protocol = `src/latency_harness.py` constants (2,000
`test.csv` domains in file order, 200 warm-up, 1,000 timed reps,
`perf_counter_ns`, serving n_jobs=1), run **inside** the profile container.

## Verdict summary

- **C5 (<1 ms median per-domain inside every profile): MET on x86+cgroup,
  with 20–40× margin.** All 12 grid cells ({DT-12·ONNX, RF-pruned·ONNX} ×
  {fast, fast_marisa} × {A, B, C}) sit at 24.0–50.3 µs p50; the worst
  p99 anywhere is 363 µs — still 2.7× inside the target. No fallback story
  needed: idle RSS fits every profile before data arrives (idle-first rule).
- **T6 (aarch64 functional parity): see §7** — wheel availability for the
  pinned serving stack (ort 1.25.1, pyahocorasick 2.3.1, marisa-trie 1.4.1,
  numpy 2.4.6) is confirmed on aarch64; QEMU is correctness-only by policy.
- **T1 (real ARM): NOT RUN — blocked on cloud account.** The complete
  copy-paste procedure is `scripts/rq3/ARM_RUNBOOK.md`; the identical
  driver, images and parity gate are committed and exercised on x86. Until
  those numbers land, every claim below is bounded to x86-under-cgroup.

## 0. Step-0 kernel affordances (prerequisite for the RSS story)

`FEATURE_KERNEL` now has three modes (`legacy` / `fast` / `fast_marisa`) and
a `DictKernel` handle that replaces the raw word set at serving time, so the
23 MiB set can be released (`initialize_serving_resources`). A2 re-asserted
after the rework: **golden-v2 (1,049 domains) bit-identical for all three
modes**, for the handle path, and for the mmapped `.marisa` artifact;
suite 66-green on Windows (`tests/test_feature_kernel.py`, 21 tests).

Serving RSS, Windows host, DT-12·ONNX end-to-end (`step0_rss.json`):

| stack | RSS | resource load |
|---|---|---|
| set retained (B5 status quo) | 121.4 MiB | 0.45 s |
| fast, kernel-only (Step 0a) | **101.6 MiB** (−19.8) | 0.44 s |
| fast_marisa, kernel-only (Step 0b) | **63.0 MiB** | **0.007 s** |

(The fast drop is ~20 MiB rather than the naive 23: the allocator keeps a
few MiB of pages after the set is freed.)

## 1. Images (E)

`python:3.11-slim-bookworm` pinned by **multi-arch index digest**
`721dc13f…` (2026-06-24 build) — one FROM line for x86-64 and aarch64.
Two images, profiles applied only at `docker run`:

- `dga-full:b6` (1.44 GB, image id `f92e977b…`): full pinned env
  (requirements.txt), tests, gcc (tl2cgen).
- `dga-serve-onnx:b6` (436 MB, id `5df76db7…`): requirements-serve.txt
  ONLY (onnxruntime 1.25.1, numpy 2.4.6, pyahocorasick 2.3.1,
  marisa-trie 1.4.1) — no sklearn/pandas/joblib/psutil. Code, both ONNX
  models (sha256 manifest `models/manifest.json`), preprocess artifacts,
  golden-v2 and the x86 parity snapshot are baked in: the image is
  self-contained for ARM.

Profile limits are captured **from inside** every run (`cgroup_{A,B,C}.json`
+ embedded in every measurement JSON): A = quota 200000/100000 + 512 MiB,
B = 100000/100000 + 512 MiB, C = 100000/100000 + 268435456 bytes, swap
capped equal to memory (no swap escape). Host: AMD Ryzen 7 7840HS, WSL2
kernel 5.15.167.4, Docker 29.6.1, **cgroup v1**.

Build-infra note (recorded because it cost three build attempts): the WSL2
NAT link reliably breaks >16 MiB PyPI transfers; the committed Dockerfiles
retry pip inside one layer with a cache mount, and `wheels/` accepts
natively pre-fetched platform wheels (`pip download --platform …`) for the
arm64 builds.

## 2. Idle RSS first (C5 rule) — x86, serve-onnx, kernel-only serving

Idle = after model session + serving resources + first inference
(`idle_*.json`; VmRSS/VmHWM from `/proc`). Values are profile-independent
(≤ ±3 MiB across A/B/C), quoted from Profile C — the 244 MiB worst case:

| model | fast (idle / peak) | fast_marisa (idle / peak) |
|---|---|---|
| DT-12·ONNX | 105.3 / 126.4 MiB | **68.3 / 68.3 MiB** |
| RF-pruned·ONNX | 138.6 / 159.4 MiB | 99.7 / 99.7 MiB |

Every cell fits Profile C **before data**; worst-case headroom 1.5×
(RF-pruned/fast peak), best 3.6× (DT-12/marisa). Resource load: marisa
mmap ~7 ms vs AC build ~0.48 s per process.

## 3. Canonical single-request latency — x86 grid (µs)

| profile | model | kernel | p50 | p95 | p99 | feat p50 | pred p50 |
|---|---|---|---|---|---|---|---|
| A | DT-12 | fast | 35.7 | 67.6 | 121.9 | 25.9 | 8.9 |
| A | RF-pruned | fast | 41.9 | 77.2 | 133.9 | 26.2 | 15.1 |
| A | DT-12 | marisa | 40.9 | 72.9 | 175.5 | 31.2 | 9.4 |
| A | RF-pruned | marisa | 50.3 | 99.9 | 191.5 | 33.4 | 16.3 |
| B | DT-12 | fast | **24.0** | 50.6 | 80.1 | 17.7 | 6.0 |
| B | RF-pruned | fast | 55.4 | 114.7 | 294.0 | 31.0 | 24.1 |
| B | DT-12 | marisa | 32.1 | 72.6 | 151.8 | 25.3 | 6.3 |
| B | RF-pruned | marisa | 50.6 | 104.6 | 362.8 | 32.6 | 15.0 |
| C | DT-12 | fast | 25.3 | 53.7 | 107.1 | 18.9 | 6.1 |
| C | RF-pruned | fast | 42.9 | 82.6 | 196.2 | 25.7 | 16.0 |
| C | DT-12 | marisa | 33.6 | 71.7 | 126.8 | 25.9 | 6.6 |
| C | RF-pruned | marisa | 47.3 | 95.0 | 161.4 | 29.8 | 14.4 |

- **C5 met everywhere**; median 20–40× inside 1 ms, p99 ≥2.7× inside.
- **Bottleneck structure holds under cgroups:** features are 65–75% of the
  total in every cell (B5's DT-12 framing, unchanged).
- The container p50s are *faster* than the Windows-host B5 numbers
  (DT-12·fast 24–36 vs 45.8 µs) — Linux syscall/timer overhead is lower;
  cross-OS absolute comparisons stay out of the claims.
- RF-pruned's p99 fattens on 1-core profiles (294/363 µs) — consistent
  with ORT's host-core-sized threadpool spinning against the quota. The
  opt-in `ORT_INTRA_OP_THREADS=1` knob exists to A/B this on ARM; medians
  were unaffected, so it was left at default for comparability.

## 4. Batch throughput (Pool.map, full 999,927-domain corpus, k=cgroup cores)

| profile | fast | fast_marisa |
|---|---|---|
| A (2c/512m) | **OOM-killed** | 73,303 dom/s (k=2) |
| B (1c/512m) | 48,007 dom/s (k=1) | 38,619 dom/s (k=1) |
| C (1c/256m) | **OOM-killed** | **OOM-killed** |

Honest bounded finding: single-request serving fits every profile, but the
*batch harness itself* (whole corpus resident in the parent + per-worker
automaton copies under `fast`) does not fit 512 MiB at k=2 (A/fast) or
256 MiB at all (C). The kills are memory-cap kills of an in-process 1M-row
protocol, not a serving failure; a streaming/chunked feed is the fix and is
RQ1 territory, not claimed here. Corollary: under tight RAM the marisa
kernel is also the *batch* survivor (0.7 MiB/worker vs ~35 MiB/worker).

## 5. Gates + bounded RQ1 datum under Profile A

- **Gates (full image, 2c/512m): 52 passed, 0 failed** — test_parallel,
  test_onnx, test_feature_kernel, test_features, test_boundary
  (`gates_A_junit.xml`).
- **RQ1 C3-under-cgroup datum** (`rq1_bounded_A.json`; rq1-adaptive tree
  as-is, uniform load, ρ∈{0.5,0.9}, reps=2, 30k subset, μ(k=2)=24.1k dom/s):
  at ρ0.5 all engines track the 12k offered; at ρ0.9 static-2 ≈ 21.5k
  (tracks offered), adaptive 19.6k (mean workers 1.65 — scaling lag),
  sequential saturates ≈16.8k. **Adaptive < static at 2 cores — B2's
  bounded ≤2-core finding reproduces under a real cgroup.** Nuance vs B2
  (Windows): under Linux fork, static-2 *beats* sequential at high load;
  B2's sequential-on-top ordering was partly a spawn-cost artifact.
- Archival gap found: the `rq1-adaptive` branch references
  `src/compact_dict.py` but never committed it (the file predated B5 in
  working trees, was first committed on the B5 line). Supplied from
  `feature-kernel` head for this run; recorded here for the repro chapter.

## 6. Linux-fork re-checks (Step 3, `fork_recheck.json`) + tl2cgen (Step 4)

| measure | Windows (B2/B5) | Linux fork (container) |
|---|---|---|
| pool startup / worker (k=8) | 0.3–0.5 s | **0.065 s** (k=1: 0.51 s) |
| AC build from set | 0.50 s | 0.43–0.46 s |
| AC pickle-loads attach (26.5 MiB blob) | 0.06 s | 0.054 s |
| AC native save/load | — | 0.53 s |
| marisa build / mmap (1 process) | — | 0.14 s / **0.0001 s** |
| kernel-mode propagation to workers | env var ONLY (spawn re-imports) | **`set_kernel_mode()` propagates** (fork inherits module state) |

Ranking unchanged (blob-attach < build < save/load); what changes is the
constant: fork amortises worker startup ~5–8× and removes the env-only
constraint (env remains the portable mechanism and stays the documented
interface).

**tl2cgen compiled `.so` (Windows-blocked since B3), RF-pruned, canonical
protocol, labels array_equal across all four runtimes**
(`tl2cgen_bench.json`): sklearn n_jobs=1 **3,950 µs** p50 → GTIL **544** →
compiled `.so` **186** (predict 127; compile 17.1 s, 2.4 MiB) → **ONNX
39.7 (predict 13.5)**. Data point only, and it *reinforces* the B4
adjudication: ONNX beats the compiled library ~9× at batch=1 (per-request
DMatrix construction dominates the compiled path).

## 7. T6 — aarch64 functional parity (QEMU, correctness ONLY): **PASS**

Wheel availability on aarch64/cp311 for the pinned serving stack is
**confirmed by download**: onnxruntime 1.25.1, pyahocorasick 2.3.1,
marisa-trie 1.4.1, numpy 2.4.6 (all manylinux aarch64). The serve image
builds for `linux/arm64` from the same digest-pinned FROM line.
**Exception: tl2cgen 1.0.0 publishes NO aarch64 wheel** — the arm64 full
image is defined by `requirements-arm64.txt` (identical minus that pin;
nothing in tests/runtime imports it), via
`--build-arg REQUIREMENTS=requirements-arm64.txt`.

Parity (`parity_check.py --check` vs the committed x86 snapshot, B3
criteria) on aarch64 under QEMU — `parity_qemu_arm64.json`: **PASS.**
Golden-v2 A2 exact for all three kernels; canonical 2,000-row
feature-matrix sha256 **equal per mode** (feature computation is
bit-identical across ISAs); DT-12 and RF-pruned labels `array_equal` and
probas **max|Δ| = 0.0** — the ort 1.25.1 TreeEnsemble output is
bit-identical on aarch64. **No timing numbers under QEMU by policy**
(Rejected list: QEMU for latency/throughput).

**Scope correction (close-out audit): the full pytest gate suite was NOT
run under QEMU.** The arm64 *full* image never built on the WSL2 host —
the pip layer died on the NAT link after 5 in-layer retries
(`logs/arm_full_build.log`) and the native wheel pre-fetch for the full
dependency set failed a hash check on a truncated download
(`logs/wheel_dl_full.log`). `gates_qemu_arm64_junit.xml` therefore does
not exist; an earlier draft of this file cited it in error. T6's verdict
is unaffected — its criteria are the parity checks above, which ran in
the self-contained serve image (`logs/arm_serve_build.log`,
`logs/parity_qemu.log`). The full gate suite on aarch64 runs natively on
real ARM per `scripts/rq3/ARM_RUNBOOK.md` (T1).

## 8. Kernel recommendation (pyahocorasick vs marisa — SamG decides)

- **Profile A/B (512 MiB):** `fast` (pyahocorasick). Lowest p50 everywhere
  it fits; RSS 105–139 MiB leaves ≥3.5× headroom at 512 MiB.
- **Profile C (256 MiB) or anything tighter / batch-heavy under RAM
  pressure:** `fast_marisa`. 63–100 MiB idle, 7 ms cold-start, survives
  batch at A where fast OOMs; costs ~25–30% p50 (still ~30× inside C5).
- DT-12 + fast_marisa is the minimal pair: 68 MiB end-to-end, 33.6 µs p50
  under Profile C.

## 9. Claim bounding (drafted for the report/viva)

> "Sub-millisecond median per-domain detection latency (24–50 µs p50,
> p99 ≤ 363 µs) and ≤160 MiB serving RSS are demonstrated **under x86
> core/memory cgroup limits (2c/512MB, 1c/512MB, 1c/256MB)**. aarch64
> functional parity (bit-identical features on the golden oracle,
> label-equal model output) is verified under emulation and the identical
> pinned images/driver are committed for server-class ARM; **the ARM+cgroup
> latency/RSS grid is pending hardware access (Oracle Ampere A1 /
> Graviton), and embedded-class ARM (e.g. Raspberry Pi) is future work.**
> All timing claims derive from real hardware; QEMU was used for
> correctness verification only."

## 10. Deviations & notes

- `docker --memory-swap == --memory` (hard cap, no swap): OOM-kill of the
  batch harness under A/fast and C is the recorded behaviour, and the three
  killed runs burned their 1 h timeout in reclaim-thrash first — deploy
  configs should set both.
- Grid ran on images `f92e977b…`/`5df76db7…`; a one-line guard was added to
  `build_kernel` afterwards (marisa_path implies fast_marisa) for Step-3
  tooling — no measured path touched.
- serve-onnx ships without FastAPI (spec: "api optional") — the C5 metric
  is in-process; HTTP stack cost is a separate, unclaimed number.
- RQ1 numbers here are a bounded spot-check datum, not a re-run of the B2
  sweep; the B2 conclusions stand unmodified.
- Build/parity logs from the WSL2 host are committed under
  `results/rq3/logs/` (arm serve/full builds, wheel pre-fetch, QEMU
  parity run). The x86 grid driver log and x86 full-image build log
  remain on the WSL2 host (`/root/b6/`) — grid provenance is already
  captured in `x86/host_info.txt`, `x86/image_ids.txt`,
  `x86/exit_codes.txt` and the per-cell JSONs.
