# RQ1 Step 6 — Cost–Benefit, Stability, 2×-Peak, ≤2-core (acceptance C3)

**Setup.** Streaming load sweep on the real ExtraHop test set (`data/test.csv`),
subset = 10,000 domains, reps: k8 = 3 / k2 = 2. Dev box 8c/16t (Ryzen 7 7840HS,
Windows). Three engines — **adaptive** (1..K, proportional), **static** (K),
**sequential** (single-process) — under **uniform / Poisson-bursty / ramp**.
Reference saturation (burst, full 200k): μ(K=8) = 62,689 dom/s, μ(K=2) = 24,699.
Artifacts: `sweep_k8.json`, `sweep_k2.json`, `fig_timeseries.png`,
`fig_cost_benefit.png`, `summary.txt`.

## Headline (honest, negative-for-throughput)
On this **streaming** workload the queue-fed parallel engine is **IPC-bound**:
single-process **sequential is fastest** (~14.8k dom/s), **static-8** ~8.4k,
**adaptive** ~5.6k. Per-domain feature work (~65 µs) is too cheap to amortise the
per-batch pickle/`multiprocessing.Queue` cost, so shipping 100-domain batches
through the queue costs more than the parallelism saves. More workers help
*within* the parallel regime (static-8 > adaptive-4.5) but the whole
streaming-parallel approach trails inline sequential.

This **empirically confirms CLAUDE.md's post-measurement re-ranking**: RQ1
adaptive parallelism is not a throughput lever for *streaming*. The 78,800 dom/s
baseline came from *batch* `Pool.map` (few large chunks); batch saturation here
(μ K=8 = 62.7k > K=2 = 24.7k) still shows parallelism helping in **batch** mode —
it is the streaming/queue-fed model that is inefficient.

## (d) Three time-series + cost–benefit curve
- **`fig_timeseries.png`** — adaptive worker-count + CPU over time (uniform /
  bursty / ramp). The proportional controller ramps **1→8 monotonically**, CPU
  tracks the ramp.
- **`fig_cost_benefit.png`** — throughput(ρ) · mean-workers(ρ) · thr-per-worker(ρ):
  - *Throughput*: flat across ρ; sequential > static > adaptive.
  - *Resource cost*: static pinned at 8 workers at all loads; adaptive 4.5–5.5.
  - *Efficiency* (thr/worker): adaptive **higher** (~1,250) than static (~1,050) —
    adaptive's real win is **per-worker efficiency**, not absolute throughput.

## 2×-peak target — **HELD (no collapse)**
Throughput at ρ=2.0 ≈ throughput at ρ=1.05 for every engine (ratios 0.99–1.01):
adaptive 5,639→5,709 · static 8,430→8,386 · sequential 14,785→14,829. The engines
**saturate gracefully** under 2× peak — no throughput collapse. ✅

## Stability — **zero oscillation**
Adaptive worker-count series has **0 direction reversals** under all three
profiles (monotonic ramp to max, no thrash). The hysteresis dead-band + 0.5 s
cooldown work as intended. ✅

## Net benefit (adaptive vs static, uniform)
At ρ=0.1 throughput is **equal** (~5,431) and adaptive saves **2.5 workers**; at
ρ≥0.3 static's extra workers buy ~2,600 dom/s more throughput that adaptive
forgoes (adaptive 4.5 vs static 8.0 workers, saving 3.5). So adaptive trades
absolute throughput for resource efficiency — favourable only at **low load**.

## (e) ≤2-core edge benefit — **small/negative (valid finding)**
k=2 uniform: adaptive ~6,500 (~1.5 workers) < static-2 ~7,800 < sequential
~15,000+. On ≤2 cores adaptivity can add at most ~1 worker, so it cannot
out-throughput static-2, and both trail sequential. **Per the brief, a
small/negative edge result is a valid RQ1 finding, not a failure** — it bounds
the claim honestly.

## Bounded RQ1 claim
Adaptive parallelism's demonstrated value on this workload is **(1) resource
efficiency** (equal throughput at low load with fewer workers; higher
throughput/worker), **(2) graceful degradation** (no collapse at 2× peak),
**(3) stability** (no oscillation), **(4) fault tolerance** (A5 worker-kill /
overload recovery, Step 4). It is **not** a throughput accelerator for streaming
— sequential wins there and the batch `Pool.map` baseline (78.8k) remains the
throughput story.

## Caveats
- Reference μ is measured on the full 200k set (burst); per-ρ runs use a 10k
  subset (spawn less amortised), so absolute achieved throughput sits below μ.
  Engine-vs-engine comparison (same subset) is valid.
- Windows spawn (~0.3–0.5 s/worker) + 234k-word dictionary unpickle per worker
  inflate parallel startup — part of the honest cost. Linux `fork` would reduce
  this; to be re-checked on the ARM/edge target (Batch 5).
