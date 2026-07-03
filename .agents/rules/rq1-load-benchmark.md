---
trigger: model_decision
description: Apply when designing, generating, or running the RQ1 adaptive-vs-static load benchmark, or any arrival-rate / throughput experiment.
---

# RQ1 load-benchmark protocol

## Core decision
The "uniform" baseline is a **constant arrival rate swept across utilization** — NOT a single rate, and NOT the batch-throughput number.

## Rules
- **Never feed the engine at ~78,800 domains/s.** That figure is *batch* throughput (workers chewing a pre-loaded list). It is not a streaming arrival rate and will over-drive a queue-fed engine.
- **First measure `μ`** = the engine's *streaming* saturation throughput on the **current host**, with per-item dispatch through the queue (not batch). Re-measure on every machine; do not reuse the original 8-core laptop number.
- **Express every arrival rate as utilization `ρ = λ / μ`** and report results in `ρ`, not absolute domains/s. This is the queueing-correct, hardware-reproducible framing.
- **Uniform profile = constant `λ`; sweep `ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` plus one brief overload `ρ ≈ 1.05`** to show graceful backpressure.
- At each `ρ`, run **both** the adaptive engine and **static K=8** on identical traffic.
- **Primary metric at low `ρ` = mean active worker count** (resources saved by scaling down), reported alongside **p95 / p99 latency**. The deliverable is a *curve* over `ρ`, not a single throughput number.
- Expected story: low-`ρ` end shows adaptive retires to 1–2 workers while static holds 8; high-`ρ` end shows adaptive ≈ static (it does not *hurt* at full load). Both ends are valid results.
- Beyond uniform, also run **bursty-Poisson** and **ramp** profiles.

## Reporting
- Report **mean ± SD with confidence intervals** over ≥5 repetitions.
- Always separate **single-request latency** from **amortised batch throughput** — never conflate them.
- Account for monitoring cost and worker spawn/teardown overhead in the adaptive numbers.
- Output raw data to a file; do **not** write conclusions or final numbers into report prose (the human runs and records them).
