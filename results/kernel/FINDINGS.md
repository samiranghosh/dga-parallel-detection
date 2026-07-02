# Batch 5 — Aho-Corasick Feature Kernel (RQ1/RQ3 extraction path): FINDINGS

**Setup.** Branch `feature-kernel` off `rq2-compression` head (`eb7758d`).
Machine: 8c/16t Ryzen 7 7840HS, Windows 11, Python 3.11, sklearn 1.8.0,
onnxruntime 1.25.1, pyahocorasick 2.3.1 (pinned this batch), marisa-trie 1.4.1.
Model under serving measurements: **DT-12** (B4-gated primary; RF-pruned·ONNX
fallback untouched). Scope guard honoured: no model/classifier/api changes;
gates (`test_parallel`, `test_features`, `test_boundary`, `test_onnx`) green
throughout — full suite **54 passed** at head.
Artifacts: `bench_backends.json`, `attach_paths.json`, `a2_full_corpus.json`,
`step5_single_request.json`, `step5_batch.json`, `step5_memory.json`;
oracle `tests/golden_features_v2.json`; scripts under `scripts/kernel/`.

## Step 0 — Golden oracle v2 (built BEFORE any kernel code)
`tests/golden_features_v2.json`: **1,049 domains** = 20 v1 curated + 29 A4 edge
cases (empty, single-char, real+synthetic 63-char max-length, digit-only,
hyphenated, punycode `xn--…`, raw-unicode IDN, unicode digits — `'٣'.isdigit()`
is True, repeated-word compounds `paypalpaypal`, zero-hit strings, case
probes, dotted FQDN) + 1,000 stratified train+test sample (seed 42).
**0 quarantined** — legacy crashed on no edge case; regeneration verified
bit-identical (`--check`). Generator: `scripts/kernel/make_golden_v2.py`.

Two measured dictionary facts that make the oracle non-trivial:
- **all 26 single letters are nltk dictionary words** → under THIS dictionary
  `meaningful_word_ratio` degenerates to alpha-coverage (e.g. `qqqq` → 1.0);
  the kernel must reproduce that via the same rules, not special-case it;
- matching is **case-sensitive** (`GOOGLE` → 0.0); `google` is *not* an nltk
  word (`the`, `cat`, `pay` are).

## Step 1 — Backend choice (measured, not assumed)
10k-domain bench (seed 42, 3 reps) of the two features' access pattern; all
three paths cross-checked **equal on every domain**:

| path | µs/domain (mwr+lms) | RSS (fresh proc) | build/load | aarch64 cp311 wheel |
|---|---|---|---|---|
| legacy set-scan | 47.0 | 22.6 MiB | 0.08 s | — |
| **pyahocorasick 2.3.1** | **9.7** | 35.3 MiB | 0.23 s build | ✓ manylinux2014 |
| marisa prefix-sweep | 15.3 | **0.7 MiB (mmap)** | **5 ms** | ✓ manylinux_2_28 |

**Winner on latency (the brief's criterion): pyahocorasick** — 4.8× vs legacy,
1.6× vs marisa; not a tie, so tie-breakers don't fire. The trade-off is
memory: the automaton structure is +12.7 MiB vs the set — but since the set
stays resident beside it (Step 5), the *deployed* serving cost is the full
+34.5 MiB; marisa would instead *return* ~22 MiB. Both aarch64 wheels verified (marisa needs
glibc ≥ 2.28 — fine on any current ARM distro). Pinned `pyahocorasick==2.3.1`
in requirements.txt with rationale; the marisa path stays available via
`src/compact_dict.py` as the memory-optimal option for tighter-than-256 MB
profiles.

## Step 2 — Extracted parity rules (from code, not intuition)
Documented above the implementation in `src/features.py` (R1–R5):
1. **Candidate set** = every occurrence of every dictionary word as a
   substring; case-sensitive; no normalisation; **no minimum word length**
   (single letters count). AC all-occurrence iteration ≡ this set.
2. **lms** = len(longest match)/len(domain); 0.0 for empty/no-match.
3. **mwr** = max characters coverable by **non-overlapping** matches /
   len(domain) — a weighted-interval DP; abutting allowed, overlap not.
   The legacy carry-forward form is equivalent to the standard end-indexed
   recurrence (covered[i] is final before it propagates).
4. **Denominator** = raw len(domain), non-alpha chars included.
5. Both values are exact int/int quotients → identical integers give
   **bit-identical** IEEE-754 doubles (no tolerance needed).

Edge found during implementation: pyahocorasick cannot finalise a
**zero-word automaton**; empty dictionary is guarded to legacy semantics
(no matches). Implementation: `FEATURE_KERNEL=legacy|fast` switch,
one automaton pass yields *both* features in `extract_features`.

## Step 3 — A2 equivalence gate: PASS (all three layers)
- **Golden v2**: bit-identical under both kernels (1,049 domains incl. edges).
- **Full corpus** (train+test, **999,927 domains**): `max|Δ| = 0.0` for both
  features; 0 non-identical values (`a2_full_corpus.json`). Sweep itself:
  legacy 49.7 s vs fast 14.7 s.
- **`tests/test_feature_kernel.py`** (9 tests): golden both modes, fresh 10k
  seed-43 sample, rule-corner small dictionaries (incl. empty dict), vector
  dtype parity, shm end-to-end.
Default flipped to `fast` only after all three passed; `legacy` selectable
(env var or `set_kernel_mode`) for A/B and rollback.

## Step 4 — Worker attach (measured wiring)
Attach-path bench (`attach_paths.json`): per-worker rebuild from set
**0.50 s** vs pickle-attach **0.06 s** (8×) vs pyahocorasick native
save/load 0.71 s.

**Measured refinement (Step 5 batch run):** for the spawn-initargs path,
shipping the 26.5 MiB serialized automaton to each worker costs more than it
saves — worker builds run **in parallel** during pool startup while the blob
transfer serializes in the parent (152.4k dom/s worker-build vs 147.6k
blob-attach). Defaults are therefore: **initargs path = per-worker warm-build
at init** (cost lands in startup, not the first chunk); **shm path = true
single-copy attach** (`SharedMemoryResources.create(automaton_blob=…)` /
`attach_automaton`, one OS-shared block, no per-worker parent cost) — the
`_init_worker_shm` signature is unchanged and `test_parallel.py` plus a new
end-to-end shm test prove the API.

Why not mmap: **pyahocorasick has no mmap**, so a per-worker private heap
copy (~35 MiB) after attach is inherent to the AC backend; the only true
zero-copy option is the marisa DAWG (0.7 MiB mmap, 5 ms) at 1.6× kernel
latency — that documented trade-off is the answer to the brief's
"wire mmap or document why". On Linux `fork` (Batch 6) the parent-built
automaton is inherited copy-on-write for free; re-measure there.

Spawn caveat (documented in `set_kernel_mode`): workers re-import
`features` and read `FEATURE_KERNEL` from the **environment** — the setter
is process-local. The first batch A/B was invalid for exactly this reason
(both arms ran fast workers); re-run with env propagation.

## Step 5 — Before/after (canonical protocol + batch + memory)
**Single-request** (CANONICAL-B4: 2,000 test.csv domains · 200 warm-up ·
1,000 reps · n_jobs=1 · idle box; same-session before/after):

| arm | total p50 | feature p50/p95/p99 | predict p50 | total p95/p99 |
|---|---|---|---|---|
| DT-12·sklearn · legacy | 136.6 µs | 74.0 / 226.3 / 370.9 | 54.0 | 365.4 / 548.8 |
| DT-12·sklearn · **fast** | **74.2 µs** | **28.5 / 57.1 / 90.8** | 44.1 | 147.6 / 285.6 |
| DT-12·ONNX · legacy | 71.5 µs | 56.8 / 173.1 / 298.6 | 13.3 | 199.6 / 334.1 |
| DT-12·ONNX · **fast** | **45.8 µs** | **30.6 / 63.9 / 110.4** | 13.6 | 92.2 / 159.3 |

- End-to-end totals drop **1.56× (ONNX) / 1.84× (sklearn)**; the brief's
  expectation (toward the ~13–45 µs model floor) is met: DT-12·ONNX lands at
  **45.8 µs**, with the model floor (13.6 µs) now ~30% of it.
- **The big win is the tail**: feature p99 371→91 µs (4.1×, sklearn arm;
  ONNX arm 299→110, 2.7×) — O(m²) cost concentrates in long domains, i.e.
  exactly the p95/p99 region; the kernel linearises it. Total p99 549→286
  (sklearn) and 334→159 (ONNX).
- Legacy totals today (136.6/71.5) sit above B4's session (102/64): run-to-run
  machine state; before/after within THIS session is the valid comparison,
  and B4's absolute numbers remain the C4 record.
- Residual fast-kernel feature cost ~29–31 µs ≈ trigram pronounceability +
  Python call overhead — the next bottleneck if anyone chases sub-30 µs.

**Batch** (Pool.map k=8, full 999,927-domain corpus, 3 reps, wall incl.
spawn; kernel mode via env):

| arm | mean dom/s | walls (s) |
|---|---|---|
| legacy | 78,862 | 12.85 / 12.45 / 12.75 |
| fast (blob attach) | 147,606 | 7.02 / 6.59 / 6.72 |
| **fast (worker build, default)** | **152,402** | 6.59 / 6.53 / 6.56 |

Legacy reproduces the **78.8k dom/s baseline** almost exactly — protocol
validated. Fast kernel: **1.93×** batch throughput. Outputs asserted
`array_equal` across all arms.

**Memory / load** (staged fresh-process serving stacks, DT-12):

| stack | total RSS | stages (MiB) | stack load |
|---|---|---|---|
| ONNX · legacy | 84.4 MiB | interp 21.8 · session 57.6 · dict 80.6 · trigram 84.4 | 0.42 s |
| ONNX · **fast** | **118.9 MiB** | … + automaton → 115.3 · trigram 118.9 | 0.48 s |
| sklearn · legacy | 181.4 MiB | model stage 153.9 · dict 177.0 · trigram 181.4 | 2.3 s |
| sklearn · **fast** | 216.2 MiB | … + automaton → 212.4 · trigram 216.2 | 2.3 s |

- Kernel cost: **+34.5 MiB on the ONNX stack / +34.8 on the sklearn stack**
  (automaton), attach 0.06 s from a prebuilt blob. ONNX·legacy 84.4 concords
  with B4's ≈83 end-to-end.
- **Profile C (244 MiB)**: DT-12·ONNX·fast = 118.9 MiB → headroom 2.9×→**2.05×**;
  every RQ3 profile still fits. If a future profile is tighter than ~128 MiB,
  switch the kernel to the marisa backend (0.7 MiB) and pay 1.6× kernel
  latency — both options are now measured.
- The dict **set stays resident** (23 MiB) beside the automaton: the feature
  API keys the automaton cache on the dictionary object (and legacy/rollback
  needs it). An automaton-only serving floor (~96 MiB) would need a small API
  affordance — noted, not done (scope).

## The Batch-5 story
Under the DT-12 primary, extraction was the single-request bottleneck
(B4: DT-12's feature p50 was 49.6–53.5 µs of its 63.5–102.3 µs totals, i.e.
52–78% — the 47–92 µs span quoted in B4 was grid-wide across all four
models; SHAP top feature `lms_percentage` is an O(m²) scan). One C
Aho-Corasick pass + an O(m+matches) DP replaces both O(m²) features with
**provably identical output** — bit-identical on a 1,049-domain edge-case
oracle and on the full 999,927-domain corpus — cutting single-request
totals 1.56× (ONNX) / 1.84× (sklearn) — DT-12·ONNX **45.8 µs p50**, total
tails compressed 1.9–2.5×, feature tails 2.7–4.1× — and batch throughput
to **152k dom/s (1.93×)** for +35 MiB RSS,
which every RQ3 profile absorbs (118.9 MiB end-to-end, 2.05× Profile-C
headroom). The identical-output claim is the contribution's spine: no
accuracy re-validation is needed because the feature values — and therefore
the model inputs — are unchanged bits.

## Caveats (honest)
- **Windows-only measurements**; spawn semantics drove the attach-path
  choice. On Linux `fork` (Batch 6) the automaton is inherited CoW — the
  worker-build vs blob trade-off disappears and per-worker RSS may improve;
  re-measure. aarch64 wheels verified for both backends but not yet run (B6).
- Batch arms include pool startup (Windows spawn; not separately
  instrumented in `step5_batch.json`) — steady-state fast-kernel throughput
  is higher than the 152k figure; the ratio 1.93× is conservative.
- Legacy absolute latencies this session (136.6/71.5 µs) differ from B4's
  (102/64) by machine state; cross-session comparisons must use ratios.
- The automaton assumes the dictionary is static (it is: preprocess
  artifact). Any dictionary change requires automaton rebuild — handled by
  construction (built from the live dictionary object, never cached to disk
  in production paths).
- `_AUTOMATON_CACHE` keys on `(id(dictionary), len)`; sets aren't
  weakref-able. Stale-hit risk needs id reuse at equal length within one
  process — implausible in these processes (one dictionary for life), noted
  for reviewers.
