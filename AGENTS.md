# AGENTS.md — DGA Parallel Detection (MTech Dissertation)

@CLAUDE.md

**All project facts live in `CLAUDE.md`** (research questions, layout,
baselines, measured results per batch, env pins, rejected directions,
critical rules). It is the single source of truth; do **not** restate its
content here — a stale duplicate of these facts in this file has already
caused drift once.

## Agent-behavior extras (not in CLAUDE.md)

- Recommended autonomy profile: **Review-driven development**. Do **NOT**
  run Agent-driven / auto-continue on changes to parallel, shared-memory,
  or benchmark code (`src/parallel_engine.py`, `src/shared_resources.py`,
  `src/chunker.py`, `src/benchmark*.py`).
- **Never fabricate, hardcode, or approximate benchmark numbers.** Produce
  the measurement code; the human runs it and records the numbers. All
  timing/RSS deliverables follow the canonical protocol cited in CLAUDE.md.
- **Do not auto-commit.** Leave commits to the human (Conventional Commits
  if asked).
- Plan first for any multi-file or architectural change and wait for human
  approval before executing. Smallest possible diff on concurrency code;
  explicitly flag race conditions and the copy-on-write RAM trap.
- The human must defend every line in a viva — prefer clarity over
  cleverness, and say so explicitly if you generate something the human
  likely can't explain.

## Scoped rules & workflows

- `.agents/rules/rq1-load-benchmark.md` — RQ1 load-benchmark protocol
  (utilization sweep, never feed the batch rate).
- `.agents/rules/rq2-serving-backend.md` — pluggable sklearn/ONNX backend
  rules (glob-triggered on serving files).
- `.agents/rules/rq3-edge-container.md` — base-image + cgroup-profiling
  rules (glob-triggered on Dockerfiles).
- `.agents/workflows/benchmark.workflow.md` — `/benchmark` measurement
  sequence (review-driven; pause between steps).
