---
trigger: glob
globs: Dockerfile, Dockerfile.*, *.dockerfile, docker/**, **/docker-compose*.yml
description: Edge container base image and cgroup-profiling rules for RQ3.
---

# RQ3 edge container

## Core decision
Base image = **`python:3.11-slim-bookworm`, pinned by digest**. **Not** alpine. **Not** custom/distroless.

## Rationale the agent must respect
- The cgroup profiles constrain **RAM (RSS)**, not disk image size. Alpine's ~50 MB-vs-~120 MB saving is *disk* — it does almost nothing for the 256 / 512 MB **RAM** budget that is the actual constraint.
- **Do not use alpine.** musl breaks manylinux wheels, so `numpy` / `scipy` / `scikit-learn` would compile from source (slow, fragile), and `onnxruntime` has historically shipped **no official musl wheel** (verify current status before assuming). Wrong trade-off for this exact scientific stack.
- **Do not use custom/distroless** — over-engineering for a dissertation. Revisit only if runtime RSS must be shaved further later.

## Rules
- Pin the base image **by digest** (`python:3.11-slim-bookworm@sha256:...`). Reproducibility and the tagged-baseline claim depend on this.
- Define **two profiles** via `--cpus` / `--memory`: **2c / 512 MB** and **1c / 256 MB**.
- **Always measure RSS *inside* the container under the cgroup limit** — never report image size as the memory result.
- To reduce *runtime* RSS, use the real levers: **ONNX-only serving, `marisa-trie` dictionary, lazy imports** (see the RQ2 rule) — not the base image.
- State in every RQ3 output that cgroups emulate **quantity** (cores/RAM), not microarchitecture; physical ARM hardware is **future work**.

## Do not
- Do not switch to alpine to "save space."
- Do not install build toolchains into the final image unless a wheel is genuinely unavailable; if so, use a multi-stage build and keep the runtime stage slim.
