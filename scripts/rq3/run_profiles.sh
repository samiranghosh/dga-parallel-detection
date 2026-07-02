#!/usr/bin/env bash
# =============================================================================
# B6 Step 2 driver - the x86/ARM cgroup measurement grid. Run on the Docker
# host (WSL2 Ubuntu for x86; the ARM VM for T1). One script for both arches.
#
# Usage:
#   bash scripts/rq3/run_profiles.sh <output-dir> [rq1-tree-dir]
#
#   <output-dir>   host dir for all JSONs/logs (mounted at /out in containers)
#   [rq1-tree-dir] optional `git archive rq1-adaptive` export; enables the
#                  bounded RQ1 adaptive-vs-static datum under Profile A
#
# Profiles (applied at run time; images are profile-agnostic):
#   A = 2 cpus / 512 MiB    B = 1 cpu / 512 MiB    C = 1 cpu / 256 MiB
# --memory-swap == --memory makes the limit hard (no swap escape hatch).
#
# Per profile, IDLE-RSS FIRST (the C5 rule: if it doesn't fit before data,
# the fallback story starts there), then the latency/RSS grid
# {dt12, rf_pruned} x {fast, fast_marisa}, then batch throughput
# {fast, fast_marisa} (extraction-only, model-independent). Gates + the RQ1
# datum run in the full image under Profile A. Exit codes are recorded and
# the grid continues on failure - an OOM kill (rc=137) under Profile C is a
# valid result, not an abort.
# =============================================================================
set -u

FULL_IMG=${FULL_IMG:-dga-full:b6}
SERVE_IMG=${SERVE_IMG:-dga-serve-onnx:b6}
OUT=${1:?usage: run_profiles.sh <output-dir> [rq1-tree-dir]}
RQ1_TREE=${2:-}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
rec() { echo "$1 rc=$2" >> "$OUT/exit_codes.txt"; }

# ---- host + image evidence -------------------------------------------------
{
  echo "date: $(date -u +%FT%TZ)"
  echo "uname: $(uname -a)"
  echo "cpu: $(grep -m1 -E 'model name|Hardware' /proc/cpuinfo | cut -d: -f2-)"
  echo "nproc: $(nproc)"
  echo "docker: $(docker version --format '{{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})')"
  echo "kernel cgroup: $(docker info --format '{{.CgroupDriver}}/v{{.CgroupVersion}}')"
} > "$OUT/host_info.txt"
docker image inspect --format '{{.RepoTags}} {{.Id}}' "$FULL_IMG" "$SERVE_IMG" \
  > "$OUT/image_ids.txt" 2>&1

cpus_of()  { case "$1" in A) echo 2;; B) echo 1;; C) echo 1;; esac; }
mem_of()   { case "$1" in A) echo 512m;; B) echo 512m;; C) echo 256m;; esac; }

# drun <name> <profile> <kernel> <timeout_s> <image> <cmd...>
drun() {
  local name=$1 prof=$2 kern=$3 tmo=$4 img=$5; shift 5
  local cpus mem rc
  cpus=$(cpus_of "$prof"); mem=$(mem_of "$prof")
  log "run $name  [profile $prof: ${cpus}c/${mem}, kernel=$kern]"
  timeout --signal=KILL "$tmo" docker run --rm \
    --cpus="$cpus" --memory="$mem" --memory-swap="$mem" \
    -e FEATURE_KERNEL="$kern" \
    ${ORT_INTRA_OP_THREADS:+-e ORT_INTRA_OP_THREADS="$ORT_INTRA_OP_THREADS"} \
    ${RQ1_TREE:+-v "$RQ1_TREE":/rq1:ro} \
    -v "$OUT":/out \
    "$img" "$@" >> "$OUT/driver.log" 2>&1
  rc=$?
  rec "$name" "$rc"
  [ "$rc" -ne 0 ] && log "  !! $name exited rc=$rc (137 = killed: OOM or timeout)"
  return 0
}

# ---- the grid ---------------------------------------------------------------
for PROF in A B C; do
  # cgroup evidence, captured from INSIDE a container under this profile
  timeout --signal=KILL 120 docker run --rm \
    --cpus="$(cpus_of $PROF)" --memory="$(mem_of $PROF)" --memory-swap="$(mem_of $PROF)" \
    "$SERVE_IMG" python scripts/rq3/cgroup_info.py > "$OUT/cgroup_${PROF}.json" 2>>"$OUT/driver.log"
  rec "cgroup_${PROF}" $?

  # 1) idle RSS first (C5 rule)
  for KERN in fast fast_marisa; do
    for MODEL in dt12 rf_pruned; do
      drun "idle_${MODEL}_${KERN}_${PROF}" "$PROF" "$KERN" 600 "$SERVE_IMG" \
        python scripts/rq3/measure_serve.py --part idle \
          --model "models/${MODEL}.onnx" --out "/out/idle_${MODEL}_${KERN}_${PROF}.json"
    done
  done

  # 2) canonical single-request latency grid
  for KERN in fast fast_marisa; do
    for MODEL in dt12 rf_pruned; do
      drun "latency_${MODEL}_${KERN}_${PROF}" "$PROF" "$KERN" 900 "$SERVE_IMG" \
        python scripts/rq3/measure_serve.py --part latency \
          --model "models/${MODEL}.onnx" --out "/out/latency_${MODEL}_${KERN}_${PROF}.json"
    done
  done

  # 3) batch throughput (extraction-only; model-independent)
  for KERN in fast fast_marisa; do
    drun "batch_${KERN}_${PROF}" "$PROF" "$KERN" 3600 "$SERVE_IMG" \
      python scripts/rq3/measure_serve.py --part batch \
        --out "/out/batch_${KERN}_${PROF}.json"
  done
done

# ---- full image under Profile A: the five gate suites ----------------------
drun "gates_A" A fast 3600 "$FULL_IMG" \
  python -m pytest tests/test_parallel.py tests/test_onnx.py \
    tests/test_feature_kernel.py tests/test_features.py tests/test_boundary.py \
    -v --junitxml=/out/gates_A_junit.xml

# ---- bounded RQ1 adaptive-vs-static datum under Profile A -------------------
if [ -n "$RQ1_TREE" ]; then
  drun "rq1_bounded_A" A fast 5400 "$FULL_IMG" \
    python scripts/rq3/rq1_bounded.py --data-path data/ --out /out/rq1_bounded_A.json
else
  log "rq1-tree-dir not given - skipping the RQ1 bounded datum"
fi

log "grid complete; exit codes in $OUT/exit_codes.txt"
