"""B6 Step 3: Linux-fork re-checks of the B5 spawn traps (run in `full` image).

Windows measured (B5 / known issues): spawn ~0.3-0.5 s/worker + dict
unpickle; automaton build-from-set 0.50 s vs blob-attach 0.06 s; workers
read FEATURE_KERNEL from the env because spawn re-imports modules.
Under Linux fork all three change shape - this script measures:

  1. pool_startup   Pool(k) creation + trivial map, k in {1,2,4,8}: the
                    per-worker startup cost under fork (initargs are
                    inherited, not pickled).
  2. attach_paths   B5 attach-path ranking re-run (build / pickle-loads /
                    native save-load), in-process, same protocol as
                    scripts/kernel/attach_bench.py.
  3. env_propagation  under fork a worker inherits the parent's ALREADY
                    IMPORTED features module, so set_kernel_mode() DOES
                    reach workers (unlike spawn). Verified empirically.
  4. build_1core    automaton build + marisa build/mmap time (relevant to
                    per-worker warm cost when the cgroup grants 1 core).

Output: results JSON to --out (stdout otherwise).
"""
import os
import sys
import json
import time
import pickle
import argparse
import multiprocessing

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(REPO, "data")
REPS = 5


def _timed(fn, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"mean_sec": round(sum(times) / len(times), 4),
            "reps_sec": [round(t, 4) for t in times]}


def _noop(_):
    return os.getpid()


def _worker_mode(_):
    from src.features import get_kernel_mode
    return get_kernel_mode()


def part_pool_startup(dictionary, ngram):
    from src.parallel_engine import _init_worker
    out = {}
    for k in (1, 2, 4, 8):
        def start_pool():
            with multiprocessing.Pool(
                    processes=k, initializer=_init_worker,
                    initargs=(dictionary, ngram, True, None)) as pool:
                pool.map(_noop, range(k))
        r = _timed(start_pool, reps=3)
        r["per_worker_sec"] = round(r["mean_sec"] / k, 4)
        out[f"k{k}"] = r
        print(f"pool k={k}: {r['mean_sec']}s total, {r['per_worker_sec']}s/worker")
    return out


def part_attach_paths(dictionary):
    import ahocorasick

    def build():
        a = ahocorasick.Automaton()
        for w in dictionary:
            a.add_word(w, len(w))
        a.make_automaton()
        return a

    automaton = build()
    blob = pickle.dumps(automaton, -1)
    save_path = "/tmp/english_dictionary.automaton"
    automaton.save(save_path, pickle.dumps)
    return {
        "build_from_set": _timed(build),
        "pickle_loads": {**_timed(lambda: pickle.loads(blob)),
                         "blob_mib": round(len(blob) / 2**20, 1)},
        "native_saveload": {**_timed(lambda: ahocorasick.load(save_path, pickle.loads)),
                            "file_mib": round(os.path.getsize(save_path) / 2**20, 1)},
    }


def part_env_propagation():
    """set_kernel_mode WITHOUT touching the env; do forked workers see it?"""
    from src import features
    assert multiprocessing.get_start_method() == "fork"
    features.set_kernel_mode("legacy")  # env still says fast (or unset)
    with multiprocessing.Pool(2) as pool:
        worker_modes = pool.map(_worker_mode, range(2))
    features.set_kernel_mode("fast")
    return {
        "start_method": "fork",
        "parent_setter_mode": "legacy",
        "env_FEATURE_KERNEL": os.environ.get("FEATURE_KERNEL"),
        "worker_modes_seen": worker_modes,
        "setter_propagates_under_fork": all(m == "legacy" for m in worker_modes),
    }


def part_build_1core(dictionary):
    from src import features
    marisa_path = os.path.join(DATA, "english_dictionary.marisa")
    out = {
        "ac_build_from_set": _timed(
            lambda: features._build_matcher(dictionary, "fast"), reps=3),
        "marisa_build_from_set": _timed(
            lambda: features._build_matcher(dictionary, "fast_marisa"), reps=3),
    }
    if os.path.exists(marisa_path):
        out["marisa_mmap"] = _timed(
            lambda: features.build_kernel(marisa_path=marisa_path), reps=3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from cgroup_info import cgroup_summary
    from src.shared_resources import initialize_shared_resources

    dictionary, ngram = initialize_shared_resources(DATA)
    result = {
        "start_method": multiprocessing.get_start_method(),
        "cgroup": cgroup_summary(),
        "windows_baselines": {
            "spawn_cost_per_worker_sec": "0.3-0.5 (B2/B5, Windows spawn)",
            "ac_build_from_set_sec": 0.50,
            "ac_pickle_loads_sec": 0.06,
            "source": "results/kernel/attach_paths.json (Windows)",
        },
        "pool_startup": part_pool_startup(dictionary, ngram),
        "attach_paths": part_attach_paths(dictionary),
        "env_propagation": part_env_propagation(),
        "build_1core": part_build_1core(dictionary),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
