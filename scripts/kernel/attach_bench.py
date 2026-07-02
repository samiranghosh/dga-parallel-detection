"""B5 Step 4a: automaton attach-path bench -> results/kernel/attach_paths.json.

A spawned worker needs the AC automaton. Candidate attach paths:
  build     - build from the dictionary set already delivered by the existing
              initializer/shm block (status quo: rebuild per worker)
  pickle    - parent pickles the finished automaton once; workers loads() it
              (what Pool initargs would transfer)
  saveload  - pyahocorasick native save() once to disk; workers load() it

Each timed in-process (structure work only; process spawn cost is common to
all paths and unchanged by this batch). RSS per worker is ~35 MiB for every
path - pyahocorasick has no mmap, so a per-worker private copy is inherent;
the only true zero-copy option is the marisa backend (0.7 MiB mmap) at 1.6x
kernel latency (results/kernel/bench_backends.json).
"""
import os
import sys
import json
import time
import pickle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "results", "kernel", "attach_paths.json")
REPS = 5

if __name__ == "__main__":
    import ahocorasick
    from src.shared_resources import initialize_shared_resources

    dictionary, _ = initialize_shared_resources(os.path.join(REPO, "data"))

    def timed(fn):
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return {"mean_sec": round(sum(times) / len(times), 4),
                "reps_sec": [round(t, 4) for t in times]}

    def build():
        a = ahocorasick.Automaton()
        for w in dictionary:
            a.add_word(w, len(w))
        a.make_automaton()
        return a

    automaton = build()
    blob = pickle.dumps(automaton, -1)
    save_path = os.path.join(REPO, "data", "english_dictionary.automaton")
    automaton.save(save_path, pickle.dumps)

    results = {
        "reps": REPS,
        "build_from_set": timed(build),
        "pickle_loads": {**timed(lambda: pickle.loads(blob)),
                         "blob_mib": round(len(blob) / 2**20, 1)},
        "native_saveload": {**timed(lambda: ahocorasick.load(save_path, pickle.loads)),
                            "file_mib": round(os.path.getsize(save_path) / 2**20, 1)},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
