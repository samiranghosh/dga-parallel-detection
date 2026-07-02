"""B5 Step 1: kernel backend micro-bench -> results/kernel/bench_backends.json.

Candidates for replacing the two O(m^2) dictionary scans:
  (a) pyahocorasick 2.3.1 - C Aho-Corasick automaton over the 234,377-word
      dictionary; one pass per domain yields ALL dictionary-word occurrences.
  (b) marisa-trie 1.4.1 (existing src/compact_dict.py groundwork) - per-position
      trie.prefixes(domain[i:]) sweep; m calls per domain, each returns the
      dictionary words starting at that position.

Both feed the SAME downstream computation (max match length -> lms; interval
DP over matches -> mwr), so the bench isolates the match-enumeration cost,
which is the access pattern the two features share. Legacy O(m^2) included as
reference. Values are cross-checked for equality on every domain (pre-A2).

Protocol: 10,000 domains stratified over train+test x label (seed 42),
3 reps per backend, per-domain mean microseconds. Build/load time and
RSS delta measured in fresh subprocesses.
"""
import os
import sys
import json
import time
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

N_DOMAINS = 10_000
REPS = 3
SEED = 42
OUT = os.path.join(REPO, "results", "kernel", "bench_backends.json")


def sample_domains(n, seed):
    import pandas as pd
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(REPO, "data", f"{split}.csv"))
        df["domain"] = df["domain"].astype(str)
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True)
    return alldf.sample(n=n, random_state=seed)["domain"].tolist()


# ── the shared downstream computation ────────────────────────────────────────
def features_from_matches(n, matches):
    """(mwr, lms) from a list of (end_pos_exclusive, length) matches.

    Replicates legacy semantics: lms = longest match / n; mwr = max chars
    coverable by non-overlapping matches / n (weighted-interval DP).
    """
    if n == 0:
        return 0.0, 0.0
    max_len = 0
    by_end = [None] * (n + 1)
    for end, length in matches:
        if length > max_len:
            max_len = length
        b = by_end[end]
        if b is None:
            by_end[end] = [length]
        else:
            b.append(length)
    cov = [0] * (n + 1)
    for j in range(1, n + 1):
        best = cov[j - 1]
        b = by_end[j]
        if b is not None:
            for length in b:
                c = cov[j - length] + length
                if c > best:
                    best = c
        cov[j] = best
    return cov[n] / n, max_len / n


def ac_matches(automaton, domain):
    return [(end + 1, length) for end, length in automaton.iter(domain)]


def marisa_matches(trie, domain):
    out = []
    for i in range(len(domain)):
        for w in trie.prefixes(domain[i:]):
            out.append((i + len(w), len(w)))
    return out


def measure_struct_memory(kind):
    """RSS delta (MiB) + build/load seconds for one structure, fresh process.

    Streams words from disk (no retained Python list) and gc.collect()s
    before the final RSS read, so the delta is the structure alone.
    """
    dict_txt = os.path.join(REPO, "data", "english_dictionary.txt")
    dict_marisa = os.path.join(REPO, "data", "english_dictionary.marisa")
    code = (
        "import os,sys,time,gc,psutil;sys.path.insert(0,r'%s');"
        "p=psutil.Process();r0=p.memory_info().rss;t0=time.perf_counter()\n" % REPO
    )
    if kind == "set":
        code += (
            "d=set(l.strip() for l in open(r'%s',encoding='utf-8') if l.strip())\n"
            % dict_txt
        )
    elif kind == "marisa":
        code += (
            "import marisa_trie\n"
            "d=marisa_trie.Trie();d.mmap(r'%s')\n"
            "assert 'the' in d\n" % dict_marisa
        )
    elif kind == "ahocorasick":
        code += (
            "import ahocorasick\n"
            "d=ahocorasick.Automaton()\n"
            "for l in open(r'%s',encoding='utf-8'):\n"
            "    w=l.strip()\n"
            "    if w: d.add_word(w,len(w))\n"
            "d.make_automaton()\n" % dict_txt
        )
    code += (
        "t1=time.perf_counter();gc.collect();r1=p.memory_info().rss;"
        "import pickle;psz=len(pickle.dumps(d,-1)) if %r!='marisa' else 0;"
        "print((r1-r0)/2**20, t1-t0, psz/2**20)" % kind
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=REPO, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    mib, secs, pickle_mib = map(float, out.stdout.split())
    return {"rss_delta_mib": round(mib, 1), "build_or_load_sec": round(secs, 3),
            "pickle_size_mib": round(pickle_mib, 1)}


if __name__ == "__main__":
    from src.shared_resources import initialize_shared_resources
    from src.features import calc_meaningful_word_ratio, calc_lms_percentage
    import marisa_trie
    import ahocorasick

    dictionary, _ = initialize_shared_resources(os.path.join(REPO, "data"))
    domains = sample_domains(N_DOMAINS, SEED)

    trie = marisa_trie.Trie()
    trie.mmap(os.path.join(REPO, "data", "english_dictionary.marisa"))

    t0 = time.perf_counter()
    automaton = ahocorasick.Automaton()
    for w in dictionary:
        automaton.add_word(w, len(w))
    automaton.make_automaton()
    ac_build_inproc = time.perf_counter() - t0

    # correctness cross-check (all three agree on every domain)
    mismatches = 0
    for d in domains:
        ref = (calc_meaningful_word_ratio(d, dictionary),
               calc_lms_percentage(d, dictionary))
        got_ac = features_from_matches(len(d), ac_matches(automaton, d))
        got_ma = features_from_matches(len(d), marisa_matches(trie, d))
        if ref != got_ac or ref != got_ma:
            mismatches += 1
    print(f"cross-check: {mismatches} mismatches over {len(domains)} domains")

    def bench(fn):
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for d in domains:
                fn(d)
            times.append((time.perf_counter() - t0) / len(domains) * 1e6)
        return {"per_domain_us_mean": round(sum(times) / len(times), 2),
                "per_domain_us_reps": [round(t, 2) for t in times]}

    results = {
        "protocol": {"n_domains": N_DOMAINS, "reps": REPS, "seed": SEED,
                     "workload": "mwr + lms per domain (the two O(m^2) features)"},
        "cross_check_mismatches": mismatches,
        "legacy_set_scan": bench(
            lambda d: (calc_meaningful_word_ratio(d, dictionary),
                       calc_lms_percentage(d, dictionary))),
        "ahocorasick": bench(
            lambda d: features_from_matches(len(d), ac_matches(automaton, d))),
        "marisa_prefix_sweep": bench(
            lambda d: features_from_matches(len(d), marisa_matches(trie, d))),
        "ac_build_inprocess_sec": round(ac_build_inproc, 3),
        "memory": {k: measure_struct_memory(k)
                   for k in ("set", "marisa", "ahocorasick")},
        "wheel_availability_aarch64_cp311": {
            "pyahocorasick-2.3.1": "manylinux2014_aarch64 (glibc >= 2.17)",
            "marisa-trie-1.4.1": "manylinux_2_28_aarch64 + musllinux_1_2 (glibc >= 2.28)",
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for k in ("legacy_set_scan", "ahocorasick", "marisa_prefix_sweep"):
        print(f"{k:22s} {results[k]['per_domain_us_mean']:8.2f} us/domain")
    print("memory:", json.dumps(results["memory"]))
    print("wrote", OUT)
