"""cgroup-limit evidence capture (B6 Step 1) - run INSIDE each container.

Reads the kernel's own view of the applied limits (cgroup v1 and v2 layouts)
plus CPU identity, so every measurement JSON carries proof of the profile it
ran under. stdlib only: works in the serve-onnx image (no psutil).
"""
import os
import json


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def effective_cpus() -> float:
    """CPUs allowed by the cgroup quota (fallback: os.cpu_count()).

    Inside `docker run --cpus=N`, os.cpu_count() still reports the HOST
    count; Pool sizing and k=cores claims must use this instead.
    """
    # cgroup v2
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max":
            return float(quota) / float(period or 100000)
    # cgroup v1
    quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return float(os.cpu_count())


def memory_limit_bytes():
    v2 = _read("/sys/fs/cgroup/memory.max")
    if v2 and v2 != "max":
        return int(v2)
    v1 = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1:
        v = int(v1)
        return v if v < 2**60 else None  # v1 'unlimited' sentinel
    return None


def cpu_model():
    txt = _read("/proc/cpuinfo") or ""
    for line in txt.splitlines():
        if line.lower().startswith(("model name", "hardware", "cpu part")):
            return line.split(":", 1)[1].strip()
    return None


def cgroup_summary() -> dict:
    mem = memory_limit_bytes()
    return {
        "effective_cpus": round(effective_cpus(), 3),
        "memory_limit_mib": round(mem / 2**20, 1) if mem else None,
        "host_visible_cpus": os.cpu_count(),
        "cpu_model": cpu_model(),
        "kernel": (_read("/proc/version") or "").split("(")[0].strip(),
        "machine": os.uname().machine,
        "raw": {
            "cgroup_v2_cpu.max": _read("/sys/fs/cgroup/cpu.max"),
            "cgroup_v2_memory.max": _read("/sys/fs/cgroup/memory.max"),
            "cgroup_v2_memory.swap.max": _read("/sys/fs/cgroup/memory.swap.max"),
            "cgroup_v1_cpu.cfs_quota_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            "cgroup_v1_cpu.cfs_period_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            "cgroup_v1_memory.limit_in_bytes": _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            "cgroup_v1_memsw.limit_in_bytes": _read("/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(cgroup_summary(), indent=2))
