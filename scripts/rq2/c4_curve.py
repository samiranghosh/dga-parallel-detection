"""B4 Step 8: the C4 trade-off chart - accuracy vs RSS vs latency,
{RF-100, RF-pruned, DT-12, LR} x {sklearn, ONNX}. Print-oriented PNG.

Form: two scatter panels sharing the accuracy y-axis (never 3D / dual-axis).
Color = model identity (4 fixed-order validated categorical slots);
runtime = marker shape (secondary encoding, CVD/print-safe);
direct labels on every point (relief rule for the sub-3:1 slots);
targets as recessive dashed reference lines. rf100-onnx has no canonical
latency (cold-start DQ) -> RSS panel only, annotated.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = r"c:/Users/samir/source/repos/dga-parallel-detection/results/rq2/fig_c4_tradeoff.png"

# palette.md parameters (validated: ALL CHECKS PASS, CVD dE 24.2)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
MODEL_COLOR = {"RF-100": "#2a78d6", "RF-pruned": "#1baf7a",
               "DT-12": "#eda100", "LR": "#008300"}          # fixed slot order
RUNTIME_MARKER = {"sklearn": "o", "ONNX": "^"}

# (model, runtime, acc%, rss_mib, latency_us_or_None)  - canonical protocol
DATA = [
    ("RF-100",    "sklearn", 93.179,  631.7, 4398.0),
    ("RF-100",    "ONNX",    93.179,  990.7, None),   # cold-start DQ
    ("RF-pruned", "sklearn", 93.315,  165.6, 4214.0),
    ("RF-pruned", "ONNX",    93.315,   69.2,   77.0),
    ("DT-12",     "sklearn", 93.4675, 155.3,  102.0),
    ("DT-12",     "ONNX",    93.4675,  56.0,   64.0),
    ("LR",        "sklearn", 90.0013, 154.8,  200.0),
    ("LR",        "ONNX",    90.0013,  54.8,   61.0),
]
# manual per-point label offsets (x-mult, y-add) to avoid collisions
OFF = {("RF-100", "sklearn"): (1.12, 0.22), ("RF-100", "ONNX"): (0.88, -0.30),
       ("RF-pruned", "sklearn"): (1.12, -0.30), ("RF-pruned", "ONNX"): (1.14, -0.32),
       ("DT-12", "sklearn"): (1.12, 0.05), ("DT-12", "ONNX"): (1.14, 0.22),
       ("LR", "sklearn"): (1.12, 0.02), ("LR", "ONNX"): (1.14, 0.22)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)

for ax, dim, xlabel, refs in (
        (axes[0], 3, "total process RSS (MiB, log)",
         [(244, "244 MiB (256 MB cgroup)"), (488, "488 MiB (512 MB)")]),
        (axes[1], 4, "single-request p50 (µs, log) — canonical protocol",
         [(1000, "1 ms target")])):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for x, lab in refs:
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
        ax.text(x, 89.62, " " + lab, color=MUTED, fontsize=7.5, rotation=90,
                va="bottom", ha="right")
    for model, runtime, acc, rss, lat in DATA:
        x = rss if dim == 3 else lat
        if x is None:
            continue
        ax.plot(x, acc, RUNTIME_MARKER[runtime], color=MODEL_COLOR[model],
                markersize=10, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3)
        mx, my = OFF[(model, runtime)]
        ax.text(x * mx, acc + my, f"{model} · {runtime.lower()}",
                fontsize=7.5, color=INK2, va="center",
                ha="left" if mx > 1 else "right", zorder=4)
    ax.set_xlim(35, 3500 if dim == 3 else 16000)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)

# DQ annotation on the RSS panel
axes[0].annotate("cold-start DQ\n(init 15–32 min)", xy=(990.7, 93.179),
                 xytext=(700, 91.9), fontsize=7.5, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

axes[0].set_ylabel("holdout accuracy (%)", color=INK2, fontsize=9)
axes[0].set_ylim(89.6, 94.0)
axes[0].text(0.01, 0.02, "y-axis not zero-based", transform=axes[0].transAxes,
             fontsize=7, color=MUTED)

handles = ([Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m],
                   markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=9,
                   label=m) for m in MODEL_COLOR]
           + [Line2D([], [], marker=RUNTIME_MARKER[r], ls="", color=INK2,
                     markersize=8, label=f"{r} runtime") for r in RUNTIME_MARKER])
fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("C4 trade-off: accuracy vs memory vs latency — model grid × runtime",
             fontsize=11, color=INK, y=1.10)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT)
