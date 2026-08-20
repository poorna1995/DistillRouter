"""Generates the three results figures for research.md / research.tex:
  fig_pareto.pdf/png   -- cost/quality tradeoff (Fig. accompanying S5.4/5.6)
  fig_table8.pdf/png   -- Table 8's 20-epoch validation curve, as a line chart
  fig_table9.pdf/png   -- Table 9's 20-epoch validation curve, as a line chart

Palette: validated categorical slots 1 (blue, route-only) and 2 (orange,
reasoning-plus-route) from the dataviz skill's reference palette (light mode) --
`node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light` -> ALL CHECKS PASS.
Same two colors used for the same two variants in every figure in this paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"      # route-only
ORANGE = "#eb6834"    # reasoning-plus-route
GRAY = "#52514e"      # teacher / reference (neutral, not a "variant")

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8a86",
    "axes.linewidth": 0.8,
    "xtick.color": "#3a3a38",
    "ytick.color": "#3a3a38",
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "font.family": "DejaVu Sans",
})

# ---------------------------------------------------------------------------
# Figure 1 (main paper figure): full routing-quality-vs-latency Pareto plot,
# all 8 student configurations plus the teacher, evaluated in both label
# spaces (the teacher makes a single native 3-way prediction per query;
# binary is a post-hoc collapse of that same prediction and the oracle
# label, so it gets its own reference point at identical latency). Binary
# and 3-way oracle F1 are not on a directly comparable absolute scale
# (different structural baselines/majority-class rates), so they are kept
# visually distinct (fill = label space) rather than implying one combined
# ranking; latency, on the x-axis, *is* directly comparable across every
# point. Every one of the 8 student points has both lower latency and
# higher oracle F1 than its own label-space's teacher anchor.
# Source: research.tex Table 6 (efficiency) and Table 7 (scaling), plus
# the teacher's own oracle-accuracy numbers (data/oracle/teacher_vs_oracle
# _test_report.json for 3-way; the same computation collapsed to binary).
# ---------------------------------------------------------------------------
points = [
    # label,                 latency_ms, f1,    f1_sd, color,  marker, filled, group
    ("Teacher (binary)",     1633.0,     0.591, 0.0,   GRAY,   "*",    True,   "teacher"),
    ("Teacher (3-way)",      1633.0,     0.380, 0.0,   GRAY,   "*",    False,  "teacher"),
    ("RO 270M binary",       10.9,       0.667, 0.010, BLUE,   "o",    True,   "student"),
    ("RR 270M binary",       11.2,       0.675, 0.007, ORANGE, "o",    True,   "student"),
    ("RO 1B binary",         20.8,       0.658, 0.011, BLUE,   "s",    True,   "student"),
    ("RR 1B binary",         22.3,       0.662, 0.007, ORANGE, "s",    True,   "student"),
    ("RO 270M 3-way",        10.6,       0.435, 0.005, BLUE,   "o",    False,  "student"),
    ("RR 270M 3-way",        11.2,       0.436, 0.008, ORANGE, "o",    False,  "student"),
    ("RO 1B 3-way",          20.8,       0.440, 0.007, BLUE,   "s",    False,  "student"),
    ("RR 1B 3-way",          22.4,       0.434, 0.009, ORANGE, "s",    False,  "student"),
]

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)

# dashed reference lines at each teacher's own oracle macro-F1
ax.axhline(0.591, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
ax.text(1450, 0.591 + 0.010, "teacher, binary (0.591)", color=GRAY, fontsize=7,
        va="bottom", ha="right")
ax.axhline(0.380, color=GRAY, linestyle=(0, (1, 2)), linewidth=1.1, zorder=1)
ax.text(1450, 0.380 + 0.010, "teacher, 3-way (0.380)", color=GRAY, fontsize=7,
        va="bottom", ha="right")

for label, latency, f1, sd, color, marker, filled, group in points:
    face = color if filled else "white"
    size = 11 if marker == "*" else 8
    ax.errorbar(latency, f1, yerr=sd if sd else None, fmt=marker, color=color,
                markerfacecolor=face, markersize=size, markeredgecolor=color,
                markeredgewidth=1.1, elinewidth=1.0, capsize=2.5, zorder=3)

# headline callout
ax.text(0.03, 0.06,
        "Every student lies on a\nsuperior Pareto frontier.",
        transform=ax.transAxes, fontsize=9, fontweight="bold", color="#0b0b0b",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f4ee", edgecolor="#c9c7bd", linewidth=0.8))

ax.set_xscale("log")
ax.set_xlim(6, 2400)
ax.set_ylim(0.30, 0.76)
ax.set_xlabel("Routing latency per query (log scale, ms)")
ax.set_ylabel("Macro-F1 vs. oracle ground truth")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.grid(True, which="major", axis="both", color="#e5e4de", linewidth=0.6, zorder=0)

# legend: color = variant, shape = backbone, fill = label space
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=8, label="Route-only"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=8, label="Reasoning-plus-route"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=GRAY, markeredgecolor=GRAY, markersize=10, label="Teacher"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="270M (circle)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="1B (square)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="Binary (filled)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#8a8a86", markersize=8, label="3-way (hollow)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
          fontsize=7.2, frameon=False, columnspacing=1.0, handletextpad=0.4)

fig.tight_layout()
fig.savefig("fig_pareto.pdf")
fig.savefig("fig_pareto.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figures 2 & 3: 20-epoch validation macro-F1 curves (Table 8, Table 9)
# ---------------------------------------------------------------------------
table8_epochs = list(range(1, 21))
table8_route_only = [0.368, 0.446, 0.448, 0.506, 0.528, 0.499, 0.517, 0.484, 0.517, 0.506,
                      0.511, 0.509, 0.513, 0.509, 0.510, 0.517, 0.518, 0.514, 0.516, 0.515]
table8_reasoning = [0.346, 0.410, 0.445, 0.483, 0.529, 0.511, 0.516, 0.530, 0.493, 0.492,
                     0.498, 0.507, 0.506, 0.505, 0.506, 0.505, 0.507, 0.503, 0.508, 0.508]
table8_route_only_peak = (5, 0.528)
table8_reasoning_peak = (8, 0.530)

table9_epochs = list(range(1, 21))
table9_route_only = [0.613, 0.679, 0.579, 0.665, 0.627, 0.657, 0.636, 0.641, 0.636, 0.635,
                      0.640, 0.636, 0.640, 0.640, 0.638, 0.636, 0.636, 0.636, 0.636, 0.638]
table9_reasoning = [0.599, 0.638, 0.563, 0.658, 0.647, 0.681, 0.654, 0.658, 0.677, 0.651,
                     0.670, 0.664, 0.665, 0.669, 0.670, 0.671, 0.667, 0.672, 0.670, 0.670]
table9_route_only_peak = (2, 0.679)
table9_reasoning_peak = (6, 0.681)


def make_epoch_figure(epochs, route_only, reasoning, ro_peak, rr_peak, ylabel, ylim, out_stem):
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    ax.plot(epochs, route_only, color=BLUE, linewidth=2, marker="o", markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.5, label="Route-only", zorder=3)
    ax.plot(epochs, reasoning, color=ORANGE, linewidth=2, marker="o", markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.5, label="Reasoning-plus-route", zorder=3)

    # mark each variant's selected (peak) epoch
    ax.scatter([ro_peak[0]], [ro_peak[1]], s=70, facecolor=BLUE, edgecolor="white",
               linewidth=1.2, zorder=4)
    ax.annotate(f"epoch {ro_peak[0]}\nF1={ro_peak[1]:.3f}", ro_peak, textcoords="offset points",
                xytext=(-8, 10), fontsize=7.5, color=BLUE, ha="right")
    ax.scatter([rr_peak[0]], [rr_peak[1]], s=70, facecolor=ORANGE, edgecolor="white",
               linewidth=1.2, zorder=4)
    ax.annotate(f"epoch {rr_peak[0]}\nF1={rr_peak[1]:.3f}", rr_peak, textcoords="offset points",
                xytext=(8, -18), fontsize=7.5, color=ORANGE, ha="left")

    ax.axvline(3, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(3.3, ylim[0] + 0.015, "3-epoch budget\n(used elsewhere)", fontsize=7,
            color=GRAY, va="bottom")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, which="major", axis="y", color="#e5e4de", linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


make_epoch_figure(table8_epochs, table8_route_only, table8_reasoning,
                   table8_route_only_peak, table8_reasoning_peak,
                   "Validation macro-F1 (3-way)", (0.30, 0.58), "fig_table8")

make_epoch_figure(table9_epochs, table9_route_only, table9_reasoning,
                   table9_route_only_peak, table9_reasoning_peak,
                   "Validation macro-F1 (binary)", (0.55, 0.70), "fig_table9")

# ---------------------------------------------------------------------------
# Figure: confusion-matrix heatmaps (replaces the two confusion-matrix
# tables) -- 2x2 grid, rows = reference (teacher / oracle), columns =
# variant (route-only / reasoning-plus-route), each cell a 2x2 binary
# confusion matrix (small/large) shown as a heatmap with row-normalized
# recall (%) annotated. Sequential single-hue colormap (not the categorical
# blue/orange, which encodes variant identity elsewhere) since this figure
# encodes magnitude, not identity.
# Source: research.md Tables 6 & 7.
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.cm as cm

LABELS2 = ["small", "large"]
matrices = {
    ("Teacher", "Route-only"):            np.array([[57.6, 42.4], [23.7, 76.3]]),
    ("Teacher", "Reasoning-plus-route"):  np.array([[55.6, 44.4], [22.4, 77.6]]),
    ("Oracle", "Route-only"):             np.array([[67.4, 32.6], [31.4, 68.6]]),
    ("Oracle", "Reasoning-plus-route"):   np.array([[66.4, 33.6], [29.3, 70.7]]),
}
row_refs = ["Teacher", "Oracle"]
col_variants = ["Route-only", "Reasoning-plus-route"]

fig, axes = plt.subplots(2, 2, figsize=(6.2, 6.0), dpi=200)
cmap = cm.get_cmap("Blues")

for i, ref in enumerate(row_refs):
    for j, variant in enumerate(col_variants):
        ax = axes[i, j]
        m = matrices[(ref, variant)]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        for r in range(2):
            for c in range(2):
                val = m[r, c]
                color = "white" if val > 55 else "#2a2a28"
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS2, fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS2, fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax.set_title(variant, fontsize=9.5, color="#0b0b0b", pad=6)
        if j == 0:
            ax.set_ylabel(f"vs. {ref}\n\ntrue label", fontsize=9)
        if i == 1:
            ax.set_xlabel("predicted label", fontsize=8.5)

fig.suptitle("Row-normalized recall (%), pooled across 10 seeds × 2 datasets, binary label space",
             fontsize=9, color="#52514e", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_confusion.pdf")
fig.savefig("fig_confusion.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: fidelity vs. correctness scatter (replaces Table 5) -- diagnostic
# for Section 5.3. x = teacher-agreement macro-F1 (fidelity), y =
# oracle-agreement macro-F1 (correctness). Eight points: route-only and
# reasoning-plus-route, at all four configurations this paper tests (2
# backbones x 2 label spaces) -- the full binary/1B grid completed after
# the binary/1B training run finished. Marker shape encodes backbone
# (circle=270M, square=1B); marker fill encodes label space (filled=binary,
# hollow=3-way); color encodes variant (blue=route-only, orange=RR), as in
# every other figure in this paper.
# Source: research.md Table 5 (all four configuration rows).
# ---------------------------------------------------------------------------
scatter_points = [
    # label,             teacher_f1, teacher_sd, oracle_f1, oracle_sd, color,  marker, filled
    ("RO Binary/270M",    0.651,      0.005,      0.667,     0.010,     BLUE,   "o",    True),
    ("RR Binary/270M",    0.645,      0.010,      0.675,     0.007,     ORANGE, "o",    True),
    ("RO 3-way/270M",     0.477,      0.025,      0.435,     0.005,     BLUE,   "o",    False),
    ("RR 3-way/270M",     0.460,      0.018,      0.436,     0.008,     ORANGE, "o",    False),
    ("RO 3-way/1B",       0.505,      0.012,      0.440,     0.007,     BLUE,   "s",    False),
    ("RR 3-way/1B",       0.529,      0.012,      0.434,     0.009,     ORANGE, "s",    False),
    ("RO Binary/1B",      0.652,      0.016,      0.658,     0.011,     BLUE,   "s",    True),
    ("RR Binary/1B",      0.676,      0.008,      0.662,     0.007,     ORANGE, "s",    True),
]

fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)

# connector between same-configuration RO/RR pairs, drawn first (under the
# points) -- carries the "does the ranking reverse" story per configuration.
connectors = [
    ((0.651, 0.645), (0.667, 0.675)),   # binary/270M
    ((0.477, 0.460), (0.435, 0.436)),   # 3-way/270M
    ((0.505, 0.529), (0.440, 0.434)),   # 3-way/1B
    ((0.652, 0.676), (0.658, 0.662)),   # binary/1B
]
for (x0, x1), (y0, y1) in connectors:
    ax.plot([x0, x1], [y0, y1], color="#8a8a86", linewidth=1.3,
            linestyle=(0, (3, 2)), zorder=2)

label_offsets = {
    "RO Binary/270M": (10, 8),
    "RR Binary/270M": (10, -16),
    "RO 3-way/270M": (-12, -14),
    "RR 3-way/270M": (10, 8),
    "RO 3-way/1B": (-14, 10),
    "RR 3-way/1B": (10, -6),
    "RO Binary/1B": (-14, -14),
    "RR Binary/1B": (10, 10),
}

for label, tf1, tsd, of1, osd, color, marker, filled in scatter_points:
    face = color if filled else "white"
    ax.errorbar(tf1, of1, xerr=tsd, yerr=osd, fmt=marker, color=color,
                markerfacecolor=face, markersize=9, markeredgecolor=color,
                markeredgewidth=1.3, elinewidth=1.1, capsize=3, zorder=3)
    dx, dy = label_offsets[label]
    ax.annotate(label.split(" ", 1)[0], (tf1, of1),
                textcoords="offset points", xytext=(dx, dy), fontsize=7,
                color=color, fontweight="bold",
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top")

ax.set_xlabel("Policy Transfer F1")
ax.set_ylabel("Routing Accuracy F1")
ax.set_title("Policy Transfer Does Not Predict Routing Accuracy", fontsize=12.5, fontweight="bold", pad=12)
ax.grid(True, which="major", axis="both", color="#e5e4de", linewidth=0.6, zorder=0)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=8, label="Route-only"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=8, label="Reasoning-plus-route"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="Binary (filled)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#8a8a86", markersize=8, label="3-way (hollow)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="270M (circle)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#8a8a86", markeredgecolor="#8a8a86", markersize=8, label="1B (square)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
          fontsize=7.5, frameon=False, columnspacing=1.0, handletextpad=0.4)

fig.tight_layout()
fig.savefig("fig_fidelity_vs_correctness.pdf")
fig.savefig("fig_fidelity_vs_correctness.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: effect of reasoning supervision across configurations -- delta
# plot (forest-plot style). Delta = F1_(reasoning-plus-route) -
# F1_(route-only), teacher-agreement macro-F1 (fidelity), one point per
# configuration with a 95% CI from a paired bootstrap over the ten seeds
# (research.tex Table 5's route-only-minus-reasoning delta, sign-flipped).
# Configurations are ordered 270M pair then 1B pair so the reader sees the
# scale-dependence directly: near/below zero on the left, reliably
# positive on the right.
# ---------------------------------------------------------------------------
configs = ["Binary\n270M", "3-way\n270M", "3-way\n1B", "Binary\n1B"]
delta =    [-0.0065, -0.017, 0.024, 0.024]
ci_lo =    [-0.0232, -0.062, 0.001, 0.003]
ci_hi =    [ 0.0118,  0.022, 0.055, 0.062]

fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)
x = np.arange(len(configs))

ax.axhline(0, color="#8a8a86", linewidth=1.0, zorder=1)

for idx, (d, lo, hi) in enumerate(zip(delta, ci_lo, ci_hi)):
    reliable = lo > 0 or hi < 0
    color = ORANGE if (reliable and d > 0) else (BLUE if (reliable and d < 0) else GRAY)
    ax.errorbar(idx, d, yerr=[[d - lo], [hi - d]], fmt="o", color=color,
                markersize=9, markeredgecolor="white", markeredgewidth=0.9,
                elinewidth=1.6, capsize=5, zorder=3)
    tag = "95% CI excludes 0" if reliable else "n.s."
    ax.annotate(f"{d:+.3f}\n({tag})", (idx, d), textcoords="offset points",
                xytext=(0, 14 if d >= 0 else -14), fontsize=7.5, color=color,
                ha="center", va="bottom" if d >= 0 else "top", fontweight="bold")

ax.text(0.02, 0.94, "Reasoning only helps at larger scale.",
        transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#0b0b0b",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f4ee", edgecolor="#c9c7bd", linewidth=0.8))

ax.axvspan(-0.5, 1.5, color="#eceae2", alpha=0.5, zorder=0)
ax.axvspan(1.5, 3.5, color="#fbeee4", alpha=0.5, zorder=0)
ax.text(0.5, -0.078, "270M backbone", ha="center", fontsize=8, color="#6b6a64")
ax.text(2.5, -0.078, "1B backbone", ha="center", fontsize=8, color="#6b6a64")

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.085, 0.075)
ax.set_ylabel(r"$\Delta$ Macro-F1 (reasoning-plus-route $-$ route-only)")
ax.grid(True, which="major", axis="y", color="#e5e4de", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_reasoning_effect.pdf")
fig.savefig("fig_reasoning_effect.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: latency comparison -- simple horizontal bar chart, primary
# configuration (binary, 270M). Log-scale x-axis since the teacher is
# roughly two orders of magnitude slower than either distilled router.
# Source: research.md Table 6 (efficiency).
# ---------------------------------------------------------------------------
routers = ["Teacher", "Route-only", "Reasoning+route"]
latencies = [1633.0, 10.9, 11.2]
colors = [GRAY, BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(5.8, 2.6), dpi=200)
y = np.arange(len(routers))
bars = ax.barh(y, latencies, color=colors, edgecolor="white", linewidth=0.6, height=0.55)
ax.set_xscale("log")
ax.set_xlim(5, 3000)
for yi, val in zip(y, latencies):
    ax.text(val * 1.15, yi, f"{val:g} ms", va="center", fontsize=8.5, color="#0b0b0b")
ax.set_yticks(y)
ax.set_yticklabels(routers, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Latency per query (log scale, ms)")
ax.grid(True, which="major", axis="x", color="#e5e4de", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_latency_bar.pdf")
fig.savefig("fig_latency_bar.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: scaling curve -- oracle macro-F1 vs. backbone size (270M -> 1B),
# both label spaces now that the binary/1B run has completed the full
# 2x2 grid. Two side-by-side panels (shared style, independent y-scale
# since 3-way and binary oracle F1 sit on very different absolute levels)
# rather than one shared axis, so neither label space's flat-line story
# gets visually swamped by the other's scale.
# Source: research.md Table 7 (scaling analysis).
# ---------------------------------------------------------------------------
backbones = ["270M", "1B"]
bx = [0, 1]
scaling_panels = [
    ("3-way", [0.435, 0.440], [0.005, 0.007], [0.436, 0.434], [0.008, 0.009], (0.38, 0.50)),
    ("Binary", [0.667, 0.658], [0.010, 0.011], [0.675, 0.662], [0.007, 0.007], (0.60, 0.72)),
]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=200)
for ax, (label_space, ro_f1, ro_sd, rr_f1, rr_sd, ylim) in zip(axes, scaling_panels):
    ax.errorbar(bx, ro_f1, yerr=ro_sd, color=BLUE, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=0.8, linewidth=2, capsize=3,
                label="Route-only", zorder=3)
    ax.errorbar(bx, rr_f1, yerr=rr_sd, color=ORANGE, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=0.8, linewidth=2, capsize=3,
                label="Reasoning-plus-route", zorder=3)
    ax.set_xticks(bx)
    ax.set_xticklabels(backbones, fontsize=10)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Distilled-router backbone")
    ax.set_title(f"{label_space} label space", fontsize=10, color="#3a3a38")
    ax.grid(True, which="major", axis="y", color="#e5e4de", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Macro-F1 vs. oracle ground truth")
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig_scaling.pdf")
fig.savefig("fig_scaling.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figures: RO-vs-oracle and RR-vs-oracle confusion matrices, split into two
# standalone heatmaps (binary label space, pooled 10 seeds x 2 datasets).
# Source: research.md Section 5.6 (error analysis); same underlying matrices
# as the combined fig_confusion grid above, oracle row only.
# ---------------------------------------------------------------------------
oracle_matrices = {
    "Route-only":            np.array([[67.4, 32.6], [31.4, 68.6]]),
    "Reasoning-plus-route":  np.array([[66.4, 33.6], [29.3, 70.7]]),
}


def make_single_confusion(variant, out_stem):
    m = oracle_matrices[variant]
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=200)
    ax.imshow(m, cmap=cm.get_cmap("Blues"), vmin=0, vmax=100, aspect="equal")
    for r in range(2):
        for c in range(2):
            val = m[r, c]
            color = "white" if val > 55 else "#2a2a28"
            ax.text(c, r, f"{val:.1f}%", ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS2, fontsize=9.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS2, fontsize=9.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("predicted label", fontsize=9)
    ax.set_ylabel("true label (oracle)", fontsize=9)
    ax.set_title(f"{variant} vs. Oracle", fontsize=10.5, color="#0b0b0b", pad=8)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


make_single_confusion("Route-only", "fig_confusion_ro")
make_single_confusion("Reasoning-plus-route", "fig_confusion_rr")

# ---------------------------------------------------------------------------
# Figure: Teacher vs. Student confusion matrix, both against oracle ground
# truth, binary label space -- the direct visual evidence for the
# under-escalation-then-recovery claim (Section 6.4). "Student" is
# route-only/binary/270M, this paper's primary distillation configuration.
# Source: teacher matrix computed directly from data/teacher/{gsm8k,math}
# /v2/test.jsonl vs. data/oracle/{gsm8k,math}/test.labels.jsonl (collapsed
# to binary); student matrix is the Route-only oracle_matrices entry above.
# ---------------------------------------------------------------------------
teacher_vs_oracle = np.array([[77.1, 22.9], [50.4, 49.6]])
student_vs_oracle = oracle_matrices["Route-only"]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), dpi=200)
for ax, (name, m) in zip(axes, [("Teacher", teacher_vs_oracle), ("Student (route-only)", student_vs_oracle)]):
    ax.imshow(m, cmap=cm.get_cmap("Blues"), vmin=0, vmax=100, aspect="equal")
    for r in range(2):
        for c in range(2):
            val = m[r, c]
            color = "white" if val > 55 else "#2a2a28"
            ax.text(c, r, f"{val:.1f}%", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS2, fontsize=9.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS2, fontsize=9.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("predicted label", fontsize=9)
    ax.set_title(name, fontsize=10.5, color="#0b0b0b", pad=8)
axes[0].set_ylabel("true label (oracle)", fontsize=9)
fig.suptitle("Recall (%) vs. oracle ground truth, binary label space",
             fontsize=9, color="#52514e", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig_confusion_teacher_vs_student.pdf")
fig.savefig("fig_confusion_teacher_vs_student.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure: cost-quality frontier. x = expected cost/query (routing latency +
# answer-generation latency of whichever candidate actually serves the
# query, ms, log scale) -- this paper has no populated dollar-cost field
# (all candidates are self-hosted, not API-billed), so compute latency is
# the real, defensible cost proxy. y = end-to-end accuracy (real task
# correctness, not routing-label agreement). "Always X" baselines serve
# every query with one fixed candidate, so their cost is that candidate's
# own mean answer-generation latency. Teacher/DistillRouter cost is the
# routing decision's own latency plus the *expected* answer-generation
# latency, weighted by each router's actual predicted-label distribution
# on the test set (binary deployment policy: "cheap" queries are served by
# the medium candidate, per Section 3.7's empirically grounded policy).
# Source: data/oracle/{gsm8k,math}/test.attempts.jsonl (per-candidate
# latency + correctness), data/teacher/{gsm8k,math}/v2/test.jsonl and
# runs_binary/{route_only,reasoning_plus_route}/seed_*/best/eval_*_test/
# predictions.jsonl (routing-label distributions), Table 3 (end-to-end
# accuracy).
# ---------------------------------------------------------------------------
cq_points = [
    # label,                    cost_ms, acc,   color,  marker
    ("Always Small",            4390,    0.045, GRAY,   "D"),
    ("Always Medium",           4683,    0.336, GRAY,   "D"),
    ("Always Large",            10226,   0.456, GRAY,   "D"),
    ("Teacher Router",          8550,    0.392, "#8a8a86", "*"),
    ("DistillRouter (RO)",      7801,    0.405, BLUE,   "o"),
    ("DistillRouter (RR)",      7894,    0.408, ORANGE, "o"),
]

fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)

ax.axhline(0.542, color="#8a8a86", linestyle=(0, (1, 2)), linewidth=1.1, zorder=1)
ax.text(4300, 0.542 + 0.010, "oracle ceiling (0.542)", color="#6b6a64", fontsize=7.5,
        va="bottom", ha="left")

label_offsets = {
    "Always Small":        (0, -14),
    "Always Medium":       (0, 12),
    "Always Large":        (0, 12),
    "Teacher Router":      (18, -16),
    "DistillRouter (RO)":  (-14, -16),
    "DistillRouter (RR)":  (0, 16),
}
for label, cost, acc, color, marker in cq_points:
    size = 12 if marker == "*" else 9
    ax.plot(cost, acc, marker=marker, color=color, markersize=size,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3, linestyle="none")
    dx, dy = label_offsets[label]
    ax.annotate(label, (cost, acc), textcoords="offset points", xytext=(dx, dy),
                fontsize=7.5, color=color if color != GRAY else "#52514e",
                ha="center", va="bottom" if dy > 0 else "top")

ax.set_xscale("log")
ax.set_xlim(3800, 13000)
ax.set_ylim(0.0, 0.60)
ax.set_xlabel("Expected cost per query (log scale, ms of compute)")
ax.set_ylabel("End-to-end accuracy")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.grid(True, which="major", axis="both", color="#e5e4de", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_cost_quality.pdf")
fig.savefig("fig_cost_quality.png", dpi=200)
plt.close(fig)

print("wrote fig_pareto.{pdf,png}, fig_table8.{pdf,png}, fig_table9.{pdf,png}, "
      "fig_confusion.{pdf,png}, fig_fidelity_vs_correctness.{pdf,png}, "
      "fig_reasoning_effect.{pdf,png}, fig_latency_bar.{pdf,png}, "
      "fig_scaling.{pdf,png}, fig_confusion_ro.{pdf,png}, fig_confusion_rr.{pdf,png}, "
      "fig_confusion_teacher_vs_student.{pdf,png}, fig_cost_quality.{pdf,png}")
