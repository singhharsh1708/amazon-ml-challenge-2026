"""Draw the charts in assets/ used by the README and the docs.

Every number below is hard-coded from the write-up and nothing is read from data files:

- docs/methodology.md: local validation scores (Section 6), blocking and candidate
  statistics (Section 3), band AUCs (Sections 4.3 and 4.4), the v11 loss split
  (Section 6) and the late increments (Section 4.9).
- docs/10-leaderboard-journey.md: public leaderboard scores of every upload with a recorded score, the
  local scores of v1 to v3 on the older unweighted metric, and the leaderboard
  snapshot taken at 18:19 IST on 27 Sep 2026 (8,269 teams).
- The e5-large AUC and the four-model stack AUC were reproduced with the stacker in
  src/final_steps/rescue_extra/gate/save15.py on the saved validation band scores;
  the same run gives back every AUC that docs/methodology.md reports.

Run from the repository root with matplotlib installed:

    python research/make_figures.py

The PNGs are written to assets/ at 150 dpi on a white background.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ASSETS = Path(__file__).resolve().parent.parent / "assets"

DATA = {
    "leaderboard": [
        ("v1", 0.940),
        ("v2", 0.950),
        ("v3", 0.968),
        ("v6", 0.974),
        ("v11", 0.982),
        ("v15", 0.986784),
        ("v17", 0.9871),
        ("v20", 0.988549),
        ("v21", 0.988642),
    ],
    "snapshot": [
        ("1st", 0.991829),
        ("50th", 0.989515),
        ("100th", 0.988705),
    ],
    "local_older_metric": {"v1": 0.9537, "v2": 0.9714, "v3": 0.9817},
    "local_weighted": {"v6": 0.98241, "v11": 0.98716, "v15": 0.98901, "v17": 0.98901},
    "local_ladder": [
        ("v6 + veto", "the 0.974 file", 0.98241),
        ("v10", "odd-one-out stage 2, sibling rule", 0.98470),
        ("v11", "ce1 cross-encoder stack, expected-F0.5 selection", 0.98716),
        ("v12", "second e5-small cross-encoder (ce2)", 0.98777),
        ("v13", "France audit rules, no US or Indian change", 0.98777),
        ("v14", "rescue for blocking misses", 0.98847),
        ("v15", "e5-base cross-encoder (kbase)", 0.98901),
    ],
    "late_increments": [
        ("v19", "e5-large as a fourth cross-encoder", 0.00013),
        ("v19", "high-confidence recheck", 0.0001),
        ("v20", "name-key rescue, held-out half", 0.00017),
        ("v21", "hc and reverse rescue, held-out half", 0.000075),
    ],
    "band_auc_single": [
        ("Stage 1 LightGBM", 0.9415, "gbm"),
        ("Stage 2 LightGBM", 0.9714, "gbm"),
        ("ce1  e5-small, 250k pairs", 0.9269, "ce"),
        ("ce2  e5-small, +500k pairs", 0.9457, "ce"),
        ("ce3  e5-small, dropped", 0.9525, "dropped"),
        ("kbase  e5-base, 1.39M pairs", 0.9591, "ce"),
        ("klarge  e5-large, 900k pairs", 0.9600, "ce"),
    ],
    "band_auc_stacked": [
        ("Stage 2 alone", 0.9714),
        ("+ ce1", 0.9777),
        ("+ ce2", 0.9807),
        ("+ kbase", 0.9828),
        ("+ klarge", 0.9834),
    ],
    "loss_v11": [
        ("True pairs scored\nbelow the decision line", 0.0073, "recall"),
        ("Blocking misses\n(true entity never a candidate)", 0.0061, "recall"),
        ("False positives\n(wrong merges)", 0.002, "precision"),
    ],
    "candidates_buckets": [
        ("0", 29),
        ("1 to 5", 190926),
        ("6 to 10", 795559),
        ("11 or more", 746030),
    ],
    "candidates_country_mean": [("US", 9.59), ("India", 11.81), ("France", 14.51)],
    "candidates_overall": {"mean": 11.37, "median": 10, "p90": 18, "p99": 37, "max": 4434},
}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#a8a79f"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "axes.unicode_minus": False,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "savefig.dpi": 150,
        "savefig.facecolor": SURFACE,
    }
)


def style(ax, grid_axis="y"):
    """Hairline grid on one axis, no box, a single baseline."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    if grid_axis == "x":
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(BASE)


def heading(fig, title, subtitle):
    """Left-aligned figure title with a secondary subtitle line."""
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.46 / height, subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def footnote(fig, text):
    """Muted note under the plot."""
    fig.text(0.012, 0.018, text, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def save(fig, name):
    """Write one PNG into assets/."""
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def dot(ax, x, y, color, size=8, hollow=False, zorder=4):
    """A filled or hollow marker with a surface ring."""
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=size,
        markerfacecolor=SURFACE if hollow else color,
        markeredgecolor=color if hollow else SURFACE,
        markeredgewidth=1.8 if hollow else 1.6,
        linestyle="none",
        zorder=zorder,
    )


def fmt_score(v):
    """Print a score with the precision it was reported at."""
    text = f"{v:.6f}".rstrip("0")
    return text if len(text.split(".")[1]) >= 3 else f"{v:.3f}"


def leaderboard_journey():
    """Public score of each upload with a recorded score, with a zoom on the last four."""
    rows = DATA["leaderboard"]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    xs = list(range(len(rows)))

    fig = plt.figure(figsize=(10, 4.9))
    heading(
        fig,
        "Public leaderboard: 0.940 to 0.988642",
        "Macro F0.5 of each upload with a recorded score, 26 to 27 Sep 2026. Right: the last four uploads against the leaderboard at 18:19 IST.",
    )
    ax = fig.add_axes([0.075, 0.13, 0.5, 0.66])
    zx = fig.add_axes([0.66, 0.13, 0.32, 0.66])

    style(ax)
    ax.plot(xs, vals, color=BLUE, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    for x, v in zip(xs, vals):
        dot(ax, x, v, BLUE)
    ax.axvspan(4.6, 8.4, color="#f3f2ee", zorder=0, linewidth=0)
    ax.text(6.5, 0.9395, "zoomed at right", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    offsets = {"v1": (0.28, -0.0006), "v2": (0.28, -0.0006), "v3": (0.28, -0.0012), "v6": (0.28, -0.0014), "v11": (0.0, 0.0022)}
    for name, x, v in zip(names, xs, vals):
        if name in offsets:
            dx, dy = offsets[name]
            ha = "left" if dx else "center"
            ax.text(x + dx, v + dy, fmt_score(v), ha=ha, va="center", fontsize=9.5, color=INK)
    ax.text(8, 0.988642 + 0.0028, "0.988642", ha="right", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(xs, names)
    ax.set_xlim(-0.5, 8.4)
    ax.set_ylim(0.936, 0.996)
    ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    ax.set_yticklabels([f"{t:.2f}" for t in ax.get_yticks()])
    ax.set_ylabel("Public F0.5")

    style(zx)
    late = rows[5:]
    lx = list(range(len(late)))
    lv = [r[1] for r in late]
    for label, v in DATA["snapshot"]:
        zx.axhline(v, color=MUTED, linewidth=1.1, zorder=1)
        zx.text(-0.4, v + 0.00006, f"{label} place at 18:19 IST: {v:.6f}", ha="left", va="bottom", fontsize=8.5, color=INK_2)
    zx.plot(lx, lv, color=BLUE, linewidth=2, solid_joinstyle="round", zorder=3)
    for x, v in zip(lx, lv):
        dot(zx, x, v, BLUE)
    zoom_labels = {0: (0.0, -0.0003, "center"), 1: (0.0, -0.0003, "center"), 2: (0.06, -0.00028, "left"), 3: (0.06, -0.00028, "left")}
    for i, v in enumerate(lv):
        dx, dy, ha = zoom_labels[i]
        weight = "bold" if i == len(lv) - 1 else "normal"
        zx.text(i + dx, v + dy, fmt_score(v), ha=ha, va="center", fontsize=9.5, color=INK, fontweight=weight)
    zx.set_xticks(lx, [r[0] for r in late])
    zx.set_xlim(-0.45, 4.1)
    zx.set_ylim(0.9862, 0.9925)
    zx.set_yticks([0.987, 0.989, 0.991])
    zx.set_yticklabels([f"{t:.3f}" for t in zx.get_yticks()])

    footnote(fig, "Only uploads with a recorded public score are shown. The final ranking comes from the private leaderboard, which uses the last upload.")
    save(fig, "leaderboard_journey.png")


def local_vs_leaderboard():
    """Local validation against the public score, and the gap between them."""
    order = ["v1", "v2", "v3", "v6", "v11", "v15", "v17"]
    lb = dict(DATA["leaderboard"])
    older = DATA["local_older_metric"]
    weighted = DATA["local_weighted"]
    xs = list(range(len(order)))

    fig = plt.figure(figsize=(10, 4.9))
    heading(
        fig,
        "The gap between local and public scores shrank from 0.0214 to 0.0019",
        "Left: local validation and public F0.5 per upload. Right: how far the local number sat above the public one.",
    )
    ax = fig.add_axes([0.075, 0.2, 0.5, 0.6])
    gx = fig.add_axes([0.66, 0.2, 0.32, 0.6])

    style(ax)
    ax.axvline(2.5, color=BASE, linewidth=1, zorder=1)
    ax.text(2.58, 0.9945, "test-weighted metric from here on", ha="left", va="top", fontsize=8.5, color=INK_2)
    ax.plot(xs, [lb[v] for v in order], color=BLUE, linewidth=2, zorder=3)
    for x, v in zip(xs, order):
        dot(ax, x, lb[v], BLUE)
    ox = [i for i, v in enumerate(order) if v in older]
    ax.plot(ox, [older[order[i]] for i in ox], color=NEUTRAL, linewidth=1.4, zorder=2)
    for i in ox:
        dot(ax, i, older[order[i]], NEUTRAL, hollow=True)
    wx = [i for i, v in enumerate(order) if v in weighted]
    ax.plot(wx, [weighted[order[i]] for i in wx], color=ORANGE, linewidth=2, zorder=3)
    for i in wx:
        dot(ax, i, weighted[order[i]], ORANGE)
    ax.set_xticks(xs, order)
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(0.935, 0.995)
    ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    ax.set_yticklabels([f"{t:.2f}" for t in ax.get_yticks()])
    ax.set_ylabel("F0.5")
    handles = [
        Line2D([0], [0], color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label="Public leaderboard"),
        Line2D([0], [0], color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, label="Local, test-weighted F0.5"),
        Line2D([0], [0], color=NEUTRAL, linewidth=1.4, marker="o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=NEUTRAL, label="Local, older unweighted metric"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.01), handlelength=2.2)

    style(gx)
    gaps = [((weighted.get(v) or older.get(v)) - lb[v]) for v in order]
    colors = [NEUTRAL if v in older else ORANGE for v in order]
    gx.bar(xs, gaps, width=0.56, color=colors, zorder=3, linewidth=0)
    for x, g in zip(xs, gaps):
        gx.text(x, g + 0.0004, f"{g:.4f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    gx.set_xticks(xs, order)
    gx.set_xlim(-0.6, 6.6)
    gx.set_ylim(0, 0.024)
    gx.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
    gx.set_yticklabels(["0", "0.005", "0.010", "0.015", "0.020"])
    gx.set_ylabel("Local minus public")
    gx.legend(
        handles=[Patch(color=NEUTRAL, label="older metric"), Patch(color=ORANGE, label="test-weighted")],
        loc="upper right",
        handlelength=1.2,
    )

    footnote(
        fig,
        "Test-weighted: a false positive from a record with no true entity counts 1.887 times. The older metric used a different held-out slice.\n"
        "v17 changed only French rows, which have no labels, so its local score equals v15's.",
    )
    save(fig, "local_vs_leaderboard.png")


def local_gains():
    """Local F0.5 gain of every step from the 0.974 file to the final file."""
    ladder = DATA["local_ladder"]
    rows = []
    for (_, _, prev), (version, change, value) in zip(ladder, ladder[1:]):
        rows.append((version, change, value - prev))
    rows += DATA["late_increments"]

    fig = plt.figure(figsize=(10, 5.8))
    heading(
        fig,
        "Gains shrank from +0.00246 at v11 to +0.000075 at v21",
        "Test-weighted F0.5 gained on the validation slice by each step after the 0.974 file (0.98241 there, 0.98901 at v15).",
    )
    ax = fig.add_axes([0.4, 0.2, 0.54, 0.63])
    style(ax, grid_axis="x")
    n = len(rows)
    ys = list(range(n))[::-1]
    split = len(ladder) - 1
    colors = [ORANGE if i < split else BLUE for i in range(n)]
    ax.barh(ys, [r[2] for r in rows], height=0.56, color=colors, zorder=3, linewidth=0)
    for y, (version, change, gain) in zip(ys, rows):
        if gain < 5e-7:
            text = "0"
        elif abs(round(gain, 5) - gain) < 1e-9:
            text = f"+{gain:.5f}"
        else:
            text = f"+{gain:.6f}"
        ax.text(gain + 0.00003, y, text, ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [f"{r[0]}   {r[1]}" for r in rows], fontsize=9.5)
    ax.axhline(ys[split - 1] - 0.5, color=BASE, linewidth=1, zorder=1)
    ax.set_xlim(0, 0.0028)
    ax.set_xticks([0, 0.0005, 0.001, 0.0015, 0.002, 0.0025])
    ax.set_xticklabels(["0", "0.0005", "0.0010", "0.0015", "0.0020", "0.0025"])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("F0.5 gain on local validation")
    ax.legend(
        handles=[
            Patch(color=ORANGE, label="v10 to v15: change over the previous build"),
            Patch(color=BLUE, label="after v15: each change measured on its own"),
        ],
        loc="lower right",
        handlelength=1.2,
    )
    footnote(
        fig,
        "v17 changed only French rows, which have no labels, so it has no local gain.\n"
        "Name-key and hc/reverse rescue were scored on the half of the slice that was not used to pick their threshold.",
    )
    save(fig, "local_gains_by_step.png")


def cross_encoder_auc():
    """Band AUC of each model alone and of the growing stack."""
    single = DATA["band_auc_single"]
    stacked = DATA["band_auc_stacked"]
    stage2 = 0.9714

    fig = plt.figure(figsize=(10, 4.9))
    heading(
        fig,
        "No cross-encoder beats stage 2 alone, yet each one we kept lifted the stack",
        "ROC AUC on the 82,133 uncertain validation pairs (41.3% true). Right: out-of-fold AUC of the LightGBM stacker as models are added.",
    )
    ax = fig.add_axes([0.235, 0.22, 0.3, 0.58])
    sx = fig.add_axes([0.68, 0.22, 0.3, 0.58])

    style(ax, grid_axis="x")
    ys = list(range(len(single)))[::-1]
    ax.axvline(stage2, color=BASE, linewidth=1, zorder=1)
    for y, (label, v, kind) in zip(ys, single):
        color = {"gbm": NEUTRAL, "ce": BLUE, "dropped": BLUE}[kind]
        ax.plot([0.92, v], [y, y], color=GRID, linewidth=1, zorder=2)
        dot(ax, v, y, color, hollow=(kind == "dropped"))
        ax.text(v + 0.0018, y, f"{v:.4f}", ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [r[0] for r in single], fontsize=9)
    ax.set_xlim(0.92, 0.99)
    ax.set_xticks([0.92, 0.94, 0.96, 0.98])
    ax.set_xticklabels([f"{t:.2f}" for t in ax.get_xticks()])
    ax.set_ylim(-0.6, len(single) - 0.4)
    ax.set_xlabel("AUC, one model alone")

    style(sx, grid_axis="x")
    sy = list(range(len(stacked)))[::-1]
    sx.plot([r[1] for r in stacked], sy, color=BLUE, linewidth=2, zorder=3)
    for y, (label, v) in zip(sy, stacked):
        dot(sx, v, y, NEUTRAL if label == "Stage 2 alone" else BLUE)
        sx.text(v + 0.0018, y, f"{v:.4f}", ha="left", va="center", fontsize=8.5, color=INK)
    sx.set_yticks(sy, [r[0] for r in stacked], fontsize=9)
    sx.set_xlim(0.92, 0.99)
    sx.set_xticks([0.92, 0.94, 0.96, 0.98])
    sx.set_xticklabels([f"{t:.2f}" for t in sx.get_xticks()])
    sx.set_ylim(-0.6, len(stacked) - 0.4)
    sx.set_xlabel("AUC, stacked with stage 2")
    sx.legend(
        handles=[
            Line2D([0], [0], color=NEUTRAL, marker="o", markersize=7, markeredgecolor=SURFACE, linestyle="none", label="LightGBM"),
            Line2D([0], [0], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE, linestyle="none", label="cross-encoder"),
            Line2D([0], [0], color=BLUE, marker="o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=BLUE, linestyle="none", label="trained, not used"),
        ],
        loc="lower left",
    )

    footnote(
        fig,
        "Stacked AUC is out-of-fold, two folds split by Source 1 entity. ce3 alone beat ce2, but on top of kbase it moved the stacked AUC\n"
        "only from 0.9828 to 0.9830 and gave no F0.5 gain, so it was dropped.",
    )
    save(fig, "cross_encoder_auc.png")


def remaining_loss():
    """Where the v11 validation loss sat."""
    rows = DATA["loss_v11"]

    fig = plt.figure(figsize=(10, 3.9))
    heading(
        fig,
        "At v11 the loss was mostly recall, not precision",
        "F0.5 regained on validation if one kind of error were fixed on its own (v11, test-weighted, 110,565 held-out entities).",
    )
    ax = fig.add_axes([0.27, 0.2, 0.68, 0.56])
    style(ax, grid_axis="x")
    ys = list(range(len(rows)))[::-1]
    colors = [BLUE if r[2] == "recall" else ORANGE for r in rows]
    ax.barh(ys, [r[1] for r in rows], height=0.52, color=colors, zorder=3, linewidth=0)
    for y, (label, v, kind) in zip(ys, rows):
        ax.text(v + 0.00008, y, f"+{v:g}", ha="left", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 0.0082)
    ax.set_xticks([0, 0.002, 0.004, 0.006, 0.008])
    ax.set_xticklabels(["0", "0.002", "0.004", "0.006", "0.008"])
    ax.set_xlabel("F0.5 regained")
    ax.legend(
        handles=[Patch(color=BLUE, label="missed matches (recall)"), Patch(color=ORANGE, label="wrong merges (precision)")],
        loc="lower right",
        handlelength=1.2,
    )
    footnote(fig, "The three buckets overlap in the per-entity score, so they do not add up to the total loss.")
    save(fig, "remaining_loss.png")


def candidates_per_entity():
    """How many candidate pairs each test Source 1 entity has in the final candidate file."""
    buckets = DATA["candidates_buckets"]
    countries = DATA["candidates_country_mean"]
    overall = DATA["candidates_overall"]

    fig = plt.figure(figsize=(10, 4.5))
    heading(
        fig,
        "Blocking keeps about 11 candidates per business",
        "Final candidate file: 19,691,692 pairs over 1,732,544 test Source 1 entities.",
    )
    ax = fig.add_axes([0.085, 0.19, 0.5, 0.6])
    cx = fig.add_axes([0.68, 0.19, 0.3, 0.6])

    style(ax)
    xs = list(range(len(buckets)))
    counts = [b[1] for b in buckets]
    ax.bar(xs, counts, width=0.56, color=BLUE, zorder=3, linewidth=0)
    for x, n in zip(xs, counts):
        ax.text(x, n + 14000, f"{n:,}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs, [b[0] for b in buckets])
    ax.set_xlabel("Candidates per Source 1 entity")
    ax.set_ylim(0, 900000)
    ax.set_yticks([0, 200000, 400000, 600000, 800000])
    ax.set_yticklabels(["0", "200k", "400k", "600k", "800k"])
    ax.set_ylabel("Source 1 entities")

    style(cx)
    cxs = list(range(len(countries)))
    cx.bar(cxs, [c[1] for c in countries], width=0.56, color=BLUE, zorder=3, linewidth=0)
    for x, (name, v) in zip(cxs, countries):
        cx.text(x, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    cx.axhline(overall["mean"], color=INK_2, linewidth=1, zorder=4)
    cx.text(-0.5, overall["mean"] + 0.25, f"all countries: {overall['mean']:.2f}", ha="left", va="bottom", fontsize=8.5, color=INK_2)
    cx.set_xticks(cxs, [c[0] for c in countries])
    cx.set_xlim(-0.55, 2.55)
    cx.set_ylim(0, 17)
    cx.set_yticks([0, 5, 10, 15])
    cx.set_ylabel("Mean candidates per entity")

    footnote(
        fig,
        f"Median {overall['median']}, 90th percentile {overall['p90']}, 99th percentile {overall['p99']}, maximum {overall['max']:,}.",
    )
    save(fig, "candidates_per_entity.png")


def main():
    """Draw every chart."""
    leaderboard_journey()
    local_vs_leaderboard()
    local_gains()
    cross_encoder_auc()
    remaining_loss()
    candidates_per_entity()
    for path in sorted(ASSETS.glob("*.png")):
        print(path.relative_to(ASSETS.parent))


if __name__ == "__main__":
    main()
