"""Figure: describability against grounding strength, four judges, five sources.

Supersedes `results/concordance_validation/tables_png/F1_curve.png`, which showed a
single judge over six fine bands and seven sources. This version uses the three
disjoint bands the paper reports in, all four judges, and the five sources that carry
the argument: a lexical ceiling, a matched random floor, the strongest domain SAE, the
general-purpose SAE that isolates domain training, and the label-supervised direction
that carries the prediction/description dissociation.

Every value is the held-out audit split and is transcribed from RESULTS.md §14.1
(hit@1) and the exact-YES recomputation in §14.4. Run:

    uv run --with matplotlib python scripts/make_concordance_figure.py
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

BANDS = ["0.1–0.3", "0.3–0.5", "> 0.5"]
JUDGES = ["Sonnet 4.6", "DeepSeek-V3", "GPT-5-mini", "Gemini 2.5 Flash"]

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs clean).
# The two bounds are deliberately NOT given categorical hues: they frame the
# comparison rather than compete in it.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CEIL, FLOOR = "#8d8d86", "#3f3f3b"
INK, MUTED, GRID = "#16171a", "#5c5c57", "#e3e3df"

#                       label                     colour  style  marker  zorder
SOURCES = [
    ("keyword", "keyword (lexical ceiling)", CEIL, (0, (5, 2)), "o", 2),
    ("jumprelu", "JumpReLU SAE (domain)", BLUE, "-", "D", 5),
    ("gemmascope", "GemmaScope SAE (general)", ORANGE, "-", "s", 4),
    ("dm", "diff-in-means (supervised)", AQUA, "-", "^", 4),
    ("random", "random directions (floor)", FLOOR, (0, (1, 1.6)), "v", 3),
]

# Per-judge values, in JUDGES order. A band absent from a dict is a band in which
# the source owns fewer than five held-out features -- a hard scarcity, not a gap.
HIT1 = {
    "keyword": {0: [50, 83, 83, 83], 1: [92, 96, 96, 100], 2: [100, 100, 100, 100]},
    "jumprelu": {0: [9, 12, 7, 9], 1: [73, 73, 75, 70], 2: [92, 92, 94, 91]},
    "gemmascope": {0: [0, 0, 0, 0], 1: [52, 46, 52, 50]},
    "dm": {0: [22, 28, 22, 28], 1: [24, 38, 29, 29], 2: [43, 29, 29, 29]},
    "random": {0: [17, 18, 23, 16], 1: [38, 44, 38, 38]},
}
EXACT = {
    "keyword": {0: [67, 50, 67, 50], 1: [85, 77, 96, 92], 2: [100, 83, 100, 83]},
    "jumprelu": {0: [0, 0, 0, 0], 1: [15, 19, 21, 19], 2: [43, 42, 60, 51]},
    "gemmascope": {0: [0, 0, 0, 0], 1: [6, 10, 12, 12]},
    "dm": {0: [6, 11, 11, 6], 1: [19, 29, 24, 19], 2: [14, 29, 29, 29]},
    "random": {0: [2, 2, 3, 2], 1: [0, 0, 0, 0]},
}
# Features per source per band; shared across judges.
N = {
    "keyword": {0: 6, 1: 26, 2: 6},
    "jumprelu": {0: 139, 1: 175, 2: 144},
    "gemmascope": {0: 100, 1: 50},
    "dm": {0: 18, 1: 21, 2: 7},
    "random": {0: 284, 1: 16},
}


# Grounded features in each band across the source's FULL dictionary, derived from
# the threshold counts in RESULTS.md §13.2 (disjoint: >0.1 minus >0.3, etc.). This is
# the population the judge pools are drawn from -- for the two SAEs the judged pool is
# a band-stratified sample of it; for every other source here it is the whole thing.
POPULATION = {
    "keyword": [6, 26, 6],
    "jumprelu": [9111, 463, 147],
    "gemmascope": [5736, 50, 4],
    "dm": [18, 21, 7],
    "random": [6929, 16, 0],
}


def median(v: list[float]) -> float:
    s = sorted(v)
    return (s[1] + s[2]) / 2 if len(s) == 4 else s[len(s) // 2]


def population_panel(ax):
    """How many features each source owns in each band -- bars, not lines, so the
    change of unit is unmistakable and no one reads it as a second rate axis."""
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    width = 0.16
    for i, (key, _label, colour, _style, _marker, _z) in enumerate(SOURCES):
        off = (i - 2) * width
        for b, count in enumerate(POPULATION[key]):
            x = b + off
            if count == 0:
                ax.text(x, 1.25, "0", ha="center", va="bottom", fontsize=6.2, color=colour)
                continue
            ax.bar(x, count, width=width * 0.88, color=colour, zorder=3, linewidth=0)

    ax.set_yscale("log")
    ax.set_ylim(1, 30000)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1k", "10k"], fontsize=8, color=MUTED)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS, fontsize=8.4, color=INK)
    ax.set_xlim(-0.55, 2.55)
    ax.tick_params(length=0, pad=5)
    ax.set_xlabel("grounding strength  $|r_{pb}|$", fontsize=8.6, color=INK, labelpad=7)
    ax.set_title(
        "Features that exist there", fontsize=9.4, color=INK, pad=9, loc="left", fontweight="bold"
    )


def panel(ax, data, title, *, chance=None):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    if chance is not None:
        ax.axhline(chance, color=MUTED, lw=0.8, ls=(0, (1, 2.5)), zorder=1)
        ax.text(
            2.15,
            chance,
            "chance",
            color=MUTED,
            fontsize=7.2,
            va="center",
            ha="right",
            bbox=dict(fc="white", ec="none", pad=1.6),
        )

    for key, label, colour, style, marker, z in SOURCES:
        pts = data.get(key, {})
        xs = sorted(pts)
        if not xs:
            continue
        ys = [median(pts[x]) for x in xs]
        # Individual judges as translucent dots: where they stack the panel shows
        # agreement, where they scatter it shows a small-n cell. This is the
        # uncertainty display -- no separate error bar is needed.
        for x in xs:
            ax.scatter(
                [x] * len(pts[x]),
                pts[x],
                s=13,
                color=colour,
                alpha=0.28,
                linewidths=0,
                zorder=z,
            )
        ax.plot(
            xs,
            ys,
            color=colour,
            ls=style,
            lw=1.9,
            marker=marker,
            ms=5.2,
            mec="white",
            mew=0.9,
            zorder=z + 5,
            clip_on=False,
            label=label,
        )

    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS, fontsize=8.4, color=INK)
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=8, color=MUTED)
    ax.tick_params(length=0, pad=5)
    ax.set_xlabel("grounding strength  $|r_{pb}|$", fontsize=8.6, color=INK, labelpad=7)
    ax.set_title(title, fontsize=9.4, color=INK, pad=9, loc="left", fontweight="bold")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [0.82, 1, 1]})
    fig.subplots_adjust(left=0.055, right=0.992, top=0.885, bottom=0.30, wspace=0.28)

    population_panel(axes[0])
    panel(axes[1], HIT1, "Forced choice, same-chapter  (hit@1)", chance=11.1)
    panel(axes[2], EXACT, "Unaided recall  (exact-YES)")

    fig.text(
        0.5,
        0.028,
        "All four judges: Claude Sonnet 4.6 · DeepSeek-V3 · GPT-5-mini · Gemini 2.5 Flash.   "
        "Line = median across the four; dots = the judges individually.",
        fontsize=7.4,
        color=MUTED,
        ha="center",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(lbl) for _, lbl, *_ in SOURCES if lbl in labels]
    leg = fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.082),
        ncol=5,
        frameon=False,
        fontsize=7.6,
        handlelength=2.0,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        text.set_color(INK)

    out = pathlib.Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_concordance_bands.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", out / "fig_concordance_bands.{pdf,png}")

    for name, data in (("hit@1", HIT1), ("exact-YES", EXACT)):
        print(f"\n{name}: median across four judges")
        for key, label, *_ in SOURCES:
            cells = [
                f"{BANDS[b]}: {median(data[key][b]):5.1f} (n={N[key][b]})"
                for b in sorted(data.get(key, {}))
            ]
            print(f"   {label:28s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
