#!/usr/bin/env python3
"""Build the M5+M6 feature-walkthrough figure(s) (paper §3/§4 boundary).

Reads the JSON/CSV summaries already pulled from the Modal `sae-artifacts`
volume (default dir `/tmp/modal_pull`) and renders a converging-evidence
walkthrough for two representative features:

  * Hypothyroidism (ICD-9 2449): JumpReLU #2265 (+ #13990 PARTIAL contrast),
    causal arm from analogous Vanilla #3485.
  * Atrial fibrillation (ICD-9 42731): JumpReLU #6701 (peak r), causal arm
    from analogous Vanilla #1387.

Each feature's three validation arms (grounding -> concordance -> ablation) are
verified against their source files before rendering (verification-before-
completion), and every emitted note fragment is checked for HIPAA identifiers.

Output formats:
  --format pdf    two side-by-side flowcharts as a .pdf  (needs matplotlib;
                  run via `uv run --with matplotlib python ...`)
  --format tikz   inline TikZ figure, hypothyroidism only (fallback)
  --format table  compact booktabs table, hypothyroidism only (fallback)

If `/tmp/modal_pull` is empty, repopulate with `modal volume get` (see the
project CLAUDE.md). NOTE: pdf/tikz/table outputs contain short de-identified
MIMIC-IV note fragments; do NOT commit the generated file to the repo.

Usage:
    uv run --with matplotlib python scripts/build_feature_walkthrough.py \
        --format pdf --out /tmp/feature_walkthrough.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

# HIPAA Safe-Harbor identifier sniff test for emitted note fragments.
_HIPAA = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{2,4}-\d{2}-\d{2}|MRN|\d{6,}|age\s*\d|\d{1,3}\s*y/?o)\b",
    re.I,
)


def scan_fragment(text: str) -> list[str]:
    """Return HIPAA-identifier matches in `text` (empty list = clean)."""
    return _HIPAA.findall(text)


# Each concept = one flowchart. `expect` values guard against wrong source files.
CONCEPTS = [
    dict(
        title="Hypothyroidism (ICD-9 2449)",
        loc="problem-list lines",
        jr_ground="2265",
        jr_partial="13990",
        van="3485",
        expect=dict(icd="Unspecified hypothyroidism", r=0.847, delta=0.672, verdict="YES"),
    ),
    dict(
        title="Atrial fibrillation (ICD-9 42731)",
        loc="problem-list headers",
        jr_ground="6701",
        jr_partial=None,
        van="1387",
        expect=dict(icd="Atrial fibrillation", r=0.864, delta=0.427, verdict="YES"),
    ),
]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_inspection(src: Path) -> dict[str, dict]:
    data = json.loads((src / "jumprelu_feature_inspection.json").read_text())
    return {str(lat["latent_idx"]): lat for lat in data["latents"]}


def load_concordance(src: Path) -> dict[str, dict]:
    return {
        r["feature_idx"]: r for r in csv.DictReader((src / "jr_concordance_results.csv").open())
    }


def load_catalog(src: Path) -> dict[str, dict]:
    return {
        r["feature_idx"]: r for r in csv.DictReader((src / "autointerp_feature_catalog.csv").open())
    }


def load_vanilla_ablation(src: Path) -> dict[str, dict]:
    """Vanilla ablation rows across pilot + extended runs, grounded targets only."""
    rows: dict[str, dict] = {}
    for fname in ("vanilla_pilot_results.csv", "vanilla_ablation_results.csv"):
        path = src / fname
        if path.exists():
            for r in csv.DictReader(path.open()):
                if r.get("kind", "grounded") == "grounded":
                    rows[r["feature"]] = r
    return rows


# --------------------------------------------------------------------------- #
# Fragment reconstruction (PHI-sensitive)
# --------------------------------------------------------------------------- #
def fragment(lat: dict, pre: int = 3, post: int = 2) -> tuple[str, str]:
    """Reconstruct a short readable fragment around the top-activating trigger.

    Returns (raw_text, trigger). Derived from the inspection JSON's context
    window so no note text is stored in this script. Section text after a colon
    and leading list markers are dropped to keep the fragment minimal.
    """
    t = lat["top_tokens"][0]
    ctx, tok = t["context"], t["token_str"]
    ti = ctx.index(tok) if tok in ctx else len(ctx) // 2
    raw = "".join(ctx[max(0, ti - pre) : ti + post + 1]).replace("\n", " / ")
    raw = re.sub(r"\s+", " ", raw).split(":")[0].strip(" /-#").strip()
    return raw, tok


def _bold(raw: str, trig: str, wrap: str) -> str:
    """Wrap the (last) trigger occurrence with `wrap` ('latex' or 'mpl')."""
    idx = raw.rfind(trig)
    if idx < 0:
        return raw
    mark = {"latex": (r"\textbf{", "}"), "mpl": (r"$\mathbf{", "}$")}[wrap]
    return raw[:idx] + mark[0] + trig + mark[1] + raw[idx + len(trig) :]


# --------------------------------------------------------------------------- #
# Gather + verify
# --------------------------------------------------------------------------- #
def gather_concept(src: Path, spec: dict) -> dict:
    insp, conc, cat, abl = (
        load_inspection(src),
        load_concordance(src),
        load_catalog(src),
        load_vanilla_ablation(src),
    )
    g = insp[spec["jr_ground"]]
    cg = conc[spec["jr_ground"]]
    v = abl[spec["van"]]
    frag_g_raw, trig_g = fragment(g)

    vals = dict(
        title=spec["title"],
        loc=spec["loc"],
        jr_ground=spec["jr_ground"],
        jr_partial=spec["jr_partial"],
        van=spec["van"],
        icd_desc=cg["concordance_icd_description"],
        ground_r=float(cg["concordance_r_pb"]),
        ground_ratio=g["firing_stats"]["firing_rate_ratio"],
        ground_verdict=cg["concordance_verdict"],
        frag_g_raw=frag_g_raw,
        trig_g=trig_g,
        ablate_delta=float(v["cliffs_delta"]),
        ablate_mag=v["delta_magnitude"],
        ablate_pbh=float(v["p_bh"]),
        ablate_npos=int(v["n_pos"]),
        category=cat[spec["jr_ground"]]["category"],
    )

    frag_p_raw = trig_p = None
    if spec["jr_partial"]:
        p, cp = insp[spec["jr_partial"]], conc[spec["jr_partial"]]
        frag_p_raw, trig_p = fragment(p)
        vals.update(
            partial_r=float(cp["concordance_r_pb"]),
            partial_verdict=cp["concordance_verdict"],
            frag_p_raw=frag_p_raw,
            trig_p=trig_p,
        )

    # --- verification against source / expected values ---
    e = spec["expect"]
    assert vals["icd_desc"] == e["icd"], f'{spec["title"]}: icd {vals["icd_desc"]!r}'
    assert math.isclose(vals["ground_r"], e["r"], abs_tol=5e-3), vals["ground_r"]
    assert math.isclose(vals["ablate_delta"], e["delta"], abs_tol=5e-3), vals["ablate_delta"]
    assert vals["ground_verdict"] == e["verdict"], vals["ground_verdict"]
    # --- PHI guard: emitted fragments must carry no HIPAA identifiers ---
    for fid, raw in [(spec["jr_ground"], frag_g_raw)] + (
        [(spec["jr_partial"], frag_p_raw)] if frag_p_raw else []
    ):
        hits = _HIPAA.findall(raw)
        assert not hits, f"HIPAA identifier in fragment #{fid}: {hits} ({raw!r})"
        print(f"[phi] #{fid}: {raw!r}  (no identifiers)")
    print(f"[ok] {spec['title']}: grounding/concordance/ablation verified")
    return vals


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def latex_sci(x: float) -> str:
    exp = math.floor(math.log10(x))
    return rf"{x / 10 ** exp:.1f}{{\times}}10^{{{exp}}}"


def render_pdf(concepts: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(concepts), figsize=(7.0, 5.4))
    axes = [axes] if len(concepts) == 1 else list(axes)
    ys = dict(feat=0.93, a1=0.72, a2=0.50, a3=0.28, claim=0.06)

    for ax, c in zip(axes, concepts, strict=False):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(c["title"], fontsize=10, fontweight="bold", pad=4)

        frag = _bold(c["frag_g_raw"], c["trig_g"], "mpl")
        conc = rf'explanation $\Rightarrow$ {c["ground_verdict"]}'
        if c["jr_partial"]:
            conc += f'\n(#{c["jr_partial"]} {_bold(c["frag_p_raw"], c["trig_p"], "mpl")}'
            conc += rf' $\Rightarrow$ {c["partial_verdict"]})'
        boxes = {
            "feat": (f'JumpReLU #{c["jr_ground"]}\nfires on "{frag}"\n({c["loc"]})', "0.92"),
            "a1": (
                rf'1. Statistical grounding'
                rf'\n$r_{{pb}}={c["ground_r"]:.3f}$'
                f'\n({c["ground_ratio"]:.1f}' + r"$\times$ more on +notes)",
                "white",
            ),
            "a2": (f"2. Semantic concordance\n{conc}", "white"),
            "a3": (
                r"3. Causal ablation$^\dagger$"
                + rf"\nCliff's $\delta={c['ablate_delta']:.3f}$ ({c['ablate_mag']})"
                + f"\n(Vanilla #{c['van']})",
                "white",
            ),
            "claim": (r"$\Rightarrow$ a genuine clinical concept", "0.92"),
        }
        for key, (text, fc) in boxes.items():
            ax.text(
                0.5,
                ys[key],
                text.replace(r"\n", "\n"),
                ha="center",
                va="center",
                fontsize=7.5,
                linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec="black", lw=0.8),
            )
        for upper, lower in (("feat", "a1"), ("a1", "a2"), ("a2", "a3"), ("a3", "claim")):
            ax.annotate(
                "",
                xy=(0.5, ys[lower] + 0.055),
                xytext=(0.5, ys[upper] - 0.055),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"),
            )

    fig.savefig(out, bbox_inches="tight")  # format inferred from suffix (.pdf/.png)
    print(f"[ok] wrote {out}")


_TIKZ = r"""% --- Feature walkthrough (hypothyroidism) -- TikZ fallback ---
% Preamble: \usepackage{tikz}\usetikzlibrary{arrows.meta}
% NOTE: contains de-identified MIMIC-IV note fragments; do NOT commit.
\begin{figure}[t]\centering
\begin{tikzpicture}[font=\small,>={Latex[length=2mm]},
  b/.style={draw,rounded corners,align=center,inner sep=5pt,text width=0.42\textwidth}]
\node[b,fill=black!5] (f) {\textbf{JumpReLU \#@JRG@} — fires on ``@FRAGG@'' (@LOC@)};
\node[b,below=4mm of f] (a1) {\textbf{1. Grounding} — $r_{\text{pb}}{=}@RG@$};
\node[b,below=4mm of a1] (a2) {\textbf{2. Concordance} — \textsc{@VG@}};
\node[b,below=4mm of a2] (a3) {\textbf{3. Causal}$^{\dagger}$ — $\delta{=}@DELTA@$ (@MAG@), Vanilla \#@VAN@};
\node[b,fill=black!5,below=4mm of a3] (c) {$\Rightarrow$ a genuine clinical concept};
\draw[->](f)--(a1);\draw[->](a1)--(a2);\draw[->](a2)--(a3);\draw[->](a3)--(c);
\end{tikzpicture}
\caption{Converging-evidence validation on a representative feature.}
\label{fig:feature-walkthrough}
\end{figure}"""


def render_tikz(c: dict) -> str:
    repl = {
        "@JRG@": c["jr_ground"],
        "@VAN@": c["van"],
        "@LOC@": c["loc"],
        "@FRAGG@": _bold(c["frag_g_raw"], c["trig_g"], "latex"),
        "@RG@": f'{c["ground_r"]:.3f}',
        "@VG@": c["ground_verdict"].lower(),
        "@DELTA@": f'{c["ablate_delta"]:.3f}',
        "@MAG@": c["ablate_mag"],
    }
    out = _TIKZ
    for k, val in repl.items():
        out = out.replace(k, val)
    return out


def render_table(c: dict) -> str:
    frag_g = _bold(c["frag_g_raw"], c["trig_g"], "latex")
    return rf"""% --- Feature walkthrough table (hypothyroidism) -- fallback ---
\begin{{table}}[t]\centering\small
\caption{{Worked example: hypothyroidism (ICD-9 2449) across three modalities.
$^{{\dagger}}$Vanilla feature shown ($p_{{\text{{BH}}}}{{=}}{latex_sci(c['ablate_pbh'])}$,
$n_{{+}}{{=}}{c['ablate_npos']}$); JumpReLU ablation not run (\S\ref{{sec:ablation}}).}}
\label{{tab:feature-walkthrough}}
\begin{{tabular}}{{@{{}}lll@{{}}}}
\toprule
Modality & Feature & Result \\
\midrule
\multicolumn{{3}}{{@{{}}l@{{}}}}{{\textit{{Activating context:}} ``{frag_g}''}} \\
\midrule
Grounding   & JR\,\#{c['jr_ground']}  & $r_{{\text{{pb}}}}{{=}}{c['ground_r']:.3f}$ \\
Concordance & JR\,\#{c['jr_ground']}  & \textsc{{{c['ground_verdict'].lower()}}} \\
Causal$^{{\dagger}}$ & Van\,\#{c['van']} & $\delta{{=}}{c['ablate_delta']:.3f}$ ({c['ablate_mag']}) \\
\bottomrule
\end{{tabular}}
\end{{table}}"""


def render_trace(trace_path: Path, src: Path, out: Path) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch

    INK, MUTE, FAINT = "#1a1a2e", "#5a6472", "#9aa3af"
    cmap = LinearSegmentedColormap.from_list("act", ["#fff7ec", "#fdae6b", "#d7301f", "#7f0000"])

    data = json.loads(Path(trace_path).read_text())
    conc, cat, abl = load_concordance(src), load_catalog(src), load_vanilla_ablation(src)
    van_by_lat = {c["jr_ground"]: c for c in CONCEPTS}  # JumpReLU latent -> concept
    traces = data["traces"]
    n = len(traces)

    fig = plt.figure(figsize=(7.6, 1.95 * n + 0.9))
    gs = fig.add_gridspec(n, 1, hspace=0.95, top=0.9, bottom=0.16, left=0.02, right=0.98)

    for idx, (lat, tr) in enumerate(traces.items()):
        ax = fig.add_subplot(gs[idx])
        toks, acts = tr["tokens"], tr["activations"]
        if scan_fragment(" ".join(toks)):
            raise AssertionError(f"HIPAA identifier in latent {lat} window")
        amax = max(acts) or 1.0
        ax.set_xlim(-0.3, len(toks) + 0.3)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for i, (tok, a) in enumerate(zip(toks, acts, strict=False)):
            shade = max(a, 0.0) / amax
            empty = not tok.strip()
            facecolor = "#f0f1f3" if (empty or shade < 1e-6) else cmap(shade)
            ax.add_patch(
                FancyBboxPatch(
                    (i + 0.08, 0.34),
                    0.84,
                    0.40,
                    boxstyle="round,pad=0,rounding_size=0.12",
                    linewidth=0,
                    facecolor=facecolor,
                )
            )
            if not empty:
                label = tok.strip()
                ax.text(
                    i + 0.5,
                    0.54,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.5 if len(label) <= 7 else 6.8,  # shrink long tokens to fit the chip
                    color="white" if shade > 0.55 else INK,
                    fontweight="bold" if shade > 0.55 else "normal",
                )

        c, k, cc = conc.get(lat, {}), cat.get(lat, {}), van_by_lat.get(lat)
        title = cc["title"] if cc else f"Feature #{lat}"
        ax.text(-0.3, 0.94, title, fontsize=10.5, fontweight="bold", color=INK, va="center")

        arms = [
            f"grounding $r$={float(c.get('concordance_r_pb', 0)):.3f}",
            f"concordance {c.get('concordance_verdict', '')}",
        ]
        if cc and cc["van"] in abl:
            arms.append(rf"causal $\delta$={float(abl[cc['van']]['cliffs_delta']):.2f}$^\dagger$")
        ax.text(
            len(toks) + 0.3,
            0.94,
            "      ".join(arms),
            fontsize=8,
            color=MUTE,
            ha="right",
            va="center",
        )
        concept = (k.get("explanation", "")[:84] + "…") if k else ""
        ax.text(
            -0.3,
            0.10,
            f"JumpReLU #{lat}  ·  {concept}",
            fontsize=7.5,
            style="italic",
            color=MUTE,
            va="center",
        )

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.40, 0.085, 0.20, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["inactive", "peak"])
    cb.ax.tick_params(length=0, labelsize=6.5, colors=MUTE)
    cb.outline.set_visible(False)

    fig.suptitle(
        "Per-token SAE feature activation over a de-identified clinical note window",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.025,
        r"$\dagger$ causal arm from the analogous Vanilla feature; JumpReLU ablation was not run.",
        ha="center",
        fontsize=6.5,
        style="italic",
        color=FAINT,
    )
    fig.savefig(out, dpi=200)
    print(f"[ok] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/modal_pull", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--format", choices=("pdf", "tikz", "table", "trace"), default="pdf")
    ap.add_argument("--trace-json", default="/tmp/modal_pull/feature_trace.json", type=Path)
    args = ap.parse_args()

    if args.format == "trace":
        render_trace(args.trace_json, args.src, args.out or Path("/tmp/feature_trace.pdf"))
        return

    concepts = [gather_concept(args.src, spec) for spec in CONCEPTS]

    if args.format == "pdf":
        render_pdf(concepts, args.out or Path("/tmp/feature_walkthrough.pdf"))
        return
    latex = render_tikz(concepts[0]) if args.format == "tikz" else render_table(concepts[0])
    if args.out:
        args.out.write_text(latex + "\n")
        print(f"[ok] wrote {args.out}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
