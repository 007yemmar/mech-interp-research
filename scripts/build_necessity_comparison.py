#!/usr/bin/env python
"""Assemble the SAE-necessity comparison (code plan C4).

Reads every audited source under ``results/necessity/`` -- all produced by one
audit harness, on one split, against one pinned 46-code panel -- and emits the
tables plan items B1-B3 call for. No Modal, no re-computation of any audit
statistic: this is reshaping plus the two coupling controls.

    uv run python scripts/build_necessity_comparison.py

Outputs (under results/necessity/comparison/):
    comparison_summary.json     tracked; every headline number
    threshold_table.csv         B-table: grounded count per method per |r|
    per_method_summary.csv      leakage / specificity / n_off_sig per method
    matched_r_comparisons.csv   B1: paired comparison at comparable |r|
    coupling_control.csv        B1 alternative: leakage residual vs strength
    monospecificity_table.csv   B3: mono profile per method per threshold
    figures/fig_necessity_specificity.png/.pdf   B1 scatter

Per-code CSVs are git-ignored (results/README.md), so re-pull them with
`modal volume get` before running if the working tree is clean.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from mech_interp_research.necessity_comparison import (
    coupling_control,
    load_many,
    matched_r_comparison,
    monospecificity_table,
    summarise_methods,
    threshold_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("necessity_comparison")

ROOT = Path(__file__).resolve().parents[1]
NEC = ROOT / "results" / "necessity"
OUT = NEC / "comparison"
FIGS = ROOT / "figures"

# Every source in the suite, keyed by the label used in the paper's tables.
# One arm per method: the thresholded random/PCA arms are near-identical to
# their dense counterparts (documented in the C5 run log), so listing all of
# them would pad the table without adding information.
SOURCES: dict[str, Path] = {
    "diff-in-means (plain)": NEC / "direction_audit" / "diff_in_means_none",
    "diff-in-means (diagonal)": NEC / "direction_audit" / "diff_in_means_diagonal",
    "diff-in-means (LDA)": NEC / "direction_audit" / "diff_in_means_full",
    "probe LR (balanced)": NEC / "direction_audit" / "probe_lr_balanced",
    "probe LR (unweighted)": NEC / "direction_audit" / "probe_lr_unweighted",
    "PCA": NEC / "pca" / "seed0" / "audit_dense",
    "random (dense)": NEC / "random_matched" / "seed0" / "audit_dense",
    "random (L0-matched)": NEC / "random_matched" / "seed0" / "audit_l0_40.92",
    "GemmaScope SAE": NEC / "sae_audit" / "gemmascope",
    "vanilla SAE": NEC / "sae_audit" / "sae_vanilla",
    "JumpReLU SAE": NEC / "sae_audit" / "sae_jumprelu",
    # BOS-free re-pool of the constructed arms. Row 0 of every note is
    # Gemma's <bos> (||x|| = 2528.6 vs a ~162 median); under max-pooling it
    # floors the note value, which can INFLATE point-biserial by collapsing
    # within-negative variance. The SAE arms are NOT re-pooled: 0 of 60
    # top-grounded latents fire at BOS in either domain-trained SAE, so
    # their pooled values are unaffected. See
    # docs/2026-08-29-bos-contamination-audit.md.
    #
    # These sit BESIDE the originals rather than replacing them: the
    # before/after pair is the result, not the corrected number alone.
    "diff-in-means (LDA) [no-BOS]": NEC / "direction_audit_nobos" / "diff_in_means_full",
    "diff-in-means (diagonal) [no-BOS]": NEC / "direction_audit_nobos" / "diff_in_means_diagonal",
    "diff-in-means (plain) [no-BOS]": NEC / "direction_audit_nobos" / "diff_in_means_none",
    "random (dense) [no-BOS]": NEC / "random_matched_nobos" / "seed0" / "audit_dense",
    "random (L0-matched) [no-BOS]": NEC / "random_matched_nobos" / "seed0" / "audit_l0_40.92",
}

# Pairs the paper actually argues about. Each SAE against each non-SAE family,
# plus the two SAE-vs-SAE contrasts the paper makes in its own right.
PAIRS = [
    ("vanilla SAE", "diff-in-means (LDA)"),
    ("vanilla SAE", "probe LR (unweighted)"),
    ("vanilla SAE", "PCA"),
    ("vanilla SAE", "random (L0-matched)"),
    ("vanilla SAE", "GemmaScope SAE"),
    ("JumpReLU SAE", "diff-in-means (LDA)"),
    ("JumpReLU SAE", "probe LR (unweighted)"),
    ("JumpReLU SAE", "GemmaScope SAE"),
    # An SAE that was never trained on this domain, against the best non-SAE
    # direction method at comparable on-target strength. This is the cleanest
    # available test of whether the ARCHITECTURE buys specificity, separate from
    # the paper's own claim that domain training buys more.
    ("GemmaScope SAE", "diff-in-means (LDA)"),
    ("GemmaScope SAE", "probe LR (unweighted)"),
]


def main() -> None:
    missing = [m for m, d in SOURCES.items() if not (d / "off_target_summary.csv").exists()]
    if missing:
        raise SystemExit(
            "Missing per-code CSVs for: "
            + ", ".join(missing)
            + "\nThese are git-ignored; re-pull with `modal volume get` (results/README.md)."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    per_code = load_many(SOURCES)
    per_code.to_csv(OUT / "per_code_all_methods.csv", index=False)

    # --- B2: per-method aggregation ------------------------------------
    summary = summarise_methods(per_code)
    summary.to_csv(OUT / "per_method_summary.csv", index=False)

    # --- threshold table -------------------------------------------------
    thresh = threshold_table(SOURCES)
    thresh.to_csv(OUT / "threshold_table.csv", index=False)

    # --- B3: monospecificity --------------------------------------------
    mono = monospecificity_table(SOURCES)
    mono.to_csv(OUT / "monospecificity_table.csv", index=False)

    # --- B1: the coupling controls ---------------------------------------
    matched = pd.DataFrame([matched_r_comparison(per_code, a, b) for a, b in PAIRS])
    matched.to_csv(OUT / "matched_r_comparisons.csv", index=False)

    ctrl = coupling_control(per_code)
    ctrl.to_csv(OUT / "coupling_control.csv", index=False)

    _figure(per_code)

    payload = {
        "n_methods": len(SOURCES),
        "n_codes": int(per_code.groupby("method").size().max()),
        "split": {"audit_shards": [281, 312], "n_audit_notes": 4911},
        "per_method_summary": summary.to_dict(orient="records"),
        "threshold_table": thresh.to_dict(orient="records"),
        "coupling_control": {
            "fit": ctrl.attrs["fit"],
            "by_method": ctrl.to_dict(orient="records"),
        },
        "matched_r_comparisons": matched.drop(columns=["codes"]).to_dict(orient="records"),
        "monospecificity": mono.to_dict(orient="records"),
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {OUT}/comparison_summary.json and 6 CSVs")

    print(summary.to_string(index=False))
    print()
    print(ctrl.to_string(index=False))


def _figure(per_code: pd.DataFrame) -> None:
    """B1 scatter: off-target leakage against on-target strength, per code.

    Plotted as leakage rather than specificity ratio on purpose -- the ratio has
    on-target r in its numerator, so a ratio-vs-r scatter would show a
    relationship that is partly arithmetic. Leakage vs strength shows only what
    is empirical.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    families = {
        "SAE": ["vanilla SAE", "JumpReLU SAE", "GemmaScope SAE"],
        "supervised direction": [
            "diff-in-means (plain)",
            "diff-in-means (diagonal)",
            "diff-in-means (LDA)",
            "probe LR (balanced)",
            "probe LR (unweighted)",
        ],
        "unsupervised / null": ["PCA", "random (dense)", "random (L0-matched)"],
    }
    colors = {"SAE": "#1b6ca8", "supervised direction": "#c8562b", "unsupervised / null": "#7a7a7a"}
    markers = {"SAE": "o", "supervised direction": "s", "unsupervised / null": "^"}

    for fam, members in families.items():
        sub = per_code[per_code["method"].isin(members)]
        ax.scatter(
            sub["abs_on_target_r"],
            sub["mean_abs_off_r"],
            s=16,
            alpha=0.55,
            c=colors[fam],
            marker=markers[fam],
            label=fam,
            edgecolors="none",
        )
    for m, c in (("vanilla SAE", "#0d3d5c"), ("diff-in-means (LDA)", "#7a2f12")):
        sub = per_code[per_code["method"] == m]
        ax.scatter(
            sub["abs_on_target_r"],
            sub["mean_abs_off_r"],
            s=22,
            facecolors="none",
            edgecolors=c,
            linewidths=0.9,
        )

    ax.set_xlabel("on-target |r| (held-out, per code)")
    ax.set_ylabel("off-target leakage  mean|r| over other codes (c-negative)")
    ax.set_title("Off-target leakage vs on-target strength (held-out, per code)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"fig_necessity_specificity.{ext}", dpi=200)
    plt.close(fig)
    logger.info(f"Wrote {FIGS}/fig_necessity_specificity.png/.pdf")


if __name__ == "__main__":
    main()
