"""Cross-method aggregation for the SAE-necessity suite (code plan C4).

Reads the canonical artefacts every source in the suite writes -- produced by
one audit harness on one split against one code panel -- and turns them into the
comparison tables and the coupling controls the meta-review's question actually
needs. No Modal, no re-computation of any statistic: everything here is
reshaping plus two controls.

Why the controls are not optional
---------------------------------
``specificity_ratio = |on_target_r| / mean|off_target_r|`` carries the on-target
correlation in its numerator. A method that grounds more strongly therefore
scores a higher ratio *even when its off-target leakage is identical*. Reading
that column down a table of methods with different on-target strength is
invalid, and plan item B1 says so: "Specificity and grounding are coupled: a
direction that barely correlates with its own code cannot show much off-target
leakage either."

Two independent ways out, both provided:

* ``mean_abs_off_r`` on its own -- off-target leakage with no on-target term in
  it, so it is not mechanically coupled and can be compared directly.
* ``matched_r_comparison`` -- restrict to codes where two methods reach
  comparable on-target |r| and compare there, paired by code.
* ``coupling_control`` -- regress leakage on on-target strength across all
  (method, code) points and compare per-method residuals, which uses every
  point instead of only the overlapping ones.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "load_source_per_code",
    "load_many",
    "summarise_methods",
    "matched_r_comparison",
    "coupling_control",
    "threshold_table",
    "monospecificity_table",
]

_PER_CODE_COLS = [
    "method",
    "code",
    "abs_on_target_r",
    "mean_abs_off_r",
    "specificity_ratio",
    "n_off_sig",
]


def load_source_per_code(audit_dir: str | Path, method: str) -> pd.DataFrame:
    """Per-code rows for one audited source, in the shared schema.

    Reads ``off_target_summary.csv`` -- the c-negative-restricted primary
    metric. The all-notes cross-check lives in
    ``off_target_summary_allnotes.csv`` and is loaded the same way by passing
    that directory's file explicitly.
    """
    audit_dir = Path(audit_dir)
    path = audit_dir / "off_target_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Per-code CSVs are git-ignored (results/README.md); "
            "re-pull with `modal volume get`."
        )
    df = pd.read_csv(path)
    df["method"] = method
    missing = [c for c in _PER_CODE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    return df[_PER_CODE_COLS].copy()


def load_many(sources: dict[str, str | Path]) -> pd.DataFrame:
    """Concatenate per-code rows for ``{method_name: audit_dir}``."""
    frames = [load_source_per_code(d, m) for m, d in sources.items()]
    out = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(out)} (method, code) rows across {len(sources)} methods")
    return out


def summarise_methods(per_code: pd.DataFrame) -> pd.DataFrame:
    """One row per method, sorted by off-target leakage (the uncoupled axis).

    Sorted by ``median_mean_abs_off_r`` rather than by specificity ratio
    deliberately: the ratio is the coupled quantity, and ordering a table by it
    is the mistake this module exists to prevent.
    """
    g = per_code.groupby("method", sort=False)
    out = pd.DataFrame(
        {
            "method": list(g.groups),
            "n_codes": g.size().to_numpy(),
            "median_abs_on_target_r": g["abs_on_target_r"].median().to_numpy(),
            "median_mean_abs_off_r": g["mean_abs_off_r"].median().to_numpy(),
            "median_specificity_ratio": g["specificity_ratio"].median().to_numpy(),
            "median_n_off_sig": g["n_off_sig"].median().to_numpy(),
        }
    )
    return out.sort_values("median_mean_abs_off_r").reset_index(drop=True)


def matched_r_comparison(
    per_code: pd.DataFrame,
    method_a: str,
    method_b: str,
    tol: float = 0.05,
) -> dict[str, Any]:
    """Compare two methods only on codes where their on-target |r| is comparable.

    This is plan item B1 made concrete. Codes are paired by ``code``; a pair is
    kept when ``|r_a - r_b| <= tol``. Within the kept set, leakage and
    specificity are compared with a paired Wilcoxon signed-rank test.

    Returns a dict with ``n_codes_matched`` (0 when the two methods never reach
    comparable strength, in which case the p-values are None and the comparison
    simply cannot be made -- which is itself a reportable result).
    """
    for m in (method_a, method_b):
        if m not in set(per_code["method"]):
            raise ValueError(f"method {m!r} is not present in the frame")

    a = per_code[per_code["method"] == method_a].set_index("code")
    b = per_code[per_code["method"] == method_b].set_index("code")
    codes = sorted(set(a.index) & set(b.index))
    a, b = a.loc[codes], b.loc[codes]

    keep = (a["abs_on_target_r"] - b["abs_on_target_r"]).abs() <= tol
    a_m, b_m = a[keep], b[keep]
    matched = list(a_m.index)

    out: dict[str, Any] = {
        "method_a": method_a,
        "method_b": method_b,
        "tol": tol,
        "n_codes_common": len(codes),
        "n_codes_matched": len(matched),
        "codes": matched,
    }
    if not matched:
        out.update(
            {
                "median_r_a": None, "median_r_b": None,
                "median_leakage_a": None, "median_leakage_b": None,
                "median_specificity_a": None, "median_specificity_b": None,
                "a_lower_leakage_on_n_codes": None,
                "leakage_wilcoxon_p": None,
                "specificity_wilcoxon_p": None,
                "note": "the two methods never reach comparable on-target |r|",
            }
        )
        return out

    def _w(x, y):
        # Wilcoxon needs at least one non-zero difference and n >= 1.
        d = np.asarray(x) - np.asarray(y)
        if len(d) < 1 or np.allclose(d, 0):
            return None
        try:
            return float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            return None

    out.update(
        {
            "median_r_a": float(a_m["abs_on_target_r"].median()),
            "median_r_b": float(b_m["abs_on_target_r"].median()),
            "median_leakage_a": float(a_m["mean_abs_off_r"].median()),
            "median_leakage_b": float(b_m["mean_abs_off_r"].median()),
            "median_specificity_a": float(a_m["specificity_ratio"].median()),
            "median_specificity_b": float(b_m["specificity_ratio"].median()),
            "a_lower_leakage_on_n_codes": int(
                (a_m["mean_abs_off_r"] < b_m["mean_abs_off_r"]).sum()
            ),
            "leakage_wilcoxon_p": _w(a_m["mean_abs_off_r"], b_m["mean_abs_off_r"]),
            "specificity_wilcoxon_p": _w(a_m["specificity_ratio"], b_m["specificity_ratio"]),
            "note": "",
        }
    )
    return out


def coupling_control(per_code: pd.DataFrame) -> pd.DataFrame:
    """Leakage residual after removing its dependence on on-target strength.

    Fits one OLS line ``mean_abs_off_r ~ abs_on_target_r`` across **all**
    (method, code) points, then reports each method's median residual. A
    negative residual means the method leaks less than its on-target strength
    predicts -- the coupling-free version of "more specific".

    Uses every point rather than only the overlapping ones, so it complements
    ``matched_r_comparison`` instead of duplicating it: the matched comparison
    is assumption-free but discards data, this uses all data but assumes the
    leakage/strength relation is roughly linear over the observed range.
    """
    df = per_code.dropna(subset=["abs_on_target_r", "mean_abs_off_r"])
    x = df["abs_on_target_r"].to_numpy(dtype=float)
    y = df["mean_abs_off_r"].to_numpy(dtype=float)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    resid = y - (slope * x + intercept)
    df = df.assign(leakage_residual=resid)

    g = df.groupby("method", sort=False)
    out = pd.DataFrame(
        {
            "method": list(g.groups),
            "n_codes": g.size().to_numpy(),
            "median_abs_on_target_r": g["abs_on_target_r"].median().to_numpy(),
            "median_mean_abs_off_r": g["mean_abs_off_r"].median().to_numpy(),
            "median_leakage_residual": g["leakage_residual"].median().to_numpy(),
        }
    )
    out.attrs["fit"] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_value),
        "p": float(p_value),
        "n_points": int(len(df)),
    }
    return out.sort_values("median_leakage_residual").reset_index(drop=True)


def threshold_table(sources: dict[str, str | Path]) -> pd.DataFrame:
    """Grounded-feature count per method at each |r| threshold.

    Read from each source's ``monospecificity.json``, whose ``n_grounded`` is
    the same statistic ``compute_grounding`` reports: BH-significant at q=0.05
    **and** |r| above the threshold.
    """
    rows: list[dict[str, Any]] = []
    for method, d in sources.items():
        mono = json.loads((Path(d) / "monospecificity.json").read_text())
        summary = json.loads((Path(d) / "audit_summary.json").read_text())
        row: dict[str, Any] = {
            "method": method,
            "k": summary["n_features"],
            "peak_abs_r": summary["max_abs_r_any_feature"],
        }
        for m in mono:
            row[f"grounded_r{m['threshold']}"] = m["n_grounded"]
        rows.append(row)
    return pd.DataFrame(rows)


def monospecificity_table(sources: dict[str, str | Path]) -> pd.DataFrame:
    """Monospecificity profile per method per threshold (plan item B3).

    Carries ``n_grounded`` alongside the fraction, because a fraction computed
    over one or two surviving features is not interpretable and must never be
    printed bare.
    """
    rows: list[dict[str, Any]] = []
    for method, d in sources.items():
        for m in json.loads((Path(d) / "monospecificity.json").read_text()):
            rows.append(
                {
                    "method": method,
                    "threshold": m["threshold"],
                    "n_grounded": m["n_grounded"],
                    "n_monospecific": m["n_monospecific"],
                    "frac_mono_of_grounded": m["frac_mono_of_grounded"],
                    "mean_codes_per_grounded": m["mean_codes_per_grounded"],
                }
            )
    return pd.DataFrame(rows)
