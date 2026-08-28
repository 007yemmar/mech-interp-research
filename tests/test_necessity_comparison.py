"""Tests for the cross-method necessity comparison (code plan C4).

The statistically load-bearing pieces are the two coupling controls. Everything
else in this module is reshaping, and is covered end to end.

Why the controls exist: `specificity_ratio = |on_target_r| / mean|off_target_r|`
has the on-target correlation in its numerator, so a method that grounds more
strongly scores a higher ratio even if its off-target leakage is identical.
Reading that column down a table of methods with different on-target strength is
therefore invalid -- which is exactly what plan item B1 warns about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mech_interp_research.necessity_comparison import (
    coupling_control,
    matched_r_comparison,
    summarise_methods,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["method", "code", "abs_on_target_r", "mean_abs_off_r",
                                       "specificity_ratio", "n_off_sig"])


def _synthetic(n_codes=20, seed=0):
    """Two methods with IDENTICAL leakage but different on-target strength.

    The specificity ratio must separate them (it is coupled); the leakage
    comparison must not (it is not). That is the whole point of the control.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_codes):
        leak = 0.05 + 0.005 * rng.standard_normal()
        for method, on in (("strong", 0.60), ("weak", 0.20)):
            r = on + 0.02 * rng.standard_normal()
            rows.append([method, f"icd9_{c}", r, leak, r / leak, 3])
    return _frame(rows)


def test_specificity_ratio_separates_methods_that_leak_identically() -> None:
    """Characterises the coupling the control has to remove."""
    df = _synthetic()
    s = summarise_methods(df)

    strong = s.loc[s["method"] == "strong", "median_specificity_ratio"].iloc[0]
    weak = s.loc[s["method"] == "weak", "median_specificity_ratio"].iloc[0]
    assert strong > 2.5 * weak, "the ratio is coupled to on-target strength by construction"


def test_leakage_comparison_does_not_separate_them() -> None:
    df = _synthetic()
    s = summarise_methods(df)

    a = s.loc[s["method"] == "strong", "median_mean_abs_off_r"].iloc[0]
    b = s.loc[s["method"] == "weak", "median_mean_abs_off_r"].iloc[0]
    assert abs(a - b) < 0.005, "leakage is not mechanically coupled and must not separate"


def test_coupling_control_reports_no_advantage_when_leakage_is_equal() -> None:
    """Residual leakage after regressing on on-target strength: both ~0."""
    df = _synthetic()
    ctrl = coupling_control(df).set_index("method")

    assert abs(ctrl.loc["strong", "median_leakage_residual"]) < 0.01
    assert abs(ctrl.loc["weak", "median_leakage_residual"]) < 0.01


def test_coupling_control_detects_a_genuine_leakage_advantage() -> None:
    df = _synthetic()
    # Halve one method's leakage; the control must now favour it (negative residual).
    df.loc[df["method"] == "strong", "mean_abs_off_r"] *= 0.5
    ctrl = coupling_control(df).set_index("method")

    assert ctrl.loc["strong", "median_leakage_residual"] < ctrl.loc["weak", "median_leakage_residual"]


def test_matched_r_comparison_restricts_to_comparable_codes() -> None:
    rows = [
        ["a", "icd9_1", 0.50, 0.02, 25.0, 1],
        ["b", "icd9_1", 0.49, 0.04, 12.3, 4],   # matched (|dr| = 0.01)
        ["a", "icd9_2", 0.60, 0.02, 30.0, 1],
        ["b", "icd9_2", 0.20, 0.04, 5.0, 4],    # NOT matched (|dr| = 0.40)
    ]
    out = matched_r_comparison(_frame(rows), "a", "b", tol=0.05)

    assert out["n_codes_matched"] == 1
    assert out["codes"] == ["icd9_1"]


def test_matched_r_comparison_reports_both_axes_and_a_paired_test() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for c in range(15):
        r = 0.4 + 0.01 * rng.standard_normal()
        rows.append(["a", f"icd9_{c}", r, 0.02, r / 0.02, 1])
        rows.append(["b", f"icd9_{c}", r, 0.05, r / 0.05, 5])
    out = matched_r_comparison(_frame(rows), "a", "b", tol=0.05)

    assert out["n_codes_matched"] == 15
    assert out["median_leakage_a"] < out["median_leakage_b"]
    assert out["a_lower_leakage_on_n_codes"] == 15
    assert out["leakage_wilcoxon_p"] < 0.01


def test_matched_r_comparison_handles_no_overlap() -> None:
    rows = [
        ["a", "icd9_1", 0.90, 0.02, 45.0, 1],
        ["b", "icd9_1", 0.10, 0.05, 2.0, 9],
    ]
    out = matched_r_comparison(_frame(rows), "a", "b", tol=0.05)

    assert out["n_codes_matched"] == 0
    assert out["leakage_wilcoxon_p"] is None


def test_matched_r_comparison_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="not present"):
        matched_r_comparison(_synthetic(), "strong", "nonexistent", tol=0.05)


def test_summarise_methods_carries_counts_and_is_sorted_by_leakage() -> None:
    """Sorted by LEAKAGE, not by specificity ratio.

    The setup is the realistic one -- "strong" has BOTH the higher on-target r
    and the lower leakage, which is how the trained SAEs actually sit. Under
    that configuration the two ascending orders genuinely disagree: by leakage
    "strong" comes first, by specificity ratio it comes last. Ordering a method
    table by the coupled quantity is the mistake this module exists to prevent,
    so the sort key is pinned to the one that disagrees.
    """
    df = _synthetic()
    df.loc[df["method"] == "strong", "mean_abs_off_r"] *= 0.5
    df["specificity_ratio"] = df["abs_on_target_r"] / df["mean_abs_off_r"]
    s = summarise_methods(df)

    assert list(s.columns[:2]) == ["method", "n_codes"]
    assert (s["n_codes"] == 20).all()
    assert s["median_mean_abs_off_r"].is_monotonic_increasing
    assert list(s["method"]) == ["strong", "weak"]
    # The two orders must genuinely disagree, or the assertion above is vacuous.
    assert list(s.sort_values("median_specificity_ratio")["method"]) == ["weak", "strong"]
