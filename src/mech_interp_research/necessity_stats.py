"""Statistics for the four-arm concordance validation.

Two jobs. First, enforce the selection/audit split: a feature chosen as the
argmax over 18,432 candidates must be scored on notes that did not participate
in choosing it, or the reported correlation is upward-biased and the bias grows
with the candidate count. Second, provide significance machinery that does not
assume the 46 ICD codes are independent — they co-occur heavily, so the paired
code-level permutation test replaces an unpaired two-proportion test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HELD_OUT_SHARD_START = 281


def split_by_shard(
    note_meta: pd.DataFrame,
    held_out_shard_start: int = DEFAULT_HELD_OUT_SHARD_START,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean row masks (selection, audit) over ``note_meta``.

    Selection = shards < held_out_shard_start. Audit = shards >= it.
    """
    if "shard" not in note_meta.columns:
        raise KeyError("note_meta must carry a 'shard' column to split on")

    shards = note_meta["shard"].to_numpy()
    selection = shards < held_out_shard_start
    audit = ~selection
    logger.info(
        "Split: %d selection notes (shards < %d), %d audit notes",
        int(selection.sum()),
        held_out_shard_start,
        int(audit.sum()),
    )
    return selection, audit


def select_feature_per_code(r_pb_selection: np.ndarray) -> list[int]:
    """Index of the strongest-|r| feature for each code, on the SELECTION set."""
    return [int(i) for i in np.argmax(np.abs(np.asarray(r_pb_selection)), axis=0)]


def selection_bias_delta(
    r_selection: np.ndarray,
    r_audit: np.ndarray,
    feature_ids: list[int],
) -> pd.DataFrame:
    """Per-code |r| on the selection set, on the audit set, and their difference.

    The difference is a direct measurement of best-of-k selection bias.
    """
    rows = []
    for code_idx, fid in enumerate(feature_ids):
        r_sel = abs(float(r_selection[fid, code_idx]))
        r_aud = abs(float(r_audit[fid, code_idx]))
        rows.append(
            {
                "code_idx": code_idx,
                "feature_id": int(fid),
                "r_selection": r_sel,
                "r_audit": r_aud,
                "delta": r_sel - r_aud,
            }
        )
    return pd.DataFrame(rows)


def paired_code_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_draws: int = 10_000,
    seed: int = 42,
) -> dict:
    """Two-sided paired permutation test on per-code outcomes.

    Pairing on code absorbs most of the dependence induced by ICD comorbidity,
    which an unpaired two-proportion test would ignore. Under the null the arm
    labels are exchangeable within each code, so each draw flips a random subset
    of the per-code differences.

    Args:
        a, b:    [n_codes] per-code outcomes (0/1 verdicts or continuous scores).
        n_draws: Permutation draws.
        seed:    RNG seed.

    Returns:
        dict with observed_diff, p_value, n_codes, n_draws.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same length, got {a.shape} and {b.shape}")

    diffs = a - b
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_draws, diffs.size))
    null = (signs * diffs).mean(axis=1)

    # +1 correction keeps the p-value strictly positive.
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_draws + 1))
    return {
        "observed_diff": observed,
        "p_value": p_value,
        "n_codes": int(diffs.size),
        "n_draws": int(n_draws),
    }


def derived_g4_threshold(a_rate: float, b1_rate: float) -> float:
    """Gate G4: retain more than half the judge's demonstrated dynamic range.

    B1 is the judge's YES rate when the answer is lexically unmistakable — its
    demonstrated ceiling. A is its rate on a direction with no clean lexical
    signature — its demonstrated floor. B2, diluted until it has no clean lexical
    signature either, must land above the midpoint of that interval.
    """
    if b1_rate <= a_rate:
        raise ValueError(
            f"dynamic range is non-positive: B1={b1_rate:.3f} <= A={a_rate:.3f}. "
            "The judge showed no measurable sensitivity, so G4 is undefined."
        )
    return (float(a_rate) + float(b1_rate)) / 2.0
