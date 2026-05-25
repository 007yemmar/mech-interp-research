"""Shuffled-explanation control for the auto-interp scorers.

Re-scores each feature's real contexts paired with a deliberately wrong
explanation to establish the Fuzzing/Detection scorer null baseline
(Paulo et al. 2024, arXiv 2410.13928: shuffled-explanation baseline ~0.51).
Reuses ``extracted_contexts.json`` + per-feature explanations from a completed
``run_auto_interp`` run; only the scoring API calls are new.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mech_interp_research.auto_interp import (
    _write_json,
    resolve_token_text,
    score_explanation_against_contexts,
)

logger = logging.getLogger(__name__)

__all__ = [
    "permute_global",
    "permute_within_tier",
    "load_existing_run",
    "aggregate_control_results",
    "compute_real_crosscheck",
    "run_shuffled_control",
]


def permute_global(feature_ids: list[int], seed: int = 42) -> dict[int, int]:
    """Map each feature id to a *different* feature id (a derangement).

    Returns ``{feature_id: other_feature_id}`` with no fixed points. Requires at
    least 2 ids. Deterministic for a fixed seed.
    """
    n = len(feature_ids)
    if n < 2:
        raise ValueError(f"need at least 2 features for a derangement, got {n}")
    ids = list(feature_ids)
    rng = np.random.default_rng(seed)
    while True:
        perm = [ids[i] for i in rng.permutation(n)]
        if all(a != b for a, b in zip(ids, perm, strict=True)):
            return dict(zip(ids, perm, strict=True))


def permute_within_tier(feature_to_tier: dict[int, str], seed: int = 42) -> dict[int, int]:
    """Derange feature ids within each tier independently.

    Tiers with fewer than 2 features are skipped (a WARNING is logged and those
    features get no mapping). Deterministic for a fixed seed.
    """
    by_tier: dict[str, list[int]] = {}
    for fid, tier in feature_to_tier.items():
        by_tier.setdefault(tier, []).append(fid)

    mapping: dict[int, int] = {}
    for tier_idx, (tier, ids) in enumerate(sorted(by_tier.items())):
        if len(ids) < 2:
            logger.warning(
                "Tier %s has %d eligible feature(s); skipping within-tier control",
                tier,
                len(ids),
            )
            continue
        mapping.update(permute_global(sorted(ids), seed=seed + tier_idx + 1))
    return mapping


def load_existing_run(
    auto_interp_dir: str | Path, model: str
) -> tuple[dict[int, dict], list[dict]]:
    """Load ``extracted_contexts.json`` + per-feature JSONs from a completed run.

    Returns ``(contexts_by_fid, feature_rows)``. Raises ``FileNotFoundError`` if
    either input is missing.
    """
    auto_interp_dir = Path(auto_interp_dir)
    contexts_path = auto_interp_dir / "extracted_contexts.json"
    if not contexts_path.is_file():
        raise FileNotFoundError(f"extracted_contexts.json not found at {contexts_path}")
    with open(contexts_path) as f:
        raw = json.load(f)
    contexts_by_fid = {int(k): v for k, v in raw.items()}

    model_dir = auto_interp_dir / "per_feature" / model.replace("/", "_")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"per-feature dir not found at {model_dir}")
    feature_rows: list[dict] = []
    for jf in sorted(model_dir.glob("feature_*.json")):
        with open(jf) as f:
            feature_rows.append(json.load(f))
    if not feature_rows:
        raise FileNotFoundError(f"no feature_*.json under {model_dir}")
    return contexts_by_fid, feature_rows


def _is_eligible(row: dict, contexts_by_fid: dict[int, dict]) -> bool:
    """A feature is eligible if it has a real explanation and >=1 pos context.

    Fallback explanations produced by ``_dead_result`` start with ``"Feature "``.
    """
    expl = row.get("explanation") or ""
    if not expl or expl.startswith("Feature "):
        return False
    ctx = contexts_by_fid.get(row["feature_idx"])
    if not ctx:
        return False
    return len(ctx.get("pos_contexts", [])) > 0


def _aggregate_block(rows: list[dict], real_key: str, shuf_key: str) -> dict:
    """Paired summary over rows where both real and shuffled scores are present."""
    pairs = [
        (r[real_key], r[shuf_key])
        for r in rows
        if r.get(real_key) is not None and r.get(shuf_key) is not None
    ]
    n = len(pairs)
    if n == 0:
        return {
            "mean_real": None,
            "mean_shuffled": None,
            "delta": None,
            "n": 0,
            "ci95_shuffled": None,
            "wilcoxon_p": None,
            "median_delta_real_minus_shuffled": None,
        }
    real = np.array([p[0] for p in pairs], dtype=float)
    shuf = np.array([p[1] for p in pairs], dtype=float)
    deltas = real - shuf

    if n > 1:
        se = float(shuf.std(ddof=1)) / np.sqrt(n)
    else:
        se = 0.0
    ci = [round(float(shuf.mean() - 1.96 * se), 4), round(float(shuf.mean() + 1.96 * se), 4)]

    if n < 10 or np.all(deltas == 0):
        p_val: float | None = None
    else:
        _, p = wilcoxon(deltas, alternative="two-sided")
        p_val = round(float(p), 6)

    return {
        "mean_real": round(float(real.mean()), 4),
        "mean_shuffled": round(float(shuf.mean()), 4),
        "delta": round(float(real.mean() - shuf.mean()), 4),
        "n": n,
        "ci95_shuffled": ci,
        "wilcoxon_p": p_val,
        "median_delta_real_minus_shuffled": round(float(np.median(deltas)), 4),
    }


def aggregate_control_results(
    per_feature_rows: list[dict],
    schemes: list[str],
    scorers: list[str],
    chance_value: float = 0.51,
) -> dict:
    """Build the summary dict (per scorer x scheme x tier(+overall))."""
    tiers = sorted({r["tier"] for r in per_feature_rows})
    results: dict[str, Any] = {}
    for scorer in scorers:
        results[scorer] = {}
        real_key = f"{scorer}_real"
        for scheme in schemes:
            shuf_key = f"{scorer}_shuf_{scheme}"
            results[scorer][scheme] = {
                "overall": _aggregate_block(per_feature_rows, real_key, shuf_key),
                "by_tier": {
                    tier: _aggregate_block(
                        [r for r in per_feature_rows if r["tier"] == tier], real_key, shuf_key
                    )
                    for tier in tiers
                },
            }
    return {
        "chance_reference": {
            "value": chance_value,
            "source": "Paulo et al. 2024 (arXiv 2410.13928)",
        },
        "schemes": schemes,
        "scorers": scorers,
        "results": results,
    }


def compute_real_crosscheck(
    per_feature_rows: list[dict],
    published_summary: dict,
    scorers: list[str],
) -> dict:
    """Cross-check the control's stored real scores against the real run's
    published means, globally and per tier.

    Closes the denominator caveat: the control's paired comparison reads each
    feature's real score from the prior run, but its aggregate ``mean_real`` is
    over the matched subset, which may differ from the real run's published
    means. This reports both side by side so any skew is visible.

    ``published_summary`` is the real run's ``scorer_summary.json`` of the form
    ``{"global": {...}, "<tier>": {...}}`` with ``mean_<scorer>`` /
    ``n_valid_<scorer>`` keys; pass ``{}`` if it is unavailable.
    """
    tiers = sorted({r["tier"] for r in per_feature_rows})

    def _block(rows: list[dict], pub: dict, scorer: str) -> dict:
        vals = [r[f"{scorer}_real"] for r in rows if r.get(f"{scorer}_real") is not None]
        ctrl_mean = round(float(np.mean(vals)), 4) if vals else None
        pub_mean = pub.get(f"mean_{scorer}")
        pub_mean = round(float(pub_mean), 4) if pub_mean is not None else None
        delta = (
            round(ctrl_mean - pub_mean, 4)
            if ctrl_mean is not None and pub_mean is not None
            else None
        )
        return {
            "control_mean_real": ctrl_mean,
            "control_n": len(vals),
            "published_mean": pub_mean,
            "published_n": pub.get(f"n_valid_{scorer}"),
            "delta_control_minus_published": delta,
        }

    out: dict[str, Any] = {}
    for scorer in scorers:
        out[scorer] = {
            "overall": _block(per_feature_rows, published_summary.get("global", {}), scorer),
            "by_tier": {
                t: _block(
                    [r for r in per_feature_rows if r["tier"] == t],
                    published_summary.get(t, {}),
                    scorer,
                )
                for t in tiers
            },
        }
    return out


def _score_one_feature(
    fid: int,
    *,
    contexts_by_fid: dict[int, dict],
    expl_by_fid: dict[int, str],
    tier_by_fid: dict[int, str],
    real_by_fid: dict[int, dict],
    perm_maps: dict[str, dict[int, int]],
    scorers: list[str],
    note_texts: dict[int, str],
    tokenizer,
    context_window: int,
    n_contexts_train: int,
    n_pos_test: int,
    client,
    model: str,
) -> dict:
    """Score one feature's held-out contexts against its permuted (wrong)
    explanation(s).

    Operates only on this feature's own (disjoint) context dicts — note
    ``resolve_token_text`` mutates them in place — and reads only this feature's
    entries from the shared inputs, so it is safe to call concurrently across
    features. Deterministic: the fuzzing
    distractor RNG is seeded by ``fid`` and the permutation maps are fixed, so
    the result is independent of call order.
    """
    ctx = contexts_by_fid[fid]
    pos = list(ctx.get("pos_contexts", []))
    neg = list(ctx.get("neg_contexts", []))
    if tokenizer is not None:
        resolve_token_text(pos + neg, note_texts, tokenizer, context_window=context_window)
    test_pos = pos[n_contexts_train : n_contexts_train + n_pos_test]
    test_neg = neg[:n_pos_test]

    row: dict[str, Any] = {
        "feature_idx": fid,
        "tier": tier_by_fid[fid],
        "real_expl_feature": fid,
    }
    for scorer in scorers:
        row[f"{scorer}_real"] = real_by_fid[fid].get(f"{scorer}_score")

    for scheme, pmap in perm_maps.items():
        wrong_fid = pmap.get(fid)
        if wrong_fid is None:  # within-tier skipped this feature's tier
            row[f"shuf_{scheme}_feature"] = None
            for scorer in scorers:
                row[f"{scorer}_shuf_{scheme}"] = None
            continue
        fuzz, det, perr = score_explanation_against_contexts(
            client=client,
            explanation=expl_by_fid[wrong_fid],
            test_pos=test_pos,
            test_neg=test_neg,
            note_texts=note_texts,
            tokenizer=tokenizer,
            model=model,
            scorers=scorers,
            context_window=context_window,
            owner_feature_id=fid,
        )
        scores = {"fuzzing": fuzz, "detection": det}
        row[f"shuf_{scheme}_feature"] = wrong_fid
        row[f"parsing_errors_{scheme}"] = perr
        for scorer in scorers:
            row[f"{scorer}_shuf_{scheme}"] = scores[scorer]
    return row


def run_shuffled_control(
    auto_interp_dir: str | Path,
    output_dir: str | Path | None = None,
    model: str = "claude-sonnet-4-6",
    schemes: list[str] | None = None,
    scorers: list[str] | None = None,
    n_contexts_train: int = 20,
    n_contexts_test: int = 10,
    context_window: int = 30,
    seed: int = 42,
    chance_value: float = 0.51,
    max_workers: int = 8,
    client_max_retries: int = 8,
    _client=None,
    _note_texts: dict[int, str] | None = None,
    _tokenizer=None,
    _commit_volume=None,
) -> dict:
    """Re-score each eligible feature's contexts with a wrong explanation.

    Reuses ``extracted_contexts.json`` + per-feature explanations under
    ``auto_interp_dir``. Writes ``shuffled_control/`` with a summary JSON, a
    per-feature CSV, and resume checkpoints. Returns the summary dict.
    """
    if schemes is None:
        schemes = ["global", "within_tier"]
    if scorers is None:
        scorers = ["fuzzing", "detection"]

    auto_interp_dir = Path(auto_interp_dir)
    output_dir = Path(output_dir) if output_dir else auto_interp_dir / "shuffled_control"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "per_feature" / model.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    contexts_by_fid, feature_rows = load_existing_run(auto_interp_dir, model)
    eligible = [r for r in feature_rows if _is_eligible(r, contexts_by_fid)]
    logger.info("Eligible features: %d / %d", len(eligible), len(feature_rows))

    expl_by_fid = {r["feature_idx"]: r["explanation"] for r in eligible}
    tier_by_fid = {r["feature_idx"]: r["tier"] for r in eligible}
    real_by_fid = {r["feature_idx"]: r for r in eligible}
    eligible_ids = [r["feature_idx"] for r in eligible]

    perm_maps: dict[str, dict[int, int]] = {}
    if "global" in schemes:
        # Sort for reproducibility: derangement is order-sensitive, so a stable
        # numeric order makes the mapping reproducible independent of file-glob order.
        perm_maps["global"] = permute_global(sorted(eligible_ids), seed=seed)
    if "within_tier" in schemes:
        perm_maps["within_tier"] = permute_within_tier(tier_by_fid, seed=seed)

    client = _client
    if client is None:
        import anthropic

        # SDK-level retries give exponential backoff for 429 rate limits and
        # transient 5xx errors — the primary defense under concurrency.
        client = anthropic.Anthropic(max_retries=client_max_retries)
    note_texts = _note_texts or {}
    tokenizer = _tokenizer

    n_pos_test = n_contexts_test // 2

    # Resume: load existing per-feature checkpoints; only re-score the rest.
    # A corrupt/truncated checkpoint is discarded and re-scored.
    per_feature_rows: list[dict] = []
    todo: list[int] = []
    for fid in eligible_ids:
        ckpt = ckpt_dir / f"feature_{fid}.json"
        if ckpt.exists():
            try:
                with open(ckpt) as f:
                    per_feature_rows.append(json.load(f))
                continue
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupt checkpoint %s (truncated write?); discarding and re-scoring", ckpt
                )
                ckpt.unlink()
        todo.append(fid)

    logger.info(
        "Scoring %d features (%d resumed from checkpoint) with max_workers=%d",
        len(todo),
        len(per_feature_rows),
        max_workers,
    )

    # Concurrency note: workers only READ their own feature's data from the
    # shared inputs and write nothing shared. Checkpoint writes and volume
    # commits happen only on the main thread (below), so there are no
    # file-write or commit races. Results are independent of completion order.
    n_errors = 0
    if todo:
        worker = partial(
            _score_one_feature,
            contexts_by_fid=contexts_by_fid,
            expl_by_fid=expl_by_fid,
            tier_by_fid=tier_by_fid,
            real_by_fid=real_by_fid,
            perm_maps=perm_maps,
            scorers=scorers,
            note_texts=note_texts,
            tokenizer=tokenizer,
            context_window=context_window,
            n_contexts_train=n_contexts_train,
            n_pos_test=n_pos_test,
            client=client,
            model=model,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, fid): fid for fid in todo}
            for done_count, fut in enumerate(as_completed(futures), start=1):
                fid = futures[fut]
                try:
                    row = fut.result()
                except Exception:
                    # A feature still failing after the SDK's retries is logged
                    # and skipped (no checkpoint written), so the run never
                    # crashes and the feature is retried on the next resume.
                    logger.exception(
                        "Feature %s failed after retries; skipping (will retry on resume)", fid
                    )
                    n_errors += 1
                    continue
                _write_json(row, ckpt_dir / f"feature_{fid}.json")
                per_feature_rows.append(row)
                if _commit_volume is not None and done_count % 50 == 0:
                    _commit_volume()

    if _commit_volume is not None:
        _commit_volume()

    summary = aggregate_control_results(
        per_feature_rows, list(perm_maps.keys()), scorers, chance_value
    )
    summary["model"] = model
    summary["n_eligible"] = len(eligible_ids)
    summary["n_errors"] = n_errors
    summary["parsing_errors"] = {
        scheme: int(sum(r.get(f"parsing_errors_{scheme}", 0) or 0 for r in per_feature_rows))
        for scheme in perm_maps
    }
    # Cross-check the matched-subset real baseline against the real run's
    # published scorer means (scorer_summary.json), so any denominator skew
    # between this control and the published numbers is visible.
    pub_path = auto_interp_dir / "scorer_summary.json"
    published_summary: dict = {}
    if pub_path.is_file():
        with open(pub_path) as f:
            published_summary = json.load(f)
    summary["real_score_crosscheck"] = compute_real_crosscheck(
        per_feature_rows, published_summary, scorers
    )
    _write_json(summary, output_dir / "shuffled_control_summary.json")
    pd.DataFrame(per_feature_rows).to_csv(
        output_dir / "shuffled_control_per_feature.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("SHUFFLED-EXPLANATION CONTROL — %d eligible features", len(eligible_ids))
    for scorer in scorers:
        for scheme in perm_maps:
            blk = summary["results"][scorer][scheme]["overall"]
            logger.info(
                "  %s/%s: real=%s shuffled=%s (delta=%s, p=%s, n=%s)",
                scorer,
                scheme,
                blk["mean_real"],
                blk["mean_shuffled"],
                blk["delta"],
                blk["wilcoxon_p"],
                blk["n"],
            )
    if n_errors:
        logger.warning(
            "  %d/%d features FAILED after retries (excluded from results)",
            n_errors,
            len(eligible_ids),
        )
    logger.info("=" * 60)
    return summary
