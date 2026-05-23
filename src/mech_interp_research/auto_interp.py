"""Auto-interpretability for clinical SAE features.

Generates LLM explanations for SAE features, scores them with Detection
and Fuzzing protocols (Paulo et al. 2024), categorizes them, and validates
grounded features against ICD-9 labels via concordance scoring.

All LLM calls use the Anthropic SDK directly — no external auto-interp
library dependency.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors

from mech_interp_research.icd_eval import (
    JumpReLUSAE,
    load_metadata,
    load_saved_correlations,
    reassemble_note_vectors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Feature selection
# ---------------------------------------------------------------------------


def select_features(
    r_pb: np.ndarray,
    p_adjusted: np.ndarray,
    significant: np.ndarray,
    code_names: list[str],
    note_vectors: np.ndarray,
    n_strong_grounded: int = 280,
    n_weak_grounded: int = 100,
    n_non_grounded: int = 1000,
    n_dead: int = 100,
    strong_threshold: float = 0.4,
    weak_lo: float = 0.1,
    weak_hi: float = 0.3,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Select features in four tiers for auto-interp.

    Args:
        r_pb: [d_sae, n_codes] point-biserial correlations.
        p_adjusted: [d_sae, n_codes] BH-adjusted p-values.
        significant: [d_sae, n_codes] boolean significance mask.
        code_names: List of ICD code column names.
        note_vectors: [n_notes, d_sae] note-level SAE activations
            (for dead-feature detection via mean activation).
        n_strong_grounded: Max features with max|r| > strong_threshold.
        n_weak_grounded: Random sample from weak_lo < max|r| <= weak_hi.
        n_non_grounded: Random sample from features with zero significant
            correlations.
        n_dead: Features with lowest mean activation.
        strong_threshold: Threshold for strong grounding (default 0.4).
        weak_lo: Lower bound for weak grounding (default 0.1).
        weak_hi: Upper bound for weak grounding (default 0.3).
        seed: Random seed for reproducible sampling.

    Returns:
        Dict with keys 'strong_grounded', 'weak_grounded', 'non_grounded',
        'dead', each mapping to a list of feature indices.
    """
    rng = np.random.default_rng(seed)
    d_sae = r_pb.shape[0]
    max_abs_r = np.abs(r_pb).max(axis=1)
    any_significant = significant.any(axis=1)

    # Tier 1: strong grounded
    strong_mask = max_abs_r > strong_threshold
    strong_ids = sorted(int(i) for i in np.where(strong_mask)[0])
    if len(strong_ids) > n_strong_grounded:
        order = np.argsort(max_abs_r[strong_ids])[::-1]
        strong_ids = [strong_ids[i] for i in order[:n_strong_grounded]]

    used = set(strong_ids)

    # Tier 2: weak grounded
    weak_mask = (max_abs_r > weak_lo) & (max_abs_r <= weak_hi) & any_significant
    weak_candidates = [int(i) for i in np.where(weak_mask)[0] if i not in used]
    n_weak = min(n_weak_grounded, len(weak_candidates))
    if weak_candidates and n_weak > 0:
        weak_ids = sorted(rng.choice(weak_candidates, size=n_weak, replace=False).tolist())
    else:
        weak_ids = []
    used |= set(weak_ids)

    # Tier 3: non-grounded (zero BH-significant correlations)
    non_grounded_mask = ~any_significant
    non_grounded_candidates = [int(i) for i in np.where(non_grounded_mask)[0] if i not in used]
    n_ng = min(n_non_grounded, len(non_grounded_candidates))
    if non_grounded_candidates and n_ng > 0:
        non_grounded_ids = sorted(
            rng.choice(non_grounded_candidates, size=n_ng, replace=False).tolist()
        )
    else:
        non_grounded_ids = []
    used |= set(non_grounded_ids)

    # Tier 4: dead (lowest mean activation, excluding already-selected)
    mean_acts = note_vectors.mean(axis=0)
    remaining = [i for i in range(d_sae) if i not in used]
    remaining_sorted = sorted(remaining, key=lambda i: mean_acts[i])
    dead_ids = remaining_sorted[: min(n_dead, len(remaining_sorted))]

    logger.info(
        f"Feature selection: strong={len(strong_ids)}, weak={len(weak_ids)}, "
        f"non_grounded={len(non_grounded_ids)}, dead={len(dead_ids)}"
    )

    return {
        "strong_grounded": strong_ids,
        "weak_grounded": weak_ids,
        "non_grounded": non_grounded_ids,
        "dead": dead_ids,
    }


# ---------------------------------------------------------------------------
# 2. Context extraction
# ---------------------------------------------------------------------------


def extract_contexts_for_feature(
    sae: JumpReLUSAE,
    feature_idx: int,
    activations_dir: Path,
    metadata: pd.DataFrame,
    n_pos: int = 20,
    n_neg: int = 10,
    context_window: int = 15,
    shard_filter: list[int] | None = None,
) -> dict:
    """Extract top-N activating and N non-activating contexts for one feature.

    Scans activation shards, encodes through SAE, and collects the
    highest-activation positions (positive contexts) and random
    zero-activation positions (negative contexts).

    Returns:
        {"pos_contexts": [...], "neg_contexts": [...]}
        Each context is a dict with keys:
            activation, position_in_shard, shard, note_idx, position_in_note
    """
    activations_dir = Path(activations_dir)

    if shard_filter is not None:
        meta = metadata[metadata["shard"].isin(shard_filter)].copy()
    else:
        meta = metadata.copy()

    shards = sorted(meta["shard"].unique())
    rng = np.random.default_rng(feature_idx)

    all_pos: list[dict] = []
    all_neg_candidates: list[dict] = []

    for shard_idx in shards:
        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            continue

        shard_data = load_safetensors(str(shard_path))
        act_key = next(iter(shard_data))
        acts = shard_data[act_key].astype(np.float32)

        encoded = sae.encode_chunked(acts)
        feature_acts = encoded[:, feature_idx]

        shard_meta = meta[meta["shard"] == shard_idx]
        note_boundaries = []
        for _, row in shard_meta.iterrows():
            note_boundaries.append(
                (int(row["row_start"]), int(row["row_end"]), int(row["note_idx"]))
            )
        note_boundaries.sort(key=lambda x: x[0])

        def _find_note(
            pos: int,
            note_boundaries: list[tuple[int, int, int]],
        ) -> tuple[int, int]:
            for rs, re_, nidx in note_boundaries:
                if rs <= pos < re_:
                    return nidx, pos - rs
            return -1, -1

        # Collect positive (activating) positions
        nonzero_positions = np.where(feature_acts > 0)[0]
        for pos in nonzero_positions:
            note_idx, pos_in_note = _find_note(int(pos))
            if note_idx == -1:
                continue
            all_pos.append(
                {
                    "activation": float(feature_acts[pos]),
                    "position_in_shard": int(pos),
                    "shard": int(shard_idx),
                    "note_idx": note_idx,
                    "position_in_note": pos_in_note,
                }
            )

        # Collect negative (zero-activation) positions
        zero_positions = np.where(feature_acts == 0)[0]
        if len(zero_positions) > 0:
            sample_size = min(n_neg * 2, len(zero_positions))
            sampled = rng.choice(zero_positions, size=sample_size, replace=False)
            for pos in sampled:
                note_idx, pos_in_note = _find_note(int(pos))
                if note_idx == -1:
                    continue
                all_neg_candidates.append(
                    {
                        "activation": 0.0,
                        "position_in_shard": int(pos),
                        "shard": int(shard_idx),
                        "note_idx": note_idx,
                        "position_in_note": pos_in_note,
                    }
                )

    # Sort positives by activation descending, take top-N
    all_pos.sort(key=lambda x: x["activation"], reverse=True)
    pos_contexts = all_pos[:n_pos]

    # Random sample from negatives
    if len(all_neg_candidates) > n_neg:
        neg_indices = rng.choice(len(all_neg_candidates), size=n_neg, replace=False)
        neg_contexts = [all_neg_candidates[i] for i in neg_indices]
    else:
        neg_contexts = all_neg_candidates[:n_neg]

    return {"pos_contexts": pos_contexts, "neg_contexts": neg_contexts}


# ---------------------------------------------------------------------------
# 3. Explanation generation
# ---------------------------------------------------------------------------

EXPLAIN_PROMPT = """\
Below are the top {n} text contexts where a sparse autoencoder feature \
activates most strongly in clinical discharge summaries. The trigger \
token in each context is marked with **asterisks**.

{contexts}

What pattern causes this feature to activate? Describe the specific \
concept, linguistic pattern, or structural element that these contexts \
share. Be as specific as possible — name clinical concepts, drugs, \
or procedures if applicable. One to two sentences."""


def explain_feature(
    client,
    pos_contexts: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 256,
) -> str:
    """Generate an explanation for a feature given its top activating contexts.

    Args:
        client: Anthropic client instance.
        pos_contexts: List of dicts with 'context_str' and 'token_str' keys.
        model: Anthropic model ID.
        max_tokens: Max response tokens.

    Returns:
        Explanation string, or "" if no contexts provided.
    """
    if not pos_contexts:
        return ""

    context_lines = []
    for i, ctx in enumerate(pos_contexts, 1):
        context_lines.append(f"{i}. {ctx['context_str']}")

    prompt = EXPLAIN_PROMPT.format(
        n=len(pos_contexts),
        contexts="\n".join(context_lines),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------

FUZZING_PROMPT = """\
A sparse autoencoder feature has been described as:
"{explanation}"

For each of the following tokens in context, predict whether \
this token would activate the feature (YES or NO).

{contexts}

Respond with one line per context: the context number and YES or NO."""

DETECTION_PROMPT = """\
A sparse autoencoder feature has been described as:
"{explanation}"

Below are {n} text contexts from clinical discharge summaries. \
Some activated this feature and some did not.

{contexts}

For each context, predict whether it activated the feature (YES or NO). \
Respond with one line per context: the context number and YES or NO."""


def _parse_yes_no_responses(text: str, n_expected: int) -> list[bool | None]:
    """Parse numbered YES/NO lines from LLM response.

    Returns a list of booleans (True=YES, False=NO, None=unparseable)
    aligned by line number (1-indexed in response, 0-indexed in output).
    """
    results: list[bool | None] = [None] * n_expected
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(\d+)\s*[.):]\s*(YES|NO)", line, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < n_expected:
                results[idx] = match.group(2).upper() == "YES"
    return results


def fuzzing_score(
    client,
    explanation: str,
    test_contexts: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 256,
) -> float:
    """Fuzzing scorer: can the LLM identify which tokens match the pattern?

    Args:
        client: Anthropic client.
        explanation: Feature explanation string.
        test_contexts: List of dicts with 'context_str' and 'is_activating'.
        model: Anthropic model ID.

    Returns:
        Accuracy (0.0-1.0), or NaN if response is unparseable.
    """
    context_lines = []
    for i, ctx in enumerate(test_contexts, 1):
        context_lines.append(f"{i}. {ctx['context_str']}")

    prompt = FUZZING_PROMPT.format(
        explanation=explanation,
        contexts="\n".join(context_lines),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    predictions = _parse_yes_no_responses(response.content[0].text, len(test_contexts))
    parsed = [
        (p, ctx["is_activating"])
        for p, ctx in zip(predictions, test_contexts, strict=False)
        if p is not None
    ]

    if not parsed:
        return float("nan")

    correct = sum(1 for pred, actual in parsed if pred == actual)
    return correct / len(parsed)


def detection_score(
    client,
    explanation: str,
    pos_contexts: list[dict],
    neg_contexts: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 256,
    _shuffle_seed: int | None = 42,
) -> float:
    """Detection scorer: can the LLM identify which contexts activated?

    Shuffles positive and negative contexts, asks the LLM to classify each.

    Args:
        client: Anthropic client.
        explanation: Feature explanation string.
        pos_contexts: Activating contexts (dicts with 'context_str').
        neg_contexts: Non-activating contexts.
        model: Anthropic model ID.
        _shuffle_seed: Seed for context shuffling. None = no shuffle (testing).

    Returns:
        Accuracy (0.0-1.0), or NaN if unparseable.
    """
    items = [(ctx, True) for ctx in pos_contexts] + [(ctx, False) for ctx in neg_contexts]

    if _shuffle_seed is not None:
        rng = np.random.default_rng(_shuffle_seed)
        indices = rng.permutation(len(items))
        items = [items[i] for i in indices]

    ground_truth = [is_pos for _, is_pos in items]

    context_lines = []
    for i, (ctx, _) in enumerate(items, 1):
        context_lines.append(f"{i}. {ctx['context_str']}")

    prompt = DETECTION_PROMPT.format(
        explanation=explanation,
        n=len(items),
        contexts="\n".join(context_lines),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    predictions = _parse_yes_no_responses(response.content[0].text, len(items))
    parsed = [(p, gt) for p, gt in zip(predictions, ground_truth, strict=False) if p is not None]

    if not parsed:
        return float("nan")

    correct = sum(1 for pred, actual in parsed if pred == actual)
    return correct / len(parsed)


# ---------------------------------------------------------------------------
# 5. Categorization
# ---------------------------------------------------------------------------

CATEGORIZE_PROMPT = """\
A sparse autoencoder feature has been described as:
"{explanation}"

Top activating tokens include: {top_tokens}

Categorize this feature into exactly one of:
- clinical_concept (a specific medical diagnosis, condition, or finding)
- clinical_vocabulary (drug names, dosages, units, lab values, procedures)
- general_language (POS, syntax, common words, generic patterns)
- structural_pattern (formatting, section headers, signatures, templates)
- noise (no coherent pattern)

Respond with just the category name."""

VALID_CATEGORIES = {
    "clinical_concept",
    "clinical_vocabulary",
    "general_language",
    "structural_pattern",
    "noise",
}


def categorize_feature(
    client,
    explanation: str,
    top_tokens: list[str],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 32,
) -> str:
    """Categorize a feature into one of 5 categories.

    Returns:
        Category string, or "unknown" if response doesn't match.
    """
    prompt = CATEGORIZE_PROMPT.format(
        explanation=explanation,
        top_tokens=", ".join(top_tokens[:10]),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower()
    if raw in VALID_CATEGORIES:
        return raw

    for cat in VALID_CATEGORIES:
        if cat in raw:
            return cat

    logger.warning(f"Unparseable categorization response: {raw!r}")
    return "unknown"


# ---------------------------------------------------------------------------
# 6. Concordance
# ---------------------------------------------------------------------------

CONCORDANCE_PROMPT = """\
A sparse autoencoder feature has been auto-interpreted as:
"{explanation}"

This feature also has the strongest statistical correlation (point-biserial \
r = {r_pb:.3f}) with the ICD-9 diagnosis code: {icd_code} ({icd_description}).

Does the auto-interp explanation describe the same concept as the ICD code?
Respond with:
- YES — the explanation clearly describes the ICD concept
- PARTIAL — the explanation overlaps but describes a related, broader, \
or narrower concept (e.g., a drug used to treat the condition, a symptom \
of the condition, or a broader category containing the condition)
- NO — the explanation describes something unrelated

Format: <verdict> | <one-sentence rationale>
Example: PARTIAL | The explanation describes warfarin dosing, which is a \
treatment for the correlated ICD code (atrial fibrillation), not the \
condition itself."""


def check_concordance(
    client,
    explanation: str,
    icd_code: str,
    icd_description: str,
    r_pb: float,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 256,
) -> tuple[str, str]:
    """Check concordance between auto-interp explanation and ICD code.

    Returns:
        (verdict, rationale) where verdict is YES/PARTIAL/NO/UNKNOWN.
    """
    prompt = CONCORDANCE_PROMPT.format(
        explanation=explanation,
        r_pb=r_pb,
        icd_code=icd_code,
        icd_description=icd_description,
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    match = re.match(r"(YES|PARTIAL|NO)\s*[|—–\-]\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        verdict = match.group(1).upper()
        rationale = match.group(2).strip()
        return verdict, rationale

    upper = raw.upper()
    for v in ("YES", "PARTIAL", "NO"):
        if upper.startswith(v):
            return v, raw[len(v) :].strip().lstrip("|—–- ").strip()

    logger.warning(f"Unparseable concordance response: {raw!r}")
    return "UNKNOWN", raw


# ---------------------------------------------------------------------------
# 7. Catalog assembly
# ---------------------------------------------------------------------------


def assemble_catalog(
    feature_results: list[dict],
    output_dir: Path,
) -> None:
    """Assemble feature_catalog.csv + summary JSONs from per-feature results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # feature_catalog.csv
    catalog_rows = []
    for r in feature_results:
        catalog_rows.append(
            {
                "feature_idx": r["feature_idx"],
                "tier": r["tier"],
                "explanation": r["explanation"],
                "fuzzing_score": r["fuzzing_score"],
                "detection_score": r["detection_score"],
                "category": r["category"],
                "top_tokens": "; ".join(r.get("top_tokens", [])),
                "model": r["model"],
            }
        )
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_df.to_csv(output_dir / "feature_catalog.csv", index=False)
    logger.info(f"Wrote feature_catalog.csv: {len(catalog_df)} features")

    # concordance_results.csv
    conc_rows = [r for r in feature_results if r.get("concordance_verdict") is not None]
    if conc_rows:
        conc_df = pd.DataFrame(
            [
                {
                    "feature_idx": r["feature_idx"],
                    "tier": r["tier"],
                    "explanation": r["explanation"],
                    "concordance_verdict": r["concordance_verdict"],
                    "concordance_rationale": r["concordance_rationale"],
                    "concordance_icd_code": r["concordance_icd_code"],
                    "concordance_r_pb": r["concordance_r_pb"],
                }
                for r in conc_rows
            ]
        )
        conc_df.to_csv(output_dir / "concordance_results.csv", index=False)
        logger.info(f"Wrote concordance_results.csv: {len(conc_df)} features")

    # categorization_summary.json
    categories = [r["category"] for r in feature_results]
    cat_counts: dict[str, int] = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    total = len(categories)
    cat_summary = {
        "counts": cat_counts,
        "fractions": {k: v / total for k, v in cat_counts.items()} if total > 0 else {},
        "total": total,
    }
    _write_json(cat_summary, output_dir / "categorization_summary.json")

    # concordance_summary.json
    if conc_rows:
        verdicts = [r["concordance_verdict"] for r in conc_rows]
        yes_count = sum(1 for v in verdicts if v == "YES")
        partial_count = sum(1 for v in verdicts if v == "PARTIAL")
        no_count = sum(1 for v in verdicts if v == "NO")
        unknown_count = sum(1 for v in verdicts if v == "UNKNOWN")
        n_conc = len(verdicts)
        conc_summary = {
            "yes_count": yes_count,
            "partial_count": partial_count,
            "no_count": no_count,
            "unknown_count": unknown_count,
            "total": n_conc,
            "concordance_rate": (yes_count + partial_count) / n_conc if n_conc > 0 else 0,
            "exact_match_rate": yes_count / n_conc if n_conc > 0 else 0,
        }
        _write_json(conc_summary, output_dir / "concordance_summary.json")

    # scorer_summary.json
    fuzz_scores = [
        r["fuzzing_score"] for r in feature_results if r.get("fuzzing_score") is not None
    ]
    det_scores = [
        r["detection_score"] for r in feature_results if r.get("detection_score") is not None
    ]
    scorer_summary = {
        "mean_fuzzing": float(np.mean(fuzz_scores)) if fuzz_scores else None,
        "median_fuzzing": float(np.median(fuzz_scores)) if fuzz_scores else None,
        "std_fuzzing": float(np.std(fuzz_scores)) if fuzz_scores else None,
        "mean_detection": float(np.mean(det_scores)) if det_scores else None,
        "median_detection": float(np.median(det_scores)) if det_scores else None,
        "std_detection": float(np.std(det_scores)) if det_scores else None,
        "n_features": len(feature_results),
        "n_valid_fuzzing": len(fuzz_scores),
        "n_valid_detection": len(det_scores),
    }
    _write_json(scorer_summary, output_dir / "scorer_summary.json")


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote {path.name}")


# ---------------------------------------------------------------------------
# 8. Orchestrator
# ---------------------------------------------------------------------------


def _get_strongest_icd(
    feature_idx: int,
    r_pb: np.ndarray,
    code_names: list[str],
) -> tuple[str, float]:
    """Return (code_name, r_pb) for the feature's strongest ICD correlation."""
    row = r_pb[feature_idx]
    best_idx = int(np.argmax(np.abs(row)))
    return code_names[best_idx], float(row[best_idx])


def _load_code_descriptions(path: str | Path | None) -> dict[str, str]:
    """Load ICD code descriptions CSV if available."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if "code" in df.columns and "description" in df.columns:
        return dict(zip(df["code"].astype(str), df["description"].astype(str), strict=False))
    return {}


def run_auto_interp(
    sae_checkpoint: str | None,
    activations_dir: str,
    icd_eval_dir: str,
    output_dir: str,
    n_strong_grounded: int = 280,
    n_weak_grounded: int = 100,
    n_non_grounded: int = 1000,
    n_dead: int = 100,
    explainer_model: str = "claude-sonnet-4-20250514",
    comparison_model: str | None = None,
    concordance_model: str = "claude-sonnet-4-20250514",
    categorization_model: str = "claude-sonnet-4-20250514",
    n_contexts_train: int = 20,
    n_contexts_test: int = 10,
    context_window: int = 15,
    scorers: list[str] | None = None,
    concordance_thresholds: list[float] | None = None,
    random_seed: int = 42,
    icd_descriptions_path: str | None = None,
    checkpoint_dir: str | None = None,
    _client=None,
    _sae: JumpReLUSAE | None = None,
    **_kwargs,
) -> dict:
    """Run the full auto-interp pipeline.

    Args:
        sae_checkpoint: Path to SAE checkpoint dir (ignored if _sae provided).
        activations_dir: Path to centered activation shards.
        icd_eval_dir: Path to ICD eval output (correlation_matrices.npz, etc.).
        output_dir: Where to write outputs.
        _client: Injected Anthropic client (for testing).
        _sae: Injected SAE (for testing).
        **_kwargs: Ignored (allows passing extra config keys).

    Returns:
        Run summary dict.
    """
    start_time = datetime.now(UTC)
    if scorers is None:
        scorers = ["fuzzing", "detection"]
    if concordance_thresholds is None:
        concordance_thresholds = [0.3, 0.4, 0.5]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    per_feature_dir = output_path / "per_feature"
    per_feature_dir.mkdir(exist_ok=True)

    # Load SAE
    sae = _sae
    if sae is None:
        sae = JumpReLUSAE.from_checkpoint(sae_checkpoint)

    # Load ICD eval data
    corr_data = load_saved_correlations(icd_eval_dir)
    r_pb = corr_data["r_pb"]
    code_names = corr_data["code_names"]

    code_descriptions = _load_code_descriptions(icd_descriptions_path)

    # Load note vectors for dead-feature detection
    shard_ckpt_dir = Path(icd_eval_dir) / "shard_ckpt"
    note_vectors, _ = reassemble_note_vectors(shard_ckpt_dir)

    # Step 1: Feature selection
    tiers = select_features(
        r_pb=r_pb,
        p_adjusted=corr_data["p_adjusted"],
        significant=corr_data["significant"],
        code_names=code_names,
        note_vectors=note_vectors,
        n_strong_grounded=n_strong_grounded,
        n_weak_grounded=n_weak_grounded,
        n_non_grounded=n_non_grounded,
        n_dead=n_dead,
        seed=random_seed,
    )

    # Build feature list with tier labels
    feature_list: list[tuple[int, str]] = []
    for tier_name, ids in tiers.items():
        for fid in ids:
            feature_list.append((fid, tier_name))

    logger.info(f"Total features to process: {len(feature_list)}")

    # Load metadata for context extraction
    metadata = load_metadata(Path(activations_dir))

    # Build Anthropic client if not injected
    client = _client
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    # Step 2-6: Per-feature processing
    feature_results: list[dict] = []
    n_errors = 0

    for idx, (feature_idx, tier) in enumerate(feature_list):
        ckpt_path = per_feature_dir / f"feature_{feature_idx}.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                feature_results.append(json.load(f))
            continue

        logger.info(f"[{idx + 1}/{len(feature_list)}] Feature {feature_idx} ({tier})")

        try:
            # Context extraction
            contexts = extract_contexts_for_feature(
                sae=sae,
                feature_idx=feature_idx,
                activations_dir=Path(activations_dir),
                metadata=metadata,
                n_pos=n_contexts_train + (n_contexts_test // 2),
                n_neg=n_contexts_test // 2,
                context_window=context_window,
            )

            pos = contexts["pos_contexts"]
            neg = contexts["neg_contexts"]

            # Split pos into train (for explanation) and test (for scoring)
            train_contexts = pos[:n_contexts_train]
            test_pos = pos[n_contexts_train : n_contexts_train + n_contexts_test // 2]
            test_neg = neg[: n_contexts_test // 2]

            train_for_explain = [
                {
                    "context_str": (
                        f"[shard {c['shard']}, pos {c['position_in_shard']}] "
                        f"activation={c['activation']:.2f}"
                    ),
                    "token_str": f"token_{c['position_in_shard']}",
                }
                for c in train_contexts
            ]

            top_tokens = [c.get("token_str", f"tok_{c['position_in_shard']}") for c in pos[:5]]

            # Explanation
            explanation = explain_feature(client, train_for_explain, model=explainer_model)

            # Scoring
            fuzz_sc = None
            det_sc = None

            test_for_scoring = [
                {
                    "context_str": (f"[pos {c['position_in_shard']}] act={c['activation']:.2f}"),
                    "is_activating": True,
                }
                for c in test_pos
            ] + [
                {
                    "context_str": f"[pos {c['position_in_shard']}] act=0.00",
                    "is_activating": False,
                }
                for c in test_neg
            ]

            if "fuzzing" in scorers and explanation and test_for_scoring:
                val = fuzzing_score(client, explanation, test_for_scoring, model=explainer_model)
                fuzz_sc = None if np.isnan(val) else val

            if "detection" in scorers and explanation and test_pos and test_neg:
                val = detection_score(
                    client,
                    explanation,
                    [
                        {
                            "context_str": (
                                f"[pos {c['position_in_shard']}] " f"act={c['activation']:.2f}"
                            )
                        }
                        for c in test_pos
                    ],
                    [
                        {"context_str": (f"[pos {c['position_in_shard']}] act=0.00")}
                        for c in test_neg
                    ],
                    model=explainer_model,
                )
                det_sc = None if np.isnan(val) else val

            # Categorization
            category = (
                categorize_feature(
                    client,
                    explanation,
                    top_tokens,
                    model=categorization_model,
                )
                if explanation
                else "noise"
            )

            # Concordance (grounded tiers only)
            conc_verdict = None
            conc_rationale = None
            conc_icd_code = None
            conc_r_pb_val = None

            if tier in ("strong_grounded", "weak_grounded") and explanation:
                icd_code, r_val = _get_strongest_icd(feature_idx, r_pb, code_names)
                icd_code_clean = icd_code.replace("icd9_", "")
                icd_desc = code_descriptions.get(icd_code_clean, icd_code)

                conc_verdict, conc_rationale = check_concordance(
                    client,
                    explanation,
                    icd_code_clean,
                    icd_desc,
                    r_val,
                    model=concordance_model,
                )
                conc_icd_code = icd_code
                conc_r_pb_val = r_val

            result = {
                "feature_idx": feature_idx,
                "tier": tier,
                "explanation": explanation,
                "fuzzing_score": fuzz_sc,
                "detection_score": det_sc,
                "category": category,
                "top_tokens": top_tokens,
                "model": explainer_model,
                "concordance_verdict": conc_verdict,
                "concordance_rationale": conc_rationale,
                "concordance_icd_code": conc_icd_code,
                "concordance_r_pb": conc_r_pb_val,
            }

            _write_json(result, ckpt_path)
            feature_results.append(result)

        except Exception:
            logger.exception(f"Error processing feature {feature_idx}")
            n_errors += 1
            feature_results.append(
                {
                    "feature_idx": feature_idx,
                    "tier": tier,
                    "explanation": "",
                    "fuzzing_score": None,
                    "detection_score": None,
                    "category": "error",
                    "top_tokens": [],
                    "model": explainer_model,
                    "concordance_verdict": None,
                    "concordance_rationale": None,
                    "concordance_icd_code": None,
                    "concordance_r_pb": None,
                }
            )

    # Step 7: Assemble catalog
    assemble_catalog(feature_results, output_path)

    # Write run_summary.json
    elapsed = (datetime.now(UTC) - start_time).total_seconds()
    summary = {
        "n_features": len(feature_results),
        "n_errors": n_errors,
        "tiers": {k: len(v) for k, v in tiers.items()},
        "explainer_model": explainer_model,
        "scorers": scorers,
        "elapsed_seconds": elapsed,
        "timestamp": start_time.isoformat(),
    }
    _write_json(summary, output_path / "run_summary.json")
    logger.info(f"Auto-interp complete: {len(feature_results)} features in {elapsed:.0f}s")

    return summary
