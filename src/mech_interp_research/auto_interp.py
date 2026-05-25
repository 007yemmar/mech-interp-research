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
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors
from tqdm import tqdm

from mech_interp_research.feature_inspector import load_tokenizer
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


def select_shards_for_features(
    note_vectors: np.ndarray,
    metadata: pd.DataFrame,
    feature_ids: list[int],
    n_top_notes: int = 50,
) -> dict[int, list[int]]:
    """Pick the minimal set of shards containing top-N notes per feature.

    Uses the already-loaded note_vectors [n_notes, d_sae] to find which
    notes activate most for each feature, then returns only the unique
    shards those notes reside in (typically 5-10 per feature instead of 20).

    Returns:
        Dict mapping feature_idx -> sorted list of shard indices.
    """
    note_to_shard = metadata.drop_duplicates("note_idx").set_index("note_idx")["shard"]

    result: dict[int, list[int]] = {}
    for fid in feature_ids:
        col = note_vectors[:, fid]
        top_k = min(n_top_notes, len(col))
        top_note_indices = np.argsort(col)[-top_k:][::-1]
        shards: set[int] = set()
        for nidx in top_note_indices:
            if int(nidx) in note_to_shard.index:
                shards.add(int(note_to_shard.loc[int(nidx)]))
        result[fid] = sorted(shards)

    shard_counts = [len(v) for v in result.values()]
    logger.info(
        f"Shard selection for {len(feature_ids)} features "
        f"(median {np.median(shard_counts):.0f} shards/feature, "
        f"range {min(shard_counts)}-{max(shard_counts)})"
    )
    return result


def _encode_selective(
    sae: JumpReLUSAE,
    x: np.ndarray,
    feature_indices: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Encode only the requested SAE features — avoids the full d_sae matmul.

    Reproduces ``sae.encode()`` but computes only the columns in
    *feature_indices*, reducing FLOPs by ``d_sae / len(feature_indices)``.
    """
    W_sub = sae.W_enc[:, feature_indices]
    b_sub = sae.b_enc[feature_indices]
    t_sub = sae.threshold[feature_indices]

    n = x.shape[0]
    if n <= chunk_size:
        x_in = (x - sae.b_dec) if sae.subtract_b_dec else x
        pre = x_in @ W_sub + b_sub
        return pre * (pre > t_sub)

    chunks: list[np.ndarray] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_in = (x[start:end] - sae.b_dec) if sae.subtract_b_dec else x[start:end]
        pre = x_in @ W_sub + b_sub
        chunks.append(pre * (pre > t_sub))
    return np.concatenate(chunks, axis=0)


def extract_all_contexts(
    sae: JumpReLUSAE,
    feature_ids: list[int],
    shard_map: dict[int, list[int]],
    activations_dir: Path,
    metadata: pd.DataFrame,
    n_pos: int = 25,
    n_neg: int = 5,
) -> dict[int, dict]:
    """Extract contexts for all features in a single shard-centric pass.

    Builds an inverted index (shard -> features), loads each shard once,
    and encodes only the needed feature columns in one batched matmul per
    shard.  Each shard is loaded and encoded exactly once regardless of
    how many features need it.
    """
    import heapq

    activations_dir = Path(activations_dir)

    shard_to_features: dict[int, set[int]] = {}
    fid_set = set(feature_ids)
    for fid, shards in shard_map.items():
        if fid in fid_set:
            for s in shards:
                shard_to_features.setdefault(s, set()).add(fid)

    shards_to_process = sorted(shard_to_features.keys())
    logger.info(
        f"Shard-centric extraction: {len(shards_to_process)} unique shards "
        f"for {len(feature_ids)} features"
    )

    pos_heaps: dict[int, list] = {fid: [] for fid in feature_ids}
    neg_pools: dict[int, list] = {fid: [] for fid in feature_ids}
    rngs = {fid: np.random.default_rng(fid) for fid in feature_ids}
    _uid = 0
    max_neg_pool = n_neg * 20

    for shard_idx in tqdm(shards_to_process, desc="shard extraction", unit="shard"):
        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            continue

        shard_data = load_safetensors(str(shard_path))
        act_key = next(iter(shard_data))
        shard_acts = shard_data[act_key].astype(np.float32)
        n_tokens_shard = shard_acts.shape[0]

        features_for_shard = sorted(shard_to_features[shard_idx])
        feat_arr = np.array(features_for_shard, dtype=np.intp)
        fid_to_local = {fid: loc for loc, fid in enumerate(features_for_shard)}

        latents = _encode_selective(sae, shard_acts, feat_arr)

        shard_meta = metadata[metadata["shard"] == shard_idx]
        note_of = np.full(n_tokens_shard, -1, dtype=np.int64)
        pos_in_note = np.full(n_tokens_shard, -1, dtype=np.int64)
        for _, row in shard_meta.iterrows():
            rs, re = int(row["row_start"]), int(row["row_end"])
            note_of[rs:re] = int(row["note_idx"])
            pos_in_note[rs:re] = np.arange(re - rs)

        for fid in features_for_shard:
            col = latents[:, fid_to_local[fid]]

            nonzero_pos = np.where(col > 0)[0]

            heap = pos_heaps[fid]
            heap_min = heap[0][0] if len(heap) >= n_pos else -np.inf
            if len(nonzero_pos) > 0 and float(col[nonzero_pos].max()) <= heap_min:
                pass  # this shard can't improve the heap
            else:
                for pos in nonzero_pos:
                    if note_of[pos] == -1:
                        continue  # token not in any note boundary; no text available
                    act_val = float(col[pos])
                    if len(heap) < n_pos:
                        heapq.heappush(
                            heap,
                            (
                                act_val,
                                _uid,
                                {
                                    "activation": act_val,
                                    "position_in_shard": int(pos),
                                    "shard": int(shard_idx),
                                    "note_idx": int(note_of[pos]),
                                    "position_in_note": int(pos_in_note[pos]),
                                },
                            ),
                        )
                        _uid += 1
                    elif act_val > heap[0][0]:
                        heapq.heapreplace(
                            heap,
                            (
                                act_val,
                                _uid,
                                {
                                    "activation": act_val,
                                    "position_in_shard": int(pos),
                                    "shard": int(shard_idx),
                                    "note_idx": int(note_of[pos]),
                                    "position_in_note": int(pos_in_note[pos]),
                                },
                            ),
                        )
                        _uid += 1

            if len(neg_pools[fid]) < max_neg_pool:
                zero_pos = np.where(col == 0)[0]
                if len(zero_pos) > 0:
                    n_sample = min(3, len(zero_pos))
                    sampled = rngs[fid].choice(zero_pos, size=n_sample, replace=False)
                    for pos in sampled:
                        neg_pools[fid].append(
                            {
                                "activation": 0.0,
                                "position_in_shard": int(pos),
                                "shard": int(shard_idx),
                                "note_idx": int(note_of[pos]),
                                "position_in_note": int(pos_in_note[pos]),
                            }
                        )

        del shard_acts, shard_data, latents

    result: dict[int, dict] = {}
    for fid in feature_ids:
        pos_list = sorted(
            [ctx for _, _, ctx in pos_heaps[fid]],
            key=lambda x: x["activation"],
            reverse=True,
        )
        neg_list = neg_pools[fid]
        if len(neg_list) > n_neg:
            indices = rngs[fid].choice(len(neg_list), size=n_neg, replace=False)
            neg_list = [neg_list[int(i)] for i in indices]
        result[fid] = {"pos_contexts": pos_list, "neg_contexts": neg_list}

    logger.info(
        f"Extraction complete: "
        f"{sum(len(v['pos_contexts']) > 0 for v in result.values())}/{len(feature_ids)} "
        f"features have positive contexts"
    )
    return result


def extract_contexts_for_feature(
    sae: JumpReLUSAE,
    feature_idx: int,
    activations_dir: Path,
    metadata: pd.DataFrame,
    n_pos: int = 20,
    n_neg: int = 10,
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
        note_boundaries: list[tuple[int, int, int]] = []
        for _, row in shard_meta.iterrows():
            note_boundaries.append(
                (int(row["row_start"]), int(row["row_end"]), int(row["note_idx"]))
            )
        note_boundaries.sort(key=lambda x: x[0])

        def _find_note(pos: int, boundaries: list[tuple[int, int, int]]) -> tuple[int, int]:
            for rs, re_, nidx in boundaries:
                if rs <= pos < re_:
                    return nidx, pos - rs
            return -1, -1

        nonzero_positions = np.where(feature_acts > 0)[0]
        for pos in nonzero_positions:
            note_idx, pos_in_note = _find_note(int(pos), note_boundaries)
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

        zero_positions = np.where(feature_acts == 0)[0]
        if len(zero_positions) > 0:
            sample_size = min(n_neg * 2, len(zero_positions))
            sampled = rng.choice(zero_positions, size=sample_size, replace=False)
            for pos in sampled:
                note_idx, pos_in_note = _find_note(int(pos), note_boundaries)
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

    all_pos.sort(key=lambda x: x["activation"], reverse=True)
    pos_contexts = all_pos[:n_pos]

    if len(all_neg_candidates) > n_neg:
        neg_indices = rng.choice(len(all_neg_candidates), size=n_neg, replace=False)
        neg_contexts = [all_neg_candidates[i] for i in neg_indices]
    else:
        neg_contexts = all_neg_candidates[:n_neg]

    return {"pos_contexts": pos_contexts, "neg_contexts": neg_contexts}


def resolve_token_text(
    contexts: list[dict],
    note_texts: dict[int, str],
    tokenizer,
    context_window: int = 15,
    max_length: int = 8192,
) -> list[dict]:
    """Resolve raw position contexts into text with **trigger token** marked.

    Adds 'context_str' and 'token_str' keys to each context dict.
    """
    tokenized_cache: dict[int, list[int]] = {}

    for ctx in contexts:
        note_idx = ctx["note_idx"]
        text = note_texts.get(note_idx)
        if text is None:
            ctx["context_str"] = f"[note {note_idx} text unavailable]"
            ctx["token_str"] = ""
            continue

        if note_idx not in tokenized_cache:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            tokenized_cache[note_idx] = encoded["input_ids"]

        token_ids = tokenized_cache[note_idx]
        pos = ctx["position_in_note"]

        if pos < 0 or pos >= len(token_ids):
            ctx["context_str"] = f"[position {pos} out of range for note {note_idx}]"
            ctx["token_str"] = ""
            continue

        trigger_str = tokenizer.decode([token_ids[pos]], skip_special_tokens=False)
        ctx["token_str"] = trigger_str.strip()

        ctx_start = max(0, pos - context_window)
        ctx_end = min(len(token_ids), pos + context_window + 1)

        parts = []
        for i in range(ctx_start, ctx_end):
            tok_text = tokenizer.decode([token_ids[i]], skip_special_tokens=False)
            if i == pos:
                parts.append(f"**{tok_text}**")
            else:
                parts.append(tok_text)
        ctx["context_str"] = "".join(parts)

    return contexts


def _make_distractor_context(
    ctx: dict,
    note_texts: dict[int, str],
    tokenizer,
    context_window: int,
    rng: np.random.Generator,
    max_length: int = 8192,
) -> dict | None:
    """Create a distractor version of an activating context.

    Takes an activating context (which already has ``context_str`` with the
    trigger token highlighted) and builds a new context that highlights a
    *different* random token from the same context window instead.

    Returns a new dict with ``context_str``, ``token_str``, and
    ``is_activating=False``, or ``None`` if no valid distractor position
    exists or the note text is unavailable.
    """
    note_idx = ctx["note_idx"]
    pos = ctx["position_in_note"]
    text = note_texts.get(note_idx)
    if text is None:
        return None

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    token_ids = encoded["input_ids"]

    if pos < 0 or pos >= len(token_ids):
        return None

    ctx_start = max(0, pos - context_window)
    ctx_end = min(len(token_ids), pos + context_window + 1)

    # Candidate positions: everything in the window except the trigger
    candidates = [i for i in range(ctx_start, ctx_end) if i != pos]
    if not candidates:
        return None

    distractor_pos = int(rng.choice(candidates))
    distractor_token_str = tokenizer.decode(
        [token_ids[distractor_pos]],
        skip_special_tokens=False,
    ).strip()

    parts = []
    for i in range(ctx_start, ctx_end):
        tok_text = tokenizer.decode([token_ids[i]], skip_special_tokens=False)
        if i == distractor_pos:
            parts.append(f"**{tok_text}**")
        else:
            parts.append(tok_text)

    return {
        "note_idx": ctx["note_idx"],
        "position_in_note": distractor_pos,
        "context_str": "".join(parts),
        "token_str": distractor_token_str,
        "is_activating": False,
    }


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

EXPLAIN_AND_CATEGORIZE_PROMPT = """\
Below are the top {n} text contexts where a sparse autoencoder feature \
activates most strongly in clinical discharge summaries. The trigger \
token in each context is marked with **asterisks**.

{contexts}

1. What pattern causes this feature to activate? Describe the specific \
concept, linguistic pattern, or structural element that these contexts \
share. Be as specific as possible — name clinical concepts, drugs, \
or procedures if applicable. One to two sentences.

2. Categorize this feature into exactly one of:
- clinical_concept (a specific medical diagnosis, condition, or finding)
- clinical_vocabulary (drug names, dosages, units, lab values, procedures)
- general_language (POS, syntax, common words, generic patterns)
- structural_pattern (formatting, section headers, signatures, templates)
- noise (no coherent pattern)

Format your response as:
EXPLANATION: <your explanation>
CATEGORY: <category_name>"""


def explain_feature(
    client,
    pos_contexts: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 256,
) -> str:
    """Generate an explanation for a feature given its top activating contexts.

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


def explain_and_categorize_feature(
    client,
    pos_contexts: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 300,
) -> tuple[str, str]:
    """Generate explanation + category in a single LLM call.

    Returns:
        (explanation, category) tuple. Category is one of the 5 valid
        categories or "unknown".
    """
    if not pos_contexts:
        return "", "noise"

    context_lines = []
    for i, ctx in enumerate(pos_contexts, 1):
        context_lines.append(f"{i}. {ctx['context_str']}")

    prompt = EXPLAIN_AND_CATEGORIZE_PROMPT.format(
        n=len(pos_contexts),
        contexts="\n".join(context_lines),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    explanation = raw
    category = "unknown"

    expl_match = re.search(r"EXPLANATION:\s*(.+?)(?:\nCATEGORY:|\Z)", raw, re.DOTALL)
    cat_match = re.search(r"CATEGORY:\s*(\S+)", raw)

    if expl_match:
        explanation = expl_match.group(1).strip()
    if cat_match:
        cat_raw = cat_match.group(1).strip().lower()
        if cat_raw in VALID_CATEGORIES:
            category = cat_raw
        else:
            for cat in VALID_CATEGORIES:
                if cat in cat_raw:
                    category = cat
                    break

    return explanation, category


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
    # Accept: "1. YES", "1: YES", "1) YES", "1 - YES", "1 YES", etc.
    pattern = re.compile(r"(\d+)(?:\s*[-–—.):]\s*|\s+)(YES|NO)", re.IGNORECASE)
    for line in text.strip().split("\n"):
        line = line.strip().strip("*")  # strip markdown bold (**1. YES**)
        if not line:
            continue
        match = pattern.match(line)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < n_expected:
                results[idx] = match.group(2).upper() == "YES"
    return results


_FORMAT_REMINDER = (
    "\n\nIMPORTANT: Respond with exactly one line per context using only this format:\n"
    "1. YES\n2. NO\n(continue for every context numbered above)"
)


def fuzzing_score(
    client,
    explanation: str,
    test_contexts: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 256,
    format_reminder: bool = False,
) -> float:
    """Fuzzing scorer: can the LLM identify which tokens match the pattern?

    Args:
        client: Anthropic client.
        explanation: Feature explanation string.
        test_contexts: List of dicts with 'context_str' and 'is_activating'.
        model: Anthropic model ID.
        format_reminder: If True, append explicit format instructions (used on retry).

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
    if format_reminder:
        prompt += _FORMAT_REMINDER

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    predictions = _parse_yes_no_responses(raw_text, len(test_contexts))
    parsed = [
        (p, ctx["is_activating"])
        for p, ctx in zip(predictions, test_contexts, strict=False)
        if p is not None
    ]

    if not parsed:
        logger.debug("fuzzing_score parse failure. Raw response: %r", raw_text[:300])
        return float("nan")

    correct = sum(1 for pred, actual in parsed if pred == actual)
    return correct / len(parsed)


def detection_score(
    client,
    explanation: str,
    pos_contexts: list[dict],
    neg_contexts: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 256,
    _shuffle_seed: int | None = 42,
    format_reminder: bool = False,
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
        format_reminder: If True, append explicit format instructions (used on retry).

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
    if format_reminder:
        prompt += _FORMAT_REMINDER

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    predictions = _parse_yes_no_responses(raw_text, len(items))
    parsed = [(p, gt) for p, gt in zip(predictions, ground_truth, strict=False) if p is not None]

    if not parsed:
        logger.debug("detection_score parse failure. Raw response: %r", raw_text[:300])
        return float("nan")

    correct = sum(1 for pred, actual in parsed if pred == actual)
    return correct / len(parsed)


def score_explanation_against_contexts(
    client,
    explanation: str,
    test_pos: list[dict],
    test_neg: list[dict],
    note_texts: dict[int, str],
    tokenizer,
    model: str,
    scorers: list[str],
    context_window: int,
    owner_feature_id: int,
) -> tuple[float | None, float | None, int]:
    """Run Fuzzing + Detection scoring of ``explanation`` against test contexts.

    Shared by ``_process_one_feature`` (real explanation) and the
    shuffled-explanation control (wrong explanation). The fuzzing distractor RNG
    is seeded by ``owner_feature_id`` so the test set is identical across
    explanations for the same feature. Contexts must already be resolved
    (``context_str``/``token_str`` populated via ``resolve_token_text``).

    Returns ``(fuzzing_score, detection_score, parsing_errors)`` with ``None`` for
    a scorer that was not requested or had no usable contexts.
    """
    fuzz_sc: float | None = None
    det_sc: float | None = None
    parsing_errors = 0

    if "fuzzing" in scorers and explanation:
        # True Fuzzing (Paulo et al. 2024): for each activating test context,
        # present the real trigger (is_activating=True) and a distractor version
        # that highlights a different random token from the same window
        # (is_activating=False).
        fuzz_rng = np.random.default_rng(owner_feature_id)
        test_for_fuzzing: list[dict] = []
        for c in test_pos:
            if c.get("token_str") is None:  # token_str="" is valid (whitespace token)
                continue
            test_for_fuzzing.append({**c, "is_activating": True})
            distractor = _make_distractor_context(
                c, note_texts, tokenizer, context_window, fuzz_rng
            )
            if distractor is not None:
                test_for_fuzzing.append(distractor)

        # Fallback: if no distractor pairs could be created, use a mix of
        # activating and non-activating contexts instead.
        if not any(not item["is_activating"] for item in test_for_fuzzing):
            test_for_fuzzing = [
                {**c, "is_activating": True} for c in test_pos if c.get("token_str") is not None
            ] + [{**c, "is_activating": False} for c in test_neg if c.get("token_str") is not None]
    else:
        test_for_fuzzing = []

    if "fuzzing" in scorers and explanation and test_for_fuzzing:
        val = fuzzing_score(client, explanation, test_for_fuzzing, model=model)
        if np.isnan(val):
            parsing_errors += 1
            val = fuzzing_score(
                client, explanation, test_for_fuzzing, model=model, format_reminder=True
            )
        fuzz_sc = None if np.isnan(val) else val

    if "detection" in scorers and explanation and test_pos and test_neg:
        det_pos = [c for c in test_pos if c.get("token_str") is not None]
        det_neg = [c for c in test_neg if c.get("token_str") is not None]
        if det_pos and det_neg:
            val = detection_score(client, explanation, det_pos, det_neg, model=model)
            if np.isnan(val):
                parsing_errors += 1
                val = detection_score(
                    client, explanation, det_pos, det_neg, model=model, format_reminder=True
                )
            det_sc = None if np.isnan(val) else val

    return fuzz_sc, det_sc, parsing_errors


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
    model: str = "claude-sonnet-4-6",
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
    model: str = "claude-sonnet-4-6",
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


def _scorer_stats(results: list[dict]) -> dict:
    """Compute scorer stats for a group of feature results."""
    fuzz = [r["fuzzing_score"] for r in results if r.get("fuzzing_score") is not None]
    det = [r["detection_score"] for r in results if r.get("detection_score") is not None]
    return {
        "mean_fuzzing": float(np.mean(fuzz)) if fuzz else None,
        "median_fuzzing": float(np.median(fuzz)) if fuzz else None,
        "std_fuzzing": float(np.std(fuzz)) if fuzz else None,
        "mean_detection": float(np.mean(det)) if det else None,
        "median_detection": float(np.median(det)) if det else None,
        "std_detection": float(np.std(det)) if det else None,
        "n_valid_fuzzing": len(fuzz),
        "n_valid_detection": len(det),
    }


def _concordance_stats(results: list[dict]) -> dict:
    """Compute concordance stats for a group."""
    conc = [
        r for r in results if r.get("concordance_verdict") in ("YES", "PARTIAL", "NO", "UNKNOWN")
    ]
    if not conc:
        return {}
    yes = sum(1 for r in conc if r["concordance_verdict"] == "YES")
    partial = sum(1 for r in conc if r["concordance_verdict"] == "PARTIAL")
    no = sum(1 for r in conc if r["concordance_verdict"] == "NO")
    unknown = sum(1 for r in conc if r["concordance_verdict"] == "UNKNOWN")
    n = len(conc)
    return {
        "yes_count": yes,
        "partial_count": partial,
        "no_count": no,
        "unknown_count": unknown,
        "total": n,
        "concordance_rate": (yes + partial) / n if n > 0 else 0,
        "exact_match_rate": yes / n if n > 0 else 0,
    }


def assemble_catalog(
    feature_results: list[dict],
    output_dir: Path,
    concordance_thresholds: list[float] | None = None,
) -> None:
    """Assemble feature_catalog.csv + summary JSONs from per-feature results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if concordance_thresholds is None:
        concordance_thresholds = [0.3, 0.4, 0.5]

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
                    "concordance_icd_description": r.get("concordance_icd_description"),
                    "concordance_r_pb": r["concordance_r_pb"],
                }
                for r in conc_rows
            ]
        )
        conc_df.to_csv(output_dir / "concordance_results.csv", index=False)
        logger.info(f"Wrote concordance_results.csv: {len(conc_df)} features")

    # categorization_summary.json — global + by tier
    tier_names = sorted({r["tier"] for r in feature_results})
    cat_summary: dict = {"global": _cat_stats(feature_results)}
    for t in tier_names:
        tier_results = [r for r in feature_results if r["tier"] == t]
        cat_summary[t] = _cat_stats(tier_results)
    _write_json(cat_summary, output_dir / "categorization_summary.json")

    # concordance_summary.json — global + by threshold
    if conc_rows:
        conc_summary: dict = {"global": _concordance_stats(conc_rows)}
        for threshold in concordance_thresholds:
            subset = [
                r
                for r in conc_rows
                if r.get("concordance_r_pb") is not None and abs(r["concordance_r_pb"]) > threshold
            ]
            conc_summary[f"r>{threshold}"] = _concordance_stats(subset)
        _write_json(conc_summary, output_dir / "concordance_summary.json")

    # scorer_summary.json — global + by tier
    scorer_summary: dict = {"global": _scorer_stats(feature_results)}
    for t in tier_names:
        tier_results = [r for r in feature_results if r["tier"] == t]
        scorer_summary[t] = _scorer_stats(tier_results)
    _write_json(scorer_summary, output_dir / "scorer_summary.json")


def _cat_stats(results: list[dict]) -> dict:
    cats = [r["category"] for r in results]
    counts: dict[str, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    total = len(cats)
    return {
        "counts": counts,
        "fractions": {k: v / total for k, v in counts.items()} if total > 0 else {},
        "total": total,
    }


def _write_json(data: dict, path: Path) -> None:
    """Write JSON atomically (temp file in same dir + os.replace) so a crash
    mid-write never leaves a truncated file that would jam checkpoint resume."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
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


def _load_code_descriptions(
    csv_path: str | Path | None = None,
    keywords_yaml_path: str | Path | None = None,
) -> dict[str, str]:
    """Load ICD code descriptions from a CSV or keywords YAML.

    Tries the CSV first (columns: code, description). Falls back to
    the keywords YAML (icd9_XXXX: {description: "..."}) which is
    already shipped to Modal for the lexical baseline.

    Returns:
        Dict mapping bare code string (e.g. "4019") to description.
    """
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        if "code" in df.columns and "description" in df.columns:
            return dict(zip(df["code"].astype(str), df["description"].astype(str), strict=False))

    if keywords_yaml_path is not None and Path(keywords_yaml_path).exists():
        import yaml

        with open(keywords_yaml_path) as f:
            raw = yaml.safe_load(f)
        descriptions: dict[str, str] = {}
        for key, val in raw.items():
            if isinstance(val, dict) and "description" in val:
                code = key.replace("icd9_", "")
                descriptions[code] = str(val["description"])
        if descriptions:
            logger.info(f"Loaded {len(descriptions)} ICD descriptions from keywords YAML")
            return descriptions

    return {}


def _needs_retry(result: dict, scorers: list[str]) -> bool:
    """Return True if any scored field in a checkpoint should be retried."""
    if "fuzzing" in scorers and result.get("fuzzing_score") is None:
        explanation = result.get("explanation", "")
        if explanation and not explanation.startswith("Feature "):
            return True
    if "detection" in scorers and result.get("detection_score") is None:
        explanation = result.get("explanation", "")
        if explanation and not explanation.startswith("Feature "):
            return True
    if result.get("concordance_verdict") == "UNKNOWN":
        return True
    return False


def _fill_missing_scores(
    result: dict,
    contexts: dict,
    note_texts: dict[int, str],
    tokenizer,
    client,
    model: str,
    concordance_model: str,
    r_pb: np.ndarray,
    code_names: list[str],
    code_descriptions: dict[str, str],
    scorers: list[str],
    n_contexts_train: int,
    n_contexts_test: int,
    context_window: int,
) -> dict:
    """Re-run only the null/failed scored fields from an existing checkpoint.

    Explanation and categorization are preserved unchanged. Only
    ``fuzzing_score`` (if None), ``detection_score`` (if None), and
    ``concordance_verdict`` (if UNKNOWN from a previous parse failure) are
    re-computed.
    """
    result = result.copy()
    explanation = result.get("explanation", "")

    if not explanation or explanation.startswith("Feature "):
        return result

    feature_idx = result["feature_idx"]
    tier = result["tier"]

    pos = list(contexts.get("pos_contexts", []))
    neg = list(contexts.get("neg_contexts", []))
    resolve_token_text(pos + neg, note_texts, tokenizer, context_window=context_window)

    # Filter out positions that have no note text (note_idx < 0).
    pos = [c for c in pos if c.get("note_idx", -1) >= 0]
    neg = [c for c in neg if c.get("note_idx", -1) >= 0]

    test_pos = pos[n_contexts_train : n_contexts_train + n_contexts_test // 2]
    test_neg = neg[: n_contexts_test // 2]

    parsing_errors = result.get("parsing_errors", 0)

    if "fuzzing" in scorers and result.get("fuzzing_score") is None:
        fuzz_rng = np.random.default_rng(feature_idx)
        test_for_fuzzing: list[dict] = []
        for c in test_pos:
            test_for_fuzzing.append({**c, "is_activating": True})
            distractor = _make_distractor_context(
                c,
                note_texts,
                tokenizer,
                context_window,
                fuzz_rng,
            )
            if distractor is not None:
                test_for_fuzzing.append(distractor)

        if not any(not item["is_activating"] for item in test_for_fuzzing):
            test_for_fuzzing = [{**c, "is_activating": True} for c in test_pos] + [
                {**c, "is_activating": False} for c in test_neg
            ]

        if test_for_fuzzing:
            val = fuzzing_score(client, explanation, test_for_fuzzing, model=model)
            if np.isnan(val):
                parsing_errors += 1
                val = fuzzing_score(
                    client,
                    explanation,
                    test_for_fuzzing,
                    model=model,
                    format_reminder=True,
                )
            result["fuzzing_score"] = None if np.isnan(val) else val
            logger.info(f"Retry fuzzing feature {feature_idx}: {result['fuzzing_score']}")

    if "detection" in scorers and result.get("detection_score") is None:
        det_pos = [c for c in test_pos if c.get("token_str") is not None]
        det_neg = [c for c in test_neg if c.get("token_str") is not None]
        if det_pos and det_neg:
            val = detection_score(client, explanation, det_pos, det_neg, model=model)
            if np.isnan(val):
                parsing_errors += 1
                val = detection_score(
                    client,
                    explanation,
                    det_pos,
                    det_neg,
                    model=model,
                    format_reminder=True,
                )
            result["detection_score"] = None if np.isnan(val) else val
            logger.info(f"Retry detection feature {feature_idx}: {result['detection_score']}")

    if result.get("concordance_verdict") == "UNKNOWN" and tier in (
        "strong_grounded",
        "weak_grounded",
    ):
        icd_code, r_val = _get_strongest_icd(feature_idx, r_pb, code_names)
        icd_code_clean = icd_code.replace("icd9_", "")
        icd_desc = code_descriptions.get(icd_code_clean, icd_code_clean)
        conc_verdict, conc_rationale = check_concordance(
            client,
            explanation,
            icd_code_clean,
            icd_desc,
            r_val,
            model=concordance_model,
        )
        result["concordance_verdict"] = conc_verdict
        result["concordance_rationale"] = conc_rationale
        logger.info(f"Retry concordance feature {feature_idx}: {conc_verdict}")

    result["parsing_errors"] = parsing_errors
    return result


def _process_one_feature(
    feature_idx: int,
    tier: str,
    contexts: dict,
    note_texts: dict[int, str],
    tokenizer,
    client,
    model: str,
    concordance_model: str,
    r_pb: np.ndarray,
    code_names: list[str],
    code_descriptions: dict[str, str],
    scorers: list[str],
    n_contexts_train: int,
    n_contexts_test: int,
    context_window: int,
    min_contexts: int = 5,
) -> dict:
    """Process a single feature: explain → score → categorize → concordance.

    Contexts must be pre-extracted via extract_all_contexts().
    """
    pos = contexts["pos_contexts"]
    neg = contexts["neg_contexts"]

    if len(pos) == 0:
        return _dead_result(feature_idx, tier, model, "no_activation")

    resolve_token_text(pos + neg, note_texts, tokenizer, context_window=context_window)

    train_contexts = pos[:n_contexts_train]
    test_pos = pos[n_contexts_train : n_contexts_train + n_contexts_test // 2]
    test_neg = neg[: n_contexts_test // 2]

    if len(train_contexts) < min_contexts:
        return _dead_result(feature_idx, tier, model, "insufficient_contexts")

    if len(test_pos) == 0 and len(test_neg) == 0:
        logger.debug(f"Feature {feature_idx}: no test contexts available for scoring")

    top_tokens = [c.get("token_str", "") for c in pos[:5] if c.get("token_str")]

    explanation, category = explain_and_categorize_feature(
        client,
        train_contexts,
        model=model,
    )

    fuzz_sc, det_sc, parsing_errors = score_explanation_against_contexts(
        client=client,
        explanation=explanation,
        test_pos=test_pos,
        test_neg=test_neg,
        note_texts=note_texts,
        tokenizer=tokenizer,
        model=model,
        scorers=scorers,
        context_window=context_window,
        owner_feature_id=feature_idx,
    )

    conc_verdict = None
    conc_rationale = None
    conc_icd_code = None
    conc_icd_desc = None
    conc_r_pb_val = None

    if tier in ("strong_grounded", "weak_grounded") and explanation:
        icd_code, r_val = _get_strongest_icd(feature_idx, r_pb, code_names)
        icd_code_clean = icd_code.replace("icd9_", "")
        icd_desc = code_descriptions.get(icd_code_clean, icd_code_clean)

        conc_verdict, conc_rationale = check_concordance(
            client,
            explanation,
            icd_code_clean,
            icd_desc,
            r_val,
            model=concordance_model,
        )
        conc_icd_code = icd_code
        conc_icd_desc = icd_desc
        conc_r_pb_val = r_val

    return {
        "feature_idx": feature_idx,
        "tier": tier,
        "explanation": explanation,
        "fuzzing_score": fuzz_sc,
        "detection_score": det_sc,
        "category": category,
        "top_tokens": top_tokens,
        "model": model,
        "parsing_errors": parsing_errors,
        "concordance_verdict": conc_verdict,
        "concordance_rationale": conc_rationale,
        "concordance_icd_code": conc_icd_code,
        "concordance_icd_description": conc_icd_desc,
        "concordance_r_pb": conc_r_pb_val,
    }


def _dead_result(feature_idx: int, tier: str, model: str, reason: str) -> dict:
    return {
        "feature_idx": feature_idx,
        "tier": tier,
        "explanation": f"Feature {reason.replace('_', ' ')}.",
        "fuzzing_score": None,
        "detection_score": None,
        "category": "noise",
        "top_tokens": [],
        "model": model,
        "parsing_errors": 0,
        "concordance_verdict": None,
        "concordance_rationale": None,
        "concordance_icd_code": None,
        "concordance_icd_description": None,
        "concordance_r_pb": None,
    }


def _load_note_texts(
    csv_path: str | Path,
    metadata: pd.DataFrame,
    join_key: str = "admission_id",
    text_col: str = "note_text",
) -> dict[int, str]:
    """Load note texts keyed by note_idx."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path, usecols=[join_key, text_col])
    meta_notes = metadata[[join_key, "note_idx"]].drop_duplicates()
    merged = meta_notes.merge(df, on=join_key, how="inner")
    texts: dict[int, str] = {}
    for _, row in merged.iterrows():
        texts[int(row["note_idx"])] = str(row[text_col])
    logger.info(f"Loaded text for {len(texts)} notes")
    return texts


def run_auto_interp(
    sae_checkpoint: str | None,
    activations_dir: str,
    icd_eval_dir: str,
    icd_csv_path: str,
    output_dir: str,
    n_strong_grounded: int = 280,
    n_weak_grounded: int = 100,
    n_non_grounded: int = 1000,
    n_dead: int = 100,
    explainer_model: str = "claude-sonnet-4-6",
    comparison_model: str | None = None,
    concordance_model: str = "claude-sonnet-4-6",
    n_contexts_train: int = 20,
    n_contexts_test: int = 10,
    max_tokens_context: int = 30,
    scorers: list[str] | None = None,
    concordance_thresholds: list[float] | None = None,
    random_seed: int = 42,
    icd_descriptions_path: str | None = None,
    icd_keywords_yaml_path: str | None = None,
    model_name: str = "google/gemma-2-2b",
    join_key: str = "admission_id",
    text_col: str = "note_text",
    _client=None,
    _sae: JumpReLUSAE | None = None,
    _tokenizer=None,
    _note_texts: dict[int, str] | None = None,
    _commit_volume=None,
    **_kwargs,
) -> dict:
    """Run the full auto-interp pipeline.

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

    sae = _sae
    if sae is None:
        sae = JumpReLUSAE.from_checkpoint(sae_checkpoint)

    corr_data = load_saved_correlations(icd_eval_dir)
    r_pb = corr_data["r_pb"]
    code_names = corr_data["code_names"]
    code_descriptions = _load_code_descriptions(icd_descriptions_path, icd_keywords_yaml_path)

    shard_ckpt_dir = Path(icd_eval_dir) / "shard_ckpt"
    note_vectors, _ = reassemble_note_vectors(shard_ckpt_dir)

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

    feature_list: list[tuple[int, str]] = []
    for tier_name, ids in tiers.items():
        for fid in ids:
            feature_list.append((fid, tier_name))
    logger.info(f"Total features to process: {len(feature_list)}")

    metadata = load_metadata(Path(activations_dir))

    all_feature_ids = [fid for fid, _ in feature_list]
    shard_map = select_shards_for_features(note_vectors, metadata, all_feature_ids)

    note_texts = _note_texts
    if note_texts is None:
        note_texts = _load_note_texts(icd_csv_path, metadata, join_key, text_col)

    tokenizer = _tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    client = _client
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=5)

    models_to_run = [explainer_model]
    if comparison_model and comparison_model != explainer_model:
        models_to_run.append(comparison_model)

    # Determine which features still need work across any model.
    features_needing_work: set[int] = set()
    for current_model in models_to_run:
        model_feature_dir = per_feature_dir / current_model.replace("/", "_")
        model_feature_dir.mkdir(exist_ok=True)
        for fid, _ in feature_list:
            if not (model_feature_dir / f"feature_{fid}.json").exists():
                features_needing_work.add(fid)

    # Phase 1: Shard-centric batch extraction with disk cache.
    contexts_cache_path = output_path / "extracted_contexts.json"
    all_contexts: dict[int, dict] = {}

    if contexts_cache_path.exists():
        logger.info(f"Loading cached contexts from {contexts_cache_path.name}")
        with open(contexts_cache_path) as f:
            raw_cache = json.load(f)
        all_contexts = {int(k): v for k, v in raw_cache.items()}
        cached_ids = set(all_contexts.keys())
        still_needed = features_needing_work - cached_ids
        if still_needed:
            logger.info(
                f"Cache has {len(cached_ids)} features; extracting {len(still_needed)} additional"
            )
        else:
            still_needed = set()
    else:
        still_needed = features_needing_work

    if still_needed:
        n_pos = n_contexts_train + (n_contexts_test // 2)
        n_neg = n_contexts_test // 2
        new_contexts = extract_all_contexts(
            sae=sae,
            feature_ids=sorted(still_needed),
            shard_map={fid: shard_map[fid] for fid in still_needed if fid in shard_map},
            activations_dir=Path(activations_dir),
            metadata=metadata,
            n_pos=n_pos,
            n_neg=n_neg,
        )
        all_contexts.update(new_contexts)
        with open(contexts_cache_path, "w") as f:
            json.dump({str(k): v for k, v in all_contexts.items()}, f)
        logger.info(f"Saved {len(all_contexts)} feature contexts to {contexts_cache_path.name}")
        if _commit_volume is not None:
            _commit_volume()
    elif not features_needing_work:
        logger.info("All features already checkpointed — skipping extraction")

    # Phase 2: Per-feature API calls with checkpointing.
    all_model_results: dict[str, list[dict]] = {}

    for current_model in models_to_run:
        model_label = "primary" if current_model == explainer_model else "comparison"
        logger.info(f"Running {model_label} model: {current_model}")

        model_feature_dir = per_feature_dir / current_model.replace("/", "_")

        feature_results: list[dict] = []
        n_errors = 0

        pbar = tqdm(
            enumerate(feature_list),
            total=len(feature_list),
            desc=f"auto-interp ({model_label})",
            unit="feat",
        )
        for _idx, (feature_idx, tier) in pbar:
            ckpt_path = model_feature_dir / f"feature_{feature_idx}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    existing = json.load(f)
                if _needs_retry(existing, scorers):
                    pbar.set_postfix(feat=feature_idx, tier=tier[:6], status="retry")
                    contexts = all_contexts.get(
                        feature_idx, {"pos_contexts": [], "neg_contexts": []}
                    )
                    try:
                        existing = _fill_missing_scores(
                            result=existing,
                            contexts=contexts,
                            note_texts=note_texts,
                            tokenizer=tokenizer,
                            client=client,
                            model=current_model,
                            concordance_model=concordance_model,
                            r_pb=r_pb,
                            code_names=code_names,
                            code_descriptions=code_descriptions,
                            scorers=scorers,
                            n_contexts_train=n_contexts_train,
                            n_contexts_test=n_contexts_test,
                            context_window=max_tokens_context,
                        )
                        _write_json(existing, ckpt_path)
                    except Exception:
                        logger.exception(f"Error retrying failed scores for feature {feature_idx}")
                feature_results.append(existing)
                continue

            pbar.set_postfix(feat=feature_idx, tier=tier[:6])

            contexts = all_contexts.get(feature_idx, {"pos_contexts": [], "neg_contexts": []})

            try:
                result = _process_one_feature(
                    feature_idx=feature_idx,
                    tier=tier,
                    contexts=contexts,
                    note_texts=note_texts,
                    tokenizer=tokenizer,
                    client=client,
                    model=current_model,
                    concordance_model=concordance_model,
                    r_pb=r_pb,
                    code_names=code_names,
                    code_descriptions=code_descriptions,
                    scorers=scorers,
                    n_contexts_train=n_contexts_train,
                    n_contexts_test=n_contexts_test,
                    context_window=max_tokens_context,
                )
                _write_json(result, ckpt_path)
                feature_results.append(result)
            except Exception:
                logger.exception(f"Error processing feature {feature_idx}")
                n_errors += 1
                feature_results.append(_dead_result(feature_idx, tier, current_model, "error"))

            if _commit_volume is not None and (_idx + 1) % 50 == 0:
                _commit_volume()

        if _commit_volume is not None:
            _commit_volume()
        all_model_results[current_model] = feature_results
        if current_model == explainer_model:
            primary_n_errors = n_errors

    primary_results = all_model_results[explainer_model]

    assemble_catalog(primary_results, output_path, concordance_thresholds)

    if len(all_model_results) > 1:
        _write_model_comparison(all_model_results, output_path)

    elapsed = (datetime.now(UTC) - start_time).total_seconds()

    n_explained = sum(
        1
        for r in primary_results
        if r["explanation"] and not r["explanation"].startswith("Feature ")
    )
    n_scored_fuzzing = sum(1 for r in primary_results if r["fuzzing_score"] is not None)
    n_scored_detection = sum(1 for r in primary_results if r["detection_score"] is not None)
    n_concordance = sum(1 for r in primary_results if r["concordance_verdict"] is not None)
    n_features_with_parsing_errors = sum(
        1 for r in primary_results if r.get("parsing_errors", 0) > 0
    )

    summary = {
        "n_features": len(primary_results),
        "n_errors": primary_n_errors,
        "n_features_with_parsing_errors": n_features_with_parsing_errors,
        "tiers": {k: len(v) for k, v in tiers.items()},
        "explainer_model": explainer_model,
        "comparison_model": comparison_model,
        "models_run": models_to_run,
        "scorers": scorers,
        "api_call_counts": {
            "explain_categorize": n_explained,
            "fuzzing": n_scored_fuzzing,
            "detection": n_scored_detection,
            "concordance": n_concordance,
        },
        "elapsed_seconds": elapsed,
        "timestamp": start_time.isoformat(),
    }
    _write_json(summary, output_path / "run_summary.json")
    logger.info(f"Auto-interp complete: {len(primary_results)} features in {elapsed:.0f}s")

    return summary


def _write_model_comparison(
    all_model_results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Produce model_comparison.json comparing primary vs comparison model."""
    models = list(all_model_results.keys())
    if len(models) < 2:
        return

    comparison: dict = {}
    for model_name in models:
        results = all_model_results[model_name]
        fuzz = [r["fuzzing_score"] for r in results if r.get("fuzzing_score") is not None]
        det = [r["detection_score"] for r in results if r.get("detection_score") is not None]
        explanations = [r["explanation"] for r in results if r["explanation"]]
        conc = [r for r in results if r.get("concordance_verdict") in ("YES", "PARTIAL", "NO")]
        cats = [r["category"] for r in results]

        conc_yes = sum(1 for r in conc if r["concordance_verdict"] == "YES")
        conc_partial = sum(1 for r in conc if r["concordance_verdict"] == "PARTIAL")

        comparison[model_name] = {
            "mean_fuzzing": float(np.mean(fuzz)) if fuzz else None,
            "mean_detection": float(np.mean(det)) if det else None,
            "mean_explanation_length": (
                float(np.mean([len(e.split()) for e in explanations])) if explanations else 0
            ),
            "concordance_rate": ((conc_yes + conc_partial) / len(conc) if conc else None),
            "exact_match_rate": conc_yes / len(conc) if conc else None,
            "n_concordance": len(conc),
            "category_distribution": {c: sum(1 for x in cats if x == c) for c in set(cats)},
        }

    _write_json(comparison, output_dir / "model_comparison.json")
