"""Latent feature inspector: token-level evidence for SAE-ICD associations.

For each top latent-code pair, identifies the tokens that maximally
activate the latent, maps them back to clinical note text with context
windows, and produces structured reports for publication.

Two-pass algorithm:
  Pass 1: Scan sampled activation shards through the SAE to find
          globally top-k token positions per target latent.
  Pass 2: Re-tokenize only the notes containing top hits to extract
          decoded token strings and context windows.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.numpy import load_file as load_safetensors

from mech_interp_research.icd_eval import (
    JumpReLUSAE,
    load_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenHit:
    """A single token that strongly activates a latent."""

    shard: int
    position_in_shard: int
    note_idx: int
    position_in_note: int
    activation: float
    token_id: int | None = None
    token_str: str | None = None
    context_tokens: list[str] | None = None
    context_token_ids: list[int] | None = None
    context_activations: list[float] | None = None


@dataclass
class LatentReport:
    """Complete inspection report for one latent-code pair."""

    latent_idx: int
    icd_code: str
    r_pb: float
    top_tokens: list[TokenHit] = field(default_factory=list)

    n_firings_icd_pos: int = 0
    n_tokens_icd_pos: int = 0
    n_firings_icd_neg: int = 0
    n_tokens_icd_neg: int = 0
    firing_rate_icd_pos: float = 0.0
    firing_rate_icd_neg: float = 0.0
    firing_rate_ratio: float = 0.0
    mean_activation_icd_pos: float = 0.0
    mean_activation_icd_neg: float = 0.0

    n_unique_tokens: int = 0
    unique_token_strings: list[str] = field(default_factory=list)
    n_unique_notes: int = 0
    diversity_score: float = 0.0
    top_token_frequency: list[tuple[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Load target latent-code pairs
# ---------------------------------------------------------------------------


def load_target_latents(
    top_associations_path: Path,
    n_pairs: int = 20,
) -> list[dict[str, Any]]:
    """Load top N latent-code pairs from top_associations.csv.

    Deduplicates by latent index, keeping the pair with highest |r_pb|
    for each latent (so we inspect each latent for its strongest code).
    """
    df = pd.read_csv(top_associations_path)
    df = df.sort_values("abs_r", ascending=False)
    df = df.drop_duplicates(subset="latent", keep="first")
    df = df.head(n_pairs).reset_index(drop=True)

    pairs = []
    for _, row in df.iterrows():
        pairs.append(
            {
                "latent": int(row["latent"]),
                "code": str(row["code"]),
                "r_pb": float(row["r_pb"]),
                "abs_r": float(row["abs_r"]),
            }
        )

    logger.info(f"Loaded {len(pairs)} target latent-code pairs")
    return pairs


# ---------------------------------------------------------------------------
# Shard selection from shard_ckpt
# ---------------------------------------------------------------------------


def select_target_shards(
    shard_ckpt_dir: Path,
    target_latent_indices: list[int],
    n_shards: int = 20,
) -> list[int]:
    """Select shards most likely to contain high-activation tokens.

    For each target latent, finds notes with highest max-pooled activation
    from existing shard_ckpt note vectors, then returns the union of
    shards containing those notes.
    """
    shard_ckpt_dir = Path(shard_ckpt_dir)
    if not shard_ckpt_dir.exists():
        raise FileNotFoundError(
            f"shard_ckpt directory not found: {shard_ckpt_dir}. "
            f"Run icd_eval.py first to generate per-shard checkpoints."
        )

    vec_files = sorted(shard_ckpt_dir.glob("shard_*_vectors.npy"))
    if not vec_files:
        raise FileNotFoundError(f"No shard vector files in {shard_ckpt_dir}")

    shard_scores: Counter[int] = Counter()

    for vec_file in vec_files:
        shard_num = int(vec_file.stem.split("_")[1])
        vecs = np.load(vec_file)  # [n_notes, d_sae]

        for lat_idx in target_latent_indices:
            if lat_idx < vecs.shape[1]:
                max_act = float(vecs[:, lat_idx].max())
                shard_scores[shard_num] += max_act

    selected = [s for s, _ in shard_scores.most_common(n_shards)]
    selected.sort()
    logger.info(f"Selected {len(selected)} shards from shard_ckpt: {selected[:10]}...")
    return selected


# ---------------------------------------------------------------------------
# Pass 1: Scan shards for top-k tokens
# ---------------------------------------------------------------------------


def scan_shard_for_top_tokens(
    sae: JumpReLUSAE,
    shard_path: Path,
    shard_idx: int,
    metadata_for_shard: pd.DataFrame,
    target_latent_indices: list[int],
    top_k: int = 50,
    icd_labels_by_note: dict[int, dict[str, int]] | None = None,
    target_codes: dict[int, str] | None = None,
) -> tuple[dict[int, list[TokenHit]], dict[int, dict[str, Any]]]:
    """Scan one shard: encode tokens, collect top-k per target latent.

    Memory-efficient: encodes in chunks of 4096 and retains only
    target latent columns (~40 MB vs 34 GB for full output).

    Returns:
        hits: Dict mapping latent_idx -> list of TokenHit
        firing_stats: Dict mapping latent_idx -> firing stat accumulators
    """
    shard_data = load_safetensors(str(shard_path))
    act_key = next(iter(shard_data))
    acts = shard_data[act_key].astype(np.float32)
    n_tokens = acts.shape[0]

    n_targets = len(target_latent_indices)
    target_indices_arr = np.array(target_latent_indices)

    # Encode in chunks, retaining only target columns
    chunk_size = 4096
    target_acts = np.empty((n_tokens, n_targets), dtype=np.float32)
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        chunk_full = sae.encode(acts[start:end])
        target_acts[start:end] = chunk_full[:, target_indices_arr]

    # Build note lookup: for each token position, which note does it belong to?
    note_boundaries = []
    for _, row in metadata_for_shard.iterrows():
        note_boundaries.append((int(row["row_start"]), int(row["row_end"]), int(row["note_idx"])))
    note_boundaries.sort(key=lambda x: x[0])

    def _find_note(pos: int) -> tuple[int, int]:
        """Return (note_idx, position_in_note) for absolute shard position."""
        for rs, re, nidx in note_boundaries:
            if rs <= pos < re:
                return nidx, pos - rs
        return -1, -1

    # Find top-k per target latent
    hits: dict[int, list[TokenHit]] = {}
    for ti, lat_idx in enumerate(target_latent_indices):
        col = target_acts[:, ti]
        k = min(top_k, n_tokens)
        if k <= 0:
            hits[lat_idx] = []
            continue

        top_positions = np.argpartition(col, -k)[-k:]
        top_positions = top_positions[np.argsort(col[top_positions])[::-1]]

        lat_hits = []
        for pos in top_positions:
            act_val = float(col[int(pos)])
            if act_val <= 0:
                continue
            note_idx, pos_in_note = _find_note(int(pos))
            if note_idx == -1:
                continue
            lat_hits.append(
                TokenHit(
                    shard=shard_idx,
                    position_in_shard=int(pos),
                    note_idx=note_idx,
                    position_in_note=pos_in_note,
                    activation=act_val,
                )
            )
        hits[lat_idx] = lat_hits

    # Compute firing statistics per note
    firing_stats: dict[int, dict[str, Any]] = {}
    if icd_labels_by_note is not None and target_codes is not None:
        for ti, lat_idx in enumerate(target_latent_indices):
            code = target_codes.get(lat_idx)
            if code is None:
                continue
            stats: dict[str, Any] = {
                "n_firings_pos": 0,
                "n_tokens_pos": 0,
                "sum_act_pos": 0.0,
                "n_firings_neg": 0,
                "n_tokens_neg": 0,
                "sum_act_neg": 0.0,
            }
            for rs, re, nidx in note_boundaries:
                labels = icd_labels_by_note.get(nidx)
                if labels is None:
                    continue
                is_positive = labels.get(code, 0) == 1
                note_col = target_acts[rs:re, ti]
                n_fire = int((note_col > 0).sum())
                s_act = float(note_col[note_col > 0].sum()) if n_fire > 0 else 0.0
                n_tok = re - rs

                if is_positive:
                    stats["n_firings_pos"] += n_fire
                    stats["n_tokens_pos"] += n_tok
                    stats["sum_act_pos"] += s_act
                else:
                    stats["n_firings_neg"] += n_fire
                    stats["n_tokens_neg"] += n_tok
                    stats["sum_act_neg"] += s_act

            firing_stats[lat_idx] = stats

    return hits, firing_stats


def merge_top_tokens(
    per_shard_hits: list[dict[int, list[TokenHit]]],
    top_k: int = 50,
) -> dict[int, list[TokenHit]]:
    """Merge per-shard top-k lists into a global top-k per latent."""
    merged: dict[int, list[TokenHit]] = {}

    all_latents: set[int] = set()
    for shard_hits in per_shard_hits:
        all_latents.update(shard_hits.keys())

    for lat_idx in sorted(all_latents):
        combined = []
        for shard_hits in per_shard_hits:
            combined.extend(shard_hits.get(lat_idx, []))
        combined.sort(key=lambda h: h.activation, reverse=True)
        merged[lat_idx] = combined[:top_k]

    return merged


def aggregate_firing_statistics(
    per_shard_stats: list[dict[int, dict[str, Any]]],
) -> dict[int, dict[str, float]]:
    """Sum per-shard firing stats into final rates and ratios."""
    aggregated: dict[int, dict[str, Any]] = {}

    for shard_stats in per_shard_stats:
        for lat_idx, stats in shard_stats.items():
            if lat_idx not in aggregated:
                aggregated[lat_idx] = {
                    "n_firings_pos": 0,
                    "n_tokens_pos": 0,
                    "sum_act_pos": 0.0,
                    "n_firings_neg": 0,
                    "n_tokens_neg": 0,
                    "sum_act_neg": 0.0,
                }
            for key in aggregated[lat_idx]:
                aggregated[lat_idx][key] += stats.get(key, 0)

    result: dict[int, dict[str, float]] = {}
    for lat_idx, agg in aggregated.items():
        rate_pos = agg["n_firings_pos"] / max(agg["n_tokens_pos"], 1)
        rate_neg = agg["n_firings_neg"] / max(agg["n_tokens_neg"], 1)
        result[lat_idx] = {
            "n_firings_pos": agg["n_firings_pos"],
            "n_tokens_pos": agg["n_tokens_pos"],
            "firing_rate_pos": rate_pos,
            "mean_activation_pos": (agg["sum_act_pos"] / max(agg["n_firings_pos"], 1)),
            "n_firings_neg": agg["n_firings_neg"],
            "n_tokens_neg": agg["n_tokens_neg"],
            "firing_rate_neg": rate_neg,
            "mean_activation_neg": (agg["sum_act_neg"] / max(agg["n_firings_neg"], 1)),
            "firing_rate_ratio": rate_pos / max(rate_neg, 1e-10),
        }

    return result


# ---------------------------------------------------------------------------
# Pass 2: Re-tokenize notes and extract context
# ---------------------------------------------------------------------------


def load_tokenizer(
    model_name: str = "google/gemma-2-2b",
    hf_token: str | None = None,
) -> Any:
    """Load only the tokenizer (no model weights)."""
    from transformers import AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    logger.info(f"Loaded tokenizer: {model_name}")
    return tokenizer


def extract_token_contexts(
    top_tokens: dict[int, list[TokenHit]],
    note_texts: dict[int, str],
    tokenizer: Any,
    context_window: int = 5,
    max_length: int = 8192,
    metadata: pd.DataFrame | None = None,
) -> dict[int, list[TokenHit]]:
    """Re-tokenize notes and fill in token strings + context for each hit.

    Modifies TokenHit objects in place and returns the same dict.
    """
    unique_note_ids = set()
    for hits in top_tokens.values():
        for h in hits:
            unique_note_ids.add(h.note_idx)

    logger.info(f"Re-tokenizing {len(unique_note_ids)} unique notes for context extraction")

    # Build metadata lookup for alignment verification
    meta_ntokens: dict[int, int] = {}
    if metadata is not None:
        for _, row in metadata.iterrows():
            meta_ntokens[int(row["note_idx"])] = int(row["n_tokens"])

    tokenized_cache: dict[int, list[int]] = {}
    alignment_warnings = 0

    for note_idx in unique_note_ids:
        text = note_texts.get(note_idx)
        if text is None:
            continue
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        token_ids = encoded["input_ids"]

        # Verify alignment with stored metadata
        if note_idx in meta_ntokens:
            expected = meta_ntokens[note_idx]
            if len(token_ids) != expected:
                alignment_warnings += 1
                if alignment_warnings <= 5:
                    logger.warning(
                        f"Token alignment mismatch for note {note_idx}: "
                        f"re-tokenized={len(token_ids)}, metadata={expected}"
                    )
        tokenized_cache[note_idx] = token_ids

    if alignment_warnings > 0:
        logger.warning(f"Total token alignment mismatches: {alignment_warnings}")

    for hits in top_tokens.values():
        for h in hits:
            token_ids = tokenized_cache.get(h.note_idx)
            if token_ids is None:
                continue

            pos = h.position_in_note
            if pos < 0 or pos >= len(token_ids):
                continue

            ctx_start = max(0, pos - context_window)
            ctx_end = min(len(token_ids), pos + context_window + 1)

            h.token_id = token_ids[pos]
            h.token_str = tokenizer.decode([token_ids[pos]], skip_special_tokens=False)
            h.context_token_ids = token_ids[ctx_start:ctx_end]
            h.context_tokens = [
                tokenizer.decode([tid], skip_special_tokens=False) for tid in h.context_token_ids
            ]

    return top_tokens


# ---------------------------------------------------------------------------
# Context diversity analysis
# ---------------------------------------------------------------------------


def assess_context_diversity(
    top_tokens: dict[int, list[TokenHit]],
) -> dict[int, dict[str, Any]]:
    """Analyze diversity of top-activating tokens per latent."""
    diversity: dict[int, dict[str, Any]] = {}

    for lat_idx, hits in top_tokens.items():
        filled = [h for h in hits if h.token_str is not None]
        if not filled:
            diversity[lat_idx] = {
                "n_unique_tokens": 0,
                "unique_token_strings": [],
                "n_unique_notes": 0,
                "diversity_score": 0.0,
                "top_token_frequency": [],
            }
            continue

        token_counter = Counter(h.token_str.strip() for h in filled)
        unique_notes = len({h.note_idx for h in filled})

        diversity[lat_idx] = {
            "n_unique_tokens": len(token_counter),
            "unique_token_strings": sorted(token_counter.keys()),
            "n_unique_notes": unique_notes,
            "diversity_score": len(token_counter) / len(filled),
            "top_token_frequency": token_counter.most_common(20),
        }

    return diversity


# ---------------------------------------------------------------------------
# Report building and serialization
# ---------------------------------------------------------------------------


def build_report(
    target_pairs: list[dict[str, Any]],
    top_tokens: dict[int, list[TokenHit]],
    firing_stats: dict[int, dict[str, float]],
    diversity: dict[int, dict[str, Any]],
) -> list[LatentReport]:
    """Assemble LatentReport objects from all analysis outputs."""
    reports = []

    for pair in target_pairs:
        lat_idx = pair["latent"]
        fstats = firing_stats.get(lat_idx, {})
        div = diversity.get(lat_idx, {})

        report = LatentReport(
            latent_idx=lat_idx,
            icd_code=pair["code"],
            r_pb=pair["r_pb"],
            top_tokens=top_tokens.get(lat_idx, []),
            n_firings_icd_pos=int(fstats.get("n_firings_pos", 0)),
            n_tokens_icd_pos=int(fstats.get("n_tokens_pos", 0)),
            n_firings_icd_neg=int(fstats.get("n_firings_neg", 0)),
            n_tokens_icd_neg=int(fstats.get("n_tokens_neg", 0)),
            firing_rate_icd_pos=float(fstats.get("firing_rate_pos", 0)),
            firing_rate_icd_neg=float(fstats.get("firing_rate_neg", 0)),
            firing_rate_ratio=float(fstats.get("firing_rate_ratio", 0)),
            mean_activation_icd_pos=float(fstats.get("mean_activation_pos", 0)),
            mean_activation_icd_neg=float(fstats.get("mean_activation_neg", 0)),
            n_unique_tokens=int(div.get("n_unique_tokens", 0)),
            unique_token_strings=div.get("unique_token_strings", []),
            n_unique_notes=int(div.get("n_unique_notes", 0)),
            diversity_score=float(div.get("diversity_score", 0)),
            top_token_frequency=div.get("top_token_frequency", []),
        )
        reports.append(report)

    return reports


def serialize_report(
    reports: list[LatentReport],
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Write feature_inspection_report.json and feature_inspection_details.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON
    latents_json = []
    csv_rows = []

    for report in reports:
        tokens_json = []
        for rank, h in enumerate(report.top_tokens, 1):
            token_entry = {
                "rank": rank,
                "activation": round(h.activation, 4),
                "token_str": h.token_str,
                "token_id": h.token_id,
                "note_idx": h.note_idx,
                "position_in_note": h.position_in_note,
                "shard": h.shard,
                "context": h.context_tokens,
                "context_token_ids": h.context_token_ids,
                "context_activations": (
                    [round(a, 4) for a in h.context_activations] if h.context_activations else None
                ),
            }
            tokens_json.append(token_entry)

            csv_rows.append(
                {
                    "latent_idx": report.latent_idx,
                    "icd_code": report.icd_code,
                    "r_pb": round(report.r_pb, 4),
                    "rank": rank,
                    "activation": round(h.activation, 4),
                    "token_str": h.token_str,
                    "token_id": h.token_id,
                    "note_idx": h.note_idx,
                    "position_in_note": h.position_in_note,
                    "context": (" ".join(h.context_tokens) if h.context_tokens else ""),
                    "firing_rate_pos": round(report.firing_rate_icd_pos, 6),
                    "firing_rate_neg": round(report.firing_rate_icd_neg, 6),
                    "firing_rate_ratio": round(report.firing_rate_ratio, 2),
                    "diversity_score": round(report.diversity_score, 4),
                }
            )

        latent_entry = {
            "latent_idx": report.latent_idx,
            "icd_code": report.icd_code,
            "r_pb": round(report.r_pb, 4),
            "firing_stats": {
                "n_firings_icd_pos": report.n_firings_icd_pos,
                "n_tokens_icd_pos": report.n_tokens_icd_pos,
                "firing_rate_icd_pos": round(report.firing_rate_icd_pos, 6),
                "n_firings_icd_neg": report.n_firings_icd_neg,
                "n_tokens_icd_neg": report.n_tokens_icd_neg,
                "firing_rate_icd_neg": round(report.firing_rate_icd_neg, 6),
                "firing_rate_ratio": round(report.firing_rate_ratio, 2),
                "mean_activation_icd_pos": round(report.mean_activation_icd_pos, 4),
                "mean_activation_icd_neg": round(report.mean_activation_icd_neg, 4),
            },
            "diversity": {
                "n_unique_tokens": report.n_unique_tokens,
                "n_unique_notes": report.n_unique_notes,
                "diversity_score": round(report.diversity_score, 4),
                "top_token_frequency": [
                    {"token": t, "count": c} for t, c in report.top_token_frequency
                ],
            },
            "top_tokens": tokens_json,
        }
        latents_json.append(latent_entry)

    # Summary stats
    firing_ratios = [r.firing_rate_ratio for r in reports if r.firing_rate_ratio > 0]
    diversity_scores = [r.diversity_score for r in reports if r.diversity_score > 0]

    output = {
        "pipeline_version": "1.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config or {},
        "summary": {
            "n_latents_inspected": len(reports),
            "mean_diversity_score": (
                round(float(np.mean(diversity_scores)), 4) if diversity_scores else 0.0
            ),
            "mean_firing_rate_ratio": (
                round(float(np.mean(firing_ratios)), 2) if firing_ratios else 0.0
            ),
        },
        "latents": latents_json,
    }

    json_path = output_dir / "feature_inspection_report.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Wrote {json_path}")

    csv_path = output_dir / "feature_inspection_details.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_feature_inspection(
    activations_dir: str | Path,
    sae_checkpoint: str | Path,
    icd_csv_path: str | Path,
    eval_output_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "google/gemma-2-2b",
    n_pairs: int = 20,
    top_k: int = 50,
    context_window: int = 5,
    n_shards: int = 20,
    max_length: int = 8192,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    text_col: str = "note_text",
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    min_notes: int = 100,
    shard_ckpt_dir: str | Path | None = None,
    rng_seed: int = 42,
    hf_token: str | None = None,
    on_shard_complete: Any = None,
) -> dict[str, Any]:
    """Run the full feature inspection pipeline."""
    activations_dir = Path(activations_dir)
    eval_output_dir = Path(eval_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load target latent-code pairs
    top_assoc_path = eval_output_dir / "top_associations.csv"
    target_pairs = load_target_latents(top_assoc_path, n_pairs)
    target_latent_indices = [p["latent"] for p in target_pairs]
    target_codes = {p["latent"]: p["code"] for p in target_pairs}

    logger.info(f"Inspecting {len(target_pairs)} latent-code pairs")

    # 2. Load SAE
    if isinstance(sae_checkpoint, (str | Path)):
        sae = JumpReLUSAE.from_checkpoint(sae_checkpoint)
    else:
        sae = sae_checkpoint

    # 3. Load metadata
    metadata = load_metadata(activations_dir)

    # 4. Select target shards
    # Sources audited outside icd_eval keep their pooled vectors elsewhere (the
    # necessity suite writes shard_ckpt_audit/ beside the run, not inside the
    # eval dir), so allow an explicit override. None preserves the original
    # eval_output_dir/shard_ckpt layout exactly.
    shard_ckpt_dir = Path(shard_ckpt_dir) if shard_ckpt_dir else eval_output_dir / "shard_ckpt"
    selected_shards = select_target_shards(shard_ckpt_dir, target_latent_indices, n_shards)
    logger.info(f"Selected {len(selected_shards)} shards for scanning")

    # 5. Build ICD labels lookup for firing stats
    icd_labels_by_note: dict[int, dict[str, int]] = {}
    try:
        icd_df = pd.read_csv(icd_csv_path)
        icd_cols = [
            c
            for c in icd_df.columns
            if c.startswith(icd_col_prefix) and pd.api.types.is_numeric_dtype(icd_df[c])
        ]
        # Join with metadata to get note_idx
        meta_for_join = metadata[[join_key, "note_idx"]].drop_duplicates()
        merged = meta_for_join.merge(icd_df[[join_key] + icd_cols], on=join_key, how="inner")
        for _, row in merged.iterrows():
            nidx = int(row["note_idx"])
            icd_labels_by_note[nidx] = {c: int(row[c]) for c in icd_cols}
        logger.info(f"Built ICD labels for {len(icd_labels_by_note)} notes")
    except Exception:
        logger.warning("Could not load ICD labels for firing stats", exc_info=True)

    # 6. Pass 1 — Scan shards
    all_shard_hits: list[dict[int, list[TokenHit]]] = []
    all_shard_stats: list[dict[int, dict[str, Any]]] = []

    for i, shard_idx in enumerate(selected_shards):
        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            logger.warning(f"Shard file not found: {shard_path}, skipping")
            continue

        meta_for_shard = metadata[metadata["shard"] == shard_idx]
        if meta_for_shard.empty:
            continue

        logger.info(
            f"Scanning shard {shard_idx} ({i + 1}/{len(selected_shards)}, "
            f"{len(meta_for_shard)} notes)"
        )

        hits, firing_stats = scan_shard_for_top_tokens(
            sae=sae,
            shard_path=shard_path,
            shard_idx=shard_idx,
            metadata_for_shard=meta_for_shard,
            target_latent_indices=target_latent_indices,
            top_k=top_k,
            icd_labels_by_note=icd_labels_by_note,
            target_codes=target_codes,
        )

        all_shard_hits.append(hits)
        all_shard_stats.append(firing_stats)

        if on_shard_complete is not None:
            on_shard_complete(shard_idx)

    # 7. Merge global top-k
    global_top_tokens = merge_top_tokens(all_shard_hits, top_k)
    logger.info(
        f"Global top-k: {sum(len(v) for v in global_top_tokens.values())} total hits "
        f"across {len(global_top_tokens)} latents"
    )

    # 8. Aggregate firing stats
    agg_firing_stats = aggregate_firing_statistics(all_shard_stats)

    # 9. Pass 2 — Re-tokenize and extract contexts
    unique_note_ids = set()
    for hits in global_top_tokens.values():
        for h in hits:
            unique_note_ids.add(h.note_idx)

    # Load note texts for matched notes
    note_texts: dict[int, str] = {}
    try:
        icd_df_text = pd.read_csv(icd_csv_path, usecols=[join_key, text_col])
        meta_notes = metadata[metadata["note_idx"].isin(unique_note_ids)][
            [join_key, "note_idx"]
        ].drop_duplicates()
        text_merged = meta_notes.merge(icd_df_text, on=join_key, how="inner")
        for _, row in text_merged.iterrows():
            note_texts[int(row["note_idx"])] = str(row[text_col])
        logger.info(f"Loaded text for {len(note_texts)} notes")
    except Exception:
        logger.warning("Could not load note texts for context extraction", exc_info=True)

    if note_texts:
        tokenizer = load_tokenizer(model_name, hf_token)
        extract_token_contexts(
            global_top_tokens,
            note_texts,
            tokenizer,
            context_window=context_window,
            max_length=max_length,
            metadata=metadata,
        )

    # 10. Diversity analysis
    diversity = assess_context_diversity(global_top_tokens)

    # 11. Build and serialize report
    reports = build_report(target_pairs, global_top_tokens, agg_firing_stats, diversity)

    run_config = {
        "n_pairs": n_pairs,
        "top_k": top_k,
        "context_window": context_window,
        "n_shards_sampled": len(selected_shards),
        "shards_used": selected_shards,
        "model_name": model_name,
        "sae_checkpoint": str(sae_checkpoint),
        "activations_dir": str(activations_dir),
        "eval_output_dir": str(eval_output_dir),
    }

    serialize_report(reports, output_dir, config=run_config)

    # Summary for return
    firing_ratios = [r.firing_rate_ratio for r in reports if r.firing_rate_ratio > 0]
    diversity_scores = [r.diversity_score for r in reports if r.diversity_score > 0]

    summary = {
        "n_latents_inspected": len(reports),
        "n_shards_scanned": len(selected_shards),
        "total_token_hits": sum(len(r.top_tokens) for r in reports),
        "mean_diversity_score": (
            round(float(np.mean(diversity_scores)), 4) if diversity_scores else 0.0
        ),
        "mean_firing_rate_ratio": (
            round(float(np.mean(firing_ratios)), 2) if firing_ratios else 0.0
        ),
        "output_dir": str(output_dir),
    }

    logger.info("=" * 60)
    logger.info("FEATURE INSPECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Latents inspected:  {summary['n_latents_inspected']}")
    logger.info(f"  Shards scanned:     {summary['n_shards_scanned']}")
    logger.info(f"  Total token hits:   {summary['total_token_hits']}")
    logger.info(f"  Mean diversity:     {summary['mean_diversity_score']}")
    logger.info(f"  Mean firing ratio:  {summary['mean_firing_rate_ratio']}")
    logger.info("=" * 60)

    return summary
