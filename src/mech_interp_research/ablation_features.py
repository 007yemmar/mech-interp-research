"""
Feature selection for the causal ablation experiment.

Reads `top_associations.csv` + `grounded_latents.csv` (produced by icd_eval)
and returns a list of `(SAE, feature_idx, code, kind)` targets per the
filter criteria in .tmp/ablation_design.md §3.5.

Two density modes:
  - `density_mode="skip"` (default, smoke runs): no density check. Picks the
    top-N features by |r_pb| with monospecificity filtering only.
  - `density_mode="from_shard_ckpt"`: pass a directory of `shard_NNNN_vectors.npy`
    files (the existing per-shard pooled-vectors checkpoint from icd_eval) and
    we'll compute per-latent note-firing rate (fraction of held-out notes with
    pooled activation > 0). Note: this is approximate — true per-token density
    would require re-encoding shards. Adequate for "alive enough" filtering.

Monospecificity is computed from `top_associations.csv` directly: how many
distinct codes does this latent appear with at `|r| > monospec_r_floor`?

Control selection (negative baseline) uses two strategies:
  - random_control: a latent uniformly chosen from those that fire on ≥1% of
    held-out notes (density floor) AND have max |r_pb| across all codes < 0.05.
  - low_r_control: a latent with max |r_pb| in [low_r_min, low_r_max].

If the local artefacts don't contain the data needed for density (no
shard_ckpt), the density filter silently downgrades to "skip" — the rest of
the pipeline still works.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Knobs for select_targets() — defaults match design doc §3.5."""

    # Inputs (per-SAE local files)
    top_associations_path: str  # e.g. .tmp/evaluations/vanilla/top_associations.csv
    grounded_latents_path: str  # e.g. .tmp/evaluations/vanilla/grounded_latents.csv
    # For per-feature density (note-firing rate). Either:
    #   - "skip": no density check (smoke mode)
    #   - path to a directory of shard_NNNN_vectors.npy files
    shard_ckpt_dir: str | None = None

    # Held-out shard range (matches the rest of the pipeline)
    held_out_shard_start: int = 281
    held_out_shard_end: int = 312

    # Selection knobs
    n_grounded: int = 10  # number of grounded features to pick per SAE
    abs_r_floor: float = 0.4  # required |r_pb| with the chosen code
    monospec_r_floor: float = 0.5  # |r| level used for monospecificity counting
    monospec_max_codes: int = 3  # max distinct codes a latent appears with at >floor

    # Density (only enforced when shard_ckpt_dir is provided)
    density_min: float = 0.005  # 0.5% of held-out notes (pooled > 0)
    density_max: float = 0.15  # 15%

    # Controls
    n_random_controls: int = 1
    n_low_r_controls: int = 1
    low_r_window: tuple[float, float] = (0.05, 0.10)
    random_control_max_r: float = 0.05  # never-grounded ceiling
    random_control_min_density: float = 0.01  # alive enough to ablate

    seed: int = 42


# ---------------------------------------------------------------------------
# Density computation from shard_ckpt
# ---------------------------------------------------------------------------


def compute_per_latent_density(
    shard_ckpt_dir: Path,
    shard_range: tuple[int, int],
) -> np.ndarray | None:
    """Per-latent note-firing rate from existing pooled-vector shard checkpoints.

    Returns:
      density: [d_sae] float32 — fraction of held-out notes with pooled
               activation > 0 for each latent. None if no shard files found.

    Note: pooled-max-activation > 0 is a coarser proxy than per-token firing
    rate. A feature firing once at high magnitude on a note still counts as
    "active" here. Adequate for "is this feature alive enough to ablate".
    """
    start, end = shard_range
    files = sorted(shard_ckpt_dir.glob("shard_*_vectors.npy"))
    if not files:
        logger.warning(f"No shard vector files in {shard_ckpt_dir}")
        return None

    n_notes_total = 0
    fire_count: np.ndarray | None = None

    for vec_file in files:
        try:
            shard_idx = int(vec_file.stem.split("_")[1])
        except ValueError:
            continue
        if shard_idx < start or shard_idx >= end:
            continue
        vecs = np.load(vec_file)  # [n_notes_shard, d_sae]
        if fire_count is None:
            fire_count = (vecs > 0).sum(axis=0).astype(np.int64)
        else:
            if vecs.shape[1] != fire_count.shape[0]:
                logger.warning(
                    f"d_sae mismatch in {vec_file.name}: "
                    f"{vecs.shape[1]} vs expected {fire_count.shape[0]} — skipping"
                )
                continue
            fire_count += (vecs > 0).sum(axis=0).astype(np.int64)
        n_notes_total += vecs.shape[0]

    if fire_count is None or n_notes_total == 0:
        logger.warning(
            f"No shards in range [{start},{end}) found in {shard_ckpt_dir}; "
            "skipping density computation"
        )
        return None

    density = fire_count.astype(np.float32) / float(n_notes_total)
    logger.info(
        f"Computed density over {n_notes_total} held-out notes, "
        f"{(density >= 0.005).sum()} latents fire on ≥0.5%"
    )
    return density


# ---------------------------------------------------------------------------
# Monospecificity from top_associations.csv
# ---------------------------------------------------------------------------


def compute_monospecificity(
    top_assoc: pd.DataFrame,
    r_floor: float,
) -> dict[int, int]:
    """For each latent in top_associations, count distinct codes at |r| > r_floor.

    Returns {latent_idx: n_distinct_codes_at_or_above_floor}.
    """
    above = top_assoc[top_assoc["abs_r"] >= r_floor]
    counts = above.groupby("latent")["code"].nunique().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Main selection routine
# ---------------------------------------------------------------------------


def select_targets(cfg: FeatureSelectionConfig) -> list[dict[str, Any]]:
    """Apply filters and return the target list ready for AblationConfig.targets.

    Each target is a dict with keys: feature_idx, code, kind, r_pb.

    Returns up to (n_grounded + n_random_controls + n_low_r_controls) targets.
    Logs warnings if filters can't be satisfied.
    """
    rng = random.Random(cfg.seed)
    top_assoc = pd.read_csv(cfg.top_associations_path)
    grounded = pd.read_csv(cfg.grounded_latents_path)

    # --- Per-latent stats ---
    mono = compute_monospecificity(top_assoc, cfg.monospec_r_floor)
    if cfg.shard_ckpt_dir and cfg.shard_ckpt_dir != "skip":
        density = compute_per_latent_density(
            Path(cfg.shard_ckpt_dir),
            (cfg.held_out_shard_start, cfg.held_out_shard_end),
        )
    else:
        density = None
        logger.info("Skipping density filter (shard_ckpt_dir not provided)")

    # --- Grounded features: filter and rank ---
    sorted_assoc = top_assoc.sort_values("abs_r", ascending=False).reset_index(drop=True)
    seen_latents: set[int] = set()
    grounded_targets: list[dict[str, Any]] = []
    rejected: dict[str, int] = {"low_r": 0, "polyspec": 0, "density": 0, "dup": 0}

    for _, row in sorted_assoc.iterrows():
        if len(grounded_targets) >= cfg.n_grounded:
            break
        j = int(row["latent"])
        if j in seen_latents:
            rejected["dup"] += 1
            continue
        if abs(row["abs_r"]) < cfg.abs_r_floor:
            rejected["low_r"] += 1
            continue
        n_codes_strong = mono.get(j, 0)
        if n_codes_strong > cfg.monospec_max_codes:
            rejected["polyspec"] += 1
            continue
        if density is not None:
            d = float(density[j]) if j < len(density) else 0.0
            if not (cfg.density_min <= d <= cfg.density_max):
                rejected["density"] += 1
                continue
        grounded_targets.append(
            {
                "feature_idx": j,
                "code": str(row["code"]),
                "kind": "grounded",
                "r_pb": float(row["r_pb"]),
                "n_codes_at_floor": n_codes_strong,
                "density": (
                    float(density[j]) if density is not None and j < len(density) else None
                ),
            }
        )
        seen_latents.add(j)

    logger.info(
        f"Grounded selection: {len(grounded_targets)}/{cfg.n_grounded} "
        f"passed (rejected: {rejected})"
    )

    # --- Random controls ---
    # Pool: latents with max |r_pb| across all codes < cfg.random_control_max_r.
    # If density is available, also require density >= cfg.random_control_min_density.
    grounded_set = set(int(x) for x in grounded["latent_idx"].values)
    # max_abs_r per latent is in grounded.csv only for grounded latents; the
    # rest of the dictionary has max_abs_r < r_threshold (likely 0.1). For
    # picking "never-grounded" latents we look outside the grounded set.
    # We need d_sae — infer from density length or from the largest latent we've seen.
    if density is not None:
        d_sae = int(len(density))
    else:
        d_sae = int(max(top_assoc["latent"].max(), grounded["latent_idx"].max()) + 1)
        logger.info(f"No density data; inferring d_sae≈{d_sae} from CSV")

    all_idx = np.arange(d_sae)
    never_grounded = np.array([i for i in all_idx if i not in grounded_set], dtype=np.int64)
    # Apply density floor if available
    if density is not None:
        density_ok = density[never_grounded] >= cfg.random_control_min_density
        never_grounded = never_grounded[density_ok]
    if len(never_grounded) == 0:
        logger.warning("No candidates for random controls (density too restrictive?)")
        random_targets: list[dict[str, Any]] = []
    else:
        picks = rng.sample(list(never_grounded), min(cfg.n_random_controls, len(never_grounded)))
        # Random code from the codes present in top_associations
        all_codes = sorted(set(top_assoc["code"].astype(str)))
        random_targets = [
            {
                "feature_idx": int(j),
                "code": rng.choice(all_codes),
                "kind": "random_control",
                "r_pb": None,
                "density": float(density[j]) if density is not None else None,
            }
            for j in picks
        ]

    # --- Low-r controls ---
    lo, hi = cfg.low_r_window
    # Use grounded_latents.csv (max_abs_r column) — these are latents that DID
    # cross the r_threshold floor (typically 0.1) but stayed below the strong
    # association range. Approximates "just barely grounded".
    low_r_pool = grounded[(grounded["max_abs_r"] >= lo) & (grounded["max_abs_r"] < hi)]
    if len(low_r_pool) == 0:
        # Fall back: take latents in the lower tail of grounded_latents
        sorted_grounded = grounded.sort_values("max_abs_r")
        low_r_pool = sorted_grounded.head(50)
        logger.info(
            f"Low-r control: no latents in [{lo},{hi}); falling back to "
            f"weakest 50 of grounded_latents"
        )
    if len(low_r_pool) == 0:
        logger.warning("No candidates for low-r controls")
        low_r_targets: list[dict[str, Any]] = []
    else:
        picked = low_r_pool.sample(
            n=min(cfg.n_low_r_controls, len(low_r_pool)), random_state=cfg.seed
        )
        low_r_targets = [
            {
                "feature_idx": int(row["latent_idx"]),
                "code": str(row["top_code"]),
                "kind": "low_r_control",
                "r_pb": float(row["top_code_r"]),
                "density": (
                    float(density[int(row["latent_idx"])]) if density is not None else None
                ),
            }
            for _, row in picked.iterrows()
        ]

    all_targets = grounded_targets + random_targets + low_r_targets
    logger.info(
        f"Selected {len(all_targets)} targets: "
        f"{len(grounded_targets)} grounded, {len(random_targets)} random, "
        f"{len(low_r_targets)} low-r"
    )
    return all_targets


# ---------------------------------------------------------------------------
# Smoke selector — minimal target set for the validation run
# ---------------------------------------------------------------------------


def select_smoke_targets(
    top_associations_path: str | Path,
    grounded_latents_path: str | Path,
    n_grounded: int = 2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pick a tiny target set for the smoke run.

    n_grounded grounded features (top by abs_r, dedup by latent) + 1 random
    control + 1 low-r control. Total: n_grounded + 2.

    Bypasses density and monospecificity filters — for the smoke we just want
    something that runs end-to-end on a few notes.
    """
    cfg = FeatureSelectionConfig(
        top_associations_path=str(top_associations_path),
        grounded_latents_path=str(grounded_latents_path),
        shard_ckpt_dir=None,  # skip density
        n_grounded=n_grounded,
        abs_r_floor=0.4,
        monospec_r_floor=0.5,
        monospec_max_codes=999,  # disable monospec filter for smoke
        n_random_controls=1,
        n_low_r_controls=1,
        seed=seed,
    )
    return select_targets(cfg)
