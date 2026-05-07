"""
ICD-9 Clinical Grounding Pipeline for SAE Features
====================================================

Evaluates whether SAE latents correspond to structured clinical concepts
(ICD-9 codes) via point-biserial correlation with Benjamini-Hochberg
FDR correction.

Pipeline stages:
  1. Load trained SAE and encode eval activations → latent space
  2. Pool token-level latents to note-level vectors (max / mean / topk-mean)
  3. Align notes with ICD-9 binary labels via admission_id
  4. Compute point-biserial correlation (vectorised, no scipy loop)
  5. Apply BH FDR correction (q < 0.05)
  6. Report grounding metrics + save artefacts

Expected inputs:
  - Centered activation shards   (safetensors, float16/32)
  - metadata.jsonl                (one JSON object per note)
  - SAE checkpoint                (sae_weights.safetensors + sae_config.yaml)
  - ICD labels CSV                (admission_id + icd9_* one-hot columns)

References:
  - Gallifant et al. (EMNLP 2025) — pooling strategies
  - O'Neill et al. (2025) — domain SAE evaluation
  - Simon & Zou (PNAS 2025) — structured-label SAE evaluation (proteins)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import yaml
from safetensors.numpy import load_file as load_safetensors
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  SAE loading and encoding
# ---------------------------------------------------------------------------


@dataclass
class JumpReLUSAE:
    """Minimal JumpReLU SAE encoder (numpy-only, no torch dependency).

    Encoding:
        pre_acts = (x - b_dec) @ W_enc + b_enc
        z = pre_acts * (pre_acts > threshold)

    Adjust weight key names in `from_checkpoint` if your safetensors
    uses different naming conventions.
    """

    # ClassVar so cls.WEIGHT_KEY_MAP is accessible from the classmethod.
    WEIGHT_KEY_MAP: ClassVar[dict[str, list[str]]] = {
        "W_enc": ["W_enc", "encoder.weight", "w_enc"],
        "b_enc": ["b_enc", "encoder.bias"],
        "b_dec": ["b_dec", "decoder.bias"],
        # VanillaSAE checkpoints have no threshold (plain ReLU = JumpReLU at 0).
        # Aliases kept for JumpReLU-trained checkpoints.
        "threshold": ["threshold", "log_threshold", "theta"],
        # Expected shape [d_sae, d_model]. PyTorch nn.Linear stores [d_model, d_sae] — transpose before saving.
        "W_dec": ["W_dec", "decoder.weight", "w_dec"],
    }

    W_enc: np.ndarray  # [d_model, d_sae]
    b_enc: np.ndarray  # [d_sae]
    b_dec: np.ndarray  # [d_model]
    threshold: np.ndarray  # [d_sae]  zeros → plain ReLU behaviour
    d_model: int
    d_sae: int
    W_dec: np.ndarray | None = None  # [d_sae, d_model]; optional — needed for decode()

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> JumpReLUSAE:
        """Load SAE from a checkpoint directory.

        Expects:
          <checkpoint_dir>/sae_weights.safetensors
          <checkpoint_dir>/sae_config.yaml   (optional, for validation)
        """
        ckpt = Path(checkpoint_dir)
        weights_path = ckpt / "sae_weights.safetensors"
        config_path = ckpt / "sae_config.yaml"

        if not weights_path.exists():
            raise FileNotFoundError(f"SAE weights not found: {weights_path}")

        tensors = load_safetensors(str(weights_path))
        available = set(tensors.keys())
        logger.info(f"SAE weight keys found: {sorted(available)}")

        def _find(canonical: str) -> np.ndarray:
            for alias in cls.WEIGHT_KEY_MAP.get(canonical, [canonical]):
                if alias in tensors:
                    t = tensors[alias].astype(np.float32)
                    if alias == "log_threshold":
                        t = np.exp(t)
                        logger.info("Converted log_threshold → threshold via exp()")
                    return t
            raise KeyError(
                f"Could not find '{canonical}' in checkpoint. "
                f"Available keys: {sorted(available)}. "
                f"Edit WEIGHT_KEY_MAP if your naming differs."
            )

        W_enc = _find("W_enc")
        b_enc = _find("b_enc")
        b_dec = _find("b_dec")
        d_model, d_sae = W_enc.shape

        # VanillaSAE checkpoints (plain ReLU) have no threshold key.
        # ReLU == JumpReLU with threshold=0, so we default to zeros.
        try:
            threshold = _find("threshold")
        except KeyError:
            threshold = np.zeros(d_sae, dtype=np.float32)
            logger.info(
                "No threshold key in checkpoint — defaulting to 0.0 "
                "(VanillaSAE / plain ReLU behaviour)"
            )

        try:
            W_dec = _find("W_dec")
        except KeyError:
            W_dec = None
            logger.info("No W_dec in checkpoint — decode() unavailable")

        logger.info(f"SAE loaded: d_model={d_model}, d_sae={d_sae}")

        # Optional config validation
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg_d_sae = cfg.get("d_sae") or cfg.get("dict_size")
            if cfg_d_sae and cfg_d_sae != d_sae:
                logger.warning(f"Config d_sae={cfg_d_sae} != weight shape d_sae={d_sae}")

        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            b_dec=b_dec,
            threshold=threshold,
            d_model=d_model,
            d_sae=d_sae,
            W_dec=W_dec,
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode activations → SAE latents.

        Args:
            x: [batch, d_model] centered activations (float32).
        Returns:
            z: [batch, d_sae] sparse latent activations.
        """
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc  # [batch, d_sae]
        z = pre_acts * (pre_acts > self.threshold)  # JumpReLU
        return z

    def encode_chunked(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Memory-friendly chunked encoding for large activation matrices."""
        n = x.shape[0]
        if n <= chunk_size:
            return self.encode(x)
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunks.append(self.encode(x[start:end]))
        return np.concatenate(chunks, axis=0)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode SAE latents → reconstructed activations.

        Args:
            z: [batch, d_sae] sparse latent activations.
        Returns:
            x_hat: [batch, d_model] reconstructed activations.
        """
        assert (
            self.W_dec is not None
        ), "W_dec not loaded. Use from_huggingface() or a checkpoint that saves W_dec."
        return z @ self.W_dec + self.b_dec

    @classmethod
    def from_huggingface(cls, repo_id: str, filename: str, token: str | None = None) -> JumpReLUSAE:
        """Load a GemmaScope SAE from HuggingFace Hub.

        Downloads a .npz weights file and maps keys via WEIGHT_KEY_MAP.
        Logs all available keys on load — useful if key names differ.

        Args:
            repo_id:  HF repo, e.g. "google/gemma-scope-2b-pt-res".
            filename: Path within repo, e.g. "layer_16/width_16k/average_l0_71/params.npz".
            token:    HF token string; if None, reads HF_TOKEN from env automatically.
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        logger.info(f"Downloaded {repo_id}/{filename} → {local_path}")

        data = np.load(local_path)
        available = set(data.files)
        logger.info(f"GemmaScope weight keys: {sorted(available)}")

        def _find(canonical: str, required: bool = True) -> np.ndarray | None:
            for alias in cls.WEIGHT_KEY_MAP.get(canonical, [canonical]):
                if alias in data:
                    t = data[alias].astype(np.float32)
                    if alias == "log_threshold":
                        t = np.exp(t)
                        logger.info("Converted log_threshold → threshold via exp()")
                    return t
            if required:
                raise KeyError(
                    f"Could not find '{canonical}' in {repo_id}/{filename}. "
                    f"Available keys: {sorted(available)}"
                )
            return None

        W_enc = _find("W_enc")
        b_enc = _find("b_enc")
        b_dec = _find("b_dec")
        W_dec = _find("W_dec", required=False)
        d_model, d_sae = W_enc.shape

        if W_dec is not None and W_dec.shape != (d_sae, d_model):
            raise ValueError(
                f"W_dec shape {W_dec.shape} != expected ({d_sae}, {d_model}). "
                f"If using a PyTorch nn.Linear checkpoint, W_dec may need transposing."
            )

        threshold = _find("threshold", required=False)
        if threshold is None:
            threshold = np.zeros(d_sae, dtype=np.float32)
            logger.info("No threshold key — defaulting to 0.0 (plain ReLU)")

        logger.info(f"GemmaScope SAE loaded: d_model={d_model}, d_sae={d_sae}")
        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            b_dec=b_dec,
            threshold=threshold,
            d_model=d_model,
            d_sae=d_sae,
            W_dec=W_dec,
        )


# ---------------------------------------------------------------------------
# 2.  Activation loading + note-level pooling
# ---------------------------------------------------------------------------

PoolingStrategy = Literal["max", "mean", "topk_mean"]


def load_metadata(activations_dir: Path) -> pd.DataFrame:
    """Load metadata.jsonl → DataFrame with one row per note."""
    meta_path = activations_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {activations_dir}")
    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info(f"Loaded metadata: {len(df)} notes across {df['shard'].nunique()} shards")
    return df


def _pool_note(
    latents: np.ndarray,
    strategy: PoolingStrategy = "max",
    topk: int = 10,
) -> np.ndarray:
    """Pool token-level latents [n_tokens, d_sae] → note-level [d_sae].

    Strategies:
      max:       element-wise max across tokens (Gallifant et al. default)
      mean:      element-wise mean
      topk_mean: for each latent dim, average the top-k activations
    """
    if strategy == "max":
        return latents.max(axis=0)
    elif strategy == "mean":
        return latents.mean(axis=0)
    elif strategy == "topk_mean":
        k = min(topk, latents.shape[0])
        # Partial sort along axis=0 for each latent dimension
        topk_vals = np.partition(latents, -k, axis=0)[-k:]
        return topk_vals.mean(axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def encode_and_pool(
    sae: JumpReLUSAE,
    activations_dir: Path,
    metadata: pd.DataFrame,
    pooling: PoolingStrategy = "max",
    topk: int = 10,
    shard_filter: list[int] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Encode all eval activations through SAE and pool to note level.

    Args:
        sae: Loaded JumpReLU SAE.
        activations_dir: Directory with shard_*.safetensors.
        metadata: DataFrame from load_metadata().
        pooling: Pooling strategy.
        topk: k for topk_mean pooling.
        shard_filter: If given, only process these shard indices.
        checkpoint_dir: If set, save per-shard results here and skip shards
            whose checkpoint files already exist (enables resume).

    Returns:
        note_vectors: [num_notes, d_sae] note-level SAE activations.
        note_meta: Aligned metadata DataFrame (same row order).
    """
    if shard_filter is not None:
        metadata = metadata[metadata["shard"].isin(shard_filter)].copy()

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    # Load any already-completed shard checkpoints.
    all_vectors: list[np.ndarray] = []
    all_meta_rows: list[dict] = []
    done_shards: set[int] = set()

    if ckpt_dir is not None and ckpt_dir.exists():
        for vec_file in sorted(ckpt_dir.glob("shard_*_vectors.npy")):
            shard_num = int(vec_file.stem.split("_")[1])
            meta_file = ckpt_dir / f"shard_{shard_num:04d}_meta.jsonl"
            if not meta_file.exists():
                continue
            vecs = np.load(vec_file)  # [n_notes, d_sae]
            all_vectors.extend(list(vecs))
            with open(meta_file) as f:
                all_meta_rows.extend(json.loads(line) for line in f if line.strip())
            done_shards.add(shard_num)
        if done_shards:
            logger.info(
                f"Checkpoint: resumed from {len(done_shards)} shards "
                f"({len(all_vectors)} notes already encoded)"
            )

    # Group notes by shard for efficient I/O
    grouped = metadata.groupby("shard")

    for shard_idx, shard_notes in grouped:
        if shard_idx in done_shards:
            logger.info(f"Shard {shard_idx}: skipping (checkpoint exists)")
            continue

        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            logger.warning(f"Shard file not found, skipping: {shard_path}")
            continue

        logger.info(f"Processing shard {shard_idx}: {len(shard_notes)} notes")
        shard_data = load_safetensors(str(shard_path))
        # safetensors stores with a key — common keys are "activations",
        # "hidden_states", or just the first key
        act_key = next(iter(shard_data))
        shard_activations = shard_data[act_key].astype(np.float32)

        shard_vectors: list[np.ndarray] = []
        shard_meta: list[dict] = []

        for _, note_row in shard_notes.iterrows():
            row_start = int(note_row["row_start"])
            row_end = int(note_row["row_end"])

            note_acts = shard_activations[row_start:row_end]  # [n_tok, d_model]
            if note_acts.shape[0] == 0:
                logger.warning(
                    f"Empty activation slice for note_idx={note_row['note_idx']}, "
                    f"shard={shard_idx}, rows=[{row_start}:{row_end})"
                )
                continue

            # Encode through SAE
            latents = sae.encode_chunked(note_acts)  # [n_tok, d_sae]

            # Pool to note level
            note_vec = _pool_note(latents, strategy=pooling, topk=topk)
            shard_vectors.append(note_vec)
            shard_meta.append(
                {
                    k: (
                        int(v)
                        if isinstance(v, np.integer)
                        else float(v)
                        if isinstance(v, np.floating)
                        else v
                    )
                    for k, v in note_row.items()
                }
            )

        if not shard_vectors:
            continue

        all_vectors.extend(shard_vectors)
        all_meta_rows.extend(shard_meta)

        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.save(ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy", np.stack(shard_vectors))
            with open(ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl", "w") as f:
                for row in shard_meta:
                    f.write(json.dumps(row) + "\n")

    note_vectors = np.stack(all_vectors, axis=0)  # [num_notes, d_sae]
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)

    logger.info(f"Encoded {note_vectors.shape[0]} notes → " f"shape {note_vectors.shape}")
    return note_vectors, note_meta


def compute_diagnostic_metrics(
    sae: JumpReLUSAE,
    activations_dir: Path,
    metadata: pd.DataFrame,
    shard_filter: list[int] | None,
    output_dir: Path,
    chunk_size: int = 4096,
) -> dict:
    """Compute L0, explained variance, and dead latent fraction in a single pass.

    Streams shards chunk-by-chunk (chunk_size tokens at a time) to bound peak RAM.
    Requires sae.W_dec — use a SAE loaded via from_huggingface().

    Args:
        sae:             Loaded JumpReLUSAE with W_dec populated.
        activations_dir: Directory with shard_NNNN.safetensors files.
        metadata:        DataFrame from load_metadata().
        shard_filter:    If set, only process these shard indices.
        output_dir:      Where to write diagnostic_metrics.json.
        chunk_size:      Tokens per forward pass (default 4096).

    Returns:
        Dict with keys: n_tokens, mean_l0, explained_variance,
        dead_latent_frac, d_sae, d_model.
    """
    if shard_filter is not None:
        metadata = metadata[metadata["shard"].isin(shard_filter)]

    shards_to_process = sorted(metadata["shard"].unique())

    n_tokens = 0
    sum_x = np.zeros(sae.d_model, dtype=np.float64)
    sum_x2 = np.zeros(sae.d_model, dtype=np.float64)
    sum_resid = np.zeros(sae.d_model, dtype=np.float64)
    sum_resid2 = np.zeros(sae.d_model, dtype=np.float64)
    sum_l0: int = 0
    ever_active = np.zeros(sae.d_sae, dtype=bool)

    for shard_idx in shards_to_process:
        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            logger.warning(f"Shard not found, skipping: {shard_path}")
            continue

        shard_data = load_safetensors(str(shard_path))
        act_key = next(iter(shard_data))
        acts = shard_data[act_key].astype(np.float32)  # [n_tok, d_model]
        n_shard = acts.shape[0]

        for start in range(0, n_shard, chunk_size):
            x = acts[start : start + chunk_size]  # [chunk, d_model]
            z = sae.encode(x)  # [chunk, d_sae]
            x_hat = sae.decode(z)  # [chunk, d_model]
            resid = x - x_hat  # [chunk, d_model]

            n_tokens += x.shape[0]
            sum_x += x.sum(axis=0).astype(np.float64)
            sum_x2 += (x**2).sum(axis=0).astype(np.float64)
            sum_resid += resid.sum(axis=0).astype(np.float64)
            sum_resid2 += (resid**2).sum(axis=0).astype(np.float64)
            sum_l0 += int((z > 0).sum())
            ever_active |= (z > 0).any(axis=0)

        logger.info(f"Diagnostic: shard {shard_idx} done ({n_shard} tokens)")

    if n_tokens == 0:
        raise RuntimeError("No tokens processed — check shard_filter and activations_dir.")

    mean_x = sum_x / n_tokens
    var_x = (sum_x2 / n_tokens) - mean_x**2
    mean_resid = sum_resid / n_tokens
    var_resid = (sum_resid2 / n_tokens) - mean_resid**2

    total_var_x = float(var_x.sum())
    total_var_resid = float(var_resid.sum())
    ev = 1.0 - total_var_resid / total_var_x if total_var_x > 1e-12 else 0.0

    metrics = {
        "n_tokens": n_tokens,
        "mean_l0": round(float(sum_l0 / n_tokens), 4),
        "explained_variance": round(ev, 4),
        "dead_latent_frac": round(float((~ever_active).sum()) / sae.d_sae, 4),
        "d_sae": sae.d_sae,
        "d_model": sae.d_model,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "diagnostic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Diagnostic metrics: {metrics}")

    return metrics


# ---------------------------------------------------------------------------
# 3.  ICD label alignment
# ---------------------------------------------------------------------------


def load_and_align_icd_labels(
    icd_csv_path: Path,
    note_meta: pd.DataFrame,
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    icd_col_prefix: str = "icd9_",
    join_key: str = "admission_id",
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load ICD binary labels and align with note-level SAE vectors.

    Args:
        icd_csv_path:     CSV with admission_id + icd9_* columns.
        note_meta:        Metadata DataFrame (from encode_and_pool).
        min_prevalence:   Drop codes with prevalence < this fraction.
        max_codes:        Keep at most this many codes (by frequency).
        icd_col_prefix:   Column name prefix for ICD indicator columns.
        join_key:         Column to join metadata with ICD labels.

    Returns:
        icd_matrix:   [num_matched_notes, num_codes] binary int8 array.
        code_names:   List of ICD code column names (length = num_codes).
        matched_meta: Metadata for matched notes only.
    """
    icd_df = pd.read_csv(icd_csv_path)
    logger.info(f"Loaded ICD labels: {len(icd_df)} rows from {icd_csv_path}")

    # Identify ICD columns
    icd_cols = [c for c in icd_df.columns if c.startswith(icd_col_prefix)]
    if not icd_cols:
        raise ValueError(
            f"No columns with prefix '{icd_col_prefix}' found in {icd_csv_path}. "
            f"Available columns: {list(icd_df.columns[:20])}..."
        )
    logger.info(f"Found {len(icd_cols)} ICD indicator columns")

    # Ensure join key exists in both
    if join_key not in note_meta.columns:
        raise KeyError(
            f"Join key '{join_key}' not in note metadata columns: {list(note_meta.columns)}"
        )
    if join_key not in icd_df.columns:
        raise KeyError(f"Join key '{join_key}' not in ICD CSV columns: {list(icd_df.columns[:20])}")

    # Inner join on admission_id
    merged = note_meta.merge(
        icd_df[[join_key] + icd_cols],
        on=join_key,
        how="inner",
    )
    n_before = len(note_meta)
    n_after = len(merged)
    logger.info(
        f"Joined on '{join_key}': {n_after}/{n_before} notes matched "
        f"({n_before - n_after} unmatched, dropped)"
    )

    if n_after == 0:
        raise RuntimeError(
            f"Zero notes matched on '{join_key}'. Check that note metadata "
            f"and ICD CSV use the same ID format/type."
        )

    # Filter by prevalence
    icd_data = merged[icd_cols].values.astype(np.int8)
    prevalences = icd_data.mean(axis=0)
    mask_prev = prevalences >= min_prevalence
    logger.info(
        f"Prevalence filter (>= {min_prevalence}): " f"{mask_prev.sum()}/{len(icd_cols)} codes pass"
    )

    passing_cols = [c for c, m in zip(icd_cols, mask_prev, strict=False) if m]
    passing_prev = prevalences[mask_prev]

    # Keep top-N by frequency
    if len(passing_cols) > max_codes:
        top_idx = np.argsort(-passing_prev)[:max_codes]
        passing_cols = [passing_cols[i] for i in top_idx]
        logger.info(f"Capped to top {max_codes} codes by prevalence")

    # Sort by prevalence descending for readability
    col_prevs = {c: merged[c].mean() for c in passing_cols}
    passing_cols = sorted(passing_cols, key=lambda c: -col_prevs[c])

    icd_matrix = merged[passing_cols].values.astype(np.int8)
    code_names = passing_cols

    logger.info(f"Final ICD matrix: {icd_matrix.shape[0]} notes × " f"{icd_matrix.shape[1]} codes")

    # Log prevalence summary
    for i, code in enumerate(code_names[:10]):
        prev = icd_matrix[:, i].mean()
        logger.info(f"  {code}: prevalence = {prev:.3f} ({int(prev * len(merged))} notes)")
    if len(code_names) > 10:
        logger.info(f"  ... and {len(code_names) - 10} more codes")

    return icd_matrix, code_names, merged


# ---------------------------------------------------------------------------
# 4.  Vectorised point-biserial correlation
# ---------------------------------------------------------------------------


def compute_point_biserial_vectorised(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised point-biserial correlation between continuous X and binary Y.

    Point-biserial r is algebraically equivalent to Pearson r when one
    variable is dichotomous.  We compute it directly for speed.

    Args:
        X: [N, D] continuous matrix (note-level SAE activations).
        Y: [N, K] binary matrix (ICD indicators).

    Returns:
        r_pb:   [D, K] correlation matrix.
        p_vals: [D, K] two-tailed p-values (t-test, df = N-2).
    """
    N = X.shape[0]
    assert Y.shape[0] == N, f"Row mismatch: X has {N}, Y has {Y.shape[0]}"

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Counts
    n1 = Y.sum(axis=0)  # [K]
    n0 = N - n1  # [K]

    # Means of X grouped by Y
    # M1[d, k] = mean of X[:, d] where Y[:, k] == 1
    M1 = (X.T @ Y) / np.maximum(n1, 1)  # [D, K]
    M0 = (X.T @ (1 - Y)) / np.maximum(n0, 1)  # [D, K]

    # Std of X (pooled, entire sample)
    X_mean = X.mean(axis=0, keepdims=True)  # [1, D]
    X_var = ((X - X_mean) ** 2).sum(axis=0) / N  # [D]
    X_std = np.sqrt(X_var)  # [D]

    # Point-biserial formula
    # r_pb = (M1 - M0) / s_x * sqrt(n1 * n0 / N^2)
    r_pb = (M1 - M0) * np.sqrt(n1 * n0) / (X_std[:, None] * N)

    # Handle zero-variance latents (dead or constant)
    zero_var_mask = X_std < 1e-12
    r_pb[zero_var_mask, :] = 0.0

    # Two-tailed p-value via t-distribution
    # t = r * sqrt((N-2) / (1 - r^2))
    r_sq = np.clip(r_pb**2, 0, 1 - 1e-12)
    t_stat = r_pb * np.sqrt((N - 2) / (1 - r_sq))
    p_vals = 2 * scipy_stats.t.sf(np.abs(t_stat), df=N - 2)

    # Clamp p-values for numerical stability
    p_vals = np.clip(p_vals, 1e-300, 1.0)
    p_vals[zero_var_mask, :] = 1.0

    return r_pb.astype(np.float32), p_vals


def apply_bh_correction(
    p_vals: np.ndarray,
    q: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction on a matrix of p-values.

    Args:
        p_vals: [D, K] p-value matrix.
        q:      FDR threshold.

    Returns:
        reject:     [D, K] boolean matrix (True = significant).
        p_adjusted: [D, K] BH-adjusted p-values.
    """
    shape = p_vals.shape
    flat = p_vals.ravel()
    m = len(flat)

    # Sort
    sorted_idx = np.argsort(flat)
    sorted_p = flat[sorted_idx]

    # BH adjusted p-values
    ranks = np.arange(1, m + 1)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity (cumulative minimum from the end)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Unsort
    p_adjusted = np.empty_like(flat)
    p_adjusted[sorted_idx] = adjusted

    reject = p_adjusted <= q

    p_adjusted = p_adjusted.reshape(shape)
    reject = reject.reshape(shape)

    n_sig = reject.sum()
    n_tests = m
    logger.info(
        f"BH correction (q={q}): {n_sig}/{n_tests} tests significant "
        f"({n_sig / n_tests * 100:.2f}%)"
    )

    return reject, p_adjusted


# ---------------------------------------------------------------------------
# 5.  Grounding metrics
# ---------------------------------------------------------------------------


@dataclass
class GroundingResults:
    """Container for all grounding analysis outputs."""

    # Core matrices
    r_pb: np.ndarray  # [d_sae, n_codes] correlation matrix
    p_vals: np.ndarray  # [d_sae, n_codes] raw p-values
    p_adjusted: np.ndarray  # [d_sae, n_codes] BH-adjusted p-values
    significant: np.ndarray  # [d_sae, n_codes] boolean reject mask
    code_names: list[str]  # length n_codes

    # Summary metrics
    n_notes: int
    n_latents: int
    n_codes: int
    n_tests: int
    n_significant: int
    frac_significant: float

    # Grounding metrics
    grounded_latent_count: int  # latents with ≥1 sig. assoc at |r| > threshold
    grounded_latent_frac: float
    grounding_threshold: float  # |r_pb| threshold used

    # Per-latent and per-code summaries
    latent_max_abs_r: np.ndarray  # [d_sae] max |r_pb| across codes
    latent_n_associations: np.ndarray  # [d_sae] count of significant assocs
    code_n_grounded_latents: np.ndarray  # [n_codes] latents grounded to each code

    # Top associations
    top_associations: list[dict]  # sorted by |r_pb|, top-N

    def summary_dict(self) -> dict:
        return {
            "n_notes": self.n_notes,
            "n_latents": self.n_latents,
            "n_codes": self.n_codes,
            "n_tests": self.n_tests,
            "n_significant_after_bh": self.n_significant,
            "frac_significant": round(self.frac_significant, 6),
            "grounding_threshold_abs_r": self.grounding_threshold,
            "grounded_latent_count": self.grounded_latent_count,
            "grounded_latent_frac": round(self.grounded_latent_frac, 4),
            "mean_max_abs_r": round(float(self.latent_max_abs_r.mean()), 4),
            "median_max_abs_r": round(float(np.median(self.latent_max_abs_r)), 4),
            "top_10_associations": self.top_associations[:10],
        }


def compute_grounding(
    r_pb: np.ndarray,
    p_adjusted: np.ndarray,
    significant: np.ndarray,
    code_names: list[str],
    n_notes: int,
    r_threshold: float = 0.1,
    top_n: int = 200,
) -> GroundingResults:
    """Compute grounding metrics from correlation analysis."""

    d_sae, n_codes = r_pb.shape

    # Per-latent stats
    abs_r = np.abs(r_pb)
    latent_max_abs_r = abs_r.max(axis=1)  # [d_sae]
    latent_n_associations = significant.sum(axis=1)  # [d_sae]

    # A latent is "grounded" if it has ≥1 significant association
    # with |r_pb| > r_threshold
    grounded_mask = (significant & (abs_r > r_threshold)).any(axis=1)
    grounded_count = int(grounded_mask.sum())

    # Per-code stats
    code_grounded = (significant & (abs_r > r_threshold)).sum(axis=0)

    # Top associations (sorted by |r_pb|)
    sig_locs = np.argwhere(significant)  # [n_sig, 2]
    top_assocs = []
    for lat_idx, code_idx in sig_locs:
        top_assocs.append(
            {
                "latent": int(lat_idx),
                "code": code_names[code_idx],
                "r_pb": round(float(r_pb[lat_idx, code_idx]), 4),
                "abs_r": round(float(abs_r[lat_idx, code_idx]), 4),
                "p_adjusted": float(p_adjusted[lat_idx, code_idx]),
            }
        )
    top_assocs.sort(key=lambda x: -x["abs_r"])
    top_assocs = top_assocs[:top_n]

    return GroundingResults(
        r_pb=r_pb,
        p_vals=np.empty(0),  # raw p-vals can be recomputed; save memory
        p_adjusted=p_adjusted,
        significant=significant,
        code_names=code_names,
        n_notes=n_notes,
        n_latents=d_sae,
        n_codes=n_codes,
        n_tests=d_sae * n_codes,
        n_significant=int(significant.sum()),
        frac_significant=float(significant.sum()) / (d_sae * n_codes),
        grounded_latent_count=grounded_count,
        grounded_latent_frac=grounded_count / d_sae,
        grounding_threshold=r_threshold,
        latent_max_abs_r=latent_max_abs_r,
        latent_n_associations=latent_n_associations,
        code_n_grounded_latents=code_grounded,
        top_associations=top_assocs,
    )


# ---------------------------------------------------------------------------
# 6.  Save artefacts
# ---------------------------------------------------------------------------


def save_results(results: GroundingResults, output_dir: Path) -> None:
    """Save all grounding artefacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = results.summary_dict()
    with open(output_dir / "grounding_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved grounding_summary.json")

    # Full correlation matrix as npz
    np.savez_compressed(
        output_dir / "correlation_matrices.npz",
        r_pb=results.r_pb,
        p_adjusted=results.p_adjusted,
        significant=results.significant.astype(np.uint8),
    )
    logger.info("Saved correlation_matrices.npz")

    # Code names
    with open(output_dir / "code_names.json", "w") as f:
        json.dump(results.code_names, f, indent=2)

    # Top associations as CSV for easy inspection
    if results.top_associations:
        pd.DataFrame(results.top_associations).to_csv(
            output_dir / "top_associations.csv",
            index=False,
        )
        logger.info(f"Saved top_associations.csv ({len(results.top_associations)} rows)")

    # Per-code grounding summary
    code_summary = pd.DataFrame(
        {
            "code": results.code_names,
            "n_grounded_latents": results.code_n_grounded_latents,
        }
    )
    code_summary.to_csv(output_dir / "per_code_summary.csv", index=False)

    # Per-latent summary (only grounded latents, to keep file small)
    grounded_mask = results.latent_max_abs_r > results.grounding_threshold
    grounded_idx = np.where(grounded_mask)[0]
    if len(grounded_idx) > 0:
        latent_summary = pd.DataFrame(
            {
                "latent_idx": grounded_idx,
                "max_abs_r": results.latent_max_abs_r[grounded_idx],
                "n_associations": results.latent_n_associations[grounded_idx],
                "top_code": [
                    results.code_names[np.abs(results.r_pb[i]).argmax()]
                    if results.latent_n_associations[i] > 0
                    else "none"
                    for i in grounded_idx
                ],
                "top_code_r": [
                    float(results.r_pb[i][np.abs(results.r_pb[i]).argmax()]) for i in grounded_idx
                ],
            }
        )
        latent_summary.sort_values("max_abs_r", ascending=False, inplace=True)
        latent_summary.to_csv(output_dir / "grounded_latents.csv", index=False)
        logger.info(f"Saved grounded_latents.csv ({len(latent_summary)} latents)")

    logger.info(f"All artefacts saved to {output_dir}")


# ---------------------------------------------------------------------------
# 7.  Orchestrator — full pipeline
# ---------------------------------------------------------------------------


def run_icd_eval(
    activations_dir: str | Path,
    sae_checkpoint: str | Path,
    icd_csv_path: str | Path,
    output_dir: str | Path,
    pooling: PoolingStrategy = "max",
    topk: int = 10,
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    r_threshold: float = 0.1,
    fdr_q: float = 0.05,
    shard_filter: list[int] | None = None,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    checkpoint_dir: str | Path | None = None,
) -> GroundingResults:
    """Run the full ICD-9 clinical grounding evaluation.

    This is the main entry point. Call it from a script or notebook.
    checkpoint_dir: directory for per-shard encode checkpoints; defaults to
        output_dir/shard_ckpt so resume works automatically on re-run.
    """
    activations_dir = Path(activations_dir)
    sae_checkpoint = Path(sae_checkpoint)
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)

    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "shard_ckpt"

    logger.info("=" * 60)
    logger.info("ICD-9 Clinical Grounding Pipeline")
    logger.info("=" * 60)

    # Step 1: Load SAE
    logger.info("Step 1: Loading SAE...")
    sae = JumpReLUSAE.from_checkpoint(sae_checkpoint)

    # Step 2: Load metadata
    logger.info("Step 2: Loading metadata...")
    metadata = load_metadata(activations_dir)

    # Step 3: Encode activations + pool to note level
    logger.info(f"Step 3: Encoding + {pooling}-pooling to note level...")
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=activations_dir,
        metadata=metadata,
        pooling=pooling,
        topk=topk,
        shard_filter=shard_filter,
        checkpoint_dir=checkpoint_dir,
    )

    # Step 4: Align ICD labels
    logger.info("Step 4: Aligning ICD labels...")
    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
    )

    # Align note_vectors with matched notes
    # matched_meta is a subset of note_meta after the inner join
    # We need the indices into note_vectors that correspond to matched rows
    matched_indices = matched_meta.index.values
    if max(matched_indices) >= note_vectors.shape[0]:
        # The merge reindexed — rebuild alignment via note_idx
        note_meta_idx = note_meta["note_idx"].values
        matched_note_idx = matched_meta["note_idx"].values
        idx_map = {nidx: i for i, nidx in enumerate(note_meta_idx)}
        aligned_indices = [idx_map[nidx] for nidx in matched_note_idx]
        X = note_vectors[aligned_indices]
    else:
        X = note_vectors[matched_indices]

    logger.info(f"Aligned activation matrix: {X.shape}")

    # Step 5: Compute correlations
    logger.info("Step 5: Computing point-biserial correlations...")
    r_pb, p_vals = compute_point_biserial_vectorised(X, icd_matrix)
    logger.info(f"Correlation matrix shape: {r_pb.shape}, " f"max |r| = {np.abs(r_pb).max():.4f}")

    # Step 6: FDR correction
    logger.info(f"Step 6: BH FDR correction (q={fdr_q})...")
    significant, p_adjusted = apply_bh_correction(p_vals, q=fdr_q)

    # Step 7: Compute grounding metrics
    logger.info("Step 7: Computing grounding metrics...")
    results = compute_grounding(
        r_pb=r_pb,
        p_adjusted=p_adjusted,
        significant=significant,
        code_names=code_names,
        n_notes=X.shape[0],
        r_threshold=r_threshold,
    )

    # Store raw p_vals in results for completeness
    results.p_vals = p_vals

    # Step 8: Save
    logger.info("Step 8: Saving results...")
    save_results(results, output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    s = results.summary_dict()
    for k, v in s.items():
        if k != "top_10_associations":
            logger.info(f"  {k}: {v}")
    logger.info("Top 10 associations:")
    for assoc in s["top_10_associations"]:
        logger.info(
            f"  Latent {assoc['latent']:>5d} ↔ {assoc['code']:<20s} "
            f"r_pb={assoc['r_pb']:+.4f}  p_adj={assoc['p_adjusted']:.2e}"
        )
    logger.info("=" * 60)

    return results
