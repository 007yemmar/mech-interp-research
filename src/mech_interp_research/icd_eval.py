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
import os
from collections.abc import Callable
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

    Two encoder conventions are supported via ``subtract_b_dec``:

    subtract_b_dec=True  (default — our VanillaSAE / custom JumpReLU checkpoints):
        pre_acts = (x - b_dec) @ W_enc + b_enc
        z = pre_acts * (pre_acts > threshold)

    subtract_b_dec=False (GemmaScope and other externally-trained SAEs):
        pre_acts = x @ W_enc + b_enc
        z = pre_acts * (pre_acts > threshold)

    GemmaScope SAEs (Lieberum et al. 2024) were trained without the b_dec
    subtraction in the encoder.  Loading them with subtract_b_dec=True shifts
    every pre-activation by −b_dec @ W_enc (mean ≈ −3.2, std ≈ 3.6 for the
    16k width model), which corrupts feature selection and yields EV ≈ −6.
    ``from_huggingface`` sets subtract_b_dec=False automatically.
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
    subtract_b_dec: bool = (
        True  # False for GemmaScope; True for our VanillaSAE/JumpReLU checkpoints
    )

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
            x: [batch, d_model] activations (float32).
        Returns:
            z: [batch, d_sae] sparse latent activations.
        """
        x_in = (x - self.b_dec) if self.subtract_b_dec else x
        pre_acts = x_in @ self.W_enc + self.b_enc  # [batch, d_sae]
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
            subtract_b_dec=False,  # GemmaScope encoder: x @ W_enc + b_enc (no b_dec subtraction)
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
    on_shard_complete: Callable[[int], None] | None = None,
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
        on_shard_complete: Optional callback invoked with the shard index
            after a shard's checkpoint files are atomically written to
            ``checkpoint_dir``. The Modal entrypoints pass
            ``lambda _: artifacts_volume.commit()`` here so each shard is
            durably synced to the remote volume before the next starts —
            otherwise a hard container crash, OOM, or network partition
            would lose every shard written since the last commit.

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
            with open(meta_file) as f:
                meta_rows = [json.loads(line) for line in f if line.strip()]
            # Integrity check — partial writes can leave .npy and .jsonl
            # with mismatched row counts. Treat as not-done and re-encode.
            if vecs.shape[0] != len(meta_rows):
                logger.warning(
                    f"Checkpoint shard {shard_num}: vectors={vecs.shape[0]} "
                    f"!= metadata rows={len(meta_rows)}. Discarding partial "
                    f"checkpoint and re-encoding."
                )
                vec_file.unlink()
                meta_file.unlink()
                continue
            all_vectors.extend(list(vecs))
            all_meta_rows.extend(meta_rows)
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
            # Atomic two-step: write metadata to a temp file, then save the
            # vectors, then rename metadata into place. The metadata rename
            # is the commit point — if either preceding write is interrupted,
            # resume sees no .jsonl and re-encodes the shard.
            meta_path = ckpt_dir / f"shard_{shard_idx:04d}_meta.jsonl"
            meta_tmp = meta_path.with_suffix(".jsonl.tmp")
            with open(meta_tmp, "w") as f:
                for row in shard_meta:
                    f.write(json.dumps(row) + "\n")
            np.save(ckpt_dir / f"shard_{shard_idx:04d}_vectors.npy", np.stack(shard_vectors))
            os.replace(meta_tmp, meta_path)

            if on_shard_complete is not None:
                on_shard_complete(int(shard_idx))

    note_vectors = np.stack(all_vectors, axis=0)  # [num_notes, d_sae]
    note_meta = pd.DataFrame(all_meta_rows).reset_index(drop=True)

    logger.info(f"Encoded {note_vectors.shape[0]} notes → shape {note_vectors.shape}")
    return note_vectors, note_meta


def compute_diagnostic_metrics(
    sae: JumpReLUSAE,
    activations_dir: Path,
    metadata: pd.DataFrame,
    shard_filter: list[int] | None,
    output_dir: Path,
    chunk_size: int = 4096,
    on_shard_complete: Callable[[int], None] | None = None,
) -> dict:
    """Compute L0, explained variance, and dead latent fraction in a single pass.

    Streams shards chunk-by-chunk (chunk_size tokens at a time) to bound peak RAM.
    Requires sae.W_dec — use a SAE loaded via from_huggingface().

    Checkpoints accumulator state to ``output_dir/diag_ckpt.npz`` after each
    shard.  On re-run, already-processed shards are skipped automatically.
    The checkpoint is removed once ``diagnostic_metrics.json`` is written.

    Args:
        sae:               Loaded JumpReLUSAE with W_dec populated.
        activations_dir:   Directory with shard_NNNN.safetensors files.
        metadata:          DataFrame from load_metadata().
        shard_filter:      If set, only process these shard indices.
        output_dir:        Where to write diagnostic_metrics.json.
        chunk_size:        Tokens per forward pass (default 4096).
        on_shard_complete: Optional callback after each shard checkpoint is
            written.  The Modal entrypoint passes
            ``lambda _: artifacts_volume.commit()`` here.

    Returns:
        Dict with keys: n_tokens, mean_l0, explained_variance,
        dead_latent_frac, d_sae, d_model.
    """
    if sae.W_dec is None:
        raise ValueError(
            "compute_diagnostic_metrics requires sae.W_dec for reconstruction. "
            "Load with from_huggingface() or a checkpoint that includes W_dec."
        )

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
    done_shards: set[int] = set()

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "diag_ckpt.npz"

    if ckpt_path.exists():
        ckpt = np.load(ckpt_path, allow_pickle=False)
        n_tokens = int(ckpt["n_tokens"])
        sum_x = ckpt["sum_x"]
        sum_x2 = ckpt["sum_x2"]
        sum_resid = ckpt["sum_resid"]
        sum_resid2 = ckpt["sum_resid2"]
        sum_l0 = int(ckpt["sum_l0"])
        ever_active = ckpt["ever_active"].astype(bool)
        done_shards = set(ckpt["done_shards"].tolist())
        logger.info(
            f"Diagnostic checkpoint: resumed from {len(done_shards)} shards "
            f"({n_tokens} tokens already processed)"
        )

    for shard_idx in shards_to_process:
        if shard_idx in done_shards:
            logger.info(f"Diagnostic: shard {shard_idx} skipping (checkpoint exists)")
            continue

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

        done_shards.add(shard_idx)

        ckpt_tmp = ckpt_path.parent / "diag_ckpt_tmp"
        np.savez(
            ckpt_tmp,
            n_tokens=np.array(n_tokens),
            sum_x=sum_x,
            sum_x2=sum_x2,
            sum_resid=sum_resid,
            sum_resid2=sum_resid2,
            sum_l0=np.array(sum_l0),
            ever_active=ever_active,
            done_shards=np.array(sorted(done_shards)),
        )
        # np.savez appends .npz to the path
        os.replace(str(ckpt_tmp) + ".npz", ckpt_path)

        if on_shard_complete is not None:
            on_shard_complete(int(shard_idx))

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

    with open(output_dir / "diagnostic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()
    logger.info(f"Diagnostic metrics: {metrics}")

    return metrics


def compute_domain_shift_analysis(
    sae: JumpReLUSAE,
    activations_dir: Path,
    metadata: pd.DataFrame,
    clinical_mean: np.ndarray,
    output_dir: Path,
    shard_filter: list[int] | None = None,
    n_shards_sample: int = 5,
    chunk_size: int = 4096,
) -> dict:
    """Diagnose domain shift between GemmaScope's training distribution and clinical data.

    Runs three analyses on a sample of shards:
      1. Mean comparison — clinical mean vs SAE's b_dec (norm, cosine, per-dim).
      2. Scale comparison — clinical activation norms vs threshold distribution.
      3. Re-centered EV — two variants to disentangle mean shift from feature mismatch:
         a. mean-only fix: replace b_dec with clinical_mean in decode only.
         b. full re-centering: center input before encode, use clinical_mean in decode.

    If variant (a) recovers EV >> 0, the negative EV was primarily mean shift.
    If both stay negative, GemmaScope's feature directions don't span clinical subspace.

    Args:
        sae:             Loaded GemmaScope SAE (subtract_b_dec=False, W_dec required).
        activations_dir: Raw (non-centered) activation shards.
        metadata:        DataFrame from load_metadata().
        clinical_mean:   [d_model] float32 mean vector from centering step.
        output_dir:      Where to write domain_shift_analysis.json.
        shard_filter:    If set, sample from these shards only.
        n_shards_sample: Number of shards to evaluate (default 5 for speed).
        chunk_size:      Tokens per forward pass.
    """
    if sae.W_dec is None:
        raise ValueError("domain shift analysis requires sae.W_dec for reconstruction.")

    clinical_mean = clinical_mean.astype(np.float32)
    b_dec = sae.b_dec.astype(np.float32)

    # --- 1. Mean comparison ---
    diff = clinical_mean - b_dec
    clinical_norm = float(np.linalg.norm(clinical_mean))
    bdec_norm = float(np.linalg.norm(b_dec))
    diff_norm = float(np.linalg.norm(diff))
    cos_sim = float(np.dot(clinical_mean, b_dec) / (clinical_norm * bdec_norm + 1e-12))
    mean_comparison = {
        "clinical_mean_norm": round(clinical_norm, 4),
        "b_dec_norm": round(bdec_norm, 4),
        "difference_norm": round(diff_norm, 4),
        "relative_difference": round(diff_norm / (clinical_norm + 1e-12), 4),
        "cosine_similarity": round(cos_sim, 6),
    }
    logger.info(f"Mean comparison: {mean_comparison}")

    # --- 2 & 3. Stream through shards for scale stats + re-centered EV ---
    if shard_filter is not None:
        metadata = metadata[metadata["shard"].isin(shard_filter)]
    shards_available = sorted(metadata["shard"].unique())
    shards_to_use = shards_available[:n_shards_sample]
    logger.info(
        f"Domain shift analysis: sampling {len(shards_to_use)} shards "
        f"from {len(shards_available)} available"
    )

    n_tokens = 0
    # accumulators for standard EV
    sum_x2_std = np.zeros(sae.d_model, dtype=np.float64)
    sum_r2_std = np.zeros(sae.d_model, dtype=np.float64)
    sum_x_std = np.zeros(sae.d_model, dtype=np.float64)
    sum_r_std = np.zeros(sae.d_model, dtype=np.float64)
    # accumulators for mean-only fix (replace b_dec with clinical_mean in decode)
    sum_r2_mfix = np.zeros(sae.d_model, dtype=np.float64)
    sum_r_mfix = np.zeros(sae.d_model, dtype=np.float64)
    # accumulators for full re-centering (center input + replace b_dec)
    sum_r2_full = np.zeros(sae.d_model, dtype=np.float64)
    sum_r_full = np.zeros(sae.d_model, dtype=np.float64)
    # scale stats
    sum_act_norms: float = 0.0
    sum_act_norms_sq: float = 0.0
    sum_preact_abs = np.zeros(sae.d_sae, dtype=np.float64)

    for shard_idx in shards_to_use:
        shard_path = activations_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            continue
        shard_data = load_safetensors(str(shard_path))
        acts = shard_data[next(iter(shard_data))].astype(np.float32)

        for start in range(0, acts.shape[0], chunk_size):
            x = acts[start : start + chunk_size]
            n = x.shape[0]
            n_tokens += n

            # activation norms
            norms = np.linalg.norm(x, axis=1)
            sum_act_norms += float(norms.sum())
            sum_act_norms_sq += float((norms**2).sum())

            # --- standard forward (GemmaScope as-is) ---
            z_std = sae.encode(x)
            x_hat_std = sae.decode(z_std)
            r_std = x - x_hat_std

            # pre-activation scale (for threshold comparison)
            pre_act = x @ sae.W_enc + sae.b_enc
            sum_preact_abs += np.abs(pre_act).sum(axis=0).astype(np.float64)

            # EV accumulators — standard
            sum_x_std += x.sum(axis=0).astype(np.float64)
            sum_x2_std += (x**2).sum(axis=0).astype(np.float64)
            sum_r_std += r_std.sum(axis=0).astype(np.float64)
            sum_r2_std += (r_std**2).sum(axis=0).astype(np.float64)

            # --- mean-only fix: same z, decode with clinical_mean instead of b_dec ---
            x_hat_mfix = z_std @ sae.W_dec + clinical_mean
            r_mfix = x - x_hat_mfix
            sum_r_mfix += r_mfix.sum(axis=0).astype(np.float64)
            sum_r2_mfix += (r_mfix**2).sum(axis=0).astype(np.float64)

            # --- full re-centering: center input, encode, decode with clinical_mean ---
            x_centered = x - clinical_mean
            z_full = sae.encode(x_centered)
            x_hat_full = z_full @ sae.W_dec + clinical_mean
            r_full = x - x_hat_full
            sum_r_full += r_full.sum(axis=0).astype(np.float64)
            sum_r2_full += (r_full**2).sum(axis=0).astype(np.float64)

    if n_tokens == 0:
        raise RuntimeError("No tokens processed for domain shift analysis.")

    def _ev(sum_x, sum_x2, sum_r, sum_r2):
        mean_x = sum_x / n_tokens
        var_x = (sum_x2 / n_tokens) - mean_x**2
        mean_r = sum_r / n_tokens
        var_r = (sum_r2 / n_tokens) - mean_r**2
        total_var_x = float(var_x.sum())
        total_var_r = float(var_r.sum())
        return round(1.0 - total_var_r / total_var_x if total_var_x > 1e-12 else 0.0, 4)

    ev_standard = _ev(sum_x_std, sum_x2_std, sum_r_std, sum_r2_std)
    ev_mean_fix = _ev(sum_x_std, sum_x2_std, sum_r_mfix, sum_r2_mfix)
    ev_full_recenter = _ev(sum_x_std, sum_x2_std, sum_r_full, sum_r2_full)

    # Scale stats
    mean_act_norm = sum_act_norms / n_tokens
    std_act_norm = (sum_act_norms_sq / n_tokens - mean_act_norm**2) ** 0.5
    mean_preact_abs = sum_preact_abs / n_tokens
    threshold = sae.threshold

    scale_comparison = {
        "clinical_activation_norm_mean": round(float(mean_act_norm), 4),
        "clinical_activation_norm_std": round(float(std_act_norm), 4),
        "preactivation_abs_mean": round(float(mean_preact_abs.mean()), 4),
        "preactivation_abs_std": round(float(mean_preact_abs.std()), 4),
        "threshold_mean": round(float(threshold.mean()), 4),
        "threshold_std": round(float(threshold.std()), 4),
        "threshold_min": round(float(threshold.min()), 4),
        "threshold_max": round(float(threshold.max()), 4),
        "frac_preact_above_threshold": round(float((mean_preact_abs > threshold).mean()), 4),
    }

    result = {
        "n_tokens_sampled": n_tokens,
        "n_shards_sampled": len(shards_to_use),
        "mean_comparison": mean_comparison,
        "scale_comparison": scale_comparison,
        "ev_standard": ev_standard,
        "ev_mean_fix_decode_only": ev_mean_fix,
        "ev_full_recentered": ev_full_recenter,
        "interpretation": {
            "if_mean_fix_recovers_ev": (
                "Negative EV is primarily mean shift (b_dec mismatch), "
                "not feature direction mismatch."
            ),
            "if_both_stay_negative": (
                "GemmaScope's learned feature directions do not span "
                "the clinical activation subspace — genuine domain mismatch."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "domain_shift_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Domain shift analysis: {json.dumps(result, indent=2)}")

    return result


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
    min_notes: int = 100,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load ICD binary labels and align with note-level SAE vectors.

    Args:
        icd_csv_path:     CSV with admission_id + icd9_* columns.
        note_meta:        Metadata DataFrame (from encode_and_pool).
        min_prevalence:   Drop codes with prevalence < this fraction.
        max_codes:        Keep at most this many codes (by frequency).
        icd_col_prefix:   Column name prefix for ICD indicator columns.
        join_key:         Column to join metadata with ICD labels.
        min_notes:        Minimum matched notes required. Raises RuntimeError
            if the join produces fewer matches (prevents meaningless
            correlations on tiny samples).

    Returns:
        icd_matrix:   [num_matched_notes, num_codes] binary int8 array.
        code_names:   List of ICD code column names (length = num_codes).
        matched_meta: Metadata for matched notes only.
    """
    icd_df = pd.read_csv(icd_csv_path)
    logger.info(f"Loaded ICD labels: {len(icd_df)} rows from {icd_csv_path}")

    # Identify ICD binary indicator columns (skip non-numeric like icd9_codes_list)
    icd_cols = [
        c
        for c in icd_df.columns
        if c.startswith(icd_col_prefix) and pd.api.types.is_numeric_dtype(icd_df[c])
    ]
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

    if n_after < min_notes:
        raise RuntimeError(
            f"Only {n_after} notes matched on '{join_key}' (minimum {min_notes}). "
            f"ICD CSV has {icd_df[join_key].nunique()} unique IDs vs "
            f"{note_meta[join_key].nunique()} in note metadata. "
            f"Check that the ICD CSV covers the same population as the "
            f"extracted activations (e.g. sample_50k.csv, not train.csv)."
        )

    # Filter by prevalence
    icd_data = merged[icd_cols].values.astype(np.int8)
    prevalences = icd_data.mean(axis=0)
    mask_prev = prevalences >= min_prevalence
    logger.info(
        f"Prevalence filter (>= {min_prevalence}): {mask_prev.sum()}/{len(icd_cols)} codes pass"
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

    logger.info(f"Final ICD matrix: {icd_matrix.shape[0]} notes × {icd_matrix.shape[1]} codes")

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
        f"BH correction (q={q}): {n_sig}/{n_tests} tests significant ({n_sig / n_tests * 100:.2f}%)"
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
# 7.  Post-hoc analysis helpers
# ---------------------------------------------------------------------------


def reassemble_note_vectors(
    checkpoint_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reload note-level SAE vectors from per-shard encode checkpoints.

    Loads shard_NNNN_vectors.npy and shard_NNNN_meta.jsonl files written
    by encode_and_pool, concatenates them in shard order.
    """
    checkpoint_dir = Path(checkpoint_dir)
    all_vectors: list[np.ndarray] = []
    all_meta: list[dict] = []

    for vec_file in sorted(checkpoint_dir.glob("shard_*_vectors.npy")):
        shard_num = int(vec_file.stem.split("_")[1])
        meta_file = checkpoint_dir / f"shard_{shard_num:04d}_meta.jsonl"
        if not meta_file.exists():
            logger.warning(f"Missing metadata for {vec_file.name}, skipping")
            continue

        vecs = np.load(vec_file)
        with open(meta_file) as f:
            meta_rows = [json.loads(line) for line in f if line.strip()]

        if vecs.shape[0] != len(meta_rows):
            logger.warning(
                f"Shard {shard_num}: vectors={vecs.shape[0]} != meta={len(meta_rows)}, skipping"
            )
            continue

        all_vectors.append(vecs)
        all_meta.extend(meta_rows)

    if not all_vectors:
        raise RuntimeError(f"No valid shard checkpoints in {checkpoint_dir}")

    note_vectors = np.concatenate(all_vectors, axis=0)
    note_meta = pd.DataFrame(all_meta).reset_index(drop=True)
    logger.info(f"Reassembled {note_vectors.shape[0]} notes from {len(all_vectors)} shards")
    return note_vectors, note_meta


def load_saved_correlations(output_dir: str | Path) -> dict:
    """Load correlation matrices and code names from a previous eval run."""
    output_dir = Path(output_dir)

    data = np.load(output_dir / "correlation_matrices.npz")
    r_pb = data["r_pb"]
    p_adjusted = data["p_adjusted"]
    significant = data["significant"].astype(bool)

    with open(output_dir / "code_names.json") as f:
        code_names = json.load(f)

    with open(output_dir / "grounding_summary.json") as f:
        summary = json.load(f)

    return {
        "r_pb": r_pb,
        "p_adjusted": p_adjusted,
        "significant": significant,
        "code_names": code_names,
        "n_notes": summary["n_notes"],
    }


def compute_partial_point_biserial(
    X: np.ndarray,
    Y: np.ndarray,
    confound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-biserial correlation after residualizing X on confound(s).

    Removes the linear effect of ``confound`` from each column of X via OLS,
    then computes point-biserial between the residuals and binary Y.  P-values
    use df = N - 2 - C (C = number of confound columns).

    Args:
        X: [N, D] continuous matrix (note-level SAE activations).
        Y: [N, K] binary matrix (ICD indicators).
        confound: [N] or [N, C] confound variable(s) to partial out.

    Returns:
        r_partial: [D, K] partial correlation matrix.
        p_vals:    [D, K] two-tailed p-values.
    """
    N = X.shape[0]
    X64 = X.astype(np.float64)
    confound = np.asarray(confound, dtype=np.float64)
    if confound.ndim == 1:
        confound = confound[:, None]
    n_confounds = confound.shape[1]

    Z = np.column_stack([np.ones(N), confound])
    beta, _, _, _ = np.linalg.lstsq(Z, X64, rcond=None)
    X_resid = X64 - Z @ beta

    r_partial, _ = compute_point_biserial_vectorised(X_resid, Y)
    r_partial = r_partial.astype(np.float64)

    df = N - 2 - n_confounds
    if df < 1:
        raise ValueError(f"Not enough observations: N={N}, confounds={n_confounds}, df={df}")
    r_sq = np.clip(r_partial**2, 0, 1 - 1e-12)
    t_stat = r_partial * np.sqrt(df / (1 - r_sq))
    p_vals = 2 * scipy_stats.t.sf(np.abs(t_stat), df=df)
    p_vals = np.clip(p_vals, 1e-300, 1.0)

    X_resid_std = X_resid.std(axis=0)
    zero_mask = X_resid_std < 1e-12
    r_partial[zero_mask, :] = 0.0
    p_vals[zero_mask, :] = 1.0

    return r_partial.astype(np.float32), p_vals


def compute_monospecificity(
    r_pb: np.ndarray,
    significant: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Count per-latent code associations at various |r| thresholds.

    Returns one dict per threshold with grounded/mono/oligo/poly counts
    and a histogram of associations-per-latent.
    """
    abs_r = np.abs(r_pb)
    d_sae = r_pb.shape[0]
    results = []

    for thresh in thresholds:
        mask = significant & (abs_r > thresh)
        n_assoc = mask.sum(axis=1)  # [d_sae]

        n_grounded = int((n_assoc >= 1).sum())
        n_mono = int((n_assoc == 1).sum())
        n_oligo = int(((n_assoc >= 2) & (n_assoc <= 3)).sum())
        n_poly = int((n_assoc >= 4).sum())

        max_assoc = int(n_assoc.max()) if n_grounded > 0 else 0
        histogram: dict[str, int] = {}
        for n in range(max(max_assoc + 1, 1)):
            count = int((n_assoc == n).sum())
            if count > 0:
                histogram[str(n)] = count
        if max_assoc >= 20:
            histogram["20+"] = int((n_assoc >= 20).sum())

        results.append(
            {
                "threshold": thresh,
                "n_grounded": n_grounded,
                "frac_grounded": round(n_grounded / d_sae, 4),
                "n_monospecific": n_mono,
                "n_oligospecific": n_oligo,
                "n_polyspecific": n_poly,
                "frac_mono_of_grounded": round(n_mono / max(n_grounded, 1), 4),
                "mean_codes_per_grounded": round(
                    float(n_assoc[n_assoc > 0].mean()) if n_grounded > 0 else 0.0, 2
                ),
                "histogram": histogram,
            }
        )

    return results


def run_posthoc_analyses(
    eval_output_dir: str | Path,
    activations_dir: str | Path,
    icd_csv_path: str | Path,
    posthoc_output_dir: str | Path,
    r_thresholds: list[float] | None = None,
    checkpoint_dir: str | Path | None = None,
    fdr_q: float = 0.05,
    min_prevalence: float = 0.02,
    max_codes: int = 50,
    join_key: str = "admission_id",
    icd_col_prefix: str = "icd9_",
    min_notes: int = 100,
) -> dict:
    """Run post-hoc analyses on existing ICD eval output.

    Three analyses on the saved correlation_matrices.npz:
      1. Recompute grounding at multiple |r| thresholds
      2. Partial correlation controlling for note length (n_tokens)
      3. Monospecificity — code-association histogram at each threshold

    The expensive shard-by-shard encoding is NOT re-run; note vectors are
    reassembled from the per-shard checkpoints in checkpoint_dir.
    """
    if r_thresholds is None:
        r_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    eval_output_dir = Path(eval_output_dir)
    activations_dir = Path(activations_dir)
    icd_csv_path = Path(icd_csv_path)
    posthoc_output_dir = Path(posthoc_output_dir)
    posthoc_output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = eval_output_dir / "shard_ckpt"
    checkpoint_dir = Path(checkpoint_dir)

    logger.info("=" * 60)
    logger.info("Post-hoc ICD Grounding Analyses")
    logger.info("=" * 60)

    # ---- Load existing results ----
    logger.info("Loading saved correlations...")
    saved = load_saved_correlations(eval_output_dir)
    r_pb = saved["r_pb"]
    p_adjusted = saved["p_adjusted"]
    significant = saved["significant"]
    code_names = saved["code_names"]
    n_notes = saved["n_notes"]

    # ---- Analysis 1: threshold sweep ----
    logger.info("Analysis 1: grounding at multiple |r| thresholds...")
    threshold_summaries = []
    for thresh in r_thresholds:
        gr = compute_grounding(
            r_pb=r_pb,
            p_adjusted=p_adjusted,
            significant=significant,
            code_names=code_names,
            n_notes=n_notes,
            r_threshold=thresh,
        )
        sub = posthoc_output_dir / f"grounding_r{thresh}"
        save_results(gr, sub)
        threshold_summaries.append({"threshold": thresh, **gr.summary_dict()})
        logger.info(
            f"  r>{thresh}: {gr.grounded_latent_count} grounded ({gr.grounded_latent_frac:.1%})"
        )

    # ---- Analysis 2: monospecificity ----
    logger.info("Analysis 2: monospecificity at each threshold...")
    mono = compute_monospecificity(r_pb, significant, r_thresholds)
    for row in mono:
        logger.info(
            f"  r>{row['threshold']}: {row['n_grounded']} grounded — "
            f"{row['n_monospecific']} mono, {row['n_oligospecific']} oligo, "
            f"{row['n_polyspecific']} poly"
        )

    # ---- Analysis 3: partial correlation (control for n_tokens) ----
    logger.info("Analysis 3: partial correlation controlling for n_tokens...")
    note_vectors, note_meta = reassemble_note_vectors(checkpoint_dir)
    logger.info(f"Reassembled note vectors: {note_vectors.shape}")

    icd_matrix, icd_code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=icd_csv_path,
        note_meta=note_meta,
        min_prevalence=min_prevalence,
        max_codes=max_codes,
        icd_col_prefix=icd_col_prefix,
        join_key=join_key,
        min_notes=min_notes,
    )
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)

    if "n_tokens" in matched_meta.columns:
        confound = matched_meta["n_tokens"].to_numpy(dtype=np.float64)
    else:
        confound = matched_meta["row_end"].to_numpy(dtype=np.float64) - matched_meta[
            "row_start"
        ].to_numpy(dtype=np.float64)
    logger.info(
        f"Confound n_tokens: mean={confound.mean():.1f}, "
        f"std={confound.std():.1f}, range=[{confound.min():.0f}, {confound.max():.0f}]"
    )

    r_partial, p_partial = compute_partial_point_biserial(X, icd_matrix, confound)
    logger.info(f"Partial correlation: max |r| = {np.abs(r_partial).max():.4f}")

    sig_partial, padj_partial = apply_bh_correction(p_partial, q=fdr_q)

    partial_dir = posthoc_output_dir / "partial"
    for thresh in r_thresholds:
        gr_partial = compute_grounding(
            r_pb=r_partial,
            p_adjusted=padj_partial,
            significant=sig_partial,
            code_names=icd_code_names,
            n_notes=X.shape[0],
            r_threshold=thresh,
        )
        save_results(gr_partial, partial_dir / f"grounding_r{thresh}")

    gr_partial_default = compute_grounding(
        r_pb=r_partial,
        p_adjusted=padj_partial,
        significant=sig_partial,
        code_names=icd_code_names,
        n_notes=X.shape[0],
        r_threshold=r_thresholds[0],
    )
    np.savez_compressed(
        partial_dir / "correlation_matrices.npz",
        r_pb=r_partial,
        p_adjusted=padj_partial,
        significant=sig_partial.astype(np.uint8),
    )
    with open(partial_dir / "code_names.json", "w") as f:
        json.dump(icd_code_names, f, indent=2)

    mono_partial = compute_monospecificity(r_partial, sig_partial, r_thresholds)
    logger.info("Partial-correlation monospecificity:")
    for row in mono_partial:
        logger.info(
            f"  r>{row['threshold']}: {row['n_grounded']} grounded — "
            f"{row['n_monospecific']} mono, {row['n_oligospecific']} oligo, "
            f"{row['n_polyspecific']} poly"
        )

    # ---- Save combined summary ----
    summary = {
        "threshold_sweep": threshold_summaries,
        "monospecificity": mono,
        "partial_correlation_monospecificity": mono_partial,
        "partial_correlation_summary": gr_partial_default.summary_dict(),
        "confound": "n_tokens",
        "r_thresholds": r_thresholds,
    }
    with open(posthoc_output_dir / "posthoc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved posthoc_summary.json to {posthoc_output_dir}")

    logger.info("=" * 60)
    logger.info("POST-HOC SUMMARY")
    logger.info("=" * 60)
    for ts in threshold_summaries:
        t = ts["threshold"]
        m = next(r for r in mono if r["threshold"] == t)
        mp = next(r for r in mono_partial if r["threshold"] == t)
        logger.info(
            f"  |r|>{t}: grounded={ts['grounded_latent_count']} "
            f"(mono={m['n_monospecific']}) | "
            f"partial: grounded={mp['n_grounded']} "
            f"(mono={mp['n_monospecific']})"
        )
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# 8.  Orchestrator — full pipeline
# ---------------------------------------------------------------------------


def _align_note_vectors_to_matched(
    note_vectors: np.ndarray,
    note_meta: pd.DataFrame,
    matched_meta: pd.DataFrame,
) -> np.ndarray:
    """Select rows of note_vectors that correspond to matched_meta, by note_idx.

    note_meta and note_vectors are aligned by row position (encode_and_pool
    builds them in lockstep). matched_meta is the result of an inner merge
    whose index has been reset by pandas — its row positions do NOT match
    note_vectors. We rebuild positions via the note_idx column.
    """
    note_meta_idx = note_meta["note_idx"].to_numpy()
    if len(set(note_meta_idx)) != len(note_meta_idx):
        raise ValueError("note_meta['note_idx'] contains duplicates; cannot align unambiguously.")
    pos = {int(nidx): i for i, nidx in enumerate(note_meta_idx)}
    matched_idx = matched_meta["note_idx"].to_numpy()
    try:
        aligned = [pos[int(nidx)] for nidx in matched_idx]
    except KeyError as exc:
        raise ValueError(
            f"matched_meta has note_idx={exc.args[0]} not present in note_meta."
        ) from None
    return note_vectors[aligned]


def run_icd_eval(
    activations_dir: str | Path,
    sae_checkpoint: str | Path | JumpReLUSAE,
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
    min_notes: int = 100,
    checkpoint_dir: str | Path | None = None,
    on_shard_complete: Callable[[int], None] | None = None,
) -> GroundingResults:
    """Run the full ICD-9 clinical grounding evaluation.

    This is the main entry point. Call it from a script or notebook.
    checkpoint_dir: directory for per-shard encode checkpoints; defaults to
        output_dir/shard_ckpt so resume works automatically on re-run.
    on_shard_complete: forwarded to encode_and_pool — see its docstring. The
        Modal entrypoints pass ``lambda _: artifacts_volume.commit()`` here.
    """
    activations_dir = Path(activations_dir)
    icd_csv_path = Path(icd_csv_path)
    output_dir = Path(output_dir)

    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "shard_ckpt"

    logger.info("=" * 60)
    logger.info("ICD-9 Clinical Grounding Pipeline")
    logger.info("=" * 60)

    # Step 1: Load SAE
    logger.info("Step 1: Loading SAE...")
    if isinstance(sae_checkpoint, JumpReLUSAE):
        sae = sae_checkpoint
    else:
        sae = JumpReLUSAE.from_checkpoint(Path(sae_checkpoint))

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
        on_shard_complete=on_shard_complete,
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
        min_notes=min_notes,
    )

    # Align note_vectors with matched notes by note_idx (always — never by
    # positional index). pandas merge resets the index, so matched_meta's
    # row positions do not correspond to positions in note_vectors. The
    # only stable join key is the note_idx column itself.
    X = _align_note_vectors_to_matched(note_vectors, note_meta, matched_meta)

    logger.info(f"Aligned activation matrix: {X.shape}")

    # Step 5: Compute correlations
    logger.info("Step 5: Computing point-biserial correlations...")
    r_pb, p_vals = compute_point_biserial_vectorised(X, icd_matrix)
    logger.info(f"Correlation matrix shape: {r_pb.shape}, max |r| = {np.abs(r_pb).max():.4f}")

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
