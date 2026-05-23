"""
Causal Ablation Experiment for SAE Features
=============================================

Provides causal evidence that grounded SAE features functionally encode
clinical concepts by ablating each feature's contribution to the residual
stream and measuring downstream LM cross-entropy on (a) notes about the
feature's correlated ICD code vs (b) notes about other codes.

Pipeline (one SAE, many notes, many features):

  for note in held_out_notes:
      x16 = forward(gemma, note)[layer=16]                # one clean forward
      loss_clean = continue_from_layer_17(x16) → CE on loss window
      for each target feature j (with associated code c):
          z = encode(x16 − μ)                              # centered for our SAEs
          x_hat = decode(z)                                # centered reconstruction
          x_hat_abl = x_hat − z[..., j:j+1] · W_dec[j]     # subtract feature contribution
          x_splice = x_hat + μ                             # un-center
          x_splice_abl = x_hat_abl + μ                     # un-center
          loss_recon = continue_from_layer_17(x_splice) → CE on loss window
          loss_abl   = continue_from_layer_17(x_splice_abl) → CE on loss window
          ablation_effect = loss_abl − loss_recon

For GemmaScope (raw-activation SAE), μ = 0 and subtract_b_dec=False — handled
by the `is_centered` flag on AblationSAEConfig.

Statistical test (per (feature, code) triple, across all held-out notes):
  Mann-Whitney U(ablation_effect | code=1, ablation_effect | code=0, alt='greater')
  Effect size: Cliff's δ
  Multiple-comparison correction: BH across all tests in one batch

References:
  - Marks et al. 2024, "Sparse Feature Circuits" — feature ablation methodology
  - Templeton et al. 2024, "Scaling Monosemanticity" — feature steering / ablation
  - Romano et al. 2006 — Cliff's δ interpretation thresholds

See `.tmp/ablation_design.md` for the full design rationale.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import yaml
from safetensors.torch import load_file as load_safetensors_torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------


@dataclass
class AblationTarget:
    """One (feature, code) ablation target. Either grounded or a control."""

    feature_idx: int
    code: str  # ICD-9 column name, e.g. "icd9_5856"
    kind: str = "grounded"  # "grounded" | "random_control" | "low_r_control"
    r_pb: float | None = None  # known correlation; None for random control


@dataclass
class AblationConfig:
    """Fully-specified configuration for one SAE's ablation experiment.

    One config = one SAE. Run separately for vanilla, JumpReLU, GemmaScope.

    Path conventions match the rest of the repo: container-relative (/out/...,
    /data/...) when running on Modal, absolute when running locally.
    """

    # ----------------- SAE identity ----------------------------------------
    sae_name: str  # human-readable, e.g. "vanilla", "jumprelu", "gemma_scope_16k"

    # One of these must be set: local checkpoint or HF download.
    sae_checkpoint_dir: str | None = None
    sae_hf_repo_id: str | None = None
    sae_hf_filename: str | None = None

    # Centering convention. True for our vanilla/JumpReLU (trained on centered
    # activations); False for GemmaScope (trained on raw activations).
    is_centered: bool = True

    # ----------------- Activations / model ---------------------------------
    # For vanilla/JumpReLU: centered activations directory (has mean.pt + metadata.jsonl).
    # For GemmaScope: raw activations directory (no centering).
    # Either way, we read metadata.jsonl from here to enumerate held-out notes.
    activations_dir: str = ""

    # Centered activations dir always — even for GemmaScope we may want metadata
    # from the centered dir for consistency. Optional; if None, fall back to
    # activations_dir.
    centered_activations_dir: str | None = None

    # Path to mean.pt for un-centering. Required when is_centered=True.
    mean_path: str | None = None

    # Gemma model
    model_name: str = "google/gemma-2-2b"
    layer: int = 16  # splice site = output of this transformer block

    # ----------------- Held-out notes --------------------------------------
    # Shard indices [start, end) used as the held-out set.
    held_out_shard_start: int = 281
    held_out_shard_end: int = 312

    # ICD label join
    icd_csv_path: str = ""
    join_key: str = "admission_id"
    text_col: str = "note_text"
    icd_col_prefix: str = "icd9_"

    # ----------------- Ablation targets ------------------------------------
    # Pre-computed list of (feature, code) pairs to ablate. Built by
    # ablation_features.select_targets() before this config is constructed.
    targets: list[dict[str, Any]] = field(default_factory=list)

    # ----------------- Forward / measurement -------------------------------
    max_length: int = 8192  # match extraction
    loss_window_frac: float = 0.25  # measure CE on last fraction of (non-pad) tokens
    min_loss_window_tokens: int = 32  # drop notes whose window < this

    # Batch size for the splice forward. 1 is safest; >1 requires per-note
    # left-padding which complicates the hook. Start with 1.
    batch_size: int = 1

    # Subset of held-out notes for the smoke run. None = all matched notes.
    max_notes: int | None = None

    # ----------------- Output ----------------------------------------------
    output_dir: str = ""
    # Per-shard checkpoint subdir. Resume: re-running skips shards with
    # results_shard_NNNN.parquet already present.
    shard_checkpoint_subdir: str = "shard_results"

    # ----------------- Diagnostics -----------------------------------------
    # Save per-(note,feature) records to a long-format parquet. Useful for
    # post-hoc filtering by length/positivity; large file (~10s of MB).
    save_per_note_records: bool = True

    # ----------------- Reproducibility -------------------------------------
    seed: int = 42

    # ----------------- Logging ---------------------------------------------
    logging_level: str = "INFO"

    @classmethod
    def from_dict(cls, d: dict) -> AblationConfig:
        """Construct, ignoring unknown keys (for forward-compatibility)."""
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - valid
        if unknown:
            print(f"WARNING: dropping unknown ablation config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})

    def __post_init__(self) -> None:
        # Validate path conventions.
        if self.is_centered and not self.mean_path:
            raise ValueError(
                "is_centered=True requires mean_path to be set "
                "(path to the float32 mean.pt produced by center.py)."
            )
        if self.sae_checkpoint_dir is None and self.sae_hf_repo_id is None:
            raise ValueError(
                "Set either sae_checkpoint_dir (for local) or "
                "sae_hf_repo_id + sae_hf_filename (for HF download)."
            )
        if not (0.0 < self.loss_window_frac <= 1.0):
            raise ValueError(f"loss_window_frac must be in (0, 1], got {self.loss_window_frac}")


def load_ablation_config(path: str | Path) -> AblationConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AblationConfig.from_dict(data)


def save_ablation_config(config: AblationConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


# ---------------------------------------------------------------------------
# 2.  TorchSAE — GPU-resident SAE inference
# ---------------------------------------------------------------------------


class TorchSAE:
    """Pure-torch SAE encoder/decoder with optional per-feature ablation.

    Mirrors the encode/decode arithmetic of icd_eval.JumpReLUSAE but stays on
    GPU and accepts torch tensors. The numpy class in icd_eval.py is the
    reference implementation; this is its tensor-fp32 sibling for the splice
    pipeline.

    Centering convention (controlled by `subtract_b_dec`):
      True  (vanilla/JumpReLU):  pre_act = (x − b_dec) @ W_enc + b_enc
      False (GemmaScope):        pre_act =  x          @ W_enc + b_enc

    Threshold convention:
      Vanilla: stored as zeros (or absent → defaults to zeros).
      JumpReLU: log_threshold key in checkpoint → exp() to get threshold.
      GemmaScope: threshold key directly present.
    """

    def __init__(
        self,
        W_enc: torch.Tensor,  # [d_model, d_sae]
        b_enc: torch.Tensor,  # [d_sae]
        W_dec: torch.Tensor,  # [d_sae, d_model]
        b_dec: torch.Tensor,  # [d_model]
        threshold: torch.Tensor,  # [d_sae], zeros for plain ReLU
        subtract_b_dec: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.W_enc = W_enc.to(device=device, dtype=dtype)
        self.b_enc = b_enc.to(device=device, dtype=dtype)
        self.W_dec = W_dec.to(device=device, dtype=dtype)
        self.b_dec = b_dec.to(device=device, dtype=dtype)
        self.threshold = threshold.to(device=device, dtype=dtype)
        self.subtract_b_dec = subtract_b_dec
        self.d_model, self.d_sae = self.W_enc.shape
        self.device = device
        self.dtype = dtype
        # Validate shapes
        assert self.b_enc.shape == (self.d_sae,), f"b_enc {self.b_enc.shape}"
        assert self.W_dec.shape == (self.d_sae, self.d_model), f"W_dec {self.W_dec.shape}"
        assert self.b_dec.shape == (self.d_model,), f"b_dec {self.b_dec.shape}"
        assert self.threshold.shape == (self.d_sae,), f"threshold {self.threshold.shape}"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: torch.device,
        subtract_b_dec: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> TorchSAE:
        """Load from a local checkpoint dir (vanilla or custom JumpReLU)."""
        ckpt = Path(checkpoint_dir)
        weights_path = ckpt / "sae_weights.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"SAE weights not found: {weights_path}")
        tensors = load_safetensors_torch(str(weights_path))

        W_enc = tensors["W_enc"].float()  # [d_model, d_sae]
        b_enc = tensors["b_enc"].float()
        W_dec = tensors["W_dec"].float()  # [d_sae, d_model]
        b_dec = tensors["b_dec"].float()

        if "log_threshold" in tensors:
            threshold = tensors["log_threshold"].float().exp()  # JumpReLU
            sae_type = "JumpReLU"
        elif "threshold" in tensors:
            threshold = tensors["threshold"].float()
            sae_type = "JumpReLU(threshold)"
        else:
            threshold = torch.zeros(W_enc.shape[1], dtype=torch.float32)  # vanilla ReLU
            sae_type = "Vanilla ReLU"

        logger.info(
            f"Loaded {sae_type} SAE from {ckpt}: "
            f"d_model={W_enc.shape[0]}, d_sae={W_enc.shape[1]}, "
            f"subtract_b_dec={subtract_b_dec}"
        )
        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            threshold=threshold,
            subtract_b_dec=subtract_b_dec,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        filename: str,
        device: torch.device,
        token: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TorchSAE:
        """Load GemmaScope (or similar HF-hosted) SAE. subtract_b_dec hard-coded False."""
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        logger.info(f"Downloaded {repo_id}/{filename} → {local_path}")
        data = np.load(local_path)
        available = set(data.files)
        logger.info(f"GemmaScope weight keys: {sorted(available)}")

        def _find(canonical: str, aliases: list[str], required: bool = True) -> np.ndarray | None:
            for a in [canonical, *aliases]:
                if a in data:
                    arr = data[a].astype(np.float32)
                    if a == "log_threshold":
                        arr = np.exp(arr)
                    return arr
            if required:
                raise KeyError(f"Missing {canonical}; available={sorted(available)}")
            return None

        W_enc = torch.from_numpy(_find("W_enc", ["encoder.weight", "w_enc"]))  # [d_model, d_sae]
        b_enc = torch.from_numpy(_find("b_enc", ["encoder.bias"]))
        b_dec = torch.from_numpy(_find("b_dec", ["decoder.bias"]))
        W_dec_arr = _find("W_dec", ["decoder.weight", "w_dec"], required=True)
        W_dec = torch.from_numpy(W_dec_arr)
        d_model, d_sae = W_enc.shape
        if W_dec.shape != (d_sae, d_model):
            # GemmaScope ships W_dec as [d_sae, d_model] in .npz format; if a
            # different convention shows up, transpose here. Be loud about it.
            if W_dec.shape == (d_model, d_sae):
                logger.warning("W_dec shape is [d_model, d_sae]; transposing.")
                W_dec = W_dec.T.contiguous()
            else:
                raise ValueError(
                    f"W_dec shape {tuple(W_dec.shape)} not in "
                    f"[({d_sae},{d_model}), ({d_model},{d_sae})]"
                )

        thresh_arr = _find("threshold", ["log_threshold", "theta"], required=False)
        if thresh_arr is None:
            threshold = torch.zeros(d_sae, dtype=torch.float32)
            logger.info("No threshold key in HF SAE — using zeros (plain ReLU).")
        else:
            threshold = torch.from_numpy(thresh_arr)

        logger.info(
            f"Loaded HF SAE {repo_id}/{filename}: "
            f"d_model={d_model}, d_sae={d_sae}, subtract_b_dec=False (GemmaScope convention)"
        )
        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            threshold=threshold,
            subtract_b_dec=False,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d_model] → [..., d_sae] sparse latents.

        x must be in the SAE's input space:
          - subtract_b_dec=True (our SAEs): centered activations (x − μ).
          - subtract_b_dec=False (GemmaScope): raw activations.
        """
        x_in = (x - self.b_dec) if self.subtract_b_dec else x
        pre_act = x_in @ self.W_enc + self.b_enc
        z = pre_act * (pre_act > self.threshold).to(self.dtype)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[..., d_sae] → [..., d_model] reconstruction (in SAE's input space).

        Inverse of encode(). To get an activation that can be spliced into
        the raw residual stream, add the saved mean (`μ`) afterwards when
        is_centered=True. GemmaScope decode output is already in raw space.
        """
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def decode_with_ablation(
        self,
        z: torch.Tensor,  # [..., d_sae]
        feature_idx: int,
    ) -> torch.Tensor:
        """Decode while zeroing out feature_idx's contribution.

        Mathematically: decode(z) − z[..., feature_idx:feature_idx+1] · W_dec[feature_idx:feature_idx+1]
        Equivalent to: decode(z_with_feature_j_zeroed).
        We compute it as a subtraction from the full decode for clarity and
        because it lets the caller share the un-ablated `x_hat` across many
        feature ablations without redundant decode() calls.
        """
        x_hat = self.decode(z)
        contribution = (
            z[..., feature_idx : feature_idx + 1] * self.W_dec[feature_idx : feature_idx + 1]
        )
        return x_hat - contribution


# ---------------------------------------------------------------------------
# 3.  Forward-hook splice mechanism
# ---------------------------------------------------------------------------


class LayerSplice:
    """Forward-hook on `model.model.layers[layer]` that swaps the layer's output.

    Three modes controlled by self.mode:
      "passthrough":  hook is no-op; the layer's normal output passes through.
      "capture":      hook stashes the output into self.captured (no replacement).
      "splice":       hook returns self.splice_tensor in place of the layer's output.

    Usage:
        splice = LayerSplice(model, layer=16)
        with splice.enabled():
            # 1. capture clean x16
            splice.mode = "capture"
            _ = model(input_ids=..., use_cache=False)
            x16 = splice.captured                      # [1, n_tok, d_model] fp16

            # 2. splice variants and measure CE on each
            splice.mode = "splice"
            splice.splice_tensor = x16_modified
            out = model(input_ids=..., use_cache=False)
            logits = out.logits                         # [1, n_tok, vocab]

    On Gemma-2-2B HF the decoder layer returns a tuple `(hidden_states, ...)`.
    We detect that and rebuild the tuple. If a non-tuple is returned (some
    HF models, or after a future HF refactor), we just return the new tensor.
    """

    def __init__(self, model: Any, layer: int) -> None:
        self.model = model
        self.layer = layer
        self.mode: str = "passthrough"
        self.captured: torch.Tensor | None = None
        self.splice_tensor: torch.Tensor | None = None
        self._handle: Any = None

    def _hook(
        self,
        module: Any,
        inputs: tuple,
        output: Any,
    ) -> Any:
        # Layer output is either a tensor or a tuple (hidden, *aux).
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()

        if self.mode == "passthrough":
            return output

        if self.mode == "capture":
            # Detach + clone so the captured tensor outlives autograd / future hooks.
            self.captured = hidden.detach().clone()
            return output

        if self.mode == "splice":
            if self.splice_tensor is None:
                raise RuntimeError("LayerSplice.mode='splice' but splice_tensor is None")
            # Cast to layer-output dtype to avoid silent dtype mismatch.
            new_hidden = self.splice_tensor.to(dtype=hidden.dtype, device=hidden.device)
            if new_hidden.shape != hidden.shape:
                raise RuntimeError(
                    f"Splice shape mismatch: layer expects {tuple(hidden.shape)}, "
                    f"got {tuple(new_hidden.shape)}"
                )
            if rest:
                return (new_hidden, *rest)
            return new_hidden

        raise ValueError(f"Unknown LayerSplice.mode: {self.mode!r}")

    def enable(self) -> None:
        if self._handle is not None:
            return
        # model.model.layers[layer] — Gemma-2 architecture
        target = self.model.model.layers[self.layer]
        self._handle = target.register_forward_hook(self._hook)
        logger.debug(f"LayerSplice hook attached to model.model.layers[{self.layer}]")

    def disable(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            logger.debug("LayerSplice hook removed")

    def enabled(self) -> LayerSplice:
        """Context-manager style: `with splice.enabled(): ...`."""
        return self

    def __enter__(self) -> LayerSplice:
        self.enable()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disable()


# ---------------------------------------------------------------------------
# 4.  CE measurement with loss window
# ---------------------------------------------------------------------------


def build_loss_window_mask(
    n_real_tokens: int,
    seq_len: int,
    window_frac: float,
) -> torch.Tensor:
    """Boolean mask [seq_len - 1] selecting prediction positions in the loss window.

    HF causal LM convention: logits[t] predicts input_ids[t+1]. We mark
    positions [floor((1 - window_frac) * n_real_tokens), n_real_tokens − 1)
    as "in window" because:
      - the last real token (index n_real_tokens − 1) has no next token to predict
        when use_cache=False and we did not pad to seq_len + 1
      - the window's last *predicted* token is at input_ids[n_real_tokens − 1],
        produced by logits at position n_real_tokens − 2 → so positions
        [start, n_real_tokens − 1) inclusive lower / exclusive upper.

    Returns a bool tensor of length seq_len - 1 (the shifted-prediction length).
    """
    mask = torch.zeros(seq_len - 1, dtype=torch.bool)
    if n_real_tokens < 2:
        return mask  # nothing to score
    start = max(0, int((1.0 - window_frac) * n_real_tokens))
    end = max(start + 1, n_real_tokens - 1)
    # Cap end at the mask length (shouldn't trigger if seq_len ≥ n_real_tokens).
    end = min(end, seq_len - 1)
    mask[start:end] = True
    return mask


def cross_entropy_in_window(
    logits: torch.Tensor,  # [1, seq_len, vocab]
    input_ids: torch.Tensor,  # [1, seq_len]
    window_mask: torch.Tensor,  # [seq_len - 1]
) -> float:
    """Mean CE over predictions whose mask position is True.

    Standard causal-LM shift: logits[t] vs input_ids[t+1].
    Returns mean over selected positions (NOT sum). Returns NaN if no
    positions are selected.
    """
    if not window_mask.any():
        return float("nan")

    # Shift to causal-LM prediction alignment.
    shift_logits = logits[:, :-1, :].float()  # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:]  # [1, seq_len-1]

    mask = window_mask.to(shift_logits.device)
    selected_logits = shift_logits[0][mask]  # [n_selected, vocab]
    selected_labels = shift_labels[0][mask]  # [n_selected]

    loss = f.cross_entropy(selected_logits, selected_labels, reduction="mean")
    return float(loss.item())


# ---------------------------------------------------------------------------
# 5.  Per-note ablation
# ---------------------------------------------------------------------------


@dataclass
class NoteAblationResult:
    """One forward pass's CE measurements at the loss window."""

    note_idx: int
    admission_id: int | None
    n_tokens_real: int
    n_tokens_in_window: int
    loss_clean: float
    # Per-feature: loss_recon is feature-independent within an SAE; loss_abl
    # is per-feature.
    loss_recon: float  # ablation-free reconstruction
    per_feature: dict[int, float] = field(default_factory=dict)
    # Per-feature mean activation of the targeted feature on this note's tokens
    # (sanity check: if a feature never fires on a note, ablation effect ≈ 0).
    per_feature_mean_act: dict[int, float] = field(default_factory=dict)


def run_ablation_for_note(
    model: Any,
    tokenizer: Any,
    text: str,
    sae: TorchSAE,
    target_feature_indices: list[int],
    splice: LayerSplice,
    mean_centering: torch.Tensor | None,  # [d_model] or None for GemmaScope
    note_idx: int,
    admission_id: int | None,
    max_length: int,
    loss_window_frac: float,
    min_loss_window_tokens: int,
    layer_dtype: torch.dtype,
) -> NoteAblationResult | None:
    """Run all ablations for one note. Returns None if note is too short.

    Steps:
      1. Tokenize text (truncate to max_length, add special tokens — matches extraction).
      2. Capture clean x16 via splice.mode='capture'.
      3. Compute loss_clean by re-running with splice.mode='splice', splice_tensor=x16.
         (We re-splice the identical tensor to be 100% sure the splice
          mechanism itself contributes nothing — this is the splice's own
          identity check.)
      4. Encode → x_hat. Compute loss_recon with splice_tensor = x_hat + μ.
      5. For each target feature j: compute x_hat_abl, splice, measure loss_abl.

    Returns NoteAblationResult or None if window is too short.
    """
    device = sae.device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    n_real = int(attn_mask.sum().item())
    seq_len = int(input_ids.shape[1])

    window_mask = build_loss_window_mask(n_real, seq_len, loss_window_frac)
    n_in_window = int(window_mask.sum().item())
    if n_in_window < min_loss_window_tokens:
        return None  # skip this note

    # ---- 1. Clean forward: capture x16, no splice ----
    splice.mode = "capture"
    splice.captured = None
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    x16 = splice.captured  # [1, seq_len, d_model], dtype = model dtype (fp16 on GPU)
    if x16 is None:
        raise RuntimeError("LayerSplice failed to capture residual — hook not firing?")
    loss_clean = cross_entropy_in_window(out.logits, input_ids, window_mask)

    # ---- 2. Compute SAE encode/decode in fp32, then splice ----
    # Slice to real tokens (positions [n_real:] are padding even with attention
    # mask; encoding them is wasted compute and could shift the decode mean).
    x16_real = x16[:, :n_real, :].to(sae.dtype)  # [1, n_real, d_model] fp32

    if mean_centering is not None:
        x16_for_sae = x16_real - mean_centering  # centered
    else:
        x16_for_sae = x16_real  # GemmaScope: raw

    z = sae.encode(x16_for_sae)  # [1, n_real, d_sae]
    x_hat = sae.decode(z)  # [1, n_real, d_model] in SAE input space

    # Build full splice tensor: real tokens get reconstructed, padding tokens
    # keep their original x16 (they don't affect the loss anyway, but the
    # shape must match the layer output exactly).
    def _build_full(x_hat_real: torch.Tensor) -> torch.Tensor:
        # x_hat_real is in SAE input space; add μ back if we centered.
        if mean_centering is not None:
            x_hat_splice_real = x_hat_real + mean_centering
        else:
            x_hat_splice_real = x_hat_real
        # Reassemble [1, seq_len, d_model] by stitching reconstructed real
        # tokens with the original padding tokens.
        full = x16.clone()
        full[:, :n_real, :] = x_hat_splice_real.to(layer_dtype)
        return full

    # ---- 3. Reconstructed (un-ablated) forward ----
    splice_tensor = _build_full(x_hat)
    splice.mode = "splice"
    splice.splice_tensor = splice_tensor
    with torch.no_grad():
        out_recon = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    loss_recon = cross_entropy_in_window(out_recon.logits, input_ids, window_mask)

    # ---- 4. Per-feature ablations ----
    per_feature_loss: dict[int, float] = {}
    per_feature_mean_act: dict[int, float] = {}

    for j in target_feature_indices:
        x_hat_abl = sae.decode_with_ablation(z, feature_idx=j)
        splice.splice_tensor = _build_full(x_hat_abl)
        with torch.no_grad():
            out_abl = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        loss_abl = cross_entropy_in_window(out_abl.logits, input_ids, window_mask)
        per_feature_loss[j] = loss_abl
        per_feature_mean_act[j] = float(z[0, :, j].mean().item())

    splice.mode = "passthrough"  # reset
    splice.splice_tensor = None

    return NoteAblationResult(
        note_idx=note_idx,
        admission_id=admission_id,
        n_tokens_real=n_real,
        n_tokens_in_window=n_in_window,
        loss_clean=loss_clean,
        loss_recon=loss_recon,
        per_feature=per_feature_loss,
        per_feature_mean_act=per_feature_mean_act,
    )


# ---------------------------------------------------------------------------
# 6.  Statistical analysis
# ---------------------------------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ — non-parametric effect size, range [−1, +1].

    δ = P(X > Y) − P(X < Y), estimated by the count of pairs.
    +1: every x exceeds every y; −1: reverse; 0: no stochastic ordering.

    Implemented via the rank-biserial identity:
        δ = (2U − n_x · n_y) / (n_x · n_y)
    where U is Mann-Whitney's U statistic (without continuity correction).
    """
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return float("nan")
    # MWU via rank-sum
    from scipy.stats import mannwhitneyu

    u_stat, _ = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return (2.0 * u_stat - n_x * n_y) / (n_x * n_y)


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. 2006 conventions for Cliff's δ magnitude."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def compute_statistics(
    per_note_results: list[NoteAblationResult],
    targets: list[dict[str, Any]],
    icd_matrix: np.ndarray,
    code_names: list[str],
    note_idx_to_row: dict[int, int],
) -> pd.DataFrame:
    """Per-target Mann-Whitney + Cliff's δ + BH correction.

    Args:
        per_note_results: list of NoteAblationResult, one per held-out note.
        targets: list of dicts with keys (feature_idx, code, kind, r_pb).
        icd_matrix: [n_notes_matched, n_codes] binary int8 from
            load_and_align_icd_labels.
        code_names: column names aligned to icd_matrix columns.
        note_idx_to_row: maps note_idx → row in icd_matrix. Notes not present
            in icd_matrix (e.g. ICD-join failed) are dropped from analysis.

    Returns:
        DataFrame indexed by target index with columns:
            feature, code, kind, r_pb_train, n_pos, n_neg,
            mean_effect_pos, mean_effect_neg, median_effect_pos, median_effect_neg,
            cliffs_delta, delta_magnitude, mw_U, p_raw, p_bh, sig_q05,
            mean_loss_clean, mean_loss_recon, mean_recon_tax,
            mean_target_activation_pos, mean_target_activation_neg
    """
    from scipy.stats import mannwhitneyu

    code_to_col = {c: i for i, c in enumerate(code_names)}

    # Build per-feature effect arrays
    rows: list[dict[str, Any]] = []
    for tgt in targets:
        j = int(tgt["feature_idx"])
        code = tgt["code"]
        if code not in code_to_col:
            # Code wasn't matched in ICD eval (filtered by min_prevalence). Skip.
            rows.append(
                {
                    "feature": j,
                    "code": code,
                    "kind": tgt.get("kind", "grounded"),
                    "r_pb_train": tgt.get("r_pb"),
                    "n_pos": 0,
                    "n_neg": 0,
                    "mean_effect_pos": float("nan"),
                    "mean_effect_neg": float("nan"),
                    "median_effect_pos": float("nan"),
                    "median_effect_neg": float("nan"),
                    "cliffs_delta": float("nan"),
                    "delta_magnitude": "n/a",
                    "mw_U": float("nan"),
                    "p_raw": float("nan"),
                    "p_bh": float("nan"),
                    "sig_q05": False,
                    "mean_loss_clean": float("nan"),
                    "mean_loss_recon": float("nan"),
                    "mean_recon_tax": float("nan"),
                    "mean_target_activation_pos": float("nan"),
                    "mean_target_activation_neg": float("nan"),
                    "note": "code not in ICD matrix",
                }
            )
            continue

        col = code_to_col[code]
        pos_effects: list[float] = []
        neg_effects: list[float] = []
        pos_acts: list[float] = []
        neg_acts: list[float] = []
        losses_clean: list[float] = []
        losses_recon: list[float] = []

        for r in per_note_results:
            row_idx = note_idx_to_row.get(r.note_idx)
            if row_idx is None:
                continue
            if j not in r.per_feature:
                continue
            effect = r.per_feature[j] - r.loss_recon
            act = r.per_feature_mean_act.get(j, 0.0)
            losses_clean.append(r.loss_clean)
            losses_recon.append(r.loss_recon)
            if icd_matrix[row_idx, col] == 1:
                pos_effects.append(effect)
                pos_acts.append(act)
            else:
                neg_effects.append(effect)
                neg_acts.append(act)

        pos = np.asarray(pos_effects, dtype=np.float64)
        neg = np.asarray(neg_effects, dtype=np.float64)
        n_pos = len(pos)
        n_neg = len(neg)

        if n_pos >= 2 and n_neg >= 2:
            u_stat, p_val = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic")
            delta = (2.0 * u_stat - n_pos * n_neg) / (n_pos * n_neg)
        else:
            u_stat, p_val, delta = float("nan"), float("nan"), float("nan")

        rows.append(
            {
                "feature": j,
                "code": code,
                "kind": tgt.get("kind", "grounded"),
                "r_pb_train": tgt.get("r_pb"),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "mean_effect_pos": float(pos.mean()) if n_pos else float("nan"),
                "mean_effect_neg": float(neg.mean()) if n_neg else float("nan"),
                "median_effect_pos": float(np.median(pos)) if n_pos else float("nan"),
                "median_effect_neg": float(np.median(neg)) if n_neg else float("nan"),
                "cliffs_delta": delta,
                "delta_magnitude": (
                    cliffs_delta_magnitude(delta) if not np.isnan(delta) else "n/a"
                ),
                "mw_U": float(u_stat),
                "p_raw": float(p_val),
                "p_bh": float("nan"),  # filled in after BH below
                "sig_q05": False,
                "mean_loss_clean": float(np.mean(losses_clean)) if losses_clean else float("nan"),
                "mean_loss_recon": float(np.mean(losses_recon)) if losses_recon else float("nan"),
                "mean_recon_tax": (
                    float(np.mean(np.asarray(losses_recon) - np.asarray(losses_clean)))
                    if losses_clean
                    else float("nan")
                ),
                "mean_target_activation_pos": float(np.mean(pos_acts))
                if pos_acts
                else float("nan"),
                "mean_target_activation_neg": float(np.mean(neg_acts))
                if neg_acts
                else float("nan"),
                "note": "",
            }
        )

    df = pd.DataFrame(rows)

    # BH-FDR across all valid p-values.
    valid_mask = ~df["p_raw"].isna()
    if valid_mask.any():
        p_raw = df.loc[valid_mask, "p_raw"].values
        m = len(p_raw)
        order = np.argsort(p_raw)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)
        adjusted = p_raw * m / ranks
        # Enforce monotonicity from the largest p down.
        sorted_adj = adjusted[order]
        sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
        adjusted[order] = sorted_adj
        adjusted = np.clip(adjusted, 0, 1)
        df.loc[valid_mask, "p_bh"] = adjusted
        df.loc[valid_mask, "sig_q05"] = adjusted <= 0.05

    return df


# ---------------------------------------------------------------------------
# 7.  Note iteration (held-out shards → text)
# ---------------------------------------------------------------------------


def iter_held_out_notes(
    metadata_path: Path,
    shard_start: int,
    shard_end: int,
    text_lookup: dict[int, str],
    max_notes: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield held-out notes with their text, in metadata order.

    Skips notes for which text_lookup has no entry (admission_id join missed).
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found: {metadata_path}")
    yielded = 0
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            shard = int(rec["shard"])
            if shard < shard_start or shard >= shard_end:
                continue
            note_idx = int(rec["note_idx"])
            text = text_lookup.get(note_idx)
            if text is None:
                continue
            yield {
                "note_idx": note_idx,
                "admission_id": rec.get("admission_id"),
                "shard": shard,
                "text": text,
                "num_tokens_truncated": int(rec.get("num_tokens_truncated", 0)),
            }
            yielded += 1
            if max_notes is not None and yielded >= max_notes:
                return


def load_note_texts(
    icd_csv_path: Path,
    metadata_df: pd.DataFrame,
    join_key: str,
    text_col: str,
) -> dict[int, str]:
    """Build a {note_idx → text} dict by joining metadata on admission_id."""
    icd_df = pd.read_csv(icd_csv_path, usecols=[join_key, text_col], low_memory=False)
    icd_df = icd_df.drop_duplicates(subset=join_key, keep="first")
    meta_keep = metadata_df[[join_key, "note_idx"]].drop_duplicates(subset="note_idx", keep="first")
    merged = meta_keep.merge(icd_df, on=join_key, how="inner")
    out = {int(r["note_idx"]): str(r[text_col]) for _, r in merged.iterrows()}
    logger.info(f"Loaded text for {len(out)} of {len(meta_keep)} notes via {join_key} join")
    return out


# ---------------------------------------------------------------------------
# 8.  Orchestrator
# ---------------------------------------------------------------------------


def _make_target_feature_list(targets: list[dict[str, Any]]) -> list[int]:
    """Extract unique feature indices from targets list, preserving first-seen order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for t in targets:
        j = int(t["feature_idx"])
        if j not in seen:
            seen.add(j)
            ordered.append(j)
    return ordered


def run_ablation(
    config: AblationConfig,
    on_shard_complete: Any = None,  # callable(shard_idx) → None
) -> dict[str, Any]:
    """End-to-end ablation experiment for one SAE.

    Writes per-shard intermediate results (resumable) and a final results
    CSV + summary JSON to config.output_dir.
    """
    from mech_interp_research.icd_eval import load_and_align_icd_labels, load_metadata
    from mech_interp_research.model import load_model_and_tokenizer

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_ckpt_dir = output_dir / config.shard_checkpoint_subdir
    shard_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load metadata + ICD labels -----
    metadata = load_metadata(Path(config.activations_dir))
    held = metadata[
        (metadata["shard"] >= config.held_out_shard_start)
        & (metadata["shard"] < config.held_out_shard_end)
    ].reset_index(drop=True)
    logger.info(
        f"Held-out shards [{config.held_out_shard_start},{config.held_out_shard_end}): "
        f"{len(held)} notes across {held['shard'].nunique()} shards"
    )

    icd_matrix, code_names, matched_meta = load_and_align_icd_labels(
        icd_csv_path=Path(config.icd_csv_path),
        note_meta=held,
        min_prevalence=0.0,  # don't re-filter — we trust targets list
        max_codes=99999,
        icd_col_prefix=config.icd_col_prefix,
        join_key=config.join_key,
        min_notes=10,
    )
    note_idx_to_row = {int(nidx): i for i, nidx in enumerate(matched_meta["note_idx"].values)}
    logger.info(f"ICD-aligned: {icd_matrix.shape[0]} notes × {icd_matrix.shape[1]} codes")

    # Build text lookup from CSV
    text_lookup = load_note_texts(
        icd_csv_path=Path(config.icd_csv_path),
        metadata_df=matched_meta,
        join_key=config.join_key,
        text_col=config.text_col,
    )

    # ----- Load model, tokenizer, SAE, mean -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(
        config.model_name, device=str(device), hf_token=os.environ.get("HF_TOKEN")
    )
    # Determine the dtype the layer outputs in (fp16 on cuda, fp32 on cpu).
    layer_dtype = next(model.parameters()).dtype

    if config.sae_checkpoint_dir:
        sae = TorchSAE.from_checkpoint(
            checkpoint_dir=config.sae_checkpoint_dir,
            device=device,
            subtract_b_dec=config.is_centered,
            dtype=torch.float32,
        )
    else:
        sae = TorchSAE.from_huggingface(
            repo_id=str(config.sae_hf_repo_id),
            filename=str(config.sae_hf_filename),
            device=device,
            token=os.environ.get("HF_TOKEN"),
            dtype=torch.float32,
        )

    if config.is_centered:
        if not config.mean_path:
            raise ValueError("is_centered=True but mean_path is None")
        mean_tensor = torch.load(config.mean_path, weights_only=True, map_location=device)
        mean_tensor = mean_tensor.to(dtype=torch.float32)
        assert mean_tensor.shape == (
            sae.d_model,
        ), f"mean.pt shape {tuple(mean_tensor.shape)} != ({sae.d_model},)"
        logger.info(f"Loaded mean from {config.mean_path}, norm={mean_tensor.norm().item():.4f}")
    else:
        mean_tensor = None

    # ----- Attach splice hook -----
    splice = LayerSplice(model, layer=config.layer)
    splice.enable()

    # ----- Iterate notes, run ablations, checkpoint per shard -----
    target_feature_indices = _make_target_feature_list(config.targets)
    logger.info(
        f"Ablating {len(target_feature_indices)} distinct features across "
        f"{len(config.targets)} (feature, code) targets"
    )

    all_results: list[NoteAblationResult] = []
    # Group held-out notes by shard for per-shard checkpointing.
    held_shards = sorted(held["shard"].unique())

    # Bookkeeping for resume: re-load any shard checkpoints already on disk.
    for shard_idx in held_shards:
        ck = shard_ckpt_dir / f"shard_{int(shard_idx):04d}_results.json"
        if ck.exists():
            recs = json.loads(ck.read_text())
            for rec in recs:
                all_results.append(
                    NoteAblationResult(
                        note_idx=int(rec["note_idx"]),
                        admission_id=rec.get("admission_id"),
                        n_tokens_real=int(rec["n_tokens_real"]),
                        n_tokens_in_window=int(rec["n_tokens_in_window"]),
                        loss_clean=float(rec["loss_clean"]),
                        loss_recon=float(rec["loss_recon"]),
                        per_feature={int(k): float(v) for k, v in rec["per_feature"].items()},
                        per_feature_mean_act={
                            int(k): float(v) for k, v in rec["per_feature_mean_act"].items()
                        },
                    )
                )
    done_shards = {int(p.stem.split("_")[1]) for p in shard_ckpt_dir.glob("shard_*_results.json")}
    if done_shards:
        logger.info(f"Resumed: {len(done_shards)} shards already done")

    note_iter = iter_held_out_notes(
        metadata_path=Path(config.activations_dir) / "metadata.jsonl",
        shard_start=config.held_out_shard_start,
        shard_end=config.held_out_shard_end,
        text_lookup=text_lookup,
        max_notes=config.max_notes,
    )

    # Process notes shard-by-shard (note_iter yields in metadata order which
    # is shard-major; we accumulate into per-shard buffers).
    current_shard: int | None = None
    shard_buffer: list[NoteAblationResult] = []
    n_processed = 0
    n_skipped_short = 0

    def _flush_shard(shard_idx: int, buffer: list[NoteAblationResult]) -> None:
        if not buffer:
            return
        out = [
            {
                "note_idx": r.note_idx,
                "admission_id": r.admission_id,
                "n_tokens_real": r.n_tokens_real,
                "n_tokens_in_window": r.n_tokens_in_window,
                "loss_clean": r.loss_clean,
                "loss_recon": r.loss_recon,
                "per_feature": {str(k): v for k, v in r.per_feature.items()},
                "per_feature_mean_act": {str(k): v for k, v in r.per_feature_mean_act.items()},
            }
            for r in buffer
        ]
        path = shard_ckpt_dir / f"shard_{int(shard_idx):04d}_results.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=None))
        os.replace(tmp, path)
        logger.info(f"Wrote {path.name} ({len(buffer)} notes)")
        if on_shard_complete is not None:
            on_shard_complete(int(shard_idx))

    for note in note_iter:
        shard = note["shard"]
        if current_shard is None:
            current_shard = shard
        if shard != current_shard:
            # Boundary: flush completed shard, reset buffer
            _flush_shard(current_shard, shard_buffer)
            shard_buffer = []
            current_shard = shard

        if int(shard) in done_shards:
            continue  # resume — entire shard already in all_results

        result = run_ablation_for_note(
            model=model,
            tokenizer=tokenizer,
            text=note["text"],
            sae=sae,
            target_feature_indices=target_feature_indices,
            splice=splice,
            mean_centering=mean_tensor,
            note_idx=note["note_idx"],
            admission_id=note["admission_id"],
            max_length=config.max_length,
            loss_window_frac=config.loss_window_frac,
            min_loss_window_tokens=config.min_loss_window_tokens,
            layer_dtype=layer_dtype,
        )
        if result is None:
            n_skipped_short += 1
            continue
        shard_buffer.append(result)
        all_results.append(result)
        n_processed += 1
        if n_processed % 25 == 0:
            logger.info(
                f"  processed {n_processed} notes (skipped {n_skipped_short} short), shard={shard}"
            )

    # Flush trailing shard
    if current_shard is not None:
        _flush_shard(current_shard, shard_buffer)

    splice.disable()

    # ----- Statistics -----
    logger.info(f"Computing statistics over {len(all_results)} notes")
    stats_df = compute_statistics(
        per_note_results=all_results,
        targets=config.targets,
        icd_matrix=icd_matrix,
        code_names=code_names,
        note_idx_to_row=note_idx_to_row,
    )
    stats_path = output_dir / "ablation_results.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Wrote {stats_path} ({len(stats_df)} rows)")

    # ----- Summary JSON -----
    summary = {
        "sae_name": config.sae_name,
        "n_held_out_notes": int(len(held)),
        "n_notes_with_text": int(len(text_lookup)),
        "n_notes_evaluated": int(len(all_results)),
        "n_notes_skipped_short_window": int(n_skipped_short),
        "n_targets": len(config.targets),
        "n_features_distinct": len(target_feature_indices),
        "n_sig_q05": int(stats_df["sig_q05"].sum()),
        "median_cliffs_delta_grounded": float(
            stats_df.loc[stats_df["kind"] == "grounded", "cliffs_delta"].median()
        ),
        "median_cliffs_delta_controls": float(
            stats_df.loc[stats_df["kind"] != "grounded", "cliffs_delta"].median()
        ),
        "mean_loss_clean": float(stats_df["mean_loss_clean"].mean()),
        "mean_loss_recon": float(stats_df["mean_loss_recon"].mean()),
        "mean_recon_tax": float(stats_df["mean_recon_tax"].mean()),
        "config": asdict(config),
    }
    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Wrote {summary_path}")

    return summary
