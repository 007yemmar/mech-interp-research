"""Import a precomputed direction matrix as a pseudo-SAE checkpoint. CPU-only.

Some feature sources are built by other pipelines that persist a direction
matrix and its thresholds but stop at *grounding* — they produce pooled
`[n_notes, k]` vectors, which is enough to correlate against ICD labels but not
enough to reach the LLM explainer. The explainer has to *encode tokens* to find
each direction's top-activating contexts, and that needs an encoder on disk.

This entrypoint is the bridge: it reads a `[d_model, k]` direction matrix and a
`[k]` threshold vector and writes them in the pseudo-SAE format
`icd_eval.JumpReLUSAE.from_checkpoint` already loads, so
`feature_inspector` → `auto_interp` run on the source unmodified.

The canonical consumer is the random-matched null (necessity baseline A4),
whose `directions.npy` and `thresholds_note_matched.npy` are already on the
volume. Without this conversion the null has grounding statistics but no
concordance verdict, and gates G2/G3 cannot be evaluated at all.

Run:
    uv run modal run modal_app/import_direction_source.py \
        --config-file configs/import_random_matched.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


# Annotations omitted: numpy is imported inside the Modal function, not at
# module scope, so a module-level np.ndarray annotation would not resolve.
def _calibrate_from_sae(config, W, log):
    """Note-level thresholds for a source that shipped directions but no thresholds.

    Reuses build_feature_source's token sampler and the cached Arm-C selection,
    so nothing already computed is recomputed and the calibration is the same
    code path the purpose-built arms use.
    """
    import json

    import numpy as np

    from mech_interp_research.feature_sources import (
        calibrate_thresholds_note_level,
        sample_tokens_note_granular,
    )
    from mech_interp_research.necessity_stats import sae_note_level_densities

    held_out_start = int(config.get("held_out_shard_start", 281))

    # The Arm-C (real SAE) latent per code, cached by build_feature_source. Read
    # rather than re-derived: re-deriving costs a full selection pass and could
    # pick a different latent if anything upstream moved.
    arm_c_path = Path(config["arm_c_selected_features_path"])
    feature_ids = json.loads(arm_c_path.read_text())["feature_ids"]

    target_rates = sae_note_level_densities(
        config["sae_shard_ckpt_dir"], feature_ids, held_out_shard_start=held_out_start
    )
    k = W.shape[1]
    if len(target_rates) != k:
        raise ValueError(
            f"Source has k={k} directions but the Arm-C reference supplies "
            f"{len(target_rates)} target rates. Note-level calibration matches "
            "each direction to one reference latent, so this path only applies "
            "to one-direction-per-code sources. Supply thresholds_npy instead."
        )

    token_sample, note_ids = sample_tokens_note_granular(config, held_out_start, log)
    theta, measured = calibrate_thresholds_note_level(W, token_sample, note_ids, target_rates)
    log.info(
        "Calibrated %d thresholds to the SAE's note-level rate "
        "(target median %.3f, achieved median %.3f)",
        k,
        float(np.median(target_rates)),
        float(np.median(measured)),
    )
    return theta.astype(np.float32)


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=32768,
    timeout=3600,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[hf_secret],
)
def import_source_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    import numpy as np

    from mech_interp_research.feature_sources import write_pseudo_sae

    logging.basicConfig(level=config.get("logging_level", "INFO"))
    log = logging.getLogger("import_direction_source")

    out_dir = Path(config["output_dir"])

    if config.get("hf_repo_id"):
        # An externally-trained SAE (GemmaScope) lives on the Hub, not on the
        # volume, so feature_inspector cannot load it directly. Re-expressing it
        # through the pseudo-SAE contract makes it just another local checkpoint
        # and every downstream stage works unmodified.
        #
        # Encoder convention matters here. GemmaScope was trained WITHOUT the
        # b_dec subtraction: its encode is `x @ W_enc + b_enc`. Our loader
        # defaults to `(x - b_dec) @ W_enc + b_enc`, so writing b_dec = 0 makes
        # the two identical and reproduces GemmaScope's encoder exactly. The
        # cost is that decode() and explained variance become meaningless, which
        # source_meta.json already flags — and the explainer path only encodes.
        from mech_interp_research.icd_eval import JumpReLUSAE

        src = JumpReLUSAE.from_huggingface(
            config["hf_repo_id"], config["hf_filename"], token=os.environ.get("HF_TOKEN")
        )
        W = np.asarray(src.W_enc, dtype=np.float32)
        theta = np.asarray(src.threshold, dtype=np.float32)
        b_enc = np.asarray(src.b_enc, dtype=np.float32)
        provenance = {
            "hf_repo_id": config["hf_repo_id"],
            "hf_filename": config["hf_filename"],
            "subtract_b_dec_original": bool(src.subtract_b_dec),
            "b_dec_norm_discarded": float(np.linalg.norm(src.b_dec)),
        }
        log.info(
            "Loaded %s/%s: W_enc %s, discarding b_dec (norm %.3f) to preserve its "
            "no-subtraction encoder",
            config["hf_repo_id"],
            config["hf_filename"],
            W.shape,
            provenance["b_dec_norm_discarded"],
        )
    else:
        directions_path = Path(config["directions_npy"])
        W = np.load(directions_path).astype(np.float32)
        b_enc = None
        provenance = {"imported_from": str(directions_path)}

        thresholds_cfg = config.get("thresholds_npy")
        if thresholds_cfg:
            theta = np.load(Path(thresholds_cfg)).astype(np.float32)
            provenance["thresholds_from"] = str(thresholds_cfg)
        else:
            # A source that shipped directions but no thresholds (the whitened
            # diff-in-means and probe arms) is calibrated here rather than being
            # rebuilt. Rebuilding would produce DIFFERENT vectors -- the existing
            # audit's correlation matrices would then describe directions the
            # checkpoint does not contain, and the r values quoted to the judge
            # would silently belong to something else.
            #
            # Calibration target is the reference SAE's NOTE-level detection
            # rate, not a token-level density: correlation is computed after
            # max-pooling, so note-level rate is the quantity that has to match.
            # Identical to the path build_feature_source uses, reusing its token
            # sampler so the two cannot drift.
            theta = _calibrate_from_sae(config, W, log)
            provenance["thresholds_from"] = "calibrated_note_level_at_import"

    # A [k, d_model] matrix is the transpose of what the contract wants; accept
    # it rather than silently writing a mis-shaped encoder.
    expected_d_model = int(config.get("expect_d_model", 2304))
    if W.shape[0] != expected_d_model and W.shape[1] == expected_d_model:
        log.warning("directions are [k, d_model]=%s; transposing to [d_model, k]", W.shape)
        W = np.ascontiguousarray(W.T)
    if W.shape[0] != expected_d_model:
        raise ValueError(
            f"directions {W.shape} has neither axis equal to d_model={expected_d_model}"
        )
    if theta.shape != (W.shape[1],):
        raise ValueError(f"thresholds {theta.shape} do not match direction count k={W.shape[1]}")

    # Non-finite thresholds are a real hazard, not a formality: `pre > NaN` is
    # always False, so a NaN theta yields a permanently dead direction that is
    # indistinguishable from a legitimately selective one. write_pseudo_sae
    # rejects them outright. A source may still legitimately carry some — a
    # direction whose pooled maxima are degenerate has no calibratable
    # threshold — so `inert` makes those directions provably never fire while
    # preserving k, which the best-of-k selection rule depends on.
    n_nonfinite = int((~np.isfinite(theta)).sum())
    policy = str(config.get("nonfinite_threshold_policy", "reject"))
    if n_nonfinite and policy == "inert":
        inert = float(np.finfo(np.float32).max)
        log.warning(
            "%d/%d thresholds are non-finite; setting them to %.3g so those "
            "directions never fire (k preserved).",
            n_nonfinite,
            theta.size,
            inert,
        )
        theta = np.where(np.isfinite(theta), theta, inert).astype(np.float32)
    elif n_nonfinite:
        raise ValueError(
            f"{n_nonfinite}/{theta.size} thresholds are non-finite. Set "
            "nonfinite_threshold_policy: inert to make those directions never "
            "fire, after confirming none of them is a selected feature."
        )

    # The contract requires theta >= 0: encoding is z = pre * (pre > theta), so a
    # negative threshold passes negative pre-activations through unchanged.
    n_negative = int((theta < 0).sum())
    if n_negative:
        log.warning("clamping %d negative thresholds to 0", n_negative)
        theta = np.maximum(theta, 0.0).astype(np.float32)

    meta: dict[str, Any] = {
        "arm": config.get("arm", "imported"),
        **provenance,
        "n_negative_thresholds_clamped": n_negative,
        "n_nonfinite_thresholds_made_inert": n_nonfinite if policy == "inert" else 0,
        "provenance_note": config.get("provenance_note", ""),
    }

    # Carry the producing run's manifest through, so the checkpoint is traceable
    # back to the seed and construction that made it.
    manifest_path = config.get("source_manifest_json")
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            meta["source_manifest"] = json.load(f)

    write_pseudo_sae(W, theta, out_dir, meta, b_enc=b_enc)
    artifacts_volume.commit()

    log.info("Imported source written: %s (d_model=%d, k=%d)", out_dir, *W.shape)
    return {"output_dir": str(out_dir), "d_model": int(W.shape[0]), "k": int(W.shape[1])}


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if detach:
        call = import_source_remote.spawn(config)
        print(f"Spawned detached: {call.object_id}")
        return
    print(json.dumps(import_source_remote.remote(config), indent=2))
