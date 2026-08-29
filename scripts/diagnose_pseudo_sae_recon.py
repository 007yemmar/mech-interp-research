"""Diagnose NaN reconstruction loss for a pseudo-SAE source. CPU only, ~2 min.

The random-matched smoke returned mean_loss_recon = NaN while mean_loss_clean
was correct, which means the failure is in encode -> decode -> splice, not in
the model or the CE. This script reproduces that path on cached activations
without a Gemma forward, and reports the three numbers that identify the cause:

  * L0 per token          -- how many directions fire
  * max |x_hat| per token -- fp16 overflows above 65504
  * BOS vs non-BOS split  -- Gemma's BOS residual is pathologically large, and
                             build_feature_source.py:411 already documents it
                             dominating the max-pool for constructed directions

If max |x_hat| at token 0 exceeds 65504 while non-BOS tokens stay well under,
the diagnosis is confirmed: ablation.py:835 casts x_hat to layer_dtype (fp16 on
cuda, model.py:38), inf propagates through layers 17+, and every note returns
NaN.

Run:
    modal run scripts/diagnose_pseudo_sae_recon.py
"""

from __future__ import annotations

from modal_app.app import app, artifacts_volume, image


@app.function(image=image, cpu=8, memory=32768, timeout=1800, volumes={"/out": artifacts_volume})
def diagnose(
    source_dir: str = "/out/sources/random_matched_note",
    activations_dir: str = (
        "/out/activations/" "google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
    ),
    shard_idx: int = 281,
    n_notes: int = 5,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    from safetensors.numpy import load_file

    FP16_MAX = 65504.0

    w = load_file(str(Path(source_dir) / "sae_weights.safetensors"))
    W_enc = w["W_enc"].astype(np.float32)  # [d_model, k]
    b_enc = w["b_enc"].astype(np.float32)
    W_dec = w["W_dec"].astype(np.float32)  # [k, d_model]
    b_dec = w["b_dec"].astype(np.float32)
    theta = w["threshold"].astype(np.float32)
    d_model, k = W_enc.shape
    print(f"source={source_dir}  d_model={d_model}  k={k}")
    print(
        f"threshold: finite={np.isfinite(theta).sum()}/{k} "
        f"min={theta[np.isfinite(theta)].min():.4g} "
        f"max={theta[np.isfinite(theta)].max():.4g} "
        f"n_at_fp32max={(theta > 1e37).sum()}"
    )

    # metadata.jsonl gives per-note row ranges inside the shard
    meta_path = Path(activations_dir) / "metadata.jsonl"
    rows = [json.loads(line) for line in open(meta_path)]
    rows = [r for r in rows if int(r.get("shard", -1)) == shard_idx][:n_notes]
    if not rows:
        raise SystemExit(f"no notes found for shard {shard_idx} in {meta_path}")

    acts = load_file(str(Path(activations_dir) / f"shard_{shard_idx:04d}.safetensors"))
    key = next(iter(acts))
    X = acts[key].astype(np.float32)  # [n_tokens_in_shard, d_model]
    print(f"shard {shard_idx}: {X.shape}, notes inspected: {len(rows)}\n")

    out = []
    for r in rows:
        start = int(r.get("row_start", r.get("start", 0)))
        n_tok = int(r.get("num_tokens_truncated", r.get("n_tokens", 0)))
        x = X[start : start + n_tok]  # [T, d_model]
        pre = x @ W_enc + b_enc  # [T, k]
        z = pre * (pre > theta)
        x_hat = z @ W_dec + b_dec  # [T, d_model]

        l0 = (z != 0).sum(axis=1)
        amax = np.abs(x_hat).max(axis=1)
        over = amax > FP16_MAX

        out.append(
            dict(
                note=r.get("admission_id"),
                n_tokens=int(n_tok),
                l0_bos=int(l0[0]),
                l0_rest_med=float(np.median(l0[1:])),
                amax_bos=float(amax[0]),
                amax_rest_max=float(amax[1:].max()),
                n_tokens_over_fp16=int(over.sum()),
                bos_over=bool(over[0]),
            )
        )
        print(f"note {r.get('admission_id')}  T={n_tok}")
        print(f"   L0      BOS={l0[0]:6d}   non-BOS median={np.median(l0[1:]):8.1f}")
        print(
            f"   max|xh| BOS={amax[0]:12.1f}   non-BOS max={amax[1:].max():12.1f}"
            f"   (fp16 max {FP16_MAX})"
        )
        print(
            f"   tokens over fp16 range: {int(over.sum())}/{n_tok}"
            f"   BOS over: {bool(over[0])}\n"
        )

    n_bos_over = sum(o["bos_over"] for o in out)
    n_any_over = sum(o["n_tokens_over_fp16"] > 0 for o in out)
    print("=" * 62)
    print(f"BOS over fp16 range in {n_bos_over}/{len(out)} notes")
    print(f"any token over fp16 range in {n_any_over}/{len(out)} notes")
    if n_bos_over == len(out) and all(o["amax_rest_max"] < FP16_MAX for o in out):
        print("VERDICT: BOS-only fp16 overflow -> fix = pass BOS through unspliced")
    elif n_any_over:
        print("VERDICT: overflow beyond BOS -> fix = bf16 layer dtype, not a BOS patch")
    else:
        print("VERDICT: no overflow; NaN comes from elsewhere (check mu / b_dec path)")
    return {"notes": out, "n_bos_over": n_bos_over, "n_any_over": n_any_over}


@app.local_entrypoint()
def main(
    source_dir: str = "/out/sources/random_matched_note",
    shard_idx: int = 281,
    n_notes: int = 5,
) -> None:
    diagnose.remote(source_dir=source_dir, shard_idx=shard_idx, n_notes=n_notes)
