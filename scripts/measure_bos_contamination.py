"""How much does the BOS token set the max-pooled value an SAE latent is grounded on?

Grounding correlates F[note, j] = max over tokens of z_j(t) against ICD labels.
Row 0 of every note is <bos>, whose layer-16 residual has ~15.6x the norm of a
typical token. So F is really max(c_j, real_max), where c_j is the latent's BOS
activation -- a FLOOR on the pooled value.

A floor is not automatically harmless. It shrinks the positive/negative mean gap
BUT also collapses within-negative variance to zero, and point-biserial is
(M1 - M0)/sigma: the second effect can dominate and INFLATE r. So "BOS only
floors, it cannot inflate" has to be checked, not assumed.

The decisive quantity per latent is:

    bos_sets_pool = fraction of notes where c_j >= real_max_j

  ~0  -> BOS never wins the max; grounding is uncontaminated; the published
         numbers stand and only the comparison arms need re-pooling.
  high on grounded latents -> the pooled value those correlations were computed
         on is partly a BOS artifact, and grounding must be recomputed.

Reported for ALL latents and, separately, for the top-grounded ones -- the
latter is what matters, since those are the features the paper reports and the
ablation targets.

Encoder convention mirrors icd_eval.JumpReLUSAE:
    subtract_b_dec=True   pre = (x - b_dec) @ W_enc + b_enc
    subtract_b_dec=False  pre =  x          @ W_enc + b_enc   (GemmaScope)
    z = pre * (pre > threshold)

Run:
    modal run scripts/measure_bos_contamination.py
    modal run scripts/measure_bos_contamination.py \
        --checkpoint-dir /out/saes/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/step_00036000 \
        --eval-dir /out/icd_eval/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z
"""

from __future__ import annotations

from modal_app.app import app, artifacts_volume, hf_secret, image


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=3600,
    volumes={"/out": artifacts_volume},
    secrets=[hf_secret],
)
def measure(
    checkpoint_dir: str = "/out/saes/sae_d2304_e8_l11e+01_20260505T205723Z/best",
    eval_dir: str = "/out/icd_eval/sae_d2304_e8_l11e+01_20260505T205723Z",
    activations_dir: str = (
        "/out/activations/" "google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
    ),
    subtract_b_dec: bool = True,
    shard_idx: int = 281,
    n_notes: int = 120,
    top_n: int = 60,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from mech_interp_research.icd_eval import JumpReLUSAE

    sae = JumpReLUSAE.from_checkpoint(checkpoint_dir)
    sae.subtract_b_dec = subtract_b_dec
    d_model, d_sae = sae.W_enc.shape
    print(
        f"SAE {checkpoint_dir}\n  d_model={d_model} d_sae={d_sae} "
        f"subtract_b_dec={subtract_b_dec}\n"
    )

    rows = [json.loads(x) for x in open(Path(activations_dir) / "metadata.jsonl")]
    rows = [r for r in rows if int(r.get("shard", -1)) == shard_idx][:n_notes]
    from safetensors.numpy import load_file

    acts = load_file(str(Path(activations_dir) / f"shard_{shard_idx:04d}.safetensors"))
    X = acts[next(iter(acts))].astype(np.float32)

    bos_wins = np.zeros(d_sae, np.int64)  # notes where c_j >= real_max_j
    c_all, n_eval = [], 0
    real_max_sum = np.zeros(d_sae, np.float64)

    for r in rows:
        s, e = int(r["row_start"]), int(r["row_end"])
        if e - s < 8:
            continue
        n_eval += 1
        z = sae.encode_chunked(X[s:e])  # [T, d_sae]
        c = z[0]  # BOS activation
        real_max = z[1:].max(axis=0)
        c_all.append(c)
        real_max_sum += real_max
        # BOS only *sets* the pool where it fires AND beats the real tokens.
        # A silent latent has c = real_max = 0; counting that as a BOS win
        # measures sparsity, not contamination, and makes sparse
        # domain-trained SAEs look worse than dense general-purpose ones.
        bos_wins += (c > 0) & (c >= real_max)

    C = np.stack(c_all)
    frac = bos_wins / max(n_eval, 1)
    c_med = np.median(C, axis=0)
    real_max_mean = real_max_sum / max(n_eval, 1)
    print(f"notes evaluated: {n_eval}\n")

    print("=== BOS activation across ALL latents ===")
    print(
        f"  latents that fire at all at BOS  : {(c_med > 0).sum():,} / {d_sae:,} "
        f"({(c_med > 0).mean():.1%})"
    )
    print(
        f"  latents where BOS sets the pool in >50% of notes: "
        f"{(frac > 0.5).sum():,} ({(frac > 0.5).mean():.1%})"
    )
    print(f"  mean fraction of notes BOS sets the pool: {frac.mean():.4f}")
    print(
        f"  c_j at BOS: median {np.median(c_med):.4f}  "
        f"p99 {np.quantile(c_med, 0.99):.4f}  max {c_med.max():.4f}\n"
    )

    # ---- the latents that actually matter: the grounded ones ----
    ta_path = Path(eval_dir) / "top_associations.csv"
    out = {"n_eval_notes": n_eval, "d_sae": int(d_sae), "frac_all_mean": float(frac.mean())}
    if ta_path.exists():
        ta = pd.read_csv(ta_path)
        ta = ta.reindex(ta["abs_r"].abs().sort_values(ascending=False).index)
        ta = ta.drop_duplicates(subset=["latent"], keep="first").head(top_n)
        idx = ta["latent"].astype(int).to_numpy()
        print(f"=== top {len(idx)} GROUNDED latents ({ta_path.name}) ===")
        print(
            f"  {'latent':>7} {'code':<12} {'r_pb':>8} {'c_j(BOS)':>10} "
            f"{'mean real_max':>14} {'BOS sets pool':>14}"
        )
        for _, row in ta.iterrows():
            j = int(row["latent"])
            print(
                f"  {j:7d} {str(row['code']):<12} {row['r_pb']:+8.4f} "
                f"{c_med[j]:10.4f} {real_max_mean[j]:14.4f} {frac[j]:13.3f}"
            )
        gf = frac[idx]
        print(
            f"\n  grounded latents: BOS sets the pool in {gf.mean():.4f} of "
            f"note-latent pairs (max over latents {gf.max():.3f})"
        )
        out.update(
            frac_grounded_mean=float(gf.mean()),
            frac_grounded_max=float(gf.max()),
            n_grounded_over_10pct=int((gf > 0.10).sum()),
            n_grounded=int(len(idx)),
        )
        print("\n" + "=" * 62)
        n_active = int((c_med[idx] > 0).sum())
        print(f"  grounded latents that fire at BOS at all: {n_active}/{len(idx)}")
        if n_active == 0:
            print()
            print("VERDICT: NO grounded latent fires at BOS (c_j = 0 for all).")
            print("         BOS cannot floor their pooled values. Grounding is")
            print("         uncontaminated; published numbers stand.")
        elif gf.max() < 0.02:
            print("VERDICT: BOS never sets the pooled value for grounded latents.")
            print("         Published SAE grounding stands. Re-pool only the")
            print("         comparison arms.")
        elif gf.mean() < 0.10:
            print("VERDICT: minor contamination, concentrated in a few latents.")
            print("         Report as a robustness check; list the affected latents.")
        else:
            print("VERDICT: grounded latents ARE partly BOS-driven. The published")
            print("         correlations need recomputing on a BOS-free pool.")
    else:
        print(f"(no top_associations.csv at {ta_path} — all-latent stats only)")
    return out


@app.local_entrypoint()
def main(
    checkpoint_dir: str = "/out/saes/sae_d2304_e8_l11e+01_20260505T205723Z/best",
    eval_dir: str = "/out/icd_eval/sae_d2304_e8_l11e+01_20260505T205723Z",
    subtract_b_dec: bool = True,
    n_notes: int = 120,
    top_n: int = 60,
) -> None:
    measure.remote(
        checkpoint_dir=checkpoint_dir,
        eval_dir=eval_dir,
        subtract_b_dec=subtract_b_dec,
        n_notes=n_notes,
        top_n=top_n,
    )
