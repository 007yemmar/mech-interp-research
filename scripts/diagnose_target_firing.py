"""Do the ABLATION TARGET directions fire on clinical tokens, or only on BOS?

Follow-up to diagnose_pseudo_sae_recon.py, which found non-BOS median L0 = 0 for
both pseudo-SAE sources. Median 0 is compatible with anything from "fires on 40%
of tokens" to "fires only on BOS", and the difference decides whether the
random-matched / diff-in-means ablation arms are measuring anything at all:
if a target direction only clears its threshold at BOS, ablating it removes no
clinical signal and the arm is void regardless of the fp16 fix.

Reports, per target direction, over real notes:
  * fraction of NOTES where it fires at all (should match its calibrated
    note-level density)
  * fraction of notes where its ONLY firing token is BOS   <- the killer number
  * firing tokens per note, excluding BOS
  * where the pooled max sits (BOS vs elsewhere), since the grounding
    correlation was computed on that max

Run:
    modal run scripts/diagnose_target_firing.py
    modal run scripts/diagnose_target_firing.py \
        --source-dir /out/sources/dim_full --targets 43,3,37,39
"""

from __future__ import annotations

from modal_app.app import app, artifacts_volume, image


@app.function(image=image, cpu=8, memory=32768, timeout=1800, volumes={"/out": artifacts_volume})
def diagnose(
    source_dir: str = "/out/sources/random_matched_note",
    targets: str = "14681,11717,601,15330",
    activations_dir: str = (
        "/out/activations/" "google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
    ),
    shard_idx: int = 281,
    n_notes: int = 60,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    from safetensors.numpy import load_file

    idxs = [int(t) for t in targets.split(",")]
    w = load_file(str(Path(source_dir) / "sae_weights.safetensors"))
    W_enc = w["W_enc"].astype(np.float32)[:, idxs]  # [d_model, n_targets]
    b_enc = w["b_enc"].astype(np.float32)[idxs]
    theta = w["threshold"].astype(np.float32)[idxs]
    print(f"source={source_dir}  targets={idxs}")
    print(f"thresholds={np.round(theta, 3).tolist()}\n")

    meta_path = Path(activations_dir) / "metadata.jsonl"
    rows = [json.loads(x) for x in open(meta_path)]
    rows = [r for r in rows if int(r.get("shard", -1)) == shard_idx][:n_notes]
    acts = load_file(str(Path(activations_dir) / f"shard_{shard_idx:04d}.safetensors"))
    X = acts[next(iter(acts))].astype(np.float32)

    n_t = len(idxs)
    fires_any = np.zeros(n_t, int)
    fires_bos_only = np.zeros(n_t, int)
    nonbos_counts = [[] for _ in range(n_t)]
    argmax_is_bos = np.zeros(n_t, int)

    for r in rows:
        start = int(r.get("row_start", r.get("start", 0)))
        n_tok = int(r.get("num_tokens_truncated", r.get("n_tokens", 0)))
        if n_tok < 8:
            continue
        pre = X[start : start + n_tok] @ W_enc + b_enc  # [T, n_targets]
        fired = pre > theta  # [T, n_targets]
        for j in range(n_t):
            f = fired[:, j]
            if f.any():
                fires_any[j] += 1
                nb = int(f[1:].sum())
                nonbos_counts[j].append(nb)
                if nb == 0:
                    fires_bos_only[j] += 1
            if int(np.argmax(pre[:, j])) == 0:
                argmax_is_bos[j] += 1

    n = len([r for r in rows if int(r.get("num_tokens_truncated", r.get("n_tokens", 0))) >= 8])
    print(f"notes evaluated: {n}\n")
    out = []
    for j, idx in enumerate(idxs):
        nb = np.array(nonbos_counts[j]) if nonbos_counts[j] else np.array([0])
        rec = dict(
            direction=idx,
            note_density=fires_any[j] / n,
            frac_bos_only=fires_bos_only[j] / max(fires_any[j], 1),
            nonbos_tokens_median=float(np.median(nb)),
            nonbos_tokens_max=int(nb.max()),
            frac_pooled_max_at_bos=argmax_is_bos[j] / n,
        )
        out.append(rec)
        print(f"direction {idx}")
        print(f"   fires in            {rec['note_density']:.3f} of notes")
        print(f"   of those, BOS-only  {rec['frac_bos_only']:.3f}   <-- 1.0 means void")
        print(
            f"   non-BOS firing tokens/note: median {rec['nonbos_tokens_median']:.1f}"
            f"  max {rec['nonbos_tokens_max']}"
        )
        print(f"   pooled max sits at BOS in {rec['frac_pooled_max_at_bos']:.3f} of notes\n")

    worst = max(r["frac_bos_only"] for r in out)
    print("=" * 62)
    if worst > 0.9:
        print("VERDICT: targets fire essentially only at BOS. The arm measures BOS,")
        print("         not clinical content. Fixing fp16 will not rescue it —")
        print("         thresholds/pooling need rework before any ablation run.")
    elif worst > 0.3:
        print("VERDICT: substantial BOS-only firing. Usable only with BOS excluded")
        print("         from BOTH the threshold calibration and the ablation.")
    else:
        print("VERDICT: targets fire on real tokens. Proceed with the fp16 fix.")
    return {"targets": out}


@app.local_entrypoint()
def main(
    source_dir: str = "/out/sources/random_matched_note",
    targets: str = "14681,11717,601,15330",
    shard_idx: int = 281,
    n_notes: int = 60,
) -> None:
    diagnose.remote(source_dir=source_dir, targets=targets, shard_idx=shard_idx, n_notes=n_notes)
