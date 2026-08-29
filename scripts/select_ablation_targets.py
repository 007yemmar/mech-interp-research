"""Pick ablation targets for a pseudo-SAE source, screened for row-0-only firing.

diagnose_target_firing.py showed that some directions clear their note-level
threshold ONLY at row 0 of a note -- dim_full latent 3 (r=0.667) did so in 100%
of notes. Ablating such a direction removes nothing from the clinical text, so
its causal effect is uninterpretable no matter how strong its grounding looks.
Selecting targets by |r_pb| alone therefore is not safe for these sources.

This screens candidates on real activations and emits a ready-to-paste
``targets:`` block:

  grounded : top-N by |r_pb| from the audit's top_associations.csv that also
             fire on non-row-0 tokens in enough notes
  controls : one direction with max |r| < random_r_max and one with
             |r| in [low_r_min, low_r_max], both passing the same screen, drawn
             from the audit's correlation_matrices.npz

Screening is bounded: only the top_associations candidates plus a sample of
low-r directions are projected, never all k.

Run:
    modal run scripts/select_ablation_targets.py
    modal run scripts/select_ablation_targets.py \
        --source-dir /out/sources/dim_full \
        --audit-dir /out/necessity/direction_audit/diff_in_means_full \
        --n-grounded 10
"""

from __future__ import annotations

from modal_app.app import app, artifacts_volume, image


@app.function(image=image, cpu=8, memory=65536, timeout=3600, volumes={"/out": artifacts_volume})
def select(
    source_dir: str = "/out/sources/random_matched_note",
    audit_dir: str = "/out/necessity/random_matched/seed0/audit_note_matched",
    activations_dir: str = (
        "/out/activations/" "google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
    ),
    shard_idx: int = 281,
    n_notes: int = 80,
    n_grounded: int = 10,
    max_row0_only: float = 0.20,  # reject if row-0-only in >20% of firing notes
    min_note_density: float = 0.02,  # must fire in >=2% of notes to be ablatable
    max_note_density: float = 0.95,
    random_r_max: float = 0.05,
    low_r_min: float = 0.05,
    low_r_max: float = 0.10,
    n_low_r_candidates: int = 400,
    seed: int = 42,
) -> dict:
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from safetensors.numpy import load_file

    rng = np.random.default_rng(seed)
    src, aud = Path(source_dir), Path(audit_dir)

    w = load_file(str(src / "sae_weights.safetensors"))
    W_enc = w["W_enc"].astype(np.float32)
    b_enc = w["b_enc"].astype(np.float32)
    theta = w["threshold"].astype(np.float32)
    d_model, k = W_enc.shape

    code_names = json.load(open(aud / "code_names.json"))
    npz = np.load(aud / "correlation_matrices.npz")
    R = np.abs(npz["r_pb"]).astype(np.float32)  # [k, n_codes]
    r_signed = npz["r_pb"].astype(np.float32)
    max_r = R.max(axis=1)
    print(f"source={src.name}  k={k}  R={R.shape}  codes={len(code_names)}")

    ta = pd.read_csv(aud / "top_associations.csv")
    print(f"top_associations: {len(ta)} rows, cols={list(ta.columns)}\n")

    # ---- candidate set: grounded (deduped, |r| order) + low-r sample ----
    ta = ta.reindex(ta["abs_r"].abs().sort_values(ascending=False).index)
    ta = ta.drop_duplicates(subset=["latent"], keep="first")
    grounded_cand = [(int(r.latent), str(r.code), float(r.r_pb)) for r in ta.itertuples()]

    rand_pool = np.flatnonzero(max_r < random_r_max)
    low_pool = np.flatnonzero((max_r >= low_r_min) & (max_r <= low_r_max))
    print(
        f"pool sizes: max|r|<{random_r_max}: {rand_pool.size}   "
        f"|r| in [{low_r_min},{low_r_max}]: {low_pool.size}"
    )
    take = min(n_low_r_candidates // 2, rand_pool.size)
    rand_sample = rng.choice(rand_pool, size=take, replace=False) if take else np.array([], int)
    take = min(n_low_r_candidates // 2, low_pool.size)
    low_sample = rng.choice(low_pool, size=take, replace=False) if take else np.array([], int)

    cand = list(
        dict.fromkeys([c[0] for c in grounded_cand] + rand_sample.tolist() + low_sample.tolist())
    )
    print(f"screening {len(cand)} candidate directions on {n_notes} notes\n")

    # ---- screen ----
    rows = [json.loads(x) for x in open(Path(activations_dir) / "metadata.jsonl")]
    rows = [r for r in rows if int(r.get("shard", -1)) == shard_idx][:n_notes]
    acts = load_file(str(Path(activations_dir) / f"shard_{shard_idx:04d}.safetensors"))
    X = acts[next(iter(acts))].astype(np.float32)

    Wc, bc, tc = W_enc[:, cand], b_enc[cand], theta[cand]
    n_c = len(cand)
    fires_any = np.zeros(n_c, int)
    row0_only = np.zeros(n_c, int)
    nonrow0_notes = np.zeros(n_c, int)
    n_eval = 0
    for r in rows:
        s, e = int(r["row_start"]), int(r["row_end"])
        if e - s < 8:
            continue
        n_eval += 1
        fired = (X[s:e] @ Wc + bc) > tc  # [T, n_c]
        any_f = fired.any(axis=0)
        nb = fired[1:].any(axis=0)
        fires_any += any_f
        nonrow0_notes += nb
        row0_only += any_f & ~nb

    density = fires_any / max(n_eval, 1)
    frac_row0_only = row0_only / np.maximum(fires_any, 1)
    pos = {c: i for i, c in enumerate(cand)}

    def passes(idx: int) -> tuple[bool, str]:
        i = pos[idx]
        if fires_any[i] == 0:
            return False, "never fires"
        if frac_row0_only[i] > max_row0_only:
            return False, f"row0-only {frac_row0_only[i]:.2f}"
        if not (min_note_density <= density[i] <= max_note_density):
            return False, f"density {density[i]:.3f}"
        return True, "ok"

    # ---- grounded picks ----
    chosen, rejected = [], []
    for idx, code, r_pb in grounded_cand:
        ok, why = passes(idx)
        (chosen if ok else rejected).append(
            (idx, code, r_pb, why, density[pos[idx]], frac_row0_only[pos[idx]])
        )
        if len(chosen) >= n_grounded:
            break
    print("=== grounded ===")
    for idx, code, r_pb, _why, den, r0 in chosen:
        print(f"  KEEP  {idx:>6} {code:<12} r={r_pb:+.4f}  density={den:.3f} row0only={r0:.2f}")
    for idx, code, r_pb, why, _den, _r0 in rejected:
        print(f"  drop  {idx:>6} {code:<12} r={r_pb:+.4f}  ({why})")

    # ---- controls ----
    def pick_control(pool: np.ndarray, kind: str):
        for idx in pool:
            idx = int(idx)
            if idx not in pos:
                continue
            ok, _ = passes(idx)
            if ok:
                c = int(np.argmax(R[idx]))
                return dict(
                    feature_idx=idx,
                    code=code_names[c],
                    kind=kind,
                    r_pb=round(float(r_signed[idx, c]), 4),
                    density=round(float(density[pos[idx]]), 3),
                )
        return None

    ctrls = [
        c
        for c in (
            pick_control(rand_sample, "random_control"),
            pick_control(low_sample, "low_r_control"),
        )
        if c
    ]
    print("\n=== controls ===")
    for c in ctrls:
        print(
            f"  {c['kind']:<15} {c['feature_idx']:>6} {c['code']:<12} "
            f"r={c['r_pb']:+.4f} density={c['density']}"
        )

    # ---- emit ----
    print("\n" + "=" * 62)
    print("targets:")
    for idx, code, r_pb, _, _, _ in chosen:
        print(
            f"  - feature_idx: {idx}\n    code: {code}\n"
            f"    kind: grounded\n    r_pb: {round(r_pb, 4)}"
        )
    for c in ctrls:
        print(
            f"  - feature_idx: {c['feature_idx']}\n    code: {c['code']}\n"
            f"    kind: {c['kind']}\n    r_pb: {c['r_pb']}"
        )
    if len(chosen) < n_grounded:
        print(
            f"\nWARNING: only {len(chosen)}/{n_grounded} grounded targets survived "
            "the screen. Widen the candidate pool or relax max_row0_only."
        )
    return {"n_grounded": len(chosen), "n_controls": len(ctrls), "n_eval_notes": n_eval}


@app.local_entrypoint()
def main(
    source_dir: str = "/out/sources/random_matched_note",
    audit_dir: str = "/out/necessity/random_matched/seed0/audit_note_matched",
    n_grounded: int = 10,
    n_notes: int = 80,
) -> None:
    select.remote(
        source_dir=source_dir, audit_dir=audit_dir, n_grounded=n_grounded, n_notes=n_notes
    )
