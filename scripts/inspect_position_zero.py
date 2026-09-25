"""What is row 0 of each note's stored activations, and why is it pathological?

diagnose_target_firing.py showed several directions firing ONLY at row 0. I
called that "BOS", but the magnitudes differed across notes (2104217, 2148138,
1843808, ...), and a shared BOS token cannot differ: position 0 attends only to
itself, so its layer-16 residual would be byte-identical everywhere. Something
else is going on, and the fix depends on which:

  * row 0 identical across notes  -> a genuine constant BOS; excluding it from
    the splice and from threshold calibration is safe and principled.
  * row 0 varies across notes     -> it is NOT a shared special token. It may be
    the first real token, or row_start may be off by one. Excluding "position 0"
    would then be discarding real content, and the fix is different.

Reports:
  1. exact-duplicate groups among the row-0 vectors, and pairwise max |diff|
  2. norm of row 0 vs the note's other tokens
  3. the actual first tokens, by re-tokenising the note text with the same call
     signature extraction used (add_special_tokens=True, truncation, max_length)
  4. stored row count vs tokenised length, which catches an off-by-one row_start

Run:
    modal run scripts/inspect_position_zero.py
"""

from __future__ import annotations

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=1800,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[hf_secret],
)
def inspect(
    activations_dir: str = (
        "/out/activations/" "google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered"
    ),
    icd_csv_path: str = "/data/sample_50k.csv",
    model_name: str = "google/gemma-2-2b",
    text_col: str = "note_text",
    join_key: str = "admission_id",
    shard_idx: int = 281,
    n_notes: int = 8,
    max_length: int = 8192,
) -> dict:
    import json
    import os
    from pathlib import Path

    import numpy as np
    from safetensors.numpy import load_file

    rows = [json.loads(x) for x in open(Path(activations_dir) / "metadata.jsonl")]
    rows = [r for r in rows if int(r.get("shard", -1)) == shard_idx][:n_notes]
    print(f"metadata fields: {sorted(rows[0].keys())}\n")

    acts = load_file(str(Path(activations_dir) / f"shard_{shard_idx:04d}.safetensors"))
    X = acts[next(iter(acts))].astype(np.float32)

    # ---- 1 + 2. row-0 vectors: identical or not, and their magnitude ----
    v0, info = [], []
    for r in rows:
        s, e = int(r["row_start"]), int(r["row_end"])
        x = X[s:e]
        v0.append(x[0].copy())
        info.append(
            (
                r.get(join_key),
                s,
                e,
                e - s,
                float(np.linalg.norm(x[0])),
                float(np.median(np.linalg.norm(x[1:], axis=1))),
            )
        )
    V = np.stack(v0)

    print("=== row 0: identical across notes? ===")
    uniq = {V[i].tobytes(): [] for i in range(len(V))}
    for i in range(len(V)):
        uniq[V[i].tobytes()].append(i)
    print(f"  {len(V)} notes -> {len(uniq)} distinct row-0 vectors")
    for k, (_b, members) in enumerate(uniq.items()):
        print(f"    group {k}: notes {[info[i][0] for i in members]}")
    d = np.abs(V[:, None, :] - V[None, :, :]).max(axis=2)
    print(
        f"  pairwise max |diff|: min={d[d > 0].min() if (d > 0).any() else 0:.4g}"
        f"  max={d.max():.4g}\n"
    )

    print("=== magnitude: row 0 vs the rest ===")
    print(f"  {'note':>12} {'rows':>7} {'||row0||':>12} {'median||rest||':>15} {'ratio':>8}")
    for a, _s, _e, n, n0, nr in info:
        print(f"  {str(a):>12} {n:7d} {n0:12.1f} {nr:15.1f} {n0 / max(nr, 1e-9):8.1f}")
    print()

    # ---- 3 + 4. what are the first tokens, really ----
    print("=== first tokens by re-tokenising the source text ===")
    try:
        import pandas as pd
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        want = {str(r.get(join_key)) for r in rows}
        df = pd.read_csv(icd_csv_path, usecols=[join_key, text_col], dtype={join_key: str})
        df = df[df[join_key].astype(str).isin(want)]
        by_id = dict(zip(df[join_key].astype(str), df[text_col], strict=False))

        for a, _s, _e, n, _, _ in info[:4]:
            txt = by_id.get(str(a))
            if txt is None:
                print(f"  note {a}: text not found in {icd_csv_path}")
                continue
            ids = tok(txt, add_special_tokens=True, truncation=True, max_length=max_length)[
                "input_ids"
            ]
            print(
                f"  note {a}: stored_rows={n}  tokenised={len(ids)}"
                f"  {'MATCH' if n == len(ids) else 'MISMATCH <-- row_start suspect'}"
            )
            print(f"    first 5 ids   : {ids[:5]}")
            print(f"    first 5 tokens: {[tok.decode([i]) for i in ids[:5]]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (tokeniser/text step failed: {exc})")

    print("\n" + "=" * 62)
    if len(uniq) == 1:
        print("VERDICT: row 0 is a single constant vector -> genuine shared BOS.")
        print("         Excluding it from the splice and from threshold")
        print("         calibration is safe.")
    else:
        print("VERDICT: row 0 VARIES across notes -> not a shared special token.")
        print("         Check the token printout above before excluding anything;")
        print("         if stored_rows != tokenised, row_start is the real bug.")
    return {"n_notes": len(V), "n_distinct_row0": len(uniq)}


@app.local_entrypoint()
def main(shard_idx: int = 281, n_notes: int = 8) -> None:
    inspect.remote(shard_idx=shard_idx, n_notes=n_notes)
