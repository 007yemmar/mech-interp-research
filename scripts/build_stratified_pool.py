"""Emit the stratified judge pool (40 latents per |r| band) as explicit_features.

Build a pool from an eval's held-out correlation matrix:

    uv run python scripts/build_stratified_pool.py \
        --npz correlation_matrices.npz --codes code_names.json --out pool.yaml

Check that an existing pool follows the same rule (pairing + banding):

    uv run python scripts/build_stratified_pool.py \
        --npz correlation_matrices.npz --codes code_names.json \
        --check-against configs/judge_vanilla_strat.yaml

Both files come from the eval dir on the sae-artifacts volume
(`modal volume get sae-artifacts <eval>/correlation_matrices.npz .`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from mech_interp_research.stratified_pool import BANDS, best_code_per_latent, sample_stratified


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True, help="correlation_matrices.npz")
    ap.add_argument("--codes", required=True, help="code_names.json (list)")
    ap.add_argument("--out", help="write the explicit_features YAML line here")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-band", type=int, default=40)
    ap.add_argument("--check-against", help="config whose explicit_features to validate")
    args = ap.parse_args()

    z = np.load(args.npz)
    code_names = json.loads(Path(args.codes).read_text())
    abs_r, codes = best_code_per_latent(z["r_pb"], z["significant"], code_names)

    if args.check_against:
        cfg = yaml.safe_load(Path(args.check_against).read_text())
        pairs = cfg["explicit_features"]
        bad = [
            (f, c)
            for f, c in pairs
            if codes[f] != c or not any(lo <= abs_r[f] < hi for lo, hi in BANDS)
        ]
        print(f"{len(pairs) - len(bad)}/{len(pairs)} pairs follow the argmax-|r| + band rule")
        if bad:
            print(f"mismatches (first 10): {bad[:10]}")
            raise SystemExit(1)
        return

    if not args.out:
        ap.error("--out is required unless --check-against is given")
    pool = sample_stratified(abs_r, codes, per_band=args.per_band, seed=args.seed)
    Path(args.out).write_text(
        "explicit_features: " + json.dumps([[f, c] for f, c in pool]) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(pool)} features -> {args.out}")


if __name__ == "__main__":
    main()
