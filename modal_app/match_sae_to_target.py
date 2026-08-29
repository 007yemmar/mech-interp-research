"""Select, per code, the SAE latent whose |r| is CLOSEST to a target |r|. CPU-only.

Why this exists
---------------
The sensitivity control (Arm B2) was specified as "blend the keyword direction
with an unrelated one until its correlation drops to match the SAE's", so that
any remaining concordance gap is about *what kind of object* a direction is
rather than how strongly it correlates. Dilution can only push |r| DOWN.

Measured after the BOS fix, the keyword directions sit at median |r| 0.375 while
the SAE's argmax latents sit at 0.580. The keyword arm is already *below* the
SAE, so there is nothing to dilute: the solve reports "unreachable" for 44 of 46
codes and every alpha stays 0. G4 is unevaluable in that direction.

Matching the other way round always works. For each code, instead of weakening
the keyword direction to the SAE's level, pick the SAE latent whose |r| is
nearest the keyword direction's. That is a selection over a correlation matrix
we already have — no new checkpoint, no re-encode, and it cannot fail to
converge.

Split discipline
----------------
Matching is a SELECTION decision, so it runs on selection shards only
(< held_out_shard_start), exactly like the argmax selection it replaces.
Auditing the matched latents on held-out notes stays the caller's job. Doing the
match on audit data would choose latents using the same notes it later reports,
which is the bias the whole split exists to remove.

Run:
    uv run modal run modal_app/match_sae_to_target.py \\
        --config-file configs/match_sae_to_keyword.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "8"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=65536,
    timeout=7200,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[hf_secret],
)
def match_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    import numpy as np

    from mech_interp_research.icd_eval import (
        _align_note_vectors_to_matched,
        compute_point_biserial_vectorised,
        load_and_align_icd_labels,
        reassemble_note_vectors,
    )
    from mech_interp_research.necessity_stats import split_by_shard

    logging.basicConfig(level=config.get("logging_level", "INFO"))
    log = logging.getLogger("match_sae_to_target")

    held_out_start = int(config.get("held_out_shard_start", 281))

    vectors, note_meta = reassemble_note_vectors(config["sae_shard_ckpt_dir"])
    Y, code_names, matched_meta = load_and_align_icd_labels(
        Path(config["icd_csv_path"]),
        note_meta,
        min_prevalence=float(config.get("min_prevalence", 0.02)),
        max_codes=int(config.get("max_codes", 50)),
        icd_col_prefix=config.get("icd_col_prefix", "icd9_"),
        join_key=config.get("join_key", "admission_id"),
        min_notes=int(config.get("min_notes", 100)),
    )
    F = _align_note_vectors_to_matched(vectors, note_meta, matched_meta)
    sel, _ = split_by_shard(matched_meta, held_out_shard_start=held_out_start)
    log.info(
        "Matching on %d selection notes, %d latents, %d codes",
        int(sel.sum()),
        F.shape[1],
        len(code_names),
    )

    r_sel, _ = compute_point_biserial_vectorised(F[sel], np.asarray(Y, dtype=np.float64)[sel])
    abs_r = np.abs(r_sel)  # [k, n_codes]

    # Targets: the arm we are matching DOWN to, per code, in the same order.
    target_meta = json.loads(Path(config["target_source_meta"]).read_text())
    if target_meta.get("code_names") != code_names:
        raise ValueError(
            "Target source's code panel differs from the one loaded here. Both "
            "must come from the same population and alignment settings, or the "
            "per-code targets line up against the wrong codes."
        )
    targets = np.array(
        [np.nan if v is None else abs(float(v)) for v in target_meta[config["target_field"]]],
        dtype=float,
    )

    matched_ids: list[int | None] = []
    matched_r: list[float | None] = []
    for c in range(len(code_names)):
        t = targets[c]
        if not np.isfinite(t):
            matched_ids.append(None)
            matched_r.append(None)
            continue
        j = int(np.argmin(np.abs(abs_r[:, c] - t)))
        matched_ids.append(j)
        matched_r.append(float(abs_r[j, c]))

    errs = [
        abs(m - t)
        for m, t in zip(matched_r, targets, strict=True)
        if m is not None and np.isfinite(t)
    ]
    argmax_ids = [int(np.argmax(abs_r[:, c])) for c in range(len(code_names))]
    n_same = sum(1 for a, b in zip(argmax_ids, matched_ids, strict=True) if a == b)

    out = {
        "code_names": code_names,
        "matched_feature_ids": matched_ids,
        "matched_abs_r_selection": matched_r,
        "target_abs_r": [None if not np.isfinite(t) else float(t) for t in targets],
        "match_abs_error": errs,
        "median_match_abs_error": float(np.median(errs)) if errs else None,
        # How often the |r|-matched latent is simply the argmax one. A high count
        # means the matching changed little and the arm adds little.
        "n_identical_to_argmax": int(n_same),
        "argmax_feature_ids": argmax_ids,
        "n_selection_notes": int(sel.sum()),
        "held_out_shard_start": held_out_start,
        "target_source_meta": config["target_source_meta"],
        "target_field": config["target_field"],
        "sae_shard_ckpt_dir": config["sae_shard_ckpt_dir"],
    }

    out_path = Path(config["output_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    artifacts_volume.commit()

    log.info(
        "Matched %d/%d codes | median |r| error %.4f | identical to argmax: %d",
        sum(1 for m in matched_ids if m is not None),
        len(code_names),
        out["median_match_abs_error"] if errs else float("nan"),
        n_same,
    )
    return {
        k: out[k] for k in ("median_match_abs_error", "n_identical_to_argmax", "n_selection_notes")
    }


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if detach:
        call = match_remote.spawn(config)
        print(f"Spawned detached: {call.object_id}")
        return
    print(json.dumps(match_remote.remote(config), indent=2))
