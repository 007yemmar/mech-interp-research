"""Build a feature source as a pseudo-SAE checkpoint. CPU-only.

Arms:
  diff_in_means  — V1/V2/V3 directions from pooled raw activations + labels
  keyword_b1     — undiluted keyword-mean token directions
  keyword_b2     — B1 diluted with an unrelated cross-chapter code's direction
                   until its selection-set on-target |r| matches Arm C's for
                   that code

(Arm B3 — a bundle of diagnosis/drug/lab keyword directions — was cut before
implementation; see docs/superpowers/plans/2026-08-27-four-arm-concordance-
validation.md, "Arm B3 dropped per spec Sec 1".)

Every arm is calibrated to the reference JumpReLU SAE's NOTE-level detection
rate (spec Sec 5.5), not its token-level firing rate: notes average ~3,089
tokens, so a direction calibrated to a 0.222% token density fires on ~99.97%
of notes, while the real SAE's selected latents fire on 67.5% of notes on
average — a ~400x mismatch on the quantity the audit pipeline actually scores
(max-pooled, per-note activations). `sae_note_level_densities` measures the
real target directly from the reference SAE's own pooled checkpoints; the
token-level density is still computed and recorded, but only for audit — see
`source_meta.json`'s `token_level_*_reported_only` keys.

Run:
    uv run modal run modal_app/build_feature_source.py --config-file configs/source_diff_in_means.yaml
    uv run modal run modal_app/build_feature_source.py --config-file configs/source_keyword_b1.yaml
    uv run modal run modal_app/build_feature_source.py --config-file configs/source_keyword_b2.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "8"))


# ---------------------------------------------------------------------------
# Arm C selection pass — shared across every constructed arm.
#
# Every arm's threshold is calibrated to match the reference JumpReLU SAE's
# note-level detection rate for its selected (latent, code) pair (Ruling 1 /
# spec Sec 5.5), and B2's dilution solve targets that same pair's on-target
# |r| (spec Sec 4.2.2). Both need the SAME 46 (latent, code) selections Arm C
# itself would make: argmax|r| per code, computed on SELECTION notes only
# (shards < held_out_shard_start) so the calibration target never touches
# held-out data. This mirrors Arm C's own selection rule exactly, so every
# arm is calibrated against a fair, non-leaked version of "how hard is this
# code, really" (docs/superpowers/plans/.../"Known gaps" note: this selection
# pass belongs to Task 8; folded in here rather than assuming a pre-existing
# artifact).
#
# Written once to a shared location and reused by every subsequent arm build
# — the selection is deterministic (argmax has no randomness), so caching is
# purely to avoid repeating the SAE-side pooled reassembly + join per arm and
# to keep one canonical, auditable copy on disk.
# ---------------------------------------------------------------------------
def _arm_c_selected_features(
    config: dict[str, Any], held_out_start: int, log: Any
) -> dict[str, Any]:
    """Arm C's own best-of-18432 selection, restricted to selection notes.

    Returns (and caches to disk) a dict with ``code_names``, ``feature_ids``
    (one latent index per code — the argmax over the reference SAE's latents
    on SELECTION notes), ``r_selection`` (the corresponding signed r_pb per
    code), and ``n_selection_notes``.
    """
    shared_path = Path(
        config.get(
            "arm_c_selected_features_path",
            "/out/sources/_shared/arm_c_selected_features.json",
        )
    )
    if shared_path.exists():
        log.info("Reusing cached Arm C selection: %s", shared_path)
        return json.loads(shared_path.read_text())

    import numpy as np

    from mech_interp_research.icd_eval import (
        _align_note_vectors_to_matched,
        compute_point_biserial_vectorised,
        load_and_align_icd_labels,
        reassemble_note_vectors,
    )
    from mech_interp_research.necessity_stats import select_feature_per_code, split_by_shard

    sae_shard_ckpt_dir = config["sae_shard_ckpt_dir"]
    vectors, note_meta = reassemble_note_vectors(sae_shard_ckpt_dir)
    Y, code_names, matched_meta = load_and_align_icd_labels(
        Path(config["icd_csv_path"]),
        note_meta,
        min_prevalence=float(config.get("min_prevalence", 0.02)),
        max_codes=int(config.get("max_codes", 50)),
        icd_col_prefix=config.get("icd_col_prefix", "icd9_"),
        join_key=config.get("join_key", "admission_id"),
        min_notes=int(config.get("min_notes", 100)),
    )
    X = _align_note_vectors_to_matched(vectors, note_meta, matched_meta)
    sel, _ = split_by_shard(matched_meta, held_out_shard_start=held_out_start)

    r_pb_sel, _ = compute_point_biserial_vectorised(X[sel], np.asarray(Y, dtype=np.float64)[sel])
    feature_ids = select_feature_per_code(r_pb_sel)
    r_selection = [float(r_pb_sel[fid, c]) for c, fid in enumerate(feature_ids)]

    result = {
        "code_names": list(code_names),
        "feature_ids": feature_ids,
        "r_selection": r_selection,
        "n_selection_notes": int(sel.sum()),
        "sae_shard_ckpt_dir": str(sae_shard_ckpt_dir),
        "held_out_shard_start": int(held_out_start),
    }
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(json.dumps(result, indent=2))
    log.info(
        "Wrote Arm C selection: %s (%d codes, %d selection notes)",
        shared_path,
        len(code_names),
        int(sel.sum()),
    )
    return result


def _sample_tokens(config: dict[str, Any], held_out_start: int, log: Any):
    """Random token rows + their note ids, from SELECTION shards.

    Returns (token_sample, note_ids): note_ids[i] is the note_idx (from
    metadata.jsonl) that token_sample[i] belongs to, derived directly from
    each shard's row_start/row_end per note rather than reconstructed after
    the fact. calibrate_thresholds_note_level needs this to group sampled
    tokens by note when computing note-level detection rates (Ruling 1).
    """
    import numpy as np
    from safetensors.numpy import load_file

    from mech_interp_research.icd_eval import load_metadata

    acts = Path(config["activations_dir"])
    n_shards = int(config.get("calibration_n_shards", 6))
    per_shard = int(config.get("calibration_tokens_per_shard", 40_000))
    rng = np.random.default_rng(int(config.get("seed", 42)))

    metadata = load_metadata(acts)
    shard_files = sorted(acts.glob("shard_*.safetensors"))
    eligible = [p for p in shard_files if int(p.stem.split("_")[1]) < held_out_start]
    chosen = rng.choice(len(eligible), size=min(n_shards, len(eligible)), replace=False)

    token_chunks = []
    note_id_chunks = []
    for i in sorted(chosen):
        shard_path = eligible[i]
        shard_idx = int(shard_path.stem.split("_")[1])
        arr = load_file(str(shard_path))["activations"].astype(np.float32)
        take = min(per_shard, arr.shape[0])
        idx = rng.choice(arr.shape[0], size=take, replace=False)

        row_note_id = np.full(arr.shape[0], -1, dtype=np.int64)
        for _, note_row in metadata[metadata["shard"] == shard_idx].iterrows():
            row_note_id[int(note_row["row_start"]) : int(note_row["row_end"])] = int(
                note_row["note_idx"]
            )

        sampled_note_ids = row_note_id[idx]
        keep = sampled_note_ids >= 0
        token_chunks.append(arr[idx][keep])
        note_id_chunks.append(sampled_note_ids[keep])

    token_sample = np.concatenate(token_chunks, axis=0)
    note_ids = np.concatenate(note_id_chunks, axis=0)
    log.info(
        "Calibration token sample: %s rows across %d notes",
        token_sample.shape,
        len(set(note_ids.tolist())),
    )
    return token_sample, note_ids


def _load_pooled_and_labels(config: dict[str, Any], held_out_start: int):
    """Pooled note vectors + aligned labels, split into selection / audit."""
    import numpy as np

    from mech_interp_research.icd_eval import (
        _align_note_vectors_to_matched,
        load_and_align_icd_labels,
        reassemble_note_vectors,
    )
    from mech_interp_research.necessity_stats import split_by_shard

    vectors, note_meta = reassemble_note_vectors(config["pooled_ckpt_dir"])
    Y, code_names, matched_meta = load_and_align_icd_labels(
        Path(config["icd_csv_path"]),
        note_meta,
        min_prevalence=float(config.get("min_prevalence", 0.02)),
        max_codes=int(config.get("max_codes", 50)),
        icd_col_prefix=config.get("icd_col_prefix", "icd9_"),
        join_key=config.get("join_key", "admission_id"),
        min_notes=int(config.get("min_notes", 100)),
    )
    X = _align_note_vectors_to_matched(vectors, note_meta, matched_meta)
    sel, aud = split_by_shard(matched_meta, held_out_shard_start=held_out_start)
    return X, np.asarray(Y, dtype=np.float64), code_names, matched_meta, sel, aud


def _build_diff_in_means(config: dict[str, Any], held_out_start: int, log: Any):
    from mech_interp_research.feature_sources import build_diff_in_means_variants

    X, Y, code_names, _, sel, _ = _load_pooled_and_labels(config, held_out_start)
    variant = config.get("variant", "v2_zscored")
    W = build_diff_in_means_variants(X[sel], Y[sel], variant=variant)
    log.info("diff-in-means %s: W %s over %d selection notes", variant, W.shape, int(sel.sum()))
    return W, {"variant": variant, "code_names": code_names, "n_selection_notes": int(sel.sum())}


# ---------------------------------------------------------------------------
# Keyword arms (B1 / B2)
# ---------------------------------------------------------------------------
def _scan_keyword_directions(config: dict[str, Any], held_out_start: int, log: Any):
    """Shared B1 scan: per-code keyword-mean token direction over selection shards.

    For each of a deterministic sample of selection shards, tokenizes every
    scanned note once per (code, keyword) pair that survives a cheap
    substring pre-filter (skips tokenizing a note that plainly doesn't
    contain the keyword at all), locates the keyword's token span via
    `find_keyword_token_spans`, and folds the matched activation rows into a
    running per-code mean via `accumulate_keyword_direction`.

    Returns (m, scan) where m is [d_model, n_codes] (unit-norm columns; zero
    column for a code with zero keyword-matched tokens) and `scan` carries
    everything keyword_b2 needs to reuse this pass without re-scanning:
    code_names, scan_meta (matched selection notes actually scanned,
    row-aligned to Y_scan), Y_scan, scan_shards, token_positions,
    underpowered_codes, d_model.
    """
    import numpy as np
    import pandas as pd
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer

    from mech_interp_research.feature_sources import (
        accumulate_keyword_direction,
        find_keyword_token_spans,
    )
    from mech_interp_research.icd_eval import load_and_align_icd_labels, load_metadata
    from mech_interp_research.lexical_baseline import load_keyword_dict
    from mech_interp_research.necessity_stats import split_by_shard

    acts_dir = Path(config["activations_dir"])
    metadata = load_metadata(acts_dir)
    Y, code_names, matched_meta = load_and_align_icd_labels(
        Path(config["icd_csv_path"]),
        metadata,
        min_prevalence=float(config.get("min_prevalence", 0.02)),
        max_codes=int(config.get("max_codes", 50)),
        icd_col_prefix=config.get("icd_col_prefix", "icd9_"),
        join_key=config.get("join_key", "admission_id"),
        min_notes=int(config.get("min_notes", 100)),
    )
    Y = np.asarray(Y, dtype=np.float64)
    sel, _ = split_by_shard(matched_meta, held_out_shard_start=held_out_start)
    sel_meta = matched_meta[sel].reset_index(drop=True)
    Y_sel = Y[sel]

    keyword_dict = load_keyword_dict(config["icd_keywords_yaml_path"], code_filter=code_names)

    rng = np.random.default_rng(int(config.get("seed", 42)))
    eligible_shards = sorted(int(s) for s in sel_meta["shard"].unique())
    n_scan = int(config.get("scan_n_shards", 24))
    chosen = rng.choice(len(eligible_shards), size=min(n_scan, len(eligible_shards)), replace=False)
    scan_shards = sorted(eligible_shards[i] for i in chosen)

    scan_bool = sel_meta["shard"].isin(scan_shards).to_numpy()
    scan_meta = sel_meta[scan_bool].reset_index(drop=True)
    Y_scan = Y_sel[scan_bool]

    join_key = config.get("join_key", "admission_id")
    text_col = config.get("text_col", "note_text")
    icd_df_text = pd.read_csv(config["icd_csv_path"], usecols=[join_key, text_col])
    text_lookup = (
        scan_meta[[join_key, "note_idx"]]
        .drop_duplicates()
        .merge(icd_df_text, on=join_key, how="inner")
    )
    note_text_by_idx = dict(
        zip(text_lookup["note_idx"].astype(int), text_lookup[text_col].astype(str), strict=False)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model_name", "google/gemma-2-2b"), token=os.environ.get("HF_TOKEN")
    )
    max_length = int(config.get("max_length", 8192))

    d_model = None
    sums: dict[str, np.ndarray | None] = dict.fromkeys(code_names)
    counts: dict[str, int] = dict.fromkeys(code_names, 0)
    token_positions: dict[str, int] = dict.fromkeys(code_names, 0)

    for shard_idx in scan_shards:
        shard_path = acts_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            log.warning("Shard file missing, skipping: %s", shard_path)
            continue
        shard_acts = load_file(str(shard_path))["activations"].astype(np.float32)
        if d_model is None:
            d_model = shard_acts.shape[1]

        shard_notes = scan_meta[scan_meta["shard"] == shard_idx]
        for _, note_row in shard_notes.iterrows():
            note_idx = int(note_row["note_idx"])
            text = note_text_by_idx.get(note_idx)
            if not text:
                continue
            row_start, row_end = int(note_row["row_start"]), int(note_row["row_end"])
            n_note_tokens = row_end - row_start
            text_lower = text.lower()

            for code in code_names:
                keywords = keyword_dict.get(code, [])
                if not keywords:
                    continue
                hit_idx: set[int] = set()
                for kw in keywords:
                    if kw.lower() not in text_lower:
                        continue  # cheap pre-filter: skip tokenizing on a guaranteed miss
                    hit_idx.update(
                        find_keyword_token_spans(text, tokenizer, kw, max_length=max_length)
                    )
                if not hit_idx:
                    continue
                valid = [i for i in hit_idx if i < n_note_tokens]
                if not valid:
                    continue
                rows = shard_acts[[row_start + i for i in valid]]
                acc = sums[code] if sums[code] is not None else np.zeros(d_model, dtype=np.float64)
                sums[code], counts[code] = accumulate_keyword_direction(acc, counts[code], rows)
                token_positions[code] += len(valid)
        del shard_acts

    m = np.zeros((d_model, len(code_names)), dtype=np.float32)
    underpowered: list[str] = []
    min_positions = int(config.get("min_token_positions", 200))
    for c, code in enumerate(code_names):
        if counts[code] == 0:
            underpowered.append(code)
            continue
        mean_vec = sums[code] / counts[code]
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1e-12:
            underpowered.append(code)
            continue
        m[:, c] = (mean_vec / norm).astype(np.float32)
        if token_positions[code] < min_positions:
            underpowered.append(code)

    log.info(
        "Keyword scan: %d/%d codes with a direction (%d underpowered, < %d token positions), "
        "%d shards scanned, %d selection notes considered",
        len(code_names) - len(underpowered),
        len(code_names),
        len(underpowered),
        min_positions,
        len(scan_shards),
        len(scan_meta),
    )

    scan = {
        "code_names": code_names,
        "scan_meta": scan_meta,
        "Y_scan": Y_scan,
        "scan_shards": scan_shards,
        "token_positions": token_positions,
        "underpowered_codes": underpowered,
        "d_model": d_model,
        "n_selection_notes": int(sel.sum()),
    }
    return m, scan


def _build_keyword_b1(config: dict[str, Any], held_out_start: int, log: Any):
    m, scan = _scan_keyword_directions(config, held_out_start, log)
    meta = {
        "code_names": scan["code_names"],
        "keyword_token_positions": scan["token_positions"],
        "underpowered_codes": scan["underpowered_codes"],
        "scan_shards": scan["scan_shards"],
        "n_selection_notes_scanned": int(len(scan["scan_meta"])),
        "n_selection_notes": scan["n_selection_notes"],
    }
    return m, meta


def _build_keyword_b2(config: dict[str, Any], held_out_start: int, arm_c: dict[str, Any], log: Any):
    """B1 diluted with an unrelated cross-chapter code's direction.

    Ruling 3: for every code, `p = X_tok @ m_c` and `q = X_tok @ m_other` are
    each computed exactly once — via a single combined matmul per scanned
    shard against ALL 46 built keyword directions at once (`P = X_tok @ m`),
    so no code's projection is ever redone inside solve_dilution_alpha's
    bisection loop. Re-projecting per alpha step would re-multiply the full
    scanned-token matrix on every one of ~36 evaluations per code instead of
    once, turning a seconds-scale solve into an hours-scale one.
    """
    import numpy as np
    from safetensors.numpy import load_file

    from mech_interp_research.concordance_multi_judge import icd9_chapter
    from mech_interp_research.feature_sources import blend_directions, solve_dilution_alpha
    from mech_interp_research.icd_eval import compute_point_biserial_vectorised

    m, scan = _scan_keyword_directions(config, held_out_start, log)
    code_names = scan["code_names"]
    scan_meta = scan["scan_meta"]
    scan_shards = scan["scan_shards"]
    Y_scan = scan["Y_scan"]
    d_model = scan["d_model"]
    n_codes = len(code_names)
    n_scan_notes = len(scan_meta)

    if code_names != arm_c["code_names"]:
        raise ValueError(
            "Code ordering mismatch between this arm's label alignment "
            "(from activations_dir) and Arm C's selection (from "
            "sae_shard_ckpt_dir). arm_c['r_selection'] is indexed by code "
            "position, so a silent mismatch here would calibrate one code's "
            "dilution target onto a different code's direction. Check that "
            "activations_dir and sae_shard_ckpt_dir cover the identical note "
            "population with identical join_key/icd_col_prefix/min_prevalence/"
            "max_codes/min_notes settings."
        )

    has_direction = [c for c in range(n_codes) if float(np.linalg.norm(m[:, c])) > 0.0]

    # Deterministic cross-chapter dilution partner per code (spec Sec 4.2.2 —
    # same cross-chapter rule the retrieval slate uses for distractors).
    rng = np.random.default_rng(int(config.get("seed", 42)))
    partner_idx: dict[int, int] = {}
    for c in sorted(has_direction):
        ch = icd9_chapter(code_names[c])
        candidates = sorted(
            c2 for c2 in has_direction if c2 != c and icd9_chapter(code_names[c2]) != ch
        )
        if not candidates:
            raise ValueError(f"No cross-chapter keyword partner available for {code_names[c]!r}")
        partner_idx[c] = candidates[int(rng.integers(0, len(candidates)))]

    # Project every built direction onto the scanned selection tokens ONCE,
    # via one combined [n_rows, d_model] @ [d_model, n_codes] matmul per
    # shard. Shards are loaded, projected, and discarded one at a time so
    # peak memory stays at one shard (~a few GB), never all scanned shards
    # at once.
    note_ordinal = {int(nidx): i for i, nidx in enumerate(scan_meta["note_idx"])}
    P_chunks = []
    note_id_chunks = []
    for shard_idx in scan_shards:
        shard_path = Path(config["activations_dir"]) / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            continue
        shard_acts = load_file(str(shard_path))["activations"].astype(np.float32)
        n_rows = shard_acts.shape[0]

        row_ordinal = np.full(n_rows, -1, dtype=np.int64)
        shard_notes = scan_meta[scan_meta["shard"] == shard_idx]
        for _, note_row in shard_notes.iterrows():
            note_idx = int(note_row["note_idx"])
            row_start, row_end = int(note_row["row_start"]), int(note_row["row_end"])
            row_ordinal[row_start:row_end] = note_ordinal[note_idx]

        keep = row_ordinal >= 0
        if np.any(keep):
            P_shard = shard_acts[keep].astype(np.float64) @ m.astype(np.float64)
            P_chunks.append(P_shard)
            note_id_chunks.append(row_ordinal[keep])
        del shard_acts

    P = np.concatenate(P_chunks, axis=0)
    note_id_tok = np.concatenate(note_id_chunks, axis=0)
    log.info("B2 projection matrix: %d tokens x %d codes", P.shape[0], P.shape[1])

    alpha_max = float(config.get("alpha_max", 32.0))
    W = np.zeros((d_model, n_codes), dtype=np.float32)
    alphas: list[float | None] = [None] * n_codes
    partners: list[str | None] = [None] * n_codes
    unreachable_codes: list[str] = []

    for c in range(n_codes):
        if c not in has_direction:
            continue  # leave zero column: no keyword hits at all for this code
        p_idx = partner_idx[c]
        partners[c] = code_names[p_idx]
        p = P[:, c]
        q = P[:, p_idx]
        mc_dot_mother = float(m[:, c] @ m[:, p_idx])
        y_c = Y_scan[:, c]

        def score_fn(
            alpha: float,
            p=p,
            q=q,
            mc_dot_mother=mc_dot_mother,
            y_c=y_c,
        ) -> float:
            denom = np.sqrt(1.0 + 2.0 * alpha * mc_dot_mother + alpha**2)
            blended = (p + alpha * q) / denom
            note_max = np.full(n_scan_notes, -np.inf, dtype=np.float64)
            np.maximum.at(note_max, note_id_tok, blended)
            note_max = np.where(np.isfinite(note_max), note_max, 0.0)
            r_pb, _ = compute_point_biserial_vectorised(note_max[:, None], y_c[:, None])
            return float(abs(r_pb[0, 0]))

        target = abs(float(arm_c["r_selection"][c]))
        try:
            alpha = solve_dilution_alpha(score_fn, target=target, alpha_max=alpha_max)
        except ValueError as exc:
            if "unreachable" not in str(exc):
                raise
            log.warning("B2 dilution unreachable for %s: %s", code_names[c], exc)
            unreachable_codes.append(code_names[c])
            alpha = 0.0
        alphas[c] = alpha
        W[:, c] = blend_directions(m[:, c], m[:, p_idx], alpha)

    meta = {
        "code_names": code_names,
        "keyword_token_positions": scan["token_positions"],
        "underpowered_codes": scan["underpowered_codes"],
        "scan_shards": scan_shards,
        "n_selection_notes_scanned": int(len(scan_meta)),
        "n_selection_notes": scan["n_selection_notes"],
        "dilution_partner_code": partners,
        "dilution_alpha": alphas,
        "dilution_target_r_selection": arm_c["r_selection"],
        "dilution_unreachable_codes": unreachable_codes,
        "dilution_partner_seed": int(config.get("seed", 42)),
    }
    return W, meta


def _build_keyword(
    config: dict[str, Any], arm: str, held_out_start: int, arm_c: dict[str, Any], log: Any
):
    if arm == "keyword_b1":
        return _build_keyword_b1(config, held_out_start, log)
    if arm == "keyword_b2":
        return _build_keyword_b2(config, held_out_start, arm_c, log)
    raise ValueError(f"unknown keyword arm {arm!r}")


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=65536,
    timeout=14400,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[hf_secret],
)
def build_source_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    import numpy as np

    from mech_interp_research.feature_sources import (
        DEFAULT_TARGET_DENSITY,
        calibrate_thresholds,
        calibrate_thresholds_note_level,
        sae_note_level_densities,
        write_pseudo_sae,
    )

    logging.basicConfig(level=config.get("logging_level", "INFO"))
    log = logging.getLogger("build_feature_source")

    arm = config["arm"]
    out_dir = Path(config["output_dir"])
    held_out_start = int(config.get("held_out_shard_start", 281))
    target_density = float(config.get("target_density", DEFAULT_TARGET_DENSITY))

    # Shared across every arm: the 46 (latent, code) pairs Arm C itself would
    # select on the selection set. Drives both the note-level calibration
    # target (Ruling 1) and, for B2, the dilution target |r| (Ruling 3).
    arm_c = _arm_c_selected_features(config, held_out_start, log)
    feature_ids = arm_c["feature_ids"]

    if arm == "diff_in_means":
        W, meta = _build_diff_in_means(config, held_out_start, log)
    elif arm.startswith("keyword"):
        W, meta = _build_keyword(config, arm, held_out_start, arm_c, log)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    # feature_ids / r_selection are indexed by code POSITION, so this arm's
    # own code_names (from its own label alignment) must match Arm C's
    # (from sae_shard_ckpt_dir) exactly — same guard as inside
    # _build_keyword_b2, repeated here so diff_in_means and keyword_b1 are
    # covered too. A shape mismatch would already raise inside
    # calibrate_thresholds_note_level below, but a same-length reordering
    # would not, so this checks equality, not just length.
    if meta.get("code_names") != arm_c["code_names"]:
        raise ValueError(
            "Code ordering mismatch between this arm's label alignment and "
            "Arm C's selection (sae_shard_ckpt_dir). Check that the pooled/"
            "activation source used to build this arm's direction and "
            "sae_shard_ckpt_dir cover the identical note population with "
            "identical join_key/icd_col_prefix/min_prevalence/max_codes/"
            "min_notes settings."
        )

    token_sample, note_ids = _sample_tokens(config, held_out_start, log)

    # Ruling 1 — calibrate on the reference SAE's NOTE-level detection rate,
    # not a token-level firing density. This is the threshold actually used.
    sae_shard_ckpt_dir = config["sae_shard_ckpt_dir"]
    target_rates = sae_note_level_densities(
        sae_shard_ckpt_dir, feature_ids, held_out_shard_start=held_out_start
    )
    theta = calibrate_thresholds_note_level(W, token_sample, note_ids, target_rates)

    # Token-level calibration is computed too, but reported-only: it is NOT
    # what theta above is set to. Recorded so the note-vs-token mismatch that
    # motivated Ruling 1 stays auditable per arm.
    theta_token = calibrate_thresholds(W, token_sample, target_density=target_density)
    pre_token = token_sample.astype(np.float32) @ W
    measured_token_density = (pre_token > theta_token).mean(axis=0)

    pre = token_sample.astype(np.float64) @ W.astype(np.float64)
    _, note_idx_compact = np.unique(note_ids, return_inverse=True)
    n_notes_sampled = int(note_idx_compact.max()) + 1 if note_idx_compact.size else 0
    note_max = np.full((n_notes_sampled, W.shape[1]), -np.inf, dtype=np.float64)
    np.maximum.at(note_max, note_idx_compact, pre)
    measured_note_rate = (note_max > theta).mean(axis=0)

    meta.update(
        {
            "arm": arm,
            "held_out_shard_start": held_out_start,
            "calibration_n_tokens": int(token_sample.shape[0]),
            "calibration_n_notes": n_notes_sampled,
            "arm_c_code_names": arm_c["code_names"],
            "arm_c_feature_ids": feature_ids,
            "arm_c_r_selection": arm_c["r_selection"],
            "sae_shard_ckpt_dir": str(sae_shard_ckpt_dir),
            "note_level_target_rate": target_rates.tolist(),
            "note_level_measured_rate": measured_note_rate.tolist(),
            "token_level_target_density_reported_only": target_density,
            "token_level_measured_density_reported_only": measured_token_density.tolist(),
        }
    )
    write_pseudo_sae(W, theta, out_dir, meta)
    artifacts_volume.commit()

    log.info("Source written: %s", out_dir)
    return {"output_dir": str(out_dir), "d_sae": int(W.shape[1]), "arm": arm}


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """CLI stub — load YAML config and dispatch remotely.

    Usage:
        uv run modal run modal_app/build_feature_source.py --config-file configs/source_diff_in_means.yaml
        uv run modal run modal_app/build_feature_source.py --config-file configs/source_keyword_b1.yaml --detach
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    print(
        f"Dispatching build_feature_source: arm={config.get('arm')} -> {config.get('output_dir')}"
    )

    if detach:
        call = build_source_remote.spawn(config)
        print(f"Spawned detached: {call.object_id}")
        return
    result = build_source_remote.remote(config)
    print(json.dumps(result, indent=2))
