"""Core extraction: one-note activation extraction and whole-run orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mech_interp_research.checks import run_checks
from mech_interp_research.config import ExtractionConfig, make_run_id
from mech_interp_research.data import load_notes
from mech_interp_research.model import choose_device, load_model_and_tokenizer
from mech_interp_research.storage import ShardedSafetensorsWriter


def extract_one_note(
    text: str,
    tokenizer: Any,
    model: Any,
    device: str,
    layer: int,
    max_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return post-block residual activations at `layer` for one text, fp32 on CPU."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)
    truncated_tokens = int(attn_mask.sum().item())

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # HF decoder hidden_states[0] is the embedding output. Post-block residual
    # for layer L is hidden_states[L + 1].
    hs_index = layer + 1
    if hs_index >= len(out.hidden_states):
        raise ValueError(
            f"Layer {layer} requested but model only has {len(out.hidden_states) - 1} layers."
        )

    activations = out.hidden_states[hs_index][0, :truncated_tokens, :].detach().cpu().float()
    meta = {
        "num_tokens_truncated": truncated_tokens,
        "d_model": int(activations.shape[1]),
    }
    return activations, meta


def run_extraction(config: ExtractionConfig) -> dict[str, Any]:
    """Run the full extraction pipeline. Returns a summary dict.

    - Resolves run_id if not set in config.
    - Loads notes, model, tokenizer.
    - Streams per-note activations into ShardedSafetensorsWriter.
    - Runs sanity checks on the first few note activations kept in RAM.
    - Writes manifest.json and metadata.jsonl to <output_root>/<run_id>/.

    Resume mode: if config.resume_from_run_id is set, reads the existing manifest
    from that run to find where it stopped, then picks up from that note index.
    New shards and metadata rows are appended to the existing output directory.
    """
    t0 = time.time()

    # --- Resume state -----------------------------------------------------------
    resume_run_id = config.resume_from_run_id
    resume_shard_idx = 0
    resume_total_tokens = 0
    resume_note_count = 0
    start_note_idx = 0

    if resume_run_id:
        resume_dir = Path(config.output_root) / resume_run_id
        existing_manifest_path = resume_dir / "manifest.json"
        if not existing_manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: manifest not found at {existing_manifest_path}"
            )
        with open(existing_manifest_path, encoding="utf-8") as f:
            existing_manifest = json.load(f)
        resume_note_count = int(existing_manifest["n_notes"])
        resume_shard_idx = int(existing_manifest["n_shards"])
        resume_total_tokens = int(existing_manifest["total_tokens"])
        start_note_idx = resume_note_count
        print(
            f"Resuming run '{resume_run_id}': "
            f"skipping {resume_note_count} already-processed notes, "
            f"continuing from shard {resume_shard_idx}."
        )
        run_id = resume_run_id
        output_dir = resume_dir
    else:
        run_id = config.run_id or make_run_id(config)
        output_dir = Path(config.output_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------------------

    device = choose_device()
    df, text_col = load_notes(config.input_csv, config.text_col, config.num_notes)
    tokenizer, model = load_model_and_tokenizer(config.model_name, device)

    # Derive d_model from a note we're actually going to process (avoids running
    # the model on an already-done note just to infer shape).
    probe_idx = start_note_idx
    first_acts, first_meta = extract_one_note(
        text=str(df.iloc[probe_idx][text_col]),
        tokenizer=tokenizer,
        model=model,
        device=device,
        layer=config.layer,
        max_length=config.max_length,
    )
    d_model = int(first_acts.shape[1])

    writer = ShardedSafetensorsWriter(
        output_dir,
        config,
        d_model=d_model,
        resume_shard_idx=resume_shard_idx,
        resume_total_tokens=resume_total_tokens,
        resume_note_count=resume_note_count,
    )

    # Feed the already-extracted probe note to the writer.
    sample_acts: list[torch.Tensor] = [first_acts]
    writer.add_note(
        note_idx=probe_idx,
        activations=first_acts,
        extra_meta=_row_meta(df.iloc[probe_idx], df, first_meta, str(df.iloc[probe_idx][text_col])),
    )

    try:
        for idx in tqdm(
            range(probe_idx + 1, len(df)),
            desc="Extracting activations",
            initial=probe_idx + 1 - start_note_idx,
            total=len(df) - start_note_idx,
        ):
            row = df.iloc[idx]
            text = str(row[text_col])
            acts, acts_meta = extract_one_note(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                layer=config.layer,
                max_length=config.max_length,
            )
            writer.add_note(
                note_idx=idx,
                activations=acts,
                extra_meta=_row_meta(row, df, acts_meta, text),
            )
            if len(sample_acts) < 5:
                sample_acts.append(acts)
    finally:
        # Always flush and write manifest/metadata — even on partial runs.
        manifest = writer.close(
            extra_manifest={"run_id": run_id, "text_col": text_col, "device": device}
        )

    checks = run_checks(sample_acts, expected_d_model=d_model)
    elapsed = time.time() - t0

    summary = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "n_notes": manifest["n_notes"],
        "n_tokens": manifest["total_tokens"],
        "n_shards": manifest["n_shards"],
        "d_model": d_model,
        "elapsed_s": round(elapsed, 2),
        "checks": checks,
    }
    return summary


def _row_meta(row: Any, df: Any, acts_meta: dict[str, Any], text: str) -> dict[str, Any]:
    """Build per-note metadata row, tolerating missing ID columns."""
    return {
        "subject_id": int(row["subject_id"]) if "subject_id" in df.columns else None,
        "admission_id": (
            int(row["admission_id"])
            if "admission_id" in df.columns
            else (int(row["hadm_id"]) if "hadm_id" in df.columns else None)
        ),
        "num_tokens_truncated": acts_meta["num_tokens_truncated"],
        "text_char_len": len(text),
    }
