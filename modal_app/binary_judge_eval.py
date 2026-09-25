"""Forced-binary (YES/NO) concordance -- PARTIAL removed from the option set.

exact-YES measures "YES vs (PARTIAL or NO)" *while PARTIAL was on offer*. This arm
asks whether PARTIAL is a real third category or an artifact of offering it: every
feature that drew PARTIAL must now resolve one way or the other.

Reads a completed auto_interp run and re-judges its explanations. No GPU, no
re-explanation -- the explanations are held fixed so only the option set changes.

Run:
    uv run modal run --detach modal_app/binary_judge_eval.py \\
        --config-file configs/binary_jumprelu.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal
import yaml

from modal_app.app import app, artifacts_volume, image, raw_volume


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=14400,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
    secrets=[
        modal.Secret.from_name("anthropic-api-key-mohit"),
        modal.Secret.from_name("openrouter-api-key"),
    ],
)
def binary_remote(config: dict[str, Any]) -> dict[str, Any]:
    import csv
    import logging

    from openai import OpenAI

    from mech_interp_research.concordance_multi_judge import (
        build_judges,
        judge_binary_batch,
        rebuild_binary_summary,
        summarize_binary_verdicts,
    )

    logging.basicConfig(level=config.get("logging_level", "INFO"))
    log = logging.getLogger("binary_judge_eval")

    src = Path(config["auto_interp_dir"]) / "concordance_results.csv"
    rows = [r for r in csv.DictReader(open(src)) if r.get("concordance_icd_code")]
    log.info("Loaded %d judged features from %s", len(rows), src)

    openrouter = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=6,
    )
    anthropic_client = None
    if any(j.get("backend") == "anthropic" for j in config["judges"]):
        import anthropic

        anthropic_client = anthropic.Anthropic(max_retries=6)
    judges = build_judges(
        config["judges"], anthropic_client=anthropic_client, openrouter_client=openrouter
    )

    out_dir = Path(config.get("output_dir") or (Path(config["auto_interp_dir"]) / "binary_eval"))
    max_workers = int(config.get("max_workers", 4))
    for j in judges:
        verdicts = judge_binary_batch(
            j,
            rows,
            max_workers=max_workers,
            progress=lambda done, total, slug=j.slug: log.info("%s: %d/%d", slug, done, total),
        )
        recs = [
            {
                "feature_idx": r["feature_idx"],
                "abs_r_pb": abs(float(r["concordance_r_pb"])),
                "icd_code": r["concordance_icd_code"],
                "original_verdict": r["concordance_verdict"],
                "binary_verdict": v["verdict"],
                "rationale": v["rationale"],
            }
            for r, v in zip(rows, verdicts, strict=True)
        ]

        d = out_dir / j.slug
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "binary_verdicts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)

        # Logged only; the roll-up below is rebuilt from the CSVs, not from this.
        blk = summarize_binary_verdicts(recs)
        log.info(
            "%s: binary-YES %.1f%% | of original PARTIALs, %s%% became YES",
            j.slug,
            blk["binary_yes_pct"],
            blk["partial_to_yes_pct"],
        )
        artifacts_volume.commit()

    # Derive the roll-up from every per-judge CSV in the directory rather than
    # overwriting it with this run's judges. Writing `summary` straight out erases
    # judges an earlier invocation recorded (each arm's Sonnet and DeepSeek legs
    # were separate runs), and an in-memory merge still loses a judge when two run
    # concurrently against the same directory. Rebuilding from disk is idempotent
    # and repairs a roll-up left stale by either.
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_volume.reload()
    rebuilt = rebuild_binary_summary(out_dir, source=src)
    (out_dir / "binary_summary.json").write_text(json.dumps(rebuilt, indent=2))
    artifacts_volume.commit()
    log.info("summary now covers judges: %s", sorted(rebuilt["judges"]))
    return rebuilt


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if detach:
        print(f"Spawned detached: {binary_remote.spawn(config).object_id}")
        return
    print(json.dumps(binary_remote.remote(config), indent=2))
