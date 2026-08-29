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

    from mech_interp_research.concordance_multi_judge import build_judges, judge_binary

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
    summary: dict[str, Any] = {"source": str(src), "n_features": len(rows), "judges": {}}

    for j in judges:
        recs = []
        for i, r in enumerate(rows):
            v = judge_binary(
                j, r["explanation"], r["concordance_icd_code"], r["concordance_icd_description"]
            )
            recs.append(
                {
                    "feature_idx": r["feature_idx"],
                    "abs_r_pb": abs(float(r["concordance_r_pb"])),
                    "icd_code": r["concordance_icd_code"],
                    "original_verdict": r["concordance_verdict"],
                    "binary_verdict": v["verdict"],
                    "rationale": v["rationale"],
                }
            )
            if (i + 1) % 50 == 0:
                log.info("%s: %d/%d", j.slug, i + 1, len(rows))

        d = out_dir / j.slug
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "binary_verdicts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)

        n = len(recs)
        y = sum(1 for x in recs if x["binary_verdict"] == "YES")
        u = sum(1 for x in recs if x["binary_verdict"] == "UNKNOWN")
        # How the forced choice resolved what the original prompt called PARTIAL.
        part = [x for x in recs if x["original_verdict"] == "PARTIAL"]
        part_yes = sum(1 for x in part if x["binary_verdict"] == "YES")
        summary["judges"][j.slug] = {
            "n": n,
            "binary_yes_pct": 100 * y / n,
            "unknown_pct": 100 * u / n,
            "n_originally_partial": len(part),
            "partial_to_yes_pct": (100 * part_yes / len(part)) if part else None,
        }
        log.info(
            "%s: binary-YES %.1f%% | of original PARTIALs, %.1f%% became YES",
            j.slug,
            100 * y / n,
            (100 * part_yes / len(part)) if part else float("nan"),
        )
        artifacts_volume.commit()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "binary_summary.json").write_text(json.dumps(summary, indent=2))
    artifacts_volume.commit()
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if detach:
        print(f"Spawned detached: {binary_remote.spawn(config).object_id}")
        return
    print(json.dumps(binary_remote.remote(config), indent=2))
