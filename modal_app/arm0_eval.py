"""Arm 0 ONLY — verbatim original-prompt ICD concordance across a judge panel.

Isolates Arm 0 from the full multi-judge job: runs the EXACT original
``CONCORDANCE_PROMPT`` (r-anchor included) through one or more OpenRouter judges
(e.g. ``openai/gpt-4o-mini``) and reports per-judge YES-only / YES+PARTIAL rates
at each |r| threshold — directly comparable to the published concordance table.

Arm 0 needs ONLY ``concordance_results.csv`` from a completed auto_interp run:
that file already carries, per grounded feature, the explanation and the exact
target ``concordance_icd_code`` / ``concordance_icd_description`` /
``concordance_r_pb`` the original judge saw. No slate, no correlation matrix, no
note text -> no PHI leaves the workspace. CPU-only.

Run:
    uv run modal run modal_app/arm0_eval.py --config-file configs/arm0_4omini.yaml
    uv run modal run --detach modal_app/arm0_eval.py --config-file configs/arm0_4omini.yaml --detach
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

from modal_app.app import app, artifacts_volume, hf_secret, image

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=8192,
    timeout=7200,
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openrouter-api-key"),
        hf_secret,
    ],
    volumes={"/out": artifacts_volume},
)
def run_arm0_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Score Arm 0 for every grounded feature with each configured judge."""
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import pandas as pd
    from openai import OpenAI

    from mech_interp_research.concordance_multi_judge import (
        aggregate_multi_judge,
        build_judges,
        judge_original,
    )

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("arm0_eval")

    openrouter = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=config.get("client_max_retries", 6),
    )
    judges = build_judges(config["judges"], openrouter_client=openrouter)
    if not judges:
        raise ValueError("no live judges — every entry is 'reuse'; add an openrouter judge")
    slugs = [j.slug for j in judges]

    ai_dir = Path(config["auto_interp_dir"])
    df = pd.read_csv(ai_dir / "concordance_results.csv")
    required = {
        "feature_idx",
        "explanation",
        "concordance_icd_code",
        "concordance_icd_description",
        "concordance_r_pb",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"concordance_results.csv is missing required columns: {sorted(missing)}")
    feats = df.to_dict("records")
    log.info("Arm 0: %d grounded features x %d judge(s): %s", len(feats), len(judges), slugs)

    def _score(rec: dict) -> dict:
        # Use the stored original target verbatim so the prompt renders exactly as
        # the published run did (concordance_icd_code is prefixed 'icd9_').
        code = str(rec["concordance_icd_code"]).removeprefix("icd9_")
        desc = str(rec["concordance_icd_description"])
        r_pb = float(rec["concordance_r_pb"])
        row: dict[str, Any] = {
            "feature_idx": int(rec["feature_idx"]),
            "tier": rec.get("tier", "grounded"),
            "r_pb": r_pb,
            "icd_code": code,
            "explanation": rec["explanation"],
        }
        for j in judges:
            v = judge_original(j, rec["explanation"], code, desc, r_pb)
            row[f"{j.slug}_orig_verdict"] = v["verdict"]
            row[f"{j.slug}_orig_rationale"] = v["rationale"]
        return row

    rows: list[dict] = []
    max_workers = config.get("max_workers", 6)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_score, rec): rec for rec in feats}
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 50 == 0:
                log.info("  scored %d/%d", done, len(feats))

    thresholds = config.get("thresholds", [0.3, 0.4, 0.5])
    summary = aggregate_multi_judge(
        rows, slugs, thresholds, verdict_key="orig_verdict", rank_key=None
    )
    summary["n_features"] = len(rows)
    summary["arm"] = "arm0_original"

    out_dir = Path(config.get("output_dir") or (ai_dir / "arm0_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "arm0_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(rows).to_csv(out_dir / "arm0_verdicts.csv", index=False)
    artifacts_volume.commit()

    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch the Arm 0 evaluation to Modal."""
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(
        f"Arm 0 eval: {config.get('auto_interp_dir')} "
        f"judges={[j['slug'] for j in config['judges']]}"
    )
    if detach:
        call = run_arm0_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        print(json.dumps(run_arm0_remote.remote(config), indent=2, default=str))
