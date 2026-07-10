"""Discriminative retrieval eval (Arm 2, clean design) across a judge panel.

For each grounded feature, builds a slate of {the grounded code + K
statistically-unrelated, cross-ICD-chapter, prevalence-matched distractors +
"none"}, shows it to a judge BLIND to the correlation statistics, and asks which
ONE the explanation best describes. Reports hit@1 (judge recovers the grounded
code) binned by |r_pb|, against the intrinsic chance floor 1/(K+2).

This is the non-circular, forced-choice, floor-anchored replacement for the
confirmatory concordance prompt. Reads only concordance_results.csv +
correlation_matrices.npz + code descriptions (+ the ICD CSV for prevalence);
no note text. CPU-only.

Run:
    uv run modal run modal_app/retrieval_eval.py --config-file configs/retrieval_gpt4o.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

from modal_app.app import app, artifacts_volume, hf_secret, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "4"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    memory=16384,
    timeout=10800,
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openrouter-api-key"),
        hf_secret,
    ],
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_retrieval_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from openai import OpenAI

    from mech_interp_research.auto_interp import _load_code_descriptions, _write_json
    from mech_interp_research.concordance_multi_judge import (
        build_discriminative_slate,
        build_judges,
        build_retrieval_prompt,
        parse_retrieval_response,
    )
    from mech_interp_research.icd_eval import load_saved_correlations

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("retrieval_eval")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        cand = {k: v for k, v in os.environ.items() if "OPENRO" in k.upper() and "KEY" in k.upper()}
        api_key = next(iter(cand.values())) if len(cand) == 1 else None
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in the mounted secret.")
    openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, max_retries=6)
    judges = build_judges(config["judges"], openrouter_client=openrouter)
    if not judges:
        raise ValueError("no live judges configured")

    ai_dir = Path(config["auto_interp_dir"])
    df = pd.read_csv(ai_dir / "concordance_results.csv")
    corr = load_saved_correlations(config["icd_eval_dir"])
    r_pb, code_names = corr["r_pb"], corr["code_names"]
    descs = _load_code_descriptions(None, config.get("icd_keywords_yaml_path"))

    # Prevalence from the ICD CSV label columns (read only the 46 code columns).
    prevalence = None
    icd_csv = config.get("icd_csv_path")
    if icd_csv and Path(icd_csv).exists():
        header = pd.read_csv(icd_csv, nrows=0).columns
        colmap = {}
        for c in code_names:
            bare = c.replace("icd9_", "")
            for cand in (c, bare, "icd9_" + bare):
                if cand in header:
                    colmap[bare] = cand
                    break
        if colmap:
            dfp = pd.read_csv(icd_csv, usecols=list(colmap.values()))
            prevalence = {bare: float(dfp[col].mean()) for bare, col in colmap.items()}
            log.info("Loaded prevalence for %d/%d codes", len(prevalence), len(code_names))
    if prevalence is None:
        log.warning("No prevalence available; distractors selected by chapter + |r| only")

    n_distractors = config.get("n_distractors", 7)
    r_unrelated = config.get("r_unrelated", 0.05)
    feats = df.to_dict("records")
    log.info(
        "Retrieval eval: %d features x %d judge(s), K=%d distractors",
        len(feats),
        len(judges),
        n_distractors,
    )

    def _score(rec: dict, judge) -> dict:
        fid = int(rec["feature_idx"])
        cstar = str(rec["concordance_icd_code"]).replace("icd9_", "")
        slate, chance = build_discriminative_slate(
            cstar,
            r_pb[fid],
            code_names,
            descs,
            prevalence=prevalence,
            n_distractors=n_distractors,
            r_unrelated=r_unrelated,
            seed=fid,
        )
        prompt = build_retrieval_prompt(rec["explanation"], slate)
        raw = judge.complete(prompt)
        ret = parse_retrieval_response(raw, slate)
        picked = ret["picked_code"]
        correct_desc = next((e["description"] for e in slate if e["is_correct"]), cstar)
        picked_desc = next((e["description"] for e in slate if e["code"] == picked), "")
        return {
            "slug": judge.slug,
            "feature_idx": fid,
            "r_pb": float(rec["concordance_r_pb"]),
            "abs_r_pb": abs(float(rec["concordance_r_pb"])),
            "correct_code": cstar,
            "correct_description": correct_desc,
            "picked_code": picked,
            "picked_description": picked_desc,
            "hit1": int(picked == cstar),
            "is_none": int(ret["is_none"]),
            "chance": chance,
            "prompt": prompt,
            "judge_raw_output": raw,
            "explanation": rec["explanation"],
        }

    tasks = [(rec, j) for rec in feats for j in judges]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.get("max_workers", 6)) as ex:
        futures = {ex.submit(_score, rec, j): None for rec, j in tasks}
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 100 == 0:
                log.info("  scored %d/%d", done, len(tasks))

    thresholds = config.get("thresholds", [0.3, 0.5])
    out_root = Path(config.get("output_dir") or (ai_dir / "retrieval_eval"))
    combined: dict[str, Any] = {}
    for j in judges:
        jr = [r for r in rows if r["slug"] == j.slug]

        def _block(subset):
            n = len(subset)
            return {
                "n": n,
                "hit1_rate": sum(r["hit1"] for r in subset) / n if n else None,
                "none_rate": sum(r["is_none"] for r in subset) / n if n else None,
                "chance_floor": float(np.mean([r["chance"] for r in subset])) if n else None,
            }

        summary = {"all_grounded(>0.1)": _block(jr)}
        for t in thresholds:
            summary[f"r>{t}"] = _block([r for r in jr if abs(r["r_pb"]) > t])
        summary["n_distractors"] = n_distractors
        summary["design"] = (
            "1 correct + K unrelated cross-chapter distractors + none; hit@1 vs 1/(K+2)"
        )

        model_dir = out_root / j.slug
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_json(summary, model_dir / "retrieval_summary.json")
        pd.DataFrame(jr).to_csv(model_dir / "retrieval_verdicts.csv", index=False)
        combined[j.slug] = summary
        b = summary["all_grounded(>0.1)"]
        log.info(
            "%s: hit@1=%.3f (chance=%.3f) none=%.3f n=%d",
            j.slug,
            b["hit1_rate"],
            b["chance_floor"],
            b["none_rate"],
            b["n"],
        )

    artifacts_volume.commit()
    print(json.dumps(combined, indent=2, default=str))
    return combined


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(
        f"Retrieval eval: {config.get('auto_interp_dir')} models={[j['slug'] for j in config['judges']]}"
    )
    if detach:
        call = run_retrieval_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        print(json.dumps(run_retrieval_remote.remote(config), indent=2, default=str))
