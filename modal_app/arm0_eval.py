"""Arm 0 ONLY — re-run the ORIGINAL concordance judging with a different model.

Reproduces the published concordance evaluation EXACTLY — same verbatim prompt
(``CONCORDANCE_PROMPT``, r-anchor included), same targets, and the **same output
form** (``concordance_summary.json`` with ``global`` + ``r>0.3/0.4/0.5`` blocks,
each carrying yes/partial/no/unknown counts, ``concordance_rate`` and
``exact_match_rate``) — but with a different judge model (e.g. ``openai/gpt-4o``).
The summary is built with the very same ``_concordance_stats`` the original run
used, so per-model numbers drop straight into the paper's table.

Arm 0 needs ONLY ``concordance_results.csv`` from the completed run (explanation +
stored ``concordance_icd_code`` / ``_description`` / ``_r_pb`` the original judge
saw). No slate, no correlation matrix, no note text -> no PHI. CPU-only.

Run:
    uv run modal run modal_app/arm0_eval.py --config-file configs/arm0_gpt4o.yaml
    uv run modal run --detach modal_app/arm0_eval.py --config-file configs/arm0_gpt4o.yaml --detach
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
        modal.Secret.from_name("anthropic-api-key-mohit"),
        modal.Secret.from_name("openrouter-api-key"),
        hf_secret,
    ],
    volumes={"/out": artifacts_volume},
)
def run_arm0_remote(config: dict[str, Any]) -> dict[str, Any]:
    """Re-judge every grounded feature with each configured model, in the
    original prompt + original summary form."""
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import pandas as pd
    from openai import OpenAI

    from mech_interp_research.auto_interp import (
        CONCORDANCE_PROMPT,
        _concordance_stats,
        _write_json,
        parse_concordance_response,
    )
    from mech_interp_research.concordance_multi_judge import build_judges

    logging.basicConfig(
        level=getattr(logging, config.pop("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("arm0_eval")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        cand = {k: v for k, v in os.environ.items() if "OPENRO" in k.upper() and "KEY" in k.upper()}
        if len(cand) == 1:
            api_key = next(iter(cand.values()))
        else:
            seen = [k for k in os.environ if "OPENRO" in k.upper()]
            raise RuntimeError(
                "OpenRouter API key not found in the mounted secret. Expected env var "
                f"OPENROUTER_API_KEY; env keys matching 'OPENRO': {seen or 'none'}."
            )
    openrouter = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=config.get("client_max_retries", 6),
    )
    # An anthropic-backed judge is built on demand, mirroring retrieval_eval.
    # Without this, arm0_eval could only reach OpenRouter -- and this account's
    # OpenRouter key is 403'd on anthropic/* and openai/*, leaving Sonnet
    # unreachable for the very prompt the published Table 2 used.
    anthropic_client = None
    if any(j.get("backend") == "anthropic" for j in config["judges"]):
        import anthropic

        anthropic_client = anthropic.Anthropic(max_retries=config.get("client_max_retries", 6))
    judges = build_judges(
        config["judges"],
        anthropic_client=anthropic_client,
        openrouter_client=openrouter,
    )
    if not judges:
        raise ValueError("no live judges — every entry is 'reuse'; add an openrouter judge")

    live = []
    for j in judges:
        try:
            j.complete("Reply with exactly: ok", max_tokens=5)
            live.append(j)
            log.info("judge preflight OK: %s (%s)", j.slug, j.model)
        except Exception as e:  # noqa: BLE001
            log.warning("dropping judge %s (%s): preflight failed: %s", j.slug, j.model, e)
    if not live:
        raise RuntimeError("no judges passed preflight")
    judges = live

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
    thresholds = config.get("thresholds", [0.3, 0.4, 0.5])
    log.info(
        "Arm 0: %d features x %d model(s): %s",
        len(feats),
        len(judges),
        [j.slug for j in judges],
    )

    def _score(rec: dict, judge) -> dict:
        # Stored code is prefixed 'icd9_'; the original judge saw the BARE code.
        code_prefixed = str(rec["concordance_icd_code"])
        code = code_prefixed.removeprefix("icd9_")
        desc = str(rec["concordance_icd_description"])
        r_pb = float(rec["concordance_r_pb"])
        prompt = CONCORDANCE_PROMPT.format(
            explanation=rec["explanation"], r_pb=r_pb, icd_code=code, icd_description=desc
        )
        try:
            raw = judge.complete(prompt)
            verdict, rationale = parse_concordance_response(raw)
        except Exception as e:  # noqa: BLE001 — one judge's failure must not kill the run
            raw = f"__error__: {e}"
            verdict, rationale = "ERROR", raw
        return {
            "slug": judge.slug,
            "feature_idx": int(rec["feature_idx"]),
            "tier": rec.get("tier", "grounded"),
            "explanation": rec["explanation"],
            "concordance_verdict": verdict,
            "concordance_rationale": rationale,
            "judge_raw_output": raw,
            "concordance_icd_code": code_prefixed,  # keep prefixed to match original CSV
            "concordance_icd_description": desc,
            "concordance_r_pb": r_pb,
        }

    tasks = [(rec, j) for rec in feats for j in judges]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.get("max_workers", 6)) as ex:
        futures = {ex.submit(_score, rec, j): (rec, j) for rec, j in tasks}
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 100 == 0:
                log.info("  scored %d/%d", done, len(tasks))

    # Per model: rebuild the ORIGINAL concordance_summary.json form + a matching
    # concordance_results.csv, so each model's artifacts are drop-in comparable.
    out_root = Path(config.get("output_dir") or (ai_dir / "arm0_eval"))
    combined: dict[str, Any] = {}
    for j in judges:
        conc_rows = [r for r in rows if r["slug"] == j.slug]
        conc_summary: dict[str, Any] = {"global": _concordance_stats(conc_rows)}
        for threshold in thresholds:
            subset = [r for r in conc_rows if abs(r["concordance_r_pb"]) > threshold]
            conc_summary[f"r>{threshold}"] = _concordance_stats(subset)

        model_dir = out_root / j.slug
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_json(conc_summary, model_dir / "concordance_summary.json")
        pd.DataFrame(
            [
                {
                    "feature_idx": r["feature_idx"],
                    "tier": r["tier"],
                    "explanation": r["explanation"],
                    "concordance_verdict": r["concordance_verdict"],
                    "concordance_rationale": r["concordance_rationale"],
                    "concordance_icd_code": r["concordance_icd_code"],
                    "concordance_icd_description": r["concordance_icd_description"],
                    "concordance_r_pb": r["concordance_r_pb"],
                }
                for r in conc_rows
            ]
        ).to_csv(model_dir / "concordance_results.csv", index=False)
        combined[j.slug] = conc_summary
        log.info(
            "%s: global concordance_rate=%.4f exact_match_rate=%.4f (n=%d)",
            j.slug,
            conc_summary["global"].get("concordance_rate", 0),
            conc_summary["global"].get("exact_match_rate", 0),
            conc_summary["global"].get("total", 0),
        )

    artifacts_volume.commit()
    print(json.dumps(combined, indent=2, default=str))
    return combined


@app.local_entrypoint()
def main(config_file: str, detach: bool = False) -> None:
    """Load YAML config and dispatch the Arm 0 evaluation to Modal."""
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(
        f"Arm 0 eval: {config.get('auto_interp_dir')} "
        f"models={[j['slug'] for j in config['judges']]}"
    )
    if detach:
        call = run_arm0_remote.spawn(config)
        print(f"Spawned: {call.object_id}")
    else:
        print(json.dumps(run_arm0_remote.remote(config), indent=2, default=str))
