"""Modal entrypoint: TF-IDF as an audited source (code plan C8).

Fits TF-IDF on train shards only and emits the full n-gram matrix for the
selection and audit shards in ``shard_ckpt`` format. Computes no audit
statistics -- ``modal_app/necessity_audit.py`` does that, with ``top_per_code``
selection so TF-IDF's 10,000-candidate search is reproduced honestly rather
than hidden behind a pre-picked winner.

Needs the note text, which is PHI on the mimic-iv-raw volume, so this must run
on Modal. CPU only.

Invoke:
    modal run modal_app/tfidf_directions.py --config-file configs/tfidf_directions.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any

from modal_app.app import app, artifacts_volume, image, raw_volume

DEFAULT_CPU = int(os.environ.get("MODAL_CPU", "16"))


@app.function(
    image=image,
    cpu=DEFAULT_CPU,
    # The dense select+audit matrix is ~9,912 x 10,000 float32 (~400 MB) and is
    # materialised twice (value + binary arms, one at a time).
    memory=65536,
    timeout=7200,
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_tfidf_sources_remote(config: dict[str, Any]) -> dict[str, Any]:
    import logging

    from mech_interp_research.tfidf_lr_baseline import run_tfidf_direction_sources

    logging.basicConfig(
        level=getattr(logging, config.get("logging_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = {k: v for k, v in config.items() if k != "logging_level"}
    if "ngram_range" in cfg and cfg["ngram_range"] is not None:
        cfg["ngram_range"] = tuple(cfg["ngram_range"])
    try:
        manifest = run_tfidf_direction_sources(**cfg)
    finally:
        artifacts_volume.commit()
    return manifest


@app.local_entrypoint()
def main(config_file: str) -> None:
    import yaml

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run_tfidf_sources_remote.remote(config)
    print(
        f"train notes: {result['n_train_notes']}  emitted: {result['n_emitted_notes']}  "
        f"features: {result['n_features']}  codes: {result['n_codes']}"
    )
    for name, arm in result["arms"].items():
        print(f"  arm {name}: {arm['n_shards']} shards, {arm['n_notes']} notes -> {arm['checkpoint_dir']}")
    print(json.dumps(result["vocabulary_sample"][:15], indent=2))
