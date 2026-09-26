"""Resolve every config under configs/ and dump {path: resolved} as JSON.

A before/after guard around registry edits: configs describing existing results
must resolve to byte-identical values when runs.yaml or the configs change.

    uv run python scripts/dump_config_resolution.py before.json
    # ... edit runs.yaml / configs ...
    uv run python scripts/dump_config_resolution.py after.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mech_interp_research.config_vars import load_config


def main(out_path: str) -> None:
    out = {}
    for p in sorted(Path("configs").rglob("*.yaml")):
        if p.name == "runs.yaml":
            continue
        try:
            out[str(p)] = load_config(p)
        except Exception as e:  # noqa: BLE001 -- recorded, compared before/after
            out[str(p)] = {"__error__": repr(e)}
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    n_err = sum(isinstance(v, dict) and "__error__" in v for v in out.values())
    print(f"{len(out)} configs -> {out_path} ({n_err} errors)")


if __name__ == "__main__":
    main(sys.argv[1])
