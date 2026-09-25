"""``${var}`` substitution for pipeline config YAMLs.

Every stage after SAE training names its inputs and outputs as literal paths built
from a run id — ``/out/saes/<run>/final``, ``/out/icd_eval/<run>/shard_ckpt`` and so
on. Run ids carry a UTC timestamp, so retraining an SAE changes every one of them,
across dozens of config files.

Configs therefore reference a run through a variable::

    sae_checkpoint: ${jumprelu_ckpt}
    output_dir:     ${jumprelu_eval}

and ``configs/runs.yaml`` holds the one definition of each::

    jumprelu_run:  jumprelu_L16_d2304_e8_l01e+01_bw1e+00_s42_20260924T101500Z
    jumprelu_ckpt: /out/saes/${jumprelu_run}/final
    jumprelu_eval: /out/icd_eval/${jumprelu_run}

Retraining is then a one-line edit. That also removes a silent failure mode: a
rerun whose output_dir still pointed at the previous run's directory would find a
full set of ``shard_ckpt/`` files, skip every shard as already done, and report
numbers computed from the *old* SAE's pooled vectors without erroring. Deriving the
output directory from the run id makes that impossible — a new run writes to a new
directory because its id is part of the path.

Substitution happens on the laptop, before the config dict is handed to Modal, so
containers only ever see fully-resolved paths and ``runs.yaml`` never needs shipping
into the image.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FILENAME = "runs.yaml"
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
# Registry entries may reference each other; this bounds the resolution passes so a
# cycle raises instead of hanging.
_MAX_RESOLUTION_PASSES = 10


def find_registry(config_path: str | Path) -> Path | None:
    """Locate the nearest ``runs.yaml`` at or above ``config_path``'s directory.

    Walking up means ``configs/fourarm/x.yaml`` finds ``configs/runs.yaml`` without
    each subdirectory needing its own copy. Returns None when there is none, which
    is fine for configs that use no variables.
    """
    start = Path(config_path).resolve().parent
    for parent in [start, *start.parents]:
        candidate = parent / REGISTRY_FILENAME
        if candidate.is_file():
            return candidate
        if (parent / ".git").exists():
            break
    return None


def load_registry(registry_path: str | Path) -> dict[str, str]:
    """Load ``runs.yaml`` and resolve references between its own entries."""
    raw = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{registry_path} must contain a mapping of name -> value")

    resolved = {str(k): "" if v is None else str(v) for k, v in raw.items()}
    for _ in range(_MAX_RESOLUTION_PASSES):
        pending = [k for k, v in resolved.items() if _VAR_PATTERN.search(v)]
        if not pending:
            return resolved
        for key in pending:
            resolved[key] = _substitute_string(resolved[key], resolved, registry_path)
    raise ValueError(
        f"{registry_path}: could not resolve variables after "
        f"{_MAX_RESOLUTION_PASSES} passes — check for a reference cycle among "
        f"{sorted(k for k, v in resolved.items() if _VAR_PATTERN.search(v))}"
    )


def _substitute_string(text: str, variables: dict[str, str], source: Any) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(
                f"{source}: unknown config variable '${{{name}}}'. "
                f"Defined variables: {sorted(variables)}"
            )
        return variables[name]

    return _VAR_PATTERN.sub(replace, text)


def substitute(value: Any, variables: dict[str, str], source: Any = "<config>") -> Any:
    """Recursively substitute ``${var}`` in every string inside ``value``."""
    if isinstance(value, str):
        return _substitute_string(value, variables, source)
    if isinstance(value, dict):
        return {k: substitute(v, variables, source) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, variables, source) for v in value]
    return value


def load_config(
    config_path: str | Path,
    registry_path: str | Path | None = None,
) -> Any:
    """Load a YAML config and resolve ``${var}`` against the run registry.

    Drop-in replacement for ``yaml.safe_load(open(path))`` in a Modal
    ``local_entrypoint``. A config containing no variables is returned unchanged, so
    this is safe to use everywhere regardless of whether a given config uses them.

    Raises KeyError naming the offending variable if a config references one the
    registry does not define — a typo fails loudly here rather than dispatching a
    multi-hour job against a path like ``/out/icd_eval/${jumprelu_ruh}``.
    """
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    registry = registry_path if registry_path is not None else find_registry(path)
    if registry is None:
        variables: dict[str, str] = {}
    else:
        variables = load_registry(registry)

    return substitute(data, variables, source=path)
