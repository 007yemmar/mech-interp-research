"""Tests for ${var} substitution in pipeline configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mech_interp_research.config_vars import find_registry, load_config, load_registry

REGISTRY = """
jumprelu_run: jumprelu_L16_s42_20260924T000000Z
jumprelu_ckpt: /out/saes/${jumprelu_run}/final
jumprelu_eval: /out/icd_eval/${jumprelu_run}
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_registry_resolves_internal_references(tmp_path):
    reg = load_registry(_write(tmp_path, "runs.yaml", REGISTRY))
    assert reg["jumprelu_ckpt"] == "/out/saes/jumprelu_L16_s42_20260924T000000Z/final"
    assert reg["jumprelu_eval"] == "/out/icd_eval/jumprelu_L16_s42_20260924T000000Z"


def test_registry_rejects_reference_cycles(tmp_path):
    path = _write(tmp_path, "runs.yaml", "a: ${b}\nb: ${a}\n")
    with pytest.raises(ValueError, match="reference cycle"):
        load_registry(path)


def test_config_substitutes_variables(tmp_path):
    _write(tmp_path, "runs.yaml", REGISTRY)
    cfg = _write(
        tmp_path,
        "eval.yaml",
        "sae_checkpoint: ${jumprelu_ckpt}\noutput_dir: ${jumprelu_eval}/test_split\n",
    )
    loaded = load_config(cfg)
    assert loaded["sae_checkpoint"] == "/out/saes/jumprelu_L16_s42_20260924T000000Z/final"
    assert loaded["output_dir"] == "/out/icd_eval/jumprelu_L16_s42_20260924T000000Z/test_split"


def test_substitution_reaches_nested_structures(tmp_path):
    _write(tmp_path, "runs.yaml", REGISTRY)
    cfg = _write(
        tmp_path,
        "audit.yaml",
        "sources:\n  - name: a\n    checkpoint_dir: ${jumprelu_eval}/shard_ckpt\n"
        "nested:\n  inner: ${jumprelu_run}\n",
    )
    loaded = load_config(cfg)
    assert loaded["sources"][0]["checkpoint_dir"].endswith("20260924T000000Z/shard_ckpt")
    assert loaded["nested"]["inner"] == "jumprelu_L16_s42_20260924T000000Z"


def test_non_string_values_are_untouched(tmp_path):
    _write(tmp_path, "runs.yaml", REGISTRY)
    cfg = _write(tmp_path, "c.yaml", "n: 31\nq: 0.05\nflag: true\nnothing: null\n")
    assert load_config(cfg) == {"n": 31, "q": 0.05, "flag": True, "nothing": None}


def test_unknown_variable_raises_naming_it(tmp_path):
    """A typo must fail here, not after dispatching a multi-hour job."""
    _write(tmp_path, "runs.yaml", REGISTRY)
    cfg = _write(tmp_path, "c.yaml", "path: ${jumprelu_ruh}\n")
    with pytest.raises(KeyError, match="jumprelu_ruh"):
        load_config(cfg)


def test_registry_is_found_from_a_subdirectory(tmp_path):
    _write(tmp_path, "runs.yaml", REGISTRY)
    cfg = _write(tmp_path, "fourarm/c.yaml", "path: ${jumprelu_run}\n")
    assert find_registry(cfg) == tmp_path / "runs.yaml"
    assert load_config(cfg)["path"] == "jumprelu_L16_s42_20260924T000000Z"


def test_config_without_variables_needs_no_registry(tmp_path):
    cfg = _write(tmp_path, "c.yaml", "a: 1\nb: plain/path\n")
    assert load_config(cfg) == {"a": 1, "b": "plain/path"}


# ---------------------------------------------------------------------------
# The real configs
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_every_repo_config_resolves():
    """No config may reference a variable the registry does not define."""
    failures = []
    for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
        if path.name == "runs.yaml":
            continue
        try:
            loaded = load_config(path)
        except KeyError as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if loaded is not None and "${" in yaml.safe_dump(loaded):
            failures.append(f"{path.name}: unexpanded ${{}} survived substitution")
    assert not failures, "\n".join(failures)


def test_no_config_hardcodes_a_known_run_id():
    """Run ids belong in runs.yaml. This is what keeps a retrain a one-line edit."""
    registry = load_registry(CONFIGS_DIR / "runs.yaml")
    run_ids = [v for k, v in registry.items() if k.endswith("_run")]
    offenders = []
    for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
        if path.name == "runs.yaml":
            continue
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue  # prose comments may name a run
            if any(run_id in line for run_id in run_ids):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
