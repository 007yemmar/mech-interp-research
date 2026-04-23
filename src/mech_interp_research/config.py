"""Run configuration for activation extraction."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExtractionConfig:
    """Fully-specified configuration for one extraction run.

    Paths are literal filesystem paths as seen by the process running extraction:
    on Modal, that means `/raw/train.csv` and `/out/activations` (volume mounts);
    locally, that's a repo-relative path like `.tmp/synthetic.csv`.
    """

    input_csv: str
    output_root: str
    text_col: str | None = None
    num_notes: int = 2000
    max_length: int = 2048
    layer: int = 16
    tokens_per_shard: int = 250_000
    model_name: str = "google/gemma-2-2b"
    run_id: str | None = None


def load_config(path: str | Path) -> ExtractionConfig:
    """Load an ExtractionConfig from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExtractionConfig(**data)


def save_config(config: ExtractionConfig, path: str | Path) -> None:
    """Write an ExtractionConfig to a YAML file (sorted keys, human-readable)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=True)


def _git_sha_short() -> str:
    """Short git SHA of HEAD, or 'nogit' if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def make_run_id(config: ExtractionConfig) -> str:
    """Build a collision-free run ID: <model>_L<layer>_<N>notes_<sha>_<utc>."""
    model_slug = config.model_name.replace("/", "-")
    sha = _git_sha_short()
    utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_slug}_L{config.layer}_{config.num_notes}notes_{sha}_{utc}"
