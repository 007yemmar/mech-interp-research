#!/usr/bin/env python
"""Run activation extraction locally (CPU/MPS/CUDA on this machine)."""

from __future__ import annotations

import argparse
import json
import sys

from mech_interp_research.config import load_config
from mech_interp_research.extraction import run_extraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Local activation extraction runner.")
    parser.add_argument("--config-file", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    summary = run_extraction(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
