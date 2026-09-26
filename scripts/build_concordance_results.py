"""Build the §14 concordance-validation tables from per-judge verdict CSVs.

Two modes:

    # 1. pull every verdict CSV named by the sources config off the volume
    uv run python scripts/build_concordance_results.py --fetch

    # 2. rebuild results/concordance_validation/F*.csv (+ a §14-style markdown)
    uv run python scripts/build_concordance_results.py \
        --out results/concordance_validation/

The fixtures dir mirrors volume paths (``<fixtures>/auto_interp/<pool>/...``). It
holds MIMIC-derived text (explanations, rationales) and must stay under the
gitignored ``.tmp/``. Nothing here prints CSV contents -- only paths and counts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mech_interp_research import concordance_tables as ct  # noqa: E402

VOLUME = "sae-artifacts"
ARM_FILES = {
    "arm0_eval": "concordance_results.csv",
    "retrieval_eval": "retrieval_verdicts.csv",
    "retrieval_eval_hardneg": "retrieval_verdicts.csv",
    "deanchored_eval": "deanchored_verdicts.csv",
    "binary_eval": "binary_verdicts.csv",
}
ARM_KIND = {
    "arm0_eval": "concordance",
    "retrieval_eval": "retrieval",
    "retrieval_eval_hardneg": "retrieval",
    "deanchored_eval": "deanchored",
    "binary_eval": "binary",
}


def _ls(remote: str) -> list[str]:
    proc = subprocess.run(
        ["modal", "volume", "ls", VOLUME, remote.rstrip("/") + "/"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [Path(line.strip()).name for line in proc.stdout.splitlines() if line.strip()]


def _get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", VOLUME, remote, str(local)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def fetch(cfg: dict, fixtures: Path) -> None:
    """Download every verdict CSV the sources config can use into ``fixtures``."""
    for src in cfg["sources"]:
        for pool in src["pool_dirs"]:
            got = []
            if _get(f"{pool}/concordance_results.csv", fixtures / pool / "concordance_results.csv"):
                got.append("concordance_results.csv")
            for arm, fname in ARM_FILES.items():
                present = set(_ls(f"{pool}/{arm}"))
                wanted = set()
                for j in cfg["judges"].values():
                    for key in (ARM_KIND[arm], f"{ARM_KIND[arm]}_s14"):
                        slugs = j.get(key, [])
                        wanted |= {slugs} if isinstance(slugs, str) else set(slugs)
                for slug in sorted(present & wanted):
                    rel = f"{pool}/{arm}/{slug}/{fname}"
                    if _get(rel, fixtures / rel):
                        got.append(f"{arm}/{slug}")
            print(f"{src['key']:<16} {pool}: {len(got)} files", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default=str(REPO / "configs/concordance_sources.yaml"))
    ap.add_argument("--fixtures", default=str(REPO / ".tmp/concordance_fixtures"))
    ap.add_argument("--out", default=str(REPO / ".tmp/concordance_tables_out"))
    ap.add_argument("--fetch", action="store_true", help="download CSVs, then exit")
    args = ap.parse_args(argv)

    cfg = ct.load_sources(args.sources)
    fixtures = Path(args.fixtures)
    if args.fetch:
        fetch(cfg, fixtures)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = ct.build_all(cfg, fixtures)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / (name + '.csv')}  ({len(df)} rows)")
    md = ct.section14_markdown(cfg, fixtures)
    (out / "SECTION14_TABLES.md").write_text(md, encoding="utf-8")
    print(f"wrote {out / 'SECTION14_TABLES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
