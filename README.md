# mech-interp-research

Pipeline for extracting residual-stream activations from Gemma-2-2b over clinical notes, targeting SAE fine-tuning. Heavy compute runs on Modal; local laptop runs are supported for fast iteration.

## Prerequisites

- Python 3.11 (via `uv`)
- `uv` installed: https://docs.astral.sh/uv/
- Modal account with membership in the `mech-interp-rmd` workspace
- Hugging Face account with access granted to `google/gemma-2-2b` (for real runs; smoke runs use `gpt2` and need no HF auth)

## First-time setup

```bash
git clone <repo-url>
cd mech-interp-research
uv sync                       # creates .venv, installs deps from uv.lock
uv run pre-commit install     # wire up lint/format hooks

modal setup                   # authenticates your Modal CLI
modal profile activate mech-interp-rmd
```

## Run a local smoke test

Requires `./test.csv` in the repo root (gitignored; obtained out-of-band from a credentialed teammate).

```bash
uv run python scripts/local_extract.py --config-file configs/smoke.yaml
```

Runs `gpt2` over 3 notes from `./test.csv`. Should finish in ~1 minute on a laptop (CSV parse dominates).

## Run on Modal — smoke

Upload `test.csv` to the shared volume once (any credentialed teammate):

```bash
modal volume put mimic-iv-raw ./test.csv /test.csv
modal run modal_app/extract.py --config-file configs/smoke_modal.yaml
```

## Run on Modal — real extraction

`/test.csv` is already on the volume from the smoke upload. Real extraction runs against the same file with full `gemma-2-2b` + `num_notes: 2000` + `max_length: 2048`:

```bash
modal run modal_app/extract.py --config-file configs/extract_2k_notes.yaml
```

Artifacts land in the `sae-artifacts` volume under `activations/<run-id>/`.

## Layout

- `src/mech_interp_research/` — model-agnostic extraction library
- `modal_app/` — Modal app, image, entrypoints
- `scripts/` — local-runnable CLI
- `configs/` — YAML configs, one per experiment
- `docs/runbook.md` — Modal admin ops, onboarding, common errors

## Contributing

See `CONTRIBUTING.md`.
