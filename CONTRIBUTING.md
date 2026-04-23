# Contributing

## Branch and PR norms

- `main` is the integration branch; never force-push to it.
- Feature work goes on `<initials>/<topic>` branches.
- Open a PR before merging, even for solo work — it gives teammates an opt-in review surface.
- Squash-merge to keep history linear.

## Pre-commit

Hooks run `ruff format`, `ruff check --fix`, trailing-whitespace, EOF-fixer, merge-conflict-marker check, and a 500KB file-size cap (the last is our guard against accidental data commits).

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Data handling rules

MIMIC-IV is credentialed, PHI-adjacent data. Three hard rules:

1. **Never commit data.** `.gitignore` blocks `*.csv`, `*.parquet`, `*.pt`, `*.safetensors`, `data/`, `outputs/`, `.tmp/`. Pre-commit rejects anything over 500KB.
2. **Never paste note text.** Not into GitHub issues, PRs, Slack, commit messages, or log statements. Keep verification outputs structural (column names, row counts, dtype, char-length ints).
3. **Credentials stay in Modal.** The HF token lives only in the Modal secret `huggingface-token`. It never goes into `.env` files, CI, or code.

When the project ends, run `modal volume rm mimic-iv-raw` to discharge the DUA.
