# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install deps and set up hooks (first time)
uv sync
uv run pre-commit install

# Run all tests
uv run pytest tests/

# Run a single test file or test
uv run pytest tests/test_sae_train.py -v
uv run pytest tests/test_smoke_pipeline.py::test_full_pipeline_smoke -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pre-commit run --all-files

# Modal — extraction
modal run modal_app/extract.py --config-file configs/smoke_modal.yaml
modal run modal_app/extract.py --config-file configs/extract_2k_notes.yaml

# Modal — post-extraction steps (run in order)
modal run modal_app/subset.py --run-id <run_id> --n-shards 3
modal run modal_app/center.py --run-id <run_id>
modal run modal_app/train_sae.py --config-file configs/sae_train_2k.yaml
MODAL_GPU=A100-40GB modal run modal_app/train_sae.py --config-file configs/sae_train_50k.yaml

# Modal — ICD-9 clinical grounding evaluation (run after SAE training)
modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml

# Modal — GemmaScope SAE baseline eval (uses raw activations, no training needed)
modal run modal_app/gemma_scope_eval.py --config-file configs/gemma_scope_eval.yaml

# Modal — post-hoc analyses (threshold sweep, partial correlation, monospecificity)
modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml

# Inspect Modal volumes (paths are relative to volume root, no /out/ prefix)
modal volume ls sae-artifacts activations/
modal volume ls sae-artifacts saes/
modal volume ls sae-artifacts icd_eval/
```

**Before any Modal run, `uv run pytest tests/` must pass.** `tests/test_smoke_pipeline.py` is the primary gating check — it runs the full `center → train → checkpoint` round-trip on CPU in < 30 s.

## Architecture

### Pipeline overview

```
scripts/prepare_sample.py          # stratified sample from train/val CSVs
    ↓  (CSV on mimic-iv-raw volume)
modal_app/extract.py               # Gemma-2-2B layer-16 activations → sharded fp16 safetensors
    ↓  (optional)
modal_app/subset.py                # random shard subset for calibration runs
    ↓
modal_app/center.py                # two-pass exact global mean subtraction
    ↓
modal_app/train_sae.py             # VanillaSAE training → checkpoints + train_summary.json
    ↓
modal_app/icd_eval.py              # ICD-9 clinical grounding eval → grounding_summary.json + artefacts
modal_app/gemma_scope_eval.py      # GemmaScope baseline: L0/EV/dead-frac + ICD grounding on raw activations
modal_app/icd_eval_posthoc.py     # Post-hoc: threshold sweep, partial-r (n_tokens confound), monospecificity
```

All heavy compute runs on Modal. The `mimic-iv-raw` volume holds input CSVs; `sae-artifacts` holds every downstream artifact (activations, centered activations, SAE checkpoints).

### `src/mech_interp_research/` — the library

| Module | Role |
|---|---|
| `config.py` | `ExtractionConfig` dataclass; `make_run_id` stamps `<model>_L<layer>_<N>notes_<sha>_<utc>` |
| `data.py` | CSV → DataFrame with text-col auto-detection |
| `model.py` | HF model/tokenizer loader; device selection (cuda > mps > cpu) |
| `extraction.py` | `extract_one_note` (fp32, CPU) + `run_extraction` orchestrator |
| `storage.py` | `ShardedSafetensorsWriter` — buffers per-note activations, flushes fp16 shards at `tokens_per_shard` boundary; writes `manifest.json` + `metadata.jsonl` |
| `checks.py` | Shape, finiteness, non-zero, diversity assertions run after extraction |
| `center.py` | `center_run` — two-pass mean subtraction (float64 accumulator, float32 arithmetic, float16 output) |
| `sae_config.py` | `SAETrainingConfig` dataclass; `d_sae` is always `d_in × expansion_factor` (never set separately) |
| `sae_data.py` | `ActivationsBuffer` — loads shards in shuffled order into a 1M-token RAM window, drains in `batch_size` chunks |
| `sae_train.py` | `VanillaSAE` + `train_step` + `resample_dead_neurons` + `train` loop |
| `icd_eval.py` | ICD-9 grounding pipeline: `JumpReLUSAE` (numpy-only encoder), `encode_and_pool`, vectorised point-biserial correlation, BH FDR correction, `run_icd_eval` orchestrator; post-hoc helpers: `reassemble_note_vectors`, `compute_partial_point_biserial`, `compute_monospecificity`, `run_posthoc_analyses` |

### `modal_app/` — Modal entrypoints

Every entrypoint imports shared primitives from `modal_app/app.py` (App, image, volumes, secrets). The image ships both `mech_interp_research` and `modal_app` as local Python sources — `modal_app` must be shipped because `extract.py` imports from `modal_app.app` at container-import time.

GPU selection: set `MODAL_GPU=<tier>` in the shell before `modal run`. The value is resolved at module import time, not per-invocation. Default is `L4`.

### Key implementation constraints

- **`W_enc` contiguity**: `W_enc` is initialized as `W_dec.data.T.contiguous()`. `.T` produces a non-contiguous view; safetensors refuses non-contiguous tensors. Always call `.contiguous()` before saving any tensor derived from a transpose.
- **Decoder norm constraint**: implemented as gradient surgery (`remove_gradient_parallel_to_decoder_directions`) _before_ the optimizer step, then hard renorm (`set_decoder_norm_to_unit_norm`) _after_. Both are required; skipping either breaks the constraint.
- **Centering uses float64 accumulation**: activations are float16 on disk, summed as float32→float64 to avoid precision loss over millions of tokens. The saved `mean.pt` is float32.
- **`ActivationsBuffer` is not a `Dataset`**: each shard is 2–5 GB. A random-access Dataset would cause near-100% shard miss rate. The buffer loads full shards sequentially, shuffles within the 1M-token window, and drains batch-by-batch. `reset_epoch()` must be called before the second and later epochs.
- **`git_sha` pre-resolution**: Modal containers have no `git` binary. `modal_app/extract.py`'s `local_entrypoint` calls `_git_sha_short()` on the laptop and injects it into the config dict before dispatching. Without this, every run is stamped `nogit`.

### Artifacts on `sae-artifacts`

```
/out/activations/<run_id>/                 # raw extraction
    shard_NNNN.safetensors                 # {"activations": [tokens, d_model], fp16}
    manifest.json                          # run metadata + centered: false
    metadata.jsonl                         # per-note shard/row_start/row_end
/out/activations/<run_id>_centered/        # after center.py
    shard_NNNN.safetensors                 # fp16, mean-subtracted
    mean.pt                                # float32 [d_model] — keep for inference
    manifest.json                          # centered: true
/out/saes/<sae_run_id>/
    best/sae_weights.safetensors           # best eval-EV checkpoint (use this for eval)
    best/sae_config.yaml
    final/sae_weights.safetensors          # W_enc, W_dec, b_enc, b_dec
    final/sae_config.yaml
    train_summary.json
/out/icd_eval/<sae_run_id>/
    grounding_summary.json                 # top-level metrics (grounded latent %, top assocs)
    correlation_matrices.npz              # r_pb [d_sae, n_codes], p_adjusted, significant
    top_associations.csv                  # latent ↔ ICD code pairs sorted by |r_pb|
    grounded_latents.csv                  # per-latent summary (grounded only)
    per_code_summary.csv                  # per-code grounded latent counts
    code_names.json
    shard_ckpt/                           # per-shard encode checkpoints (resume support)
        shard_NNNN_vectors.npy            # [n_notes, d_sae] pooled vectors
        shard_NNNN_meta.jsonl
    posthoc/                              # post-hoc analyses (icd_eval_posthoc.py)
        posthoc_summary.json              # combined threshold sweep + monospecificity + partial-r
        grounding_r0.1/ ... grounding_r0.5/  # full grounding artefacts at each threshold
        partial/                          # partial correlation (controlling for n_tokens)
            correlation_matrices.npz      # partial r_pb, p_adjusted, significant
            grounding_r0.1/ ... grounding_r0.5/
            code_names.json
```

### Data handling rules (MIMIC-IV / PHI)

1. Never commit data files. `.gitignore` blocks `*.csv`, `*.parquet`, `*.pt`, `*.safetensors`, `data/`, `outputs/`, `.tmp/`. Pre-commit rejects anything over 500 KB.
2. Never paste note text into issues, PRs, Slack, commit messages, or log output. Keep all verification structural (row counts, dtypes, character-length integers).
3. HF token lives only in the Modal secret `huggingface-token`. Never in `.env`, CI, or code.

### ICD-9 grounding eval

`icd_eval.py` runs after a trained SAE is available. Edit `configs/icd_eval.yaml` to point at the centered activations dir, SAE checkpoint (`best/`), ICD CSV, and output dir, then run:

```bash
modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml
```

Key operational notes:

- **Runtime**: the 50k run has 312 shards at ~100 s/shard → ~9 hours. `timeout=43200` (12 h) in the Modal decorator.
- **Resume after preemption or timeout**: `encode_and_pool` checkpoints each shard's pooled vectors to `output_dir/shard_ckpt/` on the `sae-artifacts` volume. Re-running the same command resumes automatically — already-done shards are skipped. Modal does **not** always auto-restart preempted `.remote()` calls; if nothing happens after a few minutes, `Ctrl+C` and re-run manually.
- **JumpReLUSAE is numpy-only**: inference uses no torch — safetensors weights are loaded with `safetensors.numpy.load_file`. Plain ReLU checkpoints (VanillaSAE) are handled by defaulting `threshold=0`.
- **Pooling strategy**: default is `max` (element-wise max across tokens per note). `mean` and `topk_mean` are also supported via the `pooling` config key.
- **`modal volume ls` paths**: use volume-relative paths without `/out/` prefix, e.g. `modal volume ls sae-artifacts activations/` not `modal volume ls sae-artifacts /out/activations/`.

### SAE calibration workflow

1. Run 2k toy training; inspect step logs for L0 (mean non-zero features per token).
2. Target: L0 in **[20, 80]**. If L0 > 100: multiply `l1_coeff` by 2–4×. If L0 < 10: divide by 2–4×.
3. Do not start the 50k run until L0 is in range on the 2k run.
4. 50k full run requires `MODAL_GPU=A100-40GB` and W&B secret `wandb-token` (set `wandb_project: sae-mimic` in config).

### ICD eval pitfalls

- **ICD CSV must match the extraction population.** `icd_csv_path` must point at the same CSV the activations were extracted from (e.g. `sample_50k.csv`), not a different split. A wrong file silently produces a near-empty join. The `min_notes` guard (default 100) in `load_and_align_icd_labels` catches this.
- **`icd9_codes_list` column**: the CSVs contain a semicolon-delimited string column alongside the binary indicators. `load_and_align_icd_labels` filters to numeric columns only — do not remove this filter.
- **Polyspecificity from max-pooling**: with `pooling: max`, longer/sicker notes produce globally higher activation values, so latents tend to correlate with many codes simultaneously (acuity confound). Diagnostic levers: raise `r_threshold` (0.3–0.5), partial-correlate out `n_tokens`, or re-run with `pooling: mean`. Higher-threshold and partial-correlation analyses can be done cheaply on the existing `correlation_matrices.npz` without re-encoding shards.
- **GemmaScope baseline must use the same pooling strategy** as the custom SAE eval for a fair comparison. Only change pooling if you re-run both.

### Post-hoc analyses (icd_eval_posthoc)

`run_posthoc_analyses` runs three cheap analyses on existing eval output (seconds, not hours):

1. **Threshold sweep** — recomputes `compute_grounding` at each `r_threshold` in the list (default 0.1–0.5). Shows how grounded-latent counts drop as the bar rises.
2. **Monospecificity** — at each threshold, counts how many grounded latents associate with exactly 1 code (monospecific), 2–3 codes (oligospecific), or 4+ (polyspecific). Key question: at `|r| > 0.5`, do surviving latents become monospecific?
3. **Partial correlation** — residualizes note-level SAE activations on `n_tokens` (OLS) before computing point-biserial, removing the max-pooling length confound. Runs BH FDR correction and grounding at every threshold. Reassembles `note_vectors` from `shard_ckpt/` — no shard re-encoding.

```bash
modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml
```
