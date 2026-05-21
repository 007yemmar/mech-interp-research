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

# Modal — activation scale measurement (for l1_coeff / lambda_l0 calibration)
modal run modal_app/measure_sigma.py --run-id <centered_run_id>

# Modal — SAE training (vanilla ReLU)
modal run modal_app/train_sae.py --config-file configs/sae_train_2k.yaml
MODAL_GPU=A100-40GB modal run modal_app/train_sae.py --config-file configs/sae_train_50k.yaml

# Modal — SAE training (JumpReLU)
modal run modal_app/train_jumprelu.py --config-file configs/jumprelu_cal.yaml
MODAL_GPU=A100-40GB modal run modal_app/train_jumprelu.py --config-file configs/jumprelu_50k.yaml

# Modal — quick SAE eval (auto-detects vanilla vs JumpReLU)
modal run modal_app/eval_sae.py \
    --checkpoint-dir /out/saes/<run_id>/best \
    --activations-dir /out/activations/<run_id>_centered

# Modal — ICD-9 clinical grounding evaluation (run after SAE training)
modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml

# Modal — GemmaScope SAE baseline eval (uses raw activations, no training needed)
modal run modal_app/gemma_scope_eval.py --config-file configs/gemma_scope_eval.yaml

# Modal — post-hoc analyses (threshold sweep, partial correlation, monospecificity)
modal run modal_app/icd_eval_posthoc.py --config-file configs/icd_eval_posthoc.yaml

# Modal — baselines (run after ICD eval produces shard_ckpt/)
modal run modal_app/lexical_baseline.py --config-file configs/lexical_baseline.yaml
modal run modal_app/tfidf_lr_baseline.py --config-file configs/tfidf_lr_baseline_jumprelu.yaml

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
modal_app/measure_sigma.py         # activation scale stats for l1_coeff / lambda_l0 estimation
    ↓
┌──────────────────────────────────────────────────────────────────┐
│  modal_app/train_sae.py          # VanillaSAE (ReLU + L1)      │
│  modal_app/train_jumprelu.py     # JumpReLU SAE (L0 + STE)     │
└──────────────────────────────────────────────────────────────────┘
    ↓
modal_app/eval_sae.py              # quick L0/EV/MSE/dead-frac eval (auto-detects flavour)
    ↓
modal_app/icd_eval.py              # ICD-9 clinical grounding eval → grounding_summary.json
modal_app/gemma_scope_eval.py      # GemmaScope baseline: diagnostics + ICD grounding on raw acts
    ↓
modal_app/icd_eval_posthoc.py      # threshold sweep, partial-r (n_tokens confound), monospecificity
    ↓
modal_app/lexical_baseline.py      # keyword co-occurrence control baseline
modal_app/tfidf_lr_baseline.py     # TF-IDF + LR classification baseline (stratified k-fold CV)
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
| `sae_data.py` | `ActivationsBuffer` — loads shards in shuffled order into a 1M-token RAM window, drains in `batch_size` chunks; `EvalAggregator` for streaming eval metrics |
| `sae_train.py` | `VanillaSAE` (ReLU + L1) + `train_step` + `resample_dead_neurons` + `train` loop with eval-driven early stopping |
| `jumprelu_config.py` | `JumpReLUConfig` dataclass; separate from `SAETrainingConfig` — adds `bandwidth`, `log_threshold_init`, `lambda_l0`, `lambda_l0_warmup_steps` |
| `jumprelu_sae.py` | `JumpReLUSAE` model, STE-based `train_step` (`_reconstruction_ste` + `_l0_surrogate`), deterministic resume (optimizer + scheduler + RNG + W&B run ID), no dead-neuron resampling |
| `icd_eval.py` | ICD-9 grounding pipeline: `JumpReLUSAE` (numpy-only encoder), `encode_and_pool`, vectorised point-biserial correlation, BH FDR correction, `run_icd_eval` orchestrator; post-hoc helpers: `reassemble_note_vectors`, `compute_partial_point_biserial`, `compute_monospecificity`, `run_posthoc_analyses` |
| `lexical_baseline.py` | Keyword co-occurrence baseline: YAML keyword dict → regex indicators → point-biserial correlation → head-to-head vs SAE; keyword-absent recall analysis |
| `tfidf_lr_baseline.py` | TF-IDF + LR baseline: per-code stratified k-fold CV (AUC-ROC/PR), Wilcoxon signed-rank paired significance, supplementary best-feature correlation comparison |

### `modal_app/` — Modal entrypoints

Every entrypoint imports shared primitives from `modal_app/app.py` (App, image, volumes, secrets). The image ships both `mech_interp_research` and `modal_app` as local Python sources — `modal_app` must be shipped because `extract.py` imports from `modal_app.app` at container-import time. Keyword YAML files are shipped via `.add_local_file()`.

GPU selection: set `MODAL_GPU=<tier>` in the shell before `modal run`. The value is resolved at module import time, not per-invocation. Default is `L4`.

### Key implementation constraints

- **`W_enc` contiguity**: `W_enc` is initialized as `W_dec.data.T.contiguous()`. `.T` produces a non-contiguous view; safetensors refuses non-contiguous tensors. Always call `.contiguous()` before saving any tensor derived from a transpose.
- **Decoder norm constraint**: implemented as gradient surgery (`remove_gradient_parallel_to_decoder_directions`) _before_ the optimizer step, then hard renorm (`set_decoder_norm_to_unit_norm`) _after_. Both are required; skipping either breaks the constraint. Identical in both vanilla and JumpReLU SAEs.
- **Centering uses float64 accumulation**: activations are float16 on disk, summed as float32→float64 to avoid precision loss over millions of tokens. The saved `mean.pt` is float32.
- **`ActivationsBuffer` is not a `Dataset`**: each shard is 2–5 GB. A random-access Dataset would cause near-100% shard miss rate. The buffer loads full shards sequentially, shuffles within the 1M-token window, and drains batch-by-batch. `reset_epoch()` must be called before the second and later epochs.
- **`git_sha` pre-resolution**: Modal containers have no `git` binary. `modal_app/extract.py`'s `local_entrypoint` calls `_git_sha_short()` on the laptop and injects it into the config dict before dispatching. Without this, every run is stamped `nogit`.
- **JumpReLU STE dual-path**: the training step maintains two separate computation paths sharing the same `pre_act` tensor. The reconstruction path (`_reconstruction_ste`) provides a downward pull on thresholds for useful features. The sparsity path (`_l0_surrogate`) provides an upward push via a rectangular pseudo-derivative of bandwidth ε. Both are required for stable threshold equilibrium.
- **JumpReLU deterministic resume**: `save_checkpoint` / `load_train_state` persist optimizer state, scheduler state, CPU+CUDA RNG state, W&B run ID, and `(step, epoch, step_in_epoch)` counters. On resume, the `ActivationsBuffer` is fast-forwarded to the exact saved position so training continues deterministically.

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
/out/saes/<sae_run_id>/                    # vanilla or JumpReLU
    best/sae_weights.safetensors           # best eval-EV checkpoint (vanilla only; JumpReLU uses final)
    best/sae_config.yaml
    step_NNNNNNNN/                         # periodic checkpoints (+ train_state.pt for resume)
    final/sae_weights.safetensors          # W_enc, W_dec, b_enc, b_dec [+ log_threshold for JumpReLU]
    final/sae_config.yaml
    train_summary.json
/out/icd_eval/<eval_id>/
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
        posthoc_summary.json
        grounding_r0.1/ ... grounding_r0.5/
        partial/
            correlation_matrices.npz
            grounding_r0.1/ ... grounding_r0.5/
            code_names.json
    lexical/                              # lexical baseline output
        lexical_baseline_summary.json
        per_code_comparison.csv
        keyword_coverage.json
    tfidf_lr/                             # TF-IDF + LR baseline output
        tfidf_lr_summary.json
        per_code_comparison.csv
        tfidf_cv_results.csv / sae_cv_results.csv
        tfidf_vocabulary.json
        cv_ckpt_tfidf/ cv_ckpt_sae/      # per-code CV checkpoints (resume support)
```

### Data handling rules (MIMIC-IV / PHI)

1. Never commit data files. `.gitignore` blocks `*.csv`, `*.parquet`, `*.pt`, `*.safetensors`, `data/`, `outputs/`, `.tmp/`. Pre-commit rejects anything over 500 KB.
2. Never paste note text into issues, PRs, Slack, commit messages, or log output. Keep all verification structural (row counts, dtypes, character-length integers).
3. HF token lives only in the Modal secret `huggingface-token`. Never in `.env`, CI, or code.

### SAE calibration workflow

**Vanilla SAE (ReLU + L1):**

1. Run `measure_sigma.py` on centered activations; note the σ (sigma) value.
2. Run 2k calibration training; inspect step logs for L0 (mean non-zero features per token).
3. Target: L0 in **[20, 80]**. If L0 > 100: multiply `l1_coeff` by 2–4×. If L0 < 10: divide by 2–4×.
4. Do not start the 50k run until L0 is in range on the 2k run.
5. 50k full run requires `MODAL_GPU=A100-40GB` and W&B secret `wandb-token` (set `wandb_project: sae-mimic` in config).

**JumpReLU SAE (L0 + STE):**

1. Run `measure_sigma.py` (same as above).
2. Run calibration with `jumprelu_cal.yaml`; monitor L0, `mean_threshold`, and `threshold_std`.
3. Target: L0 ≈ 35. Tune `lambda_l0` (higher → sparser). Watch for threshold collapse (all thresholds converging → increase `bandwidth`).
4. `bandwidth` (ε): paper default 0.001; SAELens default 0.05. Start with 0.001; bump to 0.05 if thresholds fail to move.
5. 50k full run: same GPU/W&B requirements as vanilla.

### ICD-9 grounding eval

`icd_eval.py` runs after a trained SAE is available. Edit `configs/icd_eval.yaml` to point at the centered activations dir, SAE checkpoint (`best/` for vanilla, `final/` for JumpReLU), ICD CSV, and output dir, then run:

```bash
modal run modal_app/icd_eval.py --config-file configs/icd_eval.yaml
```

Key operational notes:

- **Runtime**: the 50k run has 312 shards at ~100 s/shard → ~9 hours. `timeout=43200` (12 h) in the Modal decorator.
- **Resume after preemption or timeout**: `encode_and_pool` checkpoints each shard's pooled vectors to `output_dir/shard_ckpt/` on the `sae-artifacts` volume. Re-running the same command resumes automatically — already-done shards are skipped. Modal does **not** always auto-restart preempted `.remote()` calls; if nothing happens after a few minutes, `Ctrl+C` and re-run manually.
- **JumpReLUSAE is numpy-only**: inference uses no torch — safetensors weights are loaded with `safetensors.numpy.load_file`. Plain ReLU checkpoints (VanillaSAE) are handled by defaulting `threshold=0`.
- **Pooling strategy**: default is `max` (element-wise max across tokens per note). `mean` and `topk_mean` are also supported via the `pooling` config key.
- **`modal volume ls` paths**: use volume-relative paths without `/out/` prefix, e.g. `modal volume ls sae-artifacts activations/` not `modal volume ls sae-artifacts /out/activations/`.

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

### Baselines

Two control baselines test whether SAE grounding reflects genuine learned representation structure or is explained by surface-level text features.

**Lexical keyword co-occurrence** (`lexical_baseline.py`): builds binary keyword indicators from a curated YAML dictionary, computes point-biserial correlation vs ICD labels, and classifies each code as `sae_above_lexical`, `comparable`, or `lexical_above_sae`. Also computes keyword-absent recall — whether the best SAE latent fires on positive notes where no keyword appears (evidence of beyond-surface-level learning).

**TF-IDF + Logistic Regression** (`tfidf_lr_baseline.py`): fits TF-IDF (10k features, 1+2-grams, sublinear TF) on matched note texts, then trains per-code LR classifiers on both TF-IDF and SAE features using stratified 5-fold CV. Compares AUC-ROC and AUC-PR head-to-head per code, with Wilcoxon signed-rank paired significance test across all codes. Also computes supplementary best-feature point-biserial correlation.

Both baselines depend on `shard_ckpt/` from a completed ICD eval run and must be run after `icd_eval.py`.
