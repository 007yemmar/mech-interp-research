# Runbook

Operational notes for the `mech-interp-research` Modal workspace (`mech-interp-rmd`).

## First-time Modal workspace setup (admin, once)

1. Sign up at https://modal.com, create a workspace named `mech-interp-rmd`.
2. Invite teammates via Settings → Members.
3. Stay on Starter plan (free $30/month compute credit).
4. Dashboard → Billing → set a budget alert (recommend: email at $20, hard-stop at $50).

## Create volumes (admin, once)

```bash
modal volume create mimic-iv-raw
modal volume create sae-artifacts
modal volume list    # sanity check
```

## Create secrets (admin, once)

```bash
modal secret create huggingface-token HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

(W&B secret deferred until SAE training phase.)

## Onboarding a new teammate

1. Admin invites them to the GitHub repo and the Modal workspace.
2. New member on their laptop:
   ```bash
   git clone <repo>
   cd mech-interp-research
   uv sync
   uv run pre-commit install
   modal setup
   modal profile activate mech-interp-rmd
   ```
3. Obtain `./test.csv` out-of-band from a credentialed teammate (never via git, Slack, or email).
4. Run the local smoke:
   ```bash
   uv run python scripts/local_extract.py --config-file configs/smoke.yaml
   ```
5. Run the Modal smoke (after the admin has done the workspace setup and uploaded `/test.csv`):
   ```bash
   modal run modal_app/extract.py --config-file configs/smoke_modal.yaml
   ```
6. Verify their run ID shows up:
   ```bash
   modal volume ls sae-artifacts /activations/
   ```

## Seed MIMIC-IV into `mimic-iv-raw`

Done once by the credentialed teammate:

```bash
modal volume put mimic-iv-raw ./test.csv /test.csv
modal volume ls mimic-iv-raw
```

## Rotate the HF token

```bash
modal secret create huggingface-token --force HF_TOKEN=hf_newtoken
```

## Wipe a volume (DUA offboarding)

```bash
modal volume rm mimic-iv-raw --confirm
modal volume rm sae-artifacts --confirm
```

## Common errors

- **`modal.exception.NotFoundError: Volume 'mimic-iv-raw' not found`** — admin hasn't run `modal volume create`. See setup above.
- **`403 Forbidden` on `google/gemma-2-2b`** — HF token doesn't have Gemma access. Accept the license at https://huggingface.co/google/gemma-2-2b, then rotate the Modal secret.
- **Modal function timeout at 3600s** — scale-up run. Raise `timeout` in the `@app.function` decorator before increasing note count.
- **Cold-start latency ~30-90s on first run of a new image** — expected. Subsequent runs reuse the cached image layer.
- **`volume.commit()` error after function completes** — ensure the function returned normally; exceptions short-circuit the commit. Re-run.
- **`torch_dtype is deprecated` warning from transformers** — harmless; will resolve when the minimum transformers version is bumped and `load_model_and_tokenizer` switches to `dtype=`.
