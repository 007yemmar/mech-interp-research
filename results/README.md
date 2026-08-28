# `results/` — analysis artifacts

Outputs of the analysis pipeline, pulled from the Modal `sae-artifacts` volume so
the numbers behind the paper and the resubmission work are reproducible without
Modal access.

## What is tracked vs local-only

| Tracked in git | Local-only (git-ignored) |
|---|---|
| aggregate **summary JSON** (`*_summary.json`, `grounding_summary.json`, `monospecificity.json`, `run_summary.json`, manifests) | row-level **CSV** (`*_per_code.csv`, `off_target_long*.csv`, `concordance_results.csv`, `ablation_results.csv`, `retrieval_verdicts.csv`, …) |
| `code_names.json` | **npz** matrices (`correlation_matrices.npz`, `dm_correlation_matrix.npz`) |
| this README | per-shard checkpoints (`shard_ckpt*/`, `shard_results/`) — never pulled |

`.gitignore` blocks `*.csv`, `*.npy`, `*.npz`, `*.pt`, `*.safetensors`. The
summary JSONs carry every headline number; the CSV/npz are the backing detail,
kept locally as the staging area for the eventual non-PHI artifact release.

Re-pull any local-only file with `modal volume get sae-artifacts <path> <dest>`
— source paths are in each section below.

## PHI status of the local-only files

- **Numeric CSV / npz** (diff-in-means, random-matched off-target, ablation
  deltas): columns are feature ids, code strings, correlations, losses, Cliff's
  δ. No free text. PHI-safe.
- **`concordance_results.csv`, `retrieval_verdicts.csv`** (LLM feature
  explanations + judge rationales): scanned for MIMIC de-id markers (`[** **]`),
  raw dates, names/titles/ages/MRNs — none found. Quoted spans are the model's
  own analytical prose plus short standardized discharge-instruction templates.
  **Low risk, but do a human read-through before any public release** — this is
  the file the paper commits to releasing as a "non-PHI artifact".
- `retrieval_verdicts.csv` also contains a verbatim `prompt` column; strip it (and
  `judge_raw_output`) before release, keep only the scored columns.

---

## `necessity/` — SAE-necessity baseline suite (resubmission Block A)

Answers the meta-review's open question: *is an SAE needed to produce the audit
signals, or would a non-learned decomposition do as well?*

### `necessity/random_matched/seed0/` — A4, covariance-matched random directions

Source: `sae-artifacts:necessity/random_matched/seed0/`. 18,432 directions drawn
from `N(0, Σ_activations)` (Σ on train shards 0–3), run through the SAE's exact
pipeline (project per token → threshold → max-pool → best-per-code → audit) on
held-out shards 281–311. Four sparsity arms:

| arm | sparsity match | note-level density |
|---|---|---|
| `audit_dense/` | none (control) | 1.00 |
| `audit_l0_40.92/` | JumpReLU token-level L0 | 0.95 |
| `audit_l0_47.57/` | vanilla token-level L0 | 0.96 |
| `audit_note_matched/` | JumpReLU **per-feature note-level** distribution — the arm to report | 0.66 (vs SAE 0.61) |

Grounded-latent count by |r| threshold (held-out; cf. paper Table 1 JumpReLU
9,721 / 610 / 147 at >0.1 / >0.3 / >0.5, peak |r| 0.864):

| arm | >0.1 | >0.2 | >0.3 | >0.4 | >0.5 | peak \|r\| |
|---|---|---|---|---|---|---|
| dense | 10,988 | 538 | 40 | 2 | 0 | 0.431 |
| note_matched | 6,945 | 249 | 16 | 1 | 0 | 0.432 |
| l0_40.92 | 9,132 | 127 | 1 | 0 | 0 | 0.314 |
| l0_47.57 | 8,994 | 108 | 1 | 0 | 0 | 0.314 |

Random directions match the SAE only at the |r|>0.1 acuity floor, then collapse
15–600× by |r|>0.3 and hit zero by |r|>0.5; peak |r| never exceeds 0.43.

Per arm: `audit_summary.json` (headline), `grounding_summary.json`,
`monospecificity.json` tracked; `selected_features.csv`, `off_target_summary*.csv`,
`off_target_long*.csv`, `grounded_latents.csv`, `top_associations.csv`,
`correlation_matrices.npz` local-only.

**Not yet run:** PCA arm (A5) — `random_matched.py` computes the eigenvectors but
`run_random_matched` audits only random arms; ICA (A6); an SAE reference pass
through `necessity_audit.py` itself (needed for a strictly matched comparison).

### `necessity/diff_in_means/` — A1, difference-in-means directions

Source: `sae-artifacts:diff_in_means/`. One unit direction per code
`mean(X⁺) − mean(X⁻)` on **raw centered** pooled activations (train), projected on
held-out, audited with the c-negative off-target instrument. Vanilla SAE run
through the same instrument for comparison.

`summary.json` (tracked): diff-in-means median on-target |r| 0.121, specificity
ratio 1.23×, 17 sig. off-target codes — vs vanilla SAE 0.574 / 15.9× / 2.

⚠️ **Caveat for the write-up:** directions are built on un-whitened activations
(Σ max-variance 3950 vs mean 13), so a few length/acuity dimensions dominate and
on-target |r| collapses to 0.12. The plan called for diagonal-whitening; the code
does not do it. Re-run with a whitened variant (the steelman) alongside the raw
one before drawing conclusions. `dm_per_code.csv`, `dm_off_target_long.csv`,
`sae_vanilla_*.csv`, `dm_correlation_matrix.npz` local-only.

---

## `ablation/` — causal ablation (resubmission Block B4 + committed vanilla ablation)

Sources under `sae-artifacts:ablation/`. All on 4,911 held-out notes, loss window
= final 25% of tokens.

| dir | intervention | features | median Cliff's δ (grounded) | sig q0.05 | controls |
|---|---|---|---|---|---|
| `jumprelu_pilot_extended/` | zero | 20 | 0.090 | 15/20 | — |
| `vanilla_pilot_extended/` | zero | 20 | 0.118 | 15/20 | — |
| `vanilla_meanabl/` | **mean** | 32 | 0.169 | 23/32 | δ = −0.012 (null) |
| `vanilla_section/` | zero, section-local | 32 | 0.162 | 25/32 | δ = −0.036 (null) |
| `gemma_scope_pilot_extended/` | zero | 20 | 0.169 | 15/20 | — |

Vanilla ablation (the change committed in the rebuttal) is done and transfers
from JumpReLU as predicted; mean-ablation and zeroing agree (0.169 vs 0.162).
GemmaScope recon tax 0.648 nats vs vanilla 0.029 (22×). `vanilla_section` off-
target ICD specificity: top feature 0/45 off-target codes significant.

`ablation_summary.json` + `mean_acts.json` tracked; `ablation_results.csv`
(per-note deltas, numeric) and `posthoc_specificity.csv` local-only.

---

## `auto_interp/multi_judge/` — rebuttal Tables T1 / T2 (resubmission Block B5)

Source: `sae-artifacts:auto_interp/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/`.
380 grounded JumpReLU features, three judge families.

### `arm0/<judge>/concordance_summary.json` — verbatim-prompt concordance replication

Concordance (YES+PARTIAL)/N and exact-YES at |r| > 0.1 / 0.3 / 0.5:

| judge | concordance | exact YES |
|---|---|---|
| sonnet-4-6 | 85.5 / 95.0 / 98.6 | 22.6 / 30.7 / 43.1 |
| gpt-4o | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 |
| deepseek-v3 | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 |

### `retrieval/<judge>/retrieval_summary.json` — blind forced-choice retrieval

Judge blind to |r| and to the target recovers the code from a 9-way slate
(chance 11.1%). hit@1 at |r| > 0.1 / 0.3 / 0.5:

| judge | hit@1 |
|---|---|
| gpt-4o | 74.2 / 94.3 / 98.6 |
| sonnet-4-6 | 71.1 / 90.4 / 95.8 |
| deepseek-v3 | 70.0 / 90.4 / 94.4 |

`concordance_results.csv` and `retrieval_verdicts.csv` local-only — see PHI note.

## `auto_interp/shuffled_control/` — scorer null (resubmission Block B6)

`shuffled_control_summary.json`: fuzzing real 0.931 vs shuffled 0.496 (Δ 0.436);
detection real 0.961 vs shuffled 0.516 (Δ 0.440); Wilcoxon p ≈ 0; stable across
all tiers incl. dead. Real-score cross-check vs published within 0.007.
`shuffled_control_per_feature.csv` local-only.

---

## Pre-existing paper artifacts (already in the working tree)

`gemma/`, `jumprelu/`, `vanilla/` hold `test_split/` grounding + posthoc and
`jumprelu/auto_interp/` summaries backing the submitted paper's §4 tables.
