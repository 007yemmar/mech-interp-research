# Auto-Interpretability for Clinical SAE Features (Design v3 — Custom Build)

## Purpose

Add an **auto-interpretability** pillar to the project that answers two reviewer questions the Results section currently leaves open:

1. *"You have 18,432 features. You interpret 3 in §4.6. What do the rest mean?"*
2. *"How do you know your auto-interp explanations are actually correct?"*

Plus it produces a **novel methodological contribution** unique to this project: **ground-truth concordance validation** — comparing LLM-generated explanations against ICD-9 code labels for grounded latents. The general-purpose SAE auto-interp literature cannot do this because it lacks structured external labels.

### Why concordance is a genuine insight (not just confirmation)

The concordance *rate* on strongly-grounded features (r > 0.5) is expected to be high and is table 1. The actual findings come from three deeper analyses:

1. **Partial-match distribution reveals what the SAE actually learned.** A feature grounded to `icd9_2449` (hypothyroidism) that auto-interp labels as "thyroid medication dosing" is a PARTIAL match — the SAE learned a *sub-concept* of the ICD code, not the diagnosis itself. The YES/PARTIAL/NO distribution tells us whether the SAE learns diagnosis-level concepts, procedure/treatment-level concepts, or vocabulary-level proxies. That is a structural insight about what LLMs represent about medicine, not a confirmation of what we already know.

2. **The weakly-grounded zone (r = 0.1–0.3) is where discoveries happen.** Features with moderate statistical signal but clear clinical auto-interp explanations suggest the SAE captured real clinical concepts that point-biserial missed — perhaps because the ICD code is rare, the feature captures a broader context than one code, or the relationship is non-linear. Conversely, features with moderate r but "general language" auto-interp explanations reveal statistical artifacts from the max-pooling length confound.

3. **Non-grounded clinical features are the surprise.** If 15–25% of the non-grounded sample gets categorized as `clinical_concept`, that is evidence the SAE learned clinical structure beyond what 46 ICD codes can measure — a direct argument for richer ontologies (SNOMED-CT, etc.) in future work and evidence against the null hypothesis that grounded features are the only clinically meaningful ones.

### Why custom build over Delphi (decision locked in)

We initially designed around EleutherAI's [Delphi library](https://github.com/EleutherAI/delphi) (arXiv 2410.13928), but deep inspection of its codebase revealed critical integration friction:

- **`LatentCache` is tightly coupled to live HuggingFace model inference** — it runs a `PreTrainedModel` with forward hooks. There is no way to feed pre-computed activations. Bypassing it requires writing directly to `LatentDataset`'s undocumented on-disk format (sparse COO, split safetensors, per-module `config.json`).
- **SAE interface mismatch**: Delphi expects `EncoderOutput(top_acts, top_indices, pre_acts)` — a top-k sparse output. Our JumpReLU SAE produces dense activations with a learned threshold.
- **Token provenance mismatch**: `LatentDataset` re-tokenizes from a HuggingFace dataset in `config.json`. Our tokens come from MIMIC CSVs. Requires monkey-patching `load_tokens()`.
- **Heavy/conflicting dependencies**: `vllm>=0.10.2`, `bitsandbytes`, `eai-sparsify` — would conflict with our Modal image and may not be importable without a GPU.
- **API instability**: Delphi is v0.1.3 (March 2026). The `PotentiallyWrappedSparseCoder` protocol is not exported as a public API. Recent PRs fixed broken scorer contracts and cache path handling.

The custom build avoids all of these while producing equally defensible output. We cite Delphi/Eleuther methodology ("following the Detection and Fuzzing scoring protocols of Paulo et al. 2024") without depending on the library. The concordance validation — our actual contribution — is orthogonal to the explanation infrastructure.

## Scope

**In scope:**

- **JumpReLU SAE only** (the headline architecture)
- ~1,480 features total:
  - All **280 grounded at r > 0.4** (superset of the 144 at r > 0.5)
  - **100 weakly-grounded features** with r = 0.1–0.3 (uniform random with seed=42 from the ~330 features in this range that are BH-significant but below the r > 0.3 grounding threshold)
  - **1,000 non-grounded features** sampled uniform-random (seed=42) from the ~9,400 features with zero BH-significant correlations
  - An additional **100 dead/near-dead features** (lowest mean activation) as a noise floor control
- **Two scoring methods**: Fuzzing (primary) + Detection (supplementary)
  - Fuzzing is the primary scorer because it tests token-level discrimination within sequences, which is harder to game with vague explanations (addresses Detection confirmation bias — see §Confirmation bias mitigation)
  - Detection is supplementary for comparability with Eleuther benchmarks
- **Post-hoc categorization** via Claude (5-way: clinical_concept / clinical_vocabulary / general_language / structural_pattern / noise)
- **Ground-truth concordance** at three thresholds (r > 0.3, 0.4, 0.5) with YES/PARTIAL/NO scoring + rationale
- **Dual-model explanations**: Sonnet for all 1,480 features (production run), Haiku for all 1,480 (cost comparison + quality gap measurement in clinical domain — itself a minor contribution)
- Default LLM: **Claude Sonnet 4** via direct Anthropic API (primary); Claude 3.5 Haiku via Anthropic API (comparison run)
- New module `src/mech_interp_research/auto_interp.py` (~250–350 LOC)
- New modal entrypoint `modal_app/auto_interp.py`
- New config `configs/auto_interp_jumprelu.yaml`

**Out of scope:**

- Vanilla SAE and GemmaScope catalogs (pipeline supports them via different configs; deferred)
- Simulation scoring (requires per-token activation prediction, expensive, adds cost without proportional insight for monosemantic features)
- Intervention scoring (requires SAE+Gemma ablation infrastructure; deferred to causal-mechanism paper)
- SAGE-style agentic refinement (overkill — 94% of grounded features are monospecific at r > 0.5)
- Full SAE catalog (all 18,432 features) — achievable as a follow-up by changing config
- Delphi library integration (see rationale above)
- Web dashboard / Neuronpedia integration

**Out-of-scope reminders (project conventions):**

- No fine-tuning of Gemma
- All large artifacts persist on the `sae-artifacts` Modal volume
- No synthetic test data — use real activations extracted from `./test.csv`

## Confirmation bias mitigation

Detection scoring has a known confirmation bias: the scorer sees the explanation as a hint and can succeed by keyword-matching rather than genuinely understanding the feature. Vague explanations ("medical terminology") can score artificially high.

We address this structurally, not just by acknowledgment:

1. **Fuzzing as primary scorer.** Fuzzing tests whether the scorer can identify *which tokens within a sequence* activated the feature, not just whether the sequence activated. This requires token-level discrimination that vague explanations cannot satisfy.

2. **Specificity control via non-grounded features.** We compute Detection and Fuzzing scores for the 1,000 non-grounded features. If vague explanations ("medical text", "clinical notes") score as high as specific explanations ("atrial fibrillation rhythm terminology"), that is a red flag we report. If scores diverge (specific > vague), that is evidence the scorers discriminate.

3. **Dead-feature noise floor.** The 100 dead/near-dead features provide a floor for scorer calibration. If dead features score above chance, the scorer is broken.

4. **Dual-model cross-validation.** Running both Sonnet and Haiku explanations through the same scorers tests whether explanation quality (not just explanation presence) drives scores.

## Architecture

```
INPUT
├─ JumpReLU SAE checkpoint (existing on volume)
├─ Centered activation shards (existing on volume; 312 shards, ~155M tokens)
├─ Source note texts (sample_50k.csv on raw volume)
├─ Feature inspector output (existing: top_associations.csv, shard_ckpt/)
├─ ICD eval output (existing: correlation_matrices.npz, grounding_summary.json, code_names.json)
└─ Anthropic API key (modal secret)

FEATURE SELECTION (our code)
└─ select_features() → ~1,480 feature indices in 4 tiers

CONTEXT EXTRACTION (leverages existing feature_inspector infrastructure)
└─ For each feature: top-20 activating contexts + 10 non-activating contexts
   (drawn from feature_inspector's shard_ckpt, not re-scanning all 312 shards)

EXPLANATION GENERATION (our code, direct Anthropic SDK)
└─ explain_feature() → one natural-language explanation per feature
   (Sonnet primary, Haiku comparison run)

SCORING (our code, direct Anthropic SDK)
├─ fuzzing_score()   → per-feature token-level accuracy (primary)
└─ detection_score() → per-feature sequence-level accuracy (supplementary)

POST-PROCESSING (our code)
├─ categorize_features() → 5-way category per feature
└─ check_concordance()   → YES/PARTIAL/NO + rationale (grounded + weakly-grounded only)

OUTPUTS (sae-artifacts:auto_interp/<run_id>/)
├─ feature_catalog.csv           (feature_id, explanation, fuzzing_acc, detection_acc,
│                                 category, top_5_tokens, model_used, tier)
├─ concordance_results.csv       (feature_id, r_pb, icd_code, icd_name, explanation,
│                                 concordance, rationale, threshold_tier)
├─ categorization_summary.json   (counts per category, % breakdown, by tier)
├─ concordance_summary.json      (concordance rates at r>0.3, r>0.4, r>0.5;
│                                 YES/PARTIAL/NO distributions; partial-match analysis)
├─ scorer_summary.json           (mean/median/std Fuzzing+Detection by tier;
│                                 specificity control comparison)
├─ model_comparison.json         (Sonnet vs Haiku: explanation length, scorer delta,
│                                 concordance delta, category distribution delta)
├─ per_feature/                  (per-feature JSON: contexts, explanation, scores)
└─ run_summary.json              (config snapshot, runtime, cost, feature counts, errors)
```

## Components

### `src/mech_interp_research/auto_interp.py` (~250–350 LOC)

| Function | Responsibility |
|---|---|
| `select_features(grounding_summary, correlation_matrices, posthoc_summary, seed)` | Returns deduplicated list of feature indices in 4 tiers: `strong_grounded` (r > 0.4, N=280), `weak_grounded` (r = 0.1–0.3, N=100), `non_grounded` (zero BH-significant, N=1000), `dead` (lowest mean activation, N=100). |
| `extract_contexts(shard_ckpt_dir, sae_checkpoint, activations_dir, metadata_path, csv_path, feature_ids, n_pos, n_neg)` | For each feature, extracts top-N activating token contexts and N non-activating contexts. Leverages existing shard_ckpt note vectors for targeted shard selection (same strategy as `feature_inspector.py`). Returns `{feature_id: {pos_contexts: [...], neg_contexts: [...]}}`. Resume-safe via per-feature checkpoint. |
| `explain_feature(client, contexts, model)` | Prompts Claude with top-20 activating contexts. Returns natural-language explanation. |
| `fuzzing_score(client, explanation, test_contexts, model)` | Presents 10 test contexts with one token highlighted per context. Asks "does this token match the described pattern?" Accuracy over 10 binary judgments. Follows Paulo et al. 2024 protocol. |
| `detection_score(client, explanation, pos_contexts, neg_contexts, model)` | Presents 5 activating + 5 non-activating contexts (shuffled). Asks "which activated?" Accuracy over 10 binary classifications. |
| `categorize_feature(client, explanation, top_tokens, model)` | Prompts Claude for one of 5 categories. |
| `check_concordance(client, explanation, icd_code, icd_description, model)` | For grounded/weakly-grounded features only. YES/PARTIAL/NO + one-sentence rationale. |
| `run_auto_interp(config)` | Orchestrator: select → extract → explain → score → categorize → concordance → assemble. Writes all outputs. Emits `run_summary.json`. |

### `modal_app/auto_interp.py` (~50–70 LOC)

Standard project pattern (mirrors `modal_app/tfidf_lr_baseline.py`):

```python
@app.function(
    image=image,         # no extra deps needed — anthropic SDK already in base image
    cpu=4,
    memory=16384,
    timeout=14400,       # 4 hours
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    volumes={"/out": artifacts_volume, "/data": raw_volume},
)
def run_auto_interp_remote(config: dict) -> dict:
    summary = run_auto_interp(**config)
    artifacts_volume.commit()
    return summary

@app.local_entrypoint()
def main(config_file: str) -> None:
    ...
```

No additional image dependencies needed — `anthropic` is already in the base image. No GPU required. This is a CPU-only API-calling job.

### `configs/auto_interp_jumprelu.yaml`

```yaml
# Auto-interpretability for JumpReLU SAE features.
#
# Run:
#   modal run modal_app/auto_interp.py --config-file configs/auto_interp_jumprelu.yaml

# --- paths ---
sae_checkpoint: /out/saes/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/final
activations_dir: /out/activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z_centered
icd_eval_dir: /out/icd_eval/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z
icd_csv_path: /data/sample_50k.csv
output_dir: /out/auto_interp/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z

# --- feature selection ---
n_strong_grounded: 280       # all r > 0.4
n_weak_grounded: 100         # random sample from r = 0.1–0.3 (BH-significant)
n_non_grounded: 1000         # random sample from zero-BH-significant features
n_dead: 100                  # lowest mean activation (noise floor control)
random_seed: 42

# --- explanation ---
explainer_model: claude-sonnet-4-20250514
comparison_model: claude-3-5-haiku-20241022
n_contexts_train: 20         # activating contexts shown to the explainer
n_contexts_test: 10          # held out for scoring (5 pos, 5 neg)
max_tokens_context: 30       # ±tokens around trigger token

# --- scoring ---
scorers:
  - fuzzing       # primary: token-level discrimination
  - detection     # supplementary: sequence-level classification

# --- post-processing ---
categorization_model: claude-sonnet-4-20250514
concordance_model: claude-sonnet-4-20250514
concordance_thresholds:
  - 0.3
  - 0.4
  - 0.5

# --- resume ---
checkpoint_dir: null         # auto-detected from output_dir

logging_level: INFO
```

## Data flow

```
Feature selection
    │
    ├─ strong_grounded (280) ─────────────────────┐
    ├─ weak_grounded (100) ──────────────────────┐ │
    ├─ non_grounded (1000) ─────────────────────┐│ │
    └─ dead (100) ─────────────────────────────┐││ │
                                                ││││
Context extraction (from shard_ckpt + shards)   ││││
    │                                           ││││
    ▼                                           ││││
Explanation generation (Sonnet + Haiku)         ││││
    │                                           ││││
    ├──> Fuzzing scorer ────────────────────────┼┼┼┼──> scorer_summary.json
    ├──> Detection scorer ─────────────────────┼┼┼┤    (by tier, with specificity control)
    │                                           ││││
    ├──> Categorization ───────────────────────┼┼┼┤──> categorization_summary.json
    │                                           ││││   (by tier)
    │                                           ││││
    ├──> Concordance (grounded + weak only) ───┼┼┘┘──> concordance_summary.json
    │                                           ││      (at r>0.3, r>0.4, r>0.5;
    │                                           ││       YES/PARTIAL/NO distributions)
    │                                           ││
    └──> Model comparison (Sonnet vs Haiku) ───┘┘───> model_comparison.json

    └──> feature_catalog.csv (master join of all per-feature results)
```

## Cost & runtime

### API cost (Anthropic direct, no OpenRouter fee)

**Sonnet (claude-sonnet-4):** ~$3/M input, $15/M output.

| Call | Per feature | Notes |
|---|---:|---|
| Explanation generation | ~$0.012 | 20 contexts × ~70 tokens + prompt ≈ 1,800 input; ~200 output |
| Fuzzing scoring | ~$0.005 | 10 test contexts + explanation ≈ 1,200 input; ~100 output |
| Detection scoring | ~$0.005 | 10 contexts + explanation ≈ 1,200 input; ~100 output |
| Categorization | ~$0.002 | explanation + top tokens ≈ 500 input; ~20 output |
| Concordance (grounded only) | ~$0.005 | explanation + ICD info ≈ 600 input; ~150 output |
| **Per-feature subtotal** | **~$0.024** | (without concordance: ~$0.019) |

**Haiku (claude-3.5-haiku):** ~$0.80/M input, $4/M output. ~4x cheaper per feature.

**Totals:**

| Run | Features | Cost |
|---|---:|---:|
| Sonnet primary (all 1,480) | 1,480 | ~$28 |
| Sonnet concordance (380 grounded + weak) | 380 | ~$2 |
| Haiku comparison (all 1,480) | 1,480 | ~$8 |
| Haiku concordance (380) | 380 | ~$0.5 |
| **Total** | | **~$40** |

### Runtime breakdown

| Step | Time |
|---|---|
| Feature selection + context extraction | ~30–45 min |
| Sonnet explanations (1,480 calls @ ~2s each) | ~50 min |
| Haiku explanations (1,480 calls @ ~1s each) | ~25 min |
| Fuzzing scoring × 2 models (2 × 1,480 calls) | ~60–80 min |
| Detection scoring × 2 models (2 × 1,480 calls) | ~60–80 min |
| Categorization × 2 models (2 × 1,480 calls) | ~30 min |
| Concordance × 2 models (2 × 380 calls) | ~15 min |
| Catalog assembly + summaries | <5 min |
| **Total Modal wall time** | **~3–4.5 hours** |

### Implementation time

| Block | Time |
|---|---|
| `select_features` + tests | 30 min |
| `extract_contexts` + tests (leverages feature_inspector) | 2–3 hr |
| `explain_feature` + `fuzzing_score` + `detection_score` + tests | 2–3 hr |
| `categorize_feature` + `check_concordance` + tests | 1.5 hr |
| `run_auto_interp` orchestrator | 1 hr |
| `modal_app/auto_interp.py` entrypoint | 30 min |
| Config + CLAUDE.md update | 15 min |
| Calibration run (10 features, inspect quality) | 1–2 hr |
| **Total** | **~10–12 hr ≈ 1.5 days** |

## Testing strategy

Following project conventions (real data from `./test.csv`; no synthetic fixtures):

1. **`tests/test_auto_interp.py`** — unit tests:
   - `test_select_features` — verifies 4-tier sampling correctness (deterministic given seed); verifies deduplication; verifies dead-feature selection by mean activation
   - `test_extract_contexts` — runs on 1 shard from `./test.csv`; verifies pos/neg context extraction with expected shapes
   - `test_explain_feature` — mocks anthropic client; verifies prompt formatting includes all 20 contexts
   - `test_fuzzing_score` — mocks anthropic client; verifies binary parsing, accuracy calculation
   - `test_detection_score` — mocks anthropic client; verifies shuffled presentation, accuracy calculation
   - `test_categorize_feature` — mocks anthropic client; verifies 5-way parsing
   - `test_check_concordance` — mocks anthropic client; verifies YES/PARTIAL/NO parsing + rationale extraction
   - `test_assemble_catalog` — verifies CSV columns, row count, tier labels

2. **Integration smoke** (`test_run_auto_interp_smoke`):
   - End-to-end with 4 features (1 per tier), mocked anthropic client
   - Verifies all expected output files are written
   - Verifies concordance only runs for grounded/weak tiers

3. **Modal smoke** (manual; one-off):
   - 10 features (3 strong grounded, 2 weak grounded, 3 non-grounded, 2 dead)
   - Real Anthropic API calls (Sonnet)
   - Inspect explanation quality, scoring sanity, concordance parsing
   - Verify per-feature checkpoints enable resume

Total: 8 unit + 1 integration test; fits in 1.5–2 hr of test-writing time.

## Failure modes & recovery

| Failure | Recovery |
|---|---|
| Modal 4-hour timeout | All steps are per-feature checkpointed (`per_feature/<id>.json`). Re-dispatch resumes. |
| Anthropic rate limit / 429 | Built-in retry with exponential backoff. Default RPM is conservative. |
| Anthropic outage | Per-feature checkpoint files. Resume continues from last completed feature. |
| Malformed LLM response (non-YES/NO scoring, etc.) | Per-feature graceful degradation: log warning, set score to NaN. `run_summary.json` reports `n_features_with_parsing_errors`. Retry once with explicit format reminder in prompt. |
| Concordance: ICD code description missing | Use raw ICD code string; prompt is robust to bare codes. |
| Context extraction produces <20 activating contexts | Proceed with available contexts; log warning. Set minimum threshold at 5 (below which explanation is unreliable — mark as `insufficient_contexts`). |
| Dead features have zero activations across all shards | Expected for some. Mark as `no_activation` with explanation "Feature never fires." Score as NaN (noise floor anchor). |

## Integration with the paper

| Where | What changes |
|---|---|
| **§4.6** (qualitative case studies) | Top-grounded latents now have auto-generated explanations alongside ICD correlation values. The "3 case studies" expand to "top 280 grounded features tabulated in supplementary; 3–5 highlighted in narrative with concordance results." |
| **New §4.7** (auto-interpretability and concordance) | (a) Concordance analysis: rates at r > 0.3 / 0.4 / 0.5 with YES/PARTIAL/NO distributions and partial-match taxonomy; (b) categorization summary across all tiers; (c) Fuzzing/Detection score distributions by tier (with specificity control demonstrating scorer validity); (d) Sonnet vs Haiku quality comparison in clinical domain; (e) weakly-grounded discoveries and non-grounded clinical fraction. |
| **§4.8 limitations** | Auto-interp is JumpReLU-only; concordance limited to 46 ICD-9 codes (richer ontologies deferred); Sonnet's medical knowledge is a bottleneck for explanation quality. |
| **Supplementary** | Full `feature_catalog.csv` (1,480 features × 2 models). `concordance_results.csv` with rationales. |

## What success looks like

**Scorer quality:**
- Mean Fuzzing accuracy ≥ 0.65 for strong-grounded features (Sonnet)
- Mean Detection accuracy ≥ 0.75 for strong-grounded features (Sonnet)
- Dead features: both scores near chance (~0.50)
- Non-grounded features: scores lower than strong-grounded (specificity control)
- Sonnet scores > Haiku scores by a measurable margin

**Categorization:**
- Grounded features: majority classified as `clinical_concept` or `clinical_vocabulary`
- Non-grounded features: majority classified as `general_language` or `structural_pattern`
- Dead features: majority classified as `noise`
- If >15% of non-grounded features are `clinical_concept`, that is a finding (SAE learned clinical structure beyond ICD codes)

**Concordance:**
- Overall concordance (YES + PARTIAL) ≥ 80% at r > 0.5 (N=144)
- Overall concordance ≥ 70% at r > 0.4 (N=280)
- YES rate (exact match) ≥ 50% at r > 0.5
- PARTIAL matches produce an interpretable taxonomy (sub-concept, related-concept, vocabulary-proxy)
- Weakly-grounded features (r = 0.1–0.3): concordance drops to 40–60% — the gradient is the finding

**Model comparison:**
- Sonnet concordance rate > Haiku concordance rate by ≥ 5 percentage points
- Sonnet produces longer, more specific explanations (mean token count)
- Sonnet Fuzzing scores > Haiku Fuzzing scores

---

## Appendix A: Prompts

### Explanation prompt

```
Below are the top 20 text contexts where a sparse autoencoder feature
activates most strongly in clinical discharge summaries. The trigger
token in each context is marked with **asterisks**.

<CONTEXTS>

What pattern causes this feature to activate? Describe the specific
concept, linguistic pattern, or structural element that these contexts
share. Be as specific as possible — name clinical concepts, drugs,
or procedures if applicable. One to two sentences.
```

### Fuzzing scoring prompt

```
A sparse autoencoder feature has been described as:
"<EXPLANATION>"

For each of the following tokens in context, predict whether
this token would activate the feature (YES or NO).

<TOKEN_CONTEXTS>

Respond with one line per context: the context number and YES or NO.
```

### Detection scoring prompt

```
A sparse autoencoder feature has been described as:
"<EXPLANATION>"

Below are 10 text contexts from clinical discharge summaries.
Some activated this feature and some did not.

<SHUFFLED_CONTEXTS>

For each context, predict whether it activated the feature (YES or NO).
Respond with one line per context: the context number and YES or NO.
```

### Categorization prompt

```
A sparse autoencoder feature has been described as:
"<EXPLANATION>"

Top activating tokens include: <top_5_tokens>

Categorize this feature into exactly one of:
- clinical_concept (a specific medical diagnosis, condition, or finding)
- clinical_vocabulary (drug names, dosages, units, lab values, procedures)
- general_language (POS, syntax, common words, generic patterns)
- structural_pattern (formatting, section headers, signatures, templates)
- noise (no coherent pattern)

Respond with just the category name.
```

### Ground-truth concordance prompt

```
A sparse autoencoder feature has been auto-interpreted as:
"<EXPLANATION>"

This feature also has the strongest statistical correlation (point-biserial
r = <R_PB>) with the ICD-9 diagnosis code: <ICD_CODE> (<ICD_CODE_DESCRIPTION>).

Does the auto-interp explanation describe the same concept as the ICD code?
Respond with:
- YES — the explanation clearly describes the ICD concept
- PARTIAL — the explanation overlaps but describes a related, broader,
  or narrower concept (e.g., a drug used to treat the condition, a symptom
  of the condition, or a broader category containing the condition)
- NO — the explanation describes something unrelated

Format: <verdict> | <one-sentence rationale>
Example: PARTIAL | The explanation describes warfarin dosing, which is a
treatment for the correlated ICD code (atrial fibrillation), not the
condition itself.
```

## Appendix B: Why Sonnet (not Opus) for explanations

The literature is clear that explanation quality plateaus past Sonnet-class models for SAE auto-interp. The bottleneck is context quality (which activating examples you show the model), not model capability. For monosemantic features — which 94% of our grounded features are — the first-pass explanation from Sonnet is sufficient. Opus would cost ~4x more ($12/M input, $60/M output vs $3/$15) with marginal quality gains on features that already map cleanly to single concepts.

The interesting question is not "does Opus do better than Sonnet?" but "does Sonnet do better than Haiku in a clinical domain?" — because Haiku is the practical choice for full-catalog runs (all 18,432 features). The Sonnet-vs-Haiku comparison quantifies the quality gap, directly informing whether a future full-catalog run should use Haiku (cheap, ~$30) or Sonnet (better, ~$100).

## Appendix C: Literature positioning

**Key citations:**

| Paper | Relationship to our work |
|---|---|
| Paulo et al. 2024 (arXiv 2410.13928) | Methodology basis. We follow their Detection + Fuzzing protocols. |
| Bills et al. 2023 | Original explain-then-score paradigm. We extend to clinical domain. |
| Templeton et al. 2024 | Scaling monosemanticity. Our SAE is smaller but domain-specific. |
| O'Neill et al. 2025 (arXiv 2508.09363) | Closest prior work: JumpReLU SAEs on Gemma-2 for clinical QA. They did not validate against structured labels. |
| Wu et al. 2025 (DILA, arXiv 2409.10504) | ICD-code-connected dictionary learning, but jointly trained (not post-hoc). |
| MedSAE (arXiv 2510.26411) | SAEs on MedCLIP with CheXpert label validation; vision domain, not text. |

**Our novel contribution:** Post-hoc SAE features trained on frozen LLM activations from clinical text, with auto-interp explanations systematically validated against structured ICD-9 labels. No existing paper does this. The three-way convergence — statistical grounding (point-biserial), token-level evidence (feature inspector), semantic concordance (auto-interp vs ICD) — is the complete story.

**Addressing the random-feature interpretability concern:** Recent work has shown SAEs trained on randomly-weighted transformers produce features with auto-interp scores comparable to trained networks. Our concordance validation directly addresses this: high auto-interp scores alone are not evidence of genuine interpretability, but high concordance between auto-interp explanations and *external* structured medical labels is.

---

End of design (v3 — Custom Build).
