# Critical Publication Assessment: Clinical Mechanistic Interpretability via Sparse Autoencoders

**Date**: 2026-05-25 (revised)
**Target venue**: EMNLP 2026 (or ACL Rolling Review)
**Verdict**: Solid borderline-to-accept. The concordance validation is genuinely novel and the experimental breadth is impressive, but the paper requires careful framing to survive reviewer scrutiny given concurrent critical work on SAE validity.

---

## 1. What Was Done: Complete Experiment Inventory

### 1.1 Activation Extraction
- **Model**: Gemma-2-2B, layer 16 residual stream (d_model=2304)
- **Corpus**: 50,000 MIMIC-IV discharge summaries (stratified sample)
- **Output**: 312 shards of fp16 safetensors, mean-subtracted (float64 accumulator)

### 1.2 SAE Training

| SAE | Architecture | Key Hyperparameters | Latents | L0 (target) |
|-----|-------------|---------------------|---------|-------------|
| **JumpReLU** | L0 + STE | lambda_l0=10, bandwidth=0.5, log_threshold_init=2.3 | 18,432 | ~35 |
| **Vanilla** | ReLU + L1 | l1_coeff=10, resample_steps=5000, early_stop_patience=3 | 18,432 | [20-80] |
| **GemmaScope** | Pre-trained (Google) | width_16k, average_l0_42 | 16,384 | 42 |

Both custom SAEs trained for 3 epochs on centered activations with lr=0.0002, adam_beta1=0.0, expansion_factor=8.

### 1.3 ICD-9 Clinical Grounding Evaluation

Point-biserial correlation between note-level SAE activations (max-pooled) and 46 binary ICD-9 diagnosis indicators, with BH FDR correction.

| Metric | JumpReLU | Vanilla | GemmaScope |
|--------|----------|---------|------------|
| Top |r| | 0.864 (afib) | 0.853 (hypothyroid) | 0.574 (V5867) |
| Grounded at r>0.1 | 9,023 (48.9%) | 8,293 (45.0%) | 5,749 (35.1%) |
| Grounded at r>0.3 | 610 (3.3%) | 673 (3.7%) | 48 (0.3%) |
| Grounded at r>0.5 | 144 (0.78%) | 142 (0.77%) | 4 (0.02%) |
| Grounded at r>0.7 | 29 (0.16%) | — | — |

**Key finding**: Custom SAEs produce correlations ~50% stronger than GemmaScope at the top end (0.864 vs 0.574). The gap widens dramatically at higher thresholds: at r>0.3, custom SAEs have 12-14x more grounded latents than GemmaScope.

### 1.4 Post-Hoc Analyses (JumpReLU)

#### Monospecificity Gradient

| Threshold | Grounded | Monospecific | Mono% | Mean codes/latent |
|-----------|----------|-------------|-------|-------------------|
| r>0.1 | 9,023 | 2,719 | 30.1% | 3.79 |
| r>0.2 | 2,043 | 1,271 | 62.2% | 1.72 |
| r>0.3 | 610 | 479 | 78.5% | 1.31 |
| r>0.4 | 280 | 246 | 87.9% | 1.13 |
| r>0.5 | 144 | 135 | 93.8% | 1.06 |
| r>0.6 | 60 | 60 | 100% | 1.00 |
| r>0.7 | 29 | 29 | 100% | 1.00 |

**Key finding**: Perfect monospecificity (1 code per latent) emerges at r>0.6. This gradient is a strong signal that SAE features are encoding discrete clinical concepts, not diffuse multi-code patterns.

#### Partial Correlation (controlling for note length / n_tokens confound)

| Metric | Before | After |
|--------|--------|-------|
| Grounded at r>0.1 | 9,023 | 5,147 (43% drop) |
| Top |r| | 0.864 | 0.853 (1.3% drop) |
| Mean max |r| | 0.1166 | 0.0912 (22% drop) |

**Interpretation**: About 43% of weakly grounded latents lose significance after controlling for note length, confirming the max-pooling length confound. But the strongest associations (top-10) barely change (0.86 -> 0.85), meaning the core clinical features are robust. The confound is real but doesn't invalidate the primary findings.

#### GemmaScope Partial Correlation (for contrast)
- Grounded at r>0.1 drops from 5,749 to 894 (84% reduction)
- Mean max |r| drops 0.082 -> 0.044 (46% reduction)
- GemmaScope's grounding is largely a length confound artifact. Custom SAEs resist this much better.

### 1.5 Lexical Keyword Baseline

| Metric | JumpReLU vs Lexical | Vanilla vs Lexical | GemmaScope vs Lexical |
|--------|-------|---------|------------|
| SAE above lexical | 45/46 | 44/46 | 25/46 |
| Comparable | 1/46 | 2/46 | 9/46 |
| Lexical above SAE | 0/46 | 0/46 | 12/46 |
| Mean delta-r | 0.287 | 0.308 | 0.034 |

The sole comparable code for JumpReLU is tobacco use (icd9_3051), which is essentially a keyword detection task. Custom SAEs dominate the lexical baseline. GemmaScope does not.

**Keyword-absent recall**: SAE features fire on ICD-positive notes even when no keyword appears in the text, confirming the SAE captures beyond-surface-level representations.

### 1.6 TF-IDF + Logistic Regression Baseline (Vanilla + JumpReLU)

| Metric | TF-IDF | Vanilla SAE | JumpReLU SAE |
|--------|--------|-------------|--------------|
| Mean AUC-ROC | **0.917** | 0.881 | 0.888 |
| Mean AUC-PR | **0.590** | 0.532 | 0.543 |
| TF-IDF wins (codes) | — | 33/46 | 30/46 |
| SAE best-r > TF-IDF best-r | — | 23/46 | 21/46 |
| Mean best r (SAE / TF-IDF) | 0.519 | 0.579 | 0.566 |

**Critical nuance**: TF-IDF wins on classification AUC because it has 10,000 features (full vocabulary) vs SAE's unsupervised single-feature correlation. This is expected and not damaging: the SAE's best individual feature often produces a stronger univariate correlation than TF-IDF's best individual feature (23/46 codes for vanilla, 21/46 for JumpReLU). The SAE compresses 50k-token documents into 18k sparse features; it shouldn't beat a supervised classifier on per-code AUC. JumpReLU slightly narrows the gap vs TF-IDF relative to vanilla (Wilcoxon median Δ AUC-ROC: −0.027 vs −0.033), consistent with its marginally stronger grounding profile.

**GemmaScope**: TF-IDF baseline was abandoned for GemmaScope because the SAE-features LR did not converge (71% of features fire above threshold per token on clinical data, violating the sparsity assumption).

### 1.7 Feature Inspection

Token-level evidence extraction for top SAE-ICD associations. Two-pass algorithm scans activation shards for top-k tokens per grounded latent, re-tokenizes matched notes for context extraction, computes firing statistics and diversity metrics.

Results confirm that grounded latents fire on clinically specific tokens (e.g., "levothyroxine" for hypothyroidism, "warfarin" for atrial fibrillation) rather than generic medical vocabulary.

### 1.8 Auto-Interpretability Pipeline (Sonnet)

**964 features processed** out of 1,480 selected (280 strong_grounded r>0.4, 100 weak_grounded r=0.1-0.3, 1,000 non_grounded, 100 dead). All scored with Claude Sonnet (claude-sonnet-4-6).

#### Categorization Results

| Category | Strong Grounded | Weak Grounded | Dead | Non-Grounded |
|----------|----------------|--------------|------|-------------|
| clinical_concept | 132 (47.1%) | 14 (14.0%) | 4 (4.0%) | 0 |
| clinical_vocabulary | 120 (42.9%) | 39 (39.0%) | 20 (20.0%) | 0 |
| structural_pattern | 26 (9.3%) | 35 (35.0%) | 75 (75.0%) | 1 (0.2%) |
| general_language | 2 (0.7%) | 10 (10.0%) | 1 (1.0%) | 0 |
| noise | 0 | 0 | 0 | 483 (99.8%) |

**Key finding**: 90% of strong_grounded features are classified as clinical (concept + vocabulary). 99.8% of non_grounded features are noise. Dead features are predominantly structural patterns (75%) — they fire on coherent boilerplate/template text, not random noise.

#### Scorer Results (Fuzzing + Detection)

| Tier | N (Fuzzing) | Mean Fuzzing | N (Detection) | Mean Detection |
|------|-------------|-------------|----------------|----------------|
| Global | 432 | 0.938 | 459 | 0.962 |
| strong_grounded | 268 | 0.938 | 275 | 0.960 |
| weak_grounded | 94 | 0.941 | 90 | 0.955 |
| dead | 69 | 0.932 | 93 | 0.975 |
| non_grounded | 1 | 1.000 | 1 | 1.000 |

**Scorer ceiling effect**: Mean fuzzing is 0.93+ across all tiers including dead. No tier separation exists in scorer accuracy. Root causes documented in `docs/2026-05-24-scorer-ceiling-fix.md`: ~80% single-token features make discrimination trivial; "dead" features fire on coherent structural patterns. The shuffled-explanation control (below) was run to test this directly.

#### Shuffled-Explanation Control — scorer null baseline (NEW, 2026-05-25)

Tests whether the Fuzzing/Detection scorers measure explanation *quality* or merely exploit surface cues: each feature's held-out contexts are re-scored against a **deliberately wrong** explanation (only the explanation string changes; same contexts, same distractor seed). Two schemes — **global** derangement (reproduces Paulo et al. 2024's random-explanation baseline, chance ≈0.51) and **within-tier** permutation (controls for explanation specificity).

| Scorer | scheme | real | shuffled | Δ | Wilcoxon p | n |
|--------|--------|------|----------|----|-----------|----|
| Fuzzing | global | 0.932 | 0.496 | +0.436 | <1e-6 | 274 |
| Fuzzing | within-tier | 0.931 | 0.493 | +0.438 | <1e-6 | 278 |
| Detection | global | 0.961 | 0.522 | +0.439 | <1e-6 | 285 |
| Detection | within-tier | 0.957 | 0.516 | +0.441 | <1e-6 | 270 |

**Key finding**: shuffled explanations collapse to **chance (~0.50)** — reproducing Paulo et al.'s 0.51 baseline — while real explanations stay at 0.93–0.96, a ~0.44 gap (p≈0) that holds across **every tier, including dead**. So the scorers *do* read the explanation; the ceiling reflects genuine, verifiable explanations (monospecificity), not a vacuous metric. This converts the scorer ceiling (W4) from a liability into a data point and directly answers the Heap et al. and McCann critiques. A built-in cross-check confirms the control's real baseline (fuzzing 0.931, detection 0.960) matches the published scorer means (0.938 / 0.962) within ≤0.007, so the comparison is against the reported numbers. *Coverage: 343/481 eligible features scored (paired n 270–285) before the run hit an Anthropic credit limit; the result is decisive on the matched subset, with full coverage pending a credit top-up + resume.*

#### Concordance Validation (the novel contribution)

| Threshold | Total | YES | PARTIAL | NO | UNKNOWN | Concordance Rate (YES+PARTIAL) | Exact Match (YES) |
|-----------|-------|-----|---------|----|---------|---------------------------------|-------------------|
| All (r>0.1) | 380 | 85 | 238 | 46 | 11 | 85.0% | 22.4% |
| r>0.3 | 280 | 85 | 180 | 7 | 8 | 94.6% | 30.4% |
| r>0.5 | 144 | 61 | 81 | 1 | 1 | 98.6% | 42.4% |

*Note: r>0.3 and r>0.4 are identical (280 features each) because the strong_grounded tier boundary aligns at r>0.4 and no additional features fall in the 0.3–0.4 gap in this concordance subset. The r>0.4 row is omitted to avoid redundancy.*

**The concordance gradient is the paper's strongest result.** As the ICD correlation threshold rises:
- Concordance rate increases: 85% -> 94.6% -> 98.6%
- Exact match rate increases: 22.4% -> 30.4% -> 42.4%
- NO verdicts nearly vanish: 46 -> 7 -> 1

At r>0.5, only 1 out of 144 features has a NO concordance verdict. This means the LLM's unsupervised explanation of what the feature does almost perfectly aligns with the ICD code the feature statistically correlates with — and this alignment gets stronger as the statistical association gets stronger.

**By tier breakdown**:
- strong_grounded (r>0.4): YES=85, PARTIAL=180, NO=7, UNKNOWN=8 (concordance=94.6%)
- weak_grounded (r=0.1-0.3): YES=0, PARTIAL=58, NO=39, UNKNOWN=3 (concordance=58.0%)

The weak_grounded tier has zero YES verdicts and 39% NO — exactly what you'd expect if the LLM explanations are reflecting something real. Weaker statistical associations produce weaker semantic alignment.

### 1.9 Causal Ablation Studies

Feature ablation experiments measuring the causal effect of zeroing individual SAE features on model next-token-prediction loss. Implemented as: zero the feature's sparse code -> decode to modified activations -> patch back into forward pass -> measure CRPS/loss change. Effect sizes measured via Cliff's delta (Mann-Whitney) comparing loss changes on ICD-positive vs ICD-negative notes.

#### Vanilla SAE Ablation (Pilot Extended: 20 features, 4,911 held-out notes)

| Feature | ICD Code | r_pb | Cliff's d | Magnitude | Sig (q<0.05) |
|---------|----------|------|-----------|-----------|-------------|
| 3537 | hypothyroid (2449) | 0.711 | 0.654 | **large** | Yes |
| 11016 | ESRD (5856) | 0.807 | 0.503 | **large** | Yes |
| 1387 | afib (42731) | 0.779 | 0.427 | medium | Yes |
| 7304 | hypothyroid (2449) | 0.819 | 0.376 | medium | Yes |
| 17588 | hypothyroid (2449) | 0.766 | 0.335 | medium | Yes |
| 16170 | CHF (4280) | 0.768 | 0.211 | small | Yes |
| 11387 | ESRD (5856) | 0.736 | 0.239 | small | Yes |
| 1042 | BPH (60000) | 0.733 | 0.218 | small | Yes |
| 17076 | ESRD (5856) | 0.707 | 0.162 | small | Yes |
| 15117 | GERD (53081) | 0.755 | -0.060 | negligible | No |

**15/20 features significant** at q<0.05. Median Cliff's delta for grounded features: 0.118. Two features show large effect sizes (d>0.5). 5 features are non-significant, suggesting that not all statistically grounded features have causal effects on model predictions.

**Key control**: In the pilot run (with explicit control features), median Cliff's delta for grounded features was 0.300 vs -0.036 for controls (random/low-r features). This demonstrates that grounded features have systematically larger causal effects than ungrounded controls.

**Reconstruction tax**: Mean loss increase from SAE reconstruction alone: 0.029 nats (1.8% of base loss 1.635). The SAE introduces minimal distortion.

#### GemmaScope Ablation (Pilot Extended: 20 features, 4,911 notes)

| Feature | ICD Code | r_pb | Cliff's d | Magnitude | Sig (q<0.05) |
|---------|----------|------|-----------|-----------|-------------|
| 11234 | AKI (5849) | 0.363 | 0.471 | medium | Yes |
| 2907 | anticoag (V5867) | 0.389 | 0.462 | medium | Yes |
| 5093 | UTI (5990) | 0.380 | 0.441 | medium | Yes |
| 2513 | depression (311) | 0.343 | 0.347 | medium | Yes |
| 8281 | hyperchol (2724) | 0.369 | 0.242 | small | Yes |
| 12120 | CAD (41401) | 0.344 | 0.241 | small | Yes |
| 16233 | CHF (4280) | 0.423 | 0.210 | small | Yes |
| 12184 | gout (2749) | 0.412 | 0.218 | small | Yes |

**15/20 features significant** (same rate as custom SAE). Median Cliff's delta for grounded features: 0.169. GemmaScope features show medium causal effects despite weaker correlations (max r=0.42 vs 0.82 for Vanilla).

**Critical difference**: GemmaScope reconstruction tax is 0.648 nats (39.6% of base loss) — **22x larger** than the custom SAE (0.029). The GemmaScope SAE introduces massive reconstruction error, meaning the ablation is partially measuring the effect of removing information from an already-degraded representation.

**Pilot with controls**: Grounded median Cliff's d = 0.195 vs controls = -0.013. Same pattern as custom SAE: grounded features are causally relevant, controls are not.

#### Ablation Summary

| SAE | Features | Sig Rate | Median Cliff's d (grounded) | Median Cliff's d (controls) | Recon Tax |
|-----|----------|----------|----------------------------|-----------------------------|-----------|
| Vanilla (pilot) | 12 | 83% | 0.300 | -0.036 | 0.029 |
| Vanilla (extended) | 20 | 75% | 0.118 | — | 0.029 |
| GemmaScope (pilot) | 12 | 58% | 0.195 | -0.013 | 0.648 |
| GemmaScope (extended) | 20 | 75% | 0.169 | — | 0.648 |

**Missing**: No JumpReLU ablation was run. This is a gap — the JumpReLU SAE is the primary model in the paper, so ablation results for it would strengthen the narrative. The Vanilla ablation serves as a proxy since the two architectures produce near-identical grounding profiles.

### 1.10 Test-Split Evaluation (Held-Out Validation)

The `test_split_eval.py` module recomputes the full ICD-9 grounding pipeline on only the SAE-training held-out shards (shards 281-311, ~4,911 notes). This addresses the potential objection that grounding correlations were computed on the same data the SAE was trained on.

| Metric | JumpReLU (test) | JumpReLU (full) | Vanilla (test) | Vanilla (full) | GemmaScope (test) | GemmaScope (full) |
|--------|----------------|-----------------|----------------|----------------|-------------------|-------------------|
| Top |r| | 0.864 | 0.864 | 0.860 | 0.853 | 0.545 | 0.574 |
| Grounded r>0.1 | 9,721 (52.7%) | 9,023 (48.9%) | 8,985 (48.8%) | 8,293 (45.0%) | 5,790 (35.3%) | 5,749 (35.1%) |
| Grounded r>0.3 | 610 (3.3%) | 610 (3.3%) | 675 (3.7%) | 673 (3.7%) | 54 (0.3%) | 48 (0.3%) |
| Grounded r>0.5 | 147 (0.8%) | 144 (0.8%) | 143 (0.8%) | 142 (0.8%) | 4 (0.02%) | 4 (0.02%) |

**Key finding**: Test-split grounding is virtually identical to full-corpus grounding for the custom SAEs. The top correlations are stable on the test split (0.864 for JumpReLU on both splits). This eliminates the overfitting concern: the SAE features' clinical associations generalize to unseen data.

Partial correlation on test split also holds: JumpReLU top |r| drops only to 0.855 (from 0.864), matching the full-corpus pattern.

### 1.11 Raw Activation LR Baseline (Code exists, not yet run)

`raw_lr_baseline.py` implements a third baseline: logistic regression on raw centered Gemma-2-2B layer-16 activations (2,304-dim, max-pooled to note level), without any SAE. This tests whether the SAE decomposition adds value over the raw representation for ICD code prediction. The code exists with configs for all three SAE comparators, but no results are on Modal yet.

---

## 2. Literature Context (2025-2026)

### 2.1 Directly Competing / Closely Related Work

**O'Neill et al. (2025) — "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders"** (arXiv 2508.09363)
- Trains JumpReLU SAEs on Gemma-2 layer-20 activations using 195k clinical QA examples
- Shows domain-confined SAEs explain up to 20% more variance than general-domain SAEs
- Finds features align with clinically meaningful concepts
- **Overlap with this project**: Same model family (Gemma-2), same architecture (JumpReLU), clinical domain. But uses clinical QA text, not EHR discharge notes, and does not do ICD grounding or concordance validation.
- **Differentiation**: This project's ICD-9 grounding with point-biserial correlation against structured diagnosis labels is a substantially different and arguably more rigorous validation approach than automated/human interpretability scoring alone.

**Sainsbury et al. (2026) — "Sparse Autoencoder Decomposition of Clinical Sequence Model Representations"** (arXiv 2605.04072)
- TopK SAEs on FlatASCEND (14.5M param clinical sequence model) using MIMIC-IV
- Shows progressive abstraction across depth (layer-0 singleton detectors -> layer-6 multi-category features)
- SAE features outperform dense representations for mortality prediction
- **Overlap**: MIMIC-IV data, SAEs, clinical application. But FlatASCEND is a clinical sequence model on coded events, not a general-purpose LLM on free text.
- **Differentiation**: This project works on a general-purpose LLM (Gemma-2-2B) processing natural language, testing whether representations trained on internet-scale data encode clinical structure. Fundamentally different question.

### 2.2 Critical SAE Methodology Papers

**Leask et al. (ICLR 2025) — "Sparse Autoencoders Do Not Find Canonical Units of Analysis"**
- Shows SAE features are neither complete (smaller SAEs miss information) nor atomic (meta-SAE decomposition reveals substructure)
- **Impact on this paper**: Need to be careful not to claim SAE features are "the" decomposition. Frame as "a useful decomposition" that captures clinically relevant structure, acknowledging non-uniqueness.

**Korznikov et al. (2026) — "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?"** (arXiv 2602.14111)
- Random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72)
- On a synthetic setup with known ground-truth features, SAEs recovered only **9% of true features** despite achieving 71% explained variance — directly challenges the assumption that good reconstruction implies good feature recovery
- **Impact on this paper**: This is the most dangerous paper for the narrative. However, it evaluates SAEs on *general-domain* interpretability metrics. This project's ICD grounding provides an *external ground truth* that random baselines could not match. The point-biserial correlations (r=0.864) against structured diagnosis labels are a much harder bar than interpretability scoring. This actually strengthens the case for domain-specific grounding as a validation methodology. The 9% recovery result also motivates our concordance validation: even if the SAE only recovers a fraction of the true feature space, the features it *does* find align with real clinical concepts.

**Heap et al. (2025) — "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers"** (arXiv 2501.17727)
- SAEs trained on randomly initialised transformers produce auto-interp scores and reconstruction metrics similar to those from trained models
- Recommends treating common SAE metrics as useful but insufficient proxies; argues for routine randomised baselines and measures of feature "abstractness"
- **Impact on this paper**: Directly relevant to our scorer ceiling (§1.8). Our fuzzing/detection scores (0.93+) cannot distinguish clinical from non-clinical features, consistent with Heap et al.'s finding that these metrics lack discriminative power. This makes our concordance validation — which *does* discriminate (85% → 99% gradient) — even more important as a validation modality. Heap et al. should be cited when discussing scorer limitations.

**Ma et al. (2026) — "Falsifying Sparse Autoencoder Reasoning Features in Language Models"** (arXiv 2601.05679)
- 45-90% of contrastively selected "reasoning" features activate after injecting only a few associated tokens into non-reasoning text
- Proposes falsification-based evaluation framework
- **Impact**: Underscores the importance of this project's multi-modal validation (correlation + concordance + ablation), not just interpretability scoring.

### 2.3 Auto-Interpretability Advances

**Paulo et al. / Delphi (Eleuther AI, 2024)**: Established the fuzzing + detection scoring paradigm used in this project (arXiv 2410.13928). Open-source, widely adopted. However, manual evaluation of the first 50 features from a Gemma-2 SAE found that 38% of Delphi explanations fail in characterizable ways — a known limitation of the methodology.

**McCann (2026) — "Descriptive Collision in Sparse Autoencoder Auto-Interpretability"** (arXiv 2605.12874)
- Identifies "descriptive collision": many distinct SAE features admit the same natural-language explanation
- This is complementary to polysemanticity (one feature, many meanings) — here it's many features, one explanation
- **Impact**: Directly relevant to our concordance methodology. Descriptive collision means that high fuzzing/detection scores don't guarantee explanation specificity. Our concordance validation partially addresses this by checking whether the explanation matches the *specific* ICD code, not just whether it predicts activation. However, we should acknowledge this limitation.

**Anthropic (2025) — Circuit Tracing**: Introduced cross-layer transcoders as a new SAE variant. Suggests the field is moving beyond residual-stream SAEs toward more structured decompositions.

**EMNLP 2025 Survey — "A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models"** (ACL Anthology, Findings of EMNLP 2025)
- Comprehensive survey of SAE methods, architectures, and evaluation approaches. Useful for positioning our contribution in the context of the broader SAE evaluation literature.

### 2.4 Domain-Specific SAE Applications

**EEG Foundation Models (2026)** (arXiv 2605.13930): SAEs applied to EEG models, showing domain-specific features emerge.

**ASR Models (2026)** (arXiv 2605.12225): SAEs for speech recognition model interpretability.

**Radiology LLMs (2025)** (arXiv 2507.12950): SAEs applied to radiology-specialized multimodal LLMs.

The trend is clear: domain-specific SAE application is a growing subfield in 2025-2026, and this project fits squarely within it.

---

## 3. Critical Analysis

### 3.1 Strengths

**S1: External ground-truth validation.** The ICD-9 grounding is the project's most defensible contribution. Unlike most SAE interpretability work that relies on human judgment or LLM-based scoring (both subjective), this project validates features against structured diagnosis labels. In the context of Korznikov et al.'s critique that random baselines match SAEs on interpretability metrics, having point-biserial correlations of r=0.864 against external labels is a much harder test. Random feature directions would not produce r=0.864 against specific ICD codes.

**S2: Concordance gradient.** The monotonic increase in concordance rate (85% -> 95% -> 99%) as correlation strength increases is the strongest evidence that SAE features encode semantically meaningful clinical concepts. This is a novel validation methodology that hasn't appeared in prior work.

**S3: Comprehensive baseline battery.** Lexical (45/46 SAE wins), TF-IDF (SAE wins on best-feature correlation 23/46 despite losing on classification AUC), GemmaScope (dramatically worse grounding and near-complete collapse under partial correlation) — the baselines comprehensively establish that custom SAE grounding is non-trivial.

**S4: Confound analysis.** The partial correlation analysis honestly shows that ~43% of weak associations are confounded by note length, but the strongest associations survive nearly intact (0.86 -> 0.85). The GemmaScope contrast (84% collapse) makes the custom SAE robustness more compelling.

**S5: Causal evidence.** The ablation results, while limited to vanilla SAE and a small feature set, demonstrate that grounded features are causally relevant to model predictions (median Cliff's d = 0.30 for grounded vs -0.04 for controls). This addresses the "mere correlation" objection. The ablation infrastructure (1,297 lines in `ablation.py`) is well-engineered with per-shard checkpointing, BH FDR correction, and proper controls.

**S6: Monospecificity gradient.** The progression from 30% monospecific (r>0.1) to 100% monospecific (r>0.6) is a clean demonstration that strongly grounded features encode discrete clinical concepts rather than diffuse multi-condition patterns.

**S7: Test-split validation.** Grounding results hold identically on held-out shards never used for SAE training (top |r| 0.864 on both full and test splits for JumpReLU). This eliminates overfitting concerns and is stronger than most SAE papers, which don't validate on held-out data.

### 3.2 Weaknesses

**W1: Single model, single layer.** Only Gemma-2-2B, only layer 16. No evidence that findings generalize to other models (e.g., Llama-3, Mistral) or other layers. The layer choice is unexplained in the available materials.

**W2: Max-pooling confound partially addressed.** While partial correlation shows the strongest features survive, the 43% drop at r>0.1 is substantial. Reviewers may ask why mean-pooling wasn't used instead (or in addition). The answer is in the docs (GemmaScope must use the same pooling for fair comparison), but this needs clear justification.

**W3: No JumpReLU ablation.** The JumpReLU SAE is the primary model, but ablation was only run for the vanilla SAE. While the two have similar grounding profiles, this is a gap. A reviewer could argue that JumpReLU features might have different causal properties.

**W4: Scorer ceiling effect — now contextualized by the shuffled-explanation control.** Raw fuzzing/detection scores are uniform (0.93+ across all tiers including dead), so the *absolute* scores don't discriminate. The shuffled-explanation control (§1.8) resolves this: real explanations beat deliberately-wrong ones by ~0.44 (shuffled → chance ~0.50, p≈0, across all tiers), so the scorer *does* measure explanation quality and the ceiling reflects monospecificity. Residual: full coverage (currently 343/481) pending a credit top-up + resume.

**W5: Auto-interp run incomplete.** Only 964/1,480 features were processed. The comparison model (Haiku) was never run (comparison_model: null in config). The Sonnet-vs-Haiku comparison described in the design doc did not happen.

**W6: Small ablation sample.** 20 features is a limited sample for causal claims. The pilot had only 12 (10 grounded + 2 controls). Statistical power for per-feature significance is adequate given 4,911 notes, but the feature selection may not be representative.

**W7: No ICD-10 or SNOMED grounding.** Using ICD-9 codes (a 40-year-old coding system) when MIMIC-IV supports ICD-10 may raise reviewer questions about clinical relevance. (Counter: MIMIC-IV's discharge summaries historically use ICD-9; this is a data constraint.)

### 3.3 Gaps That Could Be Addressed Before Submission

| Gap | Effort | Impact |
|-----|--------|--------|
| ~~Run shuffled-explanation control~~ ✓ **Done (2026-05-25)** — real≫shuffled, Δ≈0.44, p≈0 (§1.8); 343/481 covered, rest pending credits | — | High — scorer validation rescued |
| Run JumpReLU ablation | ~4-8 hrs on Modal | High — ablation for the primary model; vanilla proxy is not sufficient for reviewers |
| Run raw activation LR baseline | ~4-8 hrs on Modal | Medium — tests whether SAE decomposition adds value over raw representations |
| Complete auto-interp (remaining 516 features) | ~2 hrs, ~$20 | Low — 964 is already sufficient |

### 3.4 Potential Reviewer Objections and Responses

**Q: "Korznikov et al. show SAE features don't beat random baselines. Why should we trust your findings?"**
A: Korznikov et al. evaluate on interpretability scoring, sparse probing, and causal editing — all internal metrics. Our validation uses external structured labels (ICD-9 codes) that provide independent ground truth. Random feature directions would not produce point-biserial correlations of r=0.864 with specific diagnosis codes. The ICD grounding methodology is precisely the kind of external validation the field needs to move beyond the Korznikov critique.

**Q: "Leask et al. show SAE features aren't canonical. What does your decomposition actually mean?"**
A: We don't claim our features are canonical or unique. We claim they are *useful*: they correlate with external clinical labels, their LLM explanations align with those labels (concordance), and ablating them causally affects model predictions on relevant notes. Whether an alternative decomposition might capture the same structure differently is orthogonal to the finding that *this* decomposition encodes clinically meaningful information.

**Q: "TF-IDF beats your SAE on classification AUC. Why use SAEs at all?"**
A: TF-IDF is a supervised 10,000-feature classifier; the SAE provides unsupervised single-feature correlations. The comparison tests whether SAE grounding can be trivially explained by surface text features. It cannot: (1) the best SAE feature often has a stronger univariate correlation than the best TF-IDF feature (23/46 codes); (2) SAE features fire on keyword-absent positive notes; (3) SAE features survive partial correlation for note length while GemmaScope features collapse. The SAE captures representation structure beyond surface text patterns.

**Q: "Your scorer accuracy is at ceiling. Doesn't this invalidate auto-interpretability as a validation method?"**
A: The ceiling is consistent with Heap et al. (2025), who show that auto-interp metrics fail to distinguish trained from random transformers, and McCann (2026), who identifies descriptive collision as a structural limitation. We treat scorer accuracy as a necessary-but-insufficient sanity check and rely on the concordance gradient (85% → 99%) as the discriminative validation. Unlike scorer accuracy, concordance tests whether the *content* of the explanation matches an external label, not just whether it predicts activation. The shuffled-explanation control (§1.8) now demonstrates explanation specificity directly: real explanations beat wrong ones by ~0.44 (shuffled at chance ~0.50, p≈0), so the scorer is specific to the explanation rather than exploiting surface cues.

**Q: "You only study one model and one layer. How do we know this generalizes?"**
A: Fair limitation. We frame this as a demonstration that clinical concept grounding *can* emerge in general-purpose LLM representations, validated through external labels. Generalization across models/layers is future work, and we identify it as such.

---

## 4. Novelty Assessment

### 4.1 What's Genuinely Novel

1. **ICD-code grounding of SAE features**: No prior work validates SAE features against structured medical diagnosis labels via point-biserial correlation. O'Neill uses interpretability scoring; Sainsbury uses mortality prediction probing; neither does code-level grounding.

2. **Concordance validation methodology**: Checking whether an LLM's unsupervised explanation of a feature's function aligns with the structured label the feature statistically correlates with — and showing this alignment scales with correlation strength — is a new validation paradigm. This could be cited by future work as a general-purpose SAE validation technique whenever external labels are available.

3. **Partial correlation confound analysis**: While not methodologically novel (OLS residualization is standard), applying it to SAE grounding evaluation and showing differential robustness between custom SAEs and pre-trained baselines is a useful contribution.

4. **Cross-baseline validation at clinical scale**: The combination of lexical, TF-IDF, and GemmaScope baselines evaluated on the same 50k-note corpus with the same ICD labels is unusually thorough for SAE work.

### 4.2 What's Incremental

1. **Training JumpReLU SAEs on clinical text**: O'Neill (2025) already trained JumpReLU SAEs on Gemma-2 for clinical text. The architecture and training methodology are established.

2. **Auto-interpretability with fuzzing/detection**: Standard application of Paulo et al. (2024) / Delphi methodology. The ceiling effect is a known limitation.

3. **Feature inspection / token-level evidence**: Standard exploratory analysis.

### 4.3 What's Missing for a Strong Contribution

1. **Causal story connecting features to model behavior**: The ablation results show features affect loss, but don't demonstrate a mechanistic circuit (e.g., "this feature fires on 'warfarin' -> attention head X routes it to 'INR' prediction -> affects discharge instruction generation"). Circuit-level analysis would elevate this from "SAE features correlate with diagnoses" to "here's how the model uses clinical knowledge."

2. **Downstream utility demonstration**: No evidence that SAE features improve any downstream task (diagnosis prediction, note generation quality, bias detection). The paper is purely about representation analysis.

3. **Multi-model comparison**: Testing whether the same clinical features emerge in Llama-3, Mistral, or larger Gemma models would dramatically strengthen generalization claims.

---

## 5. Framing Recommendations

### 5.1 Recommended Paper Title

"Clinical Ground Truth for Sparse Autoencoder Features: Validating Mechanistic Interpretability Against Structured Diagnosis Labels in Medical Language Models"

Alternative: "Do Language Models Learn Clinical Concepts? Structured Validation of Sparse Autoencoder Features via ICD-9 Code Grounding"

### 5.2 Central Claim

> We demonstrate that sparse autoencoder features trained on a general-purpose language model's representations of clinical text encode structured clinical concepts, validated through three independent modalities: statistical grounding against ICD-9 diagnosis labels (r_pb up to 0.864), semantic concordance between unsupervised LLM explanations and structured labels (98.6% at r>0.5), and causal ablation showing differential loss effects on diagnosis-positive notes.

### 5.3 Paper Structure

1. **Introduction**: Frame around the validation gap in SAE interpretability (Korznikov critique, Chanin non-canonicity). Argue for external ground-truth validation. Clinical text + ICD codes provide a natural setting.

2. **Related Work**: Position against O'Neill (domain-specific SAEs, no external grounding), Sainsbury (coded sequences, not free text), Korznikov/Leask et al./Heap et al./Ma et al. (SAE critique papers — this work offers a partial response through external validation).

3. **Method**:
   - SAE training (JumpReLU on Gemma-2-2B layer-16, centered activations)
   - ICD-9 grounding (point-biserial + BH FDR, max-pooling with partial-correlation control)
   - Auto-interpretability + concordance validation
   - Causal ablation protocol

4. **Results** (organized by claim, not by experiment):
   - §4.1: SAE features correlate with diagnosis codes (grounding results, threshold sweep)
   - §4.2: Correlations reflect genuine clinical structure, not surface confounds (partial correlation, lexical baseline, TF-IDF baseline, keyword-absent recall)
   - §4.3: SAE features are monospecific clinical concepts (monospecificity gradient)
   - §4.4: LLM explanations align with structured labels (concordance gradient) — the main novel result
   - §4.5: Features are causally relevant (ablation results)
   - §4.6: Custom SAEs dramatically outperform pre-trained baseline (GemmaScope comparison)

5. **Discussion**: Position concordance validation as a general methodology. Discuss scorer ceiling honestly. Acknowledge limitations (single model/layer, max-pooling confound, no downstream task).

6. **Conclusion**: Clinical text provides a uniquely suitable domain for SAE validation because structured labels exist. The concordance methodology generalizes beyond clinical NLP.

### 5.4 What to Lead With

Lead with the **concordance gradient**, not the grounding correlations. The grounding correlation (r=0.864) is a strong result but could be seen as "just correlation." The concordance gradient — showing that unsupervised LLM explanations increasingly align with structured labels as the statistical association strengthens — is the methodological contribution that makes this paper publishable rather than just a nice empirical observation.

### 5.5 What to Downplay

- **Scorer accuracy**: Report in a table but don't make it a centerpiece. Frame the ceiling as evidence of monospecificity, not as a validation failure.
- **TF-IDF classification AUC loss**: Don't apologize for it. Frame it correctly as "supervised classifier with full vocabulary vs unsupervised single-feature correlation" and move on.
- **GemmaScope comparison**: Use it as a baseline, not a punching bag. GemmaScope wasn't trained for clinical text, so it's not surprising it underperforms. The value is in showing the gap size, not in claiming superiority.

---

## 6. Publication Chances

### 6.1 Venue Fit

| Venue | Fit | Notes |
|-------|-----|-------|
| **EMNLP 2026** | Strong | Interpretability track, clinical NLP welcome |
| **ACL ARR** | Strong | NLP + interpretability |
| **NeurIPS 2026** | Moderate | Would need stronger ML methodology contribution |
| **CHIL 2026** | Strong | Clinical ML focus, interpretability-friendly |
| **ML4H Workshop** | Safe fallback | Clinical ML workshop at NeurIPS |

### 6.2 Assessment

**As-is (without filling gaps)**: Borderline accept at EMNLP. The concordance gradient is novel and compelling, but reviewers will flag the missing JumpReLU ablation and single model/layer limitation. (The scorer-ceiling concern is now addressed by the shuffled-explanation control, §1.8.) The SAE critique landscape has intensified since early 2025 (Korznikov, Heap, Ma, McCann) — a skeptical reviewer has more ammunition. Likely gets mixed reviews: one enthusiastic, one skeptical about SAE validity citing Korznikov/Heap, one asking for more models.

**With gaps filled (✓ shuffled control done; JumpReLU ablation remaining)**: Solid accept at EMNLP. The shuffled control — now run (§1.8) — converts the scorer ceiling from a weakness into a data point and directly addresses the Heap et al. and McCann critiques (real beats wrong explanations by ~0.44, shuffled at chance). JumpReLU ablation remains the most obvious open experimental gap for the primary model.

**With circuit analysis or downstream task**: Clear accept at ACL/EMNLP, competitive at NeurIPS. But this would require significant additional work (months, not days).

**Note on timing**: The EMNLP 2026 ARR submission deadline is May 25, 2026 — the shuffled control is done; the remaining gap-filling work (JumpReLU ablation ~8h, plus finishing the control's last 138 features once Anthropic credits are restored) may need to target a later ARR cycle if it can't be completed today.

### 6.3 Risk Factors

1. **Korznikov/Heap-inspired rejection**: A reviewer who takes "SAEs don't beat random baselines" (Korznikov) or "auto-interp can't distinguish trained from random" (Heap) as gospel may reject regardless of external validation. The 9% ground-truth recovery rate (Korznikov) is particularly damaging to the SAE paradigm. Mitigation: address this directly in related work, emphasize that external labels are a fundamentally different evaluation, and note that even partial feature recovery is useful if the recovered features are clinically grounded.

2. **"Just correlation" dismissal**: A reviewer may argue that ICD-code correlation is not mechanistic interpretability. Mitigation: the concordance gradient + ablation results move beyond correlation.

3. **PHI concerns**: MIMIC-IV is de-identified but reviewers in clinical ML are sensitive. Make sure no example text appears in the paper. All examples should be structural (feature fires on token X in context Y) without quoting actual discharge notes.

4. **Scope mismatch**: If framed as primarily a clinical NLP paper, interpretability reviewers may find it thin on methodology. If framed as primarily interpretability, clinical reviewers may find it thin on clinical utility. Frame as interpretability methodology validated in a clinical setting.

5. **Descriptive collision objection**: A reviewer aware of McCann (2026) may argue that high concordance merely reflects descriptive collision — the LLM gives the same explanation to many features, and that explanation happens to match the ICD code. Mitigation: the concordance *gradient* (85% → 99%) would not exist if all features received the same boilerplate explanation. Also, the weak_grounded tier's 58% concordance with 0% exact match shows the methodology is discriminative.

### 6.4 Bottom Line

This is a **solid piece of work** with a **genuine methodological contribution** (concordance validation) embedded in a **thorough experimental evaluation**. The main risk is not the quality of the work but the framing: it needs to be positioned as a response to the SAE validity critique (Korznikov, Chanin) through external ground-truth validation, not as "we trained an SAE on clinical text."

The concordance gradient is publishable. The question is whether the surrounding experiments are sufficient to pass reviewer scrutiny at a top venue. With the shuffled control (now done) and the JumpReLU ablation, I believe they are.

**Estimated probability of acceptance at EMNLP 2026**:
- As-is: 30-40% (slightly lower than a year ago due to intensified SAE critique landscape) — this baseline now *includes* the completed shuffled-explanation control (§1.8); remaining lift is from the items below
- With JumpReLU ablation added: 50-60%
- With above + raw-activation LR head-to-head + Haiku comparison: 60-70%

---

## 7. Recommended Next Steps (Priority Order)

1. ~~Run shuffled-explanation control~~ — ✓ **Done (2026-05-25)**: real≫shuffled, Δ≈0.44, shuffled at chance (~0.50, reproducing Paulo et al.), p≈0 across all tiers; scorer accuracy is now a publishable specificity metric (§1.8), addressing the Heap et al./McCann critique. *Coverage 343/481 — finish with a resume once Anthropic credits are topped up.*
2. **Run JumpReLU ablation** (~4-8 hrs on Modal) — ablation for the primary model; currently only vanilla has ablation results, and reviewers will notice
3. **Run raw activation LR head-to-head** (~4-8 hrs on Modal × 3 configs) — tests whether SAE decomposition outperforms raw representations on classification; code already exists, just needs `modal run` (see results.md §16.1)
4. **Write paper draft** — structure per §5.3 above; all core experiments are complete and TF-IDF baseline is done for both custom SAEs

Items 1-3 can run in parallel on Modal. Note: EMNLP 2026 ARR deadline is May 25, 2026.
