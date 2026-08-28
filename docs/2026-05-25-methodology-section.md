# 3 Methodology

Our study has two parts. We first build the experimental apparatus (§3.1–3.3): a corpus of clinical activations, two domain-specific SAEs trained on those activations, a general-domain pretrained SAE for comparison, and three baselines. We then apply three feature-interpretation procedures (§3.4–3.6): statistical grounding against diagnosis labels, semantic concordance between unsupervised feature explanations and those labels, and causal ablation. Grounding is computed for every SAE, which lets us compare domain-specific and general-domain dictionaries. Concordance and ablation are demonstrated on the domain-specific SAEs, with the general-domain SAE included where the comparison is informative.

## Part 1: Experimental Setup

### 3.1 Data and Activation Extraction

We use discharge summaries from MIMIC-IV (Johnson et al., 2023a) and its note module MIMIC-IV-Note (Johnson et al., 2023b), a de-identified EHR database accessible under PhysioNet credentialing. Each summary is paired with the ICD-9 codes assigned to its admission, which serve as our external labels. Because MIMIC-IV mixes ICD-9 and ICD-10, we restrict to ICD-9-coded admissions for a single consistent label space. We draw 50,000 summaries from the training and validation splits, never the test split, stratified by diagnosis rarity: notes are grouped into four prevalence strata, and the sampling budget is weighted toward the rarer strata. The sample is therefore enriched for uncommon diagnoses rather than matched to population prevalence.

We extract residual-stream activations at layer 16 of Gemma-2-2B (Gemma Team, 2024). Layer 16 is roughly 62% of the way through this 26-layer model, in the mid-depth residual stream where abstract semantic features are expected rather than surface form. The 50,000 notes yield about 15.2 million tokens at a maximum sequence length of 8192, and we mean-center the activations before training. A held-out set of 4,911 notes, about 10% of the corpus, is reserved for the ablation analysis, a grounding robustness check, and checkpoint selection for the vanilla SAE (§3.2). The appendix details the extraction precision, the centering procedure, and the automated data-quality checks.

### 3.2 SAE Models

A sparse autoencoder encodes an activation $x\in\mathbb{R}^{d_{\text{model}}}$ into a sparse code $z$ and decodes a reconstruction $\hat{x}$, minimizing $\mathcal{L}=\lVert x-\hat{x}\rVert_2^2+\lambda\,\mathcal{S}(z)$. We train two domain-specific SAEs on the centered activations, differing chiefly in their sparsity mechanism, and compare both against a general-domain SAE.

The **JumpReLU SAE** (Rajamanoharan et al., 2024) computes the code $z=\pi\odot H(\pi-\theta)$ from the pre-activation $\pi=W_{\text{enc}}(x-b_{\text{dec}})+b_{\text{enc}}$. Here $H$ is the Heaviside step and $\theta\in\mathbb{R}_{+}^{d_{\text{sae}}}$ is a per-feature threshold. Sparsity uses an $L_0$ penalty, and gradients pass through the threshold via straight-through estimators. The **vanilla ReLU SAE** (Bricken et al., 2023; Cunningham et al., 2023) uses $z=\text{ReLU}(\pi)$ with an $L_1$ penalty, and periodically resamples dead latents following Bricken et al. (2023). Both SAEs use 18,432 latents, an eightfold expansion, constrain decoder columns to unit norm by gradient projection followed by renormalization (Bricken et al., 2023), and optimize with Adam using $\beta_1=0$ to limit dead features (Gao et al., 2024). We select the best checkpoint by reconstruction explained variance, and the vanilla SAE uses the held-out set for selection and early stopping. The appendix tabulates the hyperparameters and training schedules (Table 1).

The **GemmaScope SAE** (Lieberum et al., 2024) is our general-domain baseline, a pretrained JumpReLU SAE for Gemma-2. We use the layer-16 SAE of width 16,384, with nominal average $L_0$ near 42, and apply it zero-shot, without fine-tuning or recentering. Because this SAE is applied out of distribution, we diagnose the mismatch with three explained-variance variants: the SAE as published, its codes decoded with the clinical mean replacing the decoder bias, and the SAE on fully recentered activations. The first two variants separate a decoder-bias shift from a feature-direction mismatch. We also report the fraction of pre-activations above threshold and the cosine between the decoder bias and the mean clinical activation.

For all three SAEs we report reconstruction fidelity on the held-out activations: explained variance, the mean number of active latents per token ($L_0$), and the dead-latent fraction.

### 3.3 Comparison Baselines

We compare the SAE features against three baselines, all sharing the SAE evaluation's notes, pooling, and data split for a fair comparison.

The **lexical** baseline tests whether the features merely track keyword presence. For each code we correlate a per-note keyword-presence indicator, built from a curated dictionary, with the label, add a partial correlation that residualizes on note length, and measure keyword-absent recall: the rate at which the most associated feature still fires on label-positive notes that contain no keyword.

To ask whether a full-vocabulary classifier outperforms the features, the **TF-IDF** baseline fits per-code logistic regression on 10,000 unigram and bigram features with sublinear term frequency, under stratified 5-fold cross-validation, scored by AUC-ROC and AUC-PR. The same classifier is fit on the pooled SAE features; a paired Wilcoxon signed-rank test (Wilcoxon, 1945) over the per-code AUC differences compares the two, and a supplementary analysis contrasts the strongest single feature of each type.

The **raw-activation** baseline isolates the value added by the sparse encoding, running the same cross-validated classifier on the max-pooled dense layer-16 activations. Construction details for all three are given in the appendix.

## Part 2: Validation Methodology

### 3.4 Statistical Grounding

We summarize each note by pooling its SAE activations across tokens into a per-note feature vector, a step that is standard for downstream probing (Karvonen et al., 2025). We use element-wise max-pooling. Max-pooling couples feature magnitude with note length, a confound we control for below. We retain the 46 ICD-9 codes with prevalence above 2%. For each feature and code we compute the point-biserial correlation (Tate, 1954) between the pooled activation and the binary label,

$$r_{pb}=\frac{M_1-M_0}{s}\sqrt{\frac{n_1 n_0}{N^2}},$$

where $M_1$ and $M_0$ are the mean pooled activations on the $n_1$ notes that carry the code and the $n_0$ notes that do not, $N=n_1+n_0$, and $s$ is the standard deviation of the activation over all notes. This is the Pearson correlation between the activation and a dichotomous variable, and we obtain two-tailed $p$-values from the associated $t$-statistic. Because we test the full family of $d_{\text{sae}}\times 46$ associations, we apply the Benjamini–Hochberg procedure at $q=0.05$ (Benjamini and Hochberg, 1995). A feature is grounded if it has a significant association whose $|r_{pb}|$ exceeds a threshold, which we sweep from 0.1 to 0.7. Its monospecificity is the number of such codes, classified as monospecific, oligospecific, or polyspecific. We apply this procedure identically to every SAE, so the domain-specific and general-domain dictionaries are compared on equal footing.

To control the length confound, we residualize each pooled activation on note token count by ordinary least squares, recompute the correlation, and re-apply the correction. The threshold sweep, the monospecificity counts, and the partial correlation are computed post hoc from the same correlations. We also recompute grounding on the held-out set as a robustness check. The appendix also inspects grounded features at the token level, reporting their top-activating tokens, their firing rates on diagnosis-positive versus diagnosis-negative notes, their lexical diversity, and a case study of the most strongly associated feature.

### 3.5 Semantic Concordance

Concordance asks whether two independently produced signals name the same clinical concept: an unsupervised explanation of a feature, and that feature's statistical association with a diagnosis. We apply it to the JumpReLU SAE.

**Explanations and scoring.** Following the Delphi framework (Paulo et al., 2024), an explainer LLM, Claude Sonnet, writes a natural-language explanation of each feature from its top-activating token contexts together with non-activating contexts. Each context is a 30-token window on either side of the marked triggering token. The same model assigns each feature to one of five categories: clinical concept, clinical vocabulary, structural pattern, general language, or noise. We analyze 964 features in four tiers by grounding status: 280 strongly grounded ($|r_{pb}|>0.4$), 100 weakly grounded ($0.1<|r_{pb}|\le 0.3$), 484 non-grounded, and 100 dead. We score explanations with two methods from Paulo et al. (2024): detection, in which a scorer LLM identifies which whole contexts activate the feature given the explanation, and fuzzing, its token-level analogue. We treat these scores as non-discriminative by design, since auto-interpretability scores do not separate trained from randomly initialized transformers (Heap et al., 2025). As a control on the scorers, we re-score each feature against a deliberately mismatched explanation, formed by deranging explanations globally and within tiers. A Wilcoxon signed-rank test then checks whether a feature's true explanation outscores the mismatched one. The random-explanation chance level for these scorers is about 0.51 (Paulo et al., 2024).

**Concordance validation.** Consider a grounded feature with its strongest significant code $c$ and its explanation $e$. A judge LLM rates whether $e$ is semantically related to the natural-language description of $c$, taken from the ICD-9 reference, and returns YES, PARTIAL, NO, or UNKNOWN. The concordance rate is the fraction of YES and PARTIAL verdicts, reported at several $|r_{pb}|$ thresholds. The two signals are produced independently, since the explanation comes from activation contexts alone and the association comes from labels alone. This independence is what distinguishes concordance from scorer accuracy. An explanation can predict a feature's activations without identifying a unique concept, a failure known as descriptive collision (McCann, 2026), so concordance instead tests whether the explanation's semantic content matches an external label. Prompts are given in the appendix, and a single LLM is used throughout for explanation, scoring, and concordance.

### 3.6 Causal Ablation

To test whether grounded features are causally relevant rather than merely correlated, we zero-ablate features in the vanilla domain-specific SAE and, for comparison, in the general-domain GemmaScope SAE, on the held-out set. At layer 16 we record the cross-entropy loss on a clean pass ($\ell_{\text{clean}}$), after splicing in the SAE reconstruction ($\ell_{\text{recon}}$), and after subtracting a feature's contribution $z_j W_{\text{dec},j}$ before decoding ($\ell_{\text{abl}}$). Losses are measured over a window covering each note's final 25% of non-padding tokens. The ablation effect of a feature is $\ell_{\text{abl}}-\ell_{\text{recon}}$, and the reconstruction tax is $\ell_{\text{recon}}-\ell_{\text{clean}}$. The targets are grounded features, selected by $|r_{pb}|$ and filtered for monospecificity and firing density, together with random and low-$|r_{pb}|$ control features; the appendix lists them. For each feature we compare ablation effects on label-positive against label-negative notes with a one-sided Mann–Whitney $U$ test (Mann and Whitney, 1947). We report Cliff's $\delta$ (Cliff, 1993) as the effect size, binned by magnitude following Romano et al. (2006), and apply Benjamini–Hochberg correction at $q=0.05$.

### 3.7 Reproducibility

Training and evaluation ran on cloud GPUs with a single random seed. Diagnosis labels enter only at the grounding stage, never during SAE training or selection. We release our code; MIMIC-IV is obtained through PhysioNet credentialing, and no note text or protected health information appears in this paper or its artifacts. The appendix reports the full hyperparameters, schedules, prompts, package versions, and compute.

---

## Citations referenced in this section

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *J. R. Stat. Soc. B*, 57(1), 289–300.
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread.*
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494–509.
- Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.
- Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
- Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.
- Heap, T., Lawson, T., Farnik, L., & Aitchison, L. (2025). Sparse Autoencoders Can Interpret Randomly Initialized Transformers. arXiv:2501.17727.
- Johnson, A. E. W., et al. (2023a). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10(1).
- Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023b). MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). *PhysioNet.*
- Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. arXiv:2503.09532.
- Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.
- Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other. *Ann. Math. Stat.*, 18(1), 50–60.
- McCann, J. F. (2026). Descriptive Collision in Sparse Autoencoder Auto-Interpretability. arXiv:2605.12874.
- Paulo, G., Mallen, A., Juang, C., & Belrose, N. (2024). Automatically Interpreting Millions of Features in Large Language Models. arXiv:2410.13928.
- Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). Appropriate statistics for ordinal level data. *Annual Meeting of the Florida Association of Institutional Research.*
- Tate, R. F. (1954). Correlation Between a Discrete and a Continuous Variable. Point-Biserial Correlation. *Ann. Math. Stat.*, 25(3), 603–607.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83.

---

# Appendix

Values shown are filled from verified result files; entries marked *[source: …]* are to be populated from the named artifact. No note text or PHI appears here.

## A. Data and Activation Extraction

**Table A1 — Corpus statistics**

| Quantity | Value |
|---|---|
| Notes | 50,000 |
| Tokens (total) | 15,172,037 |
| Shards | 312 |
| Max sequence length | 8,192 |
| ICD-9 codes (prevalence > 2%) | 46 |
| Held-out evaluation notes | 4,911 (≈10%) |

**A.1 Sampling.** Drawn from train + val splits only (test never accessed); notes shorter than 500 characters dropped. Stratified into four strata by the frequency of each note's rarest active top-50 ICD-9 code (quartile bins); per-stratum budget 35 / 30 / 25 / 10 % (very-rare → common); deficit carried forward. Seed 42.

**A.2 Extraction and centering.** Post-block residual stream at layer 16 (`hidden_states[17]`); fp32 compute, fp16 storage; tokenizer truncation with special tokens. Mean-centering is two-pass: a float64 accumulator over all tokens, a float32 saved mean, fp16 centered output. Post-extraction checks: shape, finiteness, non-zero, inter-note diversity.

## B. SAE Configuration and Training (Table 1)

**Table B1 — SAE configuration**

| | JumpReLU (domain) | Vanilla (domain) | GemmaScope (general) |
|---|---|---|---|
| Status | trained | trained | pretrained, zero-shot |
| d_in / d_sae / expansion | 2304 / 18,432 / 8× | 2304 / 18,432 / 8× | 2304 / 16,384 / — |
| Activation | JumpReLU | ReLU | JumpReLU |
| Sparsity penalty | L0, λ=10 | L1, l1=10 | (pretrained) |
| Bandwidth ε | 1.0 | — | — |
| log_threshold_init | −2.0 | — | — |
| Learning rate | 2e-4 | 2e-4 | — |
| Adam β | (0.0, 0.999) | (0.0, 0.999) | — |
| LR warmup steps | 2,000 | 2,000 | — |
| λ_L0 warmup steps | 5,000 | — | — |
| Epochs | 3 | 3 (early-stopped) | — |
| Batch (tokens) | 4,096 | 4,096 | — |
| Dead-latent handling | self-correcting θ | resample @5,000 (thr 1e-6) | — |
| Decoder constraint | unit-norm | unit-norm | (as released) |
| Checkpoint selection | reconstruction EV | EV on 31 held-out shards | — |
| Nominal avg L0 | — | — | 42 |

**Table B2 — Reconstruction fidelity (held-out)**

| | JumpReLU | Vanilla | GemmaScope |
|---|---|---|---|
| Explained variance | 0.899 | 0.889 | −4.21 |
| Mean L0 | *[eval_scan]* | *[train_summary]* | 50.2 |
| Dead-latent fraction | *[diagnostic_metrics]* | *[diagnostic_metrics]* | 0.028 |

## C. GemmaScope Domain-Shift Diagnostics

**Table C1** *(source: `domain_shift_analysis.json`)*

| Metric | Value |
|---|---|
| EV — as published | −4.20 |
| EV — clinical mean for decoder bias | −4.20 |
| EV — fully recentered | −6.564 |
| Fraction of pre-activations above threshold | 0.711 |
| cosine(decoder bias, clinical mean) | 0.90 |

Interpretation: re-centering does not recover EV, so the failure is feature-direction mismatch, not a mean shift.

## D. Baselines

**D.1 Lexical.** Keyword dictionary `icd9_keywords_improved.yaml` (64 codes). Matching: word-boundary regex for keywords ≤3 chars, case-insensitive substring otherwise. Length control: partial point-biserial residualized on token count. Keyword-absent recall: among label-positive notes containing no keyword, fraction on which the best-associated SAE feature fires.

**D.2 TF-IDF + LR.** Features: 10,000, unigram+bigram, sublinear TF. Classifier: logistic regression (saga, L2, max_iter 5,000). CV: stratified 5-fold, seed 42. Metrics: AUC-ROC, AUC-PR. Plus best-single-feature point-biserial comparison.

**Table D2 — Paired Wilcoxon signed-rank, SAE vs TF-IDF (n=46 codes)** *(source: `tfidf_lr_summary.json`)*

| Metric | Mean TF-IDF | Mean SAE | Median Δ (SAE−TF-IDF) | p-value | Significant (q=0.05) |
|---|---|---|---|---|---|
| AUC-ROC | 0.917 | 0.881 | −0.033 | 3.5e-9 | yes |
| AUC-PR | 0.590 | 0.532 | −0.059 | 6.2e-8 | yes |

**D.3 Raw-activation probe.** Max-pooled centered layer-16 activations; identical per-code stratified 5-fold LR protocol; compared head-to-head against the SAE-feature classifier.

## E. Statistical Grounding — extended

- **Table E1 — Threshold sweep:** grounded-latent count at |r| ∈ {0.1,…,0.7}, raw and partial, × 3 SAEs. *[source: `posthoc_summary.json`]*
- **Table E2 — Monospecificity:** fraction monospecific / oligospecific / polyspecific by threshold. *[source: `posthoc_summary.json`]*
- **Table E3 — Per-code grounding:** top-3 features per ICD code. *[source: `top_associations.csv`]*

## F. Feature Inspection

- **Table F1:** per top grounded latent — feature_idx, top code, |r_pb|, top-k tokens, pos/neg firing-rate ratio, diversity score. *[source: `feature_inspection_report.json`]*
- **Case study:** the single most strongly associated feature, shown with example token strings only (no note text).

## G. Auto-Interpretability and Concordance

**G.1 Tier sizes (planned vs analyzed)**

| Tier | Planned | Analyzed |
|---|---|---|
| Strongly grounded (\|r\|>0.4) | 280 | 280 |
| Weakly grounded (0.1<\|r\|≤0.3) | 100 | 100 |
| Non-grounded | 1,000 | 484 |
| Dead | 100 | 100 |
| **Total** | **1,480** | **964** |

The non-grounded tier is under-filled because the auto-interp pipeline was rate/time-limited.

**G.2 Prompts.** Verbatim templates for explanation+categorization, detection, fuzzing, and concordance judging. Explainer/judge: Claude Sonnet; 20 explanation contexts, 10 scoring contexts; trigger token marked. *[source: `auto_interp.py` prompt templates]*

- **Table G3 — Concordance by threshold:** YES / PARTIAL / NO / UNKNOWN counts and rate at |r| > {all, 0.3, 0.5}. *[source: `concordance_summary.json`]*
- **Table G4 — Categorization by tier:** five-category fractions per tier. *[source: `categorization_summary.json`]*
- **Table G5 — Scorer summary by tier:** mean/median/std fuzzing and detection. *[source: `scorer_summary.json`]*

**Table G6 — Shuffled-explanation control (fuzzing, global)** *(source: `shuffled_control_summary.json`; detection + within-tier + per-tier rows follow the same structure in the file)*

| Tier | Mean real | Mean shuffled | Δ | 95% CI (shuffled) | Wilcoxon p |
|---|---|---|---|---|---|
| Overall | 0.932 | 0.496 | 0.436 | [0.489, 0.502] | ≈0 |
| Strong grounded | 0.932 | 0.500 | 0.432 | [0.493, 0.507] | ≈0 |
| Weak grounded | 0.935 | 0.489 | 0.445 | [0.473, 0.506] | ≈0 |
| Dead | 0.923 | 0.490 | 0.434 | [0.469, 0.510] | ≈0 |

Chance reference ≈ 0.51 (Paulo et al., 2024).

## H. Causal Ablation

- **Table H1 — Target features:** per target — feature_idx, ICD code, kind (grounded / random_control / low_r_control), r_pb. *[source: `ablation_summary.json` → `config.targets`]*
- **Table H2 — Per-feature results:** ablation effect, Cliff's δ (+ magnitude bin), Mann–Whitney p (BH-adjusted), reconstruction tax — per SAE. *[source: `ablation_results.csv`]*

## I. Compute and Reproducibility

- **Compute:** GPU type per stage and GPU-hours (e.g., vanilla training ≈ 24,113 s; ICD-eval ≈ 9 h on the 50k run); auto-interp API token/cost. *[source: run logs / `train_summary.json`]*
- **Hyperparameter calibration:** σ measured via `measure_sigma`; L0 targeted into range before the full run.
- **Package versions:** torch, transformers, scikit-learn, scipy, safetensors, anthropic SDK. *[source: environment lockfile]*
- **Availability:** code released; MIMIC-IV obtained via PhysioNet credentialing (not redistributable).
