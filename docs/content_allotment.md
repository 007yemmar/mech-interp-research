# Content Allotment: EMNLP 2026 Submission

**Working title:** "Clinical Ground Truth for Sparse Autoencoder Features: Validating Mechanistic Interpretability Against Structured Diagnosis Labels"

**Format:** EMNLP long paper — 8 pages content + unlimited references/appendix
**Page budget:** ~8 pages ≈ ~6,400 words of body text + figures/tables

This document specifies what concepts, claims, data, and figures belong in each section. The guiding principle: the concordance validation methodology is the paper's novel contribution. Everything else — grounding, baselines, ablation — exists to make the concordance result credible. Allocate space accordingly.

---

## Abstract (~200 words, ~0.25 pages)

### Purpose
Hook the reader with the validation gap, state the method, report headline numbers, declare the contribution.

### Must contain
- **Problem statement (1–2 sentences):** Sparse autoencoders are widely used for mechanistic interpretability, but recent work (Korznikov et al. 2026; Heap et al. 2025) shows that standard evaluation metrics (auto-interp scores, reconstruction fidelity, sparse probing) fail to distinguish trained SAE features from random baselines. The field lacks external ground-truth validation.
- **Our approach (1–2 sentences):** We train JumpReLU SAEs on Gemma-2-2B representations of 50,000 MIMIC-IV clinical notes and validate the learned features against structured ICD-9 diagnosis labels through three independent modalities.
- **Headline results (3–4 sentences):**
  - Point-biserial correlation up to r=0.864 between individual SAE features and specific diagnosis codes (46 ICD-9 codes, BH-FDR corrected)
  - Concordance gradient: LLM-generated feature explanations align with the statistically associated ICD code at 85% (all features) → 94.6% (r>0.3) → 98.6% (r>0.5)
  - Causal ablation: grounded features produce differential loss effects on diagnosis-positive notes (median Cliff's δ=0.30 vs −0.04 for controls)
  - Custom domain-trained SAEs dramatically outperform a published general-domain SAE (GemmaScope) on all metrics
- **Contribution declaration (1 sentence):** We introduce concordance validation — checking whether unsupervised feature explanations align with external structured labels — as a general methodology for SAE evaluation, and demonstrate its application at clinical scale.

### Must NOT contain
- Methodology details (architecture names, hyperparameters, layer numbers)
- Baseline-specific numbers (TF-IDF AUC, lexical win counts)
- Limitations or caveats

---

## 1. Introduction (~1.25 pages)

### Purpose
Motivate the research question, establish the validation gap as a first-class problem, and preview the contribution. The introduction should make a reviewer who has read Korznikov/Heap think "this paper might actually address my concerns."

### Structure and content

**Opening paragraph — The SAE promise and its crisis of validation (~150 words)**
- SAEs have become the default tool for decomposing neural network representations into interpretable features (cite EMNLP 2025 survey)
- The core promise: sparse, monosemantic features that correspond to human-interpretable concepts
- But: Korznikov et al. (2026) show random baselines match trained SAEs on interpretability scoring, sparse probing, and causal editing; Heap et al. (2025) show auto-interp metrics can't distinguish trained from random transformers; Leask et al. (ICLR 2025) show features are neither complete nor atomic
- The result: it is unclear whether SAE features capture genuine structure in model representations, or are artifacts of the decomposition method itself

**Second paragraph — The missing piece: external ground truth (~120 words)**
- All critiques above evaluate SAEs using *internal* metrics (interpretability scores, reconstruction, probing on the same model)
- What's needed: validation against *external* structured labels that exist independently of the model and the SAE
- Clinical text provides a uniquely suitable setting: discharge notes contain rich natural language *and* are paired with structured diagnosis codes (ICD-9) assigned by human coders
- This creates a natural experiment: if SAE features genuinely encode clinical concepts, they should correlate with diagnosis labels *and* their LLM-generated explanations should name those diagnoses

**Third paragraph — What we do (~120 words)**
- We train JumpReLU SAEs (18,432 latents, 8× expansion) on Gemma-2-2B layer-16 residual stream activations from 50,000 MIMIC-IV discharge summaries
- We validate through three independent modalities:
  1. **Statistical grounding:** point-biserial correlation between per-note SAE activations and 46 binary ICD-9 indicators
  2. **Semantic concordance:** checking whether LLM-generated explanations of feature function align with the ICD code each feature correlates with
  3. **Causal ablation:** zero-ablation of individual features, measuring differential loss on diagnosis-positive vs diagnosis-negative notes
- We compare against a published general-domain SAE (GemmaScope-16k) and two text-based baselines (lexical keywords, TF-IDF + LR)

**Fourth paragraph — What we find (~120 words)**
- Individual SAE features achieve r_pb up to 0.864 with specific ICD-9 codes; 9,023 features (49%) are significantly grounded at |r|>0.1
- The concordance gradient: as correlation strength increases, the rate at which LLM explanations match the associated ICD code rises from 85% to 98.6% — near-perfect alignment at the strongest associations
- Features achieving r>0.6 are 100% monospecific (each tracks exactly one diagnosis)
- Causal ablation confirms grounded features differentially affect model predictions (median δ=0.30 for grounded vs −0.04 for controls)
- Domain-trained SAEs outperform GemmaScope by ~50% in peak correlation and 12–14× in grounded feature count at higher thresholds; GemmaScope's grounding largely collapses under length-confound control

**Fifth paragraph — Contribution summary (~80 words)**
- Enumerate contributions as a numbered list:
  1. A three-modality validation framework (statistical + semantic + causal) for SAE features using external structured labels
  2. The concordance validation methodology: a scalable, general-purpose technique for checking whether unsupervised feature explanations align with known labels
  3. Empirical demonstration at clinical scale (50k notes, 46 codes, 18k features) that domain-trained SAE features encode genuine clinical concepts, robust to confound controls and superior to surface-text baselines
  4. Evidence that general-domain SAEs suffer genuine feature-direction mismatch on out-of-distribution text, not merely mean shift

---

## 2. Related Work (~1.25 pages)

### Purpose
Position the paper at the intersection of three literatures. Demonstrate awareness of the SAE critique landscape and show how this work responds to it. Avoid being merely a list of papers — each paragraph should advance the argument for why external validation is needed.

### 2.1 Sparse Autoencoders for Mechanistic Interpretability (~0.4 pages)

**What to cover:**
- Brief history: Bricken et al. (2023), Cunningham et al. (2023) established residual-stream SAEs. Gao et al. (2024, OpenAI) scaled to GPT-4. Rajamanoharan et al. (2024, DeepMind) introduced JumpReLU
- The EMNLP 2025 survey as a consolidation point
- Anthropic (2025) circuit tracing / transcoders as a signal the field is moving beyond residual-stream SAEs

**Critical work to address head-on (devote ~0.3 pages to this):**
- Leask et al. (ICLR 2025): non-canonicity — SAE features are neither complete nor atomic. Our response: we don't claim canonicity, only *utility* — these features correlate with external labels, are causally relevant, and semantically interpretable
- Korznikov et al. (2026): random baselines match on internal metrics; 9% ground-truth recovery at 71% EV. Our response: we validate against *external* labels, a fundamentally different evaluation. Random feature directions would not produce r=0.864 against specific ICD codes
- Heap et al. (2025): auto-interp metrics don't distinguish trained from random transformers. Our response: concordance validation uses external labels, not auto-interp scoring, as the discriminative metric
- Ma et al. (2026): reasoning features activated by token injection. Our response: multi-modal validation (not just activation patterns) and partial correlation controls for surface confounds

**Tone:** Engage these critiques seriously and specifically, not dismissively. Frame our work as a partial response — we don't solve the SAE validity problem wholesale, but we demonstrate that external ground-truth validation can distinguish genuine clinical features from noise.

### 2.2 Domain-Specific SAE Applications (~0.3 pages)

**What to cover:**
- O'Neill et al. (2025) — JumpReLU SAEs on Gemma-2 for clinical QA. Closest competitor. Key differentiator: they validate via interpretability scoring and human evaluation; we validate against structured diagnosis labels. They use clinical QA text; we use EHR discharge notes with paired ICD codes
- Sainsbury et al. (2026) — TopK SAEs on FlatASCEND (clinical sequence model) using MIMIC-IV. Differentiator: they study a clinical-specific model on coded events; we study a general-purpose LLM on free text — fundamentally different question (does internet-scale pretraining encode clinical structure?)
- Brief mention of SAEs in other domains: EEG (arXiv 2605.13930), ASR (arXiv 2605.12225), radiology (arXiv 2507.12950), biological models (arXiv 2603.02952)
- Position our work: this is part of a growing trend of domain-specific SAE validation, but uniquely uses *structured external labels* rather than interpretability scoring

### 2.3 Clinical NLP and Structured Label Evaluation (~0.25 pages)

**What to cover:**
- MIMIC-IV as a benchmark dataset; ICD coding as structured supervision
- Prior work on probing LLM representations for clinical knowledge (not SAE-specific)
- The gap: no prior work connects SAE-level feature decomposition to structured clinical labels
- Auto-interpretability: Paulo et al. (2024) / Delphi for fuzzing + detection; McCann (2026) descriptive collision as a known limitation; our concordance methodology as an extension that checks *content* alignment against external labels, not just activation prediction

---

## 3. Methodology (~1.75 pages)

### Purpose
Complete, reproducible description of the pipeline. A reader should be able to replicate the experiments from this section alone. Since the methodology is the vehicle for the contribution (not the contribution itself), keep it precise but not sprawling.

### 3.1 Data and Activation Extraction (~0.3 pages)

**What to cover:**
- MIMIC-IV discharge summaries: 50,000 stratified sample, de-identified, IRB/PhysioNet credentialing
- Gemma-2-2B, layer 16, residual stream (d_model=2304). Justify layer choice briefly (mid-depth residual stream captures abstract semantic features; layer 16 is roughly 60% depth in a 26-layer model)
- Activation extraction: 312 shards of fp16 safetensors, max_length=8192
- Mean centering: float64 accumulator for numerical stability across 15.2M tokens
- Train/test split: shards 0–280 for SAE training, shards 281–311 (4,911 notes) held out for evaluation

**Table:** Data statistics (n_notes, n_tokens, n_shards, ICD codes after prevalence filtering)

### 3.2 SAE Training (~0.3 pages)

**What to cover:**
- JumpReLU architecture: learned per-feature thresholds, STE gradient estimation, L0 sparsity penalty (cite Rajamanoharan et al. 2024)
- Key hyperparameters: 18,432 latents (8× expansion), lambda_l0=10, bandwidth=0.5, lr=2e-4, adam_beta1=0.0
- Training: 3 epochs on centered activations, best checkpoint at step 36k by EV (0.899)
- Vanilla ReLU+L1 as architectural comparison (same d_sae, same data)
- GemmaScope-16k (Google) as pre-trained general-domain baseline: 16,384 latents, average L0=42, applied zero-shot to clinical activations without fine-tuning
- Reconstruction diagnostics: JumpReLU EV=0.906, Vanilla EV=0.889, GemmaScope EV=−4.21

**Table:** SAE configuration and reconstruction metrics (L0, EV, dead fraction, d_sae) for all three

### 3.3 ICD-9 Clinical Grounding (~0.3 pages)

**What to cover:**
- Per-note pooling: element-wise max of SAE activations across tokens per note → [n_notes, d_sae] matrix
- ICD-9 labels: 46 binary indicators surviving min_prevalence=0.02 filter, from the same CSV used for extraction
- Point-biserial correlation: r_pb for each (feature, ICD code) pair, with BH-FDR correction at q=0.05
- A feature is "grounded" if any |r_pb| exceeds the threshold (default 0.1)
- Confound control: partial correlation residualising activations on n_tokens (OLS) to remove the max-pooling length confound
- Monospecificity: count of ICD codes per grounded feature at each threshold

**Define clearly:** grounded latent count, grounded latent fraction, monospecificity

### 3.4 Auto-Interpretability and Concordance Validation (~0.4 pages)

**This is the methodological contribution — allocate the most space within methodology.**

**What to cover:**

*Feature explanation generation:*
- Paulo et al. (2024) / Delphi framework: present top-activating examples to an LLM, generate a natural-language explanation
- Claude Sonnet as explainer; 964 features processed across 4 tiers (280 strong grounded r>0.4, 100 weak r=0.1–0.3, 484 non-grounded, 100 dead)

*5-way categorization:*
- LLM classifies each feature as: clinical_concept, clinical_vocabulary, structural_pattern, general_language, or noise
- Definitions for each category (1 sentence each)

*Fuzzing + Detection scoring:*
- Brief description of fuzzing (does the explanation predict which tokens activate?) and detection (does the explanation predict which examples contain the feature?) — cite Paulo et al.
- Acknowledge known ceiling effect (Heap et al. 2025; McCann 2026) upfront

*Concordance validation (the novel method):*
- **Define formally:** For each grounded feature f with top-associated ICD code c and LLM-generated explanation e, a concordance judge (Claude Sonnet) is prompted: "Does this explanation of feature behaviour [e] describe a concept semantically related to [ICD code description for c]?" Verdict: YES (direct match), PARTIAL (semantically related), NO (unrelated), UNKNOWN
- **Key insight:** This checks whether two independent signals — unsupervised feature explanation and statistical label association — converge. Neither signal references the other during generation
- **Concordance rate:** (YES + PARTIAL) / total, computed at multiple |r| thresholds
- **Why this is different from scorer accuracy:** Scorer accuracy tests whether the explanation predicts *activation patterns*. Concordance tests whether the explanation's *semantic content* matches an *external label*. The distinction matters because descriptive collision (McCann 2026) can inflate scorer accuracy without implying semantic specificity

### 3.5 Causal Ablation (~0.25 pages)

**What to cover:**
- Zero-ablation protocol: set feature's sparse code to zero → decode to modified activations → splice back into Gemma's residual stream at layer 16 → measure cross-entropy change on last 25% of note tokens
- Comparison: loss change distribution on ICD-positive notes vs ICD-negative notes
- Statistical test: Mann-Whitney U (one-sided) with BH-FDR correction
- Effect size: Cliff's δ
- Reconstruction tax: mean loss increase from SAE reconstruction alone (splice without ablation), measured as baseline distortion
- Controls: random features (no grounding) and low-|r| features as negative controls

### 3.6 Baselines (~0.15 pages)

**What to cover (brief — details go in appendix):**
- Lexical keyword baseline: curated keyword dictionaries (25 keywords per ICD code), binary co-occurrence, point-biserial correlation. Tests whether SAE features detect something beyond keyword presence
- TF-IDF + LR: 10k features, 1+2-grams, sublinear TF, per-code stratified 5-fold CV. Tests whether a full-vocabulary supervised classifier outperforms SAE features. Also compare best-feature univariate correlation (SAE vs TF-IDF)
- GemmaScope as SAE baseline (described in §3.2)

---

## 4. Results (~2.5 pages)

### Purpose
Present results organised by claim, not by experiment. Each subsection advances the central argument: SAE features encode genuine clinical concepts, validated through external ground truth. This is the largest section — it carries the empirical weight.

### 4.1 SAE Features Correlate with Diagnosis Codes (~0.5 pages)

**Claim:** Domain-trained SAE features achieve strong, statistically significant correlations with specific ICD-9 diagnosis codes.

**What to present:**
- Three-SAE grounding comparison table: grounded latent count and fraction at r>0.1, 0.3, 0.5, 0.7 for JumpReLU, Vanilla, GemmaScope
- Top-5 feature–code associations per SAE (feature index, ICD code, |r|)
- Headline: JumpReLU peak r=0.864 (atrial fibrillation), Vanilla peak r=0.853 (hypothyroidism), GemmaScope peak r=0.574
- The gap widens at higher thresholds: at r>0.3, custom SAEs have 12–14× more grounded latents; at r>0.5, 36× more
- Test-split validation: grounding holds identically on held-out data (r=0.864 on both splits for JumpReLU)

**Figure 1:** Threshold sweep — log-scale grounded latent count vs |r| threshold for all three SAEs. This is the most visually compelling result after concordance.

### 4.2 Correlations Reflect Clinical Structure, Not Surface Confounds (~0.5 pages)

**Claim:** The observed correlations are not explained by note length, keyword co-occurrence, or surface text features.

**What to present:**

*Partial correlation:*
- After residualising on n_tokens, 43% of weakly grounded features drop out (r>0.1: 9,023 → 5,147) but top-10 features barely change (0.864 → 0.853)
- GemmaScope collapses: 84% reduction at r>0.1 (5,749 → 894). Its grounding is largely a length confound
- Interpretation: max-pooling inflates weak associations but the strongest clinical features are robust

*Lexical baseline:*
- Custom SAEs beat the keyword baseline on 45/46 codes (JumpReLU) and 44/46 (Vanilla); GemmaScope loses on 12/46
- Keyword-absent recall: SAE features fire on ICD-positive notes where *no* keyword appears — evidence of beyond-surface representation

*TF-IDF baseline:*
- TF-IDF LR wins on classification AUC (0.917 vs 0.888) — expected, it's a 10k-feature supervised classifier
- But SAE's best univariate feature achieves stronger correlation than TF-IDF's best feature on 21–23/46 codes
- Frame correctly: SAEs produce monosemantic single-concept detectors; TF-IDF produces polysemantic classifiers

### 4.3 Features Are Monospecific Clinical Concepts (~0.3 pages)

**Claim:** Strongly grounded SAE features encode discrete, individual clinical concepts — not diffuse multi-condition patterns.

**What to present:**
- Monospecificity gradient table: from 30% mono at r>0.1 to 100% mono at r>0.6
- At r>0.5: 93.8% monospecific, mean 1.06 codes/latent — nearly perfect one-to-one mapping
- Feature inspection: top latents fire on clinically specific tokens (e.g., "thyroidism" for hypothyroid feature, disease-specific medication names for relevant conditions)
- Diversity scores (0.087–0.091) confirm lexical concentration of top activating tokens

**Figure 2 (optional):** Monospecificity curve — % monospecific vs |r| threshold. Clean monotonic rise. Could be combined with Figure 1 as a two-panel figure.

### 4.4 LLM Explanations Align with Structured Labels — Concordance Gradient (~0.6 pages)

**THIS IS THE MAIN RESULT. Allocate the most space.**

**Claim:** Unsupervised LLM explanations of feature function increasingly match the external ICD code as statistical association strengthens — a concordance gradient that provides novel evidence for semantic validity of SAE features.

**What to present:**
- Concordance table: all (85%), r>0.3 (94.6%), r>0.5 (98.6%)
- Exact match rises: 22.4% → 30.4% → 42.4%
- NO verdicts nearly vanish: 46 → 7 → 1
- At r>0.5, only 1 of 144 features has a NO concordance verdict

*Tier contrast:*
- Strong grounded: 94.6% concordance, 30.4% exact match
- Weak grounded: 58.0% concordance, 0% exact match, 39% NO
- The weak tier has *zero* YES verdicts — exactly the expected pattern if concordance reflects genuine semantic alignment rather than LLM bias

*5-way categorization as supporting evidence:*
- 90% of strong grounded features categorised as clinical (47.1% clinical_concept + 42.9% clinical_vocabulary)
- 99.8% of non-grounded features categorised as noise
- Dead features are 75% structural patterns (boilerplate/template), not random noise

*Scorer accuracy — honest report:*
- Fuzzing 0.938, Detection 0.962 — no tier separation
- Consistent with Heap et al. (2025) finding that auto-interp metrics lack discriminative power
- Frame as evidence that concordance validation provides discrimination where standard metrics cannot

**Figure 3:** Concordance gradient bar chart — stacked YES/PARTIAL/NO bars at each |r| threshold. This is the paper's signature figure.

### 4.5 Features Are Causally Relevant (~0.3 pages)

**Claim:** Grounded features are not merely correlated with diagnosis — ablating them differentially affects model predictions on diagnosis-positive notes.

**What to present:**
- Vanilla ablation: 15/20 grounded features significant at q=0.05; two with large effect sizes (δ>0.5)
- Control comparison: grounded median δ=0.300 vs controls δ=−0.036. Controls are null — grounded features are systematically causal
- Reconstruction tax: 0.029 nats (1.8% of base loss) for custom SAE vs 0.648 nats (39.6%) for GemmaScope — 22× difference confirms domain shift at the LM-loss level
- GemmaScope ablation: 15/20 significant but with 22× reconstruction tax — the ablation measures removal from an already-degraded representation
- Note limitation: ablation is for Vanilla SAE only; JumpReLU ablation is future work (or, if run before submission, include it here)

### 4.6 Domain Shift: Custom SAEs vs GemmaScope (~0.3 pages)

**Claim:** The performance gap between domain-trained and general-domain SAEs reflects genuine feature-direction mismatch, not merely mean shift.

**What to present:**
- GemmaScope EV = −4.21 (reconstruction worse than predicting the mean)
- Domain-shift diagnostics: 71.1% of features fire above threshold (vs ~5% for custom SAEs); cosine similarity between clinical mean and b_dec is 0.90 (close but not close enough)
- Full recentering worsens EV to −6.564 — rules out mean shift as primary cause. The problem is feature directions, not the decoder bias
- GemmaScope grounding collapses under partial correlation (84% reduction at r>0.1)
- GemmaScope TF-IDF baseline abandoned (LR didn't converge — 71% feature firing violates sparsity)
- Interpretation: general-domain SAE dictionaries don't span the clinical activation subspace. Domain-specific training is necessary, not just beneficial

---

## 5. Discussion (~0.75 pages)

### Purpose
Interpret the results, position the contribution, acknowledge limitations honestly, and identify future work. Do not repeat numbers — interpret them.

### 5.1 Concordance Validation as a General Methodology (~0.25 pages)

**What to cover:**
- The concordance method generalises beyond clinical NLP: any domain where external structured labels exist (legal case codes, bug categories, gene ontology terms) can use the same framework
- The key insight is checking convergence between two independent signals: unsupervised feature explanation and statistical label association
- This provides a validation modality that is robust to the criticisms of Korznikov et al. and Heap et al. — it doesn't rely on auto-interp scoring or reconstruction metrics
- Acknowledge McCann (2026) descriptive collision concern; argue the gradient pattern (85% → 99%) mitigates it, but a shuffled-explanation control would strengthen the claim further

### 5.2 What SAE Features Do and Don't Capture (~0.2 pages)

**What to cover:**
- SAE features capture clinically meaningful concepts at the individual-feature level (monospecific, concordant, causally relevant)
- But TF-IDF LR beats SAE LR on classification AUC — the SAE doesn't capture everything useful for diagnosis prediction. It produces *interpretable single-concept detectors*, not optimal classifiers
- This is the expected trade-off between interpretability and predictive completeness (relate to Leask et al.'s non-canonicity finding)
- The 43% drop under partial correlation at weak thresholds is an honest limitation of max-pooling — report it, don't hide it

### 5.3 Limitations (~0.2 pages)

**What to cover (be upfront and specific):**
- Single model (Gemma-2-2B), single layer (layer 16) — no evidence of generalization across architectures or depths
- ICD-9 codes are a coarse labeling scheme (46 codes); finer-grained labels (ICD-10, SNOMED) might reveal more structure
- Max-pooling conflates note length with feature activation; partial correlation controls for this but doesn't eliminate it
- Auto-interp ran on 964/1,480 target features (pipeline timeout); scorer accuracy ceiling limits one validation modality
- Ablation limited to Vanilla SAE and 20 features; JumpReLU (primary model) ablation not yet available
- Concordance validation depends on LLM judge quality — different judges might produce different concordance rates

### 5.4 Future Work (~0.1 pages)

**What to cover (brief):**
- Multi-model comparison (Llama-3, Mistral, larger Gemma models)
- Multi-layer analysis (progressive abstraction as in Sainsbury et al.)
- Circuit-level analysis: trace how clinical features propagate through attention heads to affect downstream predictions
- Downstream utility: use grounded features for interpretable diagnosis prediction, bias detection, or note-quality auditing

---

## 6. Conclusion (~0.25 pages)

### Purpose
One paragraph restating the contribution. No new information.

### Must contain
- We validated SAE features against external structured labels (ICD-9 codes) through three modalities: statistical grounding (r up to 0.864), semantic concordance (98.6% at r>0.5), and causal ablation (δ=0.30 grounded vs −0.04 controls)
- The concordance gradient — unsupervised feature explanations increasingly align with structured labels as statistical association strengthens — is a novel validation methodology applicable beyond clinical NLP
- Domain-trained SAEs dramatically outperform general-domain SAEs, with the gap reflecting genuine feature-direction mismatch confirmed by recentering analysis
- Clinical text provides a uniquely suitable testbed for SAE evaluation because structured labels exist. We encourage the community to apply concordance validation wherever external labels are available

---

## References

### Must-cite papers (core narrative)

**SAE foundations:**
- Bricken et al. (2023) — Towards Monosemanticity
- Cunningham et al. (2023) — Sparse autoencoders find interpretable features
- Rajamanoharan et al. (2024) — JumpReLU SAEs (DeepMind)
- Gao et al. (2024) — Scaling and evaluating SAEs (OpenAI)

**SAE critiques (address directly):**
- Leask et al. (ICLR 2025) — Non-canonicity
- Korznikov et al. (2026) — Random baselines
- Heap et al. (2025) — Auto-interp metrics on random transformers
- Ma et al. (2026) — Falsifying reasoning features

**Auto-interpretability:**
- Paulo et al. (2024) — Delphi / Automatically interpreting millions of features
- McCann (2026) — Descriptive collision

**Domain-specific SAEs:**
- O'Neill et al. (2025) — Resurrecting the Salmon (clinical QA)
- Sainsbury et al. (2026) — Clinical sequence model SAEs

**Surveys:**
- EMNLP 2025 survey on sparse autoencoders

**Clinical NLP / data:**
- MIMIC-IV dataset papers
- ICD coding references

**Statistical methods:**
- Point-biserial correlation
- Benjamini-Hochberg FDR
- Cliff's δ / Mann-Whitney U

**Other domain SAEs (brief mention):**
- EEG (arXiv 2605.13930), Radiology (arXiv 2507.12950), ASR (arXiv 2605.12225)

---

## Appendix (unlimited pages)

### What goes here

**A. Extended tables:**
- Full threshold sweep (r>0.1 through r>0.7) for all three SAEs, both raw and partial correlation
- Per-code grounding results (top-3 features per ICD code)
- Full TF-IDF comparison per code (46-row table)
- JumpReLU training trajectory (eval_scan_summary from step 2k to 40k)

**B. Ablation details:**
- Full per-feature ablation results (20 features × 2 SAEs)
- Control feature results
- Reconstruction tax computation

**C. Auto-interpretability details:**
- Categorization breakdown by tier (full table)
- Concordance scoring prompt template
- Example concordance verdicts (2–3 curated examples showing YES, PARTIAL, NO — structural only, no PHI)
- Scorer ceiling analysis and explanation

**D. Domain-shift diagnostics:**
- Full GemmaScope domain-shift analysis (mean comparison, scale comparison, EV variants)
- Feature firing rate distribution comparison (custom SAEs vs GemmaScope)

**E. Lexical baseline details:**
- Keyword dictionary construction methodology
- Per-code keyword-absent recall rates

**F. Reproducibility:**
- Complete hyperparameter tables
- Compute budget (GPU hours, API costs for auto-interp)
- Data access instructions (PhysioNet credentialing)
- Code availability statement

---

## Figures and Tables Budget

EMNLP long papers have limited space. Prioritise figures that advance the central argument.

| # | Type | Content | Section | Priority |
|---|------|---------|---------|----------|
| Fig 1 | Line plot | Threshold sweep: grounded latent count vs |r| threshold (3 SAEs, log scale) | §4.1 | High |
| Fig 2 | Stacked bar | Concordance gradient: YES/PARTIAL/NO at each |r| threshold | §4.4 | **Critical** — the paper's signature figure |
| Fig 3 | Heatmap or bar | 5-way categorization by tier (strong/weak/dead/non-grounded) | §4.4 | Medium |
| Tab 1 | Table | SAE configuration + reconstruction metrics (3 SAEs) | §3.2 | High |
| Tab 2 | Table | ICD-9 grounding summary (3 SAEs, multiple thresholds) | §4.1 | High |
| Tab 3 | Table | Baseline comparison summary (lexical + TF-IDF) | §4.2 | Medium |
| Tab 4 | Table | Concordance results at 3 thresholds | §4.4 | **Critical** |
| Tab 5 | Table | Ablation summary (sig rate, median δ, controls, recon tax) | §4.5 | Medium |
| Tab 6 | Table | Domain-shift diagnostics (EV, firing rate, recentering) | §4.6 | Medium |

**Space allocation:** ~3 figures + ~4 in-text tables ≈ 2 pages. Remaining tables go to appendix.

---

## Content Priority Tiers

If space is tight, cut from the bottom up.

**Tier 1 — Non-negotiable (must be in main paper):**
- Concordance gradient (method + results)
- ICD-9 grounding (3-SAE comparison)
- Partial correlation confound control
- Test-split validation
- SAE critique literature engagement (Korznikov, Heap, Leask et al.)

**Tier 2 — Strongly recommended:**
- Monospecificity gradient
- Ablation results (even if vanilla only)
- Lexical baseline (SAE beats keywords on 45/46)
- GemmaScope domain-shift analysis
- 5-way categorization breakdown

**Tier 3 — Include if space allows, otherwise appendix:**
- TF-IDF baseline details (summary stat sufficient in main; per-code in appendix)
- Scorer accuracy (report briefly, acknowledge ceiling, cite Heap)
- Feature inspection token-level examples
- Training trajectory details
- Raw-activation LR baseline (if run before submission)
