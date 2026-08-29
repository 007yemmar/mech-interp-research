# Submission 15186 — Experiment Plan for Resubmission

Derived from the EMNLP/ARR meta-review (Area Chair n2to, Overall 2.5 = Borderline Findings) and the three reviews. Full review text is in `emnlp_reviews.md`.

Scope of this document: **Block A** (experiments that need to be run) and **Block B** (post-hoc computations needed for the results).

---

# Context — the meta-review

## The two paragraphs that matter

**Paragraph 1 — the single open issue:**

> The main issue raised by one reviewer (GA3f) is still open. The paper shows SAE features produce useful audit signals, but not that SAEs are needed to produce them. A comparison is required, running the same audit protocol, same label-selection rule, same explanation budget and same off-target diagnostic over raw-activation difference-in-means directions, supervised probes and TF-IDF, and asking which method better separates disease signal from treatment logic and template artifacts. Experiments with GemmaScope is meaningful, but does not fill the aforementioned checks. The complete pipeline needs to be run on a matched non-learned baseline such as random directions with matched activation statistics, and preferably PCA or ICA.

**Paragraph 2 — the committed changes:**

> Beyond that, the authors should deliver the changes they already committed to; scoping the claims to ICD codes as noisy anchors and specificity relative to the 46-code panel, report YES-only alongside YES+PARTIAL with the PARTIAL subtype breakdown, add the vanilla-SAE ablation, clarifying the multiple-testing map, add prevalence and threshold-sensitivity tables, describe ablation as diagnosis-conditioned loss relevance rather than mechanistic faithfulness, soften the GemmaScope direction-mismatch claim, fix the "test-set explained variance" wording, and expand the limitations on single seed, single model, credentialed data access, and the risk of overreading ICD concordance as clinical validation. A small blinded clinician check, not feasible during this cycle, would strengthen the paper considerably.

---

## What each phrase refers to in our paper

| Phrase in the review | What it refers to in our work | What it means for us |
|---|---|---|
| **"SAE features produce useful audit signals"** | Not the SAE architecture — the *outputs* we use as evidence: grounded features, monospecificity counts, concordance verdicts, ablation effects | The AC accepts these signals are informative. The objection is about their *source*, not their validity |
| **"but not that SAEs are needed to produce them"** | Our contribution claim in §1 and §5 | We showed the SAE is *sufficient* to produce clinical audit units. We never showed it is *necessary* — that a simpler method could not produce the same signals |
| **"the same audit protocol"** | §3.4 statistical grounding: point-biserial r between max-pooled note activations and each of the 46 ICD-9 labels, BH-FDR at q=0.05, thresholds swept 0.1–0.7, evaluated on the held-out set (shards 281–311, 4,911 notes) | Every baseline must go through this exact code path — same split, same pooling, same thresholds, same correction. Not a reimplementation per method |
| **"the same label-selection rule"** | How we pick *which* feature represents each code: top latent by \|r_pb\| | The SAE's rule (select on train/full corpus, audit on held-out) must be mirrored. Baselines with one direction per code have a trivial rule; the constraint mainly keeps the SAE side honest |
| **"the same explanation budget"** | §3.5 auto-interp: the LLM explanation + 5-way categorization + concordance judge | Each method gets the *same number* of features explained. Ours is 46 (top latent per code) for this comparison, so baselines also get 46 — not the 964 of the original tier sweep |
| **"the same off-target diagnostic"** | The c-negative off-target specificity used in `ablation_posthoc` — correlating a feature with every *other* code, restricted to notes negative for its own code | This currently exists only for ablation deltas. There is no correlation-based off-target instrument for any baseline. It must be built once and applied identically |
| **"raw-activation difference-in-means directions"** | Not in the paper. Code exists (`diff_in_means_baseline.py`), never run | Block A1 |
| **"supervised probes"** | We have `raw_lr_baseline.py` (mean AUC-ROC 0.8082) but only as a *classifier*. We never used its weight vector as a direction | Block A2 |
| **"TF-IDF"** | §4.3 / App F.2: TF-IDF + LR, 10k n-grams, 5-fold CV, AUC 0.917 vs our 0.888. A *prediction* comparison | Block A3. The AC wants TF-IDF's top n-gram audited, not benchmarked. Different experiment, same fitted model |
| **"which method better separates disease signal from treatment logic and template artifacts"** | Our 5-way categorization (Table 10): `clinical_concept` / `clinical_vocabulary` / `structural_pattern` / `general_language` / `noise` | The actual criterion. Cannot be answered from grounding numbers — requires the LLM categorization pass on baseline directions (A7). A per-code direction should smear across disease + drug + template; an SAE feature should not |
| **"Experiments with GemmaScope... does not fill the aforementioned checks"** | §4.1 and Table 1: our 12.5× / 36.8× grounded-latent advantage over GemmaScope-16k | GemmaScope is a *learned SAE in the wrong domain*. It tests domain-specificity, not whether an SAE is needed at all. This result no longer counts toward necessity |
| **"a matched non-learned baseline such as random directions with matched activation statistics"** | Nothing in the paper. Our rebuttal *argued* random directions would show a flat concordance gradient but never ran them | Block A4 — the decisive null, and the item the AC singles out. "Matched activation statistics" = sampled from the activation covariance, and firing sparsity matched to the SAE |
| **"and preferably PCA or ICA"** | Nothing in the paper | Blocks A5, A6. Non-learned unsupervised decompositions. Caveat to state: both cap at d_model = 2,304 < our 18,432, so only random and diff-in-means match our count |
| **"scoping the claims to ICD codes as noisy anchors"** | §1, §3.1, §5 language treating ICD concordance as validation of clinical concept identity | Text change. Worth adding that annotation noise *attenuates* correlation, so our grounding is a lower bound |
| **"specificity relative to the 46-code panel"** | §4.2 monospecificity claims ("features above r > 0.6 each track exactly one diagnosis code") | Text + Block C. "Specific among 46 comorbidity-correlated codes" is weaker than "monosemantic" |
| **"report YES-only alongside YES+PARTIAL with the PARTIAL subtype breakdown"** | Table 2 / Table 9 concordance = (YES+PARTIAL)/N | Block C. The subtype data exists (56% treatment/medication, 18% monitoring) and is a *result*, not a concession |
| **"add the vanilla-SAE ablation"** | §4.4 / Tables 13–15 | Now run on all three SAEs. Requires correcting the attribution in Tables 3/13/14 and the Limitations paragraph |
| **"clarifying the multiple-testing map"** | §3.4 BH at q=0.05 | Block C. Single BH over d_sae × 46 = 847,872 tests; concordance/categorization/ablation are downstream and uncorrected |
| **"describe ablation as diagnosis-conditioned loss relevance rather than mechanistic faithfulness"** | §3.6 and §4.4, both titled "Causal Ablation"; "causal ablation" in the abstract | Rename the section and the abstract phrasing, not just add a caveat |
| **"soften the GemmaScope direction-mismatch claim"** | §4.3 and App G: "The cause is direction mismatch" | Diagnostics rule out simple mean shift; they do not exclude other distributional mismatch |
| **"fix the 'test-set explained variance' wording"** | Table 5, "Checkpoint selection: test-set explained variance" | Refers to our held-out evaluation split, not the MIMIC-IV test split (never accessed) |
| **"expand the limitations"** | §6 | Single seed, single model, credentialed data, and the risk of over-reading ICD concordance as clinical validation |
| **"A small blinded clinician check"** | Not done | AC says it would strengthen the paper "considerably." Deferred last cycle |

---

## The verdict rule this implies

The comparison is **not** won on grounding strength. Diff-in-means and the supervised probe are built *from* the labels; our SAE never sees them. They should match or beat us on on-target \|r\|. Conceding that up front is free and moves the argument to the axis we can win.

The discriminating axes are:

1. **Specificity** — c-negative off-target ratio and off-target code count, at *matched* on-target grounding.
2. **Decomposition** — a per-code direction cannot split one ICD code into diagnosis / treatment / monitoring subpatterns; there is only one direction. Our CHF result (13 features at \|r\|>0.5) is only expressible in the SAE frame.
3. **Category purity** — does the method's feature get a single-category explanation, or does it mix disease, drug, and template content?

And the necessity argument only holds if **random-matched fails the audit**. If random directions ground and concord under the same selection procedure, the finding is that the *pipeline* manufactures structure, not that the SAE is unnecessary.

---

# Block A — Experiments that need to be run

## A0. Source-agnostic audit harness

**What it is.** One piece of code that takes any matrix of per-note feature values — `[n_notes × k]` — and runs the full grounding audit on it: point-biserial correlation against all 46 ICD-9 labels on held-out notes, BH-FDR at q=0.05, c-negative off-target specificity, and monospecificity counts. It does not know or care whether the features came from an SAE, a random projection, or a keyword count.

**Why it must be run.** The AC's requirement is *"the same audit protocol... same off-target diagnostic."* If each baseline gets its own analysis script, "same" is something we assert rather than something the code guarantees — and a reviewer can reasonably doubt it. Building one harness makes fairness structural. It is also the cheapest way to add PCA and ICA later, since they become new inputs rather than new pipelines.

**Status.** `diff_in_means_baseline.off_target_specificity_corr` is the seed — the only correlation-based off-target implementation that exists. `icd_eval.compute_point_biserial_vectorised`, `apply_bh_correction`, `compute_grounding`, and `compute_monospecificity` are reusable as-is.

**GPU:** none.

---

## A1. Difference-in-means directions

**What it is.** For each ICD code, one direction in activation space pointing from the average code-negative note to the average code-positive note: `d_c = normalize(mean(X_train[y_c=1]) − mean(X_train[y_c=0]))`, built on the pooled raw centered layer-16 activations `[N × 2304]`. Stack the 46 directions, project held-out notes onto them, audit through A0.

**Why it must be run.** Named explicitly by the AC (*"raw-activation difference-in-means directions"*) and by R1 (*"they do not yet show that SAE features provide a better clinical audit unit than raw activation probes, difference-in-means directions, task-specific supervised probes, or lexical methods"*). This is the **strongest simple supervised baseline** — two means and a subtraction. If it separates disease signal from treatment and template content as well as an SAE feature does, the SAE's interpretability advantage does not exist.

**Implementation notes.**

- Code is written and unit-tested in `diff_in_means_baseline.py` but has **never been run**. No output artifacts exist.
- **Z-score / diagonal-whiten before differencing.** The plain version collapses onto the max-pooling magnitude confound.
- **Build on train shards (<281), audit on held-out (≥281).** Non-negotiable — otherwise on-target grounding is circular.
- The config's `saes:` list currently contains only a vanilla entry. Add a JumpReLU entry; §4's headline numbers are JumpReLU.
- Fairness wrinkle to disclose or fix: diff-in-means does *pool → project*, the SAE does *project → threshold → pool*. Max-pooling is nonlinear, so these are not equivalent, and the SAE gets to select its single strongest token.

**GPU:** none — CPU, minutes. Reads `raw_shard_ckpt/` and `sample_50k.csv`.

---

## A2. Supervised probe as a direction

**What it is.** Fit a logistic regression per ICD code on the pooled 2,304-dim activations, then use the **fitted weight vector** as that code's direction (L2-normalized). Project held-out notes onto it and audit through A0. The probe differs from diff-in-means by accounting for the full activation covariance rather than just the class means.

**Why it must be run.** Named separately by the AC (*"supervised probes"*) and R1 (*"task-specific supervised probes"*). We already have this model — `raw_lr_baseline.py`, mean AUC-ROC 0.8082 — but only as a **classifier**. We reported how well it predicts and never asked whether its direction is an interpretable unit. That is the entire gap: *the paper compared baselines on prediction; the AC asked for baselines compared on audit properties.*

**Implementation notes.**

- Train on X = max-pooled centered layer-16 activations `[N × 2304]`, y = binary code label. 46 classifiers.
- **Refit on train shards only.** The existing run used 5-fold CV across all 50,000 notes; reusing those fits makes held-out grounding circular.
- One direction per code, so no selection rule is needed.

**GPU:** none — sklearn on cached pooled vectors. No Gemma forward pass, no SAE encode.

---

## A3. TF-IDF as an audited source

**What it is.** Take the single strongest n-gram feature per ICD code from the existing TF-IDF model and run it through the *audit* rather than a classification benchmark: held-out point-biserial grounding, c-negative off-target specificity, monospecificity, and categorization of the n-gram itself.

**Why it must be run.** All three reviewers raised TF-IDF, and all three read our §4.3 as "TF-IDF beats the SAE":

- R2 (Weakness 4): *"The keyword baseline is limited, and TF-IDF logistic regression performs better than SAE classifiers in overall AUC-ROC: 0.917 vs. 0.888 for JumpReLU."*
- R3: *"the classification baseline ordering—TF-IDF + LR outperforming SAE + LR is insufficiently discussed (Section 4.3)."*
- AC: *"...and TF-IDF."*

What exists in `tfidf_lr_summary.json` is exclusively classification: `classification_auc_roc` (0.9169 vs 0.8882, TF-IDF wins 30/46), `classification_auc_pr`, Wilcoxon tests, and a `supplementary_correlation` block. `grep -E 'off_target|monospecif'` across `tfidf_lr_baseline.py` returns **zero matches**.

**Specifically missing.**

| Missing | Why |
|---|---|
| Off-target c-negative specificity for the top n-gram | No correlation-based off-target instrument exists for any baseline |
| Monospecificity — how many of the 46 codes that n-gram tracks | Never computed for baselines |
| Held-out-only evaluation | The existing run is `n_notes=50000` with 5-fold CV; the SAE grounding headline is shards 281–311. Different split, so not "same protocol" |
| Matched selection rule | `best_tfidf_feature` is selected and scored on the same data; the SAE's latent is selected on train and audited on held-out |
| Categorization of the n-gram | The AC's actual criterion — disease term, drug name, or template token? |

The n-gram identity is already stored per code as `best_tfidf_feature` in `per_code_comparison.csv`, so the first four are CPU work on the existing TF-IDF matrix.

**Why this is winnable.** TF-IDF with 10,000 supervised n-gram features is a *ceiling on how much surface signal exists in the text*, not a rival interpretability method. Reframed that way, its classification win becomes the expected result, and the audit comparison is where we have something to say.

**GPU:** none.

---

## A4. Random-matched directions

**What it is.** Generate 18,432 random directions — the same count as our SAE dictionary — drawn to match the statistics of real activations, then run the entire pipeline on them as if they were SAE features: project every token, threshold, max-pool per note, select the best direction per code by correlation, audit, explain, categorize.

"Matched activation statistics" has two components, both needed:

1. **Covariance-matched** — sample from `N(0, Σ_act)` rather than isotropic, so directions sit in the same geometry as real activations. One Cholesky on a token subsample.
2. **Sparsity-matched** — a per-direction threshold set at the quantile that reproduces the SAE's firing density, so the comparison is against a *sparse* decomposition and not a dense projection. Free, applied post-hoc.

**Why it must be run.** This is the item the AC singles out and the one the rebuttal answered by argument instead of experiment:

- AC: *"The complete pipeline needs to be run on a matched non-learned baseline such as random directions with matched activation statistics"* — and pointedly, *"Experiments with GemmaScope is meaningful, but does not fill the aforementioned checks."*
- R1: *"they do not fully answer whether an SAE-like but non-learned sparse decomposition would also yield apparently interpretable ICD concordance after the same feature-selection procedure"* and *"Without such a target, ICD concordance risks becoming a weak form of self-confirmation."*

Our rebuttal predicted random directions would produce a flat concordance gradient. That prediction is probably right — but predicting it is exactly what the AC penalized.

**What it proves.** Two things at once. It is the **floor**: if 18,432 random directions ground and concord as well as our SAE, then the apparent structure is a product of searching many candidates against 46 codes plus an LLM explainer, and the method rather than the architecture is in question. And it **calibrates the search advantage**: reporting *max \|r\| over 18,432 matched random directions* against our 0.864 answers the obvious objection that we searched 18,432 candidates per code while diff-in-means gets exactly one.

**Implementation notes.**

- Must project **per-token, then pool.** Projection and max-pooling do not commute, so `raw_shard_ckpt/` (already pooled) cannot be reused. Read the centered shards directly.
- Give random the **same selection advantage** as the SAE — best-of-18,432 per code, chosen the same way. Anything less is rigged in our favour and a reviewer will see it.
- Run on **held-out shards only (281–311)**. That is the split every §4 headline number uses, and it cuts cost roughly tenfold.

**GPU:** ~1 L4-hour on held-out shards. The operation is arithmetically an SAE encode (`[tokens × 2304] @ [2304 × 18432]`), which is why full-corpus would cost ~9 hours.

---

## A5. PCA directions

**What it is.** Eigen-decompose the activation covariance, treat the principal components as a dictionary, project and pool exactly as in A4, select the best component per code, audit.

**Why it must be run.** AC: *"and preferably PCA or ICA."* PCA is a non-learned, unsupervised decomposition of the same activation space. If the leading variance directions ground against ICD codes about as well as SAE features, then sparse dictionary learning is not contributing beyond ordinary second-order structure.

**Caveat to state in the paper.** PCA caps at `d_model = 2,304`, below our 18,432, so it does not match the SAE's count. Only random-matched and diff-in-means do. Learning an *overcomplete* dictionary is itself part of the SAE's case, and should be argued rather than hidden.

**GPU:** minutes — eigendecomposition is seconds, and the projection reuses A4's code path with 8× fewer directions.

---

## A6. ICA directions

**What it is.** FastICA on a token subsample to recover statistically independent components, then the same project → pool → select → audit path.

**Why it must be run.** AC: *"and preferably PCA or ICA."* ICA is the closest classical analogue to what an SAE is supposed to do — recover independent latent sources — while being far simpler and non-learned in the SAE sense. It is the most informative of the two classical controls for that reason.

**GPU:** ~1 hour CPU for the fit, minutes for projection. Same 2,304 cap as PCA.

---

## A7. Explanation + 5-way categorization on baseline directions

**What it is.** For each baseline's selected direction per code, extract top-activating note contexts, have the LLM write an explanation, assign one of the five categories (`clinical_concept` / `clinical_vocabulary` / `structural_pattern` / `general_language` / `noise`), and run the concordance judge against the code — the identical pipeline used on SAE features in §3.5.

**Why it must be run.** This is the **only item that answers the question the AC actually asked**: *"asking which method better separates disease signal from treatment logic and template artifacts."* That is a statement about the semantic content of what each method finds. No grounding statistic can answer it. The AC also names the constraint directly — *"same explanation budget"* — meaning per-method parity in how many features get explained.

R1's first compelling example is the same test in different words: *"SAE features separate disease concepts from medication, procedure, monitoring, documentation-template, or comorbidity proxies better than simpler baselines."*

**The non-vacuous framing.** Taken naively the AC's criterion is trivially won — diff-in-means has one direction per code, so it has nowhere to put "treatment" separately from "disease." The well-posed version is about **category purity**: does a per-code direction's top contexts mix diagnosis mentions, drug names, and template tokens in one direction (mixed category profile), while an SAE feature's contexts are single-category? Run identically on both sides, that is a fair comparison and a meaningful result either way.

**Budget.** 46 features per method. Scoped to the decisive three (A1, A2, A4) that is **138 features**; all five methods is **230**. For reference the original tier sweep was **964** features (280 strong / 100 weak / 484 non-grounded / 100 dead) — a different analysis serving the concordance gradient and Table 10, which needed the negative tiers. So this is roughly **14–24% of the original spend, not a multiple of it.** The SAE side is also 46 (its top latent per code), giving a 46-vs-46 comparison throughout.

**GPU:** light — selective context extraction only, plus LLM credits.

---

# Block B — Post-hoc computations needed for the results

## B1. Matched-grounding comparison

**What it is.** Instead of comparing one specificity number per method, plot or tabulate specificity *as a function of* on-target grounding — per-code scatter, or a comparison restricted to codes where both methods reach comparable \|r\|.

**Why it is needed.** Specificity and grounding are coupled: a direction that barely correlates with its own code cannot show much off-target leakage either. Our own diff-in-means pilot made this concrete, with on-target \|r\| of 0.12 versus 0.57 between methods — bare medians across that gap are meaningless. Since diff-in-means and the probe are expected to *out-ground* the SAE (they see the labels), the entire comparison rests on this control. Without it, the AC's *"which method better separates..."* cannot be answered honestly in either direction.

It also pre-empts R1's scoping objection: *"A feature that is specific to one of 46 evaluated ICD-9 codes above a threshold is not necessarily globally monospecific or clinically atomic."*

---

## B2. Off-target specificity aggregation per method

**What it is.** For each method, per code: correlate the code's selected feature against every *other* code, restricted to notes that are **negative** for its own code, then report the specificity ratio (on-target \|r\| / mean off-target \|r\|) and the count of significant off-target associations.

**Why it is needed.** AC: *"the same off-target diagnostic."* R1: *"off-target ICD/proxy diagnostics."* This is the primary quantitative axis on which the SAE can win, and the c-negative restriction is what makes it fair — without it, genuine comorbidity (a diabetes feature firing on diabetes-positive notes that also carry renal codes) masquerades as non-specificity. Our existing 6.4× figure is an *ablation* off-target measure; this is its correlational counterpart and applies to methods that have no causal arm.

---

## B3. Monospecificity profile per method, per threshold

**What it is.** For each method and each \|r\| threshold, count how many selected features are monospecific (exactly one associated code), oligospecific (2–3), or polyspecific (4+).

**Why it is needed.** Our headline monospecificity claim — the progression from 30.1% at \|r\|>0.1 to 100% at \|r\|>0.6 — is currently reported for SAEs only, so it reads as a property of features in general rather than a property of *SAE* features. Computing it for diff-in-means, the probe, and random-matched turns it into a comparative result. R1 asked for exactly this scoping (*"specificity claims should be scoped to the evaluated ICD panel"*), and it is one of the three axes on which the necessity argument is decided.

---

## B4. Ablation post-hoc suite — merge and tabulate

**What it is.** Four analyses on the existing per-note ablation deltas: off-target ICD specificity, OLS residualization on note length and number of codes, effect-size calibration in nats, and section-local loss. Plus the mean-ablation replication as an alternate intervention.

**Why it is needed.** R1's Concern 1.4 lists five requirements verbatim: *"The paper should add or at least discuss matched positive/negative contrasts, alternate interventions, token/section-local loss, off-target ICD/proxy diagnostics, and absolute effect-size calibration."* **All five are already computed** — 6.4× on-target specificity, Cliff's δ 0.230 → 0.164 under residualization with 19/30 still significant, mean-ablation median δ 0.169 vs 0.162 for zeroing, and the effect magnitude at 0.086% of base loss. None appear in the paper; they exist only in the rebuttal and on the `feat/ablation-additional` branch.

R2's Weakness 3 is answered by the same material: *"Ablation measures language-model loss over the last 25% of notes, not diagnosis prediction or clinical tasks."*

**Effort:** zero new compute — merge the branch, tabulate, and report the nats magnitude honestly alongside Cliff's δ.

---

## B5. Multi-judge concordance and blind forced-choice retrieval — into the body

**What it is.** Two analyses already run: concordance reproduced under three independent judge families (Claude Sonnet 4.6 / GPT-4o / DeepSeek-V3, κ = 0.87–0.90), and a forced-choice retrieval test in which a judge blind to both the correlation and the pre-selected target must recover the code from a 9-option slate (hit@1 70–74% at \|r\|>0.1 rising to 94–99% at \|r\|>0.5, against an 11.1% chance floor).

**Why it is needed.** This is the one concern **all three reviewers raised**, and R3 states the mechanism most precisely:

> *"because the Grounding step already selects the top-associated code as the target for comparison, the gradient could simply reflect that stronger statistical correlations make the pairing more likely to be correct, rather than demonstrating independent semantic alignment"*

and

> *"the same LLM (Claude Sonnet) serves as feature explainer, feature categorizer, and concordance judge, creating a risk of systemic bias propagation."*

R2's Weakness 1: *"The same LLM is used for explanations, categorization, and judging. This may cause shared-model bias."* R1's version: *"A single Claude-based pipeline can reward its own abstraction level, phrasing, or broad clinical relatedness."*

The retrieval arm answers R3's mechanism *structurally* — the judge never sees the pairing, so the gradient cannot be an artifact of pairing correctness. The AC credited both experiments (*"the rebuttal did real work"*) but they remain rebuttal-only.

**Presentation note.** Lead §4.2 with the blind retrieval result and keep anchored concordance as the original/secondary. Three reviewers independently reading the anchored gradient as circular is a sign the presentation invites that reading.

**Effort:** zero new compute — merge from `origin/main`, tabulate.

---

## B6. Shuffled-explanation control alongside Table 11

**What it is.** Each feature's held-out contexts re-scored against a deliberately wrong explanation drawn from a different feature. Already run: fuzzing drops to 0.496 and detection to 0.522 against real scores of 0.932 and 0.961, a gap of ~0.44 at Wilcoxon p < 10⁻⁶, stable across all tiers including dead features.

**Why it is needed.** Table 11 currently shows fuzzing 0.938 and detection 0.962 uniformly across strongly grounded, weakly grounded, and dead features, with no tier separation — which reads as a failure of our own metric. The shuffled control reframes it: the scorer *is* sensitive to explanation content, it simply does not discriminate feature quality. That is the setup for the paper's actual point, that the internal metric doesn't separate and the external one does (85% → 98.6%). It also addresses R1's request for *"prompt/model robustness."*

**Effort:** zero new compute — merge from `origin/main`.

---

## B7. CHF subpattern decomposition — formal write-up

**What it is.** Show that a single ICD code resolves into multiple distinct SAE features: congestive heart failure (ICD-9 4280) yields 13 features at \|r\|>0.5 — four tracking the diagnosis string, nine tracking ejection-fraction measurements, furosemide dosing, systolic/diastolic subtype classification, and weight-monitoring instructions.

**Why it is needed.** This is R1's second compelling example, near-verbatim:

> *"An ICD diagnosis label decomposes into clinically meaningful internal subpatterns such as diagnosis mention, treatment, monitoring, complication, differential diagnosis, and ruled-out disease."*

It is also the **strongest necessity argument available**, because it is the one result no baseline can produce even in principle: diff-in-means, the probe, and TF-IDF give exactly one direction per code, so decomposition is not expressible in their frame. It should be a subsection with its own table, not a sentence — currently it appears nowhere in the paper and only in the rebuttal.

Pair it with the PARTIAL subtype data (among 180 PARTIAL verdicts at \|r\|>0.3, 101 describe treatment or medication patterns and 32 describe monitoring or diagnostic markers), which reframes PARTIAL from a weak match into the mechanism of decomposition.

**Effort:** recomputable from `top_associations.csv` and `concordance_results.csv`; no new compute.

---

# Summary

| Block | Items | New GPU |
|---|---|---|
| A — experiments to run | A0–A7 | **~1 L4-hour total** (A4; A5/A6 ride along) |
| B — post-hoc computations | B1–B7 | none |

Four of the seven Block B items (B4, B5, B6, B7) are **already computed** and sitting on unmerged branches or in the rebuttal. Branch consolidation is a prerequisite for the rewrite, not a cleanup task to defer.
