# EMNLP §3.5 Concordance: Reviewer Rebuttals

Ready-to-paste ARR author responses. Text only, no external links (per ARR rules). Post the common response first, then the per-reviewer replies. Thresholds are |r|>0.1 / >0.3 / >0.5.

---

## Response to all reviewers

We thank all three reviewers. The concern they share, that one Claude pipeline plays every role and that the grounding step could be supplying the judge with the code it then confirms, is a reasonable one to raise, and it motivated two new analyses. Both are reported inline below and will be added to §3.5 and the appendix in the final version.

**Table R1. Concordance under three independent judge families (only the judge changes; verbatim original prompt, all 380 grounded features):**

| Judge | Concordance (Y+P)/N | exact-YES |
|---|---|---|
| Claude Sonnet 4.6 (original) | 85.0 / 94.6 / 98.6 | 22.4 / 30.4 / 42.4 |
| GPT-4o (OpenAI) | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 |
| DeepSeek-V3 (open-source) | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 |

**Table R2. Discriminative forced-choice retrieval, hit@1.** An independent judge, blind to |r| and to the pre-selected target, recovers the grounded ICD code from a shuffled slate of 9 (grounded code, 7 statistically-unrelated cross-organ-system prevalence-matched distractors, and "none"). Chance = 11.1%.

| Judge | hit@1 | hard-negative picked |
|---|---|---|
| GPT-4o | 74.2 / 94.3 / 98.6 | 1.8% |
| Claude Sonnet 4.6 | 71.1 / 90.4 / 95.8 | 0.0% |
| DeepSeek-V3 | 70.0 / 90.4 / 94.4 | 0.0% |

The three families agree at Cohen κ = 0.87 to 0.90 on the exact code selected. Retrieval runs 6 to 9× the chance floor and rises with grounding while the judge cannot see the statistics, and the abstention ("none") rate falls from 23% at |r|>0.1 to 1.4% at |r|>0.5. Together these show the concordance signal is judge-invariant and does not depend on the pre-selected target. The per-reviewer replies build on these tables. We also use this response to state the scope of our claims more precisely and to position the contribution as a validation method, which the reply to Reviewer GA3f develops.

---

## Reply to Reviewer GA3f

We thank the reviewer for a detailed and constructive read. We focus here on the two concerns raised, claim calibration and the value of clinical SAE analysis.

**Claim calibration.** We state the central claim at the precision the evidence supports: some domain-trained SAE features are concordant with ICD-associated clinical semantic patterns and show diagnosis-conditioned loss relevance under the tested interventions. We align the surrounding wording to this scope so that terms such as "concept unit" are not read as claims of validated clinical identity or global monosemanticity.

- **ICD as an external anchor.** We describe ICD-9 codes explicitly as noisy, administrative, admission-level anchors shaped by coding practice, comorbidity, under-coding, documentation style, and label co-occurrence, and we state that concordance measures alignment with ICD-associated patterns rather than exact clinical concept identity. This matches how the labels function in our method, and making it explicit removes any stronger reading.
- **Scope of "specific."** Our specificity results are defined relative to the evaluated 46-code panel, and we will say so directly so the term is not read as global monosemanticity. The supporting analyses the reviewer asks for are largely already in place: we report grounding across thresholds (|r|>0.1 to 0.5) and a monospecificity sweep showing how many grounded latents associate with exactly one panel code as the threshold rises. We add per-code prevalence and positive-group sizes, the code co-occurrence structure, and a note that BH-FDR is applied once over the full d_sae × 46 correlation matrix, that is, a single hypothesis family. As a further check that this specificity is not merely an artifact of code prevalence or broad clinical relatedness, the retrieval test in the common response draws prevalence-matched distractors from different organ systems: the judge recovers the grounded code well above the 11.1% chance floor and selects these hard negatives close to 0% of the time. This establishes specificity at the organ-system level, and it complements rather than replaces the prevalence, co-occurrence, and threshold analyses above.
- **Stability across seeds.** The grounding effect sizes are large (best |r| = 0.86) and the margin over GemmaScope is more than tenfold, so the associations are not plausibly a seeding artifact. We can add a second-seed replication of the grounding step to demonstrate this directly if the reviewer would find it useful.
- **Independence of the judge.** The common response adds two analyses that speak to whether one model in three roles can validate itself. Three independent judge families, one of them open-source, agree at κ = 0.87 to 0.90 on the exact code selected; YES-only is reported; the YES/PARTIAL boundary is replaced by an exact-match retrieval metric; hard-negative ICD labels are added and selected close to 0% of the time; and the results hold across models and across two very different prompts. We are explicit that κ measures agreement across models rather than agreement with a clinician. A blinded clinician adjudication is the one check outside the LLM pipeline entirely, and we can produce a small (about 50-feature) blinded sheet during the discussion period.
- **Ablation framing** is addressed in our co-author's response, which adopts diagnosis-conditioned loss relevance rather than mechanistic faithfulness.

**Why clinical SAE analysis is worth doing.** We agree the clinical setting is best treated as a validation problem rather than an assumed transfer of established SAE interpretability, and we position the contribution accordingly. The object is the concordance protocol, which tests explanations against external structured labels the pipeline never observes, not the bare existence of ICD-associated features. Several results already in the paper bear on whether the SAE adds something beyond simpler methods, and we bring them forward:

- Domain training is what produces the effect, and it is not a property of any sparse basis. GemmaScope, a learned but non-domain SAE, yields about 54 grounded latents against 610 to 675 for the domain-trained SAEs at |r|>0.3. This is a matched learned-SAE comparison, and the effect is domain-specific.
- The features are not surface lexical matches. Our keyword-absent recall analysis shows the top SAE latent fires on positive notes that contain no matching keyword, which a keyword or TF-IDF method would not recover.
- The signal is not self-confirmation. The shuffled-explanation control returns the chance null (about 0.51), and the blind forced-choice retrieval, where the judge cannot see the statistics, runs 6 to 9× chance. So the concordance does not reduce to naming a feature from its top contexts and then re-finding the same field.
- TF-IDF+LR leads on supervised AUC, which is the expected ordering: classification is supervised and the SAE features are label-free. The result of interest is that an unsupervised, human-legible decomposition lands within about 0.03 AUC of a classifier trained directly on the labels, and we present it that way.

Two of the reviewer's suggestions, a concrete deployment decision and decomposing a single ICD label into internal sub-patterns, are natural next steps rather than claims this paper makes, and we frame them as such. On the matched non-learned null the reviewer names, GemmaScope already bounds the effect from the learned, non-domain side; we can add a random-sparse-dictionary control (random decoder directions, matched L0, identical feature-selection) to bound it from the non-learned side as well. We share the reviewer's reading of Shukla et al. (2026) that clinical SAE work is still exploratory, and we position the paper as a validation method in that spirit.

---

## Reply to Reviewer YeZF

We thank the reviewer; the shared-model and metric points shaped the two new analyses (common response, Tables R1/R2).

**Multiple judges and YES vs YES+PARTIAL (W1, W2).** Concordance is now scored by three independent families, GPT-4o, Claude Sonnet 4.6, and open-source DeepSeek-V3, with YES-only reported separately (Table R1). The independent judges reproduce the concordance ranking and its gradient with grounding, and a strong independent judge, GPT-4o, assigns exact-YES up to 59.7%, so the original Sonnet figures are conservative rather than inflated. exact-YES is a deliberately strict metric: it discards genuine partial matches and therefore understates concept alignment. Because the YES/PARTIAL boundary is a judgment call in any case, we lead with a measure that does not depend on it, the forced-choice retrieval hit@1 against an 11.1% chance floor (Table R2), which is judge-invariant at κ near 0.9. UNKNOWN verdicts appear only for the original Sonnet judge (none for the others) and come from one under-described code (V4986, DNR status); excluding them shifts exact-YES by under a point, and we report the rate.

**Keyword baseline and the TF-IDF ordering (W4).** The keyword baseline is a deliberately simple lexical control; its role is to mark what surface features alone recover, and it serves that role. TF-IDF+LR leads on AUC-ROC (0.917 vs 0.888), which is the expected ordering: classification is supervised and the SAE features are label-free. The point of interest is that an unsupervised, interpretable decomposition comes within about 0.03 AUC of a classifier trained on the labels, and we frame the comparison that way.

**Reproducibility (W5).** MIMIC-IV needs credentialed access, which is inherent to the data, and we state it plainly. Everything else is pinned: the three judges (GPT-4o, Claude Sonnet 4.6, and open-source DeepSeek-V3) are all accessed through a single LLM provider, the verbatim concordance prompt is byte-identical to the published run (verified on all 380 features), and the retrieval slate construction, seed, and parser are specified. No note text reaches any judge, so no PHI leaves the workspace.

**"Test-set explained variance" and "test split never accessed" (S1).** These two terms name different splits, and we will make that explicit to prevent a leakage reading. Table 5's "test-set explained variance" is SAE reconstruction EV on a held-out activation shard, disjoint from the SAE's training tokens. "The MIMIC test split was never accessed" refers to MIMIC's official downstream evaluation partition, which we never use for supervision. We will rename the Table 5 column to "held-out (eval-shard) explained variance" and add one sentence distinguishing the two splits.

**"Direction mismatch" wording (S2).** We will state this more precisely. Our diagnostics rule out a simple distributional (mean) shift, and we will phrase the conclusion as consistent with a direction mismatch rather than asserting it as the established mechanism.

---

## Reply to Reviewer v4ZK

We thank the reviewer; the independence and circularity points led us to add a complementary test to the concordance validation (common response, Tables R1/R2).

**Circularity (T1).** The concern is that the concordance gradient could reflect the grounding step supplying the judge with its target. We test that possibility directly. In the added retrieval design the judge is blind to |r_pb| and to the target, choosing from a shuffled slate (grounded code, 7 statistically-unrelated cross-organ-system distractors, and "none"). hit@1 still rises with grounding (74, 94, 99%; Table R2), which shows two independently derived signals, statistical grounding and blind semantic retrieval, converging rather than one confirming the other. The abstention rate falls from 23% to 1.4% as grounding increases. We present this as convergent validity.

**A second independent judge (T2).** Added, and extended to three families (GPT-4o, Sonnet, open-source DeepSeek-V3); κ = 0.87 to 0.90 on the exact code chosen (Table R2). The explanations remain Sonnet-generated, and we are precise that the added independence is in the judge, not the explainer.

**Scope of the "general-purpose" claim (T4).** We will state the scope precisely: the method applies in principle wherever paired free text and structured annotations exist, and we demonstrate it in one domain, clinical NLP with ICD-9. We present cross-domain generality as a hypothesis and future work in the abstract and §1.

**The TF-IDF ordering (E1).** This ordering is expected, and we discuss it as such (see the reply to Reviewer YeZF): classification is supervised, the SAE features are label-free, and the unsupervised decomposition still lands within about 0.03 AUC of the supervised classifier.

**Single model and seed (E2).** Results are on Gemma-2-2B with a single seed. The multi-judge analysis addresses robustness of the judge; robustness of the underlying SAE is a separate axis, and the large grounding effect sizes (best |r| = 0.86) with a tenfold-plus margin over GemmaScope make a seed artifact unlikely. We can add a second-seed grounding replication to confirm this, and we flag cross-model generalization as future work.

**Vanilla vs JumpReLU ablation (T3)** is addressed in our co-author's response.
