# EMNLP §3.5 Concordance: Rebuttal Package (v2)

**Purpose.** Everything needed to answer the three reviewers on the concordance section (§3.5) during the ARR author-response phase. This package covers concordance validation only. The causal-ablation replies (zero-ablation, the LM-loss readout, the vanilla-SAE ablation) belong to the co-author, and every place one surfaces is tagged `[ablation: co-author]` so the two packages line up. Prepared July 2026. Supersedes v1.

---

## 0. What changed from v1

Three real problems in v1:

1. **v1 assumed a revised PDF that does not exist during rebuttal.** The ARR author response is text only. No PDF upload, no external links ([ARR Authors Guidelines](https://aclrollingreview.org/authors)). v1's replies kept pointing reviewers to "(Revised PDF: §3.5, Table R1...)" as if the changes were already on the page. A reviewer who clicks that and finds nothing reads it as careless. So in v2 every result is an inline markdown table inside the reply, and manuscript edits are written as commitments for the committed/camera-ready version, not as things a reviewer can go and read now. New experiments that answer a reviewer's question directly are allowed in the response, and Tables R1/R2 are exactly that, so we open with them.

2. **v1 answered only two of the three reviewers.** The missing one, GA3f, is also the most negative (Overall 2, "Resubmit next cycle"; Soundness 2) and raises the two hardest points. A full reply to GA3f is now §5, Post A.

3. **Reviewers are addressed by ID (GA3f / YeZF / v4ZK), not "Reviewer 1/2,"** which removes the attribution guesswork from v1.

---

## 1. TL;DR

All three reviewers land on the same objection to §3.5: a single Claude pipeline writes the explanations, categorizes them, and judges them (shared-model bias); the headline metric is a lenient YES+PARTIAL with a low exact-YES (22.4 / 30.4 / 42.4%); and the grounding step hands the judge the very code it then confirms (circularity). Two new analyses answer all three at once.

- **Table R1, multi-judge replication** of the concordance table on the verbatim original prompt, scored by three independent model families (Claude Sonnet 4.6, GPT-4o, open-source DeepSeek-V3).
- **Table R2, discriminative forced-choice retrieval.** An independent judge, blind to the correlation statistics and to the pre-selected target, has to recover each feature's grounded ICD code from a slate of unrelated distractors, scored by exact-match hit@1 against a fixed chance floor.

The result is judge-invariant (three families agree at Cohen κ = 0.87 to 0.90), the retrieval hit@1 sits 6 to 9× above the 11.1% chance floor and climbs with grounding even when the judge cannot see the statistics or the target, and the judge picks a hard-negative code roughly 0% of the time.

Beyond the tables, v2 narrows the paper's claims to the wording GA3f proposed, repositions the contribution as a hypothesis-generating clinical-audit method (the framing GA3f's own limitations comment asks for), and accepts several genuine gaps (single seed, no clinician adjudication, no non-learned null yet) openly rather than talking around them.

---

## 2. New results

### Table R1. Concordance under three independent judges (verbatim original prompt)
Same prompt (byte-identical across all 380 grounded features) and same targets as the published Table 2. Only the judge changes. Concordance = (YES+PARTIAL)/N; exact-YES = YES/N. Thresholds |r|>0.1 / >0.3 / >0.5.

| Judge (family) | Concordance | exact-YES | UNKNOWN |
|---|---|---|---|
| Claude Sonnet 4.6 (original, Table 2) | 85.0 / 94.6 / 98.6 | 22.4 / 30.4 / 42.4 | 11 / 8 / 1 |
| GPT-4o (OpenAI, independent) | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 | 0 |
| DeepSeek-V3 (open source, independent) | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 | 0 |

Sonnet re-run through OpenRouter reproduces the paper (exact-YES 22.6 vs 22.4%), which confirms the setup is faithful.

### Table R2. Discriminative retrieval (blind, forced-choice)
The judge picks 1 of 9 (grounded code, plus 7 statistically-unrelated cross-organ-system prevalence-matched distractors, plus "none"), blind to |r| and to the target. Chance floor = 1/9 = 11.1%.

| Judge (family) | hit@1 (>0.1 / >0.3 / >0.5) | "none" rate | hard-neg picked |
|---|---|---|---|
| GPT-4o (OpenAI) | 74.2 / 94.3 / 98.6 | 23.2% | 1.8% |
| Claude Sonnet 4.6 (Anthropic) | 71.1 / 90.4 / 95.8 | 27.1% | 0.0% |
| DeepSeek-V3 (open source) | 70.0 / 90.4 / 94.4 | 25.5% | 0.0% |

Inter-judge agreement (n=380): all three picked the same code 85.5% of the time; unanimous hit@1 66.1%; at least 2 of 3 recover 72.6%; Cohen κ on the picked code = 0.90 / 0.90 / 0.87 ("almost perfect"); κ on hit@1 = 0.86 / 0.83 / 0.80.

### What the numbers mean (stated honestly)
- **Judge-invariance answers the shared-model concern.** Three families, one of them open-source, agree at κ near 0.9.
- **exact-YES is judge-dependent** (22 to 34%). DeepSeek, which is independent, matches Sonnet at about 24%; GPT-4o commits to YES more often, about 34%. The low exact-YES is therefore not a Sonnet artifact, and we do not claim it is.
- **(YES+PARTIAL) can be inflated by leniency** (DeepSeek reaches 96% through heavy use of PARTIAL). Both YES-based metrics are fragile, so we lead with the forced-choice hit@1, which is exact-match, floor-anchored, and judge-invariant.
- **The gradient is not circular.** hit@1 rises with grounding while the judge is blind to |r| and to the target, and the "none" rate falls from 23% to 1.4% as grounding strengthens (weakly-grounded features are correctly recognized as non-matching).
- **UNKNOWNs are a formatting issue, not semantics.** They occur only for Sonnet (0 for GPT-4o and DeepSeek) and trace to one under-described code, V4986 ("Do not resuscitate status"). Excluding them from the denominator shifts exact-YES by under 1 point; we report the rate, and fixing the V4986 description removes them.

---

## 3. Reviewer-by-reviewer concern map (concordance scope)

Legend: **[A]** answered by new work or an existing artifact; **[C]** conceded with justification; **[F]** framing or wording fix for the final version; `[ablation: co-author]` out of this package.

### Reviewer GA3f (Overall 2, Soundness 2; most negative, calls the issues "fixable")
| # | Concern | Status |
|---|---|---|
| MC1.1 | ICD codes are noisy administrative anchors, not clinical ground truth | **[F][C]** narrower claim plus limitations wording |
| MC1.2 | Scope "specific" to the 46-code panel; give prevalence, group sizes, co-occurrence, threshold sensitivity, split/seed stability, multiple-testing family map | **[A]** threshold and monospecificity done; **[C]** single seed conceded; **[F]** prevalence/co-occurrence/FDR family clarified |
| MC1.3 | The LLM judge is not an independent validator; report YES-only, PARTIAL subtypes, hard negatives, prompt/model robustness, a blinded clinician set | **[A]** Tables R1/R2 (κ near 0.9, hard-negs about 0%, YES-only, retrieval replaces PARTIAL); **[C]** clinician adjudication offered, not run |
| MC1.4 | Ablation shows diagnosis-conditioned loss relevance, not mechanistic faithfulness | `[ablation: co-author]` |
| **MC2** | **Why do the SAE analysis at all? Show it reveals something raw activations, lexical, TF-IDF, and probes miss; add a matched non-learned sparse-decomposition null** | **[F]** reposition as an audit method (GA3f's own suggested framing); **[A]** GemmaScope, keyword-absent recall, and the shuffled control already bound this; **[C]** non-learned null offered, deployment-decision demo scoped out |

### Reviewer YeZF (Overall 2.5 borderline, Soundness 3)
| # | Concern | Status |
|---|---|---|
| W1 | One LLM explains, categorizes, and judges | **[A]** Tables R1/R2 |
| W2 | YES+PARTIAL vs low exact-YES; report separately | **[A][F]** exact-YES in Table R1 plus floor-anchored hit@1 |
| W3 | Ablation is LM loss, not diagnosis; vanilla not ablated | `[ablation: co-author]` |
| W4 | Keyword baseline weak; TF-IDF+LR (0.917) beats SAE (0.888) | **[F]** reframe as interpretability-first; unsupervised, within 0.03 of the supervised ceiling |
| W5 | Reproducibility (MIMIC access, Claude reliance) | **[A][C]** §10 pins model IDs and configs; MIMIC access is inherent and stated |
| S1 | Clarify "test-set explained variance" (Table 5) against "test split never accessed" | **[F]** reconcile the split terminology (author to confirm) |
| S2 | Soften "direction mismatch rather than distributional shift" | **[F]** reword to "consistent with; diagnostics rule out a simple mean shift" |

### Reviewer v4ZK (Overall 3 Findings, Soundness 2.5; most favorable)
| # | Concern | Status |
|---|---|---|
| T1 | Circularity: the concordance gradient may conflate grounding quality with alignment | **[A]** blind forced-choice retrieval (Table R2) |
| T2 | One LLM in three roles; wants a second judge (e.g. GPT-4o) | **[A]** three families, κ near 0.9 |
| T3 | Ablation is JumpReLU only, not vanilla | `[ablation: co-author]` |
| T4 | "General-purpose / wherever paired text and labels exist" shown in one domain | **[F]** soften to "generalizable in principle; demonstrated clinically; cross-domain is future work" |
| E1 | TF-IDF+LR beats SAE, under-discussed | **[F]** same reframe as YeZF W4 |
| E2 | Single model (Gemma-2-2B), single seed | **[C]** conceded and justified; second-seed grounding replication offered |

---

## 4. ARR reply mechanics

- **Text only. No PDF. No external links** ([ARR Authors Guidelines](https://aclrollingreview.org/authors)). Markdown renders in the response box, so tables, bold, and `$LaTeX$` all work; reproduce every number inline. There is no "the figures live in the revised PDF" option here, because there is no revised PDF. Manuscript edits are commitments for the committed version.
- **Small new experiments that answer a reviewer's question are allowed** in the response. Tables R1/R2 qualify, so they lead.
- **Back-and-forth is limited** (about two turns per thread), so the first reply has to be complete on its own.
- **Structure and tone.** Use headers that state the answer rather than restate the question ("Concordance is judge-invariant," not "Is the judge biased?"). Thank the reviewer for something specific. Group the shared concern into one "Response to all" comment and cross-reference it. When we agree, write "we will clarify," not "we were wrong." When we disagree, say so plainly and point the AC to the evidence. Concede real gaps openly; it reads better to the meta-reviewer than stonewalling.
- **Layout.** One "Response to all reviewers" comment carrying Tables R1/R2, then one tight per-reviewer reply (GA3f longest). Each reply opens on that reviewer's biggest concern.

---

## 5. Ready-to-paste replies

### Post 0. Response to all reviewers
> We thank all three reviewers. The concern they share, that a single Claude pipeline writes, categorizes, and judges the explanations while the grounding step pre-selects the code the judge then confirms, is a fair one, and it prompted two new analyses. Both are reported inline below (and will be added to §3.5 and the appendix in the final version). Thresholds are |r|>0.1 / >0.3 / >0.5.
>
> **Table R1. Concordance under three independent judge families (only the judge changes; verbatim original prompt, all 380 grounded features):**
>
> | Judge | Concordance (Y+P)/N | exact-YES |
> |---|---|---|
> | Claude Sonnet 4.6 (original) | 85.0 / 94.6 / 98.6 | 22.4 / 30.4 / 42.4 |
> | GPT-4o (OpenAI) | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 |
> | DeepSeek-V3 (open-source) | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 |
>
> **Table R2. Discriminative forced-choice retrieval, hit@1.** An independent judge, blind to |r| and to the pre-selected target, recovers the grounded ICD code from a shuffled slate of 9 (grounded code, 7 statistically-unrelated cross-organ-system prevalence-matched distractors, and "none"). Chance = 11.1%.
>
> | Judge | hit@1 | hard-negative picked |
> |---|---|---|
> | GPT-4o | 74.2 / 94.3 / 98.6 | 1.8% |
> | Claude Sonnet 4.6 | 71.1 / 90.4 / 95.8 | 0.0% |
> | DeepSeek-V3 | 70.0 / 90.4 / 94.4 | 0.0% |
>
> The three families agree at Cohen κ = 0.87 to 0.90 on the exact code selected. Retrieval runs 6 to 9× the chance floor and rises with grounding while the judge cannot see the statistics, and the abstention ("none") rate falls from 23% at |r|>0.1 to 1.4% at |r|>0.5. Together these make the concordance signal judge-invariant and non-circular. The per-reviewer replies build on these tables. We also use this response to narrow our central claims and to reposition the paper as a hypothesis-generating clinical-audit method; the reply to Reviewer GA3f gives the detail.

---

### Post A. Reply to Reviewer GA3f
> We thank the reviewer. Both concerns, claim calibration and the question of why clinical SAE analysis is worth doing, changed how we present the work, and where we agree we have adopted the reviewer's own wording.
>
> **Claim calibration. We narrow the central claim to the wording the reviewer proposed.** The headline claim now reads: some domain-trained SAE features are concordant with ICD-associated clinical semantic patterns and show diagnosis-conditioned loss relevance under the tested interventions. We drop all language calling features "validated clinical concept units," "faithful clinical mechanisms," or "monosemantic disease features."
>
> - **ICD as a noisy anchor.** Agreed. Throughout the paper we now describe ICD-9 codes as noisy, administrative, admission-level anchors shaped by coding practice, comorbidity, under-coding, documentation style, and label co-occurrence, and we state that concordance tests alignment with ICD-associated patterns, not exact clinical concept identity.
> - **Scope of "specific."** Agreed. We rescope "specific" to the evaluated 46-code panel ("panel-specific," never "globally monospecific" or "atomic"). We already report threshold sensitivity (grounding recomputed at |r|>0.1 to 0.5) and a monospecificity sweep (at |r|>0.5, how many surviving grounded latents associate with exactly one panel code). The final version adds per-code prevalence and positive-group sizes, the code co-occurrence structure, and a statement that BH-FDR is applied once across the full d_sae × 46 correlation matrix (one hypothesis family). On split/seed stability our results are single-seed. We state this as a limitation, note that the effect sizes are large (best |r| = 0.86) and the GemmaScope gap is more than tenfold, which makes a seed artifact unlikely, and we offer a second-seed grounding replication if the reviewer considers it decisive.
> - **The judge is not an independent clinical validator.** We rebuilt this analysis (Tables R1/R2 in the common response). Three independent judge families, one of them open-source, agree at κ = 0.87 to 0.90; YES-only is reported; the PARTIAL subjectivity is removed by an exact-match retrieval metric; hard-negative ICD labels are added and picked roughly 0% of the time; model robustness and two-prompt robustness are shown. We are explicit that κ measures inter-model consistency, not human-validated correctness. A blinded clinician adjudication is the one check that does not share the pipeline's failure mode. We did not run it, but we can produce a small (about 50-feature) blinded sheet during the discussion period.
> - **Ablation framing** is handled in our co-author's response; we adopt "diagnosis-conditioned loss relevance," not mechanistic faithfulness.
>
> **Why clinical SAE analysis is worth doing.** We agree the paper should not assume that SAE interpretability is settled and merely transfer it, and that the clinical setting is a validation problem. We reposition the contribution as a hypothesis-generating clinical-audit and validation method, which is the framing the reviewer recommends in the limitations comment. The object is the concordance protocol, using external structured labels the explanation pipeline never sees, not the bare existence of ICD-associated features. Three results already in the paper speak to whether the SAE shows something simpler methods miss, and we bring them forward:
> - Domain training is necessary; this is not "any sparse basis." GemmaScope, a learned but non-domain SAE, yields about 54 grounded latents against 610 to 675 for the domain-trained SAEs at |r|>0.3. A matched learned-SAE comparison already shows the effect is domain-specific, not a generic property of sparse-coding clinical text.
> - Beyond lexical. Our keyword-absent recall analysis shows the top SAE latent fires on positive notes that contain no matching keyword, so the feature is not a surface string match that a keyword or TF-IDF method would trivially reproduce.
> - Not self-confirmation. The shuffled-explanation control returns the chance null (about 0.51), and the blind forced-choice retrieval (judge cannot see the statistics) runs 6 to 9× chance, so the concordance is not an artifact of naming a feature from its top contexts and then re-finding the same field.
> - On TF-IDF+LR beating the SAE, we agree and reframe rather than contest. Classification is a supervised task where TF-IDF is expected to win. The SAE's contribution is a label-free, sparse, human-legible decomposition usable as an audit unit, and it lands within about 0.03 AUC of the supervised ceiling without ever seeing labels.
>
> We also concede the two strongest asks are beyond this paper: a concrete deployment or debugging decision, and decomposing an ICD label into internal sub-patterns (diagnosis mention, treatment, monitoring, ruled-out). We scope the contribution to method plus validation and flag these as the natural next step. On the specific request for a matched SAE-like null, we can add a non-learned random-sparse-dictionary control (random decoder directions, matched L0, identical feature-selection): GemmaScope bounds this from the learned side, and the random-dictionary run would bound it from the non-learned side. We share the reviewer's reading of Shukla et al. (2026) that clinical SAE work is still exploratory, and we now position the paper that way.

---

### Post B. Reply to Reviewer YeZF
> We thank the reviewer; these points drove the two new analyses (common response, Tables R1/R2).
>
> **Shared-model bias (W1) and YES vs YES+PARTIAL (W2).** Concordance is now judged by three independent families, GPT-4o, Claude Sonnet 4.6, and open-source DeepSeek-V3, and we report exact-YES separately (Table R1). exact-YES turns out to be judge-dependent: DeepSeek, which is independent, matches Sonnet at about 24%, while GPT-4o commits more, about 34%. So the low exact-YES is not a Sonnet artifact, and conversely (YES+PARTIAL) can be inflated by leniency (DeepSeek reaches 96%). Because the YES/PARTIAL boundary is subjective, we add a metric that does not depend on it, the forced-choice retrieval hit@1 against an 11.1% chance floor (Table R2), which is judge-invariant (κ near 0.9). UNKNOWNs occur only for the original Sonnet judge (0 for the others), trace to one under-described code (V4986, DNR status), and are excluded from the denominator (effect under 1 point; rate reported). A blinded clinician set is added to the limitations and can be run on a small sample if the reviewer considers it essential.
>
> **Keyword baseline and TF-IDF+LR beating the SAE (W4).** We agree the keyword baseline is limited and that TF-IDF+LR wins on AUC-ROC (0.917 vs 0.888). We reframe rather than contest. Classification is a supervised task where TF-IDF is expected to win; the SAE's contribution is a label-free, human-legible decomposition. An unsupervised feature set that lands within about 0.03 AUC of a classifier trained on the labels, while producing interpretable features, is a property worth reporting, not a shortfall. We revise the framing to say so.
>
> **Reproducibility (W5).** MIMIC-IV needs credentialed access, which is inherent to the data, and we state it plainly. Everything else is pinned: all model IDs are given via OpenRouter (`openai/gpt-4o`, `anthropic/claude-sonnet-4.6`, `deepseek/deepseek-chat`), the verbatim concordance prompt is byte-identical to the published run (verified on all 380 features), and the retrieval slate construction, seed, and parser are specified. No note text reaches any judge, so no PHI leaves the workspace.
>
> **"Test-set explained variance" vs "test split never accessed" (S1).** Thank you; this wording is genuinely confusing and we will fix it. The two phrases refer to different splits. Table 5's "test-set EV" is SAE reconstruction explained variance on a held-out activation shard, disjoint from the SAE's training tokens. "The MIMIC test split was never accessed" refers to MIMIC's official downstream evaluation partition, which we never use for supervision. We will rename the Table 5 column to "held-out (eval-shard) explained variance" and add a sentence separating the two splits, so there is no appearance of leakage.
>
> **"Direction mismatch rather than distributional shift" (S2).** We will soften this. The diagnostics rule out a simple mean shift; they do not by themselves establish direction mismatch as the mechanism. We will rephrase to "consistent with a direction mismatch; our diagnostics rule out a simple distributional (mean) shift but do not isolate the mechanism."

---

### Post C. Reply to Reviewer v4ZK
> We thank the reviewer; this critique led us to redesign the concordance validation (common response, Tables R1/R2).
>
> **Circularity (T1).** We removed it. The judge is now blind to |r_pb| and to the pre-selected target, choosing from a shuffled slate (grounded code, 7 statistically-unrelated cross-organ-system distractors, and "none"). hit@1 still rises with grounding (74, 94, 99%; Table R2), but this now reflects two independently derived signals, statistical grounding and blind semantic retrieval, converging, rather than a judge confirming a code it was handed. The abstention rate falls from 23% to 1.4% as grounding increases. We present this as convergent validity and note that explanation sharpness co-varies with grounding.
>
> **A second independent judge (T2).** Done, and extended to three families (GPT-4o, Sonnet, open-source DeepSeek-V3); κ = 0.87 to 0.90 on the exact code chosen (Table R2). The explanations remain Sonnet-generated, and we are explicit that we establish independence of the judge, not of the explainer.
>
> **The "general-purpose" claim (T4).** Agreed, and softened. The method is in principle applicable wherever paired free text and structured annotations exist, but we demonstrate it in one domain, clinical NLP with ICD-9. We reword the abstract and §1 to present cross-domain generality as a hypothesis and future work, not a demonstrated result.
>
> **TF-IDF+LR ordering (E1).** It needs discussion and we give it (see the reply to Reviewer YeZF). Classification is a supervised task where TF-IDF is expected to lead; the SAE's value is label-free interpretability, landing within about 0.03 AUC of the supervised ceiling without seeing labels.
>
> **Single model, single seed (E2).** We concede both. The results are on Gemma-2-2B, single seed. The multi-judge analysis speaks to robustness of the judge, not of the underlying SAE, and we now state that scoping directly. The large effect sizes (best |r| = 0.86) and the tenfold-plus GemmaScope gap make a seed artifact unlikely, and we offer a second-seed grounding replication if the reviewer considers it decisive. Cross-model generalization we flag as future work.
>
> **Vanilla vs JumpReLU ablation (T3)** is addressed in our co-author's response.

---

## 6. Unaddressed and accepted concordance-scope points

Every concordance-scope point, with whether the current package fully answers it and, where it does not, whether we rebut, partly answer, or accept, plus the justification that keeps an accepted point from reading as a fatal gap. `[co-author]` marks the ablation points, out of scope here.

| # | Reviewer, point | Fully answered now? | Disposition and justification |
|---|---|---|---|
| 1 | GA3f MC1.1, ICD as noisy anchor | No (v1 left it as framing) | **Accept, reword.** Cheap and builds credibility; use the reviewer's own wording. |
| 2 | GA3f MC1.2, prevalence and group sizes | No | **Accept, add a supp table.** Data already exists (`per_code_summary.csv`); just report it. |
| 3 | GA3f MC1.2, code co-occurrence | No | **Accept, report the matrix.** Cheap, and it pre-empts "specificity is confounded by comorbidity." |
| 4 | GA3f MC1.2, threshold sensitivity | **Yes** | Grounding and monospecificity already computed at r>0.1 to 0.5; surface it. |
| 5 | GA3f MC1.2, split/seed stability | **No** | **Accept as a limitation, offer a second seed.** Real gap (also v4ZK E2). Justify: |r|=0.86 and a tenfold GemmaScope gap make a seed artifact unlikely. This is the most technically real of the accepted gaps; do not wave it off. |
| 6 | GA3f MC1.2, multiple-testing family map | No | **Accept, clarify.** One BH-FDR family across the d_sae × 46 matrix; one sentence. |
| 7 | GA3f MC1.3, YES-only, PARTIAL subtypes, hard-neg, robustness | **Yes** | Tables R1/R2; the strongest part of the response. |
| 8 | GA3f MC1.3, blinded clinician adjudication | **No** | **Accept, offer a small blinded set.** The only non-LLM check, and honestly the thing most likely to move GA3f. Offer about 50 features; author decision (§8). |
| 9 | GA3f MC2, matched non-learned (SAE-like) null | **No** | **Partly rebut, offer.** GemmaScope (learned, non-domain) and the shuffled control bound it; a random-sparse-dictionary null is the exact control requested and is cheap. Currently not run; say so. |
| 10 | GA3f MC2, SAE beats raw-activation / probes / lexical as an audit unit | Partial | **Partly rebut.** Keyword-absent recall and the GemmaScope gap are concrete "beyond baseline" evidence, but we do not show the SAE beating a raw-activation probe head to head. Scope the claim; flag the probe comparison as future work. |
| 11 | GA3f MC2, ICD decomposition into sub-patterns; deployment-decision demo | No | **Accept, scope out.** Beyond this paper. GA3f frames these as things a paper "could" show, not as requirements. |
| 12 | YeZF S1, Table 5 "test-set EV" vs "test split never accessed" | No | **Accept, reconcile wording.** Reads as leakage if ignored, so it is the highest-priority factual fix. Needs the author to confirm the actual split semantics (§8). |
| 13 | YeZF S2, soften "direction mismatch vs distributional shift" | No | **Accept, soften.** Diagnostics rule out a mean shift only; one sentence. |
| 14 | YeZF W4, v4ZK E1, TF-IDF+LR beats SAE | Partial | **Reframe, not rebut.** Interpretability-first; unsupervised, within 0.03 of supervised. Do not claim the SAE should win at classification. |
| 15 | YeZF W5, reproducibility (MIMIC, Claude) | Partial | **Accept the inherent part, pin the rest.** MIMIC credentialing is unavoidable; state it. The reproducibility score is unlikely to move without open data. |
| 16 | v4ZK T4, "general-purpose" over-claim | No | **Accept, soften.** One-domain demonstration; generality is a hypothesis. Cheap. |
| 17 | v4ZK E2, GA3f, single model (Gemma-2-2B) | No | **Accept as a limitation.** Cross-model is future work; concordance rates are model-specific and we say so. |
| 18 | GA3f, v4ZK, prompt robustness (systematic paraphrase sweep) | Partial | **Accept.** Two very different prompts agree; a systematic sweep is not done, and we note it. |
| 19 | Optional, about 75% of hit@1 explanations share a surface keyword with the code | Partial | **Accept, offer a keyword-masking control.** Pre-empts "the judge is string-matching." Expected under genuine concordance, but a sharp reviewer will probe it. |
| n/a | GA3f MC1.4, YeZF W3, v4ZK T3, ablation framing / LM-loss / vanilla | n/a | `[co-author]`. Their package should adopt "diagnosis-conditioned loss relevance" and either ablate vanilla or argue transfer. |

**Reading of the list.** Items 1 to 4, 6, 7, 13, 14, and 16 are cheap or already in hand. The real open gaps are #5 (seed), #8 (clinician), #9 (non-learned null), and #12 (Table 5 wording). Of these, #12 must be fixed because it can look like leakage, and #8 and #9 are the two offers most likely to move GA3f from "resubmit" toward "findings." The rest is framing.

---

## 7. Manuscript edit checklist (for the committed / camera-ready version, not the rebuttal)

Commitments, staged for camera-ready. None can appear in a rebuttal PDF, because there is none.

1. Narrow the central claim to GA3f's wording; remove "validated concept units / faithful mechanisms / monosemantic disease features" everywhere.
2. Reframe §1 and the abstract as a hypothesis-generating clinical-audit and validation method; soften "general-purpose."
3. Add Table R1 (three-judge YES/PARTIAL panel) and Table R2 (three-judge blind retrieval), plus a small figure; make hit@1 the primary concordance measure and reword "rises with |r|, so it validates alignment" to convergent-validity phrasing.
4. Table 2 footnote: exact-YES reported excluding UNKNOWN (Sonnet-only format failures on V4986); state the UNKNOWN rate.
5. Fix the Table 5 wording to "held-out (eval-shard) explained variance"; add one sentence separating it from the MIMIC downstream test split (item 12).
6. Soften the GemmaScope "direction mismatch" sentence (item 13).
7. Describe ICD codes as noisy administrative admission-level anchors (item 1).
8. Add per-code prevalence and group sizes, the co-occurrence structure, and the BH-FDR family statement (items 2, 3, 6).
9. Reframe the TF-IDF/AUC discussion (interpretability-first).
10. Bring keyword-absent recall and the tenfold GemmaScope gap into §4/§5 as the beyond-baseline and domain-specific evidence for MC2.
11. Expand the limitations (§9 below).
12. Data hygiene: add the V4986 ("DNR status") description to the ICD keyword file.

---

## 8. Author decisions

1. **Clinician adjudication (#8).** Run a small blinded set of about 50 features, or offer only in the limitations? This is the single most credibility-moving item for GA3f. Recommend offering it in the rebuttal and running it if the AC or reviewer signals it is pivotal.
2. **Non-learned random-sparse-dictionary null (#9).** Commit to run within the discussion window, or argue that GemmaScope and the shuffled control already bound it? Recommend offering to run it; it is cheap and GA3f names it directly.
3. **Second-seed grounding replication (#5).** Feasible before the response deadline? Even one extra seed turns a conceded limitation into a strength.
4. **Table 5 split semantics (#12).** Confirm the exact meaning of "test-set EV" so the reconciliation sentence in Post B is correct. This blocks that reply.
5. **TF-IDF framing (#14).** Confirm the interpretability-first positioning, rather than trying to improve the SAE-classifier numbers.
6. **Keyword-masking control (#19).** Optional. Decide whether to pre-empt the "judge is string-matching" objection.
7. **Coordinate with the co-author** so the ablation replies (GA3f MC1.4, YeZF W3, v4ZK T3) adopt "diagnosis-conditioned loss relevance" and cover the vanilla-SAE ablation; reviewers raised these in the same threads.

---

## 9. Honest limitations (state these in the paper)

- ICD-9 codes are noisy, administrative, admission-level anchors; concordance tests alignment with ICD-associated patterns, not exact clinical concept identity.
- "Specific" is scoped to the 46-code panel, not a claim of global monospecificity or clinical atomicity.
- Table R2 uses cross-organ-system distractors, so it establishes organ-system-level discrimination, not fine-grained code identity.
- About 75% of hit@1 explanations share a surface keyword with the code (expected under genuine concordance, but noted).
- κ measures inter-model consistency, not human-validated ground truth; no clinician adjudication was performed.
- Explanations remain Sonnet-generated; the judge is independent, the explainer is not.
- Single model (Gemma-2-2B), single seed; concordance rates are LLM-specific; cross-model and cross-seed generalization is untested.
- No non-learned matched null and no raw-activation-probe head-to-head yet. The method is framed as a hypothesis-generating audit tool that needs external validation, clinician review, and subgroup and site-level checks before any clinical use.

---

## 10. Reproducibility

- Model IDs (all via OpenRouter): `openai/gpt-4o`, `anthropic/claude-sonnet-4.6`, `deepseek/deepseek-chat` (DeepSeek-V3).
- Verbatim-prompt concordance (Table R1): the exact original `CONCORDANCE_PROMPT`, byte-identical to the published run (verified on all 380 features); same verdict parser.
- Retrieval (Table R2): slate = grounded (argmax) code, plus 7 distractors that are (i) statistically unrelated to the feature (|r|<0.05), (ii) in a different ICD-9 chapter than the grounded code, and (iii) prevalence-matched, plus "none"; shuffled, seed = feature id. Metric = hit@1 (exact-match) against chance 1/9.
- Code and configs (repo): `modal_app/arm0_eval.py`, `modal_app/retrieval_eval.py`, `src/mech_interp_research/concordance_multi_judge.py`; configs `configs/arm0_panel.yaml`, `configs/retrieval_panel.yaml`.
- Inputs read only `concordance_results.csv`, `correlation_matrices.npz`, and the code descriptions (plus the ICD CSV for prevalence). No note text is sent to any judge, so no PHI leaves the workspace.
- MIMIC-IV needs credentialed PhysioNet access, which is inherent to the dataset and stated openly.

---

## 11. Artifacts (figures and tables, in `~/Downloads`)

- `concordance_final_results.png`: both tables, the reviewer-concern mapping, and limitations (consolidated figure, for camera-ready).
- `concordance_yespartial_panel.png`: Table R1 standalone.
- `retrieval_panel_3judges.png`: Table R2 standalone.
- `retrieval_gpt4o_results.png`: single-judge retrieval breakdown.
- `retrieval_gpt4o_verification.xlsx`: row-level audit (prompt, GT code, judge output, |r|, verdict).

None of these images can be attached to the ARR response, which is text only. They belong in the committed/camera-ready PDF. Reproduce all numbers as inline markdown tables in the response, per §4 and §5.
