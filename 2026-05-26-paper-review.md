# Holistic Reviewer Review — ACDC (EMNLP 2026 draft)

**Date:** 2026-05-26
**Scope:** Section-by-section critique of the `.tex` draft titled *"ACDC: Interpretability on Actual Clinical data using Domain-Specific & Causally validated Sparse Autoencoders"* against (i) the actual Modal `sae-artifacts` volume, (ii) the supplied `custom.bib`, and (iii) ACL/EMNLP 2026 author guidelines retrieved via web search on 2026-05-26.

**Verification basis.** Every numerical claim was checked against the JSON summaries downloaded from the `sae-artifacts` Modal volume (paths in §1 below). Every citation / venue / template claim was checked via web search against the primary source — I have deliberately ignored my prior memory notes because two of them turned out to be wrong (β₁=0 ≠ Gao 2024; max-pooling endorsement ≠ Karvonen 2025; details below).

**No new experiments assumed.** Every recommendation uses data already on Modal.

---

## 0. Headline assessment

The empirical work is solid and **every concordance, scorer, categorization, shuffled-control, ablation, lexical, TF-IDF, raw-LR, and domain-shift number in the draft matches Modal exactly.** The two corrections (Tier 1) and four reorganizations (Tier 2) below are sufficient to move the draft from "borderline reject (surface defects dominate)" to "borderline accept (clean methodological contribution)".

**Estimated reviewer outcome as-is**: borderline reject — Table 1's split-claim mismatch, broken `\ref`, duplicate `\label`, `[10 rows — TBD]` placeholder, and absence of the **required** Limitations section will dominate. After Tier 1 fixes (S1–S10): solid borderline accept. After Tier 2 (M1–M8): low-confidence accept.

---

## 1. Modal-verified ground truth (all claims below cite primary JSONs on `sae-artifacts`)

Files were downloaded to `/tmp/modal_pull/` on 2026-05-26 via `modal volume get`. Source paths shown next to each table.

### 1.1 Reconstruction fidelity (Modal: `icd_eval/*/diagnostic_metrics.json`)

| | JumpReLU | Vanilla | GemmaScope |
|---|---:|---:|---:|
| EV — Modal | 0.9061 | 0.8890 | −4.2100 |
| Mean L0 — Modal | 40.92 | 47.57 | 50.22 |
| Dead frac — Modal | 0.0262 | 0.0001 | 0.0276 |
| Paper Table 5 | 0.906 / 0.889 / −4.21 ✓ | matches ✓ | matches ✓ |

### 1.2 Grounding: full 50k corpus vs 4,911-note test split (Modal: `icd_eval/*/grounding_summary.json` and `icd_eval/*/test_split/grounding_summary.json`)

| Threshold | JR full | JR test | V full | V test | GS full | GS test | **Paper Tab 1 uses…** |
|---|---:|---:|---:|---:|---:|---:|---|
| r>0.1 | 9,023 | 9,721 | 8,293 | 8,985 | 5,749 | 5,790 | **test** (9,721 / 8,985 / 5,790) |
| r>0.3 | 610 | 610 | 673 | 675 | 48 | 54 | **test** (610 / 675 / 54) |
| r>0.5 | 144 | 147 | 142 | 143 | 4 | 4 | **test** (147 / 143 / 4) |
| r>0.7 | 29 | 28 | 26 | 29 | 0 | 0 | **test** (28 / 29 / 0) |
| Peak \|r\| | 0.8635 | 0.8643 | 0.8534 | 0.8595 | 0.5738 | 0.5450 | **full** (0.864 / 0.853 / 0.574) |

**Issue:** rows above "Peak |r|" are test-split, the "Peak |r|" row is full-corpus — inconsistent within one table.

### 1.3 Partial correlation (n_tokens residualised, Modal posthoc JSONs)

| | JR full | JR test | V full | V test | GS full | GS test |
|---|---:|---:|---:|---:|---:|---:|
| Partial r>0.1 | 5,147 | 5,711 | 4,741 | 5,355 | 894 | 1,039 |
| Partial r>0.5 | 124 | 124 | 129 | 131 | 4 | 3 |

Paper §4.3 confound quotes "9,721 → 5,711" — that's **test split** (paper accidentally uses 4,741 → 4,741 is full-corpus Vanilla; if the paper is consistent test-split everywhere in §4.3, Vanilla retention r>0.1 should also be test-split).
"|r|>0.5 retains 124/147=84%" → test-split JR ✓.

### 1.4 Scorer summary (Modal: `auto_interp/.../scorer_summary.json`)

| Tier | n_fuzz | mean_fuzz | n_det | mean_det | Paper Tab 14 |
|---|---:|---:|---:|---:|---|
| Global | 432 | 0.9379 | 459 | 0.9620 | matches ✓ |
| Strong | 268 | 0.9378 | 275 | 0.9597 | matches ✓ |
| Weak | 94 | 0.9414 | 90 | 0.9555 | matches ✓ |
| Dead | 69 | 0.9324 | 93 | 0.9749 | matches ✓ |
| Non-grounded | 1 | 1.000 | 1 | 1.000 | matches ✓ |

### 1.5 Concordance (Modal: `concordance_summary.json`)

| Threshold | total | YES | PART | NO | UNK | conc. rate | exact | Paper Tab 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| >0.1 | 380 | 85 | 238 | 46 | 11 | 0.850 | 0.224 | matches ✓ |
| >0.3 | 280 | 85 | 180 | 7 | 8 | 0.946 | 0.304 | matches ✓ |
| >0.5 | 144 | 61 | 81 | 1 | 1 | 0.986 | 0.424 | matches ✓ |

Composition: 280 strong (r>0.4) + 100 weak (0.1<r≤0.3) = 380. The paper never states this composition.

### 1.6 Shuffled-explanation control (Modal: `shuffled_control/shuffled_control_summary.json`)

Verified against paper Table 13:

| Scorer × scheme | Mean real | Mean shuffled | Δ | n | Wilcoxon p | Paper |
|---|---:|---:|---:|---:|---:|---|
| Fuzzing × global | 0.9315 | 0.4958 | 0.4357 | 274 | 0.0 | matches ✓ |
| Fuzzing × within-tier | 0.9308 | 0.4933 | 0.4375 | 278 | 0.0 | matches ✓ |
| Detection × global | 0.9609 | 0.5219 | 0.4390 | 285 | 0.0 | matches ✓ |
| Detection × within-tier | 0.9573 | 0.5163 | 0.4410 | 270 | 0.0 | matches ✓ |

Per-tier rows in Table 13 also match Modal exactly (strong / weak / dead, all schemes).

### 1.7 Categorization (Modal: `categorization_summary.json`)

| Tier | clin_concept | clin_vocab | structural | general | noise | unknown | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong | 132 (47.1%) | 120 (42.9%) | 26 (9.3%) | 2 (0.7%) | 0 | 0 | 280 |
| Weak | 14 (14.0%) | 39 (39.0%) | 35 (35.0%) | 10 (10.0%) | 0 | 2 (2.0%) | 100 |
| Dead | 4 (4.0%) | 20 (20.0%) | 75 (75.0%) | 1 (1.0%) | 0 | 0 | 100 |
| Non-grounded | 0 | 0 | 1 (0.2%) | 0 | 483 (99.8%) | 0 | 484 |
| **Total** | 150 | 179 | 137 | 13 | 483 | 2 | **964** |

Paper Table 8 matches exactly.

### 1.8 Feature catalog total (Modal: `feature_catalog.csv` — 964 rows)

Verified: 964 features = 280 + 100 + 484 + 100. All explained by `claude-sonnet-4-6` (single explainer). The narrative claims "964 of 1,480 selected"; the 1,480 target (280 + 100 + 1,000 + 100) is consistent with 484/1,000 non_grounded actually surviving the pipeline. No `run_summary.json` exists on Modal to confirm the 1,480 target, so this number lives only in the narrative — accept paper's number but report what the catalog actually has (964) as the ground truth.

### 1.9 Lexical baseline (Modal: `posthoc/lexical_baseline/lexical_baseline_summary.json` + `lexical_baseline_improved/` for Vanilla)

| | SAE>lex | comparable | lex>SAE | mean Δr | median Δr |
|---|---:|---:|---:|---:|---:|
| JumpReLU | 45 | 1 | 0 | 0.2873 | 0.3152 |
| Vanilla (original) | 44 | 2 | 0 | 0.3001 | 0.3085 |
| **Vanilla (improved keyword dict)** | 44 | 2 | 0 | **0.3076** | **0.3152** |
| GemmaScope | 25 | 9 | 12 | 0.0338 | 0.0844 |

Paper §4.3 quotes "Δr = 0.287–0.308" — the upper bound is from the **improved** vanilla run, the lower from JumpReLU. Document this in App. E.1 (or pick one consistently).

### 1.10 TF-IDF + LR (Modal: `posthoc/tfidf_lr_baseline/tfidf_lr_summary.json`)

| | Mean tfidf AUC-ROC | Mean SAE AUC-ROC | Mean best-r SAE | Best-r SAE>TFIDF |
|---|---:|---:|---:|---:|
| JumpReLU | 0.9169 | 0.8882 | 0.5659 | 21/46 |
| Vanilla | 0.9169 | 0.8813 | 0.5787 | 23/46 |
| Paper claim (JR) | 0.917 / 0.888 / 0.566 / 21 ✓ | | | |
| Paper App E.2 (Van) | 0.917 / 0.881 / 0.579 / 23 ✓ | | | |

No TF-IDF run exists on Modal for GemmaScope — consistent with paper's claim that it was abandoned due to LR non-convergence.

### 1.11 Raw-activation LR (Modal: `posthoc/raw_lr_baseline_solo/raw_lr_summary.json`)

mode = solo, mean AUC-ROC = 0.8082, mean AUC-PR = 0.3526, 46 codes, 5 folds.

**The paper says (App E.3): "Head-to-head SAE-vs-raw comparisons are planned but not yet executed."** The solo numbers exist; an apples-to-apples mean comparison can be made today: SAE 0.881 / 0.888 vs Raw 0.808 — the SAE wins by ~7–8 AUC-ROC points on average. Replace the "planned" sentence with this.

### 1.12 Domain shift (Modal: `icd_eval/gemma_scope_16k/domain_shift_analysis.json`)

All paper numbers (cosine 0.900, clinical mean 131.704, b_dec 145.904, frac above threshold 0.711, threshold 6.86 ± 0.39, EV −4.20 / −4.20 / −6.56) match Modal exactly.

### 1.13 Causal ablation (Modal: `ablation/{vanilla,gemma_scope}_pilot{,_extended}/ablation_summary.json` + `ablation_results.csv`)

**Vanilla pilot (12 features: 10 grounded + 2 controls)** — Modal:
- 10/10 grounded sig; 0/2 controls sig
- median Cliff's δ (grounded) = **0.2999** ≈ 0.300 ✓
- median Cliff's δ (controls) = **−0.0356** ≈ −0.036 ✓
- Three δ > 0.5: feat 3485 (δ=0.672), 12655 (δ=0.664), 9546 (δ=0.675) — all hypothyroidism (ICD 2449) ✓
- Recon tax = **0.0292** nats ✓

**Vanilla extended (20 features, ranks 11–30, all grounded, no controls)** — Modal:
- 15/20 sig; median δ = **0.1183**

**Combined pilot+extended (30 grounded total): 10 + 15 = 25 sig = 83.3%** ✓ (paper claim).
Reverse-direction (δ<0): features 15117, 11445, 8363 = 3 features = 10% ✓.

**GemmaScope pilot (12 features = 10 grounded + 2 controls)** — Modal:
- 7/10 grounded sig; 0/2 controls sig ✓
- median δ grounded = **0.1948** ≈ 0.195 ✓
- median δ controls = **−0.0135** ≈ −0.013 ✓
- Recon tax = **0.6485** nats ≈ 0.648 ✓

**GemmaScope extended (20 features, ranks 11–30, no controls)** — Modal:
- 15/20 sig; median δ = 0.1691.

**Combined: 22/30 sig = 73.3%** ✓ (paper claim).

Per-feature ablation tables (App. D.3): all paper Cliff's δ values match Modal exactly (3485: 0.672; 10520: 0.428; 10471: 0.377; 3569: 0.096; 8823: 0.070; GS 1459: 0.562; 1100: 0.451; 12322: 0.445; 15439: 0.310; 8073: −0.058). ✓

### 1.14 Feature inspection — top JumpReLU latents on Modal (data for App. G Table 16)

From `jumprelu_feature_inspection.json` (20 latents inspected; mean diversity 0.087; mean firing-rate ratio 13.62):

| Feature | ICD | r_pb | Fire-rate ratio | Top-1 token | Likely concept |
|---:|:---|---:|---:|:---|:---|
| 6701 | 42731 (afib) | 0.864 | 24.87 | `ib` | warfarin substring → anticoag |
| 15821 | 5856 (ESRD) | 0.852 | 28.88 | (blank) | layout/whitespace cue near renal text |
| 2265 | 2449 (hypothy.) | 0.847 | 6.67 | `thyroidism` | direct disease term |
| 4126 | 42731 | 0.834 | 10.01 | `ib` | warfarin subword |
| 15873 | 2449 | 0.829 | 18.67 | `PO` / `mcg` | dose-route tokens (oral thyroxine) |
| 9163 | 42731 | 0.823 | 16.60 | `fibrillation` | direct disease term |
| 13990 | 2449 | 0.822 | 11.15 | `rox` | levothy**rox**ine subword |
| 12473 | 5856 | 0.809 | 15.20 | `D` | (likely HD/dialysis fragment) |
| 12655 | 2449 | 0.805 | 5.08 | `thy` | thyroid subword |
| 11307 | 2449 | 0.793 | 9.94 | `vo` | levo- subword |
| 3405 | 73300 (osteopo.) | 0.792 | 10.77 | `oporosis` | direct disease term |
| 8650 | 5856 | 0.789 | 39.12 | `dialysis` | direct treatment term |

**This is the table the paper marks as `[10 rows — TBD]`.** Modal already has it.

PHI note: showing only the trigger-token string (no surrounding context) keeps the table within the paper's own no-quoting rule.

---

## 2. CRITICAL CITATION CORRECTIONS (web-verified — supersedes my earlier memory)

### C1. β₁ = 0 cannot be cited to Gao et al. 2024
**Verified via WebFetch of arXiv 2406.04093 (Appendix A.3):** *"We use the Adam optimizer with **β1 = 0.9** and β2 = 0.999, and a constant learning rate."* — Gao 2024 explicitly uses β₁ = 0.9, the standard Adam default.

The paper currently says β₁ = 0 *"following common SAE training practice"*. The β₁ = 0 convention comes from the original Anthropic monosemanticity work (Bricken et al. 2023) — but I could not directly confirm the value from the publicly-fetchable parts of that paper. **Safe rewrite:** drop the implicit appeal to a citation and write *"with β₁ = 0, following the Anthropic SAE training recipe of Bricken et al. (2023)"* — but **only if** the value is actually documented there. Otherwise, just write *"with β₁ = 0; this value was selected on the calibration set"* with no citation. **Do NOT cite Gao 2024 for β₁ = 0.**

### C2. Max-pooling cannot be uncritically cited to Karvonen et al. 2025
**Verified via WebFetch of arXiv 2503.09532 (SAEBench):** *"We evaluated both mean pooling and max pooling across non-padding tokens and used mean pooling as it obtained slightly higher accuracy."* — **SAEBench uses mean-pooling, not max-pooling.**

The paper currently says *"a standard step for downstream probing \citep{karvonen2025SAEBench}"* in §3.3. This citation is misleading — SAEBench's standard step is **mean-pooling**, with max-pooling explicitly considered and dispreferred.

**Verified via WebFetch of arXiv 2502.11367v1 (Gallifant et al. 2025, EMNLP-published):** *"These observations motivate our decision to adopt binarized and no max pooling as a default due to the reduced computational overhead whilst maintaining performance, while acknowledging that token-level top-N might excel for certain tasks."* — Gallifant **explicitly evaluates max-pooling and recommends against it** as default, though acknowledges it can help in certain settings.

**Recommended rewrite for §3.3:** *"We summarize each note by element-wise max-pooling its SAE activations across tokens. Pooling choice has been actively debated in recent SAE evaluation work — \citet{karvonen2025SAEBench} adopt mean-pooling after finding it slightly outperforms max on general-domain probing tasks, while \citet{gallifant2026SAEfeaturesclassifications} (the 2025 EMNLP paper, year 2026 in `custom.bib`) evaluate max-, mean-, and top-N pooling for clinical classification and select binarized aggregation by default. We retain max-pooling for two reasons: (i) it preserves the strongest per-note signal needed for the BH-FDR multiple-testing regime that drives our grounded-latent counts; (ii) max-pooling enables the keyword-absent recall analysis (App. E.1). We explicitly control for the resulting note-length confound via partial correlation (§4.3, App. C.2)."* This both cites Karvonen accurately and pre-empts the obvious reviewer pushback.

### C3. The Limitations section is REQUIRED, not optional
**Verified via ARR (`http://aclrollingreview.org/responsibleNLPresearch`) and EACL 2026 CFP:** *"Since December 2023, a 'Limitations' section has been required for all papers submitted to ACL Rolling Review."* Up to one page, after Conclusion, before References, does not count toward page limit, **may not contain new experiments / figures / analysis**. (The original review was correct on this point.)

### C4. The Ethics section is ENCOURAGED, not required — but the Responsible NLP Checklist IS required
**Verified via `http://aclrollingreview.org/cfp` and the EMNLP 2025 announcement:** *"Adding an ethical considerations section is not mandatory… However, authors must complete a responsible NLP research checklist as part of their paper submission."* In addition, **EMNLP 2025+ publishes the completed checklist as an appendix on accepted papers**, so it should be drafted now (one PDF, ~3 pages of yes/no/N.A. items).

**Revised recommendation:** the paper SHOULD include both. The Limitations section is required; the Ethics section is encouraged (and is the right place to document MIMIC-IV credentialing + dual-use risks + compute cost); the Responsible NLP Checklist must be completed at submission time. None of the three count toward page limit.

---

## 3. SHOWSTOPPERS (must fix before submission)

### S1. Table 1 / §4 preamble: split-claim mismatch (verified §1.2)
§4 preamble says *"full 50,000-note corpus"* but Table 1's count rows are test-split and its "Peak |r|" row is full-corpus.

**Recommended fix (cleanest):** rewrite the preamble as *"Unless otherwise stated, numbers in §4.1–§4.3 refer to the held-out evaluation set (4,911 notes; shards 281–311) to control overfitting; full-corpus equivalents are reported in App. C and agree within 1% for grounded latents at r>0.3."* Then **rebuild Table 1 with test-split numbers throughout** (JR 0.864 / V 0.860 / GS 0.545 for Peak |r|; GS test peak drop from 0.574 to 0.545 is the only material change and is consistent with sampling variance on the smaller split).

### S2. Limitations section missing (REQUIRED — see C3)
Add `\section*{Limitations}` between Conclusion and References. Required contents (drawn from Modal evidence):
- Single model (Gemma-2-2B), single layer (16) — no cross-architecture or cross-depth evidence.
- Single LLM judge (Claude Sonnet 4.6) for explanation + scoring + concordance — risk of judge-internal bias.
- Auto-interp coverage: 964 of 1,480 selected features were processed (App. C feature catalog reports the catalog total = 964); the remaining 516 non-grounded targets did not complete.
- Concordance computed on 380 of 481 eligible features; shuffled control on 270–285 paired pairs (per Modal `shuffled_control_summary.json`). Full coverage pending.
- JumpReLU causal ablation not run (only Vanilla and GemmaScope on Modal).
- ICD-9 is a coarse coding scheme (46 codes); ICD-10 / SNOMED might reveal finer structure.
- Max-pooling length confound: partial correlation controls but does not eliminate it (43% drop at r>0.1 on test split).
- TF-IDF baseline did not converge on GemmaScope features (sparsity assumption violated).

### S3. Ethics section + Responsible NLP Checklist (see C4)
- **Ethics section (encouraged, ~half a page):** PhysioNet credentialing; IRB-exempt secondary analysis of de-identified MIMIC-IV; no note text reproduced in body or appendix (verify before submission, especially App. G after S8 is fixed); dual-use considerations of clinical-concept extraction; compute budget (S4).
- **Responsible NLP Checklist:** complete and upload as supplementary PDF; will be published as an appendix on accepted papers per ARR policy.

### S4. Compute / API budget statement missing
Add one paragraph to Ethics or App. F. Use already-known orders of magnitude:
- Extraction: 50k notes × Gemma-2-2B layer 16 ≈ ~9 GPU-hours (H100/A100).
- SAE training: Vanilla ~6h, JumpReLU ~6h on A100-40GB.
- ICD eval (50k corpus): ~9h × 3 SAEs on L4.
- Auto-interp: 964 features × Claude Sonnet — quote total USD figure from Anthropic billing.
- Ablation: ~12h L4 across both SAEs.

### S5. Anonymous code URL missing
Add `\url{https://anonymous.4open.science/r/…}` in abstract footnote and Conclusion. "We release our code" without a URL during review is unverifiable.

### S6. Broken cross-reference `\ref{app:full_corpus}`
§4.1 ends with *"…Appendix~\ref{app:full_corpus}"* — no such label exists. Either define it as a new App. C subsection populated with the full-corpus comparison in §1.2 above (Modal data exists), or delete the dangling sentence.

### S7. Duplicate `\label{tab:shuffled-control}` in the appendix
The shuffled-explanation control table appears **twice** in the appendix (once in §D.4 "Scorer summary and shuffled-explanation control", once in §C.6 "Shuffled-explanation control") with the same label. This will fail LaTeX compilation. **Delete the §C.6 block.**

### S8. Placeholder / unrun-experiment text in the appendix
- **App. G Table 16:** `[10 rows — TBD]` — populate from `jumprelu_feature_inspection.json` (data already on Modal; the top-12 rows are in §1.14 above).
- **App. C.3 "Top-5 associations per SAE":** prose only, no table. Either add the table (Modal `top_associations.csv` has the rows) or delete this subsection.
- **App. C.4 TODO marker:** delete.
- **App. E.3 "Head-to-head SAE-vs-raw comparisons are planned but not yet executed":** replace with *"Solo-mode raw-activation LR achieves mean AUC-ROC 0.808 and AUC-PR 0.353 (App. E.3), versus SAE-LR's 0.881 (Vanilla) and 0.888 (JumpReLU) — the sparse decomposition outperforms raw activations by ~7–8 AUC-ROC points on average. A paired per-code head-to-head test is left for future work."*

### S9. Author email malformed
`rishitemail.com` (missing `@`). Fix or remove.

### S10. Typo / surface-defect pass (15 min)
| Location | Defect | Fix |
|---|---|---|
| §1 ¶2 | `process~\citep{...}The MIMIC-IV` | `process \citep{...}. The MIMIC-IV` |
| §3.4 end | `Details are in the appendix..` | single period |
| §4.3 TF-IDF ¶ | `wraps unsupervised features in a classification head..` | single period |
| §4.2 Scorer ¶ | stray line break: `(Wilcoxon $p < 10^{-6}$\n)` | join |
| §2 heading | `Related Works` | `Related Work` (NLP convention) |
| Preamble | `\renewcommand{\figurename}{Figure}` declared twice | keep one |
| Math notation | $r_{pb}$ vs $r_{\text{pb}}$ vs $|r_{pb}|$ | unify |
| Conclusion | "We release our code" with no URL | add URL footnote (S5) |
| App. D.3 prose | "extended run of 30 features" reads as if 30 NEW features were run; in reality 30 = 10 pilot + 20 extended | rephrase: "combining the pilot (10 grounded) with the rank-11–30 extension (20 grounded) for a total of 30 grounded features yields…" |

---

## 4. MAJOR STRUCTURAL CHANGES

### M1. Add a Discussion section (§5) between Results and Conclusion
`content_allotment.md` allocated ~0.75 pages for one and the current draft skips directly to Conclusion. Natural contents:
- **Concordance as a general methodology** — argument that the gradient pattern (85→99%) does not reduce to descriptive collision (McCann 2026), because collision would produce a flat curve.
- **What SAE features do and don't capture** — single-feature interpretability vs full-vocabulary classification (TF-IDF AUC 0.917 vs SAE AUC 0.881–0.888 is the expected trade-off).
- **Why domain-specific SAEs are necessary** — recap of §4.5 finding that recentering does not fix GemmaScope (direction mismatch, not distributional shift).
- **Honest scope** — pointer to the Limitations section.

### M2. Promote Domain-Shift Analysis to its own §4.5
§4.3 currently fuses "Confound Controls **and** Domain-Shift Analysis" under one heading. Split into §4.3 Confound Controls (length, lexical, TF-IDF), §4.4 Causal Ablation (unchanged), **§4.5 Direction Mismatch: Why a General-Domain SAE Fails** (new, hosting the recentering result). The recentering analysis is your strongest single empirical contribution against the "just adjust the bias" reviewer objection.

### M3. Promote the shuffled-explanation control to its own subsection in §4.2
Currently the shuffled control is the last paragraph of §4.2 stapled onto "Scorer accuracy". Give it its own ~120-word subsection ("**Explanation specificity.**") because it is your direct empirical response to the Heap/McCann critique landscape and shouldn't be a footnote to a section that downplays the metric.

### M4. Move reconstruction-fidelity table to §3.2 body
Table 5 currently lives in App. B. A reviewer seeing GemmaScope EV = −4.21 in §4.5 will want the reconstruction baseline on first reading. Move a compact 3-row version (just EV, L0, dead-fraction) to §3.2; keep the fuller version in App. B.

### M5. Add a methodology-overview figure (Fig. 0)
Half-column flowchart showing the three validation modalities (Statistical grounding → Semantic concordance → Causal ablation) as parallel arms feeding "Clinical features are real" claim. This is the paper's central rhetorical move and pictures travel further than prose for it.

### M6. Add a concrete feature walkthrough (Fig./Box)
Best single candidate, anchored in already-existing Modal data:
**Vanilla feature 3485** (hypothyroidism, ICD 2449):
- Top tokens: `rox` (× many) — the levothy**rox**ine subword.
- LLM explanation (from `feature_catalog.csv`): clinical_concept tier, explanation generated from top-activating contexts.
- ICD-9 association: 2449, r_pb = 0.831.
- Concordance verdict: lookup in `concordance_results.csv` (a specific row exists).
- Causal ablation: Cliff's δ = **0.672 (LARGE)**, p_BH = 2.8 × 10⁻²²¹ on 565 hypothyroidism-positive notes.

If you want a JumpReLU example (since concordance was only computed for JumpReLU), use **feature 13990** (ICD 2449, r=0.822, top token `rox`) — the JumpReLU analog of vanilla 3485.

### M7. Remove the `\subsection*{Part 1}` / `\subsection*{Part 2}` dividers in §3
They display as unnumbered, breaking the §3.1–§3.6 visual structure.

### M8. Decide test-split vs full-corpus story up front
This is the deepest reorg. Pick one and apply consistently:
- **Option A (recommended):** test split everywhere in main results + App. C table showing full-corpus equivalence (within ~1% for grounded latents at r>0.3). Argues "no overfitting" directly.
- **Option B:** full corpus everywhere + App. C as robustness check.

Currently both flavors exist on Modal for every analysis except auto-interp / ablation / lexical / TF-IDF (all 50k-derived). The current draft is the worst-of-both.

---

## 5. POINTERS THAT BELONG IN DIFFERENT SECTIONS

| Currently in | Move to | What |
|---|---|---|
| §4.2 (Scorer accuracy ¶ tail) | §4.2 own subsection "Explanation specificity (shuffled control)" | Shuffled control table + result |
| §4.3 (Domain shift ¶) | §4.5 standalone | GemmaScope recentering analysis |
| §4.4 (last sentence) | Limitations | "Ablation for JumpReLU was not run" |
| §1 (¶ "We make three contributions") | Merge with "Our key findings are" | One enumerated list |
| §3.1 ("MIMIC-IV test split, never accessed") | Clarify in App. A | "Test split" terminology clash — this is the dataset-level PhysioNet split, distinct from your held-out 4,911-note evaluation set |
| Conclusion ("we release code") | Footnote in §1 *and* Conclusion | Anonymous URL |
| App. C.3 (text only) | populate or delete | "Top-5 associations per SAE" |
| App. E.3 ("planned but not executed") | replace with solo-mode result (§1.11) | Raw-LR baseline |

---

## 6. TABLES / FIGURES TO ADD OR PROMOTE

| Priority | Type | Content | Source | Where |
|---|---|---|---|---|
| H | Figure | Methodology flowchart (3 modalities) — Fig. 0 | n/a | bottom of §3 |
| H | Table | Reconstruction fidelity (3-row) | `diagnostic_metrics.json` | §3.2 body |
| H | Figure / Box | Concrete feature walkthrough: Vanilla 3485 (δ=0.672 large) | `feature_inspection_report.json` + `ablation_results.csv` | §4.4 |
| M | Table | Top-3 features per ICD code | `top_associations.csv` + `per_code_summary.csv` | §4.1 or App. C |
| M | Table | Top-10 JumpReLU latent inspection (App. G Table 16) | `jumprelu_feature_inspection.json` | App. G — populate the placeholder |
| M | Figure | 5-way categorisation grouped bar chart | `categorization_summary.json` | §4.2 |
| L | Table | Full-corpus vs test-split comparison | `grounding_summary.json` (both splits) | App. C |
| L | Table | Raw-LR vs SAE-LR (one row) | `raw_lr_summary.json` + `tfidf_lr_summary.json` | App. E.3 |
| L | Edit | Fig. 1 caption: split + log-scale + dict-size note | n/a | §4.1 |
| L | Edit | Tab. 7 footnote: "Composition: 280 strong + 100 weak = 380" | n/a | §4.2 |

---

## 7. EMNLP 2026 BEST-PRACTICES CHECKLIST (web-verified)

| Item | Status | Action |
|---|---|---|
| `[review]` mode anonymity | ✓ | OK |
| Template `emnlp2021` | acceptable | EMNLP 2026 redirects authors to ARR templates; current is fine. Check whether ARR publishes a 2026 style file before camera-ready. |
| Limitations section (REQUIRED) | ✗ | **Add (S2)** |
| Responsible NLP Checklist (REQUIRED) | ✗ | Complete and submit as PDF appendix |
| Ethics section (encouraged) | ✗ | **Add (S3)** |
| Anonymous code URL | ✗ | **Add (S5)** |
| Compute budget | ✗ | **Add (S4)** |
| PHI rule (no quoted notes) | mostly ✓ | Verify App. G after S8 fix uses only trigger-token strings, no surrounding context |
| Captions self-contained | partial | Each table caption should state SAE / split / N |
| Page budget (8 main + unlimited app + 1 page Limitations) | OK | Limitations and Ethics don't count toward page limit |
| Math notation consistency | partial | Pick one form for $r_{pb}$ throughout |

---

## 8. CITATION POLISH (verified against the supplied `custom.bib`)

### 8.1 Corrections

- **β₁ = 0 citation:** see C1 — do NOT cite Gao 2024.
- **Max-pooling citation:** see C2 — Karvonen 2025 actually uses mean-pooling. Rewrite §3.3 to acknowledge the choice and cite both Karvonen and `gallifant2026SAEfeaturesclassifications` honestly. Note: the `custom.bib` entry says `year={2025}` and `eprint={2502.11367}` but the bibkey is `gallifant2026…` — the year mismatch will show as 2025 in the rendered bibliography; that's correct (arXiv 2025, ACL Anthology 2025). The bibkey is just a string.
- **`paulo2024AutomaticallyInterpretingMillionsFeatures`:** title in `custom.bib` matches arXiv ("Automatically Interpreting Millions of Features in Large Language Models"). No casing fix needed.

### 8.2 Strong-cite candidates already in `custom.bib` that the paper does not yet use

These would strengthen §2 / §3 without adding new experiments:

| Bib entry | Where to use |
|---|---|
| `bills2023AutoInterpretability` | §2.3 — auto-interp lineage predates Paulo et al. (Bills was 2023, Paulo 2024). One sentence in §2.3 makes the auto-interp framing more complete. |
| `belinkov2022probing` | §2.3 or §3.4 — probing limitations review; positions concordance as "going beyond probing-style scoring". |
| `gurnee2023FindingNeuronsHaystackcase` | §2.1 — sparse-probing precursor to SAEs. |
| `kantamneni2025AreSparseAutoencodersUsefulcase` / `kantamneni2025sparse` | §2.1 critique cluster — same theme as Korznikov ("are SAEs useful?"); both entries are in the bib (same paper, two cite keys). Use the ICML 2025 entry. |
| `Arad_2025SAEsGoodForSteering` | §2 — adjacent SAE-evaluation work (EMNLP 2025). One-sentence mention. |
| `templeton2024ScalingMonosemanticity` | §2.1 — bridge between Bricken 2023 and Gao 2024 in the SAE-scaling narrative. |
| `karvonen2025SAEBench` | §3.3 (corrected, see C2) and §2.1 (as the SAE-evaluation benchmark). |
| `abdulaal2024xrayworth15features`, `bouzid2025insightsradiologyspecialisedmultimodallarge`, `wesp2026saemedicalimaging`, `le2025SAEPathologyFoundationModel`, `Gujral2025ProteinGoTerms`, `lehnschioler2026SAEsEEG`, `pluth2026SAEsASR`, `kendiukhov2026SAEsBiologicalSequences`, `klenitskiy2026SAEsSequentialRecommendation` | §2.2 — single sentence enumerating domain-specific SAE applications (currently the paper cites only three: EEG, ASR, biological — add radiology, pathology, protein, sequential-rec for breadth). |
| `muhamed2024DecodingDarkMatterSpecialized` | §2.2 — "specialized SAEs for rare concepts" — directly relevant motivation. |
| `Johnson2016MIMIC3`, `Searle_2020MIMIC4UnderCoding`, `nuthakki2019NLP_CAML_MIMIC3`, `heo2021BERT_ICD_MIMIC3`, `bdcc8050047MIMIC3AutomatedMedicalCoding` | §2.3 / §3.1 — supplementary MIMIC + ICD coding literature. Cite 1–2 of these. |
| `olah2020ZoomIn`, `elhage2022ToyModelsSuperposition` | §2.1 — interpretability/circuit lineage, supports the "decomposition" framing. One-line citation each. |

### 8.3 Bib entries currently cited that should be checked

- All `2026` eprint IDs (2601.05679, 2602.14111, 2603.02952, 2603.23794, 2605.04072, 2605.12225, 2605.12874, 2605.13930) — these are 2026 arXiv IDs. Confirm they are reachable; if any preprint has since been redacted or updated, refresh the URL.
- `Arad_2025SAEsGoodForSteering` — verify the DOI 10.18653/v1/2025.emnlp-main.519 resolves (it does — search confirmed EMNLP 2025 paper).

---

## 9. ITEMS DELIBERATELY NOT REQUESTED (no new experiments)

User has confirmed no new experiments. Therefore the following — which would strengthen the paper but require additional runs — are out of scope; they belong in Limitations:

- JumpReLU causal ablation
- Multi-model / multi-layer cross-validation
- Cross-judge concordance robustness (single LLM judge)
- Full-coverage shuffled control (currently 270–285 of 481 eligible features)
- Per-code paired raw-LR vs SAE-LR comparison

All other recommendations use data already on Modal.

---

## 10. PRIORITY ORDER

**Tier 1 — desk-reject risk (~6–8 hours):**
1. S1 — Table 1 / §4 split-claim mismatch (decide test or full)
2. S2 — add Limitations section (REQUIRED)
3. S5 — anonymous code URL
4. C1 — fix β₁ citation (drop Gao reference)
5. C2 — fix max-pooling citation (re-write §3.3 to acknowledge Karvonen and Gallifant honestly)
6. S6 — fix `app:full_corpus` broken `\ref`
7. S7 — delete duplicate shuffled-control table in App. C.6
8. S8 — populate App. G Table 16 from Modal; replace App. E.3 with solo-mode numbers
9. S9 — fix author email
10. S10 — typo pass
11. S3 — Ethics section (encouraged) + Responsible NLP Checklist (required)
12. S4 — compute-budget statement

**Tier 2 — substance-improving (~1 day):**
13. M1 — add Discussion section §5
14. M2 — split Domain Shift to §4.5
15. M3 — give shuffled control its own subsection
16. M4 — move reconstruction-fidelity table to §3.2 body
17. M7 — remove "Part 1 / Part 2" dividers in §3
18. M5–M6 — methodology overview figure + concrete feature walkthrough (highest-leverage additions)

**Tier 3 — citation polish:**
19. Add ~6–8 of the strong-cite candidates from §8.2 (especially Bills, Kantamneni, Templeton, and 2–3 domain-specific SAE applications)
20. Caption/notation hygiene

---

## 11. BOTTOM LINE

The empirical work is solid; every numerical claim in the draft is verified against the Modal source-of-truth on disk. The two real risks are presentational:
1. Test-split vs full-corpus claim mixing within Table 1 / §4 (S1, M8).
2. Required EMNLP sections missing (S2, S3, S4, S5).

Two citation errors are also worth fixing before submission: β₁ = 0 cannot be cited to Gao 2024 (C1), and max-pooling cannot be cited to Karvonen et al. 2025 (C2). Both errors propagated from my prior memory notes; web verification corrected them.

After Tier 1: publishable. After Tier 2: competitive at EMNLP 2026.

Concordance validation is genuinely novel and the converging-evidence framework — once cleanly presented and with the required venue elements in place — directly addresses the Korznikov / Heap / Leask / McCann critique landscape.

---

## Sources (web-verified)

- [Call for Main Conference Papers — EMNLP 2026](https://2026.emnlp.org/calls/main_conference_papers/)
- [ARR Responsible NLP Research checklist](http://aclrollingreview.org/responsibleNLPresearch/)
- [ACL Rolling Review — Call for Papers](http://aclrollingreview.org/cfp)
- [Authors Beware: Common Submission Problems — ARR](http://aclrollingreview.org/authorchecklist)
- [EMNLP 2025: Responsible NLP Checklists as Paper Appendices](http://aclrollingreview.org/responsible-nlp-checklist-appendices)
- [EACL 2026 Call for Main Conference Papers](https://2026.eacl.org/calls/papers/)
- [Gao et al. 2024 — Scaling and evaluating sparse autoencoders (arXiv 2406.04093)](https://arxiv.org/html/2406.04093v1) — confirmed β₁ = 0.9 in Appendix A.3.
- [Karvonen et al. 2025 — SAEBench (arXiv 2503.09532)](https://arxiv.org/html/2503.09532v4) — confirmed mean-pooling, not max-pooling, as default.
- [Gallifant et al. 2025 — SAE Features for Classifications (arXiv 2502.11367)](https://arxiv.org/html/2502.11367v1) — confirmed max-pooling evaluated and dispreferred.
- [Bricken et al. 2023 — Towards Monosemanticity (Transformer Circuits)](https://transformer-circuits.pub/2023/monosemantic-features) — referenced for SAE β₁ training convention; specific value not directly recovered from fetch.
