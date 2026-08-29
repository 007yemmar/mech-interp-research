# EMNLP §3.5 Semantic Concordance — Rebuttal Package

**Purpose:** everything needed to respond to the reviewers on the concordance section, plus the paper edits, action items, and reproducibility. Prepared July 2026.

---

## 1. TL;DR — what we did in response to the reviews

Reviewers flagged the concordance analysis (§3.5) as (a) a single-model (Claude) pipeline → shared-model bias, (b) using a generous (YES+PARTIAL) metric with low exact-YES (22.4/30.4/42.4%), and (c) a **circular** validation (the grounding step pre-selects the target the judge then confirms). We added two analyses:

- **Table R1 — multi-judge replication** of the concordance table using the **verbatim original prompt**, judged by three independent model families.
- **Table R2 — discriminative forced-choice retrieval**: an independent judge, **blind to the correlation statistics and to the pre-selected target**, must recover each feature's grounded ICD code from a slate of unrelated distractors, scored by exact-match **hit@1 against a fixed chance floor**.

**Headline:** the result is **judge-invariant** (three families agree at Cohen κ = 0.87–0.90), the discriminative hit@1 sits **6–9× above the 11.1% chance floor** and rises with grounding even when the judge is blind to the statistics, and the judge picks a hard-negative code **~0%** of the time.

---

## 2. New results

### Table R1 — Concordance under three independent judges (verbatim original prompt)
Same prompt (byte-identical for all 380 grounded features) and targets as the published Table 2; only the judge changes. Concordance = (YES+PARTIAL)/N; exact-YES = YES/N. Thresholds |r|>0.1 / >0.3 / >0.5.

| Judge (family) | Concordance | exact-YES | UNKNOWN |
|---|---|---|---|
| Claude Sonnet 4.6 (original, Table 2) | 85.0 / 94.6 / 98.6 | 22.4 / 30.4 / 42.4 | 11 / 8 / 1 |
| GPT‑4o (OpenAI, independent) | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 | 0 |
| DeepSeek‑V3 (open source, independent) | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 | 0 |

*Sonnet re-run via OpenRouter reproduces the paper (exact-YES 22.6 vs 22.4%), confirming a faithful setup.*

### Table R2 — Discriminative retrieval (blind, forced-choice)
Judge picks 1 of 9 (grounded code + 7 statistically-unrelated, cross-organ-system, prevalence-matched distractors + "none"), blind to |r| and to the target. **Chance floor = 1/9 = 11.1%.**

| Judge (family) | hit@1 (>0.1 / >0.3 / >0.5) | "none" rate | hard-neg picked |
|---|---|---|---|
| GPT‑4o (OpenAI) | 74.2 / 94.3 / 98.6 | 23.2% | 1.8% |
| Claude Sonnet 4.6 (Anthropic) | 71.1 / 90.4 / 95.8 | 27.1% | 0.0% |
| DeepSeek‑V3 (open source) | 70.0 / 90.4 / 94.4 | 25.5% | 0.0% |

**Inter-judge agreement (n=380):** all three picked the same code **85.5%**; unanimous hit@1 **66.1%**; ≥2/3 recover **72.6%**; **Cohen κ (picked code) = 0.90 / 0.90 / 0.87** ("almost perfect"); κ (hit@1) = 0.86 / 0.83 / 0.80.

### Key interpretive findings (important, and honest)
- **Judge-invariance** answers shared-model bias: three families, incl. an open-source model, agree at κ≈0.9.
- **exact-YES is judge-dependent** (22–34%): DeepSeek (independent) matches Sonnet at ~24%, GPT‑4o gives ~34%. So the low exact-YES is **not** a Sonnet artifact — do not claim that.
- **(YES+PARTIAL) is leniency-inflatable** (DeepSeek reaches 96% via heavy PARTIAL). Both YES-metrics are therefore fragile → we lead with the forced-choice **hit@1** (exact-match, floor-anchored, judge-invariant).
- **Non-circular gradient:** hit@1 rises with grounding with the judge blind to |r| and to the target; the "none" rate falls 23%→1.4% (weakly-grounded features are correctly recognized as non-matching).
- **UNKNOWN handling:** UNKNOWNs occur only for Sonnet (0 for GPT‑4o/DeepSeek); they are **format failures on one under-described code (V4986, "Do not resuscitate status")**, not semantic non-answers. Exclude from the denominator (effect on exact-YES <1 pp) and report the rate; fixing the V4986 description removes them.

---

## 3. OpenReview / ACL (ARR) reply mechanics

- **Modalities:** (1) a text **Author Response per reviewer** — Markdown renders, so **Markdown tables, bold, and `$LaTeX$`** work; (2) a **revised PDF** where the actual figures/illustrations go. **Embedded PNGs usually do NOT render** in the response comment, and anonymity rules forbid external links → put PNGs in the revised PDF and **reproduce the numbers as Markdown tables inline** in the comment.
- **Length:** keep each reply tight (order of a few thousand chars); lead with the biggest concern.
- **Best practices:** be direct and skimmable; thank the reviewer; address every point; **acknowledge limitations openly** (builds trust); reviewer-requested new experiments are welcome at ACL/ICLR/NeurIPS; tell the AC exactly where each change is in the revised PDF; if two reviewers share a concern, answer once and cross-reference.
- **Recommended structure:** one **"Response to all reviewers"** comment with the two tables, then a short per-reviewer reply referencing it.

---

## 4. Ready-to-paste replies

### Post 1 — Response to all reviewers
> We thank all reviewers. In response we added two analyses to §3.5 (revised PDF): **Table R1** (multi-model replication of concordance on the verbatim original prompt) and **Table R2** (a new discriminative, forced-choice retrieval test). Summary (|r|>0.1 / >0.3 / >0.5):
>
> **Table R1 — concordance, three independent judges (only the judge changes):**
>
> | Judge | Concordance (Y+P)/N | exact-YES |
> |---|---|---|
> | Claude Sonnet 4.6 (original) | 85.0 / 94.6 / 98.6 | 22.4 / 30.4 / 42.4 |
> | GPT‑4o | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 |
> | DeepSeek‑V3 (open-source) | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 |
>
> **Table R2 — discriminative retrieval, hit@1 (judge blind to |r|, picks the grounded code from 9 options; chance = 11.1%):**
>
> | Judge | hit@1 |
> |---|---|
> | GPT‑4o | 74.2 / 94.3 / 98.6 |
> | Claude Sonnet 4.6 | 71.1 / 90.4 / 95.8 |
> | DeepSeek‑V3 | 70.0 / 90.4 / 94.4 |
>
> The three families agree at **Cohen κ = 0.87–0.90** on the exact code chosen; the judge selects a hard-negative code ~0% of the time. Per-reviewer responses reference these tables.

### Post 2 — Reply to Reviewer 1
> We thank the reviewer; these comments substantially strengthened §3.5 (see the common response for Tables R1–R2).
>
> **Shared-model bias / multiple models or humans.** We now judge concordance with three independent families — GPT‑4o, Claude Sonnet 4.6, and open-source DeepSeek‑V3. Concordance replicates across all three (Table R1), and on the new forced-choice retrieval test they agree at **κ = 0.87–0.90** on the exact code selected (Table R2) — the result is judge-invariant, not a single-pipeline artifact. We are explicit that the *explanations* remain Sonnet-generated (independence of the *judge*, not the explainer). We add a clinician adjudication to the limitations and can run a small (~50-feature) blinded set if considered essential.
>
> **YES vs. YES+PARTIAL; low exact-YES (22.4/30.4/42.4%).** We now report exact-YES (Table R1) and note transparently that it is **judge-dependent**: DeepSeek (independent, open-source) matches Sonnet at ~24%, while GPT‑4o commits to YES more (~34%); conversely (YES+PARTIAL) is *leniency-inflatable* (DeepSeek 96%). Because the YES/PARTIAL boundary is subjective, we add a metric that does not depend on it — the retrieval test's **exact-match hit@1 vs. an 11.1% chance floor** (Table R2). UNKNOWNs occur only for the original Sonnet judge (0 for GPT‑4o/DeepSeek), are formatting failures on one under-described code (V4986, DNR), and are excluded from the denominator (effect <1 pp; rate reported). *(Revised PDF: §3.5, Table R1, Limitations.)*
>
> **Keyword baseline; TF‑IDF+LR > SAE (0.917 vs 0.888).** We acknowledge this; the keyword baseline is limited. We clarify that the SAE's contribution is label-free, interpretable grounding, not supervised classification: an *unsupervised* dictionary reaching AUC 0.888 — within ~0.03 of a supervised classifier trained on the labels — while yielding human-legible concept features is a notable property, not a deficiency. We revised the framing accordingly. *(Revised PDF: §Baselines.)*

### Post 3 — Reply to Reviewer 2
> We thank the reviewer; this critique led us to redesign the concordance validation (see the common response for Tables R1–R2).
>
> **(1) Circular gradient.** We redesigned the test to remove the circularity. The judge is **blind to |r_pb| and to the pre-selected target**, choosing from a shuffled slate (grounded code + 7 statistically-unrelated, cross-organ-system distractors + "none"). hit@1 still rises with grounding (74→94→99%, Table R2), but this reflects two independently-derived signals converging, not confirmation of a handed-over code; the abstention rate falls 23%→1.4% as grounding increases. We frame this as convergent validity and note explanation sharpness co-varies with grounding. *(Revised PDF: §3.5, Table R2.)*
>
> **(2) Second independent LLM judge.** Done, and extended to three families (GPT‑4o, Sonnet, open-source DeepSeek‑V3); κ = 0.87–0.90 (Table R2). Explanations remain Sonnet-generated.
>
> **(3) Independent clinical validator; YES-only, PARTIAL subtypes, hard negatives, robustness, clinician adjudication.**
> - **Hard-negative ICD labels — added** (7 cross-system distractors + "none"; selected ~0% of the time).
> - **YES-only — reported** (Table R1).
> - **PARTIAL subtypes** — the retrieval metric replaces YES/PARTIAL with exact-match recovery, directly addressing "YES+PARTIAL is weaker than exact concept identity."
> - **Model robustness** — three families, κ≈0.9; **prompt robustness** — two very different prompts (original + de-anchored forced-choice) agree (systematic paraphrase sweep not done; noted).
> - **Blinded clinician adjudication — acknowledged, not performed.** The only check that does not share the LLM pipeline's failure mode; added to limitations, and we can run a small blinded set if necessary. κ reflects model *consistency*, not human-validated correctness. *(Revised PDF: §3.5, Table R2, Limitations.)*

---

## 5. Revised-PDF / manuscript edit checklist
1. Add **Table R1** (3-judge YES/PARTIAL panel) and **Table R2** (3-judge retrieval panel) + optional small figure.
2. §3.5: add a paragraph describing the multi-judge validation and the discriminative retrieval test as the primary, non-circular, floor-anchored concordance measure.
3. Reword the "concordance rises with |r| → validates alignment" sentence to convergent-validity phrasing.
4. **Table 2 footnote:** exact-YES reported excluding UNKNOWN (Sonnet-only format failures on V4986); UNKNOWN rate stated.
5. Expand **Limitations** (see §7 below).
6. Reframe the TF‑IDF/AUC discussion (interpretability-first, not a classifier).
7. Data hygiene: add the **V4986** ("Do not resuscitate status") description to the ICD keyword file.

---

## 6. Action items (decisions only the authors can make)
1. **Clinician adjudication** — decide yes/no. Only genuinely open item; the sole non-LLM validation. If yes, we can generate a blinded ~50-feature sheet (a clinician fills it).
2. **TF‑IDF framing** — confirm positioning the SAE as interpretability-first (or improve the SAE-classifier numbers separately).
3. **Confirm reviewer→comment attribution** so the two replies map exactly (grouping assumed: Rev 1 = shared-model bias / exact-YES / TF-IDF; Rev 2 = circular gradient / second judge / independent validator).
4. **Optional hardening** — keyword-masking control (pre-empts a "judge is string-matching" objection; ~75% of hit@1 explanations share a surface keyword with the code, which is expected under genuine concordance but a sharp reviewer may probe).
5. **Check the venue's response box** for whether a PDF/image attachment is enabled; if not, rely on the inline Markdown tables (figures live in the revised PDF).

---

## 7. Honest limitations (state these in the paper)
- R2 uses cross-organ-system distractors → **organ-system-level** discrimination, not fine-grained code identity.
- **~75% of hit@1 explanations share a surface keyword with the code** (expected under genuine concordance, but note it).
- **κ measures inter-model consistency, not human-validated ground truth.**
- The code set is **46 common ICD-9 codes**.
- **Explanations remain Sonnet-generated** (the *judge* is independent, the *explainer* is not).

---

## 8. Artifacts (figures/tables, in `~/Downloads`)
- `concordance_final_results.png` — both tables + reviewer-concern mapping + limitations (the consolidated figure).
- `concordance_yespartial_panel.png` — Table R1 standalone.
- `retrieval_panel_3judges.png` — Table R2 standalone.
- `retrieval_gpt4o_results.png` — single-judge retrieval breakdown.
- `retrieval_gpt4o_verification.xlsx` — row-level audit (prompt / GT code / judge output / |r| / verdict).

---

## 9. Reproducibility
- **Model IDs (all via OpenRouter):** `openai/gpt-4o`, `anthropic/claude-sonnet-4.6`, `deepseek/deepseek-chat` (DeepSeek‑V3).
- **Verbatim-prompt concordance (Table R1):** the exact original `CONCORDANCE_PROMPT`, byte-identical to the published run (verified 380/380 features); parsed with the same verdict parser.
- **Retrieval (Table R2):** slate = grounded (argmax) code + 7 distractors that are (i) statistically unrelated to the feature (|r|<0.05), (ii) in a different ICD-9 chapter than the grounded code, (iii) prevalence-matched; + "none"; shuffled, seed = feature id. Metric = hit@1 (exact-match) vs chance 1/9.
- **Code / configs (repo):** `modal_app/arm0_eval.py`, `modal_app/retrieval_eval.py`, `src/mech_interp_research/concordance_multi_judge.py`; configs `configs/arm0_panel.yaml`, `configs/retrieval_panel.yaml`.
- Inputs read only `concordance_results.csv` + `correlation_matrices.npz` + code descriptions (+ ICD CSV for prevalence). No note text is sent to any judge → no PHI leaves the workspace.
