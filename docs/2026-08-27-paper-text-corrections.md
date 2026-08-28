# Paper text corrections — Submission 15186 resubmission

Ordered by impact on the AC's open issue and the reviewers' stated grounds for the
2.5. Each item cites its source: `[paper L###]` = line numbers in the submitted PDF
(`Interp_csv__Concordance...pdf`, via `pdftotext -layout`); `[results/...]` = an
artifact on disk in this repo; `[AC]`, `[R1 GA3f]`, `[R2 YeZF]`, `[R3 v4ZK]` = review text.

Companion document: `docs/2026-08-27-code-plan.md` (the runs that produce the numbers
several of these items need).

**Out of scope by author decision (2026-08-27):** the JumpReLU-vs-vanilla attribution
of the ablation section (§4.4, Tables 3/13/14, Limitations L686–L690). Handled directly
by the authors; no item below depends on it.

---

## Tier 1 — Items that decide the open issue

### T1. Retire the `|r| > 0.1` headline; rebuild Table 1 around the threshold where the null dies

**What is wrong.** Table 1's first row and the derived "1.6×" GemmaScope ratio
[paper L468–L472, Table 1] present grounded-latent count at `|r| > 0.1` as an
SAE property. It is not. Covariance-matched random directions, run through the
identical audit, ground *as well or better*:

| \|r\| | JumpReLU | Vanilla | GemmaScope | Random (L0-matched) | Random (dense) |
|---|---|---|---|---|---|
| 0.1 | 9,721 | 8,985 | 5,790 | **9,132** | **10,988** |
| 0.2 | 2,075 | 2,063 | 295 | 127 | 538 |
| 0.3 | 610 | 675 | 54 | **1** | 40 |
| 0.4 | 276 | 291 | 13 | 0 | 2 |
| 0.5 | 147 | 143 | 4 | 0 | 0 |
| 0.6 | 61 | 73 | 0 | 0 | 0 |
| 0.7 | 28 | 29 | 0 | 0 | 0 |
| peak \|r\| | .864 | .853 | .574 | **.314** | .431 |

Sources: SAE/GemmaScope rows from `results/{jumprelu,vanilla,gemma}/test_split/posthoc/posthoc_summary.json
→ threshold_sweep`; random rows from `results/necessity/random_matched/seed0/audit_l0_40.92/audit_summary.json
→ monospecificity[].n_grounded` and `audit_dense/`, `run_summary.json → arms.*.max_abs_r_any_feature`.

The same holds for BH-significant association counts: random-matched has 403,497 of
847,872 significant (47.6%) [`audit_l0_40.92/audit_summary.json → grounding.n_significant_after_bh`]
against JumpReLU's 305,552 (36.0%) [`results/jumprelu/test_split/grounding_summary.json`].
**Neither "grounded latent count at r > 0.1" nor "number of significant associations"
discriminates a learned SAE from an arbitrary direction in the same covariance geometry.**

**What to write instead.** Report the full table above, including the row where the
SAE loses. Move the headline to `|r| ≥ 0.2`, where separation is ~16×, and to
`|r| > 0.3`, where it is 610–675 vs 1. State explicitly that the threshold is
**chosen by the null, not by the SAE** — 0.2 is where the matched random baseline's
grounded count collapses — which is what keeps it from being a post-hoc cut.

**Why it matters.** This is the AC's single open issue: *"The paper shows SAE features
produce useful audit signals, but not that SAEs are needed to produce them… The complete
pipeline needs to be run on a matched non-learned baseline such as random directions with
matched activation statistics"* [AC]. Conceding the r>0.1 row is what makes the r≥0.3
result credible rather than assertive.

**Sections touched:** Abstract [L026–L030], contributions [L106–L116], §4.1 [L460–L479],
Table 1, Figure 2, Conclusion [L650–L658].

---

### T2. New section: SAE necessity against matched non-learned baselines

**What is missing.** The submission has no matched non-learned baseline. §3.3
["Comparison Baselines", L319–L349] lists lexical, TF-IDF, and raw-activation, all of
which the AC explicitly ruled insufficient: *"Experiments with GemmaScope is meaningful,
but does not fill the aforementioned checks"* [AC]. R1 states the gap precisely:
*"they do not fully answer whether an SAE-like but non-learned sparse decomposition would
also yield apparently interpretable ICD concordance after the same feature-selection
procedure"* [R1 GA3f].

**What to write.** A dedicated section (suggest §4.2, before concordance) covering:

1. **The audit protocol as a single code path.** State that every source — SAE, random-matched,
   diff-in-means, probe-direction, PCA — is scored by one harness
   (`src/mech_interp_research/necessity_audit.py`) with the same 46-code panel, the same
   held-out split (shards 281–311, 4,911 notes), the same BH-FDR at q=0.05, the same
   off-target diagnostic, and the same one-feature-per-code selection rule. This is what
   makes "same audit protocol… same off-target diagnostic" [AC] structural rather than asserted.
2. **The selection split.** Best-of-*k* selection scored on the notes it was selected on is
   upward-biased, and the bias grows with *k* — exactly the regime under test. All sources
   select on shards 0–30 and report on 281–311. (Requires code item C1 before this can be
   written; see companion doc.)
3. **The matched-random result** (T1's table) and the calibration number: **max |r| over
   18,432 covariance-matched random directions = 0.314** (L0-matched) / 0.431 (dense),
   against the SAE's 0.864. This is the quantitative answer to "you searched 18,432
   candidates against 46 codes."
4. **Specificity and monospecificity per method** (numbers from code items C1–C5).

**Rebuttal text that must not be reused.** The general response argued *"If random directions
were run through the same pipeline, they would show flat concordance across |r| thresholds"*
— a prediction, which is what the AC penalized. Replace with the measured result.

---

### T3. Lead §4.2 with blind forced-choice retrieval; make anchored concordance secondary

**What is wrong.** §4.2 [L486–L506] leads with the anchored concordance gradient, in which
the judge is shown the pre-selected target code. All three reviewers independently read this
as circular. R3 states the mechanism: *"because the Grounding step already selects the
top-associated code as the target for comparison, the gradient could simply reflect that
stronger statistical correlations make the pairing more likely to be correct, rather than
demonstrating independent semantic alignment"* [R3 v4ZK]. Also [R2 YeZF] Weakness 1,
[R1 GA3f] Concern 1.3.

**What to write.** Lead with the retrieval result — a judge blind to both |r| and the
pre-selected target recovers the code from a 9-option slate (7 cross-chapter prevalence-matched
distractors + "none"), against an 11.1% chance floor:

| Judge | hit@1 at \|r\|>0.1 / >0.3 / >0.5 | "none" rate |
|---|---|---|
| GPT-4o | 74.2 / 94.3 / 98.6 | 23.2 → 1.4% |
| Claude Sonnet 4.6 | 71.1 / 90.4 / 95.8 | 27.1 → 4.2% |
| DeepSeek-V3 | 70.0 / 90.4 / 94.4 | 25.5 → 3.5% |

Source: `results/auto_interp/multi_judge/retrieval/{gpt-4o,sonnet-4-6,deepseek-v3}/retrieval_summary.json`.

Then present anchored concordance under three judge families as the secondary, original-protocol
result:

| Judge | Concordance (Y+P) at >0.1 / >0.3 / >0.5 | Exact YES |
|---|---|---|
| Claude Sonnet 4.6 | 85.5 / 95.0 / 98.6 | 22.6 / 30.7 / 43.1 |
| GPT-4o | 90.5 / 98.6 / 100 | 33.9 / 46.1 / 59.7 |
| DeepSeek-V3 | 96.3 / 98.9 / 100 | 23.7 / 32.1 / 42.4 |

Source: `results/auto_interp/multi_judge/arm0/*/concordance_summary.json`.

**Note a numeric discrepancy to resolve before submission:** the paper's Table 2 reports
85.0 / 94.6 / 98.6 and YES 22.4 / 30.4 / 42.4 [paper L487–L492, Table 2]; the Arm-0 replication
JSON for the same judge and the same 380 features gives 85.5 / 95.0 / 98.6 and 22.6 / 30.7 / 43.1
(Sonnet UNKNOWN counts 11 / 9 / 2 are excluded from the denominator in one and not the other).
Pick one convention, state it in the caption, and make Table 2, Figure 3, the abstract [L020–L021],
and the rebuttal tables agree.

---

## Tier 2 — Claim calibration (R1's Major Concern 1)

### T4. Scope claims to ICD codes as noisy anchors and to the 46-code panel

**What is wrong.** The abstract and contributions assert *"features above r > 0.6 are fully
monospecific, each tracking exactly one diagnosis code"* [paper L114–L116, L025–L026] and
the Conclusion asserts *"full monospecificity above r > 0.6"* [L651–L652]. R1: *"A feature
that is specific to one of 46 evaluated ICD-9 codes above a threshold is not necessarily
globally monospecific or clinically atomic"* [R1 GA3f]. The AC endorses:
*"scoping the claims to ICD codes as noisy anchors and specificity relative to the 46-code panel"* [AC].

**What to write.**
- Everywhere "monospecific" appears unqualified, write "monospecific **with respect to the
  46-code panel**". The measured quantity is `mean_codes_per_grounded = 1.00` at |r|>0.6
  over 46 codes [`results/jumprelu/test_split/posthoc/posthoc_summary.json → monospecificity`],
  not atomicity.
- Add to §3.4 or §6 that ICD codes are administrative admission-level labels, and that
  **annotation noise attenuates point-biserial correlation** — so the reported grounding is a
  lower bound on the true feature–concept association, not an upper bound. This converts a
  concession into a supporting argument. Cite Searle et al. (2020), already in the bibliography [L678].
- Reframe "validated clinical concept units" language toward R1's own formulation:
  *"some domain-trained SAE features are concordant with ICD-associated clinical semantic
  patterns and show diagnosis-conditioned loss relevance under the tested interventions."*

---

### T5. Report YES-only alongside YES+PARTIAL, with the PARTIAL subtype breakdown

**What is wrong.** Table 2's `Concord. = (YES+PARTIAL)/N` is the headline; exact YES appears
only as a trailing column [paper Table 2, L492]. [R2 YeZF] Weakness 2: *"Concordance counts
YES and PARTIAL. Exact YES rates are much lower: 22.4%, 30.4%, and 42.4%."* AC endorses [AC].

**What to write.** Both rates as co-equal headline numbers in the abstract and §4. Then the
subtype decomposition as a *result*, not a concession: of the 180 PARTIAL verdicts at |r|>0.3,
101 (56%) describe treatment or medication patterns for the associated diagnosis and 32 (18%)
describe monitoring or diagnostic markers. Frame: PARTIAL is the signature of the SAE splitting
one administrative code into its constituent clinical subpatterns — which is what T7 formalizes.

**Verify before writing:** the 101/32 split is quoted from the rebuttal; regenerate it from
`results/jumprelu/auto_interp/concordance_results.csv` so the paper cites a reproducible artifact.

---

### T6. Rename causal ablation to diagnosis-conditioned loss relevance

**What is wrong.** §3.6 and §4.4 are both titled "Causal Ablation" [L427, L624], and the
abstract says "causal ablation" [L022]. R1: *"the ablation experiments should be described as
diagnosis-conditioned loss relevance, not full mechanistic faithfulness"*; the AC repeats it
verbatim [AC]. The authors conceded the point in the general response.

**What to write.** Rename both section headings and the abstract phrase — not merely add a
caveat. Add the effect-size calibration in absolute terms: the per-target ablation effect is
0.0075–0.011% of base loss [`results/ablation/vanilla_pilot_extended/posthoc_specificity.csv
→ pct_of_base_loss`], against a mean clean loss of 1.635 nats
[`ablation_summary.json → mean_loss_clean`]. R1 asked for exactly this
(*"absolute effect-size calibration"*), and reporting a specific-but-small effect honestly is
stronger than reporting Cliff's δ alone.

---

### T7. New subsection: one ICD code decomposes into distinct SAE features (CHF)

**What is missing.** R1's second named example of what a compelling clinical SAE paper could
show: *"An ICD diagnosis label decomposes into clinically meaningful internal subpatterns such
as diagnosis mention, treatment, monitoring, complication, differential diagnosis, and ruled-out
disease"* [R1 GA3f]. Currently in the rebuttal only.

**What to write.** A subsection with its own table: CHF (ICD-9 4280) yields 13 features at
|r|>0.5 — four tracking the diagnosis string (YES verdicts), nine tracking ejection-fraction
measurements, furosemide dosing, systolic/diastolic subtype, and weight-monitoring instructions
(PARTIAL verdicts). Pair with T5's subtype counts.

**Why this is the strongest necessity argument available.** It is the one result no per-code
baseline can produce even in principle: diff-in-means, the supervised probe, and TF-IDF each
yield exactly one direction per code, so decomposition is not expressible in their frame. Say
that explicitly in the necessity section (T2).

**Verify before writing:** regenerate the 13 / 4 / 9 counts from
`results/jumprelu/auto_interp/concordance_results.csv` joined to the held-out
`top_associations.csv`; the current numbers come from the rebuttal, not from a checked artifact.

---

### T8. Shuffled-explanation control alongside Table 11

**What is wrong.** Table 11 reports fuzzing 0.938 and detection 0.962 essentially uniformly
across strongly grounded, weakly grounded, and dead features — which reads as a failure of the
paper's own metric.

**What to write.** The shuffled control reframes it: re-scoring each feature's contexts against
a deliberately wrong explanation drops fuzzing to 0.496 and detection to 0.522 against real
scores of 0.932 and 0.961 — a gap of ~0.44, Wilcoxon p ≈ 0, stable across all tiers including
dead features [`results/auto_interp/shuffled_control/shuffled_control_summary.json`;
compare the Paulo et al. (2024) ~0.51 null recorded in that file's `chance_reference`].

The scorer **is** sensitive to explanation content; it simply does not discriminate feature
quality. That is the setup for the paper's actual thesis — the internal metric doesn't
separate and the external one does. Also addresses R1's request for *"prompt/model robustness"*.

---

### T9. Clarify the multiple-testing map

**What is wrong.** §3.4 states BH at q=0.05 over `d_sae × 46` [paper L363–L365] but never says
which analyses are inside that family and which are downstream. R1: *"a clearer map of which
hypothesis families are covered by each multiple-testing correction"* [R1 GA3f]; AC endorses.

**What to write.** One explicit paragraph: a single BH procedure at q=0.05 over the full family
of `d_sae × 46 = 847,872` associations for the 18,432-latent SAEs; concordance, categorization,
and ablation are downstream analyses on the selected grounded features and receive no additional
correction; the ablation Mann-Whitney tests carry their own separate BH across ablation targets.

---

## Tier 3 — Named corrections and supplement

### T10. Soften the GemmaScope direction-mismatch claim

§4.3 asserts *"The cause is direction mismatch"* [paper L612–L613] and the Conclusion repeats
*"its feature directions do not span the clinical activation subspace"* [L656–L658].
[R2 YeZF]: *"the claim… should be softened; the diagnostics mainly rule out simple mean shift."*
AC endorses. Rewrite to: the diagnostics (decoder-bias substitution leaves EV at −4.20;
full recentering worsens it to −6.56 [L608–L612]) rule out a simple mean shift but do not
exclude other forms of distributional mismatch.

### T11. Fix the "test-set explained variance" wording

Table 5's "Checkpoint selection: test-set explained variance" [paper L817 region] contradicts
the claim that the MIMIC-IV test split was never accessed. [R2 YeZF]. Replace "test-set" with
"held-out evaluation split (4,911 notes, shards 281–311)" in Table 5 and anywhere else the
phrase appears.

### T12. Expand Limitations

The AC lists four additions [AC]: single seed, single model, credentialed data access, and the
risk of over-reading ICD concordance as clinical validation. The current §6 [L669–L698] covers
single seed and single model [L682–L686] and the single-LLM pipeline [L690–L695] — which is
now partly superseded by T3's three-judge result, so rewrite rather than delete. Add:

- **Credentialed access.** MIMIC-IV requires credentialed access; state what is released
  (see T14) so a credentialed reader can reproduce end to end. Both R1 and R2 scored
  Reproducibility = 2.
- **Over-reading.** Per R1's Limitations note: describe concordance validation as a
  **hypothesis-generating audit tool**; state that external validation, clinician review, and
  subgroup/site-level checks are required before clinical use; state that ICD labels reflect
  coding and population biases and the method could reinforce site-specific patterns if treated
  as ground truth.
- **The r>0.1 null result** (T1) belongs here too, as a scope limit on the method: below
  |r| ≈ 0.2 the audit does not distinguish learned features from matched random directions.

### T13. Supplement tables: prevalence, group sizes, threshold sensitivity, co-occurrence

Requested by R1 and endorsed by the AC. Per-code prevalence and positive/negative group sizes
for the 46-code panel; the full threshold sweep (T1's table, all methods); code co-occurrence;
and the partial-correlation survival rates already computed
[`results/jumprelu/test_split/posthoc/posthoc_summary.json → partial_correlation_summary`].

### T14. Reproducibility statement

Neither the plan nor the paper carries the release commitment the rebuttal made
(*"release all code, configurations, and non-PHI artifacts"*). Both R1 and R2 scored
Reproducibility = 2 and R2 scored Datasets = 1 — this is cheap score movement. Add a short
statement naming: the repository, SAE checkpoints, correlation matrices, concordance CSVs, and
the `results/` aggregate JSONs, with the PHI policy (`CLAUDE.md`, "Data handling rules") stated.

### T15. Reconcile the TF-IDF single-feature numbers

§4.3 reports *"the SAE's best feature achieves a stronger correlation than TF-IDF's best on
21/46 codes (mean best-r: 0.566 vs. 0.519)"* [paper L598–L601]. The rebuttal reports 23/46 and
0.579 vs 0.519. The on-disk vanilla run gives `n_sae_above_tfidf = 23`, `mean_best_r_sae = 0.5787`,
`mean_best_r_tfidf = 0.5190` [`results/vanilla/tfidf_lr/tfidf_lr_summary.json → supplementary_correlation`],
so the paper's 21/0.566 is the **JumpReLU** run and the rebuttal's is **vanilla**. Both are
correct; they are not labelled. Label the architecture on every such number, and check the same
for AUC: TF-IDF 0.917 and SAE 0.8813 both come from the vanilla run's JSON, while the paper
quotes SAE = 0.888 (JumpReLU) against the same 0.917 [L585–L596].

---

## Dependency map

| Item | Blocked on code work? |
|---|---|
| T1 | No — numbers exist today |
| T2 | **Yes** — C1 (SAE through harness), C2 (whitened diff-in-means), C3, C4, C5 |
| T3, T8, T9, T10, T11, T12, T13, T14 | No |
| T4, T6 | No (T6's calibration numbers exist) |
| T5, T7 | Regeneration from existing CSVs only (no new compute) |
| T15 | No — reconcile against existing JSONs |

---

# Run log — artifacts produced, and what they change in the text

Appended as each code-plan item lands. Every number below was regenerated from an
artifact on disk in this repo; nothing is carried over from the rebuttal. Where a
figure contradicts what is written above, that is called out explicitly.

---

## C1 — all three SAEs through `necessity_audit` with a proper selection split ✅ DONE

**Status:** complete. Artifacts in `results/necessity/sae_audit/{sae_vanilla,sae_jumprelu,gemmascope}/`
plus `comparison_summary.json`. Modal output at `/out/necessity/sae_audit/`.

**What was built.**
- `modal_app/necessity_audit.py` — thin Modal entrypoint (CPU, minutes, reads existing
  `shard_ckpt/`, no re-encode, no GPU).
- `configs/necessity_audit_sae.yaml` — **one** config with a `sources:` list, not three
  per-SAE configs as the code plan proposed. Deviation taken deliberately: a shipped
  test (`test_shipped_comparison_config_is_coherent`) and `CLAUDE.md` already specified
  this shape, and it is the stronger form of the AC's requirement. The split, the code
  panel and the `AuditConfig` live at the top level and `NecessityComparisonConfig`
  **refuses** a `sources:` entry that carries any of them, so a source physically cannot
  be audited under a different protocol. Three sibling files could only ever *agree*.
- `necessity_audit.NecessityComparisonConfig` + `run_comparison()` — appended to the
  module. The audit core (`audit`, `audit_from_checkpoints`, `off_target_specificity_corr`)
  is byte-identical; the only deletion in the diff is an extended `dataclasses` import.
  No published number can move.
- 11 new tests. `uv run pytest tests/` is now **371 passed, 0 failed** — it was
  **1 failed** before this item (`configs/necessity_audit_sae.yaml` did not exist), so
  the CLAUDE.md pre-Modal gate had not actually been green.

**Acceptance criterion — met.** All three: `in_sample_selection: false`,
`selection_mode: "top_per_code"`, `n_select_notes: 5001`, `n_audit_notes: 4911`.
Structurally identical to the random-matched summaries.

**Validation.** The new code path reproduces the existing `test_split` artifacts
*exactly* on every shared statistic — `n_significant_after_bh`, `grounded_latent_count`,
and peak |r| for all three SAEs (e.g. JumpReLU 305,552 / 9,721 / 0.8643). Independently,
`on_target_r` (computed inside `off_target_specificity_corr`) equals `r_audit` (read from
the grounding matrix) to 0.00e+00 for all 138 selected features.

### C1-a. **Finding: the selection penalty is negligible, for every source including the null**

Median |r| of the selected feature, on the selection split (shards 0–30) vs the
held-out audit split (281–311):

| Source | k | select-split | audit-split | shift |
|---|---|---|---|---|
| JumpReLU | 18,432 | 0.5824 | 0.5739 | −1.5% |
| Vanilla | 18,432 | 0.5739 | 0.5736 | −0.1% |
| GemmaScope | 16,384 | 0.3070 | 0.3094 | +0.8% |
| Random (L0-matched) | 18,432 | 0.1468 | 0.1491 | +1.5% |
| Random (dense) | 18,432 | 0.2161 | 0.2186 | +1.2% |

**Every shift is under 1.5%, and three of five are positive.** With ~5,000 notes per
split, best-of-18,432 selection is dominated by signal that replicates, not by noise
that does not. Two consequences, both usable:

1. **The paper's grounding numbers were not materially inflated by selection** — and
   this is now demonstrated on a held-out split rather than assumed.
2. **The null was not paying a penalty the SAE avoided.** The concern in code-plan C1
   was real in principle; measured, it is ~0. The SAE-vs-random gap (0.574 vs 0.149)
   is not a selection artifact in either direction.

Also: `max |r| over the selected 46` equals `max |r| over the whole k×46 matrix` for
**every** source. The out-of-sample peak the code plan asked to report alongside the
in-sample 0.864 **is** 0.864 — the gap is zero. Write it that way.

### C1-b. **Correction to T1's table: two peak-|r| cells have the wrong provenance**

T1's grounded-count rows are all correct and reproduce exactly. But the `peak |r|` row
mixes two different note populations:

| | T1 as written | source of that number | correct held-out value |
|---|---|---|---|
| JumpReLU | .864 | test split (4,911) ✅ | **0.8643** |
| Vanilla | .853 | **full corpus (50,000)** ❌ | **0.8595** |
| GemmaScope | .574 | **full corpus (50,000)** ❌ | **0.5450** |
| Random (L0-matched) | .314 | audit split (4,911) ✅ | **0.3143** |
| Random (dense) | .431 | audit split (4,911) ✅ | **0.4307** |

Comparing a 50,000-note argmax against the null's 4,911-note argmax is not the clean
comparison the table implies. Use the held-out column throughout: it *raises* vanilla
(.853 → .860) and *lowers* GemmaScope (.574 → .545). `results/vanilla/grounding_summary.json`
and `results/gemmascope/grounding_summary.json` are the 50k artifacts the old cells came from.

### C1-c. **Finding: the monospecificity progression is NOT SAE-specific — reframe T4**

T4 and the abstract present "30.1% → 100% monospecific" as an SAE property. Run on one
code path, the null shows the same shape:

**Fraction of grounded features that are monospecific** (held-out, 46-code panel):

| \|r\| | JumpReLU | Vanilla | GemmaScope | Random (L0-m) | Random (dense) |
|---|---|---|---|---|---|
| 0.1 | 0.299 | 0.313 | 0.219 | **0.354** | 0.261 |
| 0.2 | 0.612 | 0.619 | 0.759 | **0.724** | 0.613 |
| 0.3 | 0.793 | 0.816 | 0.870 | 1.000 *(n=1)* | 0.700 |
| 0.4 | 0.884 | 0.904 | 0.846 | — *(n=0)* | 1.000 *(n=2)* |
| 0.5 | 0.925 | 0.944 | 1.000 | — | — |
| 0.6 | **1.000** | **1.000** | — | — | — |

At |r|>0.1 and |r|>0.2 the random null is **more** monospecific than JumpReLU. Rising
monospecificity with |r| is a property of the *statistic* — fewer surviving associations
per feature as the bar rises — not of learned features. Cells at n≤2 must be printed
with their count or they are misleading.

**What is SAE-specific** is how many features survive at all: 61–73 grounded features at
|r|>0.6 for the SAEs against **zero** for both random arms. Reframe the claim from
*"SAE features become monospecific"* to *"SAE features remain grounded at thresholds
where matched random directions are extinct, and those survivors are monospecific with
respect to the 46-code panel."* Same evidence, and it survives the null.

### C1-d. The specificity result, now on one enforced code path

Selected feature per code, held-out, c-negative-restricted off-target:

| Source | median on-target \|r\| | median specificity ratio | median n_off_sig | worst-code \|r\| | best-code \|r\| |
|---|---|---|---|---|---|
| JumpReLU | 0.574 | **14.94** | 3.0 | 0.251 | 0.864 |
| Vanilla | 0.574 | **15.88** | 2.0 | 0.272 | 0.859 |
| GemmaScope | 0.309 | 6.29 | 4.0 | 0.126 | 0.545 |
| Random (L0-matched) | 0.149 | 2.12 | 12.0 | — | 0.314 |
| Random (dense) | 0.219 | 3.00 | 13.0 | — | 0.431 |

This is the T2 §4.2 table. Note the coupling caveat the code plan raises as B1/C4 applies
here too: random's specificity ratio is low *partly because* its on-target |r| is low.
C4 must add the restricted comparison over codes where two methods reach comparable |r|
before this table is load-bearing.

**Not yet answered by C1:** every row above is an SAE or a random null. The
label-supervised baselines (whitened diff-in-means C2, probe directions C3, PCA C5,
TF-IDF C8) are what R1 named, and they are still missing from this table.

---

## C2 — whitened difference-in-means ✅ DONE (acceptance gate met on the corrected criterion)

**Status:** complete. Artifacts in `results/necessity/direction_audit/{diff_in_means_none,
diff_in_means_diagonal,diff_in_means_full}/` and `results/necessity/diff_in_means_whitened/
directions_manifest.json`. `results/necessity/diff_in_means/` is marked **SUPERSEDED**
(see the `SUPERSEDED.md` written into that directory).

**What was built.**
- `diff_in_means_baseline.estimate_pooled_covariance()` — shrunk pooled-space covariance
  with the Ledoit-Wolf analytic coefficient, plus the anisotropy diagnostics the code plan
  asked to *measure* rather than infer from the token-level `sigma_stats.json`.
- `build_directions(..., whiten=)` — all three arms are one estimator under a different
  metric, `d_eff = M⁻¹d` with `M ∈ {I, diag(Σ), Σ}`. The `none` path is unchanged and still
  passes its original regression test.
- `write_direction_source()` + `run_direction_sources()` — emit `shard_ckpt`-format sources.
  **This module computes no audit statistics of its own.** Grounding, off-target and
  monospecificity all come from `necessity_audit`, so the diff-in-means rows and the SAE rows
  are produced by the same code.
- `modal_app/diff_in_means_directions.py`, `configs/diff_in_means_directions.yaml`,
  `configs/necessity_audit_directions.yaml`. 13 new tests; suite at **387 passed**.

### C2-a. Research: which whitening, and why

Marks & Tegmark (2023), *The Geometry of Truth*, define mass-mean probing as
`θ = μ⁺ − μ⁻` with an IID variant `p(x) = σ(θᵀΣ⁻¹x)`, where `Σ⁻¹` "tilts the decision
boundary to accommodate interference" from non-orthogonal features; they prove it coincides
on average with the logistic-regression direction under Gaussian assumptions. Because
`θᵀΣ⁻¹x = (Σ⁻¹θ)ᵀx`, the correction is expressible as a *direction*, which is what an audit
unit must be. That is the theoretical reason to expect the full form to work here, and our
own LR probe at 0.808 AUC on these exact features is the empirical one.

Covariance estimation used **Ledoit & Wolf (2004)** analytic shrinkage toward the scaled
identity, chosen over a hand-set ridge because it needs no tuning — and anything tuned would
have to be tuned on data, which here would mean the audit split. This was not optional: the
pooled-space sample covariance is **singular** (`var_min` exactly 0.0, condition number
6.9 × 10¹⁶). Ledoit-Wolf picked α = 0.00097, which brings the condition number to 3.7 × 10⁵.

**Measured pooled-space anisotropy** (the code plan asked for this specifically, because the
3,950/13.1 figure in `sigma_stats.json` is token-level while the confound acts after
max-pooling): `var_max/var_mean = 104.7` over 2,304 dimensions, 40,088 train notes.

### C2-b. **The acceptance gate as written was unmeetable — its arithmetic used the wrong prevalence**

The code plan sets "expect ≥ 0.4 given the probe's 0.808 AUC", derived from converting
r_pb → AUC "at p ≈ 0.15". **The actual median prevalence of the 46-code panel is 0.073**, not
0.15. Redoing the conversion in the correct direction and *per code* with each code's real
prevalence (`d = √2·Φ⁻¹(AUC)`, `r = d√(pq)/√(d²pq+1)`), the raw-LR probe corresponds to a
**median |r| of 0.307**, not ≥ 0.4. Source: `raw_cv_results.csv` from the Baseline-3 run,
46/46 codes.

So the correct gate — *does whitened diff-in-means reach the LR ceiling on identical
features?* — is **met**:

| method | median on-target \|r\| | beats LR ceiling on |
|---|---|---|
| diff-in-means, unwhitened | 0.1209 | **0/46** codes |
| diff-in-means, diagonal | 0.1266 | **0/46** codes |
| *raw-LR probe (the ceiling)* | *0.3073* | — |
| **diff-in-means, full/LDA** | **0.3395** | **39/46** codes (Wilcoxon p = 1.7e-8) |
| vanilla SAE | 0.5736 | 45/46 codes |
| JumpReLU SAE | 0.5739 | 45/46 codes |

The plan's own defect criterion — "a near-cousin of LR landing 0.22 AUC below LR is a
baseline defect, not a finding" — is cleared: the whitened baseline now slightly *exceeds* LR.

### C2-c. **Finding: the plan's proposed fix (diagonal / z-score) does not work, and the reason is diagnostic**

The plan prescribed "Z-score / diagonal-whiten before differencing". Measured, it moves the
median from 0.1209 to 0.1266 — essentially nothing. The mechanism, from the pairwise geometry
of the 46 directions:

| arm | mean pairwise \|cos\| | frac \|cos\| > 0.9 | **effective dimensionality** |
|---|---|---|---|
| none | 0.685 | 0.103 | **1.89** |
| diagonal | 0.691 | 0.159 | **1.82** |
| full | 0.060 | 0.000 | **33.68** |

(Effective dimensionality = participation ratio of the Gram spectrum of the 46 unit directions;
46 would mean fully distinct.)

The unwhitened baseline's "46 concept directions" are **~2 directions**. That is the direct
measurement of what the plan inferred from the flat off-target profile, and it explains the
1.25 specificity ratio: when every code's direction is the same axis, an on-target correlation
is not distinguishable from an off-target one. **Diagonal whitening does not fix it because the
confound lives in the off-diagonal covariance, not in per-dimension scale.** Only the full
inverse-covariance form decorrelates the directions.

### C2-d. Robustness: the result is not a conditioning artifact

Shrinkage swept over a 100× range, `full` arm:

| α | cond(Σ_shrunk) | median \|r\| | specificity ratio | median n_off_sig |
|---|---|---|---|---|
| 0.00097 (Ledoit-Wolf) | 3.7e5 | 0.3395 | **5.68** | **8** |
| 0.01 | 3.6e4 | 0.3412 | 5.64 | 8 |
| 0.1 | 3.3e3 | 0.3510 | 5.32 | 10 |

On-target moves +3% and specificity *degrades* as α rises. Ledoit-Wolf did not under-shrink;
0.34 is where one linear direction per code tops out on max-pooled activations. **Keep the
parameter-free Ledoit-Wolf arm as the headline**, and report the sweep as the robustness check.

### C2-e. The necessity table, all seven sources on one code path

Held-out shards 281–311 (4,911 notes), pinned 46-code panel, identical `AuditConfig` except
`selection`, which `k` forces (`identity` for one-direction-per-code sources, `top_per_code`
for the rest — stated in two configs rather than hidden as a per-source override):

| source | k | median \|r\| | peak \|r\| | spec. ratio | n_off_sig | grounded @0.2 / @0.3 / @0.5 |
|---|---|---|---|---|---|---|
| diff-in-means (none) | 46 | 0.121 | 0.291 | 1.25 | 17 | 42 / 0 / 0 |
| diff-in-means (diagonal) | 46 | 0.127 | 0.310 | 1.37 | 17 | 41 / 3 / 0 |
| **diff-in-means (full/LDA)** | 46 | **0.339** | **0.699** | **5.68** | **8** | 43 / 28 / 7 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 2.12 | 12 | 127 / 1 / 0 |
| GemmaScope | 16,384 | 0.309 | 0.545 | 6.29 | 4 | 295 / 54 / 4 |
| vanilla SAE | 18,432 | 0.574 | 0.859 | 15.88 | 2 | 2,063 / 675 / 143 |
| JumpReLU SAE | 18,432 | 0.574 | 0.864 | 14.94 | 3 | 2,075 / 610 / 147 |

The SAE beats whitened diff-in-means on **45/46 codes** (Wilcoxon p = 1.4e-13).

### C2-f. ⚠️ **Risk flagged for C4: specificity may not survive the matched-\|r\| control**

The plan's B1 warning now has teeth. Note the pair closest in on-target strength:

- whitened diff-in-means: |r| = 0.339 → specificity ratio **5.68**
- GemmaScope: |r| = 0.309 → specificity ratio **6.29**

At comparable on-target |r| these are **effectively tied**. The vanilla SAE's 15.88 is
attached to an on-target |r| of 0.574, so the 15.9-vs-5.7 headline is confounded by exactly the
coupling B1 exists to control for. **The paper must not lead with the raw specificity ratio.**
C4's restricted comparison — codes where two methods reach comparable |r| — is now load-bearing
rather than a nicety, and it is possible it will show the SAE's specificity advantage is largely
a by-product of its higher on-target correlation. That must be reported either way.

What is *not* in doubt from C2: the SAE reaches on-target correlations no one-direction-per-code
method approaches (0.574 vs 0.339, 45/46 codes), and it still grounds 143–147 features at
|r| > 0.5 where diff-in-means yields 7 and the random null yields 0.
