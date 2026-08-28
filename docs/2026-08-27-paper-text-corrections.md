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

---

## C3 — supervised probe as a direction ✅ DONE

**Status:** complete. Artifacts in `results/necessity/direction_audit/{probe_lr_balanced,
probe_lr_unweighted}/` and `results/necessity/probe_directions/probe_manifest.json`.

**What was built.** `raw_lr_baseline.build_probe_directions()` +
`run_probe_direction_sources()`, `modal_app/probe_directions.py`,
`configs/probe_directions.yaml`. 11 new tests, suite at **398 passed**. Like C2 the module
emits only `shard_ckpt` sources; every statistic comes from `necessity_audit`.

### C3-a. Two corrections to the published probe, both material

1. **Circularity.** The 0.808 run cross-validated across all 50,000 notes
   [`tfidf_lr_summary.json → n_notes: 50000, cv_folds: 5`], so its fits had already seen
   shards 281–311. Refit here on shards [31, 281) only, disjoint from both the selection and
   audit splits. **The fix costs nothing**: median CV AUC is 0.8085 (balanced) / 0.8073
   (unweighted) against the published 0.808, so removing the leakage does not weaken the
   baseline.
2. **Unstandardized penalty.** That run fitted LR on raw features with an isotropic L2
   penalty in a space whose per-dimension variance spans 104× (C2). That is not one penalty,
   it is 2,304 different ones. Features are now standardized with train statistics and the
   coefficient mapped back to raw space as `coef/σ`.

**A hyperparameter defect caught and fixed before publishing.** The first fit used
`C ∈ [1e-3, 1.0]` and selected `1e-3` — the grid's lower boundary — for all 46 codes in both
arms, so the optimum lay outside the grid and the "selection" was not one. sklearn penalizes
`1/C` against a *summed* loss (verified empirically: ‖coef‖ scales ~3× when n scales 10×), so
effective strength is `1/(C·n)` and even `C = 1e-3` is weak at n = 40,088. Grid extended to
`[1e-6 … 1e-1]`; the chosen C is now **interior** — balanced: 33 codes at 1e-4, 13 at 1e-3;
unweighted: 40 at 1e-3, 6 at 1e-4.

### C3-b. **Finding: the probe direction is in the LDA family, exactly as ridge theory predicts**

For centered X, `β_ridge ∝ (Σ + (λ/n)I)⁻¹d` — the shrunk LDA form. So C3 and C2 should not be
independent methods. Mean |cos| between matched per-code directions, 46 codes:

| | dm_none | dm_diagonal | dm_full (LDA) | probe (bal.) | probe (unw.) |
|---|---|---|---|---|---|
| **dm_none** | 1.000 | 0.533 | 0.224 | 0.220 | 0.208 |
| **dm_full (LDA)** | 0.224 | 0.281 | 1.000 | **0.714** | **0.756** |
| **probe (bal.)** | 0.220 | 0.421 | 0.714 | 1.000 | 0.950 |

The probe directions sit at 0.71–0.76 with the closed-form LDA direction and only 0.21–0.22
with the plain mean difference. The two probe arms are 0.950 collinear with each other, so
**class weighting barely moves the direction** — and where it does, unweighted is slightly
better on both axes (0.333 vs 0.327 on-target; 4.42 vs 3.67 specificity). Report the
unweighted arm; note class weighting was tested and did not matter.

### C3-c. **Finding: a supervised probe is a WORSE audit unit than the closed-form LDA direction**

| | median on-target \|r\| | specificity ratio | median n_off_sig | effective dims of the 46 directions |
|---|---|---|---|---|
| diff-in-means (full/LDA) | **0.339** | **5.68** | **8.0** | **33.68** |
| probe LR (unweighted) | 0.333 | 4.42 | 17.0 | 9.41 |
| probe LR (balanced) | 0.327 | 3.67 | 16.5 | 9.20 |

On-target they are indistinguishable (0.327–0.339). On **specificity the probe is clearly
worse** — ratio 3.7–4.4 vs 5.7, and roughly *twice* the off-target hits (16.5–17 vs 8).

The mechanism is the effective dimensionality: the probe's 46 directions occupy ~9 dimensions,
the LDA directions ~34. A set of 46 "concept directions" crammed into 9 dimensions must
correlate with many codes at once, which is precisely what a specificity audit penalizes. The
plausible reason is that the logistic objective is free to exploit a discriminative axis shared
across codes (acuity / note length) because it helps classification, whereas `Σ⁻¹` explicitly
divides that shared covariance out.

**This is a usable result for the paper**: the best-classifying direction is not the
best-auditing direction. It is direct evidence for the paper's framing that *classification
performance and audit quality are different axes* — and it comes from the baseline the
reviewers asked for, not from the SAE.

### C3-d. The necessity table, nine sources, one code path

Held-out shards 281–311 (4,911 notes), pinned 46-code panel, identical `AuditConfig` except
`selection` (which `k` forces):

| source | k | median \|r\| | peak \|r\| | spec. ratio | n_off_sig | grounded @0.3 / @0.5 |
|---|---|---|---|---|---|---|
| diff-in-means (none) | 46 | 0.121 | 0.291 | 1.25 | 17.0 | 0 / 0 |
| diff-in-means (diagonal) | 46 | 0.127 | 0.310 | 1.37 | 17.0 | 3 / 0 |
| **diff-in-means (full/LDA)** | 46 | 0.339 | 0.699 | 5.68 | 8.0 | 28 / 7 |
| probe LR (balanced) | 46 | 0.327 | 0.630 | 3.67 | 16.5 | 32 / 5 |
| probe LR (unweighted) | 46 | 0.333 | 0.637 | 4.42 | 17.0 | 31 / 5 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 2.12 | 12.0 | 1 / 0 |
| GemmaScope | 16,384 | 0.309 | 0.545 | 6.29 | 4.0 | 54 / 4 |
| vanilla SAE | 18,432 | 0.574 | 0.859 | 15.88 | 2.0 | 675 / 143 |
| JumpReLU SAE | 18,432 | 0.574 | 0.864 | 14.94 | 3.0 | 610 / 147 |

**Every non-SAE method tops out near |r| ≈ 0.34 and none exceeds 0.70 on any single code**,
against the SAE's 0.574 median and 0.86 peak. All three label-supervised baselines — which see
the labels, unlike the SAE — plateau together, which is a stronger statement than any one of
them alone.

**The C2-f risk is unchanged and still the main threat to the specificity claim:** GemmaScope
reaches specificity 6.29 at |r| = 0.309, above every one-direction-per-code baseline at
comparable |r|. C4's matched-|r| restriction remains load-bearing.

---

## C5 — PCA directions ✅ DONE

**Status:** complete. Artifacts in `results/necessity/pca/seed0/` (four arms: dense,
L0-matched ×2, note-matched), plus `directions_manifest.json`.

**What was built.** `random_matched.pca_directions()` + a `directions_mode: random|pca`
field on `RandomMatchedConfig`, so PCA runs the **identical** pipeline as the A4 null —
same Σ, same train shards, same ridge, same projection, same threshold calibration, same
max-pooling, same splits, same panel, same `AuditConfig`. Only the dictionary differs.
7 new tests, suite at **406 passed**.

**A labelling bug the tests caught.** `source_name` is written into `audit_summary.json` and
keys the comparison table; without a fix a PCA run would have been written as
`random_matched_dense` and silently mislabelled an entire method in the necessity table.
Added `RandomMatchedConfig.source_prefix`, pinned by test.

### C5-a. **Finding: PCA grounds no better than arbitrary directions in the same geometry**

| | k | median on-target \|r\| | peak \|r\| | spec. ratio | grounded @0.2 / @0.3 / @0.5 |
|---|---|---|---|---|---|
| PCA (dense) | 2,304 | **0.141** | **0.441** | 2.88 | 18 / 1 / 0 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 2.12 | 127 / 1 / 0 |
| random (dense) | 18,432 | 0.219 | 0.431 | 3.00 | 538 / 40 / 0 |
| vanilla SAE | 18,432 | 0.574 | 0.859 | 15.88 | 2,063 / 675 / 143 |

The principal components of the clinical activation space — the canonical unsupervised linear
decomposition — reach a median on-target |r| of **0.141**, statistically indistinguishable from
covariance-matched *random* directions (0.149), and produce **1** grounded direction at
|r| > 0.3 and **zero** at |r| > 0.5.

**State the search-budget difference honestly**: PCA has 2,304 candidates against random's
18,432, an 8× smaller search. Per candidate PCA is somewhat better — its peak |r| of 0.441
exceeds L0-matched random's 0.314 and matches dense random's 0.431 with 8× fewer tries. But the
conclusion is unchanged: **variance-ordered directions are not clinically grounded directions.**
This is the cleanest available answer to "preferably PCA or ICA" [AC], and it argues that ICA is
not worth the spend (see Deferred).

### C5-b. Thresholding is inert for PCA

All three sparsity arms (dense, L0 = 47.57, L0 = 40.92) give **identical** numbers to three
decimals. Same mechanism already documented for random directions: after max-pooling over
thousands of tokens, essentially every direction has some token above its threshold, so the
sparsity match is nominal. Report the dense arm; note the others as a null result rather than
padding the table with three identical rows.

### C5-c. Correction to the code plan's cost estimate — in the plan's favour

I predicted ~1–1.5 h from the A4 docstring's claim that the projection is I/O-bound on ~140 GB
of shard reads. **Measured: ~11 minutes**, against ~1h45 for A4 at k = 18,432. Cutting k by 8×
cut wall time ~9×, so the matmul dominates at k = 18,432 and the run is *not* I/O-bound. The
code plan's "minutes" estimate was right and the A4 docstring's characterisation is misleading
for anyone sizing a future run.

---

## Necessity table — twelve sources, one enforced code path

Held-out shards 281–311 (4,911 notes), pinned 46-code panel, identical `AuditConfig` except
`selection` (which `k` forces: `identity` where k = 46, `top_per_code` otherwise).
**This is the table for T1/T2.**

| source | k | median \|r\| | peak \|r\| | spec. ratio | n_off_sig | @0.1 | @0.2 | @0.3 | @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| diff-in-means (none) | 46 | 0.121 | 0.291 | 1.25 | 17.0 | 46 | 42 | 0 | 0 |
| diff-in-means (diagonal) | 46 | 0.127 | 0.310 | 1.37 | 17.0 | 46 | 41 | 3 | 0 |
| diff-in-means (full/LDA) | 46 | 0.339 | 0.699 | 5.68 | 8.0 | 46 | 43 | 28 | 7 |
| probe LR (balanced) | 46 | 0.327 | 0.630 | 3.67 | 16.5 | 46 | 46 | 32 | 5 |
| probe LR (unweighted) | 46 | 0.333 | 0.637 | 4.42 | 17.0 | 46 | 46 | 31 | 5 |
| PCA (dense) | 2,304 | 0.141 | 0.441 | 2.88 | 4.0 | 256 | 18 | 1 | 0 |
| random (dense) | 18,432 | 0.219 | 0.431 | 3.00 | 13.0 | 10,988 | 538 | 40 | 0 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 2.12 | 12.0 | 9,132 | 127 | 1 | 0 |
| GemmaScope | 16,384 | 0.309 | 0.545 | 6.29 | 4.0 | 5,790 | 295 | 54 | 4 |
| **vanilla SAE** | 18,432 | **0.574** | **0.859** | **15.88** | **2.0** | 8,985 | 2,063 | 675 | 143 |
| **JumpReLU SAE** | 18,432 | **0.574** | **0.864** | **14.94** | **3.0** | 9,721 | 2,075 | 610 | 147 |

**What the table supports, safely:**

1. **No non-SAE source exceeds |r| = 0.70 on any single code**, against the SAE's 0.86 peak and
   0.574 median. Three label-supervised methods (which see the labels the SAE never does) all
   plateau together near 0.33.
2. **At |r| ≥ 0.3 the separation is categorical**: 610–675 SAE features against 28 (best
   baseline), 1 (PCA), 1–40 (random). At |r| ≥ 0.5: 143–147 against 7, 0, 0.
3. **The |r| > 0.1 row remains the SAE's weakest**, exactly as T1 says — dense random directions
   ground *more* latents (10,988) than either SAE.

**What the table does NOT yet support — and must not be claimed until C4:** the raw specificity
ratio. GemmaScope reaches 6.29 at |r| = 0.309 and PCA reaches 2.88 at |r| = 0.141, while the
SAEs' 14.9–15.9 sits at |r| = 0.574. Specificity and on-target strength are coupled, so the
ratio column cannot be read down the page. C4's matched-|r| restriction is the only thing that
turns it into a claim.

---

## C4 — the necessity comparison assembled, and the B1 control run ✅ DONE

**Status:** complete. `src/mech_interp_research/necessity_comparison.py` (9 tests, both
coupling controls mutation-tested), `scripts/build_necessity_comparison.py`,
outputs in `results/necessity/comparison/` + `figures/fig_necessity_specificity.png/.pdf`.
Regenerable with `uv run python scripts/build_necessity_comparison.py`, no Modal. Suite at
**415 passed**.

### C4-a. ⚠️ **Correction to my own C2-f / C5 risk flag — I over-stated it**

I flagged that the 15.9-vs-5.7 specificity headline was "confounded by the coupling B1 exists
to control" and that the SAE's advantage might be "largely a by-product of its higher on-target
correlation". **Measured, that flag was wrong in its strong form**, for a reason worth stating
in the paper.

`specificity_ratio = |on_target_r| / mean|off_target_r|` *is* arithmetically coupled. But
`mean_abs_off_r` — off-target leakage on its own — is not, and on that uncoupled axis:

| method | median \|on_r\| | **median mean\|off_r\|** ↓ | median spec. ratio |
|---|---|---|---|
| **vanilla SAE** | 0.574 | **0.0324** | 15.88 |
| **JumpReLU SAE** | 0.574 | **0.0373** | 14.94 |
| GemmaScope SAE | 0.309 | 0.0468 | 6.29 |
| PCA | 0.141 | 0.0484 | 2.88 |
| diff-in-means (LDA) | 0.339 | 0.0572 | 5.68 |
| random (L0-matched) | 0.149 | 0.0751 | 2.12 |
| random (dense) | 0.219 | 0.0786 | 3.00 |
| probe LR (unweighted) | 0.333 | 0.0885 | 4.42 |
| probe LR (balanced) | 0.327 | 0.0921 | 3.67 |
| diff-in-means (diagonal) | 0.127 | 0.0938 | 1.37 |
| diff-in-means (plain) | 0.121 | 0.0940 | 1.25 |

**All three SAEs occupy the three lowest-leakage positions, and the two trained SAEs are also
the two highest on on-target |r|. They dominate on both axes at once.** When one method is
better on both axes there is no trade-off to control for, and the coupling objection does not
arise. **This, not the specificity ratio, is how the paper should state the specificity result.**

### C4-b. **Finding: B1's premise is empirically backwards between methods**

[plan, B1] assumes "a direction that barely correlates with its own code cannot show much
off-target leakage either" — i.e. positive coupling. Pooled across all 506 (method, code)
points the OLS slope of leakage on on-target strength is **−0.0365 (r = −0.220, p = 5.5e-7)**:
methods that ground *more* strongly leak *less*.

The mechanism is C2's: the weak methods are weak *because* their directions collapse onto a
shared axis (effective dimensionality 1.8–1.9 for plain/diagonal diff-in-means), and a shared
axis correlates weakly with **everything**, which is high leakage, not low. So the failure mode
is "correlates weakly with everything", not "correlates with nothing".

**Report this honestly with its caveat**: *within* a method, across codes, the coupling is
positive for most baselines (diff-in-means LDA ρ = +0.387, p = 0.008; probe LR ρ = +0.425,
p = 0.003; random dense ρ = +0.346, p = 0.019) and negative but non-significant for all three
SAEs (ρ = −0.16 to −0.22, p > 0.14). So B1's concern is real within the supervised baselines
and does not transfer to the between-method ordering.

### C4-c. **The matched-|r| control: underpowered for the trained SAEs, decisive for GemmaScope**

Restricting to codes where two methods reach on-target |r| within 0.05, paired by code:

| A | B | matched codes | median leak A | median leak B | A lower on | Wilcoxon p |
|---|---|---|---|---|---|---|
| vanilla SAE | diff-in-means (LDA) | **5** | 0.0825 | 0.0758 | 2/5 | 1.00 |
| vanilla SAE | probe LR (unweighted) | **5** | 0.0843 | 0.0885 | 4/5 | 0.63 |
| vanilla SAE | PCA | **0** | — | — | — | — |
| vanilla SAE | random (L0-matched) | **0** | — | — | — | — |
| JumpReLU SAE | diff-in-means (LDA) | **7** | 0.0775 | 0.0427 | 2/7 | 0.69 |
| JumpReLU SAE | probe LR (unweighted) | **6** | 0.0574 | 0.0594 | 5/6 | 0.31 |
| **GemmaScope SAE** | **diff-in-means (LDA)** | **15** | **0.0376** | **0.0441** | **11/15** | 0.107 |
| **GemmaScope SAE** | **probe LR (unweighted)** | **14** | **0.0363** | **0.0742** | **13/14** | **0.0017** |

**The trained SAEs cannot be compared this way and the paper must say so.** They are so much
stronger than every baseline that only 5–7 codes overlap in on-target |r| — and against PCA and
random, **zero** codes overlap. Every trained-SAE row is non-significant (p ≥ 0.31); those rows
are *inconclusive*, not supportive, and must not be quoted as if they were.

**GemmaScope is the only SAE for which the control has power, and it wins.** With 14–15
overlapping codes it leaks less than a supervised probe direction on **13 of 14 codes
(p = 0.0017)** and less than the LDA direction on 11 of 15 (p = 0.107, directional). Since
GemmaScope was never trained on MIMIC, this isolates the contribution of the **SAE
architecture** from the contribution of **domain training** — and the paper's separate claim
that domain training adds more on top (leakage 0.0324 vs 0.0468) is untouched.

**This reverses the concern in C2-f.** GemmaScope's mid-table position is not a liability for
the necessity argument; it is the only place the matched-|r| test is estimable, and it is the
cleanest evidence that the architecture rather than the search budget produces the specificity.

### C4-d. What T2 should claim, and what it must not

**Claim, in this order:**
1. The trained SAEs are simultaneously the **strongest-grounding** and the **lowest-leaking**
   sources. Dominance on both axes needs no coupling control.
2. At |r| ≥ 0.3 the separation is categorical: 610–675 SAE features vs 28 (best baseline), 1
   (PCA), 1–40 (random); at |r| ≥ 0.5, 143–147 vs 7, 0, 0.
3. No non-SAE source exceeds |r| = 0.70 on any code, against the SAE's 0.86 peak — and the
   three *label-supervised* methods, which see labels the SAE never does, all plateau near 0.33.
4. At matched on-target strength, a **domain-mismatched** SAE still leaks significantly less
   than a supervised probe direction (13/14 codes, p = 0.0017), which attributes the
   specificity to the architecture.

**Do not claim:**
- That the matched-|r| control supports the *trained* SAEs. It is underpowered there (n = 5–7,
  all p ≥ 0.31) and undefined against PCA and random (n = 0). Say so explicitly — a reviewer
  who runs the restriction will find this immediately.
- The bare specificity ratio as a cross-method headline. Report leakage; keep the ratio as a
  secondary column with its coupling stated.
- That |r| > 0.1 grounded counts favour the SAE. They do not (T1).

---

## C7 — ablation post-hoc artifacts pulled and verified ✅ DONE (2 of 4 rebuttal figures need correction)

**Status:** complete. Pulled to `results/ablation/{vanilla_pilot_extended,vanilla_section}/posthoc_specificity/`.
Pure `modal volume get`; nothing recomputed, `ablation.py` and the ablation configs untouched.

### C7-a. Two corrections to the code plan's premise

1. **The directory is `posthoc_specificity/`, not `posthoc/`.** The plan says the outputs "are
   still in the Modal `posthoc/` directories". There is no `posthoc/` on the volume for any run.
2. **Post-hoc exists for only 2 of the 5 runs the plan lists.** `vanilla_pilot_extended` and
   `vanilla_section` have it; **`vanilla_meanabl`, `jumprelu_pilot_extended` and
   `gemma_scope_pilot_extended` have none at all** — only `ablation_summary.json`,
   `ablation_results.csv` and `shard_results/`.

   So R1's Concern 1.4 checklist (off-target #2, length/#codes #3, effect-size #4,
   section-local #5) is satisfied **for the vanilla SAE only**. Running it for the other three
   is a `modal run ablation_posthoc.py` away but is out of scope by the standing
   instruction; flagging it as an open item rather than doing it.

   The plan's claim that "all five are computed" is true only for vanilla.

### C7-b. Verification of the four rebuttal figures

| Rebuttal figure | Reproduces? | Regenerated value | Source run |
|---|---|---|---|
| on-target specificity **6.4×** | ✅ | **6.380** | `vanilla_section` |
| **2 of ~1,350** off-target tests significant | ⚠️ **numerator yes, denominator no** | **2 of 1,470** | `vanilla_section` |
| Cliff's δ **0.230 → 0.164**, **19/30** still significant | ✅ | **0.2299 → 0.1638, 19/30** | `vanilla_section` |
| mean-ablation median δ **0.169** vs **0.162** zeroing | ✅ | **0.1687** vs **0.1622** | `vanilla_meanabl` / `vanilla_section` |

Three of four reproduce exactly. **All the rebuttal's ablation numbers come from
`vanilla_section` (30 grounded targets), not `vanilla_pilot_extended` (20)** — the two runs give
materially different numbers (specificity 6.38 vs 4.75; δ 0.230 → 0.164 with 19/30 vs
0.169 → 0.097 with 11/20), so **every ablation figure in the paper must name its run**.

**The "~1,350" denominator is wrong; use 1,470.** It looks like a back-of-envelope 30 × 45
(30 targets × 45 off-codes). The artifact tests a mean of 49 off-codes per target, giving
1,470 grounded-target tests, of which 2 are BH-significant at q = 0.05. Note the correct
restriction: `off_target_summary.csv` also carries a `low_r_control` and a `random_control`
target; summing over *all* rows gives 3 of 1,568, and the random control accounts for the extra
hit — which is the expected behaviour of a control, not a failure. Report the grounded-only
figure, **2 of 1,470**, and say that the controls are excluded.

### C7-c. ⚠️ **T6's effect-size range does not reproduce and must not be used**

T6 states the per-target ablation effect is **"0.0075–0.011% of base loss"**, citing
`posthoc_specificity.csv → pct_of_base_loss`. Regenerated from that exact column:

| run | targets | min | median | max |
|---|---|---|---|---|
| `vanilla_pilot_extended` (grounded) | 20 | **−0.0095%** | **+0.0193%** | **+0.6030%** |
| `vanilla_section` (grounded) | 30 | −0.0095% | **+0.0330%** | +0.6030% |

The quoted 0.0075–0.011% appears nowhere in the artifact and I could not reconstruct it from
any combination of the stored fields. **Replace it.** The reproducible statement is:

> The per-target ablation effect has a median of **0.019%** of base loss
> (`vanilla_pilot_extended`) / **0.033%** (`vanilla_section`), ranging from −0.010% to +0.60%
> across targets, against a mean clean loss of **1.635 nats**. In absolute terms the mean
> on-target effect is **0.0011–0.0014 nats**, which is **3.7–4.8% of the SAE's own
> reconstruction tax** (0.029 nats).

The reconstruction-tax ratio is the more informative framing and is stored directly
(`mean_ratio_to_recon_tax`). It says the ablation effect is small *relative to the cost of
using the SAE at all* — which is exactly the honest calibration R1 asked for, and stronger for
being stated plainly.

**Also note** the pre-existing local file `results/ablation/*/posthoc_specificity.csv` is a
renamed copy of `effect_size_calibration.csv` (header verified identical); the real directory
carries four to five files per run.
