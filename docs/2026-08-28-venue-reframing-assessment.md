# Reframing assessment — Submission 15186 after the necessity suite

**Date:** 2026-08-28
**Scope:** an independent read of the EMNLP/ARR reviews against the artifacts produced by code-plan
items C1–C5, C7, C8, plus the current state of the SAE-evaluation literature. Written to answer:
*what are the real contributions now, and how should the paper be reframed for InterpScience
(NeurIPS workshop) and ICLR?*

Companion documents: `docs/2026-08-27-code-plan.md` (the work queue),
`docs/2026-08-27-paper-text-corrections.md` (per-item run log, sections C1–C8).

---

## 0. The deadline that shapes everything

**InterpScience ("Interpretability as a Science", NeurIPS 2026) closes 1 September 2026.** Long
papers 9 pages, references and appendices excluded, ICLR/NeurIPS format accepted.

Its call for papers solicits, verbatim:

- *"Measurement validity, identifiability, and evaluation design"*
- *"Falsifiability and experimental designs that distinguish mechanisms from artifacts"*
- *"Causal and interventional methods for grounding interpretability claims"*
- *"Pathways connecting interpretability research to reproducible science"*

That is a description of the necessity suite. This is the best-fitting venue currently open to us,
and the fit is driven by the parts of our result that are *negative*.

---

## 1. What the evidence now supports — and what it no longer supports

The submitted paper's thesis: internal SAE metrics cannot separate trained from random features
(Korznikov et al. 2026; Heap et al. 2025), therefore validate against **external** structured labels
that the pipeline never observed.

The first half of that sentence is well-supported and well-cited. **The second half has been partly
falsified by our own experiments.**

### 1.1 The headline table, ten sources, one enforced code path

Held-out shards 281–311 (4,911 notes), pinned 46-code panel, identical `AuditConfig`.

| source | k | median \|r\| | peak \|r\| | leakage | @0.1 | @0.2 | @0.3 | @0.5 |
|---|---|---|---|---|---|---|---|---|
| JumpReLU SAE | 18,432 | **0.574** | **0.864** | 0.0373 | 9,721 | 2,075 | **610** | **147** |
| vanilla SAE | 18,432 | **0.574** | 0.859 | 0.0324 | 8,985 | 2,063 | **675** | **143** |
| TF-IDF (binary) | 10,000 | 0.531 | 0.831 | **0.0273** | 2,879 | 740 | 265 | 60 |
| TF-IDF (value) | 10,000 | 0.519 | 0.848 | **0.0251** | 2,614 | 743 | 250 | 58 |
| diff-in-means (LDA) | 46 | 0.339 | 0.699 | 0.0572 | 46 | 43 | 28 | 7 |
| probe LR (unweighted) | 46 | 0.333 | 0.637 | 0.0885 | 46 | 46 | 31 | 5 |
| GemmaScope | 16,384 | 0.309 | 0.545 | 0.0468 | 5,790 | 295 | 54 | 4 |
| random (dense) | 18,432 | 0.219 | 0.431 | 0.0786 | **10,988** | 538 | 40 | 0 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 0.0751 | 9,132 | 127 | 1 | 0 |
| PCA | 2,304 | 0.141 | 0.441 | 0.0484 | 256 | 18 | 1 | 0 |

"leakage" = median `mean|off_target_r|`, c-negative restricted. Unlike `specificity_ratio` it has no
on-target term in it, so it is not mechanically coupled to grounding strength.

### 1.2 Four claims in the submitted paper that the artifacts do not support

1. **The `|r| > 0.1` grounded-count headline is not an SAE property.** Dense random directions ground
   **10,988** latents against JumpReLU's 9,721. External-label grounding fails at this threshold in
   exactly the way internal metrics fail. Table 1's first row and the derived 1.6× GemmaScope ratio
   must go.
2. **Monospecificity is a property of the statistic, not of SAEs.** Random-matched is *more*
   monospecific than JumpReLU at |r|>0.1 (0.354 vs 0.299) and at |r|>0.2 (0.724 vs 0.612). The
   abstract's "features above r > 0.6 are fully monospecific" survives only because the null is
   *extinct* there, which is a different and weaker claim than the one currently made.
3. **Grounding + specificity do not separate SAEs from a supervised lexical baseline.** Given the
   same best-of-k search the SAE gets, TF-IDF leaks *less* than either SAE and has fewer significant
   off-target associations. Normalised by dictionary size, the SAE's grounded-yield advantage is
   1.29–1.38×, not the 2.3–2.7× absolute gap.
4. **Two reported numbers do not reproduce from any artifact.** See §4.

### 1.3 What is genuinely strong and untouched

- **The blind forced-choice retrieval result.** A judge blind to |r| and to the pre-selected target
  recovers the code from a 9-option slate at 74.2 / 94.3 / 98.6 against an 11.1% floor, across three
  judge families. This is the single best asset in the paper and no per-code direction can produce it.
- **Three-judge concordance agreement** (κ = 0.87–0.90).
- **The |r| ≥ 0.3 separation is categorical**: 610–675 SAE features against 1 (L0-matched random),
  1 (PCA), 28 (best supervised direction).
- **Selection bias is negligible.** Best-of-18,432 selected out-of-sample shifts median |r| by under
  1.5% for every source, and the 0.864 peak survives an honest split unchanged. The "you searched
  18,432 candidates against 46 codes" objection is now closed empirically rather than argued.

---

## 2. The reframe

**Current framing.** *"We introduce concordance validation. It works. Domain-trained SAE features
encode real clinical concepts."*
Vulnerable on exactly R1's Major Concern 2 — *why perform this SAE analysis at all?* — because our
own TF-IDF result now answers "on these metrics, perhaps you needn't."

**Proposed framing.** *"External structured labels are the field's proposed escape from internal
metrics. We build the first parity-enforced harness to test that escape, run ten feature sources
through it, and characterise where external validation is and is not valid as a measurement."*

Under this framing the negative results are the product, not a concession. R1's Major Concern 2
dissolves rather than needing an answer: we are no longer asserting that clinical SAE analysis is an
established tool; we are measuring whether the evaluation is able to tell.

**The single most important sentence change: clinical text becomes the *setting*, not the
contribution.** It is the right setting because it is one of the few domains where the labels were
produced by humans, independently, before any model touched the data. That justifies the domain
without requiring SAEs to win.

---

## 3. The three contributions to claim

### Contribution 1 — A parity-enforced audit harness for feature-source comparison

`src/mech_interp_research/necessity_audit.py`. Consumes any `[n_notes × k]` matrix of per-note
feature values; one code path; pinned 46-code panel; disjoint selection (shards 0–30) and audit
(281–311) splits; and a config schema that **refuses at parse time** to let a source carry its own
split, panel, or audit threshold. Ten sources have gone through it: 3 SAEs, random-matched (4
sparsity arms), PCA, diff-in-means (3 whitening arms), LR probe (2 arms), TF-IDF (2 arms).

Korznikov et al. compare SAEs against random baselines. No prior work puts supervised directions,
PCA, a lexical baseline, and SAEs in a single harness with parity enforced structurally rather than
asserted in prose. This is the "experimental design that distinguishes mechanisms from artifacts"
the InterpScience CFP asks for.

### Contribution 2 — A measurement-validity result: the operating range of external-label grounding

- **Below |r| ≈ 0.2 external grounding does not discriminate**; random beats SAEs. This extends
  Korznikov's conclusion *into the external-validation regime that was proposed as the remedy*.
- **Above |r| ≈ 0.3 it discriminates categorically** (610–675 vs 1).
- **The threshold is set by where the null dies, not by where the SAE wins.** Stating this up front is
  what makes it a principled cut rather than a post-hoc one.
- **Monospecificity does not discriminate at any threshold where the null is alive.**
- **Best-of-k selection bias is negligible** at n ≈ 5,000 (<1.5% for every source including the null).
- **Grounding and off-target specificity do not separate SAEs from supervised lexical features.**

### Contribution 3 — What survives, and the methodology that reveals it

The concordance gradient under three judge families and the blind retrieval result, neither of which
any one-direction-per-code method can produce. Plus three methodological findings that are
independently useful:

- **Whitening is decisive for concept directions.** The plain difference of class means yields 46
  "directions" with an effective dimensionality of **1.89** — they collapse onto ~2 shared axes.
  Full inverse-covariance whitening (mass-mean / LDA, Marks & Tegmark 2023) gives **33.68**. Diagonal
  z-scoring, the intuitive fix, does not work (1.82): the confound lives in the off-diagonal
  covariance.
- **The best-classifying direction is not the best-auditing direction.** A fitted L2 logistic probe
  matches closed-form LDA on on-target grounding (0.333 vs 0.339) but leaks substantially more
  (0.0885 vs 0.0572) because its 46 directions occupy ~9 effective dimensions against LDA's ~34.
- **PCA grounds no better than arbitrary directions** in the same covariance geometry (0.141 vs
  0.149), which answers the AC's "preferably PCA or ICA" and argues ICA is not worth the spend.

---

## 4. Numbers that must be corrected before any submission

| Claim | Status | Correct value |
|---|---|---|
| Abstract: "median Cliff's δ=0.30 for grounded features" | needs qualifier + architecture check | Table 3 labels it **top-10**; the value reproduces from the `vanilla_pilot` artifact (0.2999), while `jumprelu_pilot_extended` gives **0.090**. GemmaScope's .195 matches `gemma_scope_pilot` (0.1948). This is the known JumpReLU-vs-vanilla attribution item. |
| Table 1 peak \|r\|, vanilla | wrong split | .853 is a 50,000-note argmax. Held-out is **0.8595**. |
| Table 1 peak \|r\|, GemmaScope | wrong split | .574 is a 50,000-note argmax. Held-out is **0.5450**. |
| T6: "0.0075–0.011% of base loss" | **does not reproduce** | Regenerated from the cited column: min −0.0095%, **median +0.0193%**, max +0.6030%. Replace with the reconstruction-tax framing: the mean on-target effect is 0.0011–0.0014 nats, **3.7–4.8% of the SAE's own 0.029-nat reconstruction tax**. |
| Rebuttal: "2 of ~1,350 off-target tests" | denominator wrong | **2 of 1,470** (grounded targets only; the two control targets add 1 more hit over 98 further tests). |
| 12.5× GemmaScope ratio at \|r\|>0.3 | ✅ verified | 675 vs 54. |
| Ablation figures generally | must name the run | Every rebuttal ablation number comes from `vanilla_section` (30 grounded targets); `vanilla_pilot_extended` (20) gives materially different values — specificity 4.75 vs 6.38, 11/20 vs 19/30 surviving residualization. |

---

## 5. Venue strategy

### 5.1 InterpScience — submit, 1 September

Lead with **Contributions 1 and 2**. The paper is *"we stress-tested external-label validation for
SAEs and here is its operating range."* Clinical NLP is a one-paragraph setting justification in §3.
Negative results are the product. The blind-retrieval result is the evidence that something survives
the stress test.

**Cut for 9 pages:** the GemmaScope domain-shift analysis (§4.3 + Appendix G), the scorer-ceiling
material, most of the 5-way categorization, and the causal-ablation section. None of them carry
Contributions 1–2, and the ablation section additionally carries the unresolved attribution issue.

**Prerequisites:** the five corrections in §4. All tables regenerate from `results/` with no Modal
call via `uv run python scripts/build_necessity_comparison.py`.

### 5.2 ICLR — the fuller paper, needs two more things

Contributions 1 + 2 + a *positive* claim. The positive claim has to be **comparative category purity
(code-plan C6)**, which is now load-bearing twice over: it is the AC's still-open *"same explanation
budget"* requirement, and it is the only remaining evidence that can distinguish an SAE feature from
an n-gram. An n-gram is a string; it cannot separate a disease mention from a medication that shares
the token. **TF-IDF must be one of C6's sources.**

ICLR also realistically needs the **second seed**. Sainsbury et al. (2026) — the paper R1 cites as
Shukla et al. — report **21% feature reproducibility across random seeds** on clinical SAEs and
conclude individual features are "illustrative rather than stable". R1 and R3 both raised single seed.
With that result in the literature, a single-seed clinical SAE paper will be asked to justify itself,
and a stated limitation will not cover it. This is now the highest-value deferred item in the plan.

### 5.3 Estimated odds

| | as-is | reframed | + C6 | + second seed |
|---|---|---|---|---|
| InterpScience | ~45% | **~75%** | ~80% | — |
| ICLR | ~10% | ~20% | ~35% | ~45% |

InterpScience is a strong bet *because* the results are partly negative — that is the venue's stated
purpose. ICLR remains hard: effect sizes are small, the central claim is contested by concurrent
work, and the clinical setting is a weaker draw there than at *ACL.

---

## 6. Reviewer-by-reviewer status

| Concern | Source | Status |
|---|---|---|
| SAE necessity vs matched non-learned baselines | AC, R1 | Run in full. Procedurally complete; substantively **partly refuted**. |
| PCA / ICA | AC | ✅ PCA done (≈ random). ICA correctly deferred. |
| diff-in-means directions | AC, R1 | ✅ Done, after fixing a defective first implementation. |
| supervised probes as directions | AC, R1 | ✅ Done. Yields a novel result (best-classifying ≠ best-auditing). |
| **"same explanation budget"** | AC | ❌ **Open — this is C6.** |
| TF-IDF | R2 W4, R3, AC | ❌ Answered honestly, and the honest answer is worse for the paper. |
| concordance circularity | R1, R2 W1, R3 #1 | ✅ Blind retrieval + 3 judge families. Strongest asset. |
| monospecificity scoping | R1 | ⚠️ Worse than R1 assumed — needs reframing, not just scoping. |
| YES-only + PARTIAL subtypes | R2 W2, AC | ⚠️ 101/32 split still unverified against `concordance_results.csv`. |
| ablation five requirements | R1 | ⚠️ Computed for **vanilla only**; absent for JumpReLU, GemmaScope, mean-ablation. |
| ablation renaming | R1, AC | Text change, not started. |
| single seed / single model | R1, R3, AC | ❌ Deferred; now the most dangerous deferral. |
| multiple-testing map | R1, AC | Verified: one BH family over d_sae × 46; off-target is a separate per-feature family. Text change. |
| GemmaScope claim softening, test-set EV wording | R2, AC | Text changes, not started. |
| Reproducibility = 2, Datasets = 1 | R1, R2 | ❌ C9 not started. Note: 42 commits on this branch carry `Co-Authored-By: Claude` trailers that become public on release. |

---

## 7. Recommended next four days

1. Fix the five numbers in §4 (hours, no compute).
2. Write the InterpScience paper around Contributions 1–2. Every table regenerates from `results/`.
3. Start C6 in parallel so it lands for ICLR rather than being rushed into the workshop version.
4. Price a second JumpReLU seed against the ICLR deadline.
