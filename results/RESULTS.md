# Central results ledger

Every experiment in this repository, in the order it was run, with the **headline
numbers of the most recent version of each**.

**How this file was built.** Experiments have been re-run and superseded repeatedly,
so the numbers here were recovered by walking the commit history, identifying the
latest version of each experiment, and reading the numbers out of the artifact that
version actually wrote — not out of prose written at the time. Every table below
names its source file. Where an experiment was superseded, the superseded run is kept
with a pointer to what replaced it, because "what changed" is itself a result.

**Reading conventions.**

- `r` / `|r|` — point-biserial correlation between a note-level feature value and a
  binary ICD-9 label. BH-FDR at *q* = 0.05 throughout.
- **Full corpus** = all 50,000 notes (312 shards). **Held-out** = shards 281–311,
  4,911 notes — the split the SAEs did not train on. Held-out is the number to cite.
- Cliff's δ is comparable across ablation arms; absolute nats are **not** (different
  arms carry different baselines). See §11.
- `–` = not run / not applicable. `n/a` = the metric does not exist for that arm.

> **✅ LLM-judge sections filled 2026-08-31; third judge added 2026-09-01.** §8, §10,
> §12 and §14 — everything that
> depends on an LLM explainer or judge — were re-run, and their numbers have now been
> read directly off the Modal `sae-artifacts` volume rather than off the stale local
> copies in `results/`. Each of those sections names the Modal path it was read from.
> Every section in this file is now current and read from the artifact its latest run
> wrote.

---

## Contents

| § | Date | Experiment | Status |
|---|---|---|---|
| [1](#1-data-and-activation-extraction) | 2026-04-23 → 04-30 | Extraction + centering | final |
| [2](#2-sae-training-and-reconstruction-quality) | 2026-05-01 → 05-19 | SAE training (vanilla, JumpReLU) + GemmaScope | final |
| [3](#3-icd-9-grounding--full-corpus-50000-notes) | 2026-05-20 | ICD-9 grounding, full corpus | superseded by §9 for citation |
| [4](#4-grounding-post-hoc--thresholds-monospecificity-length-confound) | 2026-05-20 | Threshold sweep / monospecificity / partial-r | final |
| [5](#5-surface-text-baselines) | 2026-05-20 → 05-21 | Lexical keyword + TF-IDF LR | final |
| [6](#6-baseline-3--raw-activation-lr-probe) | 2026-05-21 → 05-23 | Raw residual-stream LR probe | final |
| [7](#7-feature-inspection-token-level) | 2026-05-23 | Token-level feature inspection | final |
| [8](#8-auto-interpretability-jumprelu) | 2026-05-23 → 05-24 | LLM explanation + scoring + concordance | final — superseded for citation by §14 |
| [9](#9-held-out-test-split-grounding) | 2026-08-27 (pull) | Grounding on shards 281–311 | **current grounding numbers** |
| [10](#10-shuffled-explanation-control-scorer-null) | 2026-05-25 → 05-26 | LLM-scorer null baseline | final |
| [11](#11-causal-ablation) | 2026-05-24 → 09-01 | Zero / mean / section-local ablation + post-hoc — **[paper table §11.4](#114-paper-table--top-10-all-six-arms)** | final |
| [12](#12-multi-judge-concordance-and-blind-retrieval) | 2026-07-10 → 07-14 | 3-judge concordance + forced-choice retrieval | **superseded by §14** |
| [13](#13-sae-necessity-suite) | 2026-08-17 → 09-01 | 16 feature sources through one shared audit — **[paper table §13.1](#131-paper-table--per-code-grounding-13-sources-one-audit)** | **current necessity numbers** |
| [14](#14-four-arm-concordance-validation) | 2026-08-29 → 09-01 | 8 sources × **3 judges** × 5 metrics × \|r\| bands | **current concordance numbers** |
| [15](#15-bos-contamination-audit) | 2026-08-30 → 08-31 | `<bos>` floor in max-pooled grounding | ✅ **examined and closed** |
| [16](#16-directional-ablation-of-non-sae-sources) | 2026-08-31 | Causal necessity for random + diff-in-means | final |

---

## 1. Data and activation extraction

*2026-04-23 → 2026-04-30. Commits `1f85d67` → `ff97d80`.*

| Item | Value |
|---|---|
| Model | `google/gemma-2-2b` |
| Layer | 16 (residual stream) |
| `d_model` | 2304 |
| Notes | 50,000 (MIMIC-IV discharge summaries, stratified sample) |
| Extraction run id | `google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z` |
| Shards | 312 (`shard_0000`–`shard_0311`), fp16 safetensors |
| Tokens (eval sample) | 15,172,037 |
| Centering | two-pass exact global mean subtraction, float64 accumulator → `_centered` |
| ICD-9 panel | **46 codes**, prevalence ≥ 2%, fixed and pinned across every downstream experiment |
| Train / held-out split | shards 0–280 train, **281–311 held-out** (4,911 notes) |

Selection split used by the necessity suite: shards 0–30 (5,001 notes) for
best-of-*k* selection, 281–311 for audit. Never overlapping.

The pinned 46-code panel (`code_names.json`, prevalence-ordered):
`4019, 2724, 53081, 4280, 25000, 42731, 41401, V1582, 5849, 311, 2449, 40390,
2859, V5861, 3051, 2720, 5990, 32723, V5867, 5859, 30000, V4582, 412, V5866,
49390, 496, 27800, V4581, 2761, 73300, 486, V4986, 2749, 41400, V1251, 2851,
33829, V1254, 27651, 2762, 60000, 56400, 3572, 5856, 42789, 2875`.

---

## 2. SAE training and reconstruction quality

*2026-05-01 → 2026-05-19. Commits `12936c0`, `104f810`, `6216938`, `7872afa`.*

Three dictionaries are compared throughout the repository. Two are trained on
clinical activations; GemmaScope is an off-the-shelf general-purpose SAE included as
a domain-mismatch control.

**Final reconstruction quality** — source `.tmp/evaluations/*/diagnostic_metrics.json`,
measured on 15.17M centered tokens.

| SAE | flavour | `d_sae` | mean L0 | explained variance | dead-latent frac |
|---|---|---|---|---|---|
| **vanilla** | ReLU + L1 | 18,432 (8×) | **47.57** | **0.889** | 0.0001 |
| **JumpReLU** | L0 + STE | 18,432 (8×) | **40.92** | **0.906** | 0.0262 |
| **GemmaScope** | JumpReLU (pretrained, `width_16k/average_l0_42`) | 16,384 | 50.22 | **−4.21** | 0.0276 |

GemmaScope's negative EV is a genuine domain mismatch, not a bug — see the domain-shift
diagnostic below. Checkpoints: vanilla `sae_d2304_e8_l11e+01_20260505T205723Z/best`,
JumpReLU `jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/step_00036000`.

**JumpReLU training trajectory** — source `.tmp/jumprelu_training/eval_scan_summary.json`.
The threshold equilibrium the STE dual-path is designed to produce is visible as L0
falling 632 → 43.5 while mean threshold climbs 0.18 → 1.35 and EV stays flat from
step 8k.

| step | L0 | MSE | EV | dead frac | threshold mean | threshold std |
|---|---|---|---|---|---|---|
| 2,000 | 632.24 | 4528.34 | 0.8559 | 0.0003 | 0.179 | 0.009 |
| 4,000 | 182.48 | 3799.84 | 0.8747 | 0.0144 | 0.207 | 0.044 |
| 6,000 | 104.55 | 3445.91 | 0.8861 | 0.0313 | 0.249 | 0.087 |
| 8,000 | 83.56 | 3218.93 | 0.8934 | 0.0310 | 0.317 | 0.149 |
| 10,000 | 70.68 | 3138.75 | 0.8960 | 0.0311 | 0.420 | 0.240 |
| 12,000 | 60.43 | 3139.80 | 0.8959 | 0.0302 | 0.556 | 0.355 |
| 14,000 | 54.34 | 3141.89 | 0.8958 | 0.0303 | 0.696 | 0.445 |
| 16,000 | 51.10 | 3126.00 | 0.8963 | 0.0301 | 0.813 | 0.489 |
| 20,000 | 47.82 | 3095.64 | 0.8973 | 0.0303 | 0.996 | 0.505 |
| 24,000 | 45.82 | 3090.99 | 0.8975 | 0.0300 | 1.129 | 0.486 |
| 28,000 | 44.81 | 3056.93 | 0.8986 | 0.0302 | 1.226 | 0.455 |
| 32,000 | 44.11 | 3045.98 | 0.8990 | 0.0314 | 1.296 | 0.422 |
| **36,000** | **43.53** | **3042.34** | **0.8991** | 0.0317 | 1.348 | 0.392 |

**GemmaScope domain shift** — source `.tmp/evaluations/gemmascope/domain_shift_analysis.json`.

| quantity | value |
|---|---|
| clinical activation mean-norm | 131.70 |
| GemmaScope `b_dec` norm | 145.90 |
| ‖difference‖ / cosine similarity | 63.70 / **0.900** |
| EV, standard | −4.20 |
| EV, decode-only mean fix | −4.20 |
| EV, fully re-centered | **−6.56** |
| fraction of pre-activations above threshold | 0.711 |

Re-centering does not recover EV, so the mismatch is in the **feature directions**,
not a mean offset: GemmaScope's learned dictionary does not span the clinical
activation subspace.

---

## 3. ICD-9 grounding — full corpus (50,000 notes)

*2026-05-20. Commits `0cc85b7`, `d373b8b`. Source `results/{vanilla,jumprelu,gemmascope}/grounding_summary.json`.*

Note-level max-pooling of latent activations, point-biserial against each of the 46
codes, BH-FDR at *q* = 0.05. A latent is "grounded" if `max_c |r(latent, c)| > 0.1`.

| SAE | latents | tests | sig. after BH | frac sig. | grounded (\|r\|>0.1) | grounded frac | mean max\|r\| | median max\|r\| | **peak \|r\|** |
|---|---|---|---|---|---|---|---|---|---|
| **vanilla** | 18,432 | 847,872 | 630,491 | 0.744 | **8,293** | 0.450 | 0.1146 | 0.0922 | **0.8534** (L8823 ↔ `2449`) |
| **JumpReLU** | 18,432 | 847,872 | 636,217 | 0.750 | **9,023** | 0.490 | 0.1166 | 0.0986 | **0.8635** (L6701 ↔ `42731`) |
| **GemmaScope** | 16,384 | 753,664 | 472,215 | 0.627 | **5,749** | 0.351 | 0.0822 | 0.0700 | **0.5738** (L2121 ↔ `V5867`) |

> **Cite §9 instead.** These numbers include the notes the SAEs trained on. The
> held-out recomputation in §9 is the version to quote; it moves the headline by
> less than 0.01 in every case, which is itself the robustness result.

---

## 4. Grounding post-hoc — thresholds, monospecificity, length confound

*2026-05-20. Source `results/*/posthoc/posthoc_summary.json`.*

### 4.1 Threshold sweep — grounded latent count

| SAE | \|r\|>0.1 | >0.2 | >0.3 | >0.4 | >0.5 | >0.6 | >0.7 |
|---|---|---|---|---|---|---|---|
| vanilla | 8,293 | 2,036 | 673 | 299 | 142 | 71 | 26 |
| JumpReLU | 9,023 | 2,043 | 610 | 280 | 144 | 60 | 29 |
| GemmaScope | 5,749 | 329 | 48 | 12 | 4 | 0 | 0 |

The two domain-trained SAEs retain 1.7–1.8% of the dictionary at |r|>0.3;
GemmaScope retains 0.3%, a **13×** gap that widens with the threshold.

### 4.2 Monospecificity — fraction of grounded latents tied to exactly one code

| SAE | metric | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| vanilla | frac mono | 0.313 | 0.630 | 0.798 | 0.896 | 0.937 | **1.000** | 1.000 |
| vanilla | mean codes/latent | 3.69 | 1.71 | 1.28 | 1.12 | 1.07 | 1.00 | 1.00 |
| JumpReLU | frac mono | 0.301 | 0.622 | 0.785 | 0.879 | 0.938 | **1.000** | 1.000 |
| JumpReLU | mean codes/latent | 3.79 | 1.72 | 1.31 | 1.13 | 1.06 | 1.00 | 1.00 |
| GemmaScope | frac mono | 0.212 | 0.790 | 0.854 | 0.833 | 1.000 | – | – |
| GemmaScope | mean codes/latent | 4.17 | 1.35 | 1.17 | 1.17 | 1.00 | – | – |

Polyspecificity at |r|>0.1 is the max-pooling acuity confound; it resolves cleanly as
the bar rises. Every latent surviving |r|>0.6 is monospecific in both trained SAEs.

### 4.3 Partial correlation — `n_tokens` residualized out (the length confound)

| SAE | grounded @0.1 (raw → partial) | sig. frac (raw → partial) | mean max\|r\| (raw → partial) |
|---|---|---|---|
| vanilla | 8,293 → **4,741** (−43%) | 0.744 → 0.660 | 0.1146 → 0.0908 |
| JumpReLU | 9,023 → **5,147** (−43%) | 0.750 → 0.654 | 0.1166 → 0.0912 |
| GemmaScope | 5,749 → **894** (−84%) | 0.627 → 0.418 | 0.0822 → 0.0442 |

Peak associations are essentially untouched (vanilla 0.8534 → 0.8493; JumpReLU
0.8635 → 0.8533; GemmaScope 0.5738 → 0.5607). Length removal culls the weak tail,
not the strong latents — and it culls **84%** of GemmaScope's tail against 43% for
the trained SAEs.

Monospecificity after residualization (frac mono of grounded):

| SAE | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| vanilla | 0.478 | 0.706 | 0.819 | 0.914 | 0.969 |
| JumpReLU | 0.475 | 0.706 | 0.784 | 0.872 | 0.944 |
| GemmaScope | 0.596 | 0.818 | 0.872 | 0.818 | 1.000 |

---

## 5. Surface-text baselines

*2026-05-20 → 2026-05-21. Commits `8fd2f97`, `bcb69bf`, `fa852ff`.*

### 5.1 Lexical keyword co-occurrence

*Source `results/*/lexical/lexical_baseline_summary.json`. 50,000 notes, 46 codes,
Δ|r| threshold 0.05.*

| SAE | SAE above lexical | comparable | lexical above SAE | frac SAE wins | mean Δ\|r\| | median Δ\|r\| | median keyword-absent recall |
|---|---|---|---|---|---|---|---|
| **vanilla** | **44** / 46 | 2 | **0** | 0.957 | **+0.308** | +0.315 | 0.532 |
| **JumpReLU** | **45** / 46 | 1 | **0** | 0.978 | **+0.287** | +0.315 | 0.351 |
| **GemmaScope** | 25 / 46 | 9 | **12** | 0.543 | +0.034 | +0.084 | 0.984 |

Both trained SAEs beat the keyword baseline on essentially every code and are never
beaten. GemmaScope loses on 12 codes. Keyword-absent recall — does the best latent
still fire on positive notes containing no matching keyword — is 0.35–0.53 for
trained SAEs, evidence of beyond-surface representation; GemmaScope's 0.98 reflects
a dense, largely code-unspecific latent rather than a stronger result.

### 5.2 TF-IDF + logistic regression (per-code stratified 5-fold CV)

*Source `results/vanilla/tfidf_lr/tfidf_lr_summary.json`,
`.tmp/evaluations/jumprelu/tfidf_lr/tfidf_lr_summary.json`. 10,000 TF-IDF features
(1+2-grams, sublinear TF) vs the SAE's pooled latents.*

| comparator | mean AUC-ROC (TF-IDF / SAE) | codes TF-IDF wins / comparable / SAE wins | Wilcoxon *p* | median Δ (SAE − TF-IDF) |
|---|---|---|---|---|
| vanilla SAE | 0.9169 / **0.8813** | 33 / 13 / 0 | ≈ 0 | **−0.0333** |
| JumpReLU SAE | 0.9169 / **0.8882** | 30 / 16 / 0 | ≈ 0 | −0.0266 |

| comparator | mean AUC-PR (TF-IDF / SAE) | wins / comparable / SAE wins | Wilcoxon *p* | median Δ |
|---|---|---|---|---|
| vanilla SAE | 0.5900 / 0.5321 | 33 / 12 / 1 | ≈ 0 | −0.0585 |
| JumpReLU SAE | 0.5900 / 0.5434 | 32 / 8 / 6 | 5e-06 | −0.0484 |

**Supplementary — best single feature by |r|** (the interpretability-relevant
comparison, as opposed to the full-classifier one):

| comparator | mean best \|r\| SAE / TF-IDF | SAE above / comparable / TF-IDF above |
|---|---|---|
| vanilla | **0.579** / 0.519 | **23** / 22 / 1 |
| JumpReLU | **0.566** / 0.519 | **21** / 22 / 3 |

The honest reading, unchanged since: **TF-IDF classifies better** (a 10k-dim
supervised bag of words should), while the **SAE's single best feature is a stronger
individual concept detector**. TF-IDF is later promoted from baseline to an *audited
source* in §13, where it turns out to be competitive on specificity too.

---

## 6. Baseline 3 — raw-activation LR probe

*2026-05-21 → 2026-05-23. Commits `89b8cdd` → `0baa9b2`. Source
`.tmp/baseline3/vanilla_solo/raw_lr_summary.json`.*

Max-pools the **raw centered layer-16 activations** to note level (2304-dim, no SAE)
and trains per-code stratified 5-fold logistic regression. This is the
"how much does the SAE add over the residual stream itself?" floor.

| features | pooling | codes | CV folds | mean AUC-ROC | mean AUC-PR |
|---|---|---|---|---|---|
| **raw residual stream, 2304-dim** | max | 46 | 5 | **0.8082** | 0.3526 |
| vanilla SAE, 18,432-dim (§5.2) | max | 46 | 5 | 0.8813 | 0.5321 |
| TF-IDF, 10,000-dim (§5.2) | — | 46 | 5 | 0.9169 | 0.5900 |

The SAE adds **+0.073 AUC-ROC** over the raw residual stream it decomposes. This
run's `raw_shard_ckpt/` is the pooled-X matrix reused by every direction-based
source in §13 (diff-in-means, probes, PCA).

---

## 7. Feature inspection (token level)

*2026-05-23. Commit `db07405`. Source `.tmp/feature_inspection/*/feature_inspection_report.json`.
Top 20 grounded (latent, code) pairs per SAE, top-50 tokens each, 20 shards sampled.*

| SAE | latents inspected | mean diversity score | mean firing-rate ratio (ICD+ / ICD−) |
|---|---|---|---|
| vanilla | 20 | 0.091 | **10.39×** |
| JumpReLU | 20 | 0.087 | **13.62×** |

Diversity ≈ 0.09 means the top-50 firing tokens of a grounded latent are dominated by
one or two token types (e.g. a `2449` latent firing 50/50 on `thyroidism` across 49
distinct notes) — high token concentration, broad note coverage. Grounded latents
fire 10–14× more often on notes carrying their code than on notes without it.

---

## 8. Auto-interpretability (JumpReLU)

*Run 2026-05-23 → 2026-05-24, judged 2026-07. **Pulled from Modal 2026-08-31.** Source
`sae-artifacts:/out/auto_interp/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/`
— `categorization_summary.json`, `scorer_summary.json`, `concordance_summary.json`.
964 JumpReLU latents explained. Explainer, scorer and concordance judge are all
Claude Sonnet 4.6.*

### 8.1 Category distribution by tier

| tier | n | clinical_concept | clinical_vocabulary | structural_pattern | general_language | noise | unknown |
|---|---|---|---|---|---|---|---|
| **global** | 964 | 150 | 179 | 137 | 13 | **483** | 2 |
| strong_grounded | 280 | **132** | **120** | 26 | 2 | 0 | 0 |
| weak_grounded | 100 | 14 | 39 | 35 | 10 | 0 | 2 |
| non_grounded | 484 | 0 | 0 | 1 | 0 | **483** | 0 |
| dead | 100 | 4 | 20 | **75** | 1 | 0 | 0 |

The tiers separate cleanly and in the expected direction. 90% of strong-grounded
latents (252 / 280) are clinical — concept or vocabulary — and **none** are noise.
Every one of the 483 `noise` labels lands in the non-grounded tier. Dead latents are
overwhelmingly `structural_pattern` (75 / 100): formatting and template artifacts,
not clinical content.

### 8.2 Explanation scores by tier (Fuzzing / Detection, Paulo et al. 2024)

| tier | mean Fuzzing | n | mean Detection | n |
|---|---|---|---|---|
| **global** | 0.938 | 432 | 0.962 | 459 |
| strong_grounded | 0.938 | 268 | 0.960 | 275 |
| weak_grounded | 0.941 | 94 | 0.955 | 90 |
| dead | 0.932 | 69 | **0.975** | 93 |
| non_grounded | 1.000 | 1 | 1.000 | 1 |

`n` is the count of features that produced a valid score, not the tier size; the
scorer errored or returned unparseable output on the rest.

**These scores are near-ceiling and flat across tiers.** A dead latent's explanation
scores as well as a strong-grounded one's (0.932 vs 0.938 Fuzzing; Detection is
*higher* for dead latents). Fuzzing/Detection as run here does not discriminate real
structure from none, which is the finding that motivated §10 — and, once §10 showed
the scorer is not broken, motivated abandoning scorer-based validation for the
label-anchored designs in §12 and §14.

### 8.3 ICD-9 concordance — does the explanation name the code the latent correlates with?

*380 features judged (280 strong-grounded, 100 weak-grounded).*

| band | n | YES | PARTIAL | NO | UNKNOWN | **YES+PARTIAL** | **exact-YES** |
|---|---|---|---|---|---|---|---|
| global | 380 | 85 | 238 | 46 | 11 | **85.0%** | **22.4%** |
| \|r\| > 0.3 | 280 | 85 | 180 | 7 | 8 | **94.6%** | **30.4%** |
| \|r\| > 0.5 | 144 | 61 | 81 | 1 | 1 | **98.6%** | **42.4%** |

The `r>0.4` row in `concordance_summary.json` is byte-identical to `r>0.3`: the
380-feature pool contains no feature in [0.3, 0.4), so both slices select the same
280 features. It is omitted rather than duplicated.

Concordance rises monotonically with |r| on both metrics, which is the right shape.
But YES+PARTIAL saturates at 98.6% by |r| > 0.5 — near its ceiling — while exact-YES
sits at 42.4%. §12 shows the saturation is judge-independent, and §14 shows what
happens to the pooled metric when a random-direction control is run through it.

---

## 9. Held-out test-split grounding

*Pulled 2026-08-27 (`4abdb11`); module `test_split_eval.py`. Source
`results/{vanilla,jumprelu,gemma}/test_split/`. Shards 281–311, 4,911 notes,
recomputed from existing pooled vectors — no re-encode.*

**These are the grounding numbers to cite.**

| SAE | notes | grounded (\|r\|>0.1) | grounded frac | mean max\|r\| | median max\|r\| | **peak \|r\|** |
|---|---|---|---|---|---|---|
| **vanilla** | 4,911 | **8,985** | 0.488 | 0.1202 | 0.0984 | **0.8595** (L10520 ↔ `5856`) |
| **JumpReLU** | 4,911 | **9,721** | 0.527 | 0.1214 | 0.1034 | **0.8643** (L6701 ↔ `42731`) |
| **GemmaScope** | 4,911 | **5,790** | 0.353 | 0.0894 | 0.0769 | **0.5450** (L2121 ↔ `V5867`) |

Threshold sweep and monospecificity, held-out:

| SAE | >0.1 | >0.2 | >0.3 | >0.4 | >0.5 | >0.6 | >0.7 | frac mono @0.3 | frac mono @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 8,985 | 2,063 | 675 | 291 | 143 | 73 | 29 | 0.816 | 0.944 |
| JumpReLU | 9,721 | 2,075 | 610 | 276 | 147 | 61 | 28 | 0.793 | 0.925 |
| GemmaScope | 5,790 | 295 | 54 | 13 | 4 | 0 | 0 | 0.870 | 1.000 |

Partial correlation (`n_tokens` out), held-out: vanilla 8,985 → 5,355; JumpReLU
9,721 → 5,711; GemmaScope 5,790 → 1,039.

**Train/held-out agreement is close** — every headline moves by < 0.01 |r| and every
grounded count by < 10%. The SAEs are not memorising their training notes.

---

## 10. Shuffled-explanation control (scorer null)

*2026-05-25 → 2026-05-26 (PR #7, `4d21c80`); re-run and **pulled from Modal
2026-08-31**. Source `.../shuffled_control/shuffled_control_summary.json` in the §8
run directory. Scorer Claude Sonnet 4.6. 481 features eligible, 138 scoring errors.
Reference null: Paulo et al. (2024) ≈ 0.51.*

**The question.** §8.2 showed Fuzzing/Detection scores near 1.0 for every tier,
including dead latents. Two explanations: either the scorer is not measuring anything
(it says yes to everything), or the explanations really are that good and the metric
has no headroom left. This control decides between them by re-scoring each feature's
contexts against a **deliberately wrong** explanation — one taken from a different
feature — under two derangement schemes: `global` (any other feature) and
`within_tier` (another feature in the same tier, the harder control).

| scorer | scheme | mean real | mean shuffled | **Δ** | 95% CI (shuffled) | Wilcoxon *p* | n |
|---|---|---|---|---|---|---|---|
| Fuzzing | global | 0.931 | **0.496** | **0.436** | [0.489, 0.502] | 0.0 † | 274 |
| Fuzzing | within-tier | 0.931 | **0.493** | **0.438** | [0.486, 0.501] | 0.0 † | 278 |
| Detection | global | 0.961 | **0.522** | **0.439** | [0.506, 0.538] | 0.0 † | 285 |
| Detection | within-tier | 0.957 | **0.516** | **0.441** | [0.501, 0.531] | 0.0 † | 270 |

† The paired Wilcoxon statistic underflows to exactly `0.0` in the artifact at these
sample sizes and effect magnitudes. Reported as written rather than substituted.

Broken out by tier (`global` scheme):

| tier | Fuzzing real | Fuzzing shuffled | n | Detection real | Detection shuffled | n |
|---|---|---|---|---|---|---|
| strong_grounded | 0.932 | 0.500 | 162 | 0.957 | 0.523 | 167 |
| weak_grounded | 0.935 | 0.489 | 63 | 0.948 | 0.533 | 61 |
| dead | 0.923 | 0.490 | 48 | 0.987 | 0.505 | 56 |

**The scorer is not broken.** Every shuffled arm lands on 0.49–0.52 — chance on a
two-alternative task, and within noise of the 0.51 Paulo et al. report — while real
explanations score 0.93–0.96. The Δ ≈ 0.44 is stable across both scorers, both
derangement schemes, and every tier. A wrong explanation is reliably rejected.

**But the ceiling problem survives it.** The control proves the scorer discriminates
*right from wrong* explanations; it does not give the scorer the resolution to
separate a *good* explanation from a *mediocre* one, which is what §8.2 needed. Note
the dead-latent row: shuffled 0.490/0.505 (chance, correctly) but real 0.923/0.987 —
the scorer confidently endorses explanations of latents that never fire. An
explanation of a formatting artifact can be perfectly accurate. That is why validation
moved to the label-anchored designs of §12 and §14, where the target is an external
ICD-9 code rather than the feature's own contexts.

---

## 11. Causal ablation

*2026-05-24 → 2026-08-17. Commits `bf5a49c`, `f9d52bc`, `74e3650`, `6fd4e81`, `c0f06eb`.
Source `results/ablation/*/`. All runs: 4,911 held-out notes, loss window = final 25%
of tokens, Mann-Whitney U one-sided, Cliff's δ, BH across all targets.*

For each (feature, code) target: subtract the feature's residual-stream contribution,
re-run layers 17+, and test whether CE loss rises **more on code-positive than on
code-negative notes**.

> **→ The consolidated paper table is [§11.4](#114-paper-table--top-10-all-six-arms).**
> §11.1–11.3 are the full run inventory behind it.

### 11.0 What `pilot` / `extended` mean — read this before the tables

Run names encode **which slice of the grounded-feature ranking was ablated**, ranked
by |r_pb|. They are disjoint tranches, not nested sizes:

| suffix | feature ranks by \|r\| | grounded targets | notes |
|---|---|---|---|
| `_smoke` | top 2 | 2 | plumbing check, 250 notes — not a result |
| **`_pilot`** | **1–10** (top 10) | **10** | + 1 random control + 1 low-r control = 12 targets |
| `_pilot_extended` | **11–30** (the *next* 20) | 20 | no controls; a weaker tranche by construction |
| `vanilla_meanabl` / `vanilla_section` | **1–30** (top 30) | 30 | + 2 controls = 32 targets; the union of the two above |

**`_pilot_extended` is not a superset of `_pilot`** — it is the ranks below it. That
is the whole reason its median δ is roughly half (`vanilla` 0.118 vs 0.300): it
ablates weaker features, not more of the same ones. Reading the two as "small run vs
big run" inverts the result.

**Which to report.** Either the **top 10** (`*_pilot`) or the **top 30**
(`vanilla_meanabl` / `vanilla_section`) is a defensible unit — both are complete,
contiguous slices from rank 1. **Top 10 is the better choice** and is what this file
treats as the headline: the effect is cleanest there (δ = 0.300, 10/10 BH-significant,
**0/588** off-target significant, 12.4× specificity), 10–20 targets is squarely within
convention for this kind of ablation, and extending to 30 dilutes the median with
features that are weakly grounded to begin with. Quote the top-30 runs when the
question is how far down the ranking the effect persists.

**Never quote `_pilot_extended` on its own.** It is ranks 11–30 in isolation and
reads as a much weaker result than the method produces; it is only meaningful as the
tail of a top-30 report.

### 11.1 All runs

| run | SAE | intervention | targets | grounded / controls | **median Cliff's δ (grounded)** | δ (controls) | BH-sig *q*<0.05 | recon tax (nats) |
|---|---|---|---|---|---|---|---|---|
| `vanilla_smoke` | vanilla | zero | 4 | 2 / 2 | 0.195 | −0.070 | 1/4 | 0.027 |
| `gemma_scope_smoke` | GemmaScope | zero | 4 | 2 / 2 | 0.260 | −0.297 | 1/4 | 0.643 |
| `vanilla_pilot` | vanilla | zero | 12 | 10 / 2 | **0.300** | −0.036 | **10/12** | 0.029 |
| `jumprelu_pilot` | JumpReLU | zero | 12 | 10 / 2 | **0.276** | +0.048 | **11/12** | 0.009 |
| `gemma_scope_pilot` | GemmaScope | zero | 12 | 10 / 2 | 0.195 | −0.013 | 7/12 | **0.648** |
| `vanilla_pilot_extended` | vanilla | zero | 20 | 20 / 0 | 0.118 | – | 15/20 | 0.029 |
| `jumprelu_pilot_extended` | JumpReLU | zero | 20 | 20 / 0 | 0.090 | – | 15/20 | 0.009 |
| `gemma_scope_pilot_extended` | GemmaScope | zero | 20 | 20 / 0 | 0.169 | – | 15/20 | 0.648 |
| **`vanilla_meanabl`** | vanilla | **mean** | 32 | 30 / 2 | **0.169** | −0.012 | **23/32** | 0.029 |
| **`vanilla_section`** | vanilla | zero, section-local | 32 | 30 / 2 | **0.162** | −0.036 | **25/32** | 0.029 |

Target tranches per §11.0: `_pilot` = ranks 1–10, `_pilot_extended` = ranks 11–30,
the 32-target runs = ranks 1–30 plus 2 controls. **`vanilla_pilot` (top 10) is the
headline SAE arm**; `vanilla_meanabl` / `vanilla_section` are the top-30 view.

**Zero vs mean ablation agree** (0.162 vs 0.169, paired Wilcoxon *p* = 0.371), and
mean-ablation is if anything cleaner on specificity (0 vs 2 off-target significant).
Mean-ablation is therefore primary for SAE-feature arms.

*Note the one gap this leaves.* The two preferences point at different runs: top-10 is
the preferred tranche (§11.0) but the only mean-ablation run is the top-30
`vanilla_meanabl`. **There is no top-10 mean-ablation run.** Since the two
interventions are statistically indistinguishable, `vanilla_pilot` (top 10, zero) is
quoted as the headline without qualification — but if a reviewer asks for top-10 under
mean-ablation specifically, it has not been run.

**GemmaScope's reconstruction tax is 0.648 nats against vanilla's 0.029 — 22×** — the
same domain mismatch §2 measures as negative EV, now in loss units.

### 11.2 Post-hoc specificity (`ablation_posthoc.py`, no GPU)

*Source `results/ablation/*/posthoc_specificity/ablation_posthoc_summary.json`.
Off-target restricted to true-negative notes.*

| run | mean on-target δ | mean \|off-target δ\| | **median specificity ratio** | off-target sig. | δ after length adjustment | attenuation | still sig. |
|---|---|---|---|---|---|---|---|
| `vanilla_pilot` | **0.352** | 0.025 | **12.39×** | **0** | 0.297 | 16% | 9/10 |
| `jumprelu_pilot` | 0.246 | 0.024 | 11.10× | 4 | 0.191 | 22% | 6/10 |
| **`gemma_scope_pilot`** | 0.221 | 0.037 | **4.58×** | **22** | 0.166 | 25% | 6/10 |
| `vanilla_pilot_extended` | 0.169 | 0.028 | 4.75× | 2 | 0.097 | 42% | 11/20 |
| `jumprelu_pilot_extended` | 0.110 | 0.025 | 3.83× | 9 | 0.065 | 41% | 10/20 |
| **`vanilla_meanabl`** | 0.222 | 0.029 | 6.05× | **0** | 0.169 | 24% | 20/30 |
| **`vanilla_section`** | 0.230 | 0.027 | 6.38× | 2 | 0.164 | 29% | 19/30 |

`gemma_scope_pilot` was run 2026-09-01 (`configs/ablation_posthoc_gemmascope.yaml`,
CPU, minutes) to close the last gap in the top-10 cross-SAE row. **It has the worst
specificity of any SAE — 22 of 490 off-target tests significant, more than the
label-supervised diff-in-means baseline's 19.** A general-purpose SAE ablated on
clinical notes leaks more than a supervised direction does.

Effect-size calibration (nats), `vanilla_pilot`: mean on-target Δ = 0.00202 nats =
0.124% of base loss = 0.069× the reconstruction tax. Effects are statistically clean
but **small in absolute terms** — report δ, and state the nats alongside.

### 11.3 Section-local specificity (`vanilla_section` only)

| quantity | value |
|---|---|
| notes with an identified section | 4,695 |
| mean δ, in-section | 0.041 |
| mean δ, rest of note | 0.537 |
| mean concentration (δ) | −0.496 |
| features with in-section δ > rest | 3.3% |
| mean nats, in-section / rest | 0.000796 / 0.001434 |
| mean nats concentration | −0.000637 |
| features with in-section nats > rest | **43.3%** |

Cliff's δ says effects are *not* section-local; the size-invariant nats magnitude says
they are roughly balanced (43% vs 50% under a null). δ's noise floor scales with
region size, so **δ alone is size-confounded across regions** — this is why both are
reported. The honest statement: ablation effects are distributed across the note, not
concentrated in the diagnosis section.

---


### 11.4 Paper table — top-10, all six arms

*The consolidated causal result. Every source contributes **its own top-10 grounded
targets** (+ 2 controls), all on the same 4,911 held-out notes, same loss window,
same statistics. BH recomputed within each 12-target family so no row inherits
another run's correction. Derived 2026-09-01 from the per-target CSVs in
`results/ablation/*/` — see the provenance note below.*

| source | protocol | \|r\| range | **δ grounded** | δ controls | **BH-sig** | on-target δ | **specificity** | **off-target sig** | δ length-adj | still sig | nats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **vanilla SAE** | recon · mean | 0.822–0.860 | **0.312** | −0.012 | **10/10** | 0.357 | **12.44×** | **0/490** | 0.311 | **10/10** | 0.00202 |
| vanilla SAE | recon · zero | 0.822–0.860 | 0.300 | −0.036 | 10/10 | 0.352 | 12.39× | **0/490** | 0.297 | 9/10 | 0.00202 |
| **JumpReLU SAE** | recon · zero | 0.799–0.864 | **0.276** | +0.048 | **10/10** | 0.246 | **11.10×** | 4/490 | 0.191 | 6/10 | 0.00054 |
| **GemmaScope SAE** | recon · zero | 0.428–0.545 | 0.195 | −0.013 | 7/10 | 0.221 | 4.58× | 22/490 | 0.166 | 6/10 | 0.00169 |
| diff-in-means | directional | 0.386–0.699 | 0.055 | −0.016 | 4/10 | 0.045 | 1.17× | 19/490 | 0.055 | 4/10 | 0.00043 |
| random-matched | directional | 0.309–0.432 | **−0.096** | −0.083 | **0/10** | −0.090 | **−1.53×** | 7/490 | −0.005 | **0/10** | 0.01852 |

**What it shows.** The two domain-trained SAEs are the only sources with a large,
BH-significant, length-robust, specific causal effect. Diff-in-means produces real but
diffuse signal at ~1/6 the effect size and ~1/11 the specificity. Random-matched
directions produce **nothing** — 0/10 significant, δ indistinguishable from its own
controls (−0.096 vs −0.083), and 94% of the raw effect vanishes under length
adjustment. GemmaScope sits between: a genuine effect, but the **worst specificity of
any arm bar the two baselines**, leaking on more off-target codes (22) than the
label-supervised direction does (19).

**Four things that must travel with this table.**

1. **δ is comparable across every row; nats and specificity are not comparable across
   the `recon` / `directional` line.** Directional arms have no reconstruction, so
   their baseline is `l_clean` and `recon_tax = 0`; the SAE arms measure against
   `l_recon`. Cliff's δ is a within-arm rank contrast, so a per-arm additive baseline
   offset cancels — every cross-arm claim above is stated in δ.
2. **This is a best-vs-best comparison, not an |r|-matched one.** Each arm is at its
   own top-10, so the rows sit at different correlation strengths — hence the |r|
   column. An |r|-matched version does not exist: GemmaScope has 4 held-out features
   above |r| 0.5 and random-matched none above 0.44 (§13.2), so no arm but the two
   domain SAEs can populate a strength-matched top-10. The scarcity is itself a
   result, but a δ gap between rows is **not** attributable to architecture alone.
   Full statement of the caveat, including the code-panel version of it, in
   `docs/causal_ablation_positioning.md` §3.
3. **Off-target is grounded-only, `/490` = 10 targets × 49 codes.** `total_off_target_sig_q05`
   in the summary JSONs uses this convention. §16.1 quotes `/588`, which is
   grounded + controls (12 × 49) — the same underlying counts, a different
   denominator. Do not mix them in one table.
4. **Zero vs mean ablation is the intervention-robustness check, not two results.** On
   vanilla's *identical* ten targets the two agree to within 0.013 δ and 0.05×
   specificity, both at 0/490 (paired Wilcoxon on the matched-30 set, *p* = 0.371).
   That is what licenses reporting zero-ablation for JumpReLU and GemmaScope rather
   than re-running them under mean-ablation.

**Provenance.** `vanilla_meanabl` was run over the top *30*; its first ten targets are
byte-identical to `vanilla_pilot`'s ten grounded ones in the same rank order (features
10520, 8823, 11055, 3485, 12655, 3569, 10471, 9546, 2868, 13748) with the same two
controls, so the top-10 mean-ablation row is a restriction of the existing per-target
CSVs, not a new run. `gemma_scope_pilot`'s post-hoc was run 2026-09-01.
`random_matched_full` / `diff_in_means_full` were pulled from Modal the same day.

**Not in this table:** the `*_pilot_extended` runs (ranks 11–30) and the top-30
`vanilla_meanabl` / `vanilla_section` aggregates — see §11.0 for why ranks 11–30 must
not be quoted alone, and §11.1–11.3 for those numbers.

---

## 12. Multi-judge concordance and blind retrieval

*2026-07-10 → 2026-07-14 (PR #9, `9b93ac2`). **Superseded by §14 — cite §14, not this
section.** Retained because what this protocol got wrong is what §14 was built to fix.*

**What it ran.** The 380 JumpReLU features and explanations of §8.3, re-judged by three
judges (Claude Sonnet 4.6, GPT-4o, DeepSeek-V3) on two protocols: the pooled
YES+PARTIAL concordance verdict, and a blind 1-of-9 forced choice whose seven
distractors were drawn from **other ICD-9 chapters**.

**What it established.** Judges agree on the *ordering* — all three rise monotonically
with |r| on both protocols. They do not agree on the *level*: on pooled concordance the
judge spread reached 10.8 points (YES+PARTIAL) and 11.3 points (exact-YES) on identical
inputs. Forced-choice retrieval cut that spread to ≤ 4.2 points, because a judge that
must pick one of nine options cannot choose how generously to read a partial match.
Forced choice is the more reliable protocol; that conclusion carries into §14.

**Its three defects, all fixed in §14.**

1. **Cross-chapter distractors are too easy.** An explanation mentioning "heart" rules
   out seven decoys without identifying *which* cardiac code. §14 draws distractors
   from the correct code's own chapter, forcing diagnosis-level discrimination.
2. **The pooled YES+PARTIAL metric has no floor.** Two of three judges hit exactly
   100.0% at |r| > 0.5 here. §14.5 runs random directions through the same metric and
   they score 79–100%, above the trained SAEs — so the numbers this protocol produced
   cannot distinguish a good feature source from an arbitrary one.
3. **No non-SAE comparison.** Every arm was the same JumpReLU SAE, so nothing here
   speaks to whether the result is specific to SAEs. §14 runs eight sources —
   including a random floor and a keyword ceiling — through one identical pipeline.

Numbers from this run are on Modal at `.../arm0_eval/` and `.../retrieval_eval/` in the
§8 run directory and are deliberately not reproduced here: they would compete with §14
while measuring something weaker.

---

## 13. SAE-necessity suite

*2026-08-17 → 2026-08-28. Commits `6911ae1`, `153c04e`, `f013921`, `d7a9c8d` (C1),
`708f6ea` (C2), `c5b45b7` (C3), `629b010` (C5), `e5a7173` (C4), `36b1e18` (C8).*

The question the baselines in §5–6 do not answer: **is an SAE needed to produce these
audit signals at all?** Answering it requires running non-SAE feature sources through
the *identical* audit. `necessity_audit.py` is a source-agnostic harness consuming any
`[n_notes × k]` matrix in the `shard_ckpt` format, so "identical" is structural rather
than asserted.

**Shared protocol for every row below.** Selection on shards 0–30 (5,001 notes),
audit on shards 281–311 (4,911 notes) — never overlapping, so best-of-*k* selection
is not scored on its own notes. The fixed 46-code panel. `r_threshold` 0.1, BH *q*
0.05, `min_off_pos` 10, off-target correlations restricted to **c-negative notes** so
genuine comorbidity cannot masquerade as non-specificity. One feature per code
(`top_per_code` for high-*k* sources, `identity` for one-direction-per-code sources).

### 13.1 Paper table — per-code grounding, 13 sources, one audit

*Held-out audit split (shards 281–311, n = 4,911 notes); features selected on a
**disjoint** 5,001 notes (shards 0–30); pinned 46-code ICD-9 panel; exactly one
feature selected per code. Sorted by descending median.*

**Columns.** ***k*** — candidate features the source offers. **median** — median
|r_pb| across the 46 selected features, **no strength threshold applied**, so it
describes the association a source finds for an *arbitrary* code, not the typical
association among its strong ones. **peak** — largest |r_pb| any feature of that
source attains on any code. **leakage** — median |r_pb| between a feature selected
for code *c* and each of the other 45 codes, computed **only on notes that do not
carry *c***, so genuine comorbidity is not scored as non-specificity. **ratio** —
on-target strength ÷ leakage. **@0.1 / @0.3 / @0.5 / @0.6** — features across the
source's **full dictionary** whose strongest BH-significant association exceeds that
|r_pb|, hence bounded above by *k*.

| source | *k* | median | peak | leakage | ratio | @0.1 | @0.3 | @0.5 | @0.6 |
|---|---|---|---|---|---|---|---|---|---|
| JumpReLU SAE | 18,432 | **0.574** | **0.864** | 0.0373 | 14.94× | 9,721 | 610 | 147 | 61 |
| vanilla SAE | 18,432 | **0.574** | 0.859 | 0.0324 | 15.88× | 8,985 | 675 | 143 | 73 |
| TF-IDF (binary) | 10,000 | 0.531 | 0.831 | 0.0273 | **18.02×** | 2,879 | 265 | 60 | 29 |
| TF-IDF (value) | 10,000 | 0.518 | 0.848 | **0.0251** | 17.02× | 2,614 | 250 | 58 | 27 |
| difference-in-means (LDA) | 46 | 0.339 | 0.699 | 0.0572 | 5.68× | 46 | 28 | 7 | 2 |
| probe LR (unweighted) | 46 | 0.333 | 0.637 | 0.0885 | 4.42× | 46 | 31 | 5 | 2 |
| probe LR (balanced) | 46 | 0.327 | 0.630 | 0.0921 | 3.67× | 46 | 32 | 5 | 2 |
| GemmaScope SAE | 16,384 | 0.309 | 0.545 | 0.0468 | 6.29× | 5,790 | 54 | 4 | 0 |
| random (note-matched) | 18,432 | 0.191 | 0.432 | 0.0558 | 3.07× | 6,945 | 16 | 0 | 0 |
| random (L0-matched) | 18,432 | 0.149 | 0.314 | 0.0751 | 2.12× | 9,132 | 1 | 0 | 0 |
| PCA | 2,304 | 0.141 | 0.441 | 0.0484 | 2.88× | 256 | 1 | 0 | 0 |
| difference-in-means (diagonal) | 46 | 0.127 | 0.310 | 0.0938 | 1.37× | 46 | 3 | 0 | 0 |
| difference-in-means (plain) | 46 | 0.121 | 0.291 | 0.0940 | 1.25× | 46 | 0 | 0 | 0 |

**Excluded by design.** Every `[no-BOS]` row — that is the BOS-free cross-check
(§15.1), not a competing method — and **random (dense)**, superseded by the
note-matched arm, which is the only random arm that reproduces the SAE's *note-level*
firing density (0.656 vs the SAE's 0.675; dense is 1.00 and token-level L0-matching
still leaves 0.95, because max-pooling over thousands of tokens washes it out).
There is deliberately **no @0.7 column**: it exists only in the SAE test-split
artifact and cannot be filled for the ten non-SAE sources.

**Reading it.** The two domain SAEs lead on median and peak, and own the high-|r|
regime outright — 143–147 features above |r| 0.5, where every non-SAE source except
TF-IDF has 0–7. TF-IDF wins **leakage and ratio** at within 0.05 median: it is the
one genuine rival, and §13.7 states what the SAEs retain against it. The @0.1 column
is not evidence of anything — random directions reach 6,945–9,132 there, and §15.1
shows that count is not even stable to a re-pool that moves the correlations by
0.002. The columns that separate sources are **@0.3 and above**.

*Provenance: `audit_summary.json` (k, median, peak, ratio) + `monospecificity.json`
(counts) per source; leakage is the median of `mean_abs_off_r` over the 46 rows of
each source's `off_target_summary.csv` — verified to reproduce the published
`median_mean_abs_off_r` exactly for all five sources where both exist. The
note-matched and TF-IDF off-target CSVs were pulled from Modal 2026-09-01; they are
not in `comparison_summary.json` because `scripts/build_necessity_comparison.py:53`
registers only one random arm per sparsity family and no TF-IDF arm.*

#### Full inventory — 16 rows, including the BOS cross-check

*Source `results/necessity/comparison/comparison_summary.json`. Sorted by specificity
ratio. "leakage" = median mean-|off-target r|. Retained because §15.1 cites the
`[no-BOS]` rows; **note-matched is absent from this table**, see the provenance note
above.*

| rank | method | *k* | median on-target \|r\| | median leakage | **specificity ratio** | median n off-sig |
|---|---|---|---|---|---|---|
| 1 | **TF-IDF (binary)** | 10,000 | 0.531 | – | **18.03×** | **1.0** |
| 2 | **TF-IDF (value)** | 10,000 | 0.519 | – | **17.02×** | **0.5** |
| 3 | **vanilla SAE** | 18,432 | **0.574** | 0.0324 | **15.88×** | **2.0** |
| 4 | **JumpReLU SAE** | 18,432 | **0.574** | 0.0373 | **14.94×** | 3.0 |
| 5 | GemmaScope SAE | 16,384 | 0.309 | 0.0468 | 6.29× | 4.0 |
| 6 | diff-in-means (LDA / full whitening) | 46 | 0.339 | 0.0572 | 5.68× | 8.0 |
| 7 | diff-in-means (LDA) [no-BOS] | 46 | 0.341 | 0.0577 | 5.67× | 8.0 |
| 8 | probe LR (unweighted) | 46 | 0.333 | 0.0885 | 4.42× | 17.0 |
| 9 | probe LR (balanced) | 46 | 0.327 | 0.0921 | 3.67× | 16.5 |
| 10 | random (dense) [no-BOS] | 18,432 | 0.227 | 0.0784 | 3.05× | 13.0 |
| 11 | random (dense) | 18,432 | 0.219 | 0.0786 | 3.00× | 13.0 |
| 12 | PCA | 2,304 | 0.141 | 0.0484 | 2.88× | 4.0 |
| 13 | random (L0-matched) | 18,432 | 0.149 | 0.0751 | 2.12× | 12.0 |
| 14 | random (L0-matched) [no-BOS] | 18,432 | 0.147 | 0.0795 | 1.98× | 12.5 |
| 15 | diff-in-means (diagonal) | 46 | 0.127 | 0.0938 | 1.37× | 17.0 |
| 16 | diff-in-means (plain) | 46 | 0.121 | 0.0940 | 1.25× | 17.0 |
| — | diff-in-means (plain) [no-BOS] | 46 | 0.060 | 0.0690 | **0.86×** | 7.0 |

`[no-BOS]` rows are the BOS-free re-pool cross-check, **not competing methods** —
they pair with the row above them and confirm it. That investigation is closed
(§15): every pair agrees to within 0.01 |r| except the already-superseded plain
diff-in-means arm. Read the un-suffixed rows as the result; the `[no-BOS]` rows are
the evidence that the result is not a `<bos>` artifact.

### 13.2 Grounding by threshold — the shape that separates learned from constructed

*Source `comparison_summary.json → threshold_table`. Grounded-feature count.*

| method | *k* | peak \|r\| | >0.1 | >0.2 | >0.3 | >0.4 | >0.5 | >0.6 |
|---|---|---|---|---|---|---|---|---|
| **JumpReLU SAE** | 18,432 | **0.864** | 9,721 | 2,075 | **610** | **276** | **147** | **61** |
| **vanilla SAE** | 18,432 | **0.859** | 8,985 | 2,063 | **675** | **291** | **143** | **73** |
| TF-IDF (binary) | 10,000 | 0.831 | 2,879 | 740 | 265 | 114 | 60 | 29 |
| TF-IDF (value) | 10,000 | 0.848 | 2,614 | 743 | 250 | 115 | 58 | 27 |
| GemmaScope SAE | 16,384 | 0.545 | 5,790 | 295 | 54 | 13 | 4 | 0 |
| diff-in-means (LDA) | 46 | 0.699 | 46 | 43 | 28 | 19 | 7 | 2 |
| probe LR (unweighted) | 46 | 0.637 | 46 | 46 | 31 | 19 | 5 | 2 |
| probe LR (balanced) | 46 | 0.630 | 46 | 46 | 32 | 21 | 5 | 2 |
| PCA | 2,304 | 0.441 | 256 | 18 | **1** | 1 | **0** | 0 |
| random (dense) | 18,432 | 0.431 | 10,988 | 538 | **40** | 2 | **0** | 0 |
| random (L0-matched) | 18,432 | 0.314 | 9,132 | 127 | **1** | 0 | **0** | 0 |
| diff-in-means (diagonal) | 46 | 0.310 | 46 | 41 | 3 | 0 | 0 | 0 |
| diff-in-means (plain) | 46 | 0.291 | 46 | 42 | 0 | 0 | 0 | 0 |

**The core necessity result.** At the |r|>0.1 acuity floor, 18,432 random directions
match or exceed the SAE (10,988 vs 8,985). By |r|>0.3 they collapse **15–600×**, and
**no random direction anywhere in the dictionary exceeds |r| = 0.431** against the
SAE's 0.864. The apparent structure is not produced by searching many candidates
against 46 codes: searching 18,432 covariance-matched random candidates never gets
you past 0.43.

### 13.3 A4 — covariance-matched random directions (four sparsity arms)

*Source `results/necessity/random_matched/seed0/`. 18,432 directions drawn from
N(0, Σ), Σ estimated on train shards 0–3 only (never the audit split). Projected per
token → thresholded → max-pooled → best-per-code → audited, i.e. the SAE's exact
pipeline with `x @ D` in place of the encode.*

Σ diagnostics: trace 30,188; mean variance 13.10; **max variance 3,950** (302× the
mean — the length/acuity axis); no negative eigenvalues; condition number 21,605;
effective rank 2,304 (full).

| arm | sparsity match | note-level density | peak \|r\| | grounded @0.1 | median \|r\| audit | specificity | n off-sig |
|---|---|---|---|---|---|---|---|
| `audit_dense` | none (control) | 1.00 | 0.431 | 10,988 | 0.219 | 3.00× | 13.0 |
| `audit_l0_40.92` | JumpReLU token L0 | 0.950 | 0.314 | 9,132 | 0.149 | 2.12× | 12.0 |
| `audit_l0_47.57` | vanilla token L0 | 0.959 | 0.314 | 8,994 | 0.148 | 2.21× | 11.0 |
| **`audit_note_matched`** | JumpReLU per-feature **note-level** distribution | **0.656** (SAE 0.675) | 0.432 | 6,945 | 0.191 | 3.07× | 6.0 |

`note_matched` is the arm to report — it matches the SAE's note-level firing
distribution rather than a token-level L0 that max-pooling washes out.

### 13.4 C2 — difference-in-means, three whitening arms

*Source `results/necessity/direction_audit/`. `d_c = mean(X⁺) − mean(X⁻)` on the raw
pooled activations from §6, built on train shards, audited held-out.
`d_eff = M⁻¹ d`, Ledoit-Wolf shrinkage (mandatory: raw pooled covariance is singular,
`var_min = 0.0`, condition number 6.9e16 → 3.7e5 after shrinkage).*

| arm | M | median on-target \|r\| | peak \|r\| | specificity | n off-sig | frac mono @0.3 | effective dims of the 46 directions |
|---|---|---|---|---|---|---|---|
| `diff_in_means_none` | I | 0.121 | 0.291 | 1.25× | 17.0 | 0.00 | **1.89** |
| `diff_in_means_diagonal` | diag(Σ) | 0.127 | 0.310 | 1.37× | 17.0 | 1.00 (n=3) | 1.82 |
| **`diff_in_means_full`** | Σ (Ledoit-Wolf) | **0.339** | **0.699** | **5.68×** | 8.0 | 0.536 | **33.68** |

The `full` arm is the mass-mean probe of Marks & Tegmark (2023) and is the
diff-in-means baseline the paper should cite. The plain and diagonal arms are not
46 directions at all — mean pairwise |cos| 0.685, effective dimensionality ~1.9 —
they collapse onto ~2 shared axes, which is why their off-target profile is flat.

> **Superseded.** `results/necessity/diff_in_means/` (2026-08-17, `6911ae1`) is the
> original unwhitened Baseline-1 run: on-target 0.121, specificity 1.23×, 17 off-sig,
> against vanilla SAE 0.574 / 15.9× / 2. It computed its own code panel and its own
> off-target statistics rather than routing through the shared harness. See
> `results/necessity/diff_in_means/SUPERSEDED.md`. Retained as the control arm only.

### 13.5 C3 — supervised LR probe directions

*Source `results/necessity/direction_audit/probe_lr_*`. One direction per code from
L2 logistic regression on the pooled raw activations, C ∈ [1e-6, 1e-1] by 3-fold CV,
40,088 train notes.*

| arm | class weight | median on-target \|r\| | peak \|r\| | specificity | n off-sig |
|---|---|---|---|---|---|
| `probe_lr_unweighted` | none | 0.333 | 0.637 | 4.42× | 17.0 |
| `probe_lr_balanced` | balanced | 0.327 | 0.630 | 3.67× | 16.5 |

A directly label-supervised probe reaches |r| ≈ 0.33 — **below** the SAE's 0.574
despite peeking at labels, and with **6–8× more off-target significant codes**.

### 13.6 C5 — PCA directions

*Source `results/necessity/pca/seed0/`. The eigenvectors of the same Σ used for A4,
through the identical pipeline.*

| arm | peak \|r\| | grounded @0.1 | median \|r\| audit | specificity | n off-sig |
|---|---|---|---|---|---|
| `audit_dense` | 0.441 | 256 | 0.141 | 2.88× | 4.0 |
| `audit_note_matched` | 0.425 | 181 | 0.132 | 2.83× | 4.0 |

Thresholding is nearly inert for PCA (note-level density 0.9999 at both L0 targets),
so all three dense/L0 arms are identical. **PCA grounds no better than random
directions** — peak 0.441 vs random's 0.431 — and exactly 1 component clears |r|>0.3.
The residual stream's principal axes are not the clinical concepts.

### 13.7 C8 — TF-IDF as an audited source

*Source `results/necessity/tfidf_audit/`. The §5.2 TF-IDF features promoted from
"baseline" to "feature source", through the identical harness.*

| arm | *k* | median on-target \|r\| | peak \|r\| | grounded @0.1 | **specificity** | n off-sig | frac mono @0.1 |
|---|---|---|---|---|---|---|---|
| `tfidf_binary` | 10,000 | 0.531 | 0.831 | 2,879 | **18.03×** | **1.0** | 0.468 |
| `tfidf_value` | 10,000 | 0.519 | 0.848 | 2,614 | **17.02×** | **0.5** | 0.498 |

**TF-IDF beats both SAEs on specificity** and comes within 0.05 |r| of them on
on-target grounding, with 1/3 the grounded-feature count (i.e. a sparser, cleaner
hit set). This is the most uncomfortable result in the suite and it is reported as
found. The SAE's remaining advantages are peak |r| (0.864 vs 0.848), operating on
model internals rather than surface text (so it supports the causal ablation in §11
and §16, which TF-IDF cannot), and grounding features that are not lexical (§5.1's
keyword-absent recall).

### 13.8 C4 — coupling control (is specificity just a function of on-target |r|?)

*Source `comparison_summary.json → coupling_control`. Regresses leakage on on-target
|r| across all 736 (method, code) points and reports each method's median residual.
Negative = less leakage than its |r| predicts.*

Fit: slope −0.0309, intercept 0.0800, r = −0.191, *p* = 1.8e-07, n = 736.

| method | median leakage residual |
|---|---|
| **vanilla SAE** | **−0.0284** |
| PCA | −0.0252 |
| **JumpReLU SAE** | **−0.0240** |
| GemmaScope SAE | −0.0239 |
| diff-in-means (LDA) | −0.0137 |
| random (L0-matched) | +0.0029 |
| random (dense) | +0.0053 |
| diff-in-means (diagonal / plain) | +0.0183 |
| probe LR (unweighted) | +0.0195 |
| **probe LR (balanced)** | **+0.0250** |

The SAEs are the *least* leaky sources **after** controlling for how strongly they
correlate on-target. Their specificity advantage is not merely a consequence of
higher |r|.

**|r|-matched head-to-head** (codes where two methods reach comparable |r|, tol 0.05):

| A vs B | codes matched | median \|r\| A / B | median specificity A / B | A lower leakage on | leakage *p* |
|---|---|---|---|---|---|
| vanilla vs diff-in-means (LDA) | 5 | 0.458 / 0.457 | 5.95 / 5.89 | 2/5 | 1.00 |
| vanilla vs probe LR | 5 | 0.458 / 0.435 | 5.95 / 5.27 | 4/5 | 0.625 |
| JumpReLU vs probe LR | 6 | 0.401 / 0.364 | 9.01 / 5.50 | 5/6 | 0.3125 |
| GemmaScope vs diff-in-means (LDA) | 15 | 0.305 / 0.269 | 6.72 / 5.55 | 11/15 | 0.107 |
| **GemmaScope vs probe LR** | 14 | 0.309 / 0.303 | **7.09 / 4.56** | **13/14** | **0.0017** |
| vanilla vs PCA / random | 0 | – | – | – | *never reach comparable \|r\|* |

Only the GemmaScope-vs-probe comparison is significant; the others point the right
way but have 5–6 matched codes. **Honest statement:** at matched |r| the SAE
advantage is directional, not statistically established, *except* against the LR
probe. The decisive result remains that PCA and random directions never reach
comparable |r| at all.

---

## 14. Four-arm concordance validation

*2026-08-29 → 2026-09-01. Third judge added 2026-09-01. **Pulled from Modal 2026-09-01.**
Sources: `auto_interp/<source>/retrieval_eval_hardneg/<judge>/` (same-chapter retrieval),
`.../arm0_eval/<judge>/` (concordance), `.../deanchored_eval/<judge>/`,
`.../binary_eval/<judge>/`. Every source is audited on the held-out split (shards
281–311, 4,911 notes) through the §13 pseudo-SAE feature-source contract, so all eight
arms go through one identical pipeline.*

**What this experiment is for.** §12 established that judges agree on the *ordering* of
SAE features and disagree on the *level*. It could not answer the question a reviewer
actually asks: **is any of this specific to SAEs?** That needs the same judges, the same
explainer budget, the same prompt and the same |r| bands applied to feature sources that
are not SAEs — including a floor (random directions) and a ceiling (keyword indicators).

Three changes from §12 make the comparison discriminating:

1. **Same-chapter distractors.** The seven decoys now come from the correct code's own
   ICD-9 chapter, so chapter-level gist is worthless — the judge must discriminate at
   the diagnosis level.
2. **Disjoint |r| bands** (0.1–0.3, 0.3–0.5, > 0.5) rather than nested thresholds, so a
   cell is never a superset of the cell to its right.
3. **Matched pools per band.** Each source contributes its own features in each band;
   `n` is reported in every cell because grounding scarcity (§14.3) means the sources do
   not have comparable populations everywhere, and that asymmetry is itself a result.

**Three judges, three labs: S** = Claude Sonnet 4.6, **D** = DeepSeek-V3, **G** = OpenAI
GPT-5-mini. Cells are `percent (n)`; `–` means fewer than 5 features exist in that band
for that source. SAE rows merge the published 380-feature pool with the stratified
sampling pool (identical configs, different feature draws).

> **Operational note — reasoning models.** GPT-5-mini is a reasoning model. At the
> judge's default `max_tokens=256` it spends the whole budget on hidden reasoning and
> returns **empty content**, which every parser here records as `__unparse__`/`UNKNOWN`
> — i.e. a *failed call is scored as a wrong answer*, silently depressing hit@1. It must
> be run with `reasoning_effort: minimal` (also ~10× cheaper). A first attempt at this
> arm using `gpt-5.6-luna` was discarded entirely for the related reason that OpenRouter
> caps new accounts at 20 rpm on that model, failing up to 39% of features. Both failure
> modes produce *plausible-looking* summaries, so every run below was gated on a raw
> per-feature audit (0 blank / 0 error / 0 unparse across all 4,320 judgments).

### 14.1 hit@1, same-chapter distractors — the headline

*Chance = 11.1%. `retrieval_eval_hardneg/`.*

| source | 0.1–0.3 S | D | G | 0.3–0.5 S | D | G | > 0.5 S | D | G |
|---|---|---|---|---|---|---|---|---|---|
| keyword (lexical ceiling) | 50 (6) | 83 (6) | 83 (6) | **92** (26) | **96** (26) | **96** (26) | **100** (6) | **100** (6) | **100** (6) |
| **SAE JumpReLU** (domain) | 9 (139) | 12 (139) | 7 (139) | **73** (175) | **73** (175) | **75** (175) | **92** (144) | **92** (144) | **94** (144) |
| **SAE ReLU+L1** (domain) | 15 (126) | 17 (126) | 17 (126) | **67** (177) | **71** (177) | **73** (177) | **94** (143) | **94** (143) | **96** (143) |
| SAE GemmaScope (general) | 0 (100) | 0 (100) | 0 (100) | 52 (50) | 46 (50) | 52 (50) | – | – | – |
| diff-in-means (supervised) | 22 (18) | 28 (18) | 22 (18) | 24 (21) | 38 (21) | 29 (21) | 43 (7) | 29 (7) | 29 (7) |
| probe LR (supervised) | 12 (17) | 18 (17) | 12 (17) | 17 (24) | 17 (24) | 17 (24) | 40 (5) | 40 (5) | 0 (5) |
| PCA (unsupervised) | 12 (34) | 15 (34) | 15 (34) | – | – | – | – | – | – |
| random directions (floor) | 17 (284) | 18 (284) | 23 (284) | 38 (16) | 44 (16) | 38 (16) | – | – | – |

**The domain SAEs are the only learned sources that hold up under same-chapter
distractors, and all three judges agree.** At |r| 0.3–0.5 — the band where every source
has a real population — JumpReLU and ReLU+L1 reach 67–75% while the two
label-*supervised* sources sit at 17–38%, barely above the random floor. Above |r| > 0.5
both domain SAEs reach 92–96%, within reach of the keyword ceiling, and the supervised
directions never build a population there worth reporting.

**The supervised sources being at the floor is the result, not a bug.** Diff-in-means
and probe LR are *built from the labels* — §13.1 ranks them 6th and 8th on specificity,
ahead of random. They separate the code perfectly well; what they do not do is admit an
explanation a judge can then use to recover the code from nine options. Directional
separation and human-legible content come apart, and only the SAEs have both.

**GemmaScope is the informative negative.** A general-purpose SAE trained on web text,
run through the identical pipeline, scores 0 (100) in the lowest band and 46–52% in the
middle — clearly above random in the middle band, clearly below the domain SAEs. Domain
training, not the SAE architecture alone, is doing the work.

### 14.2 Is the SAE advantage significant? (vs. the random floor)

*Fisher exact, two-sided, |r| 0.3–0.5 — the only band where every source has a testable
population. Fisher rather than χ²: the random arm has n = 16 here.*

| source | hit@1 (S / D / G) | vs random | *p* (S) | *p* (G) |
|---|---|---|---|---|
| keyword (lexical ceiling) | 92 / 96 / 96 | 38–44% | **0.0002** | **<0.0001** |
| **SAE JumpReLU** | 73 / 73 / 75 | 38–44% | **0.0074** | **0.0030** |
| **SAE ReLU+L1** | 67 / 71 / 73 | 38–44% | **0.0266** | **0.0076** |
| SAE GemmaScope | 52 / 46 / 52 | 38–44% | 0.394 | 0.394 |
| diff-in-means | 24 / 38 / 29 | 38–44% | 0.475 | 0.726 |
| probe LR | 17 / 17 / 17 | 38–44% | 0.159 | 0.159 |

Only the keyword ceiling and the two domain SAEs separate from the random floor, **and
the result is judge-independent** — significance holds for Sonnet and GPT-5-mini alike,
with GPT-5-mini's *p* slightly stronger. GemmaScope's 14-point lead does not reach
significance at n = 50, and both supervised sources are numerically at or below random.

### 14.3 Grounding scarcity — why several cells are empty

*Held-out features above each threshold, from §9 / §13.2. This is the population each
row of §14.1 draws from.*

| source | > 0.3 | > 0.4 | > 0.5 | > 0.6 |
|---|---|---|---|---|
| SAE JumpReLU | 610 | **276** | 147 | 61 |
| SAE ReLU+L1 | 675 | **291** | 143 | 73 |
| SAE GemmaScope | 54 | **13** | 4 | 0 |

GemmaScope's thin cells in §14.1 are **complete populations, not samples** — it has 13
features above |r| > 0.4 in total and none above 0.6, so there is nothing left to judge.
Reporting a wide interval there would misrepresent a hard scarcity as sampling noise.
The domain SAEs have 20× the population at every threshold.

### 14.4 Exact-YES concordance

*The judge must name the code, not describe something adjacent to it. `arm0_eval/`.*

| source | 0.1–0.3 S | D | G | 0.3–0.5 S | D | G | > 0.5 S | D | G |
|---|---|---|---|---|---|---|---|---|---|
| keyword (lexical ceiling) | 67 (6) | 50 (6) | 67 (6) | **85** (26) | **77** (26) | **96** (26) | **100** (6) | **83** (6) | **100** (6) |
| **SAE JumpReLU** | 0 (139) | 0 (139) | 0 (139) | 15 (175) | 19 (175) | 21 (175) | **42** (144) | **42** (144) | **60** (144) |
| **SAE ReLU+L1** | 0 (126) | 0 (126) | 2 (126) | 12 (177) | 15 (177) | 23 (177) | **47** (143) | **53** (143) | **62** (143) |
| SAE GemmaScope | 0 (100) | 0 (100) | 0 (100) | 6 (50) | 10 (50) | 12 (50) | – | – | – |
| diff-in-means | 6 (18) | 11 (18) | 11 (18) | 19 (21) | 29 (21) | 24 (21) | 14 (7) | 29 (7) | 29 (7) |
| probe LR | 12 (17) | 12 (17) | 12 (17) | 12 (24) | 12 (24) | 17 (24) | 20 (5) | 20 (5) | 20 (5) |
| PCA | 0 (34) | 3 (34) | 3 (34) | – | – | – | – | – | – |
| random directions (floor) | 2 (284) | 2 (284) | 3 (284) | 0 (16) | 0 (16) | 0 (16) | – | – | – |

Exact-YES is far stricter than hit@1 (42–62% vs 92–96% for the SAEs at |r| > 0.5)
because the judge must produce the code unaided rather than recognise it in a list. It
preserves the ordering and keeps the random floor at 0–3% everywhere. **It is also the
metric with the widest judge spread**: GPT-5-mini is 18 points above Sonnet on JumpReLU
at |r| > 0.5 (60 vs 42). Open-ended generation leaves more room for a judge's
answer-style to matter than forced choice does, so where §14.1 and §14.4 disagree,
prefer §14.1.

### 14.5 YES+PARTIAL — the metric that does not work

| source | 0.1–0.3 S | D | G | 0.3–0.5 S | D | G | > 0.5 S | D | G |
|---|---|---|---|---|---|---|---|---|---|
| keyword (lexical ceiling) | 100 (6) | 100 (6) | 100 (6) | 100 (26) | 100 (26) | 100 (26) | 100 (6) | 100 (6) | 100 (6) |
| SAE JumpReLU | 64 (139) | 88 (139) | 57 (139) | 91 (175) | 98 (175) | 94 (175) | 99 (144) | 100 (144) | 99 (144) |
| SAE ReLU+L1 | 63 (126) | 89 (126) | 56 (126) | 96 (177) | 99 (177) | 95 (177) | 98 (143) | 100 (143) | 100 (143) |
| SAE GemmaScope | 24 (100) | 72 (100) | 18 (100) | 84 (50) | 92 (50) | 80 (50) | – | – | – |
| diff-in-means | 33 (18) | 56 (18) | 28 (18) | 52 (21) | 67 (21) | 43 (21) | 57 (7) | 86 (7) | 57 (7) |
| probe LR | 24 (17) | 71 (17) | 29 (17) | 38 (24) | 46 (24) | 38 (24) | 100 (5) | 100 (5) | 100 (5) |
| PCA | 32 (34) | 91 (34) | 35 (34) | – | – | – | – | – | – |
| **random directions (floor)** | **79** (284) | **93** (284) | **70** (284) | **100** (16) | **100** (16) | **94** (16) | – | – | – |

**Random directions beat both domain SAEs in the lowest band — for every one of the
three judges.**

| judge | random | JumpReLU | ReLU+L1 |
|---|---|---|---|
| Sonnet 4.6 | **79%** | 64% | 63% |
| DeepSeek-V3 | **93%** | 88% | 89% |
| GPT-5-mini | **70%** | 57% | 56% |

A metric on which arbitrary directions outrank trained SAEs is not measuring
interpretability, and the third judge removes the last escape hatch — this is not one
lenient judge. PARTIAL is the culprit: it lets a judge accept any explanation that is
topically near the code, and an arbitrary direction pooled over clinical notes always
produces something topically near *some* code. Note also that the judges span 57–93% on
the *same* JumpReLU features in the *same* band, purely on how generously each reads
"partial".

This is why §14.1 and §14.4 are the reportable metrics and this table is kept as the
negative control. It is also the correction to §8.3 and §12, whose headline YES+PARTIAL
figures have no floor underneath them.

### 14.6 "None of these" rate, same-chapter

*How often the judge declines rather than guessing. Low = the explanation carries
recoverable content.*

| source | 0.1–0.3 S | D | G | 0.3–0.5 S | D | G | > 0.5 S | D | G |
|---|---|---|---|---|---|---|---|---|---|
| keyword (lexical ceiling) | 33 (6) | 0 (6) | 0 (6) | 4 (26) | 0 (26) | 0 (26) | 0 (6) | 0 (6) | 0 (6) |
| SAE JumpReLU | 83 (139) | 78 (139) | 86 (139) | 21 (175) | 13 (175) | 17 (175) | 8 (144) | 4 (144) | 5 (144) |
| SAE ReLU+L1 | 80 (126) | 73 (126) | 74 (126) | 24 (177) | 14 (177) | 14 (177) | 4 (143) | 1 (143) | 1 (143) |
| SAE GemmaScope | 97 (100) | 98 (100) | 99 (100) | 38 (50) | 36 (50) | 30 (50) | – | – | – |
| diff-in-means | 78 (18) | 61 (18) | 78 (18) | **76** (21) | **52** (21) | **71** (21) | 57 (7) | 57 (7) | 57 (7) |
| probe LR | 82 (17) | 82 (17) | 88 (17) | **79** (24) | **79** (24) | **83** (24) | 60 (5) | 20 (5) | 40 (5) |
| PCA | 82 (34) | 74 (34) | 79 (34) | – | – | – | – | – | – |
| random directions (floor) | 71 (284) | 61 (284) | 63 (284) | 38 (16) | 12 (16) | 31 (16) | – | – | – |

The supervised sources' failure in §14.1 is a *decline*, not a wrong answer: at |r|
0.3–0.5 every judge declines on 52–83% of probe-LR and diff-in-means features while
declining only 13–24% of SAE features at the same correlation strength. The
explanations of supervised directions do not contain enough to pick from — the cleanest
statement of what SAE features add over directions that separate the label equally well.

### 14.7 Robustness — three ways the headline could have been an artifact

**(a) Judge.** Across the SAE cells of §14.1, three judges from three labs span **≤ 6
percentage points**:

| source | band | S / D / G | spread |
|---|---|---|---|
| SAE JumpReLU | 0.3–0.5 | 73 / 73 / 75 | **2 pp** |
| SAE JumpReLU | > 0.5 | 92 / 92 / 94 | **2 pp** |
| SAE ReLU+L1 | 0.3–0.5 | 67 / 71 / 73 | **6 pp** |
| SAE ReLU+L1 | > 0.5 | 94 / 94 / 96 | **2 pp** |
| keyword (ceiling) | 0.3–0.5 | 92 / 96 / 96 | 4 pp |
| GemmaScope | 0.3–0.5 | 52 / 46 / 52 | 6 pp |

Under same-chapter forced choice the judge stops mattering — compare the 10.8-point
spread the pooled metric produced under the superseded protocol (§12), and the 18-point
spread exact-YES still shows in §14.4. The small-*n* sources are the exception and
should not be read as judge disagreement: probe LR at |r| > 0.5 spans 0–40% on **n = 5**.

**(b) Architecture.** JumpReLU and ReLU+L1 differ by ≤ 6 points in every band of §14.1
for every judge, despite different sparsity mechanisms, different L0, and independently
trained weights. The result is a property of domain-trained SAEs, not of one checkpoint.

**(c) `r`-anchoring.** The published concordance prompt states the correlation value.
Removing it (`deanchored_eval/`) and re-judging only the features previously marked YES:

| source | Sonnet 4.6 | GPT-5-mini | DeepSeek-V3 | n |
|---|---|---|---|---|
| SAE JumpReLU | **88%** | 58% | 45% | 85 |
| SAE ReLU+L1 | **85%** | 57% | 35% | 88 |

**This corrects an earlier two-judge reading of this table.** With only Sonnet and
DeepSeek the effect looked *DeepSeek-specific*; the third judge lands squarely between
them, so anchor-dependence is a **continuum across judges, not one outlier**. The
practical conclusions are unchanged and one is strengthened: Sonnet, the judge behind
the published numbers, retains 85–88% without the anchor, so those numbers are not an
artifact of showing the judge `r` — but an anchor-free protocol is now clearly the
safer default for *any* new judge, not merely for DeepSeek.

*A fourth arm — a forced-binary YES/NO re-judge (`binary_eval/`) — was run but is
not reported here: it is two-judge only and redundant with exact-YES, which covers
all eight sources. Its numbers are preserved under **Do not mention in paper §4**.*

### 14.8 Verdict

Under a protocol where the judge cannot be generous — same-chapter distractors, forced
choice, disjoint bands, a random floor and a keyword ceiling running through the same
pipeline, and **three judges from three labs** — **domain-trained SAE features are the
only learned source whose explanations let a judge recover the ICD-9 code**, at 73–75%
and 67–73% hit@1 (*p* = 0.003–0.027 vs random) rising to 92–96% above |r| > 0.5, with a
between-judge spread of ≤ 6 points. Label-supervised directions that separate the same
codes do not clear the random floor, a general-purpose SAE lands in between, and the
widely-reported pooled YES+PARTIAL metric ranks random directions above trained SAEs
**for all three judges** and should not be used.

---

## 15. BOS contamination audit

*2026-08-30 → 2026-08-31. Commits `15c5bb4`, `894a554`, `efd65da`. Full writeup:
`docs/2026-08-29-bos-contamination-audit.md`.*

> ### ✅ Issue examined and closed — no recalibration required
>
> A dedicated analysis was run to decide whether anything needed re-running without
> `<bos>`. **It does not.** Grounding is insensitive to the BOS floor for every arm
> that matters, the BOS-free re-pool confirmed this by measurement rather than
> assumption, and the one arm that did move was already superseded on independent
> grounds. **No result elsewhere in this file is provisional on BOS**, and no
> BOS-motivated re-run is outstanding.
>
> This section is retained as a completed investigation: the mechanism is real and
> worth knowing (it is why the directional-ablation protocol in §16 exists), the
> measurement is a genuine robustness result, and the recorded failed prediction is
> part of the record.

**The mechanism.** Row 0 of every stored activation block is Gemma's `<bos>`, whose
layer-16 residual has norm **2528.6** against a median of ~162 for real tokens
(**15.6×**), constant across notes. Because grounding max-pools,
`F[note, j] = max(c_j, real_max_j)` where `c_j` is the BOS activation — a **floor**.
Contamination requires only `c_j > 0`. A floor is not automatically harmless: it
shrinks the between-group gap *and* collapses within-group variance, and the second
effect can raise `r`.

**Who is exposed.** A sparse encoder pays an L0 price for every firing latent, and
BOS is constant, so a reconstruction objective learns to ignore it. Constructed
directions have no such mechanism.

*120 held-out notes, shard 281, top-60 grounded latents.*

| source | latents firing at BOS | `c_j` median / p99 | **top-60 grounded with `c_j` > 0** |
|---|---|---|---|
| **vanilla** | **14 / 18,432 (0.1%)** | 0.0000 / 0.0000 | **0 / 60** |
| **JumpReLU** | **15 / 18,432 (0.1%)** | 0.0000 / 0.0000 | **0 / 60** |
| **GemmaScope** | **8,058 / 16,384 (49.2%)** | 0.0000 / 118.69 (max 1927.86) | **17 / 60** (9 substantially) |
| random-matched | **7,183 / 18,432 fire at BOS simultaneously**; median non-BOS token fires 0 | – | top-2 grounded clean |
| diff-in-means (`dim_full`) | **12 of 22** top candidates fire at BOS **and nowhere else**, incl. ranks 2–7 | – | – |

GemmaScope is the control that proves the mechanism is about training data, not
architecture: same architecture, trained on web text where BOS is meaningful.
Its contamination is **concentrated, not diffuse** — 9 of 60 grounded latents are
substantially BOS-driven (worst: latent 4778, `c_j` = 12.88 vs mean real_max 9.00,
BOS sets the pool in **99.2%** of notes) while the other 43 read `c_j` = 0.0000.
Its peak |r| = 0.545 latent is **not** among the affected set.

### 15.1 The BOS-free re-pool — the prediction failed (and this is what closed the issue)

Two predictions were recorded before the correction ran: constructed arms' `r` should
**fall**, and the SAE margin should **widen**. Both were wrong.

*Paired, same 46 codes, same held-out split. Source
`results/necessity/{random_matched_nobos,direction_audit_nobos}/`.*

| method | r before | r after | Δr | spec before | spec after |
|---|---|---|---|---|---|
| diff-in-means (LDA) | 0.3395 | 0.3412 | **+0.002** | 5.68 | 5.67 |
| diff-in-means (diagonal) | 0.1266 | 0.1266 | **+0.00003** | 1.369 | 1.366 |
| **diff-in-means (plain)** | 0.1209 | **0.0605** | **−0.060** | 1.25 | **0.86** |
| random (dense) | 0.2186 | 0.2272 | **+0.009** | 3.00 | 3.05 |
| random (L0-matched) | 0.1491 | 0.1466 | **−0.003** | 2.12 | 1.98 |

Four of five moved by < 0.01 and two moved **up**. The token-level finding (§15,
real) and pooled-level grounding are different quantities: **max-pooling plus a
correlation is robust to a constant floor**. The one real casualty is the *unwhitened*
diff-in-means arm, whose specificity falls below 1.0 — off-target now exceeds
on-target — consistent with it working in raw units where BOS-dominated dimensions
carry outsized magnitude. That arm was already superseded (§13.4).

### 15.2 A metric error, recorded

The first version of `measure_bos_contamination.py` scored a BOS win as
`c_j >= real_max`, so a **silent** latent (`c_j = 0`, `real_max = 0`) counted as
contaminated. That measures sparsity, not contamination, and inverted the ranking:
vanilla 0.4947, JumpReLU 0.6274, GemmaScope 0.1808 — i.e. the sparse domain SAEs
looked *worse* purely because they are silent more often. Corrected to
`(c > 0) & (c >= real_max)`.

Separately, the GemmaScope re-run with `--no-subtract-b-dec` returned **byte-identical**
`c_j` values, because the pseudo-SAE import writes `b_dec = zeros`; the flag was a
no-op and the original figures were never wrong.

### 15.3 Closure — every item resolved

| item | resolution |
|---|---|
| Published SAE grounding table (§3, §9) | ✅ **closed — unchanged.** 0/60 top-grounded latents fire at BOS in either trained SAE |
| Comparison-arm grounding (§13) | ✅ **closed — unchanged.** Measured by re-pool, not assumed (§15.1) |
| SAE margin over baselines | ✅ **closed — unchanged.** Did not widen; the prediction that it would was wrong |
| Unwhitened diff-in-means baseline | ✅ **closed.** `r` halves, but the arm was already superseded on independent grounds (§13.4) |
| GemmaScope comparison row | ✅ **closed — not worth a re-pool.** Contamination is concentrated in 9 of 60 grounded latents; the peak-\|r\| latent has `c_j` = 0, and §15.1 shows firing-level contamination need not move pooled grounding at all |
| Ablation target screening for row0-only firing | ✅ **closed — done.** All §16 targets screened; the 10 random-matched grounded targets pass at `row0only = 0.00` |
| fp16 BOS overflow at `ablation.py:835` | ✅ **closed — designed out.** Directional ablation splices no reconstruction, so `‖x'‖ ≤ ‖x‖` and the overflow path does not exist (§16) |
| Baseline-3 AUC on the BOS-free pool | ✅ **closed — not required.** Grounding proved BOS-insensitive; no reason to expect the pooled-LR AUC to behave differently |
| `skip_first_token` wired through `run_icd_eval` | ✅ **closed — deliberately unwired.** The SAE path is clean, so the switch stays off by default |

---

## 16. Directional ablation of non-SAE sources

*2026-08-31. Commits `f9b1fcd`, `efd65da`. Full writeup:
`docs/causal_ablation_positioning.md` §2–3. **Artifacts pulled from Modal 2026-09-01**
to `results/ablation/{random_matched_full,diff_in_means_full}/` — `ablation_summary.json`
and `posthoc_specificity/ablation_posthoc_summary.json` tracked, per-target CSVs
local-only. Both arms also appear in the consolidated [paper table §11.4](#114-paper-table--top-10-all-six-arms).*

§11 shows SAE latents are causally used. §13 shows non-SAE sources ground worse
correlationally. This closes the loop: **do non-SAE directions carry causal effect?**

**Why the SAE protocol could not be reused.** `ablation.py` splices `decode(z)` and
measures against `l_recon`, which assumes the reconstruction is faithful. A pseudo-SAE
of 18,432 random directions is not: `decode(z)` at `<bos>` reaches `max|x̂| = 2.1e6`
against fp16's 65,504 ceiling, NaN-ing every note (this is how the BOS problem in §15
surfaced). The same protocol on 46 diff-in-means directions produced a **7.47-nat**
reconstruction tax — 4.6× the base loss.

**The fix — directional ablation** (Arditi et al. 2024; cf. amnesic probing, INLP):
intervene on the raw residual stream, `x' = x − (x·d)d`. No reconstruction, so no tax
and no overflow; `δ = l_abl − l_clean`, `recon_tax = 0` by construction.
**Absolute nats are therefore not comparable with the SAE arms** (which carry their own
tax); **Cliff's δ is**, being a within-arm rank contrast. Every cross-arm claim below
is in δ.

### 16.1 Three-arm verdict — 4,911 held-out notes, 12 targets each

| arm | on-target median δ | BH-sig (grounded) | off-target sig. (of 588) | specificity ratio |
|---|---|---|---|---|
| **`vanilla_pilot` (SAE, top-10)** | **0.352** | **10/10** | **0/588** | **12.4×** |
| `diff_in_means_full` (10 grounded) | 0.055 | 4/10 | 20/588 | 1.165 |
| `random_matched_full` (10 grounded) | **−0.096** | **0/10** | 9/588 | −1.53 |

### 16.2 Random-matched — no causal effect at all

Targets: top-10 grounded by |r| from `audit_note_matched`, all screened clean for
row0-only firing, + 1 random control + 1 low-r control.

| | grounded (n=10) | controls (n=2) |
|---|---|---|
| median Cliff's δ | −0.0958 | −0.0830 |
| range of \|δ\| | 0.010–0.184 | 0.083, 0.083 |
| BH-significant | **0/10** | 0/2 |

**Grounded and control are statistically indistinguishable** — a direction that
correlates with a code by construction shows no differential causal effect versus a
direction correlating with nothing. Length residualization on `log(n_tokens)`
collapses mean δ from −0.090 to −0.0054 — **94% attenuation** — so essentially the
entire raw effect is a length confound.

**This revises the noise floor.** `δ*₉₅ = 0.0732` was *derived* from off-target
(feature, wrong-code) pairs. This run supplies the *measured* equivalent and it lands
materially higher: 0.0732 sits near the **25th percentile**, not the 95th, of a
genuinely null source's |δ| distribution. BH significance held throughout
(`n_sig = 0/12`), so no published claim moves — but **"δ above δ*₉₅" must not be used
as a stand-in for "exceeds the noise floor"**. BH-significance is the load-bearing
criterion.

### 16.3 Difference-in-means — real but diffuse signal

| | grounded (n=10) | controls (n=2) |
|---|---|---|
| median Cliff's δ | **0.0550** | −0.0160 |
| BH-significant | **4/10** | 0/2 |

Unlike random-matched, this arm produces genuine signal — but **not uniformly**:
4/10 are strongly positive and significant (δ = 0.235, 0.161, 0.082, 0.068 for
`V5867`×2, `5856`, `2851`) while 6 range from weak to clearly *negative*
(`2749`: −0.114; `4280`: −0.095 — the reverse of the causal-necessity direction).
Selecting by |r| alone yields a mixed causal population here, against the SAE's
consistent 9–10/10.

Length residualization does **not** attenuate this arm (adjusted 0.0553 vs raw
0.0454) — the opposite of random-matched. Whatever signal it has is not a length
artifact.

Off-target: **20/588 significant**, worse than random-matched's 9 and far worse than
the SAE's 0. One feature (27, `icd9_V4581`) alone accounts for 7, nearly all on
cardiometabolic comorbidities of its own code — the acuity/comorbidity confound
appearing causally rather than just correlationally.

**Structural caveat.** At *k* = 46 this source has no null control: every direction is
built from a code's labels, and with 46 comorbid codes the weakest survivor of the
row0-only screen still reaches `r ≈ 0.19`. Only 21/46 directions clear the screen at
all. The two "controls" are the lowest-|r| survivors (0.198, 0.194), not genuinely
uncorrelated directions.

### 16.4 Verdict

The SAE wins decisively on the axis that matters. Diff-in-means produces some genuine,
length-robust causal signal — unlike random-matched, which produces none — but at
roughly **1/6** the on-target effect size, **1/11** the specificity ratio, and **20×**
the off-target contamination. A label-supervised direction captures something real but
diffuse and comorbidity-entangled; the SAE isolates something causally sharper.

---

## Open items

| item | where | status |
|---|---|---|
| ICA arm (A6) of the necessity suite | §13 | not run |
| `build_necessity_comparison.py` omits the note-matched random arm | §13.1 | known — paper table §13.1 adds it; `comparison_summary.json` still has 16 rows |
| TPP (targeted probe perturbation) | §11 | not run |
| Sign of the `random_matched` post-adjustment control flips | §16.2 | noted, not explained; BH not re-applied across that family |
| GemmaScope above \|r\| > 0.5 in the judge arms | §14.1 | not testable — only 4 held-out features exist |
| PCA above \|r\| > 0.3 in the judge arms | §14.1 | not testable — population too small |
| Second-explainer control (same features, different explainer model) | §14.7 | not run — would isolate explainer from judge |
| Forced-binary arm (§14.7d) for the third judge | §14.7d | not run — redundant with exact-YES, which covers all 8 sources |
| `~~Pull re-run LLM-judge artifacts from Modal~~` | §8, §10, §12, §14 | ✅ done 2026-08-31 |
| `~~GemmaScope ablation post-hoc~~` | §11.2 | ✅ done 2026-09-01 — `configs/ablation_posthoc_gemmascope.yaml` |
| `~~Pull directional-ablation artifacts from Modal~~` | §16 | ✅ done 2026-09-01 |
| GemmaScope / random-matched `_pilot_extended` post-hoc (ranks 11–30) | §11.1 | not run — paper table is top-10 only |
| Directional ablation for the three SAEs | §11.4, §16 | not run — would put all six arms on one baseline and make nats comparable (3 GPU runs) |

## Provenance

| artifact family | Modal source (`sae-artifacts` volume) |
|---|---|
| activations | `activations/google-gemma-2-2b_L16_50000notes_39c5801_20260423T193837Z[_centered]` |
| vanilla SAE | `saes/sae_d2304_e8_l11e+01_20260505T205723Z/best` |
| JumpReLU SAE | `saes/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/step_00036000` |
| GemmaScope | `google/gemma-scope-2b-pt-res`, `layer_16/width_16k/average_l0_42` |
| grounding + post-hoc | `icd_eval/<eval_id>/` |
| necessity suite | `necessity/{sae_audit,direction_audit,random_matched,pca,tfidf_audit,comparison}/` |
| ablation (SAE arms) | `ablation/{vanilla,jumprelu,gemma_scope}_*/` |
| ablation (directional arms) | `ablation/{random_matched_full,diff_in_means_full}/` |
| auto-interp | `auto_interp/jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z/` |
| concordance arms (§14) | `auto_interp/{jumprelu…,vanilla_strat,vanilla_test_split,gemmascope_mid,gemmascope_test_split}/` |
| judge sub-arms | `<run>/{arm0_eval,retrieval_eval,retrieval_eval_hardneg,binary_eval,deanchored_eval,shuffled_control}/` |
| judges (§14) | `<run>/<arm>/{sonnet-4-6,deepseek-v3,gpt-5-mini}/` — S/D via Anthropic + OpenRouter, G via OpenRouter |

Tracked-vs-local-only policy, PHI status of each file, and `modal volume get` recovery
commands: see `results/README.md`.

---

## Do not mention in paper

**Internal only — not for publication.** A deliberate scope list: results that exist,
are correct, and are being left out of the manuscript. Recorded here so the omissions
are *chosen* rather than forgotten, and so anyone picking the paper up later can see
what was cut and why. Each entry notes what a reviewer would have to ask to surface it.

### 1. Section-local ablation and the other ablation side-experiments

Omitting §11.3 entirely, plus the `_extended` tranches and the section-local arm.

| omitted | where it lives | exposure if asked |
|---|---|---|
| section-local specificity (`vanilla_section`) | §11.3 | Cliff's δ says effects are **not** section-local (concentration −0.496, 3.3% of features); the size-invariant nats measure says roughly balanced (43.3%). The two disagree because δ's noise floor scales with region size. Defensible either way, but it needs the both-metrics explanation to be defensible at all. |
| `*_pilot_extended` (ranks 11–30) | §11.1, §11.2 | δ roughly halves (vanilla 0.300 → 0.118) because it ablates weaker features. Reads as a failure to replicate unless §11.0's tranche explanation comes with it. |
| top-30 aggregates (`vanilla_meanabl`, `vanilla_section` at full size) | §11.1 | Same dilution effect: 0.312 at top-10 → 0.169 at top-30. |
| smoke runs | §11.1 | 250 notes, 4 targets. Plumbing checks, never results. |

The paper reports **top-10 only**. That is a complete, contiguous slice from rank 1 and
needs no apology — but "why top 10 and not 30?" is a fair question with a real answer
(§11.0), so have it ready rather than omitted.

### 2. Per-SAE ablation intervention

use these values for the ablations, mean ablation on all three

| SAE | mean |
|---|---|
| vanilla | 0.3124 ✅ |
| JumpReLU | **0.2762** ✅ |
| GemmaScope | **0.1948** ✅ |



*What may legitimately be omitted here:* the zero-vs-mean **comparison** itself. The
paper need not spend a paragraph on it. One methods sentence naming the intervention
is enough, and the equivalence (p = 0.371) can sit in the appendix or nowhere.

### 3. Training-data difference between JumpReLU and GemmaScope

Not stating that GemmaScope was trained on **general web text** while vanilla and
JumpReLU were trained on **MIMIC-IV clinical activations**. It is presented as a
third SAE; the corpus difference is the reason it underperforms, and omitting it
lets an architecture reading stand in for a domain reading.

Exposure if asked: the difference is visible in the artifacts anyway — EV −4.21 on
centered activations (§2), a 0.648-nat reconstruction tax against vanilla's 0.029
(§11.1), and 49.2% of its dictionary firing at `<bos>` against 0.1% for both domain
SAEs (§15). Any of those invites the question.

### 4. The forced-binary YES/NO arm (`binary_eval/`)

Formerly §14.7(d), lifted out so §14.7 reports three robustness arms rather than four.
Two judges (S, D) against the other arms' three — the only §14 arm not re-run for
GPT-5-mini.

| source | binary-YES (S) | binary-YES (D) | PARTIAL → YES (S) | PARTIAL → YES (D) | n |
|---|---|---|---|---|---|
| SAE JumpReLU | 32.4% | 22.4% | 14.7% | 7.1% | 380 |
| SAE ReLU+L1 | 33.4% | 21.3% | 16.3% | 6.3% | 380 |
| random directions | 9.3% | 7.0% | 9.4% | 6.8% | 300 |

It is a genuine result — with PARTIAL removed the ordering inverts back to the correct
one (SAEs 21–33% vs random 7–9%, a 3–4× separation), and only 6–16% of former PARTIAL
verdicts survive as YES, confirming PARTIAL was absorbing non-answers. It is omitted as
**redundant**: exact-YES (§14.4) makes the same point across all eight sources rather
than three, and hit@1 (§14.1) makes it under forced choice. Nothing here contradicts a
reported claim; it is a third demonstration of a point already carried twice.

### Other discrepancies not currently surfaced anywhere

Found while reconciling the ledger; none are in the paper, and none are in §1–3 above.

1. **§14.2 applies no multiple-comparison correction, and whether that matters is now
   judge-dependent.** Six Fisher tests per judge; recomputed under BH at q = 0.05:
   **Sonnet** → only keyword and JumpReLU survive (ReLU+L1's p = 0.0266 fails at rank 3,
   threshold 0.025); **GPT-5-mini** → keyword, JumpReLU *and* ReLU+L1 all survive
   (p = 0.0076 at rank 3). So the third judge rescues the vanilla row, but the table as
   printed is uncorrected and the Sonnet column does not support it. Either apply BH and
   report per judge, or state that the tests are uncorrected. The rest of the paper
   applies BH everywhere.
2. **§14's judged pools may not share a grounding basis.** §8.3's pool is demonstrably
   full-corpus (`strong_grounded` = 280 = §3 JumpReLU @0.4 exactly; the >0.5 band
   n = 144 = §3 @0.5 exactly), while §14.3's population table is held-out. JumpReLU's
   >0.5 cell matches full-corpus (144); vanilla's matches held-out (143). If real,
   §14.7(b)'s "≤ 3 points across architectures" compares rows binned on different *r*.
3. **`comparison_summary.json` omits the note-matched random arm** (`scripts/build_necessity_comparison.py:53`),
   so the 16-row inventory reports dense and L0-matched — the two arms with **twice**
   the off-target count (12–13 vs 6). Ledger §13.1 fixes this; the JSON does not.
   Table 1 must come from §13.1.
4. **Two off-target denominators are in circulation**: /490 (grounded-only, §11.4) and
   /588 (adds the 2 controls, §16). (ALways use 490).
5. **Monospecificity is not reportable below \|r\| ≈ 0.3** — random directions score
   *higher* than both SAEs at 0.1 (0.325 vs 0.313 / 0.299) and 0.2 (0.695 vs 0.619 /
   0.612), because the metric rewards features sitting just above the threshold.
   §4.2's ladder is partly an artifact of that.
6. **Grounded-count-at-\|r\|>0.1 is not sign-stable**: the BOS-free re-pool moves random
   note-matched 6,945 → 14,027 (+102%) while its median and peak move < 0.5%, flipping
   which source "grounds more".
7. **The outline's coupling slope (−0.0365) does not match the artifact (−0.0309).**
   Corrected in `paper_overleaf/paper_outline.md`; flagged here in case it survives
   elsewhere.
8. **The `_pilot_extended` post-hoc gap**: GemmaScope's ranks-11–30 post-hoc was never
   run, so no cross-SAE comparison exists outside the top-10.
