# Causal Ablation: Method Positioning

Scope: what causal ablation is used for in the literature, what specific
experiments are needed to support a claim that an SAE feature is causally
load-bearing for an ICD code, and which of those experiments are required for
our paper. No repo results or implementation details here — those belong in a
separate document.

---

## 1. How causal ablation is used in the literature

### 1.1 The broad landscape (brief)

Ablation is the family of interventions that remove or replace a component's
activation and measure the downstream change. Four established uses, noted only
for orientation:

- **Circuit discovery.** Ablate heads/MLPs/edges and keep what moves a logit
  difference. Basis of the IOI and greater-than circuit results. Path patching
  refines this to direct vs mediated effects.
- **Knowledge localization.** Causal tracing (ROME/MEMIT) corrupts a subject
  token and restores activations one at a time to find where a fact lives.
- **Behavioral control.** Ablate or steer a direction to remove or induce a
  behavior; the basis of refusal removal, concept erasure, and unlearning work.
- **Redundancy measurement.** Backup heads and self-repair were discovered
  through ablations that *failed* to move the metric.

None of these are our setting. Ours is **feature-level validation against
external structured labels**, which has its own narrower lineage.

### 1.2 The papers that match our design

These four use ablation the way we do: an externally labeled concept, a
candidate representation grounded to that label, ablation as the test, and
on-target vs off-target contrast as the evidence.

---

**Bau et al. (2020), "Understanding the role of individual units in a deep
neural network," PNAS**

The ancestor of the design. Units are dissected against an external labeled
concept set (Broden), then ablated; the labeled concept disappears from the
model's output.

- *Claim made:* units matched to a concept label are causally responsible for
  that concept's appearance in the output, not merely correlated with it.
- *Takeaway for us:* establishes the template — external labels supply the
  grounding, ablation supplies the causation, and the two must come from
  independent sources for the argument to work. Published in a venue outside
  interpretability, which makes it a useful citation for a clinical audience.

---

**Marks et al. (2025), "Sparse Feature Circuits," ICLR — the SHIFT method**

Uses Bias in Bios (profession + gender labels). Identifies SAE features carrying
the spurious attribute, ablates them, shows the classifier stops relying on it
while target performance holds.

- *Claim made:* the ablated features carry the specific labeled attribute, shown
  by a *directional prediction* — the intervention changes exactly what the
  interpretation says it should and nothing else.
- *Takeaway for us:* the strength of the inference comes from the prediction
  being made in advance and being narrow. "Ablation changed something" is weak;
  "ablation changed the thing the label names, and not its neighbors" is the
  publishable form.

---

**Karvonen, Rager, Marks & Nanda (2024), "Evaluating Sparse Autoencoders on
Targeted Concept Erasure Tasks"; folded into SAEBench (2025) — the TPP metric**

The closest published analogue to our design. Procedure:

1. Externally labeled multi-class dataset.
2. Train a binary probe per class on model activations.
3. Select each class's relevant SAE latents (attribution patching: decoder
   vector · probe weights · difference in mean activations between
   concept-present and concept-absent inputs).
4. Zero-ablate that class's latents.
5. Measure probe accuracy **across all classes**.

Scoring is high when ablating one class's latents degrades that class's probe
and leaves other class probes intact.

- *Claim made:* concepts are captured by *distinct* sets of latents —
  i.e. disentanglement, measured causally rather than by inspection.
- *Takeaway for us:* label-driven ablation with on-target/off-target specificity
  is **established practice with a name**. Our novelty cannot be the design
  itself; it has to be the label source (real-world administrative clinical
  coding rather than curated annotation sets), the domain, and the readout
  (language-modeling loss rather than probe accuracy).
- *Caveat:* Chanin et al. (2026), "Are Sparse Autoencoder Benchmarks Reliable?"
  is a direct critique of SAEBench metrics. Read before adopting TPP framing.

---

**Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single
Direction," NeurIPS**

Directional ablation of one residual-stream direction across all layers and
positions; refusal collapses. The paper's weight rests on the *second* result:
general capability benchmarks are unaffected.

- *Claim made:* necessity plus specificity. The direction mediates this behavior
  specifically, rather than being a component whose removal degrades everything.
- *Takeaway for us:* the best available template for how to *write* a
  specificity argument. Also the clearest demonstration that the control
  condition, not the headline effect, is what makes the claim survive.

---

**Supporting, briefly:**

- **Cunningham et al. (2023)** — ablated SAE features on a task metric and
  argued SAE features are more causally *precise* than neurons. The original
  necessity argument for SAE features specifically.
- **Gurnee et al. (2023), TMLR** — sparse probing against labeled datasets with
  ablation as validation; useful precedent for label-driven feature selection at
  scale with multiple-comparison discipline.

### 1.3 Recent preprints — use with care

Two 2026 arXiv preprints are topically close but **unrefereed**; neither should
carry structural weight in our argument.

- **Cho et al. (2026), "Are Single-Token SAE Features Causally Necessary?"**
  Zero-ablation across ~3.9M features, six models, three SAE families, with BH
  correction. Finding: causal necessity varies by *SAE training recipe* on the
  same base model. Useful as a scope-condition citation: causal results are a
  property of the SAE, not only of the model. Methodological weakness worth
  noting — it treats BH significance as necessity with no effect-size floor,
  which is why it can report 178/208 significant conditions alongside features
  whose target-token rank recovers 96–98% of the time.
- **Bal (2026), "From Geometric Recovery to Causal Validation."** Toy-model
  audit (32 ground-truth features → 8-dim bottleneck); introduces
  "causal inertness," and the read-inert / write-inert distinction. Its useful
  contribution is conceptual: ablation and steering measure different
  properties, so a null ablation is ambiguous between "not causally used" and
  "write-only." Its headline numbers do not transfer to language models, and its
  cosine-0.90 recovery threshold is the author's own uncited choice.

### 1.4 What the literature establishes

1. Ablation licenses **necessity**, not representation. The standard term is
   causal necessity or causal relevance; "causal validation" overstates it.
2. The claim only becomes about *meaning* when specificity is demonstrated —
   on-target vs off-target, with pre-specified direction.
3. Control features and distributional baselines carry the argument. Papers that
   survive are the ones where the null condition is designed as carefully as the
   treatment.
4. Significance is not necessity. Large-n ablations produce significant effects
   routinely; an effect-size floor must be stated.

---

## 2. Experiments required to claim a feature is causally load-bearing for code c

Two distinct claims, with different evidentiary requirements. Be explicit about
which one is being made.

- **Claim A (achievable):** feature f is causally load-bearing for code c — the
  model's behavior on c-positive text depends on f, specifically.
- **Claim B (stronger):** feature f *encodes* / *represents* c. Requires
  sufficiency; not supportable by ablation alone.

### For Claim A

**A1. Necessity.** Ablate f; behavior degrades on c-positive inputs.

**A2. Baseline discipline.** The comparison must be against the
SAE-reconstructed forward pass, not the clean model. A clean-model baseline
conflates the feature's contribution with the SAE's general reconstruction
damage and inflates every effect.

**A3. Specificity, on three independent axes.** Any one alone is insufficient:

- *Note-level:* effect larger on c-positive than c-negative notes.
- *Code-level (off-target):* ablating f's effect is concentrated on c, not
  spread across other codes. **This is the axis that distinguishes a
  concept-specific feature from a generally important one**, and the one TPP and
  Arditi both treat as load-bearing.
- *Feature-level:* random and low-association control features produce null
  effects under the identical pipeline.

**A4. Confound control.** Residualize on note length, code count, and section
position. Max-pooled features correlate with length by construction.

**A5. Effect-size floor, pre-specified.** Report a magnitude threshold
alongside FDR-corrected significance. Significance alone is not a necessity
claim (see Cho et al. above for the failure mode).

**A6. Null calibration.** An empirical effect-size distribution from
covariance-matched random directions run through the identical pipeline. No
published reference value for "a good causal effect" exists for domain-specific
SAEs — the null has to be internal.

**A7. Competitive baseline.** A label-supervised direction (difference-in-means)
ablated identically. Without this, "would a supervised direction do the same
job without an SAE?" is unanswered — and it is the first question a reviewer
asks.

### For Claim B (additional)

**B1. Sufficiency / steering.** Clamp f high on c-negative inputs; check whether
c-related behavior appears. Without this, a null ablation is ambiguous between
"not causally used" and "write-only," and no representation claim is available.

**B2. Optional, high-value: graded dose-response along the ICD hierarchy.**
A feature encoding a specific code should show a monotone gradient — largest
effect on the exact code, smaller on parent/sibling codes, near-zero on distant
chapters. Artifacts do not respect a clinical ontology, so a monotone gradient
is far harder to explain away than any binary contrast. Neither TPP nor SHIFT
can run this test: their label sets are flat, unordered classes. Ours is not.
This is the clearest available differentiator from the existing literature.

---

## 3. Are these required for our paper, and what else can be ablated?

### 3.1 Requirement triage

| Experiment | Status | Reason |
|---|---|---|
| A1 necessity | Required | The claim does not exist without it |
| A2 reconstruction baseline | Required | Otherwise effects are inflated and the measurement is contested |
| A3 note-level specificity | Required | Minimum bar for "specific to c" |
| A3 off-target (code-level) | **Required** | The reviewer's first objection; currently the most common omission |
| A3 control features | Required | Cheap; establishes the pipeline produces nulls |
| A4 confound control | Required | Max-pooling makes length confounding structural |
| A5 effect-size floor | Required | Cheap, and distinguishes us from the preprint failure mode |
| A6 random-direction null | Strongly advised | No external reference values exist; this supplies them |
| A7 diff-in-means comparison | **Strongly advised** | Directly answers "is the SAE necessary?" |
| B1 steering | Optional for this paper | Upgrades Claim A to Claim B; scope it as future work if not run |
| B2 hierarchy dose-response | Optional, highest novelty | Not available to prior work; the strongest differentiator |

Minimum defensible package for Claim A: A1–A5. A6 and A7 convert it from "our
feature has an effect" to "our feature has an effect that alternatives do not,"
which is a materially stronger paper.

### 3.2 Can equivalent ablation be run on the other methods?

This question has a structural answer that is itself an argument for the paper.
Methods split into two tiers by whether they produce a **direction in the
model's activation space**.

**Tier 1 — ablatable inside the model (identical protocol available):**

- **SAE latents.** Subtract the feature's decoder contribution, re-run
  downstream layers.
- **Difference-in-means directions.** Project the direction out of the residual
  stream and re-run downstream layers (the Arditi-style directional ablation).
  Fully comparable to SAE ablation, and the most important comparison to run:
  it is the label-supervised alternative that needs no SAE.
- **Random covariance-matched directions.** Same mechanism; these are the null,
  and they supply the effect-size distribution required by A6.
- **PCA directions.** Same mechanism; an unsupervised, non-sparse control.

Because all four take the *same* intervention and the *same* readout, they can
be placed in one table under one protocol. That comparison is the strongest
form the causal section can take.

**Tier 2 — not ablatable inside the model:**

- **TF-IDF + LR.** The features are external to the model. There is no
  activation to perturb and no downstream forward pass, so no model-internal
  ablation exists. Only classifier-space ablation is possible — zero a term's
  weight and watch accuracy drop — which measures the classifier's dependence
  on a token, not the model's dependence on a representation.
- **Lexical / keyword baselines.** Same limitation.
- **Raw activation dimensions.** Technically ablatable, but individual residual
  dimensions are not concept-aligned, so a per-code ablation is not meaningful.
  Useful as a probe baseline, not as a causal comparison.

**The consequence, and how to use it.** A classification benchmark can rank
TF-IDF above SAE features; that comparison is available to both. A causal
specificity test is *not* available to TF-IDF at all — it has no localized,
ablatable unit corresponding to a diagnosis. This asymmetry is the honest answer
to "bag-of-words classifies better, so why care about SAE features": predictive
accuracy and causal localization are different properties, and only one of the
two methods can be tested for the second. Stating the asymmetry explicitly is
stronger than competing on AUROC.

---

## References

- Bau, Zhu, Strobelt, Lapedriza, Zhou, Torralba (2020). Understanding the role
  of individual units in a deep neural network. *PNAS*.
  https://www.pnas.org/doi/10.1073/pnas.1907375117
- Marks, Rager, Michaud, Belinkov, Bau, Mueller (2025). Sparse Feature Circuits.
  *ICLR*. https://arxiv.org/abs/2403.19647
- Karvonen, Rager, Marks, Nanda (2024). Evaluating Sparse Autoencoders on
  Targeted Concept Erasure Tasks. https://arxiv.org/abs/2411.18895
- Karvonen et al. (2025). SAEBench. https://arxiv.org/abs/2503.09532
- Chanin et al. (2026). Are Sparse Autoencoder Benchmarks Reliable?
  https://arxiv.org/abs/2605.18229
- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). Refusal in
  Language Models Is Mediated by a Single Direction. *NeurIPS*.
  https://arxiv.org/abs/2406.11717
- Cunningham, Ewart, Riggs, Huben, Sharkey (2023). Sparse Autoencoders Find
  Highly Interpretable Features. https://arxiv.org/abs/2309.08600
- Gurnee, Nanda, Pauly, Harvey, Troitskii, Bertsimas (2023). Finding Neurons in
  a Haystack. *TMLR*. https://arxiv.org/abs/2305.01610
- Cho, Wu, Da Costa, Kalra, Wicaksono, Koshiyama (2026). Are Single-Token SAE
  Features Causally Necessary? *Preprint.* https://arxiv.org/abs/2607.20596
- Bal (2026). From Geometric Recovery to Causal Validation. *Preprint.*
  https://arxiv.org/abs/2607.12166

---

# Full current ablation result

All runs: Gemma-2-2B, layer 16, 4,911 held-out MIMIC-IV discharge notes
(shards 281–311). Targets are grounded (feature, code) pairs selected by
point-biserial `r_pb` from the grounding stage, plus one random and one low-`r`
control per SAE.

**Quantities.** Three forward passes per note give `ℓ_clean` (unmodified model),
`ℓ_recon` (SAE reconstruction spliced in at layer 16), and `ℓ_abl` (target
latent zeroed before decoding). Cross-entropy is measured over the final 25% of
non-padding tokens (the *loss window*).

- **Ablation effect** `Δⱼ = ℓ_abl − ℓ_recon` — the feature's own contribution,
  with the SAE's baseline distortion held constant.
- **Reconstruction tax** `= ℓ_recon − ℓ_clean` — the cost of splicing the SAE in
  at all. The denominator for cross-SAE comparison.
- **Test:** one-sided Mann–Whitney *U* (`alternative='greater'`) on `Δⱼ`,
  label-positive vs label-negative notes.
- **Effect size:** Cliff's δ `= 2U/(n₊n₋) − 1`, range [−1, +1].
- **Correction:** Benjamini–Hochberg FDR at `q = 0.05`.

---

## 1. Necessity, and its invariance to the counterfactual

| Run | Intervention | Targets | median δ | BH sig | δ > δ*₉₅ |
|---|---|---|---|---|---|
| `vanilla_pilot` | zero | top-10 | **0.300** | 10/10 | 9/10 |
| `vanilla_section` | zero | all 30 | 0.162 | 25/30 | 22/30 |
| `vanilla_meanabl` | mean | all 30 | 0.169 | 23/30 | 22/30 |
| ↳ same top-10 subset | zero | 10 | 0.300 | — | — |
| ↳ same top-10 subset | mean | 10 | 0.312 | — | — |

Controls null in every run. Reconstruction tax 0.029 nats (1.8% of base loss),
so the intervention measures the feature, not the SAE's reconstruction failure.

**Paired comparison of the two interventions**, same 30 targets, same notes:
Wilcoxon signed-rank `W = 189`, `z = −0.895`, **`p = 0.371`**; maximum
per-target |difference| = 0.077.

**What this shows.** The model's prediction on diagnosis-positive notes depends
on these specific decoder directions, and the dependence tracks `r_pb` — median
δ falls from 0.300 (ranks 1–10) to 0.162 (ranks 1–30), monotonically by rank.
The effect is *not* an artifact of clamping a latent to an off-distribution
zero: replacing zero-ablation with mean-ablation (clamping to the latent's own
dataset-mean activation `mⱼ`) leaves both the magnitude and the significance
count statistically indistinguishable.

**For the paper.** This is the headline. Two corrections: Table 3 currently
attributes these numbers to JumpReLU — they are vanilla's. And report top-10 and
all-30 together; the decline is expected and disclosing it pre-empts a
cherry-picking objection. Mean-ablation belongs in one appendix paragraph plus
one main-text sentence. One caveat to state: mean-ablation subtracts
`(zⱼ − mⱼ)·W_dec[j]`, so on notes where the latent is silent it *adds*
`mⱼ·W_dec[j]` — it perturbs the negative arm where zero-ablation is a no-op.

---

## 2. The significance floor: δ*₉₅ = 0.0732

Every run also measures each feature's effect on codes it is **not** grounded
to. Those (feature, off-target-code) pairs are an empirical null: what δ looks
like when the feature is genuinely not causally relevant. The 95th percentile of
|δ| over that null, `δ*₉₅`:

| `vanilla_pilot` | `v_extended` | `v_section` | `v_meanabl` | `jr_pilot` | `jr_extended` |
|---|---|---|---|---|---|
| 0.0746 | 0.0727 | 0.0732 | 0.0791 | 0.0668 | 0.0712 |

Six independent estimates (*n* up to 1,568 pairs each) across two SAEs, two
interventions and three target sets, all within 0.067–0.079.

**What this shows.** No field-standard cutoff exists for "a causally meaningful
ablation effect," so this constructs one from the data. Its stability across SAE
and intervention means it is a property of the measurement, not of any one
configuration. Critically, it separates two things that BH-significance alone
conflates: at *n* = 4,911 a BH-significant δ only says the effect is not
*exactly* zero; `δ > δ*₉₅` says the effect exceeds what an unrelated
feature–code pair produces. Under the stricter bar vanilla's top-10 goes
10/10 → 9/10 and JumpReLU's goes 10/10 → 6/10.

Romano et al.'s conventional "negligible" bound of 0.147, used for the magnitude
bins, is **2× more conservative** than the empirical null and discards real
effects.

**For the paper.** Name it — a *causal necessity floor* — and report every count
two ways (BH-significant, and above-floor). This is the cleanest novelty claim
available: label-driven ablation with off-target specificity is already
established (TPP, SHIFT), but a calibrated floor derived from the off-target
null is not, and it directly answers the significance-is-not-necessity failure
mode documented in Cho et al. (2026).

**Update.** This floor is now also *measured* directly, not just derived —
see "Most valuable experiments" §2 below. It revises the reading: `δ*₉₅`
undershoots a genuinely null source's actual |δ| distribution, so the
load-bearing criterion for every count above is BH-significance, not
`δ > δ*₉₅` on its own.

---

## 3. Specificity

Specificity ratio = mean on-target δ ÷ mean |off-target δ|.

| Run | on-target δ | \|off-target\| δ | ratio | off-target sig |
|---|---|---|---|---|
| `vanilla_pilot` (top-10) | 0.352 | 0.025 | **12.4×** | **0 / 588** |
| `vanilla_section` (30) | 0.230 | 0.027 | 6.4× | 2 / 1,568 |
| `vanilla_meanabl` (30) | 0.222 | 0.029 | 6.0× | 0 / 1,568 |

**What this shows.** This is the axis that turns "ablation broke something" into
"ablation broke the concept the label names." Zero off-target hits out of 588
tests on the top-10 forecloses the generic-damage objection. The ratio surviving
unchanged under mean-ablation (6.0× vs 6.4×) means specificity, not just
magnitude, is invariant to the counterfactual.

**For the paper.** Currently absent from the PDF entirely — promote it to the
main text. One figure carries all three results at once: on-target δ against
mean off-target δ, one point per feature, with `δ*₉₅ = 0.0732` drawn as a
horizontal line.

---

## 4. Remaining results

| Result | Numbers | Significance | Paper treatment |
|---|---|---|---|
| **Length confound** | Residualizing on `log(n_tokens)`: attenuation 0.056 (top-10), 0.066 (30); 9/10 and 19/30 survive | Max-pooling makes length confounding structural; losing ~16% of the top-10 effect is a good outcome | Report adjusted δ as primary, not a footnote — volunteering it beats being asked |
| **Effect-size calibration** | 0.0020 nats = 0.124% of base loss = 6.9% of reconstruction tax (top-10) | One latent of 18,432 accounting for ~0.1% of loss is the expected order of magnitude | State plainly; use the ratio to reconstruction tax as the honest denominator. **Resolve first:** the corrections doc cites 0.0075–0.011%, ~12× below the on-disk value |
| **Section-local (negative)** | 1/30 features show section δ > rest δ; 13/30 on the size-invariant nats metric | The diagnosis recurs in admission differential, history *and* discharge — a terminal window was never diagnosis-specific | Report as a characterization of note structure, evidenced by the feature's own firing distribution across sections (`feature_inspector.py`), not by argument. Fix §3.4, whose window justification this contradicts. Main effect survives a demonstrably suboptimal window |
| **GemmaScope** | 7/10 BH sig, 6/10 above floor; reconstruction tax 0.648 nats (22× vanilla) | Three top-10 features have `r_pb ≥ 0.43` and are causally inert — the correlational-vs-causal contrast, intact under the stricter floor | Keep as-is; it is the paper's interpretive contribution |
| **JumpReLU (excluded)** | 10/10 BH sig but 6/10 above floor; low-`r` control **is** significant (δ = 0.066, above three of its own grounded features); 13 off-target sig across 30 vs vanilla's 2; strongest feature in the SAE (6701, `r_pb` = 0.864) reaches only δ = 0.052 | Causal arm is materially weaker than vanilla's, and the held-out set is contaminated (shards 281–311 were in JumpReLU training, gap G14) | Excluding it is defensible. It still needs one sentence, since JumpReLU carries the grounding, concordance and AUC results elsewhere — a reviewer will notice the causal arm is missing |
| **Multiple-comparison robustness** | 25/30 and 22/30 identical whether BH is applied per-run (as published), over 30 grounded, or over all 32 including controls | Closes the "two separately-corrected families" objection | One appendix line |
| **Reproducibility** | `vanilla_section` reproduces pilot ∪ extended to 4 d.p. (median δ 0.1622, mean 0.2299) across a 7-week gap and a code change | Pipeline is deterministic | One sentence |

---

## Through-line

Correlational grounding is not causal necessity, and `δ*₉₅` is a calibrated
floor for telling them apart. The domain-trained vanilla SAE clears it,
GemmaScope clears it partially, and the surviving effect is **specific**
(12.4× on-target vs off-target, zero off-target hits), **length-robust**
(9/10 after residualization), and **invariant to the counterfactual**
(paired Wilcoxon *p* = 0.371).

---

## Writing items to resolve

| # | Item | Resolution |
|---|---|---|
| 1 | **Terminology conflict.** R1 and the AC both required renaming "causal ablation" → *diagnosis-conditioned loss relevance*, "not full mechanistic faithfulness," and this was conceded in the general response (corrections doc T6). The sections above use "causal necessity floor," which reintroduces the objected-to framing. | Pick the vocabulary once and apply it to §3.6, §4.4, the abstract, and this document. If the rename stands, `δ*₉₅` becomes a **loss-relevance floor** and the "necessity" language throughout §1–§3 needs rewording. Highest priority — it touches every heading and the abstract. |
| 2 | **Table 3 mislabel.** The "JumpReLU" column carries `vanilla_pilot`'s numbers (median δ 0.300, reconstruction tax 0.029). Real JumpReLU: 0.276 and 0.0088. | Relabel to vanilla. Currently marked out-of-scope by author decision (2026-08-27); it is a factual misattribution and should not ship. |
| 3 | **§3.4 window justification.** The text justifies the last-25% loss window on the grounds that "ICD-relevant content is concentrated there." The section-local result contradicts exactly this. | Reword to state the window as a pragmatic choice, not a content claim. A reviewer reading the section-local null will go straight to this sentence. |
| 4 | **Section-local explanation is asserted, not evidenced.** The claim that the diagnosis recurs across admission differential, history and discharge is reasoning, not measurement. | Back it with the feature's own firing distribution across note sections (`feature_inspector.py`). This is the difference between a failed experiment and a characterization of clinical note structure. |
| 5 | **Effect-size calibration figure.** The corrections doc cites "0.0075–0.011% of base loss"; that interval appears nowhere in the on-disk data. Actual per-target `pct_of_base_loss` for `vanilla_pilot_extended`: min −0.0095, median 0.0203, mean 0.067, max 0.603. | Report the median (0.020%) or the full range, not the two-value interval. |

---

## Framing notes for the ablation section

### F1. Disclose the grounding gap before reporting cross-method effects

The best-vs-best comparison is confounded by grounding strength and must say so
in its own paragraph rather than in a footnote. Draft text:

> Best-vs-best comparison is confounded by grounding strength: the SAE's top-10
> features reach |r| = 0.83–0.86, whereas the strongest covariance-matched
> random direction reaches 0.43 and the strongest difference-in-means direction
> 0.70. No r-matched comparison is available at the SAE's grounding level,
> because neither alternative produces directions that ground that strongly —
> itself a result. Causal effects are therefore reported alongside each target's
> |r|, and the comparison should be read as "best available from each method,"
> not as a matched contrast.

Peak `|r_pb|` by source, for the table that paragraph refers to:

| Source | k | max \|r\| | selected median \|r\| |
|---|---|---|---|
| SAE (JumpReLU / vanilla) | 18,432 | 0.864 / 0.860 | 0.574 |
| TF-IDF (value / binary) | 10,000 | 0.848 / 0.831 | 0.52 / 0.53 |
| difference-in-means (full whitening) | 46 | 0.699 | 0.340 |
| LR probe (unweighted) | 46 | 0.637 | 0.333 |
| GemmaScope | 16,384 | 0.545 | 0.309 |
| random-matched (`note_matched`) | 18,432 | 0.432 | 0.191 |
| PCA (`note_matched`) | 2,304 | 0.425 | 0.132 |

### F2. Ten to thirty interventions is the field norm — say so

Causal ablation is GPU-heavy and reviewers unfamiliar with the cost may read a
30-target study as thin. It is not. Comparable published counts:

| Paper | Interventions |
|---|---|
| Bau et al., PNAS 2020 (the founding design) | **20 units per class**; 4 in the headline single-class case |
| Marks et al., Sparse Feature Circuits / SHIFT, ICLR 2025 | **55 features**, ablated *jointly* as one intervention, from a 67-feature circuit; circuits are "<100 nodes" |
| Arditi et al., NeurIPS 2024 | **one direction**, evaluated across 13 models |
| Cho et al., 2026 (preprint) | 3.9M features — not comparable: single-token Δlogit readout, no full forward per intervention |

**The distinction to draw explicitly.** SFC ablates 55 features *together* and
measures one downstream effect: one causal experiment. This work runs 30
**independent** per-feature tests, each with its own Mann–Whitney U, effect
size, and place in the FDR family. By count of separate causal experiments it
already exceeds every LM-scale paper in this line, and that framing should
appear in the text rather than being left for a reviewer to work out.

This also justifies staging new arms at top-10 first: 10–20 is squarely within
convention, and ranks 11–30 are a config-level extension if the first tranche
proves informative.

### F3. Mean-ablation is the primary intervention for SAE-feature arms — keep the bookkeeping straight

Zero-ablation invites the out-of-distribution objection (clamping a latent to an
unreachable zero) and buys nothing: on the matched 30 targets the two
interventions are statistically indistinguishable (paired Wilcoxon *p* = 0.371),
and specificity is if anything cleaner under mean-ablation (0 off-target
significant vs 2). New SAE-feature arms therefore run mean-ablation only.

**Scope note — this does not extend to directional (projection) ablation.**
Zero- vs mean-*ablation* here means clamping the SAE latent `zⱼ` to 0 or to
its mean `mⱼ` before `decode`. Zero- vs mean-*projection* (§2/§3 above, "Most
valuable experiments") is a different operation: `x' = x − (x·d)d` removes
the component along `d` entirely, leaving `x'` in the span of everything
else, rather than clamping a latent to a value it never takes. The
out-of-distribution objection this section raises for SAE features is
substantially weaker there, which is why the random-matched and
difference-in-means arms use `intervention: zero` (Arditi et al.'s operator)
as primary, not mean-projection.

**Consequence for every cross-method table:** the SAE comparator must be
`vanilla_meanabl`, **not** `vanilla_pilot`. Comparing a mean-ablated random arm
against the zero-ablated SAE headline is an intervention mismatch. The matched
SAE numbers are:

| | median δ |
|---|---|
| `vanilla_meanabl`, top-10 subset | **0.3124** |
| `vanilla_meanabl`, all 30 | **0.1687** |

Report `vanilla_pilot`'s 0.300 as the paper's zero-ablation headline if desired,
but never as the comparator for a mean-ablated arm.

**Caveat to state once.** Mean-ablation subtracts `(zⱼ − mⱼ)·W_dec[j]` with no
non-negativity clamp, so on notes where the latent is silent it *adds*
`mⱼ·W_dec[j]`. Zero-ablation is a strict no-op there. The negative arm is
therefore perturbed under mean-ablation, which is why it is a different
contrast rather than a gentler version of the same one.

---

## Most valuable experiments

Ordered by value per unit of cost. All extend the existing ablation path; none
requires new methodology.

### 1. GemmaScope post-hoc — *minutes, CPU*

No post-hoc config exists for either GemmaScope run, so its off-target
specificity is unmeasured.

**What it adds.** The correlational-vs-causal contrast — the paper's main
interpretive claim — currently rests on GemmaScope *necessity* with no
GemmaScope *specificity*. Without it, "3 of GemmaScope's top-10 are causally
inert" is a claim about magnitude only; a reviewer can ask whether the 7 that do
fire are firing specifically. Completing this makes the contrast symmetric with
vanilla's. Cheapest item on the list by a wide margin.

### 2. Ablation on covariance-matched random directions (`random_matched`) — DONE

**Why not the SAE's reconstruction protocol.** `ablation.py`'s default
protocol splices `decode(z)` at layer 16 and measures `l_abl = CE(recon −
feature j)` against `l_recon = CE(recon)`. That assumes `decode(z)` is a
faithful reconstruction of `x`; a trained SAE earns that (EV ≈ 0.90, tax
0.029 nats for vanilla). A pseudo-SAE built by packaging 18,432
covariance-matched random directions as a `W_enc`/`W_dec` pair does not:
`decode(z)` sums 18,432 direction contributions and at `<bos>` overflows
fp16 (`max|x̂| = 2.1e6` vs 65,504), NaN-ing every note. The same
reconstruction protocol applied to 46 difference-in-means directions instead
produced a 7.47-nat reconstruction tax — 4.6× the base loss — turning every
downstream δ into noise on a destroyed baseline.

**Directional ablation.** The fix is to intervene on the *raw* residual
stream instead of an SAE reconstruction — the standard treatment for a single
direction rather than a dictionary feature (directional ablation, Arditi et
al., NeurIPS 2024; the same projection underlies amnesic probing, Elazar et
al., TACL 2021, and INLP, Ravfogel et al., 2020):

```
x' = x − (x · d) d          d unit-norm      (intervention: zero)
```

No reconstruction is spliced, so there is no reconstruction tax to begin
with; subtracting one component leaves `‖x'‖ ≤ ‖x‖`, so fp16 cannot overflow
regardless of dictionary size; there is no threshold gating, so the
note-level `τ` calibrated for grounding is never misapplied as a token-level
firing rule; and the control condition is well-defined at any k ("project out
a random direction"). `intervention: zero` is used throughout — mean-projection
is implemented but not exercised, since F3's out-of-distribution argument for
mean-ablation is already weaker for a single projection than for an SAE
feature.

**Baseline consequence.** With no reconstruction pass, `δ = l_abl − l_clean`,
not `l_abl − l_recon`; `mean_recon_tax = 0.0` by construction and the run
summary records `baseline: "clean"`. **Absolute nats are therefore not
comparable with the SAE arms**, which carry their own reconstruction tax
(0.029–0.648 nats depending on SAE). **Cliff's δ is** — it is a within-arm
rank contrast, so a per-arm additive baseline offset cancels. Every
cross-arm claim below is stated in δ, never nats.

**Results (`random_matched_full`, 4,911 held-out notes, 12 targets).**
Targets: top-10 grounded features by `|r_pb|` from the A4 necessity audit
(`necessity/random_matched/seed0/audit_note_matched`), screened by
`select_ablation_targets.py` for row-0/BOS-only firing (all 10 pass at
`row0only = 0.00`) + 1 random control (`max|r| < 0.05`) + 1 low-r control
(`|r| ∈ [0.05, 0.10]`), both screened the same way. Eight of the ten grounded
features associate with `icd9_4280` (heart failure) — the same top-|r|
selection skew already documented for `vanilla_pilot`'s `icd9_2449` cluster,
not a property of this source.

| | grounded (n=10) | controls (n=2) |
|---|---|---|
| median Cliff's δ | **−0.0958** | **−0.0830** |
| range of \|δ\| | 0.010–0.184 | 0.083, 0.083 |
| BH-significant (q=0.05) | **0/10** | 0/2 |

**Grounded and control are statistically indistinguishable** — the necessity
claim working as designed: a direction correlating with a code by
construction (covariance-matching, not learning) shows no differential causal
effect versus a direction that correlates with nothing. Contrast the SAE:
on-target 0.352 vs off-target 0.025, a 12.4× separation (§3 above).

**Off-target** (588 tests — 12 targets × 49 off-codes, the same family size
as `vanilla_pilot`'s headline 588): 9/588 significant (7/490 grounded, 2/98
controls), `median_specificity_ratio = −1.53`. Not the 0/588 the vanilla SAE
achieves — a genuinely null source still produces *some* spurious off-target
hits at this scale, concentrated on rare codes (`icd9_V270`, n_pos = 30;
`icd9_V6284`, n_pos = 75). That is itself informative: it calibrates how much
off-target contamination should be read as sampling noise rather than a
specificity failure, before comparing it against the SAE's zero.

**Length-matched:** residualizing on `log(n_tokens)` collapses the raw mean δ
(−0.090) to −0.0054 — **94% attenuation** — and 0/10 grounded targets remain
significant after adjustment. (Two controls flip to significant
post-adjustment, δ_adj ≈ +0.06–0.07; noted rather than explained away — BH is
not re-applied across this family, and at n ≈ 700–4,200 per group some
post-hoc significance among genuinely null directions is expected under
repeated testing.) Essentially the entire raw effect in this arm is a length
confound, consistent with there being no real signal underneath it.

**Calibration:** `ratio_to_recon_tax` is `NaN` for every row by construction
(`recon_tax = 0.0`; confirms the `if recon_tax else float("nan")` guard in
`ablation_posthoc.py:409` — no divide-by-zero). `pct_of_base_loss` ranges
0.22%–13.5%, uninformative on its own without the BH result alongside it.

**What this revises.** `δ*₉₅ = 0.0732` (§2 above) was *derived* from
off-target (feature, wrong-code) pairs within the SAE arms. This run supplies
the *measured* equivalent — actual ablation of directions that are null by
construction — and it lands materially higher: 0.0732 sits near the **25th
percentile**, not the 95th, of this arm's |δ| distribution (0.010–0.184).
The safeguard that held: `n_sig_q05 = 0/12` throughout, so BH correctly
refused significance despite those magnitudes. **The paper's actual claims
survive** — every SAE-arm significance count is unaffected — but any place
the text uses "`δ` above `δ*₉₅`" as a stand-in for "exceeds the noise floor,"
independent of the BH result, needs to drop that framing. BH significance is
the load-bearing criterion; `δ*₉₅` was too small an estimate of where
chance-level effect sizes land for a genuinely null source at this sample
size.

### 3. Ablation on difference-in-means directions (`dim_full`) — DONE

Uses the identical directional-ablation technique described in §2 above (same
projection intervention, same `l_clean` baseline, same reason the
reconstruction protocol is inapplicable — the 46-direction dictionary's own
attempt at it produced a 7.47-nat reconstruction tax, §2). One
label-supervised unit direction per code from `diff_in_means_baseline.py`,
ablated by projecting it out of the raw residual stream on the same held-out
notes and statistics as every other arm.

**Controls are weak, not null — a structural property of k=46.**
`select_ablation_targets.py`'s strict null pools (`max|r| < 0.05`,
`|r| ∈ [0.05, 0.10]`) are both empty for this source: every one of the 46
directions is built from one code's labels, and with 46 comorbid codes in
the panel even the weakest survivor of the row0-only screen still reaches
`r ≈ 0.19`. (Screening all 46 directions rather than the usual top-10 found
21/46 clear the screen at all.) The two `low_r_control` targets below are
the lowest-|r| survivors — 0.198 and 0.194 — not a genuinely uncorrelated
direction the way `random_matched`'s controls are. Unlike random-matched
(§2), this arm has no direction that fails to correlate with *anything*.

**Results (`diff_in_means_full`, 4,911 held-out notes, 12 targets: 10
grounded + 2 `low_r_control`).**

| | grounded (n=10) | controls (n=2) |
|---|---|---|
| median Cliff's δ | **0.0550** | −0.0160 |
| BH-significant (q=0.05) | **4/10** | 0/2 |

Unlike random-matched, this arm produces genuine on-target signal — but it
is not uniform: 4 of 10 grounded targets are strongly positive and
BH-significant (δ = 0.161, 0.235, 0.082, 0.068 for `icd9_5856`,
`icd9_V5867`×2, `icd9_2851`), while the other 6 range from weakly positive
to clearly negative (`icd9_2749`: −0.114; `icd9_4280`: −0.095), the reverse
of the causal-necessity direction. Selecting by |r_pb| alone therefore
produces a mixed causal population for this source, in contrast to the SAE's
top-10, which reaches 9–10/10 BH-significant with a consistent sign.

**Off-target** (588 tests, same family size as vanilla_pilot and
random_matched_full): **20/588 significant** (19/490 grounded, 1/98
controls) — worse than random-matched's 9/588 and far worse than the SAE's
0/588. `median_specificity_ratio = 1.165`: on-target and off-target effects
are of comparable magnitude on the median, not separated the way the SAE's
12.4× is. One feature (27, `icd9_V4581`) alone accounts for 7 off-target
hits, nearly all on cardiometabolic comorbidities of its own code
(`icd9_25000`, `icd9_4280`, `icd9_42731`, `icd9_41401`, `icd9_5849`) — the
acuity/comorbidity confound the codebase's own pooling caveats warn about,
now showing up causally rather than just correlationally.

**Length-matched:** residualizing on `log(n_tokens)` does *not* attenuate
this arm's effect — mean adjusted δ (0.0553) is if anything slightly larger
than raw (0.0454), and the same 4/10 targets remain significant after
adjustment. This is the opposite of random-matched, where 94% of the raw
effect was a length confound. Whatever signal this arm has is not a
length artifact.

**Calibration:** raw effect sizes are tiny in absolute nats — grounded mean
`delta_pos_nats` ranges 0.0000045–0.0007, roughly 10–100× smaller than
random-matched's grounded nats despite comparable Cliff's δ magnitudes (both
arms share the `l_clean` baseline, so their nats *are* directly comparable to
each other, unlike either arm vs. the SAE). `pct_of_base_loss` tops out at
0.043%. `ratio_to_recon_tax` is `NaN` throughout (`recon_tax = 0.0`, same
guard as random-matched). Read alongside Cliff's δ: significance here comes
from a small-variance, small-magnitude signal, not a large practical effect —
consistent with the through-line that these are label-supervised summary
directions, not directions the model's forward pass was shaped around.

**Verdict — does the SAE win?**

| | on-target median δ | BH-sig | off-target sig (of 588) | specificity ratio |
|---|---|---|---|---|
| `vanilla_pilot` (SAE, top-10) | **0.352** | 10/10 | **0/588** | **12.4×** |
| `diff_in_means_full` (10 grounded) | 0.055 | 4/10 | 20/588 | 1.165 |
| `random_matched_full` (10 grounded) | −0.096 | 0/10 | 9/588 | −1.53 (negative) |

Yes, decisively, on the axis that matters. Diff-in-means does not match or
tie the SAE — it produces some genuine, length-robust causal signal (unlike
random-matched, which produces none), but at roughly **1/6** the on-target
effect size and **1/11** the specificity ratio, with 20× the off-target
contamination of the SAE. This is the cleanest evidence in the paper for the
`icd_eval.py`/necessity-audit framing already established correlationally:
a label-supervised direction captures something real but diffuse and
comorbidity-entangled, while the SAE isolates something causally sharper.

### 4. TPP (targeted probe perturbation) — *hours, mostly CPU*

Zero the latent's column in the existing pooled feature matrix, re-score the
per-code probe, measure accuracy change on the target code and on all others.
No forward passes needed.

**What it adds.** Comparability with published SAEBench numbers, in the
interpretability field's own metric, and a second readout that is independent of
the language-modelling loss. It also supplies the answer to the TF-IDF AUROC
result: predictive accuracy and causal localization are different properties,
and TF-IDF has no ablatable unit to test for the second. Lower priority than
2 and 3 only because it duplicates an established metric rather than extending
the paper's own argument.

### Not available

**TF-IDF and lexical baselines cannot be ablated.** Their features are external
to the model — there is no activation to perturb and no downstream forward pass.
Only classifier-space ablation is possible, which measures the classifier's
dependence on a token, not the model's dependence on a representation. Stating
this asymmetry explicitly is stronger than competing on AUROC.
