# BOS contamination in max-pooled grounding — audit

Status: vanilla and JumpReLU cleared. GemmaScope contaminated, needs a re-run
with the correct encoder convention. Constructed sources (random-matched,
diff-in-means) confirmed contaminated separately.

---

## 1. What BOS is, and how it can reach a correlation

Every note is tokenised with `add_special_tokens=True`, so row 0 of every
stored activation block is Gemma's `<bos>`. Verified directly
(`scripts/inspect_position_zero.py`, 8 notes): the first five tokens are
identical everywhere — `<bos>, ' ', '\n', 'Name', ':'` — and `stored_rows ==
tokenised_length` in every note, so the row indexing is correct and row 0 really
is the special token.

`<bos>` is not an ordinary token. Its layer-16 residual has a norm of **2528.6**
against a median of ~162 for real tokens — **15.6×** — the attention-sink /
massive-activation effect. The norm is constant across notes to within fp16
storage noise (max pairwise component difference 0.5 on a vector of norm 2528,
0.02% relative).

### The exact point of entry

Grounding never looks at tokens. It correlates a **note-level** value against a
binary ICD label, and that value is a max over tokens:

```
F[note, j] = max over tokens t of  z_j(t)          # icd_eval.encode_and_pool
r_pb       = point-biserial( F[:, j], y[:, code] ) # icd_eval
```

With row 0 included, this is really

```
F[note, j] = max( c_j , real_max_j(note) )      c_j = the BOS activation
```

which is a **floor** at `c_j`. So contamination requires one thing: `c_j > 0`.
If a latent does not fire on BOS at all, `c_j = 0`, the max is unchanged, and
grounding is untouched no matter how large the BOS residual is.

### Why a floor is not automatically harmless

The intuition "a floor can only attenuate r" is wrong. Point-biserial is
`(M₁ − M₀)/σ · √(pq)`, and flooring does two things at once:

- **shrinks the numerator** — negatives rise from ~0 to `c_j`, narrowing the gap
- **collapses σ** — every below-floor note lands on exactly `c_j`, so
  within-group variance goes to zero

The second effect can dominate, leaving the two groups cleanly separated at
`c_j` and producing a *higher* r than uncontaminated data. Attenuation to r ≈ 0
only happens in the extreme where `c_j` exceeds essentially every note.

---

## 2. Why constructed directions are vulnerable and trained encoders are not

The floor requires `c_j > 0`. Whether a source clears that bar is decided by how
its directions come into existence, and the three families differ structurally.

### Random-matched directions — vulnerable by construction

`D` is sampled from `N(0, Σ_activations)` with unit-norm columns. For *any* unit
direction, the projection of a vector scales with that vector's norm: BOS at
‖x‖ = 2528 projects roughly **15.6× larger** than a typical token at ‖x‖ = 162,
on average, for every direction in the dictionary. Nothing in the sampling
procedure can prefer directions that avoid BOS, because the directions are drawn
before any data is scored.

Thresholds make it worse. `τ` is a quantile of the projection distribution — and
BOS is *in* that distribution. Calibrating to a sparse firing rate therefore
places `τ` near the top of a distribution whose top is BOS, so BOS is exactly
the token that clears it. Measured: at BOS, **7,183 of 18,432** directions fire
simultaneously, while the **median non-BOS token fires 0**.

### Difference-in-means directions — vulnerable at the pooling stage

`d_c = mean(X[y=1]) − mean(X[y=0])` is computed on the **max-pooled** note
matrix `X` `[n_notes, 2304]`, not on tokens. Any dimension where BOS wins the
max is therefore pinned near-constant in `X` *before* the direction is built.
That the pooled covariance is singular (`var_min = 0.0`, `cond = 6.9e16`,
recorded when the whitened arm was added) confirms zero-variance dimensions
exist in that matrix.

#### A whitening hypothesis, raised and refuted — recorded so it is not re-raised

It was proposed here that full-covariance whitening (`d_eff = Σ⁻¹d`) *amplifies*
low-variance BOS-pinned dimensions, and that `whiten: full` reaching peak
|r| = 0.699 against `diagonal` 0.310 and `none` 0.291 was partly this artifact.

The hypothesis makes two predictions, and both fail against evidence produced
when the whitened arm was built:

| Prediction | Result |
|---|---|
| Diagonal whitening divides by per-dimension variance, so it should show the same amplification | **Fails.** 0.121 → 0.127 only, and effective dimensionality *fell*, 1.89 → 1.82. The confound is in the off-diagonal covariance, not per-dimension scale. |
| Gains from near-singular directions should collapse under stronger shrinkage | **Fails.** Flat across shrinkage α ∈ [0.001, 0.1] — two orders of magnitude. |

Ledoit–Wolf shrinkage (mandatory given the singularity) regularises exactly the
directions the hypothesis relies on, and the α-insensitivity shows they are not
carrying the result. Full whitening is the mass-mean probe of Marks & Tegmark
(2023), a principled estimator; its advantage over the diagonal and plain arms
is the ordinary one.

**What remains open** is narrower and needs no covariance argument: whether the
*pooled values* the correlations are computed on are BOS-set for specific
directions. Note that the row0-only screen in §3 is **token-level** (project each
token, compare to threshold) while diff-in-means grounding is **pooled-level** —
different quantities, not yet reconciled. Re-pooling with
`skip_first_token=True` and rebuilding the directions settles both at once, and
is step 2 of the action list rather than a separate investigation.

### Trained SAEs — protected by their objective

A sparse autoencoder pays an L0/L1 price for every latent it fires. BOS is
constant across notes, so it carries no information a reconstruction objective
can use, and spending a latent on it costs sparsity for nothing. The encoder
therefore learns to ignore it. Measured: **14 of 18,432** vanilla latents fire
at BOS at all, and **none** of the top-60 grounded ones.

GemmaScope is the control that proves the mechanism is about training data
rather than architecture: same architecture, trained on general web text where
BOS *is* a meaningful position, and **49.2%** of its dictionary fires there.

---

## 3. How this was found, and how it was confirmed

Not by looking for it. It surfaced as a crash and was traced back.

**Step 1 — an impossible smoke result.** The random-matched ablation smoke
returned `mean_loss_recon = NaN` and `mean_recon_tax = NaN`, while
`mean_loss_clean = 1.6156` was correct and identical to the vanilla SAE's. A
correct clean forward with a NaN reconstruction localises the fault to
encode → decode → splice, not to the model or the loss.

**Step 2 — reproduce without the model** (`diagnose_pseudo_sae_recon.py`).
Replaying encode/decode on cached activations, no GPU: at BOS, 7,183 directions
fired at once and `max|x̂|` reached **2,104,217** against fp16's ceiling of
65,504. `ablation.py:835` casts the spliced tensor to `layer_dtype` (fp16, per
`model.py:38`), so BOS became `inf`, propagated through layers 17+, and produced
NaN in every note. Confirmed on 5/5 notes.

**Step 3 — the incidental finding that mattered more.** The same output showed
**non-BOS median L0 = 0.0** for both pseudo-SAE sources. The overflow was a
symptom; the real problem was that these directions barely fire anywhere else.

**Step 4 — is it the targets, or just the dictionary?**
(`diagnose_target_firing.py`) Per-direction firing on 60 notes, reporting the
fraction of firing notes where row 0 is the *only* token above threshold.
`dim_full` latent 3 — its second-strongest direction, r = 0.667 — came back
row0-only in **100%** of notes.

**Step 5 — confirm row 0 really is BOS** (`inspect_position_zero.py`). This
mattered because row-0 magnitudes differed slightly across notes, and a genuine
shared BOS should be byte-identical. Three checks settled it: re-tokenising the
source text gives `<bos>, ' ', '\n', 'Name', ':'` in every note;
`stored_rows == tokenised_length` everywhere, ruling out an off-by-one in
`row_start`; and `‖row0‖ = 2528.6` in all 8 notes against a median of ~162. The
apparent variation is fp16 storage noise — max pairwise component difference 0.5
on a vector of norm 2528, 0.02% relative.

**Step 6 — how widespread** (`select_ablation_targets.py`). Screening every
candidate rather than a hand-picked few: **12 of 22** `dim_full` candidates were
row0-only, including ranks 2–7. Random-matched's top grounded directions were
clean; two of the four controls I had picked blind were not — which is what
motivated screening in the first place.

**Step 7 — do the SAEs share the problem?**
(`measure_bos_contamination.py`) The answer is section 4. The first version of
that script got its own metric wrong; that is recorded next, because the error
inverted the result.

## 4. A metric error worth recording

The first version of `measure_bos_contamination.py` counted a BOS win as
`c_j >= real_max`. When a latent is simply **silent** on a note, `c_j = 0` and
`real_max = 0`, so `0 >= 0` scored as a win — even though nothing fired and the
pooled value is 0 either way.

That measures **latent sparsity, not contamination**, and it inverts the
ranking: sparse domain-trained SAEs score worse than dense general-purpose ones
purely because they are silent more often.

| SAE | broken metric | what it actually measured |
|---|---|---|
| vanilla | 0.4947 | 49% of note-latent pairs had the latent silent |
| JumpReLU | 0.6274 | 63% silent |
| GemmaScope | 0.1808 | denser, so fewer silent pairs |

Corrected: `bos_wins += (c > 0) & (c >= real_max)`. The decisive prior question
is simply **does the latent fire at BOS at all**.

---

## 5. Results (120 held-out notes, shard 281, top-60 grounded latents)

### Domain-trained SAEs — clean

| | vanilla | JumpReLU |
|---|---|---|
| latents firing at BOS (all) | **14 / 18,432** (0.1%) | **15 / 18,432** (0.1%) |
| `c_j` median / p99 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| **top-60 grounded with `c_j > 0`** | **0 / 60** | **0 / 60** |

Every grounded latent reads `c_j(BOS) = 0.0000`. A latent that never fires on
BOS cannot be floored by it. **Grounding for both domain-trained SAEs is
uncontaminated, and the published correlations stand.**

This is the expected result rather than a lucky one: these SAEs were trained on
clinical activations, and a sparse encoder has no reason to spend capacity on a
constant token carrying no information.

### GemmaScope — contaminated

Re-run with `--no-subtract-b-dec`; figures below are final.

| | GemmaScope |
|---|---|
| latents firing at BOS (all) | **8,058 / 16,384 (49.2%)** |
| `c_j` median / p99 / max | 0.0000 / 118.69 / 1927.86 |
| latents where BOS sets the pool in >50% of notes | 5,347 (32.6%) |
| **top-60 grounded with `c_j > 0`** | **17 / 60** |
| of those, BOS sets the pool in >45% of notes | **9 / 60** |
| grounded mean fraction of pairs BOS sets | 0.1224 |

The nine grounded latents whose reported correlation rests substantially on BOS:

| latent | code | `r_pb` | `c_j` | mean real_max | BOS sets pool |
|---|---|---|---|---|---|
| 4778 | icd9_5856 | +0.2976 | 12.88 | 9.00 | **0.992** |
| 12184 | icd9_2749 | +0.4507 | 27.23 | 19.95 | **0.917** |
| 2402 | icd9_60000 | +0.3162 | 23.27 | 22.95 | 0.767 |
| 14799 | icd9_5856 | +0.3625 | 12.02 | 12.12 | 0.658 |
| 27 | icd9_5990 | +0.2941 | 26.00 | 28.11 | 0.633 |
| 47 | icd9_496 | +0.3188 | 22.12 | 24.60 | 0.625 |
| 12448 | icd9_V5867 | +0.2854 | 27.37 | 24.98 | 0.625 |
| 11544 | icd9_V5867 | +0.3637 | 15.47 | 21.45 | 0.525 |
| 12120 | icd9_41401 | +0.3343 | 37.13 | 50.96 | 0.492 |

Latent 4778 is the clearest case: `c_j = 12.88` against a mean real_max of 9.00,
so BOS sets its pooled value in 99.2% of notes. Its r = 0.298 is a correlation
between an ICD label and a quantity that is almost always the same constant.

Note the pattern in the remaining 43 grounded latents: `c_j = 0.0000` and
`BOS sets pool = 0.000`. Contamination is **concentrated, not diffuse** — a
minority of latents are entirely BOS-driven while the majority are untouched.
That is a per-latent audit problem, not a global correction factor.

**On the `subtract_b_dec` flag.** The first run used `subtract_b_dec=True`,
which is the wrong convention for GemmaScope. The re-run with
`--no-subtract-b-dec` returned **byte-identical** `c_j` values (12184: 27.2313
in both). The reason: `/out/sources/gemmascope_16k` is a pseudo-SAE import, and
`feature_sources.write_pseudo_sae` writes `b_dec = zeros`, so `x − b_dec = x`
either way. The flag was a no-op for this checkpoint and the original figures
were never wrong. Harmless here, since GemmaScope's correct convention is not to
subtract `b_dec` at all — but worth knowing that this imported source cannot
express the distinction.

### Constructed sources — contaminated (measured separately)

From `scripts/diagnose_target_firing.py`:

| source | finding |
|---|---|
| diff-in-means (`dim_full`) | **12 of 22** top candidates clear threshold at BOS **and nowhere else**, including ranks 2–7 (latents 3, 5, 6, 8, 16, 31, 13 at r = 0.469–0.667) |
| random-matched | top-2 grounded clean (row0-only = 0.00); one sampled control was row0-only = 1.00 |

Random directions and label-constructed directions have no mechanism that
avoids a high-magnitude constant token — unlike a trained sparse encoder, which
learns not to spend a latent on it.

---

## 6. Conclusions

1. **Published SAE grounding is safe.** 0 of 60 top-grounded latents in either
   domain-trained SAE fire at BOS. Peak r, monospecificity and the concordance
   join are unaffected.
2. **The comparison arms are not safe**, and the error inflates them. Correcting
   it should *widen* the SAE's margin, not narrow it.
3. **GemmaScope is contaminated in a concentrated way**: 17 of 60 grounded
   latents fire at BOS, 9 of them substantially, and the other 43 are clean.
   Its comparison-table row should be recomputed on a BOS-free pool, or the
   affected latents named. Its peak |r| = 0.545 is not itself among the
   affected set (latent 2121, `c_j` = 0), so the headline is not at risk.
4. This is a difference in kind between trained and constructed feature sources,
   and it is a defensible point in its own right: a trained sparse encoder
   learns to ignore a constant high-magnitude token; random, PCA and
   label-constructed directions cannot.

## 7. Actions

| Step | Status |
|---|---|
| `skip_first_token` on `project_and_pool`, `pool_raw_activations`, `encode_and_pool` (default `False`) | done, behaviourally verified |
| Re-run GemmaScope measurement with `--no-subtract-b-dec` | done — no-op flag, figures unchanged and final |
| Whitening hypothesis (§2) | refuted by existing evidence; no separate check needed |
| Re-pool + re-audit random-matched and diff-in-means with `skip_first_token: true`, into **new** output dirs | pending |
| Thread `skip_first_token` from `run_icd_eval` to `encode_and_pool` | deliberately not done — the SAE path is clean, so the switch stays unwired |

Write corrected artifacts to new directories. The BOS-inclusive numbers are what
the paper currently reports; keeping both is what makes this a correction rather
than a silent replacement.
