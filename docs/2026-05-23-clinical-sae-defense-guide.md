# Clinical SAE Defense Guide

A complete, ground-up walkthrough of what this repository is doing, why each step is there, and how to defend the work to a committee or reviewer. Written to be self-contained — a reader who has never seen the code can read this end-to-end and come out with both intuition and the math.

The structure of the guide is the same as the teaching arc:

- Part I — Pipeline at a glance
- Part II — Inside Gemma (what an activation is, why we extract it)
- Part III — The SAE itself (encoder, decoder, training, JumpReLU)
- Part IV — From SAE codes to "this feature represents sepsis" (pooling, correlation, FDR)
- Part V — Baselines as defense backbone
- Part VI — Defense playbook (15 anticipated attacks + crisp responses)
- Part VII — Appendices: hyperparameter table, key files in the repo, glossary, references

Math is introduced only after the intuition is in place. The conventions are: ASCII diagrams for structure, plain English first, formulae callouts second.

---

## Part I — The pipeline at a glance

What this repository does, end to end:

```
clinical note text (MIMIC-IV discharge summary)
        ↓ run through Gemma-2-2B's first 16 transformer blocks
2304-dim activation, per token, per note
        ↓ subtract the global mean (centering)
2304-dim centered activation
        ↓ push through the trained SAE encoder
18432-dim sparse code, per token, per note (~20-80 nonzero per token)
        ↓ combine all tokens of a note (max pooling per feature)
18432-dim note vector
        ↓ correlate each feature with each ICD-9 code across all notes
correlation matrix [18432 features × ~50 codes]
        ↓ Benjamini-Hochberg FDR correction at q = 0.05
significance mask
        ↓ filter by |r| > 0.1 (default; swept up to 0.5)
"grounded latents" — features whose firing patterns correlate
                     significantly with at least one ICD-9 code
```

The headline scientific claim the pipeline is designed to evaluate:

> Gemma-2-2B, despite never being fine-tuned on clinical text, internally represents clinical concepts in a way that can be extracted via sparse autoencoding. Many SAE latents correlate meaningfully with ICD-9 diagnosis codes, and this exceeds what is achievable with reasonable baselines (a generic pre-trained SAE, keyword regex, TF-IDF + LR, and a linear probe on raw activations).

That claim has hedges, which we'll be explicit about throughout.

---

## Part II — Inside Gemma: what an activation is, and why we extract it

### II.1  Gemma as a chain of identical blocks

Gemma-2-2B is a function. Text in, next-token probability out. That's the whole job. Internally, it does this through a chain of 26 transformer blocks:

```
   text
    ↓
  ┌──────────┐
  │ tokenizer │   integer IDs (one per token)
  └──────────┘
    ↓
  ┌──────────┐
  │ embedding │   each token ID becomes a 2304-dim vector
  └──────────┘
    ↓
  ┌──────────┐
  │  Block 1  │  ← all 26 blocks have the same architecture
  └──────────┘
    ↓
   ...26 times...
    ↓
  ┌──────────┐
  │  unembed  │   convert final vector to next-token probabilities
  └──────────┘
    ↓
predictions
```

Each block reads in `[seq_len, 2304]` (a vector per token) and writes out `[seq_len, 2304]`. The block doesn't change the shape; it changes the content.

### II.2  Inside one block

Each block does two operations:

```
                input  (shape: [seq_len, 2304])
                  │
                  ├──────────────┐
                  │              ↓
                  │        ┌──────────┐
                  │        │ Attention │   lets tokens "look at" each other
                  │        └──────────┘
                  │              │
                  ↓              ↓
                  + ─────────────┘    add the attention output back to input
                  │
                  ├──────────────┐
                  │              ↓
                  │        ┌──────────┐
                  │        │   MLP    │   lets each token "think" about itself
                  │        └──────────┘   (a 2-layer feedforward net)
                  │              │
                  ↓              ↓
                  + ─────────────┘    add the MLP output back
                  │
                  ↓
                output (shape: [seq_len, 2304])
```

Attention is the cross-token operation: each token "looks at" other tokens and pulls in relevant information. This is how the model integrates context — when processing "she" it can attend back to "the patient" earlier and resolve the reference.

MLP is the within-token operation: a small feedforward net that operates on each token's vector independently. Think of it as the "compute" step — making decisions based on the information collected so far.

Crucially, each block doesn't *replace* the input vector — it **adds** to it:

```
output = input + attention(input) + mlp(input + attention(input))
```

The `+ input` is the key. The "main pathway" of the vector flows through every block unchanged. Each block contributes a small additive update.

### II.3  The residual stream — the most important concept here

The thing that flows through the model — getting added to at every block — is the **residual stream**. Think of it as a bus:

```
   token enters as a 2304-dim vector
   │
   ├──── block 1 reads, computes Δ, writes Δ back
   │
   ├──── block 2 reads, computes Δ, writes Δ back
   │
   ├──── block 3 reads, computes Δ, writes Δ back
   │
   ...
```

At any point the value on the bus is *the original embedding plus all updates so far*. It's a running accumulation, never overwritten. This is the canonical interpretability tap point: the value at layer 16 is the model's **complete state of belief** about this token at this point in its computation, including what word it is, what grammatical role it plays, which earlier tokens it refers to, and any abstract concept it triggers.

Reference: Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits" — this is the paper that formalized the "residual stream as bus" view that the whole interpretability field now uses.

### II.4  Why layer 16 specifically

Empirical observation from the field. Different layers do different things:

| Layer range | What lives there |
|---|---|
| 1-4 | Token-level stuff: syntax, basic word identity, tokenization quirks |
| 5-12 | Local structure: phrases, simple grammar, coreference |
| 13-20 | Abstract concepts — the "semantic" sweet spot (cardiovascular condition, drug name, lab value) |
| 21-26 | Output preparation: features that directly influence next-token prediction |

For finding clinical concepts, the abstract-concept layers are best. Layer 16 of Gemma-2-2B is the conventional middle pick, borrowed from DeepMind's GemmaScope work which trained SAEs at every layer and found middle layers richest in semantic features.

This is one of the weakest defenses of any single choice in the pipeline. No layer sweep has been done. Honest position: layer 16 is a defensible starting point from prior work, not the result of an ablation in this work.

### II.5  Why we don't fine-tune Gemma on MIMIC

The scientific question is: *does Gemma, as released, contain clinical knowledge from internet training?*

If we fine-tuned Gemma on MIMIC first, we'd be answering a different question: "can Gemma learn clinical concepts when given clinical data?" The answer to that is trivially yes — any modern LM can. The harder, more interesting question is whether the off-the-shelf model already has it.

Practical consequence: Gemma was trained on a huge mix of internet text, which contains lots of medical content (UpToDate excerpts, PubMed abstracts, MedlinePlus, clinical forums, Wikipedia medicine articles). It's plausible — but not certain — that this exposure gave Gemma representations of common clinical concepts. The SAE evaluation is a test of that plausibility.

### II.6  How extraction actually runs (`extraction.py`)

For one note:

1. Tokenize: clinical note text → list of token IDs (max length 8192).
2. Run through Gemma with `output_hidden_states=True`: this asks HuggingFace to return the residual stream at every layer.
3. Take `hidden_states[17]` — the residual stream value after block 16, just before block 17 would read it. (Off-by-one: `hidden_states[0]` is the embedding output, so layer L's output sits at index L+1.)
4. Convert to fp16, move to CPU, save to a shard file.

Output per note: a tensor of shape `[n_tokens, 2304]`, where each row is Gemma's complete internal state about that token at that point in its computation.

### II.7  Why fp16 and "shards"

fp16 (16-bit floats) costs 2 bytes per number; fp32 costs 4. With 50,000 notes × ~500 tokens × 2304 dims:
- fp32: ~230 GB
- fp16: ~115 GB

fp16 has less precision (~3 significant digits vs ~7), but activations are noisy and the rounding error is much smaller than natural activation variance.

Sharding into ~300 files of ~400 MB each (instead of one 115 GB file) lets us stream chunks into RAM during SAE training, resume after crashes, parallelize work, and avoid memory-mapping disasters. `ShardedSafetensorsWriter` (`storage.py`) handles this: it batches activations as they're computed and flushes to a new shard every time the buffer hits a configured token count.

### II.8  Centering

After extraction, `center.py` does a two-pass mean subtraction in float64 (the saved `mean.pt` is float32). Why: SAEs reconstruct `x` as `x_hat = z @ W_dec + b_dec`. If the data has a large constant mean, the SAE will burn capacity learning that mean as a single dense direction instead of finding sparse meaningful features. Centering removes that pressure.

The two-pass version is in float64 because summing 150M float16 values in float32 accumulates ~1e-2 error per dim, which would shift the centered mean by enough to matter for sparsity dynamics. This matches standard practice from Anthropic's *Towards Monosemanticity* (Bricken et al. 2023).

### II.9  Defense anchors — Stage II

> **Q: What exactly are you tapping?**
> The residual stream value after Gemma's 16th transformer block — a 2304-dim vector per token. This is the model's complete internal state about that token at that point in its computation.

> **Q: Why does the residual stream contain "everything the model has computed so far"?**
> Because of the additive structure of transformer blocks: `output = input + Δ`. Each block adds a small update to the running state but never replaces it. So the value at layer 16 is the embedding plus the cumulative updates from the first 16 blocks. Elhage et al. 2021 is the canonical reference.

> **Q: Why layer 16?**
> Empirical convention from GemmaScope and prior interpretability work — middle layers hold abstract concepts. A layer sweep is on the to-do list; layer 16 is a defensible starting point but not the result of an ablation in this work.

> **Q: Why no fine-tuning?**
> Our scientific question is whether pretraining alone gave Gemma usable clinical representations. Fine-tuning would change the answer.

---

## Part III — The SAE itself

### III.1  The SAE as a translator

The SAE is a tiny neural network. Two pieces back-to-back:

```
   Gemma activation (2304 numbers, confusing soup)
              ↓
         ┌──────────┐
         │ ENCODER   │   linear layer + a "gate" function
         └──────────┘
              ↓
   SAE codes (18432 numbers, mostly zero)
              ↓
         ┌──────────┐
         │ DECODER   │   linear layer (no gate)
         └──────────┘
              ↓
   Reconstructed activation (2304 numbers, should ≈ the input)
```

The whole SAE is **encode-then-decode**. The interesting middle vector — 18432 sparse codes — is what we actually care about. That's our list of "feature presences." The decoder exists only to prove the encoder is doing its job (by showing the codes can rebuild the original).

So really: the SAE is a translator, and we only care about its translations, not the round-trip output.

### III.2  What the encoder does — one feature at a time

The encoder runs 18432 "feature detectors" in parallel. Each detector is a simple operation:

> Take the dot product of the input activation with my detector vector. If the result is large enough, output that number. If not, output zero.

In symbols:

```
for feature j = 1, 2, ..., 18432:
    score_j = (input activation) · (detector vector for feature j)
    if score_j > threshold:
        z_j = score_j        ← feature j "fires" with strength score_j
    else:
        z_j = 0              ← feature j stays silent
```

Each feature has its own detector vector (one row of the `W_enc` matrix) and its own threshold (the `b_enc` bias term, or for JumpReLU a learned per-feature `θ`). The encoder runs all 18432 of these simultaneously via a matrix multiply.

Intuition: each feature is like a clinical specialist focused on one concept. *Resident #14,283 specializes in sepsis.* When a token's activation vector "looks like" sepsis in some learned direction, the resident outputs a number; otherwise it stays quiet. After all 18432 residents have voted, you have a vector of 18432 vote magnitudes. Most are zero (most residents had nothing to say). About 20-80 of them are nonzero per token.

### III.3  What the decoder does

The decoder takes the 18432 codes and rebuilds the original 2304-dim Gemma activation:

> Each feature has a "direction vector" in the 2304-dim space. To reconstruct, add up all the direction vectors, each scaled by its code value.

```
reconstruction = z_1 · direction_1 + z_2 · direction_2 + ... + z_18432 · direction_18432
```

Most `z_j` are zero, so most terms drop out. Only the ~20-80 active features contribute. The reconstruction is a sparse linear combination of direction vectors.

If feature 14,283 ("sepsis") fires at magnitude 4.7 on a sepsis-mentioning token, the decoder adds `4.7 × sepsis_direction` to the reconstruction. Other active features add their contributions; the sum should land close to the original Gemma activation.

The direction vector for feature j is a row of the `W_dec` matrix.

**Important pairing.** `W_dec[j]` (decoder direction) and `W_enc[:, j]` (encoder detector) refer to the same feature j. They're a matched pair — the encoder's job is to detect "is this direction present?" and the decoder's job is to "write this direction back."

### III.4  The complete forward pass

In one block, what the vanilla SAE does (`sae_train.py:82-92`):

```
(1)  x_in  =  x − b_dec                       ∈ ℝ^d_in
(2)  π     =  x_in · W_enc  +  b_enc          ∈ ℝ^d_sae          ("pre-activations")
(3)  z     =  ReLU(π)  =  max(0, π)           ∈ ℝ^d_sae          ("codes")
(4)  x̂     =  z · W_dec  +  b_dec             ∈ ℝ^d_in           ("reconstruction")
```

Why each line:

- **(1) subtract `b_dec` before encoding.** Re-centering. The dataset is already globally centered, but `b_dec` can move slightly during training to absorb residual offset. By subtracting it before the encoder, the encoder operates on the truly centered signal. The same `b_dec` is added back in (4).
- **(2) linear encode.** Projection onto each feature's encoder direction.
- **(3) nonlinear gate.** ReLU enforces `z_j ≥ 0`. Non-negativity comes from the dictionary-learning convention ("feature presence is a non-negative magnitude") and removes a gauge ambiguity (we could otherwise encode the same feature as `+z·d` or `−z·(−d)`).
- **(4) linear decode.** Each row of `W_dec` is a feature direction. Reconstruction is `Σ_j z_j · W_dec[j]`.

### III.5  What "training the SAE" actually means

Training learns the four parameter sets: 18432 encoder detector vectors, 18432 decoder direction vectors, encoder biases, and the decoder bias. (For JumpReLU, also a per-feature threshold.)

The procedure:

```
1. Initialize all weights randomly.
2. Feed in millions of Gemma activations, one batch at a time.
3. For each batch:
     a. Encode → get sparse codes
     b. Decode → get reconstruction
     c. Measure how bad the reconstruction is
     d. Measure how non-sparse the codes are
     e. Combine into a single "loss" number
     f. Use backpropagation to compute weight gradients
     g. Adam optimizer step
4. Repeat for hundreds of thousands of batches.
```

No labels. No supervision. Just *"reconstruct using as few features as possible"* repeated many times. The bet is that the most efficient way to satisfy this is to discover the actual concepts the model uses internally — because real concepts are the natural sparse decomposition; anything else requires more pieces.

### III.6  The two competing goals

The training loss has two parts that fight each other:

**Goal A — Reconstruction.** Decoder output should match the original Gemma activation. Measured by mean squared error (MSE). To get good reconstruction the SAE wants to use lots of features with big values.

**Goal B — Sparsity.** Most codes should be zero. To be sparse the SAE wants to use few features with small values.

Perfect reconstruction is trivial (set every code huge). Perfect sparsity is trivial (set all codes to zero). The loss balances them:

```
loss  =  reconstruction_error   +   λ · sparsity_penalty
         ───────────────────       ───────────────────
         "fix the reconstruction"   "use fewer features"
                                    λ controls how strict
```

λ is a knob we set. Higher λ → more pressure to be sparse → fewer features fire, but reconstruction degrades. Lower λ → less pressure → more features fire, reconstruction improves.

The bet of the field: at some sweet spot of λ, the SAE is *forced* to find the actual concepts, because they're the only way to reconstruct well *and* be sparse.

In this repo λ is called `l1_coeff` (vanilla SAE) or `lambda_l0` (JumpReLU). The 50k-note config sets it to 10, calibrated from a smaller 2k-note run by looking for the sweet spot where mean active features per token (the L0 metric) lands in [20, 80].

### III.7  Why L1 produces sparsity (the math, briefly)

For a single feature `j` with the rest of the network frozen, let `g_j = ∂MSE/∂z_j`. The total gradient is:

```
∂L/∂z_j  =  g_j  +  λ · sign(z_j)            for z_j ≠ 0
∂L/∂z_j  ∈  g_j + λ · [−1, +1]               for z_j = 0   (subgradient)
```

Gradient descent sets `z_j` to a point where this is zero. For `z_j > 0`: `0 = g_j + λ → g_j = −λ` — the feature is "worth" exactly λ in MSE units. For `z_j = 0`: the optimum requires `0 ∈ g_j + λ·[−1,1]`, i.e., `|g_j| ≤ λ`. Every feature whose MSE signal `|g_j|` is below threshold λ gets shoved to zero. That's the source of sparsity.

This is the classical "soft thresholding" you've probably seen in Lasso — same math.

### III.8  The decoder unit-norm constraint

There's an obvious cheat the SAE could pull:

> *"What if I made all my direction vectors 100× bigger, and made all my code values 100× smaller? The reconstruction is identical (100 × 0.01 = 1), but the code values look way sparser because they're tiny!"*

This would let the SAE satisfy the sparsity penalty without doing real work. The "sparsity" would be a scaling artifact.

**Fix:** force every direction vector to have length exactly 1 (unit norm). Now the SAE can't cheat by scaling. Code magnitudes have to mean what they say — "feature j is present with strength `code_j`."

**Implementation** (`sae_train.py:94-97`): after every training step, divide each direction vector by its current length, snapping it back to length 1. Like keeping a ball on the surface of a sphere — every step you slide it around, then push it back to the surface.

This is one of those "this seems pedantic" constraints that's actually critical. Without it, the whole notion of "feature j fires with strength `code_j`" is meaningless.

**The gradient surgery** (`sae_train.py:99-113`) makes this efficient. Let `u = W_dec[j]` (unit vector) and `g = ∇_{W_dec[j]} L`. Decompose:

```
g  =  g_‖  +  g_⊥           where  g_‖ = (g · u) u   (parallel to u)
                                    g_⊥ = g − g_‖    (tangent to sphere)
```

Only `g_⊥` rotates the direction; `g_‖` tries to grow/shrink it (which renormalization immediately undoes). The fix: subtract `g_‖` before the Adam step, so Adam's first/second moment estimates only see the tangent component. In one line:

```
g  ←  g  −  (g · u) u
```

### III.9  Dead neurons

A "dead feature" is one that never fires across the whole dataset — its code value is always zero. It contributes nothing. The mechanism is a trap:

```
Feature j is dead when:
  for every input we see, the detector score for feature j is ≤ 0
  → ReLU clamps to zero
  → code_j = 0
  → feature j contributes nothing to reconstruction
  → reconstruction error doesn't depend on feature j
  → gradient with respect to feature j's weights is zero
  → feature j's weights never update
  → the situation never changes
```

Once dead, a feature stays dead. In a fresh SAE, 30-50% of features can be born dead due to bad random initialization.

**The rescue: dead-neuron resampling** (`sae_train.py:159-227`). Every 5000 training steps:

1. Check which features haven't fired recently.
2. Find the inputs the SAE is currently reconstructing worst.
3. Reassign each dead feature's direction to point at one of those problem inputs. *"If reconstruction is bad in this direction, give a dead feature a new home there — maybe it can help."*
4. Reset the optimizer's memory for that feature so old momentum doesn't immediately re-kill it.

Result: dead fraction drops from 30-50% to under 5%.

**JumpReLU doesn't need this** — its threshold mechanism automatically lowers thresholds on features that never fire. Self-correcting.

### III.10  ReLU's subtler problem: shrinkage

Even features that fire are mis-measured under ReLU + L1.

Consider a feature that should fire at magnitude 5.0 to reconstruct perfectly. Under L1 penalty, what does the SAE output? **Less than 5.0.** The sparsity penalty rewards smaller code values, so the SAE finds it cheaper to under-report (say, fire at 4.0 instead of 5.0) and accept slightly worse reconstruction.

The math is the equation we derived in III.7: at the optimum, `g_j = −λ`, not `g_j = 0`. The MSE optimum and SAE optimum differ by λ.

Two consequences:
1. *Magnitude analysis becomes unreliable.* Absolute scales are off; relative orders are still mostly preserved.
2. *Weak features get killed.* A feature that should fire at 0.6 gets shrunk to 0.4, then below activation threshold, then dead.

This is the shrinkage problem. Reference: [Addressing Feature Suppression in SAEs (LessWrong)](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes).

### III.11  JumpReLU — the principled fix

JumpReLU changes the gate. Instead of ReLU's "ramp from zero":

```
ReLU:       0 ────────╱
                     ╱
                    ╱

JumpReLU:   0 ──────│  jump
                    │
                    │╱
                    ╱
                   ╱
```

- **Below threshold θ:** feature is *exactly off*, code = 0.
- **Above threshold θ:** feature is *on at full pre-activation magnitude*, code = the detector score itself.

The sparsity penalty switches from L1 (penalize magnitude) to L0 (penalize count). Each active feature pays a flat cost regardless of magnitude. Once a feature is over its threshold, growing its magnitude is free from a sparsity perspective. No shrinkage.

In symbols:

```
π     = (x − b_dec) @ W_enc + b_enc           (pre-activations)
z     = π · H(π − θ)                          (JumpReLU gate; H = Heaviside step)
x_hat = z @ W_dec + b_dec
L     = MSE + λ_L0 · mean_batch( Σ_j H(π_j − θ_j) )
```

**The catch:** the gate is discontinuous (a jump). Standard gradient descent can't pass gradients through a discontinuity. The fix is a **straight-through estimator** (STE): in the forward pass use the true discontinuous gate, but in the backward pass pretend the gate is smooth.

There are two STEs in `jumprelu_sae.py`:

1. **For the reconstruction gradient through `π`:** treat the gate as a detached constant scalar. `∂z/∂π ≈ gate`. Gradient passes through `π` whenever the feature is active.
2. **For the threshold gradient:** rectangular kernel `K_ε` of half-width `ε/2` around the firing boundary. `∂L/∂θ_j ≈ −K_ε(π_j − θ_j) / ε`. Only features whose pre-activation is within `ε/2` of the current threshold receive threshold updates.

The two STEs together produce an equilibrium: MSE pulls `θ` down for useful features (raises `z_j`, lowers MSE), L0 pushes all `θ` up (lowers count of active features). A feature's threshold settles where these forces balance.

**Why no dead-neuron resampling for JumpReLU.** If a feature never fires, the L0 penalty's gradient on `θ_j` is `−K_ε(π_j − θ_j) / ε`. The kernel has support near `θ_j`. If `π_j` is within `ε/2` of `θ_j`, the threshold gets pushed down — making the feature easier to fire next time. The threshold acts as a self-correcting wake-up mechanism.

### III.12  Three diagnostic metrics

After training, we measure three things:

**L0 — average number of active features per token.**
```
L0 = (count of nonzero codes) / (count of tokens)
```
Target [20, 80]. Too low → over-sparse, reconstruction bad. Too high → not sparse enough, defeats the purpose. L0 is the sparsity diagnostic.

**EV — explained variance (reconstruction quality).**
```
EV = 1 − var(reconstruction error) / var(input)
```
Target > 0.85. EV = 1 means perfect reconstruction. This repo's best ReLU SAE achieved EV ≈ 0.896.

**Dead fraction — wasted capacity.**
```
Dead frac = (count of features that never fire) / (total features)
```
Target < 10%. With resampling, this repo's runs settle around 2-5%.

Together: L0 in range + high EV + low dead frac = healthy SAE.

### III.13  Defense anchors — Stage III

> **Q: What does the SAE encoder do?**
> Runs 18432 feature detectors in parallel. Each detector takes the dot product of the input with a learned direction; if the result exceeds a threshold (via ReLU or JumpReLU), outputs that score, otherwise zero. The result is a sparse code where each nonzero entry says "this feature is present at this strength."

> **Q: What's the training objective?**
> Reconstruction loss plus a sparsity penalty: minimize MSE between decoded output and original input, while keeping codes as sparse as possible. The coefficient λ controls the tradeoff. No labels — pure self-supervised reconstruction.

> **Q: Why are decoder rows constrained to unit norm?**
> To prevent a gauge cheat where the SAE scales directions up and codes down by the same factor — same reconstruction, fake sparsity. Unit-norm enforcement closes this loophole and makes code magnitudes meaningful.

> **Q: What are dead features and how do you handle them?**
> Features that never fire. They happen when a pre-activation is always negative and ReLU clamps to zero, after which no gradient flows. Vanilla SAEs use periodic resampling to assign dead features new directions toward poorly-reconstructed inputs. JumpReLU handles it automatically via learnable per-feature thresholds.

> **Q: What is the shrinkage problem?**
> L1 sparsity penalizes magnitude, so even features that should fire strongly get systematically under-reported. Active feature values are biased low. JumpReLU fixes this by switching to L0 (count) — once a feature is over its threshold, its magnitude is no longer penalized.

> **Q: How do you know your SAE is healthy?**
> Three metrics: L0 (target 20-80), EV (target > 0.85), and dead fraction (target < 10%). All three in range simultaneously.

> **Q: JumpReLU has zero gradient almost everywhere. How do you train it?**
> Straight-through estimators. For the pre-activation: treat the gate as a detached constant — gradient passes through `π` whenever the gate is open. For the threshold: pseudo-derivative via a rectangular kernel of bandwidth ε centered at `π_j = θ_j`. Only features near the firing boundary receive threshold updates.

---

## Part IV — From SAE codes to "this feature represents sepsis"

### IV.1  The mismatch we need to bridge

After running the SAE encoder on a clinical note, we have one matrix per note of shape `[n_tokens, 18432]`. A typical note has 500-3000 tokens.

The ICD-9 labels we want to compare against are **per-admission, not per-token**. Each admission has a binary indicator for each diagnosis code. There's no per-token labeling — we have no idea which token in the note "is" the sepsis token.

So we need to collapse the per-note matrix down to a single vector of shape `[18432]` per note. That's pooling.

### IV.2  Max pooling — what it captures

The default in this repo is max pooling:

```
note_vector[j]  =  max over all tokens t of (code[t][j])
```

In words: *"Did feature j ever fire strongly anywhere in this note?"*

If the sepsis feature fired at magnitude 4.7 on the token "septic" and 3.9 on "shock" and zero everywhere else, the note's value for that feature is 4.7 (the peak).

Why this is a reasonable default:
- ICD codes represent diagnoses, usually mentioned briefly in the note (the diagnosis word might appear once in a 2000-token note).
- Max preserves "strong signal anywhere" — even a single firing token is captured.
- Matches "is this concept present?" semantics rather than "how dominant is this concept?"

### IV.3  Mean pooling — the alternative

```
note_vector[j]  =  mean over all tokens t of (code[t][j])
```

In words: *"How much does feature j fire on average in this note?"*

For the sepsis example: 4.7 + 3.9 averaged over 2000 tokens with zeros elsewhere = 0.0043. Tiny.

Mean captures "dominance" — does this feature fire on most tokens? Right when the concept is diffusely present (e.g., "this note is about a critically ill patient"). Wrong for diagnosis-style concepts mentioned once.

This repo uses max because diagnoses are mention-based. That choice has a serious side effect.

### IV.4  The acuity confound — max pooling's failure mode

The most important caveat in the whole pipeline. Internalize it for defense.

In one sentence: **longer notes have more chances for any feature to fire strongly, and longer notes correlate with sicker patients.**

Mechanically:
1. A note has 3000 tokens. Each token has some chance of activating any given feature.
2. The max of 3000 noisy values is larger than the max of 500 noisy values, even from the same underlying distribution. (Basic property of order statistics — max of n samples grows with n.)
3. So `max_pool(feature_j)` has a baseline that grows with note length, regardless of clinical content.
4. Longer notes describe sicker patients (a patient with sepsis + AKI + heart failure + pneumonia generates more documentation than one with a single straightforward issue).
5. Therefore every feature's max correlates weakly with patient severity, just because severity correlates with length.

Downstream effect: when you correlate the sepsis feature with the sepsis label, you get a positive correlation. Good! But you *also* get positive correlations between the sepsis feature and the heart failure label, the AKI label, the pneumonia label, etc. — because all those labels are more common in long notes too.

This is the **acuity confound**. It's why the repo's results show **polyspecific** latents (one feature correlating with many codes). Some is genuine clinical co-occurrence (sepsis really does come with AKI). Some is artifact of max pooling on length-correlated data.

### IV.5  What "correlation" means

We have, for each note: one number (a feature's pooled value, e.g. `note_vector[14283]`) and one binary label (`has_sepsis: 0 or 1`). We want to ask: *"Does the feature value relate to the label?"*

**Pearson correlation** `r` is the standard measure:

```
r =  +1     →  perfect positive: as X goes up, Y goes up proportionally
r =   0     →  no linear relationship
r =  -1     →  perfect negative: as X goes up, Y goes down
```

Plain words: *"If you sort all data points by X, do the Y values tend to increase or decrease as you go?"*

Magnitude tells you strength:
- |r| > 0.7 — strong, very visible in a scatter plot
- 0.3 < |r| < 0.7 — moderate, visible but noisy
- 0.1 < |r| < 0.3 — weak, barely visible, often statistically detectable
- |r| < 0.1 — effectively no relationship for practical purposes

**This matters enormously for defense.** The repo finds plenty of latents with |r| > 0.1 (the default "grounded" threshold). But |r| = 0.15 is a *weak* relationship — overlaid histograms at r = 0.15 look nearly identical. With N=20,000 notes, the statistical test declares it significant (because sample size is huge) but the effect size is small. That's why we also do a threshold sweep at 0.3, 0.5 — to know how many features have meaningfully strong associations, not just statistically detectable ones.

### IV.6  Point-biserial correlation

X is continuous (the feature's pooled value); Y is binary (0 or 1 for the ICD code). When one variable is binary, Pearson correlation has a special form:

```
r_pb  =  (mean_X_when_Y=1  −  mean_X_when_Y=0)  ÷  std_X  ×  √(n_1 × n_0) / N
         ─────────────────────────────────────       ────────────────────────
         "how different are the two groups"          "scaled by class balance"
```

Plain words: *"How much higher is the mean feature value among patients with the label, compared to those without? Standardized by overall spread, weighted by class balance."*

Point-biserial r is mathematically **identical** to Pearson r when one variable is 0/1 — just a different name for the same number.

The repo computes this for every (feature, code) pair: 18432 features × ~50 codes = ~920,000 correlation tests, all at once via a vectorized matrix multiply (`compute_point_biserial_vectorised`, `icd_eval.py:944`).

### IV.7  The multiple testing problem

Run 920,000 tests at p < 0.05 on **pure noise**: expect 46,000 "significant" correlations by chance. Without correction, declaring "feature j is grounded to code k" at uncorrected p < 0.05 is meaningless.

Two main fixes:

**Bonferroni.** Divide your threshold by the number of tests: 0.05 / 920,000 ≈ 5.4 × 10⁻⁸. Very strict. Controls the probability of *any* false positive. Kills almost everything, including real but weak effects.

**Benjamini-Hochberg (BH) FDR.** Controls the *expected fraction* of false positives among declared findings. Less conservative, more practical for exploratory analysis. This repo uses BH.

### IV.8  BH-FDR — what it controls

FDR = False Discovery Rate. The conceptual distinction:

- Bonferroni controls: *"the probability any one of my findings is a false positive."* Very strict.
- BH-FDR controls: *"the expected fraction of my findings that are false positives."* More relaxed.

If you set FDR = 0.05 and declare 100 features grounded, you expect about 5 to be false positives. You don't know which 5, but you trust the bulk (95) is real.

The procedure (`apply_bh_correction`, `icd_eval.py:1002`):

1. Sort all p-values from smallest to largest.
2. For each p-value at position k (out of m total), check: is `p_k ≤ (k/m) × q`?
3. Find the largest k where this holds. Declare all p-values up to k significant.

After BH-FDR with q = 0.05 on this repo's 920k tests, typically a few thousand pass. Those are the "significant" associations.

### IV.9  The dependence caveat

BH-FDR assumes tests are independent or exhibit "positive regression dependence." Our tests are **not** independent — many features encode overlapping things and many codes co-occur (sepsis ↔ shock).

A more conservative variant — **Benjamini-Yekutieli** — handles arbitrary dependence by dividing the threshold by a log(m) factor (~14× here). This repo uses plain BH. Whether the dependence is "positive enough" for plain BH to hold is technically not proven, but plain BH is what everyone uses in genomics and most multiple-testing applications, and it generally gives reasonable results.

### IV.10  Partial correlation — controlling for the length confound

Recall from IV.4: max pooling causes every feature to weakly correlate with note length, and length correlates with patient severity. Partial correlation removes the linear effect of a confound (here `n_tokens`) before computing the main correlation.

Procedure:
1. For each feature j, run a linear regression: `feature_j = α + β × n_tokens + residual`.
2. Compute the residual (the part of `feature_j` *not* explained by note length).
3. Compute point-biserial between the residual and the ICD label.

In code (`compute_partial_point_biserial`, `icd_eval.py:1310`):

```python
Z = np.column_stack([np.ones(N), n_tokens])      # design matrix
beta, _, _, _ = np.linalg.lstsq(Z, X, rcond=None) # OLS fit
X_resid = X - Z @ beta                            # residuals
# now compute point-biserial r between X_resid and Y
```

What this **catches**: any feature whose correlation with a code is mediated linearly through note length.

What this **misses**: non-linear length effects, and other confounds (ward type, hospital service, year, attending preferences). The repo controls only for length.

### IV.11  "Grounded latent" — the final definition

A feature is **grounded** if all three hold:

1. At least one ICD code's (feature, code) correlation passes BH-FDR at q = 0.05 (statistical significance).
2. The absolute correlation `|r_pb|` for at least one such code exceeds the threshold (default 0.1, swept up to 0.5).
3. (Optionally, post-hoc): the same holds after controlling for note length.

Sub-classifications by how many codes a feature is associated with at a given threshold:
- **Monospecific** — exactly 1 code (clean: this feature ↔ this one concept)
- **Oligospecific** — 2 or 3 codes (plausibly clinically related)
- **Polyspecific** — 4+ codes (suspect: either acuity confound or genuine many-way overlap)

A good SAE result has many monospecific latents at high thresholds (r > 0.3). A suspicious result has many polyspecific latents at low thresholds that collapse at high thresholds — that's classic acuity-confound behavior.

### IV.12  The full evaluation pipeline

```
SAE codes per token, per note
        ↓ max-pool across tokens per note
note vectors [n_notes, 18432]
        ↓ join with ICD CSV on admission_id
matched note vectors + ICD label matrix [n_notes, ~50 codes]
        ↓ point-biserial correlation for all (feature, code) pairs
correlation matrix [18432, ~50]   +   p-value matrix [18432, ~50]
        ↓ BH-FDR correction at q = 0.05
significance mask [18432, ~50]
        ↓ filter by |r| > threshold (sweep 0.1, 0.2, 0.3, 0.4, 0.5)
"grounded" latents per threshold
        ↓ (post-hoc) residualize on n_tokens, repeat
partial-correlation grounded latents
        ↓ (post-hoc) count codes per latent
monospecific / oligospecific / polyspecific breakdown
```

Output files in `/out/icd_eval/<run_id>/` and `/out/icd_eval/<run_id>/posthoc/` are exactly these artifacts saved to disk.

### IV.13  Defense anchors — Stage IV

> **Q: Why do you pool tokens to notes?**
> Because our labels are per-admission, not per-token. We need to collapse the per-note matrix [n_tokens, 18432] to a single vector [18432] per note before correlating with the ICD label vector.

> **Q: Why max pooling specifically?**
> Diagnoses are mention-based — they appear in a few specific tokens of a note. Max preserves single-mention signals. Mean would dilute them into noise.

> **Q: What's the cost of max pooling?**
> Length confound. Max grows with token count, and longer notes correlate with patient severity. Every feature inherits a weak length-driven correlation with every common diagnosis. We address this with a threshold sweep and partial-correlation control for n_tokens.

> **Q: What does point-biserial measure?**
> The standardized mean difference of a continuous variable between two groups defined by a binary variable. Mathematically identical to Pearson correlation when one variable is 0/1. It answers "do feature values differ systematically between patients with and without this diagnosis?"

> **Q: Why is BH-FDR appropriate, and what does it leave out?**
> ~920k tests; uncorrected p<0.05 would yield ~46k false positives by chance. Bonferroni is too conservative. BH controls the expected fraction of false positives among findings, giving a usable list for follow-up. Caveat: BH assumes independence or positive regression dependence; our tests have nontrivial correlation structure. The fully-conservative fallback is Benjamini-Yekutieli, which would tighten the bound by a log(m) factor.

> **Q: What does "grounded latent" actually mean, and how strong is the claim?**
> Grounded = passes BH-FDR with |r_pb| > 0.1 for at least one ICD code. This is a permissive bar — it includes weak-but-detectable effects. The threshold sweep up to |r| > 0.5 stratifies by effect size; monospecificity analysis identifies clean one-feature-one-code associations. The strong claim is at high threshold + monospecific; the weak claim is at low threshold.

---

## Part V — Baselines as defense backbone

### V.1  Why baselines exist at all

Suppose you train an SAE and find 1,200 of the 18,432 features are "grounded" to ICD codes. Headline: *"Gemma's clinical knowledge is interpretable via SAE!"*

A skeptic asks: how do you know this is meaningful? Specifically:

- *"Maybe a random feature would correlate with ICD codes at that rate."*
- *"Maybe a keyword regex would do just as well."*
- *"Maybe TF-IDF + LR would predict ICD codes better than your fancy SAE."*
- *"Maybe Gemma's raw activations already linearly contain everything; the SAE is just a re-encoding."*
- *"Maybe DeepMind's pre-trained SAE on Gemma (trained on the open internet) would find the same features without any clinical training."*

Each "maybe" is a null hypothesis. Each baseline is a controlled experiment that says: *"if this null were true, here's what we'd see. Let's check."*

The four baselines in this repo each kill one of these "maybes" — or fail to, which is also informative.

The scientific structure of the work:

```
Claim:    "Custom-trained SAE on clinical activations finds meaningful clinical features."

For each baseline B:
  Null hypothesis: "What you're attributing to your SAE is actually due to B alone."
  Test:            Run baseline B on the same task, compare per-code AUC / r_pb.
  Outcome 1:       SAE >> B  →  null rejected, your SAE adds value.
  Outcome 2:       SAE ≈ B   →  cannot reject null; B explains the result; SAE adds nothing here.
  Outcome 3:       SAE < B   →  embarrassment; baseline is doing better than your method.
```

Each outcome is publishable. "SAE wins" is the headline; "SAE matches X" is a useful negative result.

### V.2  Baseline #1 — GemmaScope SAE (the "did training matter?" baseline)

**What it is.** GemmaScope is a suite of SAEs DeepMind released July 2024, trained on Gemma-2-2B at every layer on general internet text (no clinical filtering). You can download GemmaScope's layer-16 SAE off the shelf.

**What null it kills.** *"Maybe domain-specific training is unnecessary. Maybe a generic SAE on Gemma would find the same clinical features."*

If GemmaScope finds the same number of grounded latents at the same correlation strengths, your training added nothing — you could have used GemmaScope. If your MIMIC-trained SAE meaningfully exceeds GemmaScope, the domain-specific training is doing real work.

**How it's run.** `gemma_scope_eval.py` downloads `google/gemma-scope-2b-pt-res/layer_16/width_16k/average_l0_42/params.npz` from HuggingFace, then runs the same ICD evaluation pipeline using GemmaScope's SAE in place of yours. Output to `/out/icd_eval/gemma_scope_16k/`.

**The catch.** GemmaScope was trained on **raw, non-centered** Gemma activations. Your custom SAE was trained on **centered** activations. The repo correctly evaluates GemmaScope on raw activations (the `activations_dir` in `gemma_scope_eval.yaml` points to the non-centered directory) and uses `subtract_b_dec=False` in `JumpReLUSAE.from_huggingface` because GemmaScope's encoder convention differs from the custom SAE's.

### V.3  Baseline #2 — Lexical keyword regex (the "is it just keywords?" baseline)

**What it is.** For each ICD code, build a regex that matches obvious surface forms — for sepsis: `\b(sepsis|septic|bacteremia|septicaemia)\b`. For each note, check whether any keyword matches → binary indicator. Then compute point-biserial between **the lexical indicator** and the ICD label.

**What null it kills.** *"Maybe your SAE features are just glorified keyword detectors."*

If your SAE feature's correlation with sepsis is roughly equal to the lexical sepsis regex's correlation, your feature is doing nothing more than detecting the word "sepsis." Not interesting — a regex is interpretable in a different way. If your SAE feature's correlation substantially exceeds the lexical baseline, the feature is capturing more than surface lexical matching (synonyms, negation, context).

**Three-way classification** (`lexical_baseline.py`):

```
For each (feature, code) pair:
  If SAE_r >> lexical_r + delta:  →  "SAE wins"  (feature is more than keywords)
  If |SAE_r - lexical_r| < delta: →  "SAE ≈ lexical" (feature is just a keyword)
  If SAE_r << lexical_r - delta:  →  "SAE < lexical" (feature does worse than keywords)
```

**Limitation.** The lexical baseline is only as good as the keyword list. If you forget synonyms (septicemia, bacteremia, bloodstream infection), the lexical baseline looks artificially weak. The repo's `configs/lexical_keywords.yaml` was assembled by hand. UMLS-expanded synonym lists would tighten this comparison.

### V.4  Baseline #3 — TF-IDF + Logistic Regression (the "old-school NLP" baseline)

**What it is.** TF-IDF (term frequency – inverse document frequency) is the 1970s NLP standard for representing text. For each note: count how often each word/phrase appears (TF), down-weight common terms (IDF), get a sparse vector of ~10,000 weights. Then train per-code logistic regression on TF-IDF features. Evaluate per-code AUC-ROC and AUC-PR via cross-validation.

**What null it kills.** *"Maybe none of this neural network stuff is needed. Old-school sparse bag-of-words + linear classifier might predict ICD codes just as well."*

If TF-IDF + LR matches or beats your SAE on per-code AUC, the predictive task doesn't actually need the SAE. A 1970s pipeline works just as well. If your SAE meaningfully beats TF-IDF, the SAE features capture something beyond surface word frequency.

**Why this matters more than people think.** Clinical NLP has a strong tradition of TF-IDF + LR baselines because they're shockingly hard to beat. Many "deep learning beats baseline" papers turn out to have weak baselines. If your SAE doesn't beat TF-IDF, the headline becomes "neural features no better than 1970s NLP for ICD prediction."

### V.5  Baseline #4 — Raw activations + Logistic Regression (the killer baseline)

**This is the most important baseline. It's the current branch of the repo (`feat/baseline-3-raw-activation-probe`).**

**What it is.** Take the raw centered Gemma activations (the 2304-dim vectors *before* the SAE). Max-pool per note. Train per-code logistic regression directly on those 2304-dim vectors. Compare per-code AUC against the same protocol on the SAE's 18432-dim features.

**What null it kills.** *"Maybe Gemma's raw residual stream already linearly contains all the clinical information. The SAE just re-expresses it in a wider, sparser form — useful for interpretability per-feature, but not for prediction."*

This is the most fundamental version of "does the SAE actually do anything?" If a linear probe on raw 2304-dim activations matches a linear probe on 18432-dim SAE features, then:

- For **prediction**, the SAE is decorative. You could drop it and lose nothing.
- For **interpretability**, the SAE may still matter — a linear probe doesn't give one-feature-per-concept attribution; it gives opaque coefficients on dense dimensions. So the SAE's value would be "naming features," not "improving prediction."

**Outcomes:**

| Outcome | Interpretation |
|---|---|
| SAE >> Raw | SAE's expansion+sparsity adds genuine predictive structure beyond what a linear probe extracts. Strong defense. |
| SAE ≈ Raw | SAE is a re-encoding for interpretability only. Defense pivots to "we get human-readable features at the same predictive power." |
| SAE < Raw | SAE is losing information during encoding. Training failed or shrinkage is destroying signal. Bad. |

The config explicitly warns (`configs/raw_lr_baseline.yaml:8-9`): *"The 'pooling' field MUST match the icd_eval run that produced sae_results_csv — sae_cv_results.csv carries no provenance for the pooling strategy used to produce it, so a mismatch silently invalidates the comparison."*

### V.6  The decision tree your results imply

```
Does custom SAE beat GemmaScope?
├── YES → Domain SAE training is meaningful. Move on.
└── NO  → Defense pivots to "GemmaScope already does this; we validate it on clinical
          data" — still publishable but a different paper.

Does SAE beat lexical regex?
├── YES → Features capture beyond keywords. Move on.
└── NO  → Defense is weak; need to upgrade lexical baseline to UMLS-expanded
          synonyms before claiming the comparison.

Does SAE beat TF-IDF + LR?
├── YES → Features add value over bag-of-words NLP. Move on.
└── NO  → Headline shifts: "neural features no better than TF-IDF for ICD prediction" —
          interesting negative result, but the SAE-adds-classification-value claim dies.

Does SAE beat raw activations + LR?  ← THE BIG ONE
├── YES → SAE adds predictive value beyond a linear probe. Strong defense.
└── NO  → Defense pivots entirely to interpretability:
            "SAE matches raw activations' predictive power but gives us human-readable
             features per-concept rather than opaque coefficients."
          Respectable position but a different paper.
```

Walk in already knowing what each outcome means and what your fallback story is for each.

### V.7  Comparison to InterPLM's baseline lineup

The protein-SAE paper (Simon & Zou, Nature Methods 2025) used a different lineup because clinical text and protein sequences have different failure modes:

| InterPLM (proteins) | This repo (clinical) |
|---|---|
| **Random SAE** (same architecture, random init) | Implicit via lexical baseline |
| **Individual neurons** of ESM-2 (no SAE) | Raw activations + LR (more principled "no SAE") |
| **Cross-layer comparisons** | Not done; would require multi-layer extraction |
| **Smaller SAEs (lower expansion)** | Not done; could be added |
| n/a — proteins have no lexical analog | **Lexical regex baseline** |
| n/a — protein labels are clean curated annotations | **TF-IDF + LR** |

What's different: protein analyses didn't need lexical or TF-IDF baselines because protein sequences aren't text in the bag-of-words sense. But protein analyses did include random-SAE and bare-neuron baselines, which this repo doesn't quite replicate.

### V.8  Headline claims and honest hedges

**The strong claim:**

> *Gemma-2-2B, despite never being fine-tuned on clinical text, internally represents clinical concepts in a way that can be extracted via sparse autoencoding. We demonstrate this by: (a) training a JumpReLU SAE on Gemma layer-16 activations from MIMIC-IV notes; (b) showing that many SAE latents correlate with ICD-9 codes after BH-FDR correction at q=0.05; (c) showing this exceeds what is achievable with a generic pre-trained SAE (GemmaScope), keyword regex, TF-IDF+LR, and — critically — a linear probe on the raw activations themselves.*

**Honest hedges:**
- Correlation, not causation. We haven't shown ablating a "sepsis latent" changes Gemma's behavior on sepsis-related inputs.
- The acuity confound (max pooling × note length) is partially but not fully controlled.
- The lexical baseline depends on a hand-curated keyword list of uncertain coverage.
- We did not sweep layer choice or expansion factor.
- BH-FDR's dependence assumption is technically violated; results would be more conservative under Benjamini-Yekutieli.

**Bottom line:** if the SAE beats raw-LR, your claim is empirically grounded *up to confounders we name explicitly*. If the SAE matches raw-LR, your claim has to be repositioned to "interpretability without prediction gain" — still a real contribution, but a smaller one.

### V.9  Defense anchors — Stage V

> **Q: Why have baselines at all?**
> Without them, "our SAE found X grounded features" is meaningless. Baselines turn observations into evidence by ruling out alternative explanations.

> **Q: Walk me through each baseline and what it controls for.**
> GemmaScope (generic pre-trained SAE on Gemma) — controls for whether domain-specific training is needed. Lexical regex — controls for whether features are just keyword detectors. TF-IDF + LR — controls for whether old-school NLP would do as well. Raw activations + LR — controls for whether the SAE adds predictive value beyond a linear probe on the same activations.

> **Q: Which baseline matters most?**
> Raw activations + LR. The most direct test of "does the SAE add anything?" If the SAE matches raw-LR, our predictive-utility claim collapses and we fall back on interpretability claims.

> **Q: What happens if your SAE doesn't beat raw-LR?**
> We reposition the paper around interpretability rather than prediction. The SAE still provides one-feature-per-concept attribution that a linear probe cannot, even if predictive power is comparable. Legitimate but different contribution.

> **Q: Are these baselines exhaustive?**
> They cover the four most plausible alternative explanations. Visible gap: a random-SAE control (same architecture, random weights), which InterPLM used. We argue lexical regex subsumes random-SAE for clinical NLP, but it's a fair criticism that a random-SAE baseline would be more directly comparable to protein literature.

---

## Part VI — Defense playbook (15 anticipated attacks)

Memorize the top six. Mid-tier should be familiar. Bottom-tier can be handled on the fly with the meta-skills at the end.

### Top tier — almost guaranteed

#### Q1. *"Correlation isn't causation. Do these features actually drive Gemma's behavior?"*

**Core response:**
> This work establishes observational grounding — SAE latents correlate with structured clinical labels under multiple controls. Causal demonstration via feature ablation is the natural next step, following the Sparse Feature Circuits methodology (Marks et al. 2024). Explicitly out of scope here; what we've shown is necessary but not sufficient for a full mechanistic claim.

**If probed:** The full causal protocol would be: identify a sepsis-grounded latent, intervene by zeroing its activation during a forward pass on a sepsis-related input, and measure whether the model's downstream behavior degrades. Requires causal mediation infrastructure we don't yet have. Observational grounding is a prerequisite — there's no point doing causal ablation on a feature that doesn't even correlate with the concept. This work is the necessary first step.

**Concession:** Without intervention, the strongest claim is associative. Explicit in limitations.

#### Q2. *"What does |r| = 0.15 actually mean? That sounds weak."*

**Core response:**
> |r| = 0.15 is a weak effect size. That's why we report a threshold sweep up to |r| > 0.5 and stratify by monospecificity. The headline claim is at the high-threshold + monospecific subset, not at the |r| > 0.1 floor.

**If probed:** r = 0.15 means r² ≈ 2.25% variance explained — barely visible in a scatter plot. But for thousands of features simultaneously across dozens of codes, even modest effects accumulate. If random, they'd be unstable across thresholds; they're not. At r > 0.3 surviving latents are predominantly monospecific; at r > 0.5 even more so. Strong claim in the high-threshold band; weak claim is "many latents detect something."

**Concession:** If we only had the |r| > 0.1 number the result would be weak. The threshold sweep is what carries the result.

#### Q3. *"Max pooling × note length = acuity confound. Aren't your correlations just measuring patient severity?"*

**Core response:**
> Yes, the acuity confound is real, and we explicitly address it via partial correlation controlling for n_tokens. Results that survive partial correlation aren't purely length-driven. Post-hoc analysis is in `posthoc_summary.json`.

**If probed:** Mechanically, max-pooling means longer notes produce larger maxes for any feature, and longer notes correlate with sicker patients. We regress out the linear effect of n_tokens — this removes the first-order length confound. We don't claim it removes non-linear length effects, ward/service confounders, or admission-source confounders. Mainline results survive linear length adjustment, which is the dominant component.

**Concession:** Partial correlation handles linear length effect, not all possible confounders. A more complete analysis would adjust for ward, service, admission characteristics.

#### Q4. *"Why this specific layer? Why this expansion factor? Did you do a sweep?"*

**Core response:**
> Layer 16 follows GemmaScope and prior interpretability convention for middle-layer semantic features. Expansion factor 8 is the standard 4-32× range midpoint. Neither is the result of an exhaustive ablation in this work.

**If probed:** GemmaScope trained at every layer of Gemma-2-2B and reports middle layers most semantically rich; layer 16 is the standard pick. For expansion factor: 8× balances dictionary capacity against dead-feature fraction and compute. We did not run a full sweep; doing so is straightforward and on the roadmap. We used field-standard defaults but didn't validate them empirically for this specific domain.

**Concession:** Fair weakness. The right experiment is a layer sweep and an expansion sweep. We chose to invest in baselines instead.

#### Q5. *"ICD codes are billing artifacts, not clinical ground truth. How can you trust them?"*

**Core response:**
> ICD-9 codes are noisy proxies for clinical concepts, which lowers the ceiling on our correlations but doesn't invalidate the comparisons. All baselines face the same noisy labels — relative performance differences are still meaningful.

**If probed:** ICD codes are imperfect. A "sepsis" code may be applied loosely or missed entirely. This caps our maximum achievable r_pb — even a perfect sepsis-detector feature would correlate at less than 1.0. However every method we evaluate faces the same label noise. A more rigorous evaluation would use clinician-adjudicated labels — clear upgrade for future work.

**Concession:** Label quality is a real ceiling. Strongest version replicates the headline findings on clinician-adjudicated labels.

#### Q6. *"How is this different from InterPLM (Simon & Zou)?"*

**Core response:**
> InterPLM established the methodology — train SAE on a domain language model, correlate features with structured labels, baseline against bare neurons. We apply that methodology to clinical text, which has different failure modes than protein sequences: a length confound that doesn't exist for proteins, ICD label noise heavier than Swiss-Prot, and a need for text-specific baselines (lexical, TF-IDF) that proteins don't require. Contributions: (a) the clinical application, (b) the broader baseline lineup, (c) the polyspecificity analysis arising from pooling — novel because protein annotations are per-residue and pooling isn't needed.

**If probed:** InterPLM's three baselines were random-SAE, bare neurons, and cross-layer comparison. Our four are GemmaScope, lexical regex, TF-IDF+LR, raw-activation+LR. The raw-activation+LR baseline serves the same role as InterPLM's bare-neuron baseline but is more principled — a linear probe rather than per-neuron correlation. We lose the random-SAE baseline, which I'd add if doing the work over.

### Mid tier — likely

#### Q7. *"BH-FDR assumes independence. Your tests aren't independent."*

**Core response:**
> Plain BH-FDR holds under positive regression dependence, a reasonable assumption when most test-test correlations are positive — which they are in our setting. The fully-conservative fallback is Benjamini-Yekutieli, which tightens by a log(m) factor and would give a smaller list. The directionally strongest results would survive BY.

**Concession:** Should report BY-corrected results as a sensitivity analysis. Easy to add.

#### Q8. *"What's the actual clinical use case? Why does this matter?"*

**Core response:**
> Two horizons. Near-term: clinical NLP models are increasingly deployed for triage, summarization, and decision support; understanding their internal representations is foundational to auditing, debiasing, and certifying them. Long-term: if foundation models are going to be components of clinical systems, mechanistic interpretability is prerequisite for trust and regulatory acceptance. This work demonstrates the methodology on a tractable testbed.

#### Q9. *"What if your 'grounded' features are just keyword detectors?"*

**Core response:**
> Exactly what the lexical baseline tests. For each (feature, code) pair we compare the SAE's correlation against the keyword regex's correlation. Three classifications: SAE >> lexical (beyond keywords), SAE ≈ lexical (essentially keyword detector — uninteresting), SAE < lexical (underperforms keywords — bad). Headline results are SAE >> lexical features only.

**Concession:** Lexical baseline is only as strong as our keyword list. UMLS-expanded synonyms would tighten this.

#### Q10. *"Why Gemma-2-2B rather than a clinically-tuned model like MedGemma?"*

**Core response:**
> Scientific question: does pretraining on the open internet — which contains substantial medical text — induce clinical representations in a general-purpose model? Using a clinically-tuned model would conflate "what was in pretraining" with "what we added via fine-tuning." Running the same pipeline on MedGemma is the natural follow-up and would directly answer "does clinical tuning improve recoverable representations?"

#### Q11. *"Polyspecificity is a problem with your method, isn't it?"*

**Core response:**
> Polyspecificity at low thresholds is partly the acuity confound (length × severity), partly genuine clinical co-occurrence (sepsis ↔ shock), and partly imperfect monosemanticity. We disentangle these via threshold sweep and partial correlation. At |r| > 0.3 with partial correlation, the monospecific fraction increases — what we'd expect if the low-threshold polyspecificity is mostly confound-driven rather than fundamental.

**Concession:** Polyspecificity won't go to zero even with perfect controls — some clinical concepts genuinely co-occur. The right metric is whether polyspecificity *decreases* with stricter thresholds, which it does.

### Bottom tier — possible

#### Q12. *"How do you handle train/eval split? Any risk of data leakage?"*

**Core response:**
> SAE training uses the first ~90% of shards (~281 of 312); the last 31 shards are held out for early-stop EV monitoring. ICD evaluation runs on all shards including held-out — the SAE has never seen the eval activations during training. PHI is handled via the standard MIMIC credentialed-access protocol; data lives only on the gated Modal volume.

**If probed on split structure:** No patient-level leakage check in the current pipeline — same patient could appear in train and eval shards. Right fix is patient-stratified splitting, known improvement on the to-do list.

#### Q13. *"Did you replicate with different seeds?"*

**Core response:**
> Seed 42 throughout. We haven't run multi-seed replication; doing so would let us report mean ± std for headline metrics. Clear gap.

**Concession:** Single-seed results are not best practice. Multi-seed replication is the right next step.

#### Q14. *"Why max pooling and not [mean / topk / weighted attention]?"*

**Core response:**
> Max pooling matches the semantics of ICD diagnoses — they're mentioned at specific tokens, not diffusely across the note. Mean would dilute single mentions to nothing. We support `topk_mean` in the code (line 314 of `icd_eval.py`); it gives similar results to max in pilots. Right pooling depends on what the labels represent; we chose for the labels we have.

#### Q15. *"The Linear Representation Hypothesis isn't proven. What if features aren't linear?"*

**Core response:**
> You're right — the LRH is an empirically-supported hypothesis, not a theorem. If clinical concepts are encoded non-linearly in Gemma, SAEs will systematically miss them. The "features we find" are then a lower bound on what's there, not a complete inventory. Meaningful limitation but doesn't invalidate the features we do find — those still correlate with labels and survive baselines.

**Concession:** Deep conceptual limitation of the whole SAE program, not just our work. We inherit the assumption from the field.

### VI.16  The meta-skill: framing answers

Patterns that work across most attacks.

**The two-step concede-redirect.**
1. Acknowledge the legitimate part of the criticism (don't be defensive).
2. Explain why it doesn't undermine the specific claim you make.

Example: *"You're right that ICD codes are noisy. That's why we don't claim our absolute correlations represent the true SAE-concept alignment — we claim relative superiority over baselines that face the same noisy labels."*

**The "necessary but not sufficient" framing.**
Most criticism of mechanistic interp at this stage is about "this isn't the final word." Agree, and position your work as *necessary infrastructure* for the final word.

Example: *"This work establishes observational grounding. Causal demonstration is the natural next step but requires the observational result first."*

**Owning the to-do list.**
You will have unanswered questions. Owning them explicitly is much stronger than dodging.

**Knowing when to lose gracefully.**
Some criticism is correct. *"You're right, we should report Benjamini-Yekutieli as a sensitivity analysis. It's an easy fix and we'll include it."* Builds credibility for the points you do defend.

**Avoid:**
- Overclaiming. "We *prove*" — no, you provide evidence.
- Hiding weaknesses. The reviewer will find them; better that you raise them first.
- Conflating statistical significance with effect size.
- Dismissing baselines. Treat them as honest comparisons.

### VI.17  The 100-second elevator pitch

If asked "in one minute, what's the contribution?":

> *We adapt the InterPLM methodology — SAEs as a probe for foundation-model representations of structured concepts — to clinical text. Concretely: we train a sparse autoencoder on Gemma-2-2B's layer-16 activations from MIMIC-IV discharge summaries, and we evaluate whether the resulting latents correlate with ICD-9 diagnosis codes. We benchmark against a generic pre-trained SAE (GemmaScope), a lexical keyword baseline, TF-IDF+LR, and a linear probe on raw activations. The headline finding is that our domain-trained SAE produces clinically-grounded latents at rates above all four baselines. The work is observational, not causal, and we are explicit about the acuity confound, label noise, and limitations of the linear representation hypothesis.*

Practice until automatic. Every committee or reviewer interaction can start from there.

---

## Part VII — Appendices

### A.  Hyperparameter table

Numbers from `configs/sae_train_50k.yaml` and `configs/jumprelu_50k.yaml`.

| Hyperparam | Value (repo) | What it does mathematically | Failure if wrong |
|---|---|---|---|
| `layer` | 16 (of 26) | Where in Gemma we tap | Early: token features; late: next-token logits |
| `d_in` | 2304 | Gemma residual width | Fixed by model |
| `expansion_factor` | 8 → `d_sae` = 18432 | Dictionary capacity | Too small: superposed; too big: dead frac high |
| `l1_coeff` (vanilla) | 10 | Sparsity strength via L1 | Too high: features die; too low: L0 explodes |
| `lambda_l0` (JumpReLU) | 10 | Sparsity strength via L0 | Same direction; JumpReLU more robust |
| `bandwidth` (JumpReLU ε) | 0.5 | STE kernel half-width | Too narrow: no gradient; too wide: biased |
| `log_threshold_init` | 2.3 → θ₀ ≈ 10 | Initial gate cutoff | Too low: everything fires; too high: everything dies |
| `lr` | 2e-4 | Adam learning rate | Standard |
| `adam_beta1` | 0.0 | Adam first-moment momentum | Higher → more dying features (Anthropic finding) |
| `train_batch_size_tokens` | 4096 | Batch size | Smaller: noisier; bigger: OOM |
| `l1_warmup_steps` | 3000 | Linear ramp 0 → λ | Too short: mass feature death |
| `lr_warmup_steps` | 2000 | Linear ramp 0 → lr | Skipping → divergence |
| `resample_steps` | 5000 (vanilla only) | Dead-neuron rescue cadence | Too frequent: thrashing |
| `dead_feature_threshold` | 1e-6 | What counts as dead | Too high: live-but-rare features killed |
| `eval_n_shards` | 31 of 312 (~9.9%) | Held out for early-stop | Smaller: noisier EV |
| `early_stop_patience` | 3 evals | Allows transient EV dips | Too low: stop on noise |
| `pooling` | max | Token → note collapse | See acuity confound (IV.4) |
| `min_prevalence` | 0.02 | Code filter | Too low: noisy r_pb; too high: lose interesting codes |
| `max_codes` | 50 | Caps multiple-testing burden | Bigger: more BH stringency on real effects |
| `r_threshold` | 0.1 | "Grounded" threshold | Generous; sweep shows real picture |
| `fdr_q` | 0.05 | False discovery rate | Higher: more permissive |

### B.  Key files in the repo

| File | Role |
|---|---|
| `extraction.py` | Activation extraction; `extract_one_note` and `run_extraction` |
| `center.py` | Two-pass float64 mean subtraction |
| `sae_train.py` | Vanilla ReLU SAE training, dead-neuron resampling |
| `jumprelu_sae.py` | JumpReLU model + STE + training loop |
| `sae_data.py` | `ActivationsBuffer` — shuffled shard streaming |
| `icd_eval.py` | Full eval pipeline: encode → pool → correlate → FDR → grounding |
| `icd_eval_posthoc.py` | Threshold sweep + monospecificity + partial-correlation |
| `gemma_scope_eval.py` | GemmaScope baseline eval |
| `lexical_baseline.py` | Keyword regex baseline |
| `tfidf_lr_baseline.py` | TF-IDF + LR baseline |
| `raw_lr_baseline.py` | Raw-activation + LR baseline (current branch) |
| `configs/sae_train_50k.yaml` | Full 50k vanilla SAE training config |
| `configs/jumprelu_50k.yaml` | Full 50k JumpReLU training config |
| `configs/icd_eval.yaml` | Main eval config |
| `configs/raw_lr_baseline.yaml` | Raw-LR baseline config |

### C.  Glossary

- **Activation** — vector of numbers representing the model's internal state at a particular point in the computation.
- **Residual stream** — the running 2304-dim vector that flows through and is added to by every transformer block.
- **Polysemanticity** — the empirical observation that individual neurons fire for many unrelated reasons.
- **Superposition** — the hypothesis that the model packs more concepts than dimensions by giving each concept a non-orthogonal direction in activation space.
- **Sparse autoencoder (SAE)** — a neural network that encodes high-dimensional activations into a wider, sparser code, then decodes back. The codes are the "feature presences" we analyze.
- **Encoder direction** — `W_enc[:, j]`, the detector vector for feature j.
- **Decoder direction** — `W_dec[j]`, the direction in activation space that feature j writes back during reconstruction.
- **Unit-norm constraint** — every decoder direction has length exactly 1. Prevents the SAE from cheating on sparsity via scaling.
- **ReLU** — `max(0, x)`. The vanilla SAE's nonlinear gate.
- **JumpReLU** — `x · H(x − θ)`. Per-feature learnable threshold; zero below, full magnitude above.
- **L1 penalty** — sum of `|z_j|`. Vanilla sparsity penalty. Causes shrinkage of active features.
- **L0 penalty** — count of nonzero `z_j`. JumpReLU's sparsity penalty. Doesn't shrink magnitudes.
- **Dead feature** — a feature that never fires. ReLU SAEs need resampling to rescue them.
- **Shrinkage** — systematic under-reporting of active feature magnitudes under L1 penalty.
- **STE (straight-through estimator)** — a trick for training discontinuous functions: use the discontinuous gate in the forward pass, pretend it's smooth in the backward pass.
- **L0** (as a metric) — mean number of active features per token. Sparsity diagnostic.
- **EV** — explained variance, reconstruction quality diagnostic.
- **Dead fraction** — fraction of features that never fire. Capacity diagnostic.
- **Pooling** — collapsing per-token codes to a single per-note vector (here, max).
- **Point-biserial correlation** — Pearson correlation when one variable is binary (0/1). Identical math, special name.
- **BH-FDR (Benjamini-Hochberg false discovery rate)** — multiple-testing correction controlling the expected fraction of false positives among findings.
- **Partial correlation** — correlation after regressing out a confound (here, note length).
- **Grounded latent** — feature with at least one significant correlation passing |r| > threshold.
- **Monospecific / oligospecific / polyspecific** — features associated with 1, 2-3, or 4+ codes respectively.
- **Acuity confound** — the length × severity × max-pool interaction that inflates correlations between any feature and any common diagnosis.
- **Linear Representation Hypothesis (LRH)** — the conjecture that concepts in transformer activations are encoded as linear directions. SAEs assume this.

### D.  Key references

**Foundational mechanistic interp:**
- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* — https://transformer-circuits.pub/2021/framework/index.html
- Elhage et al. 2022, *Toy Models of Superposition* — https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al. 2023, *Towards Monosemanticity* — https://transformer-circuits.pub/2023/monosemantic-features/
- Templeton et al. 2024, *Scaling Monosemanticity* — https://transformer-circuits.pub/2024/scaling-monosemanticity/

**SAE methods:**
- Cunningham et al. 2023, *Sparse Autoencoders Find Highly Interpretable Features in LMs* — https://arxiv.org/abs/2309.08600
- Rajamanoharan et al. 2024, *Jumping Ahead: JumpReLU SAEs* — https://arxiv.org/abs/2407.14435
- Rajamanoharan et al. 2024, *Gated SAEs* — https://arxiv.org/abs/2404.16014
- Gao et al. 2024, *Scaling and Evaluating SAEs (OpenAI TopK)* — https://arxiv.org/abs/2406.04093
- Lieberum et al. 2024, *Gemma Scope* — https://arxiv.org/abs/2408.05147

**Domain applications and precedents:**
- Simon & Zou 2025, *InterPLM (proteins, Nature Methods)* — https://www.nature.com/articles/s41592-025-02836-7
- InterPLM GitHub — https://github.com/ElanaPearl/InterPLM

**Causal follow-on:**
- Marks et al. 2024, *Sparse Feature Circuits* — https://arxiv.org/abs/2403.19647

**Statistical machinery:**
- Tibshirani 1996, *Lasso* — https://www.jstor.org/stable/2346178
- Benjamini & Hochberg 1995, *FDR* — https://www.jstor.org/stable/2346101
- Benjamini & Yekutieli 2001, *FDR under dependence* — https://www.jstor.org/stable/2674075
- Hewitt & Liang 2019, *Designing and Interpreting Probes* — https://arxiv.org/abs/1909.03368

**Intuitive walkthroughs (recommended):**
- Adam Karvonen, *An Intuitive Explanation of SAEs* — https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
- *Addressing Feature Suppression in SAEs (LessWrong)* — https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes
- Neel Nanda, *Mechanistic Interpretability Glossary* — https://www.neelnanda.io/mechanistic-interpretability/glossary
- *Toy Models of Superposition: Simplified by Hand* — https://www.lesswrong.com/posts/8CJuugNkH5FSS9H2w/toy-models-of-superposition-simplified-by-hand

**Background data:**
- MIMIC-IV (PhysioNet) — https://physionet.org/content/mimiciv/
- Gemma-2 tech report — https://arxiv.org/abs/2408.00118

---

End of guide.
