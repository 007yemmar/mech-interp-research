# Four-Arm Concordance — Minimum Path to LLM-Explainer Results

**Goal:** concordance verdicts (explanation + YES/PARTIAL/NO) for **every** arm, at the least compute that gets there.
**Date:** 2026-08-28 · supersedes the cost/sequence sections of `2026-08-28-fourarm-run-plan.md`

---

## What changed since the last plan

Two discoveries make this much cheaper than the earlier estimate:

1. **Arm A's evaluation stage is already complete.** `/out/necessity/random_matched/seed0/` holds `shard_ckpt_audit/` (all 31 audit shards, standard `shard_NNNN_vectors.npy` + `_meta.jsonl`) and `audit_note_matched/` containing `top_associations.csv`, `correlation_matrices.npz`, `code_names.json`, `grounded_latents.csv`, `per_code_summary.csv` — i.e. everything `icd_eval` produces and exactly what `feature_inspector` reads as its `eval_output_dir`. **Arm A needs no encode and no eval. Only a checkpoint file.**
2. **The collaborator's `necessity_audit.audit()` is the source-agnostic evaluator** we would otherwise have to write. It runs grounding + BH-FDR + one-feature-per-code selection on any `[n_notes × k]` matrix, which is precisely Arm C's matched-protocol problem.

**Consequence: Task 9's `icd_eval` configs are largely unnecessary.** The minimum path needs the *explainer* half of the pipeline only.

---

## What is required — the complete list

### A. Code to write (~half a day, no compute)

| # | Item | Size | Why it is required |
|---|---|---|---|
| **W1** | **Arm A → pseudo-SAE checkpoint.** Script reading `directions.npy` + `thresholds_note_matched.npy`, calling the existing `write_pseudo_sae(...)`. | ~30 lines | `feature_inspector` and `auto_interp` must *encode tokens* to find top-activating contexts. Arm A's pooled vectors exist; its encoder does not. **Without this the floor has no concordance number and gates G2/G3 cannot be evaluated at all.** |
| **W2** | **Merge `origin/feat/necessity-audit-harness`** into the working branch. | merge | Supplies `necessity_audit.audit()` for W3, and is the source of Arm A's artifacts anyway. |
| **W3** | **Arm C matched-protocol eval.** Call `necessity_audit.audit()` on Arm C's existing `shard_ckpt/`, restricted to audit shards, selecting on shards < 281. | ~40 lines | Arm C is the bridge to the published numbers. Its current eval is full-corpus and selection-biased; a matched one is required or no comparison is like-for-like. **No re-encode** — its 312 shards are already pooled. |
| **W4** | **Widen `explicit_features` to `list[tuple[int, str]]`.** | ~15 lines + test | Spec §5.1 requires **by-construction** pairing for gates G1 and G4. Today auto-interp pairs by `argmax\|r\|`, so a keyword direction can be judged against the wrong code. **Without this the control gates measure the wrong thing.** |
| **W5** | **Per-arm configs:** `feature_inspector` × 5 and `auto_interp` × 5. | 10 YAML | Copy-edits of `configs/feature_inspector_jumprelu.yaml` and `configs/auto_interp_jumprelu.yaml`. |

### B. Not required for this goal — explicitly cut

| Cut | Why it is safe to cut |
|---|---|
| `arm0_eval` × 3 judges (S5) | `auto_interp` already emits a Sonnet concordance verdict. The 3-judge panel is a *robustness* layer on top of a verdict you will already have. |
| `retrieval_eval` × 3 judges (S6) | Same — blind retrieval is a second measurement of the same features. |
| Forced-binary pass (Task 11) | Answers "would conclusions change without a PARTIAL option." Needs the verdicts to exist first. |
| Assembly script (Task 10) | Five `concordance_summary.json` files can be read directly. Automate once the numbers are worth automating. |
| Diff-in-means V1 / V3 builds | Only the chosen variant goes downstream. Build them later if the §4.4 variant question needs settling; V2 is the spec's recommendation. |
| `icd_eval` configs for Arm A and C | A's is done; C's comes from `necessity_audit.audit()`. |

---

## The minimum sequence

### Phase 0 — human gate, before any compute

**Eyeball the 46 core keywords** (spec §4.2.1). ~20 min. Gates every Arm B job. Expect several codes to come back underpowered now that only the core term counts — `icd9_40390`'s core term is `hypertensive kidney`, roughly one hit in 41k notes.

### Phase 1 — build sources ⟨CPU, ~105 min wall clock⟩

```bash
# 1a — ALONE FIRST. Populates /out/sources/_shared/arm_c_selected_features.json.
#      Two jobs launched together will race on that write.
uv run modal run modal_app/build_feature_source.py \
    --config-file configs/source_diff_in_means.yaml            # ~15 min

modal volume ls sae-artifacts sources/_shared/                 # verify cache landed

# 1b + 1c — now safe in parallel.
uv run modal run --detach modal_app/build_feature_source.py \
    --config-file configs/source_keyword_b1.yaml --detach      # ~45 min

uv run modal run --detach modal_app/build_feature_source.py \
    --config-file configs/source_keyword_b2.yaml --detach      # ~90 min
```

Arm A needs no build — run **W1** locally instead (seconds).

**Check `source_meta.json` on each:** `keyword_per_code` (the P0 audit artifact), `underpowered_codes`, `note_level_measured_rate` vs target, and for B2 `dilution_achieved_r_post_threshold` and `dilution_unreachable_codes`.

### Phase 2 — evaluate ⟨CPU, ~20 min⟩

| Arm | Action | Cost |
|---|---|---|
| A | **none** — `audit_note_matched/` is already the eval dir | 0 |
| C | W3: `necessity_audit.audit()` on existing `shard_ckpt/` | ~5 min |
| B1, B2, D | `icd_eval` on audit shards 281–311, k=46 | ~5 min each |

All four are parallelisable.

### Phase 3 — token contexts ⟨CPU, ~80 min⟩

`feature_inspector` per arm, `n_pairs: 46`, `n_shards: 20` (of the 31 audit shards).

| Arm | k | Cost | Parallel? |
|---|---|---|---|
| A | 18,432 | ~33 min | yes |
| C | 18,432 | ~33 min | yes |
| B1, B2, D | 46 | ~5 min each | yes |

All five run concurrently — wall clock ≈ the slowest, ~35 min.

### Phase 4 — explanations + concordance ⟨API-bound, ~2 h⟩ ← **the deliverable**

`auto_interp` per arm with `explicit_features` = the 46 (feature_id, code_name) pairs.

- **230 features total** (46 × 5 arms)
- ~4 API calls each ⇒ **~920 calls**
- **`max_workers: 3`** — the org is tier-1 at 50 RPM; more will 429-storm and silently drop features
- Emits per arm: `feature_catalog.csv`, `concordance_results.csv`, `concordance_summary.json`

**Run arms sequentially at 3 workers**, not in parallel — the rate limit is per-org, so parallel arms contend for the same budget and gain nothing.

---

## Total

| Phase | Compute | Wall clock |
|---|---|---|
| 1 — builds | CPU | ~105 min |
| 2 — eval | CPU | ~20 min |
| 3 — contexts | CPU | ~35 min (parallel) |
| 4 — explain + judge | API | ~2 h |
| | | **≈ 4.5 h, no GPU** |

Plus ~half a day of code (W1–W5).

Against the earlier plan's 6–7 CPU-hours plus a full day of build, this is roughly **half the compute and half the implementation**, because Arm A's eval turned out to be done and the multi-judge layers are genuinely optional for this goal.

---

## What you get at the end

Five `concordance_summary.json` files — A, B1, B2, C, D — each with YES / PARTIAL / NO counts over 46 features on the same 4,911 held-out notes, plus per-arm `feature_catalog.csv` with the explanations themselves.

That is enough to evaluate three of the four pre-registered gates:

| Gate | Evaluable? | Needs |
|---|---|---|
| **G1** liveness — B1 YES > 0.85 | ✅ | B1 |
| **G2** specificity — A YES < 0.15 | ✅ | A |
| **G3** C − A > 0.25 | ✅ | A + C |
| **G4** B2 ≥ (A+B1)/2 | ✅ | A + B1 + B2 |

All four, in fact — **provided W4 lands**, since G1 and G4 depend on by-construction pairing.

What you will *not* have: multi-judge agreement, blind-retrieval hit@1, the forced-binary robustness check, and the assembled comparison table. All four are additive layers over verdicts you will already hold, and each can be run later against unchanged inputs.

---

## Recommended order

1. **P0 keyword eyeball** — 20 min, gates everything Arm B.
2. **W1 + W2** — Arm A converter and the necessity-audit merge. This alone makes the floor measurable, which is the single highest-value unblock in the plan.
3. **Phase 1 builds** while writing **W3–W5**.
4. **Phases 2–4.**

If you want one thing first: **W1**. Arm A is complete except for a checkpoint file, and without it gates G2 and G3 — the two that address the published prior-art threat directly — cannot be evaluated at all.
