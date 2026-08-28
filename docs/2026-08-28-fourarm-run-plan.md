# Four-Arm Concordance — Modal Run Plan

**Date:** 2026-08-28
**Branch:** `feat/four-arm-concordance-harness` (12 commits, 300 tests passing, unmerged)
**Spec:** `docs/superpowers/specs/2026-08-27-four-arm-concordance-validation-design.md`

---

## TL;DR

**Runnable today:** Stage 1 only — building 3–5 feature sources. ~2.5 h of CPU, no GPU.

**Blocked:** Stages 2–5. Tasks 9, 10 and 11 of the plan were never in scope, so the per-arm run configs, the assembly script and the forced-binary pass do not exist. Roughly a day of implementation stands between Stage 1 and any concordance verdict.

**Also blocked:** Arm A cannot enter the shared path yet — it exists on the volume in a different file format and needs a short conversion.

Nothing downstream produces a number until those gaps close, so **Stage 1 is worth running now only if you want the sources materialised and the underpowered-code count answered early.** That is a real reason — see Gate P0 — but it is not "starting the experiment."

---

## Prerequisites — human, before any compute

| # | Gate | Who | Time | Blocks |
|---|---|---|---|---|
| **P0** | **Eyeball the 46 core keywords.** Spec §4.2.1 calls this "the single highest-leverage manual check in this spec." The code now takes `keyword_dict[code][0]` per code; you need to confirm those 46 strings are the clinical terms you want. | you | ~20 min | all Arm B work |
| **P1** | **Sign off gates G1–G4** (spec §5.7) before any arm runs. Their whole value is being fixed in advance. | you | ~10 min | interpretation, not compute |
| **P2** | **Decide the Arm B code-description source** (spec §11 item 3) — keyword YAML (makes B1 tautological by design, which is arguably right for a liveness check) or official ICD-9 text (cleaner, but changes Arm C's published numbers and forces a re-run). | you | ~10 min | Stage 4 |

**P0 is the one that matters.** The offline proxy suggests several codes will come back *underpowered* once only the core term counts — `icd9_40390`'s core term is `hypertensive kidney`, roughly **1 hit in 41k notes**. Candidates to check by eye: `icd9_40390`, `icd9_41400`, `icd9_41401`, `icd9_V1582`, `icd9_5856`, `icd9_33829`. If a code's core term is near-absent, Arm B is simply undefined there and the positive control covers fewer than 46 codes.

---

## Stage 1 — Build feature sources ✅ RUNNABLE

CPU only. Every job writes `/out/sources/<id>/` in the pseudo-SAE format the rest of the pipeline reads.

### Ordering constraint — read this before launching

The **first** job to run computes `/out/sources/_shared/arm_c_selected_features.json` (Arm C's argmax-per-code on selection shards) and caches it. Every other job reads that cache and validates its provenance.

**Two jobs launched simultaneously will both try to compute and write it.** So:

```
1a  (alone)  ──►  then 1b, 1c, 1d in parallel
```

Run the cheapest job first so the cache lands quickly.

### Commands

```bash
# 1a — FIRST, ALONE. Populates the shared Arm C cache. ~15 min.
uv run modal run modal_app/build_feature_source.py \
    --config-file configs/source_diff_in_means.yaml

# Verify the cache landed before launching anything else:
modal volume ls sae-artifacts sources/_shared/

# 1b, 1c, 1d — now safe in parallel.

# 1b — diff-in-means V1, for the §4.4 variant comparison. ~15 min.
#      Edit variant: v1_plain and output_dir: /out/sources/dim_v1_plain first.
uv run modal run modal_app/build_feature_source.py \
    --config-file configs/source_diff_in_means_v1.yaml

# 1c — diff-in-means V3. ~15 min. Same edit, v3_diag_lda.
uv run modal run modal_app/build_feature_source.py \
    --config-file configs/source_diff_in_means_v3.yaml

# 1d — Arm B1, undiluted keyword. ~45 min (scans 24 shards, ~55 GB of reads).
uv run modal run --detach modal_app/build_feature_source.py \
    --config-file configs/source_keyword_b1.yaml --detach

# 1e — Arm B2, difficulty-matched. ~90 min (scans 24 shards TWICE, plus the alpha solve).
uv run modal run --detach modal_app/build_feature_source.py \
    --config-file configs/source_keyword_b2.yaml --detach
```

`--detach` goes **both** before the entrypoint and after the config for the long jobs, or the app is cancelled ~5 min in.

### What to check the moment each finishes

```bash
modal volume get sae-artifacts sources/<id>/source_meta.json .
```

| Field | What it tells you |
|---|---|
| `keyword_per_code` | The 46 strings actually used — this is P0's audit artifact |
| `underpowered_codes` | Codes below `min_token_positions: 200`. **Expect several.** |
| `note_level_measured_rate` vs the target | Whether calibration hit the SAE's per-code note rates |
| `token_level_measured_rate_reported_only` | Recorded for the record; not the calibration target |
| `dilution_achieved_r_post_threshold` (B2) | The residual against `dilution_target_r_selection` |
| `dilution_unreachable_codes` (B2) | Codes where the keyword direction is weaker than the SAE latent |

### Two things Stage 1 answers on its own

1. **How many codes Arm B actually covers.** If `underpowered_codes` is large, the positive control is thinner than the design assumes, and that changes what G1/G4 can claim.
2. **Whether diff-in-means whitening works.** V2's selection-set |r| against the measured random null of **0.1906** — if the whitened variant still sits below the null the way the unwhitened one does (0.1213), that is a result in itself.

---

## Stage 2–5 — ⛔ BLOCKED, needs implementation

| Stage | What it does | Blocked on | Est. build |
|---|---|---|---|
| **2** | `icd_eval` per source on audit shards 281–311 | `configs/fourarm/*_icd_eval.yaml` (Task 9) | ~1 h |
| **3** | `feature_inspector` — top-50 tokens, diversity score | `configs/fourarm/*_feature_inspector.yaml` | ~30 min |
| **4** | `auto_interp` — LLM explains + Sonnet concordance | `configs/fourarm/*_auto_interp.yaml`; needs `explicit_features` per arm | ~1 h |
| **5a** | `arm0_eval` ×3 judges | `configs/fourarm/*_arm0.yaml` | ~30 min |
| **5b** | `retrieval_eval` ×3 judges | `configs/fourarm/*_retrieval.yaml` | ~30 min |
| **5c** | Forced-binary pass over PARTIALs | `modal_app/forced_binary_eval.py` (Task 11) | ~2 h |
| **6** | Assemble table, apply gates G1–G4 | `scripts/assemble_fourarm_table.py` (Task 10) | ~2 h |

Tasks 9 and 10 are mostly config-writing plus one script; the plan already specifies both in full. **Roughly one day of work to unblock the whole downstream.**

### Two spec commitments that are NOT delivered

Recorded in the spec as implementation gaps on 2026-08-28:

- **§4.4 variant selection is unimplemented.** Nothing scores V1/V2/V3 on the selection set. That is why 1b and 1c exist above — you compare them by hand and record all three numbers.
- **§5.6 shipped `explicit_features: list[int]`, not `list[tuple[int, str]]`.** So auto-interp gets *argmax* pairing only. Spec §5.1 requires **by-construction** pairing for gates G1 and G4. Whoever builds Task 9 must widen that parameter or supply the pairing another way, or the two control gates measure the wrong thing.

---

## Arm A — done, but not yet plumbed

`/out/necessity/random_matched/seed0/` is complete and its grounding numbers are in hand:

| Variant | note density | max \|r\| | median \|r\| (audit) |
|---|---|---|---|
| dense | 1.0000 | 0.4307 | 0.2186 |
| token-matched L0 47.57 | 0.9590 | 0.3143 | 0.1481 |
| token-matched L0 40.92 | 0.9504 | 0.3143 | 0.1491 |
| **note-matched** ← report this | 0.6565 | **0.4322** | **0.1906** |

**But it has no concordance verdicts, and the entire experiment is about the verdict stage.**

To enter the shared path it needs converting from its native layout (`directions.npy` + `thresholds_note_matched.npy`) into the pseudo-SAE contract. That is a ~30-line script calling `write_pseudo_sae(directions, thresholds_note_matched, out_dir, meta)`. **Small job, high value — without it the floor has no concordance number and gates G2/G3 cannot be evaluated at all.**

One caveat to record when reporting it: Arm A's `note_matched` variant was calibrated against the **vanilla** SAE's note-level densities (its `sae_shard_ckpt_dir` points at `sae_d2304_e8_l11e+01_...`), while the headline arm is JumpReLU. Our own configs correctly point at the JumpReLU `shard_ckpt`.

---

## Cost summary

| Stage | Compute | Wall clock | API |
|---|---|---|---|
| 1a–1c (diff-in-means ×3) | CPU | ~45 min total | — |
| 1d (Arm B1) | CPU | ~45 min | — |
| 1e (Arm B2) | CPU | ~90 min | — |
| Arm A conversion | local | minutes | — |
| Stages 2–3 × 5 sources | CPU | ~2 h | — |
| Stage 4 × 5 sources | CPU | ~1 h | ~920 calls, **≤3 workers** (tier-1, 50 RPM) |
| Stage 5a/5b × 5 sources | CPU | ~1 h | 46 × 3 judges × 5, OpenRouter |

**No GPU anywhere.** Total once unblocked: roughly 6–7 CPU-hours plus modest API spend.

---

## Recommended decision

**Do P0 first — the 20-minute keyword eyeball — before spending any compute.** It gates every Arm B job, and if the core terms turn out to be wrong or mostly absent, Stage 1's B arms produce artifacts you would throw away.

Then either:

- **(A) Run Stage 1 now** to materialise the sources, get the underpowered-code count, and settle the diff-in-means variant question — accepting that no concordance number arrives until Tasks 9–11 are built. ~2.5 h CPU.
- **(B) Build Tasks 9–11 and the Arm A converter first** (~1 day), then run everything through in one sweep with the full table at the end.

**(B) is the better use of the compute**, because Stage 1's artifacts are inputs, not results, and re-running them later is cheap. **(A) is worth it anyway if the underpowered-code count would change the design** — and given `icd_9_40390`'s core term appears roughly once in 41k notes, it plausibly might.
