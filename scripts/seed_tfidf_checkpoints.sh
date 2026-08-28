#!/usr/bin/env bash
# Seed per-code TF-IDF CV checkpoints for JumpReLU and GemmaScope eval runs
# by copying from Vanilla SAE's completed TF-IDF baseline output.
#
# Rationale: TF-IDF features depend only on note text, not the SAE under evaluation.
# When the matched-notes set is identical across runs (verified: code_names.json
# and n_positive agree), the TF-IDF per-code CV results are identical across runs.
# Seeding JR/GS checkpoint dirs with Vanilla's cached results lets the TF-IDF
# baseline pipeline skip the TF-IDF arm entirely and only compute the per-model
# SAE arm — saving ~30 minutes of compute per run.
#
# Trade-off: invalidates the cross-run TF-IDF AUC diff sanity check (it would
# trivially pass because we copied). The matched-notes parity check
# (code_names.json + sample n_positive) replaces it as the integrity guarantee.
#
# Implementation note: the sae-artifacts volume is V1, so `modal volume cp -r`
# is not supported. Instead we download vanilla's 46 checkpoint files locally,
# then `modal volume put -f` each one to the JR + GS destinations.
#
# Run from repo root: bash scripts/seed_tfidf_checkpoints.sh

set -euo pipefail

VOLUME="sae-artifacts"
VANILLA_RUN="sae_d2304_e8_l11e+01_20260505T205723Z"
JUMPRELU_RUN="jumprelu_d2304_e8_l01e+01_bw1e+00_20260519T084742Z"
GEMMASCOPE_RUN="gemma_scope_16k"

SRC="icd_eval/${VANILLA_RUN}/posthoc/tfidf_lr_baseline/cv_ckpt_tfidf"
JR_DST="icd_eval/${JUMPRELU_RUN}/posthoc/tfidf_lr_baseline/cv_ckpt_tfidf"
GS_DST="icd_eval/${GEMMASCOPE_RUN}/posthoc/tfidf_lr_baseline/cv_ckpt_tfidf"

LOCAL_STAGE="/tmp/seed_tfidf_checkpoints"

echo "Source:  ${VOLUME}:${SRC}"
echo "Targets:"
echo "  ${VOLUME}:${JR_DST}"
echo "  ${VOLUME}:${GS_DST}"
echo

echo "Step 1/4: Download vanilla's 46 TF-IDF per-code checkpoints to ${LOCAL_STAGE}/"
rm -rf "${LOCAL_STAGE}"
mkdir -p "${LOCAL_STAGE}"
uv run modal volume get sae-artifacts "${SRC}/" "${LOCAL_STAGE}" --force >/dev/null 2>&1
LOCAL_DIR="${LOCAL_STAGE}/cv_ckpt_tfidf"
LOCAL_COUNT=$(ls "${LOCAL_DIR}" | wc -l | tr -d ' ')
if [ "${LOCAL_COUNT}" != "46" ]; then
  echo "ERROR: downloaded ${LOCAL_COUNT} files, expected 46." >&2
  exit 1
fi
echo "  downloaded 46 files ✓"
echo

upload_all() {
  local dst_dir="$1"
  local label="$2"
  local count=0
  for f in "${LOCAL_DIR}"/*.json; do
    local fname
    fname=$(basename "${f}")
    uv run modal volume put -f "${VOLUME}" "${f}" "${dst_dir}/${fname}" >/dev/null 2>&1
    count=$((count + 1))
    if [ $((count % 10)) -eq 0 ]; then
      echo "  ${label}: ${count}/46"
    fi
  done
  echo "  ${label}: ${count}/46 ✓"
}

echo "Step 2/4: Upload 46 checkpoints to JumpReLU ${JR_DST}"
upload_all "${JR_DST}" "JR"
JR_COUNT=$(uv run modal volume ls "${VOLUME}" "${JR_DST}/" | wc -l | tr -d ' ')
if [ "${JR_COUNT}" != "46" ]; then
  echo "ERROR: JumpReLU dst has ${JR_COUNT} files after upload, expected 46." >&2
  exit 1
fi
echo "  JumpReLU dst has 46 files on volume ✓"
echo

echo "Step 3/4: Upload 46 checkpoints to GemmaScope ${GS_DST}"
upload_all "${GS_DST}" "GS"
GS_COUNT=$(uv run modal volume ls "${VOLUME}" "${GS_DST}/" | wc -l | tr -d ' ')
if [ "${GS_COUNT}" != "46" ]; then
  echo "ERROR: GemmaScope dst has ${GS_COUNT} files after upload, expected 46." >&2
  exit 1
fi
echo "  GemmaScope dst has 46 files on volume ✓"
echo

echo "Step 4/4: Cleanup local stage"
rm -rf "${LOCAL_STAGE}"
echo "  removed ${LOCAL_STAGE}"
echo

echo "Done. Next: re-dispatch JR + GS tfidf_lr_baseline runs;"
echo "the library will load cv_ckpt_tfidf/ on entry and skip the TF-IDF arm,"
echo "running only the per-model SAE arm."
