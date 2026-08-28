"""Unit tests for the torch-free helpers behind #5 (section-local loss).

build_section_mask and real_prediction_mask are pure (numpy in/out), so they
test without a GPU. The torch pieces (decode_with_mean_ablation, the section CE
per pass) are validated by the Modal smoke runs — see ablation_smoke_*.yaml.
"""

from __future__ import annotations

import re

import numpy as np

from mech_interp_research.ablation import build_section_mask, real_prediction_mask

HEADERS = [r"discharge diagnos[ie]s", r"final diagnos[ie]s", r"discharge diagnosis"]


def _synthetic_tokenization(text: str):
    """BOS token + whitespace tokens, each with (char_start, char_end) offsets."""
    toks = ["<bos>"] + [m.group() for m in re.finditer(r"\S+", text)]
    offs = [(0, 0)] + [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return toks, offs


def test_section_mask_marks_diagnosis_span_only():
    text = "hpi note . Discharge Diagnosis : sepsis and AKI . exam normal . Final Diagnosis : pneumonia ."
    toks, offs = _synthetic_tokenization(text)
    n_real = seq_len = len(toks)

    def pos_predicting(word: str) -> int:  # position t predicts token t+1
        return toks.index(word) - 1

    mask = build_section_mask(text, offs, n_real, seq_len, HEADERS)
    assert mask.any()
    # inside the discharge-diagnosis section (up to the next header)
    assert mask[pos_predicting("sepsis")]
    assert mask[pos_predicting("AKI")]
    # outside: before the section, and after the next header bounds it
    assert not mask[pos_predicting("hpi")]
    assert not mask[pos_predicting("pneumonia")]


def test_section_and_rest_partition_real_positions():
    text = "a . Discharge Diagnosis : sepsis . b c d"
    toks, offs = _synthetic_tokenization(text)
    n_real = seq_len = len(toks)
    section = build_section_mask(text, offs, n_real, seq_len, HEADERS)
    real = real_prediction_mask(n_real, seq_len)
    rest = real & ~section
    assert section.dtype == bool and rest.dtype == bool
    assert (section & rest).sum() == 0  # disjoint
    assert np.array_equal(section | rest, real)  # cover exactly the real positions
    assert section.sum() + rest.sum() == real.sum()
    # section is a subset of real prediction positions
    assert np.array_equal(section & real, section)


def test_no_header_returns_empty_mask():
    text = "no relevant section headers appear in this note at all"
    toks, offs = _synthetic_tokenization(text)
    n_real = seq_len = len(toks)
    mask = build_section_mask(text, offs, n_real, seq_len, HEADERS)
    assert not mask.any()


def test_special_tokens_and_short_notes_are_safe():
    # all-special / too-short inputs must not raise and yield empty masks
    assert not build_section_mask("x", [(0, 0)], 1, 1, HEADERS).any()
    assert real_prediction_mask(1, 1).sum() == 0
    assert real_prediction_mask(4, 4).sum() == 3  # positions 0,1,2 predict tokens 1,2,3
