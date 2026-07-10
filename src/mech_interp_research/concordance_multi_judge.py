"""Multi-judge discriminative concordance for the JumpReLU auto-interp run.

Re-validates ICD concordance with a cross-provider judge panel, a top-5
retrieval arm, a shuffled null, a human-adjudication sheet, and an independent
second explainer. Reuses a completed auto_interp run; no GPU.
"""

from __future__ import annotations

import logging
import string

import numpy as np

logger = logging.getLogger(__name__)


class Judge:
    """Uniform ``.complete(prompt) -> str`` over Anthropic and OpenRouter backends."""

    def __init__(self, slug, backend, model=None, client=None, max_retries=6):
        self.slug = slug
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self._client = client

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        if self.backend == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        if self.backend == "openrouter":
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        raise ValueError(f"unknown backend {self.backend!r}")


def build_judges(judge_cfgs, *, anthropic_client=None, openrouter_client=None):
    """Build callable Judges, skipping ``backend == 'reuse'`` entries."""
    judges = []
    for cfg in judge_cfgs:
        backend = cfg["backend"]
        if backend == "reuse":
            continue
        client = anthropic_client if backend == "anthropic" else openrouter_client
        judges.append(Judge(cfg["slug"], backend, model=cfg.get("model"), client=client))
    return judges


_DESC_FALLBACK = {"V4986": "Do not resuscitate status"}


def _bare(code: str) -> str:
    return code.replace("icd9_", "")


def _describe(code: str, code_descriptions: dict) -> str:
    bare = _bare(code)
    if bare in code_descriptions:
        return code_descriptions[bare]
    if code in code_descriptions:
        return code_descriptions[code]
    if bare in _DESC_FALLBACK:
        return _DESC_FALLBACK[bare]
    logger.warning("No description for code %s; using bare code", code)
    return bare


def build_slate(
    feature_idx, r_pb, code_names, code_descriptions, n_candidates=5, n_hard_neg=3, seed=42
):
    """Build a shuffled slate: top-|r_pb| candidates + hard negatives + none."""
    row = np.abs(r_pb[feature_idx])
    order = np.argsort(row)[::-1]  # descending |r_pb|
    cand_idx = list(order[:n_candidates])
    argmax_code = _bare(code_names[int(cand_idx[0])])

    # Hard negatives: lowest |r_pb| codes not already candidates.
    low = [int(i) for i in order[::-1] if int(i) not in {int(c) for c in cand_idx}]
    hard_idx = low[:n_hard_neg]

    entries = []
    for rank, i in enumerate(cand_idx, start=1):
        entries.append(
            {
                "code": _bare(code_names[int(i)]),
                "description": _describe(code_names[int(i)], code_descriptions),
                "rank_by_rpb": rank,
            }
        )
    for i in hard_idx:
        entries.append(
            {
                "code": _bare(code_names[int(i)]),
                "description": _describe(code_names[int(i)], code_descriptions),
                "rank_by_rpb": None,
            }
        )

    rng = np.random.default_rng(seed + int(feature_idx))
    rng.shuffle(entries)
    entries.append({"code": "__none__", "description": "none of these", "rank_by_rpb": None})

    for e, letter in zip(entries, string.ascii_lowercase, strict=False):
        e["letter"] = letter
    return entries, argmax_code
