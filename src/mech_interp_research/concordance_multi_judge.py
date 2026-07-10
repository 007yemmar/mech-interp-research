"""Multi-judge discriminative concordance for the JumpReLU auto-interp run.

Re-validates ICD concordance with a cross-provider judge panel, a top-5
retrieval arm, a shuffled null, a human-adjudication sheet, and an independent
second explainer. Reuses a completed auto_interp run; no GPU.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter

import numpy as np

from mech_interp_research.auto_interp import parse_concordance_response
from mech_interp_research.shuffled_control import permute_global

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
    row = np.nan_to_num(np.abs(r_pb[feature_idx]), nan=-1.0)
    asc = np.argsort(row)  # ascending |r_pb|
    order = asc[::-1]  # descending |r_pb|
    cand_idx = list(order[:n_candidates])
    argmax_code = _bare(code_names[int(cand_idx[0])])

    # Hard negatives: lowest |r_pb| codes not already candidates.
    low = [int(i) for i in asc if int(i) not in {int(c) for c in cand_idx}]
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


_PARTIAL_SUBTYPES = {
    "treatment",
    "symptom",
    "broader-category",
    "narrower-concept",
    "related-other",
}

DEANCHORED_CONCORDANCE_PROMPT = """\
A sparse autoencoder feature has been auto-interpreted as:
"{explanation}"

Consider the ICD-9 diagnosis code {code} ({description}).

Does the explanation describe the same clinical concept as the code?
- YES  — the explanation clearly describes the code's concept
- PARTIAL — related but not identical; also name the subtype: one of
  treatment, symptom, broader-category, narrower-concept, related-other
- NO   — unrelated

Format for YES/NO:   <verdict> | <one-sentence rationale>
Format for PARTIAL:  PARTIAL | <subtype> | <one-sentence rationale>"""


def parse_deanchored_response(raw: str) -> dict:
    """Parse a de-anchored reply into {verdict, subtype, rationale}; unknown subtype → related-other."""
    raw = raw.strip()
    m = re.match(r"PARTIAL\s*[|]\s*([a-z\-]+)\s*[|]\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        subtype = m.group(1).strip().lower()
        return {
            "verdict": "PARTIAL",
            "subtype": subtype if subtype in _PARTIAL_SUBTYPES else "related-other",
            "rationale": m.group(2).strip(),
        }
    verdict, rationale = parse_concordance_response(raw)
    return {"verdict": verdict, "subtype": None, "rationale": rationale}


def judge_deanchored(judge, explanation, code, description) -> dict:
    """Arm 1: ask a judge whether the explanation matches the code, no r_pb anchor."""
    prompt = DEANCHORED_CONCORDANCE_PROMPT.format(
        explanation=explanation, code=code, description=description
    )
    return parse_deanchored_response(judge.complete(prompt))


RETRIEVAL_PROMPT_HEADER = """\
A sparse autoencoder feature has been auto-interpreted as:
"{explanation}"

Which ONE of these ICD-9 codes does the explanation best describe? If none
fit, choose the 'none' option. Answer with one letter, then a rationale.

{options}

Format: <letter> | <one-sentence rationale>"""


def build_retrieval_prompt(explanation, slate) -> str:
    """Build a retrieval-ranking prompt from explanation and slate."""
    options = "\n".join(f"({e['letter']}) {e['code']}  {e['description']}" for e in slate)
    return RETRIEVAL_PROMPT_HEADER.format(explanation=explanation, options=options)


def parse_retrieval_response(raw, slate) -> dict:
    """Parse a retrieval response into {picked_letter, picked_code, picked_rank, is_none}.

    Returns:
        dict with keys:
        - picked_letter: the matched letter or None if unparseable
        - picked_code: the matched code, or "__unparse__" if unparseable
        - picked_rank: the rank_by_rpb value or None
        - is_none: True if the picked code is "__none__"
    """
    by_letter = {e["letter"]: e for e in slate}
    s = raw.strip()
    # Accept the documented "<letter> | rationale" form, or a bare/parenthesized
    # letter alone. A letter followed by prose (no pipe) is NOT a pick — this
    # prevents rationales like "A more specific code..." from being read as
    # letter 'a'. Mirrors parse_concordance_response's whole-token discipline.
    m = re.match(r"\(?([a-z])\)?\s*(?:[|]|$)", s, re.IGNORECASE)
    letter = m.group(1).lower() if m else None
    if letter is None or letter not in by_letter:
        return {
            "picked_letter": None,
            "picked_code": "__unparse__",
            "picked_rank": None,
            "is_none": False,
        }
    e = by_letter[letter]
    return {
        "picked_letter": letter,
        "picked_code": e["code"],
        "picked_rank": e["rank_by_rpb"],
        "is_none": e["code"] == "__none__",
    }


def judge_retrieval(judge, explanation, slate) -> dict:
    """Arm 2: ask a judge to rank codes in a slate by relevance to the explanation."""
    return parse_retrieval_response(
        judge.complete(build_retrieval_prompt(explanation, slate)), slate
    )


def derange_feature_codes(feature_to_code, seed=42, max_tries=64) -> dict:
    """Assign each feature a *wrong* ICD code (different from its own).

    Deranges feature ids, but the guarantee is on the code: a feature paired
    with its own true code would make the judge answer the real question and
    contaminate the Arm-3 null. Random derangements are rejection-sampled until
    every assigned code differs from the feature's own; if a single code
    dominates (>n/2 features) that is impossible, so a sorted-rotation fallback
    minimises residual collisions and logs how many remain.
    """
    fids = sorted(feature_to_code)
    n = len(fids)
    if n < 2:
        raise ValueError(f"need at least 2 features, got {n}")

    for attempt in range(max_tries):
        perm = permute_global(fids, seed=seed + attempt)
        assigned = {fid: feature_to_code[perm[fid]] for fid in fids}
        if all(assigned[fid] != feature_to_code[fid] for fid in fids):
            return assigned

    # Fallback for dominant-code cases: sort by code (same-code features become
    # contiguous blocks) and rotate assignment by the largest block size, which
    # pushes each feature's source out of its own code block whenever feasible.
    order = sorted(fids, key=lambda f: (str(feature_to_code[f]), f))
    max_group = max(Counter(feature_to_code.values()).values())
    assigned = {}
    collisions = 0
    for i, fid in enumerate(order):
        src = order[(i + max_group) % n]
        code = feature_to_code[src]
        assigned[fid] = code
        if code == feature_to_code[fid]:
            collisions += 1
    if collisions:
        logger.warning(
            "derange_feature_codes: %d/%d features kept their own code "
            "(a dominant ICD code exceeds half the features)",
            collisions,
            n,
        )
    return assigned
