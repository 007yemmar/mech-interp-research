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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from mech_interp_research.auto_interp import (
    CONCORDANCE_PROMPT,
    _load_code_descriptions,
    _write_json,
    explain_and_categorize_feature,
    parse_concordance_response,
    resolve_token_text,
)
from mech_interp_research.icd_eval import load_saved_correlations
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


def judge_original(judge, explanation, icd_code, icd_description, r_pb) -> dict:
    """Arm 0: the VERBATIM original CONCORDANCE_PROMPT (r-anchor included), via any
    judge — the apples-to-apples multi-model replication of the published table."""
    prompt = CONCORDANCE_PROMPT.format(
        explanation=explanation, r_pb=r_pb, icd_code=icd_code, icd_description=icd_description
    )
    verdict, rationale = parse_concordance_response(judge.complete(prompt))
    return {"verdict": verdict, "rationale": rationale}


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


def fleiss_kappa(rating_counts) -> float:
    """Compute Fleiss' kappa from per-item category counts.

    Args:
        rating_counts: numpy array of shape [n_items, n_categories] containing
            per-item category tallies.

    Returns:
        Fleiss' kappa (float), or nan if undefined (≤1 rater, zero variance,
        unequal raters per item, or no items).
    """
    counts = np.asarray(rating_counts, dtype=float)
    n_items, _ = counts.shape
    n_raters = counts.sum(axis=1)
    if n_items == 0 or np.any(n_raters < 2) or len(set(n_raters.tolist())) != 1:
        return float("nan")
    n = n_raters[0]
    p_j = counts.sum(axis=0) / (n_items * n)
    P_i = (np.square(counts).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = float(np.square(p_j).sum())
    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return float((P_bar - P_e) / (1 - P_e))


def _binarize(v: str) -> str | None:
    if v in ("YES", "PARTIAL"):
        return "concordant"
    if v == "NO":
        return "discordant"
    return None  # UNKNOWN excluded


def aggregate_multi_judge(
    per_feature_rows, judge_slugs, thresholds, verdict_key="verdict", rank_key="pick_rank"
) -> dict:
    """Aggregate per-feature multi-judge rows into rate summaries + agreement stats.

    Args:
        per_feature_rows: list of dicts, each with ``feature_idx``, ``tier``,
            ``r_pb`` (float) plus, for each judge slug ``S``:
            ``f"{S}_{verdict_key}"``, ``f"{S}_subtype"``, and (when
            ``rank_key`` is not None) ``f"{S}_{rank_key}"`` (int|None),
            ``f"{S}_is_none"`` (bool).
        judge_slugs: list of judge slug strings.
        thresholds: list of |r_pb| thresholds to filter rows on.
        verdict_key: suffix (after ``f"{slug}_"``) to read verdicts from.
            Defaults to ``"verdict"`` (Arm 1, de-anchored). Pass
            ``"orig_verdict"`` for Arm 0 (original, anchored prompt).
        rank_key: suffix (after ``f"{slug}_"``) to read retrieval ranks
            from, or ``None`` to omit hit@k entirely (Arm 0 has no
            retrieval ranks). Defaults to ``"pick_rank"``.

    Returns:
        dict with ``per_judge`` (rate summaries per judge/threshold),
        ``agreement`` (fleiss/cohen kappa + majority counts over all rows),
        ``judge_slugs``, and ``thresholds``.
    """
    per_judge = {}
    for s in judge_slugs:
        per_judge[s] = {}
        for thr in thresholds:
            rows = [r for r in per_feature_rows if abs(r["r_pb"]) > thr]
            n = len(rows)
            verdicts = [r[f"{s}_{verdict_key}"] for r in rows]
            ranks = [r[f"{s}_{rank_key}"] for r in rows] if rank_key is not None else None
            yes = sum(1 for v in verdicts if v == "YES")
            yp = sum(1 for v in verdicts if v in ("YES", "PARTIAL"))

            def hit(k, ranks=ranks):
                return sum(1 for rk in ranks if rk is not None and rk <= k)

            entry = {
                "n": n,
                "yes_only_rate": (yes / n) if n else None,
                "yes_partial_rate": (yp / n) if n else None,
            }
            if ranks is not None:
                entry["hit1_rate"] = (hit(1) / n) if n else None
                entry["hit3_rate"] = (hit(3) / n) if n else None
                entry["hit5_rate"] = (hit(5) / n) if n else None
            per_judge[s][f"r>{thr}"] = entry

    # Agreement over all rows, binarized (YES∪PARTIAL vs NO; UNKNOWN excluded).
    labels_by_judge = {s: [r[f"{s}_{verdict_key}"] for r in per_feature_rows] for s in judge_slugs}
    majority = Counter()
    for r in per_feature_rows:
        vs = [r[f"{s}_{verdict_key}"] for s in judge_slugs]
        top = Counter(vs).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            majority["NO_CONSENSUS"] += 1
        else:
            majority[top[0][0]] += 1

    # Binarized Fleiss over complete cases (every judge non-UNKNOWN → equal
    # raters per item = n_judges). Each vote maps to concordant (YES/PARTIAL)
    # or discordant (NO); rows with any UNKNOWN vote are dropped so the two
    # "binarized" agreement metrics share one basis.
    bin_counts = []
    for r in per_feature_rows:
        vs = [r[f"{s}_{verdict_key}"] for s in judge_slugs]
        if any(v == "UNKNOWN" for v in vs):
            continue
        concordant = sum(1 for v in vs if v in ("YES", "PARTIAL"))
        discordant = sum(1 for v in vs if v == "NO")
        bin_counts.append([concordant, discordant])
    fleiss_bin = fleiss_kappa(np.array(bin_counts, dtype=float)) if bin_counts else float("nan")

    bin_labels = {s: [_binarize(v) or "UNKNOWN" for v in labels_by_judge[s]] for s in judge_slugs}
    agreement = {
        "fleiss_binarized": fleiss_bin,
        "pairwise_cohen_binarized": pairwise_cohen(bin_labels),
        "majority_counts": dict(majority),
    }
    return {
        "per_judge": per_judge,
        "agreement": agreement,
        "judge_slugs": judge_slugs,
        "thresholds": thresholds,
    }


def pairwise_cohen(labels_by_judge) -> dict:
    """Compute pairwise Cohen's kappa across judge pairs.

    Args:
        labels_by_judge: dict mapping judge names to lists of labels (str).

    Returns:
        dict mapping "judgeA__judgeB" to Cohen's kappa score, or nan if fewer
        than 2 usable (non-UNKNOWN) label pairs.
    """
    out = {}
    for a, b in combinations(sorted(labels_by_judge), 2):
        la, lb = labels_by_judge[a], labels_by_judge[b]
        pairs = [(x, y) for x, y in zip(la, lb, strict=True) if x != "UNKNOWN" and y != "UNKNOWN"]
        if len(pairs) < 2:
            out[f"{a}__{b}"] = float("nan")
            continue
        out[f"{a}__{b}"] = float(cohen_kappa_score([p[0] for p in pairs], [p[1] for p in pairs]))
    return out


def run_concordance_multi_judge(
    auto_interp_dir,
    icd_eval_dir,
    output_dir=None,
    judges=None,
    thresholds=None,
    n_candidates=5,
    n_hard_neg=3,
    seed=42,
    icd_keywords_yaml_path=None,
    run_shuffled=True,
    arm6=None,
    max_workers=6,
    _judges=None,
    _corr=None,
    _code_descriptions=None,
    _explainer_client=None,
    _contexts_by_fid=None,
    _note_texts=None,
    _tokenizer=None,
    _commit_volume=None,
) -> dict:
    """Orchestrate the multi-judge concordance re-validation over a completed auto-interp run.

    Per feature, per judge, runs Arm 0 (``judge_original``, the verbatim
    anchored prompt — apples-to-apples with the paper's table), Arm 1
    (``judge_deanchored``, no r_pb anchor), then Arm 2 (``judge_retrieval``,
    top-k slate ranking), in that order. Aggregates Arm 0 and Arm 1 into two
    separate summary blocks and writes both a JSON summary and a flat
    per-feature/per-judge verdict-matrix CSV.

    Args:
        auto_interp_dir: directory containing a completed auto-interp run's
            ``concordance_results.csv`` (feature_idx, tier, explanation,
            concordance_r_pb).
        icd_eval_dir: directory containing a completed ICD eval run (used to
            load saved correlations) unless ``_corr`` is injected.
        output_dir: where to write ``multi_judge_summary.json`` and
            ``verdict_matrix.csv``. Defaults to ``auto_interp_dir / "multi_judge"``.
        judges: list of judge config dicts (``slug``, ``backend``, ``model``),
            passed to ``build_judges`` unless ``_judges`` is injected.
        thresholds: list of |r_pb| thresholds for aggregation. Defaults to
            ``[0.3, 0.4, 0.5]``.
        n_candidates, n_hard_neg, seed: passed through to ``build_slate``.
        icd_keywords_yaml_path: fallback YAML path for code descriptions,
            used when ``_code_descriptions`` is not injected.
        run_shuffled: when True, runs Arm 3 (shuffled null) — deranges each
            feature's argmax code via ``derange_feature_codes`` and re-judges
            each explanation against the wrong code with ``judge_deanchored``,
            adding ``summary["shuffled_null"]``.
        arm6: when a config dict (with ``model`` and optional ``sample``) is
            provided alongside ``_explainer_client`` and ``_contexts_by_fid``,
            runs Arm 6 (independent explainer) — regenerates explanations via
            ``regenerate_explanations``, re-judges them with
            ``judge_deanchored``, writes ``explainer_comparison.csv``, and
            sets ``summary["arm6"]``.
        max_workers: reserved for parallelizing judge calls (Task 12); unused
            in this orchestrator core.
        _judges, _corr, _code_descriptions, _explainer_client,
            _contexts_by_fid, _note_texts, _tokenizer, _commit_volume:
            test-injection hooks mirroring ``run_shuffled_control`` — when
            unset, production values are built/loaded from disk or the
            configured backends.

    Returns:
        dict with ``"arm0_original"`` and ``"arm1_deanchored"`` aggregation
        blocks (see ``aggregate_multi_judge``), plus ``"shuffled_null"`` when
        ``run_shuffled`` and ``"arm6"`` when the Arm 6 hooks are provided.
    """
    if judges is None:
        judges = []
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5]

    auto_interp_dir = Path(auto_interp_dir)
    output_dir = Path(output_dir) if output_dir else auto_interp_dir / "multi_judge"
    output_dir.mkdir(parents=True, exist_ok=True)

    corr = _corr or load_saved_correlations(icd_eval_dir)
    r_pb, code_names = corr["r_pb"], corr["code_names"]
    code_descriptions = (
        _code_descriptions
        if _code_descriptions is not None
        else _load_code_descriptions(None, icd_keywords_yaml_path)
    )

    df = pd.read_csv(auto_interp_dir / "concordance_results.csv")
    feats = df.to_dict("records")

    judge_objs = _judges if _judges is not None else build_judges(judges)
    judge_slugs = [j.slug for j in judge_objs]

    per_feature_rows = []
    for rec in feats:
        fid = int(rec["feature_idx"])
        expl = rec["explanation"]
        slate, argmax_code = build_slate(
            fid, r_pb, code_names, code_descriptions, n_candidates, n_hard_neg, seed
        )
        desc = next((e["description"] for e in slate if e["code"] == argmax_code), argmax_code)
        row = {
            "feature_idx": fid,
            "tier": rec["tier"],
            "r_pb": float(rec.get("concordance_r_pb", 0.0)),
            "explanation": expl,
            "argmax_code": argmax_code,
        }
        for j in judge_objs:
            orig = judge_original(j, expl, argmax_code, desc, row["r_pb"])  # Arm 0
            d = judge_deanchored(j, expl, argmax_code, desc)  # Arm 1
            ret = judge_retrieval(j, expl, slate)  # Arm 2
            row[f"{j.slug}_orig_verdict"] = orig["verdict"]
            row[f"{j.slug}_orig_rationale"] = orig["rationale"]
            row[f"{j.slug}_verdict"] = d["verdict"]
            row[f"{j.slug}_subtype"] = d["subtype"]
            row[f"{j.slug}_rationale"] = d["rationale"]
            row[f"{j.slug}_pick_rank"] = ret["picked_rank"]
            row[f"{j.slug}_is_none"] = ret["is_none"]
        per_feature_rows.append(row)

    summary = {
        "arm0_original": aggregate_multi_judge(
            per_feature_rows, judge_slugs, thresholds, verdict_key="orig_verdict", rank_key=None
        ),
        "arm1_deanchored": aggregate_multi_judge(
            per_feature_rows, judge_slugs, thresholds
        ),  # verdict/pick_rank defaults
    }
    _write_json(summary, output_dir / "multi_judge_summary.json")
    pd.DataFrame(per_feature_rows).to_csv(output_dir / "verdict_matrix.csv", index=False)

    if run_shuffled and per_feature_rows:
        feature_to_code = {r["feature_idx"]: r["argmax_code"] for r in per_feature_rows}
        wrong = derange_feature_codes(feature_to_code, seed=seed)
        shuffled_null = {}
        for j in judge_objs:
            yp = 0
            for r in per_feature_rows:
                wc = wrong[r["feature_idx"]]  # a different feature's code
                d = judge_deanchored(j, r["explanation"], wc, _describe(wc, code_descriptions))
                yp += 1 if d["verdict"] in ("YES", "PARTIAL") else 0
            n = len(per_feature_rows)
            shuffled_null[j.slug] = {"yes_partial_rate": yp / n if n else None, "n": n}
        summary["shuffled_null"] = shuffled_null

    if arm6 and _explainer_client is not None and _contexts_by_fid is not None:
        sub = [r["feature_idx"] for r in per_feature_rows][
            : arm6.get("sample") or len(per_feature_rows)
        ]
        alt = regenerate_explanations(
            _explainer_client, sub, _contexts_by_fid, _note_texts or {}, _tokenizer, arm6["model"]
        )
        comp = []
        for r in per_feature_rows:
            if r["feature_idx"] not in alt:
                continue
            slate, argmax_code = build_slate(
                r["feature_idx"],
                r_pb,
                code_names,
                code_descriptions,
                n_candidates,
                n_hard_neg,
                seed,
            )
            desc = next((e["description"] for e in slate if e["code"] == argmax_code), argmax_code)
            per_judge = {}
            for j in judge_objs:
                per_judge[j.slug] = judge_deanchored(j, alt[r["feature_idx"]], argmax_code, desc)[
                    "verdict"
                ]
            comp.append(
                {
                    "feature_idx": r["feature_idx"],
                    "sonnet_explanation": r["explanation"],
                    "alt_explanation": alt[r["feature_idx"]],
                    **{f"{k}_verdict": v for k, v in per_judge.items()},
                }
            )
        pd.DataFrame(comp).to_csv(output_dir / "explainer_comparison.csv", index=False)
        summary["arm6"] = {"n_reexplained": len(comp), "model": arm6["model"]}

    if _commit_volume is not None:
        _commit_volume()
    return summary


def regenerate_explanations(
    explainer_client,
    feature_ids,
    contexts_by_fid,
    note_texts,
    tokenizer,
    model,
    n_contexts_train=20,
    context_window=30,
) -> dict:
    """Regenerate explanations with an alternate explainer over cached contexts.

    This is the ONLY function that reads note text; run it on the route the
    operator has configured for the corpus.

    Args:
        explainer_client: Anthropic or OpenAI-shaped client (the ._client from a Judge).
        feature_ids: list of feature indices to regenerate explanations for.
        contexts_by_fid: dict mapping feature_id to context dict with pos_contexts/neg_contexts.
        note_texts: dict mapping note_id to note text (used by resolve_token_text).
        tokenizer: tokenizer for resolve_token_text (or None to skip text resolution).
        model: model string to pass to explain_and_categorize_feature.
        n_contexts_train: number of top contexts to use for explanation.
        context_window: context window for resolve_token_text.

    Returns:
        dict mapping feature_idx to explanation string.
    """
    out = {}
    for fid in feature_ids:
        ctx = contexts_by_fid.get(fid)
        if not ctx or not ctx.get("pos_contexts"):
            continue
        pos = list(ctx["pos_contexts"])
        if tokenizer is not None:
            resolve_token_text(pos, note_texts, tokenizer, context_window=context_window)
        explanation, _ = explain_and_categorize_feature(
            explainer_client, pos[:n_contexts_train], model=model
        )
        out[fid] = explanation
    return out
