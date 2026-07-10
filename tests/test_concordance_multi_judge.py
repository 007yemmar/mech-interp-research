import numpy as np

from mech_interp_research.auto_interp import parse_concordance_response
from mech_interp_research.concordance_multi_judge import (
    Judge,
    aggregate_multi_judge,
    build_judges,
    build_retrieval_prompt,
    build_slate,
    judge_deanchored,
    judge_retrieval,
    parse_deanchored_response,
    parse_retrieval_response,
)


def test_parse_wellformed_verdict():
    assert parse_concordance_response("PARTIAL | warfarin treats the code") == (
        "PARTIAL",
        "warfarin treats the code",
    )


def test_parse_prefix_only():
    v, r = parse_concordance_response("YES the explanation names the condition")
    assert v == "YES"
    assert "explanation names" in r


def test_parse_garbage_is_unknown():
    v, r = parse_concordance_response("mumble mumble no verdict here")
    assert v == "UNKNOWN"
    assert r == "mumble mumble no verdict here"


class _StubMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        text = self._text

        class _Resp:
            content = [type("C", (), {"text": text})()]

        return _Resp()


class _StubAnthropic:
    def __init__(self, text="YES | ok"):
        self.messages = _StubMessages(text)


class _StubOpenAI:
    def __init__(self, text="NO | nope"):
        outer = self

        class _Completions:
            def create(self, **kwargs):
                msg = type("M", (), {"content": outer._text})()
                choice = type("Ch", (), {"message": msg})()
                return type("R", (), {"choices": [choice]})()

        self._text = text
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_judge_anthropic_complete():
    j = Judge(
        "sonnet-4-6", "anthropic", model="claude-sonnet-4-6", client=_StubAnthropic("  YES | ok  ")
    )
    assert j.complete("hi") == "YES | ok"


def test_judge_openrouter_complete():
    j = Judge("gpt-4o", "openrouter", model="openai/gpt-4o", client=_StubOpenAI("  NO | nope  "))
    assert j.complete("hi") == "NO | nope"


def test_build_judges_skips_reuse():
    cfgs = [
        {"slug": "sonnet-4-6", "backend": "reuse"},
        {"slug": "gpt-4o", "backend": "openrouter", "model": "openai/gpt-4o"},
    ]
    judges = build_judges(cfgs, openrouter_client=_StubOpenAI("NO | nope"))
    assert [j.slug for j in judges] == ["gpt-4o"]


CODES = [
    "icd9_4019",
    "icd9_4280",
    "icd9_42731",
    "icd9_25000",
    "icd9_5849",
    "icd9_311",
    "icd9_V4986",
    "icd9_2449",
]
DESCS = {
    "4019": "hypertension",
    "4280": "heart failure",
    "42731": "atrial fibrillation",
    "25000": "diabetes",
    "5849": "acute kidney failure",
    "311": "depression",
    "2449": "hypothyroidism",
}  # V4986 intentionally absent → fallback


def test_slate_has_candidates_hardneg_and_none():
    r = np.zeros((1, len(CODES)))
    r[0] = [0.05, 0.60, 0.50, 0.40, 0.30, 0.20, 0.01, 0.02]  # ranks by |r|
    slate, argmax = build_slate(0, r, CODES, DESCS, n_candidates=5, n_hard_neg=2, seed=1)
    codes = [e["code"] for e in slate]
    assert argmax == "4280"  # highest |r|
    assert codes[-1] == "__none__"  # none option last after shuffle-then-append
    cand = [e for e in slate if e["rank_by_rpb"] is not None]
    assert len(cand) == 5  # top-5
    hard = [e for e in slate if e["rank_by_rpb"] is None and e["code"] != "__none__"]
    assert len(hard) == 2  # two hard negatives, low |r|
    assert {e["code"] for e in hard} == {"V4986", "2449"}  # pin the lowest-|r| codes
    letters = [e["letter"] for e in slate]
    assert letters == sorted(set(letters))  # unique, ordered letters


def test_slate_v4986_description_fallback():
    r = np.zeros((1, len(CODES)))
    r[0] = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1]  # V4986 becomes a top candidate
    slate, _ = build_slate(0, r, CODES, DESCS, n_candidates=2, n_hard_neg=0, seed=1)
    v = next(e for e in slate if e["code"] == "V4986")
    assert v["description"] == "Do not resuscitate status"


def test_parse_deanchored_partial_subtype():
    out = parse_deanchored_response("PARTIAL | treatment | warfarin treats it")
    assert out["verdict"] == "PARTIAL"
    assert out["subtype"] == "treatment"
    assert "warfarin" in out["rationale"]


def test_parse_deanchored_yes_no_subtype():
    out = parse_deanchored_response("YES | names the condition")
    assert out["verdict"] == "YES"
    assert out["subtype"] is None


def test_parse_deanchored_unrecognized_subtype_maps_to_related_other():
    out = parse_deanchored_response("PARTIAL | causal-link | some rationale")
    assert out["verdict"] == "PARTIAL"
    assert out["subtype"] == "related-other"  # unrecognized subtype normalized


def test_judge_deanchored_uses_prompt_without_rpb():
    captured = {}

    class _Rec(Judge):
        def complete(self, prompt, max_tokens=256):
            captured["prompt"] = prompt
            return "NO | unrelated"

    j = _Rec("x", "anthropic", model="m", client=None)
    out = judge_deanchored(j, "anticoagulation", "42731", "atrial fibrillation")
    assert out["verdict"] == "NO"
    assert "r =" not in captured["prompt"] and "correlation" not in captured["prompt"].lower()


_SLATE = [
    {"code": "42731", "description": "atrial fibrillation", "rank_by_rpb": 2, "letter": "a"},
    {"code": "4280", "description": "heart failure", "rank_by_rpb": 1, "letter": "b"},
    {"code": "25000", "description": "diabetes", "rank_by_rpb": None, "letter": "c"},
    {"code": "__none__", "description": "none of these", "rank_by_rpb": None, "letter": "d"},
]


def test_parse_retrieval_picks_rank():
    out = parse_retrieval_response("b | best match", _SLATE)
    assert out["picked_code"] == "4280"
    assert out["picked_rank"] == 1
    assert out["is_none"] is False


def test_parse_retrieval_none():
    out = parse_retrieval_response("d | nothing fits", _SLATE)
    assert out["is_none"] is True
    assert out["picked_rank"] is None


def test_parse_retrieval_garbage():
    out = parse_retrieval_response("???", _SLATE)
    assert out["picked_code"] == "__unparse__"


def test_retrieval_prompt_lists_all_letters():
    p = build_retrieval_prompt("heart failure text", _SLATE)
    for letter in ("a", "b", "c", "d"):
        assert f"({letter})" in p
    assert "rank" not in p.lower()  # ranks/r_pb never leak into the prompt


def test_parse_retrieval_prose_not_mistaken_for_letter():
    # A rationale starting with the word "A" must NOT be read as picking letter 'a'.
    out = parse_retrieval_response("A more specific code would be better here", _SLATE)
    assert out["picked_code"] == "__unparse__"


def test_parse_retrieval_valid_letter_absent_from_slate():
    out = parse_retrieval_response("e | not in slate", _SLATE)  # _SLATE has a-d only
    assert out["picked_code"] == "__unparse__"


def test_parse_retrieval_bare_and_parenthesized_letter():
    assert parse_retrieval_response("b", _SLATE)["picked_code"] == "4280"
    assert parse_retrieval_response("(b)", _SLATE)["picked_code"] == "4280"


def test_judge_retrieval_sends_prompt_and_parses(monkeypatch):
    from mech_interp_research.concordance_multi_judge import Judge

    captured = {}

    class _Rec(Judge):
        def complete(self, prompt, max_tokens=256):
            captured["prompt"] = prompt
            return "b | best match"

    j = _Rec("x", "anthropic", model="m", client=None)
    out = judge_retrieval(j, "heart failure text", _SLATE)
    assert out["picked_code"] == "4280"
    assert "rank" not in captured["prompt"].lower()


def test_derange_no_feature_keeps_own_code():
    from mech_interp_research.concordance_multi_judge import derange_feature_codes

    f2c = {10: "4280", 11: "42731", 12: "25000", 13: "5849"}
    wrong = derange_feature_codes(f2c, seed=3)
    assert set(wrong) == set(f2c)
    for fid, code in wrong.items():
        assert code != f2c[fid]  # every feature gets a wrong code
        assert code in f2c.values()


def test_derange_wrong_code_with_duplicate_codes():
    from mech_interp_research.concordance_multi_judge import derange_feature_codes

    # Two features share code "4280"; a valid id-derangement could swap them and
    # hand each its own code back. The code-level guarantee must prevent that.
    f2c = {10: "4280", 11: "4280", 12: "25000", 13: "5849"}
    wrong = derange_feature_codes(f2c, seed=7)
    for fid, code in wrong.items():
        assert code != f2c[fid]
        assert code in f2c.values()


def test_derange_dominant_code_warns_and_terminates(caplog):
    import logging

    from mech_interp_research.concordance_multi_judge import derange_feature_codes

    # Code "X" dominates (3 of 4) → some collision unavoidable; must still return.
    f2c = {1: "X", 2: "X", 3: "X", 4: "Y"}
    with caplog.at_level(logging.WARNING):
        wrong = derange_feature_codes(f2c, seed=1)
    assert set(wrong) == set(f2c)  # terminates, all features assigned
    assert any("kept their own code" in r.message for r in caplog.records)


def test_fleiss_perfect_agreement():
    from mech_interp_research.concordance_multi_judge import fleiss_kappa

    # 3 items, 3 raters, all agree → kappa 1.0
    counts = np.array([[3, 0], [0, 3], [3, 0]], dtype=float)
    assert abs(fleiss_kappa(counts) - 1.0) < 1e-9


def test_pairwise_cohen_perfect():
    from mech_interp_research.concordance_multi_judge import pairwise_cohen

    labels = {"a": ["YES", "NO", "YES"], "b": ["YES", "NO", "YES"]}
    out = pairwise_cohen(labels)
    assert abs(out["a__b"] - 1.0) < 1e-9


def test_pairwise_cohen_excludes_unknown():
    from mech_interp_research.concordance_multi_judge import pairwise_cohen

    labels = {"a": ["YES", "UNKNOWN", "NO"], "b": ["YES", "YES", "NO"]}
    out = pairwise_cohen(labels)  # only items 0 and 2 counted
    assert abs(out["a__b"] - 1.0) < 1e-9


def _row(fid, tier, r, **kw):
    base = {"feature_idx": fid, "tier": tier, "r_pb": r}
    base.update(kw)
    return base


def test_aggregate_yes_only_and_hits():
    rows = [
        _row(
            1,
            "strong_grounded",
            0.6,
            gpt_verdict="YES",
            gpt_subtype=None,
            gpt_pick_rank=1,
            gpt_is_none=False,
        ),
        _row(
            2,
            "strong_grounded",
            0.45,
            gpt_verdict="PARTIAL",
            gpt_subtype="treatment",
            gpt_pick_rank=3,
            gpt_is_none=False,
        ),
    ]
    out = aggregate_multi_judge(rows, ["gpt"], [0.4])
    g = out["per_judge"]["gpt"]["r>0.4"]
    assert g["yes_only_rate"] == 0.5
    assert g["yes_partial_rate"] == 1.0
    assert g["hit1_rate"] == 0.5  # only feature 1 picked rank 1
    assert g["hit3_rate"] == 1.0  # both within rank 3


def test_aggregate_majority_and_kappa_keys():
    rows = [
        _row(
            1,
            "strong_grounded",
            0.6,
            a_verdict="YES",
            a_subtype=None,
            a_pick_rank=1,
            a_is_none=False,
            b_verdict="YES",
            b_subtype=None,
            b_pick_rank=1,
            b_is_none=False,
        ),
    ]
    out = aggregate_multi_judge(rows, ["a", "b"], [0.3])
    assert "fleiss_binarized" in out["agreement"]
    assert "a__b" in out["agreement"]["pairwise_cohen_binarized"]
    assert out["agreement"]["majority_counts"]["YES"] == 1
