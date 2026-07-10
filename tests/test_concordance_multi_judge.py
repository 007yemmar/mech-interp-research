import numpy as np

from mech_interp_research.auto_interp import parse_concordance_response
from mech_interp_research.concordance_multi_judge import Judge, build_judges, build_slate


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
