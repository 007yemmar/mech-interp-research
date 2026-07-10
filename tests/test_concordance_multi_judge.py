from mech_interp_research.auto_interp import parse_concordance_response
from mech_interp_research.concordance_multi_judge import Judge, build_judges


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
