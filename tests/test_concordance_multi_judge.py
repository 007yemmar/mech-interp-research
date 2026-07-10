from mech_interp_research.auto_interp import parse_concordance_response


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
