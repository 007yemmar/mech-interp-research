"""Tests for the §14 concordance-table builder.

Unit tests run on tiny synthetic verdict frames (no MIMIC data). The reproduction
gate, ``test_reproduces_published_tables``, needs the real verdict CSVs pulled with
``uv run python scripts/build_concordance_results.py --fetch`` into the gitignored
``.tmp/concordance_fixtures/`` and is skipped when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mech_interp_research import concordance_tables as ct
from mech_interp_research.concordance_tables import (
    BANDS3,
    BANDS6,
    band3_of,
    band_of,
    banded_counts,
    cell,
    fisher_p,
    judge_gap,
    merge_pools,
    pct,
    range_cell,
    retention,
    stars,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / ".tmp" / "concordance_fixtures"
PUBLISHED = REPO / "results" / "concordance_validation"
SOURCES = REPO / "configs" / "concordance_sources.yaml"


# ------------------------------------------------------------------ primitives


def test_band_edges_are_left_closed():
    assert band_of(0.0999) is None
    assert band_of(0.1) == "0.1-0.2" and band_of(0.2999) == "0.2-0.3"
    assert band_of(0.6) == "0.6+" and band_of(0.95) == "0.6+"
    assert band_of(-0.45) == "0.4-0.5"


def test_band3_edges_are_left_closed():
    assert band3_of(0.05) is None
    assert band3_of(0.1) == "0.1-0.3" and band3_of(0.2999) == "0.1-0.3"
    assert band3_of(0.3) == "0.3-0.5" and band3_of(0.5) == ">0.5"


def test_cell_format():
    assert cell(0, 0) == "–"
    assert cell(18, 136) == "13 (136)"


def test_pct_rounds_half_to_even():
    # The published tables show 52.5 -> 52 and 87.5 -> 88 (banker's rounding).
    assert pct(42, 80) == 52
    assert pct(35, 40) == 88
    assert pct(3, 40) == 8
    assert pct(1, 40) == 2


def test_stars_thresholds():
    assert stars(0.0009) == "***"
    assert stars(0.009) == "**"
    assert stars(0.049) == "*"
    assert stars(0.05) == "ns"


def test_fisher_two_sided_symmetric():
    p = fisher_p(12, 16, 6, 16)
    assert 0 < p < 1
    assert p == pytest.approx(fisher_p(6, 16, 12, 16))


def _frame(rs, **cols):
    return pd.DataFrame({"feature_idx": range(len(rs)), "abs_r": rs, **cols})


def test_banded_counts_unknown_stays_in_denominator():
    df = _frame([0.15, 0.15, 0.15, 0.45], verdict=["YES", "UNKNOWN", "NO", "YES"])
    c = banded_counts(df, df["verdict"] == "YES")
    assert c["0.1-0.2"] == (1, 3)
    assert c["0.4-0.5"] == (1, 1)
    assert c["0.3-0.4"] == (0, 0)
    assert set(c) == set(BANDS6)
    c3 = banded_counts(df, df["verdict"] == "YES", "3")
    assert c3 == {"0.1-0.3": (1, 3), "0.3-0.5": (1, 1), ">0.5": (0, 0)}
    assert set(c3) == set(BANDS3)


def test_merge_pools_first_pool_wins():
    pub = pd.DataFrame({"feature_idx": [1, 2], "abs_r": [0.5, 0.6], "hit1": [1, 1]})
    strat = pd.DataFrame({"feature_idx": [2, 3], "abs_r": [0.61, 0.2], "hit1": [0, 0]})
    m = merge_pools([pub, strat, None])
    assert sorted(m["feature_idx"]) == [1, 2, 3]
    assert m.set_index("feature_idx").loc[2, "hit1"] == 1


def test_judge_gap_is_unrounded_rate_difference():
    a = {"0.4-0.5": (24, 136), "0.6+": (0, 0)}
    b = {"0.4-0.5": (29, 136), "0.6+": (1, 2)}
    # 17.6% vs 21.3% -> 3.7 -> "4"; an empty side -> dash
    assert judge_gap(a, b) == {"0.4-0.5": "4", "0.6+": "–"}


def test_retention():
    df = _frame(
        [0.5] * 4,
        anchored=["YES", "YES", "PARTIAL", "YES"],
        deanchored=["YES", "NO", "YES", "UNKNOWN"],
    )
    assert retention(df) == (1, 3)


def test_range_cell():
    assert range_cell({"S": (1, 4)}) == "–"
    assert range_cell({"S": (0, 139), "D": (0, 139)}) == "0 (139)"
    assert range_cell({"S": (21, 175), "D": (37, 175)}) == "12–21 (175)"


def test_table_functions_on_synthetic_store(tmp_path):
    """End-to-end over a fake fixtures dir: columns match the published schema."""
    pool = "auto_interp/fake"
    rs = [0.15, 0.25, 0.45, 0.55, 0.65]
    conc = pd.DataFrame(
        {
            "feature_idx": range(5),
            "concordance_r_pb": rs,
            "concordance_verdict": ["NO", "PARTIAL", "YES", "YES", "UNKNOWN"],
        }
    )
    (tmp_path / pool).mkdir(parents=True)
    conc.to_csv(tmp_path / pool / "concordance_results.csv", index=False)
    for arm in ["retrieval_eval", "retrieval_eval_hardneg"]:
        for slug in ["sonnet-4-6", "deepseek-v3"]:
            d = tmp_path / pool / arm / slug
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "feature_idx": range(5),
                    "abs_r_pb": rs,
                    "hit1": [0, 1, 1, 1, 0],
                    "is_none": [1, 0, 0, 0, 0],
                }
            ).to_csv(d / "retrieval_verdicts.csv", index=False)
    cfg = {
        "judges": {
            "Sonnet": {
                "concordance": "auto_interp",
                "retrieval": "sonnet-4-6",
                "deanchored": "x",
                "binary": "x",
            },
            "DeepSeek": {
                "concordance": "deepseek-v3",
                "retrieval": "deepseek-v3",
                "deanchored": "x",
                "binary": "x",
            },
        },
        "sources": [
            {"key": "a", "label": "A", "type": "t", "pool_dirs": [pool]},
            {"key": "r", "label": "R", "type": "t", "pool_dirs": [pool], "fisher_ref": True},
        ],
    }
    tables = ct.build_all(cfg, tmp_path)
    f1 = tables["F1_exactYES"]
    assert list(f1.columns) == ["Source", "Type", "Judge", *BANDS6]
    assert f1.iloc[0].to_dict()["0.4-0.5"] == "100 (1)"
    assert f1.iloc[0].to_dict()["0.6+"] == "0 (1)"
    f8 = tables["F8_significance"]
    assert f8.iloc[0]["0.2-0.3"] == "+0 ns"
    f8c = tables["F8_significance_coarse"]
    assert f8c.iloc[0]["0.1-0.3"] == "50 vs 50  +0 ns (n=2)"


# --------------------------------------------------------- reproduction gate

# Rows the gate must reproduce exactly. Other published rows (gemmascope, keyword,
# diff-in-means, probe, PCA, random spanning) are compared and reported but only
# the gate rows fail the test.
GATE_SOURCES = {
    "SAE jumprelu (pub 380)",
    "SAE jumprelu (strat 200)",
    "SAE jumprelu (380)",
    "SAE vanilla (380)",
    "SAE vanilla (strat 200)",
    "random (300)",
}
KEYS = {
    "F1_exactYES": ["Source", "Type", "Judge"],
    "F2_yespartial": ["Source", "Type", "Judge"],
    "F3_no": ["Source", "Type", "Judge"],
    "F4_hit1_crosschapter": ["Source", "Type", "Judge"],
    "F4b_hit1_samechapter": ["Source", "Type", "Judge"],
    "F5_none_crosschapter": ["Source", "Type", "Judge"],
    "F5b_none_samechapter": ["Source", "Type", "Judge"],
    "F6_interjudge": ["Source", "Metric"],
    "F7_binary": ["Source", "Judge", "Metric"],
    "F8_significance": ["Source", "Condition", "Judge"],
    "F8_significance_coarse": ["Source", "Condition", "Judge"],
    "F9_anchoring": ["Source", "Judge", "Prompt"],
}


def compare_published(tables: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    """``({table: (n_match, n_total)} over gate cells, [mismatch records])``.

    Records cover every published row (gate or not); ``gate`` flags which count.
    """
    tally, mismatches = {}, []
    for name, keys in KEYS.items():
        pub = pd.read_csv(PUBLISHED / f"{name}.csv", dtype=str, keep_default_na=False)
        got = tables[name].astype(str)
        got_idx = {tuple(r[k] for k in keys): r for _, r in got.iterrows()}
        ok = total = 0
        for _, row in pub.iterrows():
            key = tuple(row[k] for k in keys)
            gate = row["Source"] in GATE_SOURCES
            mine = got_idx.get(key)
            for col in pub.columns:
                if col in keys:
                    continue
                want = row[col]
                have = None if mine is None else str(mine.get(col, ""))
                if gate:
                    total += 1
                    ok += have == want
                if have != want:
                    mismatches.append(
                        {
                            "table": name,
                            "row": key,
                            "col": col,
                            "published": want,
                            "computed": have,
                            "gate": gate,
                        }
                    )
        tally[name] = (ok, total)
    return tally, mismatches


# §14 four-judge values for the merged SAE rows, transcribed from RESULTS.md §14.
S14_EXPECTED_HIT1 = {  # §14.1 (n, S, D, G, M) -- every row with n >= 5
    ("keyword (lexical ceiling)", "0.3-0.5"): (26, 92, 96, 96, 100),
    ("keyword (lexical ceiling)", ">0.5"): (6, 100, 100, 100, 100),
    ("keyword (lexical ceiling)", "0.1-0.3"): (6, 50, 83, 83, 83),
    ("SAE GemmaScope", "0.3-0.5"): (50, 52, 46, 52, 50),
    ("SAE GemmaScope", "0.1-0.3"): (100, 0, 0, 0, 0),
    ("diff-in-means (supervised)", "0.3-0.5"): (21, 24, 38, 29, 29),
    ("diff-in-means (supervised)", ">0.5"): (7, 43, 29, 29, 29),
    ("diff-in-means (supervised)", "0.1-0.3"): (18, 22, 28, 22, 28),
    ("probe LR (supervised)", "0.3-0.5"): (24, 17, 17, 17, 17),
    ("probe LR (supervised)", ">0.5"): (5, 40, 40, 0, 20),
    ("probe LR (supervised)", "0.1-0.3"): (17, 12, 18, 12, 12),
    ("PCA", "0.1-0.3"): (34, 12, 15, 15, 12),
    ("SAE JumpReLU", "0.3-0.5"): (175, 73, 73, 75, 70),
    ("SAE JumpReLU", ">0.5"): (144, 92, 92, 94, 91),
    ("SAE JumpReLU", "0.1-0.3"): (139, 9, 12, 7, 9),
    ("SAE ReLU+L1", "0.3-0.5"): (177, 67, 71, 73, 62),
    ("SAE ReLU+L1", ">0.5"): (143, 94, 94, 96, 96),
    ("SAE ReLU+L1", "0.1-0.3"): (126, 15, 17, 17, 13),
    ("random directions (floor)", "0.3-0.5"): (16, 38, 44, 38, 38),
    ("random directions (floor)", "0.1-0.3"): (284, 17, 18, 23, 16),
}
S14_EXPECTED_RANGES = {  # §14.4 / 14.5 / 14.6: [0.1-0.3, 0.3-0.5, >0.5]
    "exact": {
        "keyword (lexical ceiling)": ["50–67 (6)", "77–96 (26)", "83–100 (6)"],
        "SAE GemmaScope": ["0 (100)", "6–12 (50)", "–"],
        "diff-in-means (supervised)": ["6–11 (18)", "19–29 (21)", "14–29 (7)"],
        "probe LR (supervised)": ["12 (17)", "12–17 (24)", "20 (5)"],
        "PCA": ["0–3 (34)", "–", "–"],
        "SAE JumpReLU": ["0 (139)", "15–21 (175)", "42–60 (144)"],
        "SAE ReLU+L1": ["0–2 (126)", "12–23 (177)", "47–62 (143)"],
        "random directions (floor)": ["2–3 (284)", "0 (16)", "–"],
    },
    "yp": {
        "keyword (lexical ceiling)": ["100 (6)", "100 (26)", "100 (6)"],
        "SAE GemmaScope": ["18–72 (100)", "80–92 (50)", "–"],
        "diff-in-means (supervised)": ["28–56 (18)", "38–67 (21)", "57–86 (7)"],
        "probe LR (supervised)": ["24–71 (17)", "38–46 (24)", "100 (5)"],
        "PCA": ["32–91 (34)", "–", "–"],
        "SAE JumpReLU": ["57–88 (139)", "92–98 (175)", "99–100 (144)"],
        "SAE ReLU+L1": ["56–89 (126)", "95–99 (177)", "98–100 (143)"],
        "random directions (floor)": ["65–93 (284)", "94–100 (16)", "–"],
    },
    "none": {
        "keyword (lexical ceiling)": ["0–33 (6)", "0–4 (26)", "0 (6)"],
        "SAE GemmaScope": ["97–99 (100)", "30–38 (50)", "–"],
        "diff-in-means (supervised)": ["61–78 (18)", "52–76 (21)", "57 (7)"],
        "probe LR (supervised)": ["82–88 (17)", "79–83 (24)", "20–60 (5)"],
        "PCA": ["74–82 (34)", "–", "–"],
        "SAE JumpReLU": ["78–86 (139)", "13–22 (175)", "4–8 (144)"],
        "SAE ReLU+L1": ["73–81 (126)", "14–28 (177)", "1–4 (143)"],
        "random directions (floor)": ["61–71 (284)", "12–44 (16)", "–"],
    },
}
# §14.5 lowest-band per-judge sub-table: S, D, G, M
S14_EXPECTED_YP_LOW = {
    "random directions (floor)": (79, 93, 70, 65),
    "SAE JumpReLU": (65, 88, 57, 59),
    "SAE ReLU+L1": (63, 89, 56, 56),
}
S14_EXPECTED_FISHER = {  # §14.2: judge -> (pct, p as printed)
    "SAE JumpReLU": {
        "Sonnet": (73, "0.0074"),
        "GPT-5-mini": (75, "0.0030"),
        "Gemini": (70, "0.0125"),
    },
    "SAE ReLU+L1": {
        "Sonnet": (67, "0.0266"),
        "GPT-5-mini": (73, "0.0076"),
        "Gemini": (62, "0.0681"),
    },
    "SAE GemmaScope": {
        "Sonnet": (52, "0.394"),
        "GPT-5-mini": (52, "0.394"),
        "Gemini": (50, "0.407"),
    },
}
S14_EXPECTED_ANCHOR = {  # §14.7c: S, M, G, D retained %; n
    "SAE jumprelu (pub 380)": ({"Sonnet": 88, "Gemini": 73, "GPT-5-mini": 58, "DeepSeek": 45}, 85),
    "SAE vanilla (380)": ({"Sonnet": 85, "Gemini": 52, "GPT-5-mini": 57, "DeepSeek": 35}, 88),
}
S14_EXPECTED_BINARY = {  # §14.7d: judge -> "binary / partial->yes"; n
    "SAE jumprelu (pub 380)": (
        {
            "Sonnet": "32.4 / 14.7",
            "DeepSeek": "22.4 / 7.1",
            "GPT-5-mini": "33.7 / 18.9",
            "Gemini": "31.1 / 14.7",
        },
        380,
    ),
    "SAE vanilla (380)": (
        {
            "Sonnet": "33.4 / 16.3",
            "DeepSeek": "21.3 / 6.3",
            "GPT-5-mini": "32.4 / 17.2",
            "Gemini": "26.6 / 9.2",
        },
        380,
    ),
    "random (300)": (
        {
            "Sonnet": "9.3 / 9.4",
            "DeepSeek": "7.0 / 6.8",
            "GPT-5-mini": "9.3 / 9.8",
            "Gemini": "10.7 / 11.5",
        },
        300,
    ),
}
S14_EXPECTED_SLATE = {  # §14.7e: (source, band) -> (n, S, D, G, M)
    ("SAE jumprelu (pub 380)", "0.4-0.5"): (136, "-8.8", "-11.0", "-11.0", "-10.3"),
    ("SAE jumprelu (strat 200)", "0.3-0.4"): (40, "-10.0", "-22.5", "+0.0", "-10.0"),
    ("SAE jumprelu (strat 200)", "0.4-0.5"): (40, "-7.5", "-5.0", "-17.5", "-15.0"),
    ("keyword (39)", "0.3-0.4"): (13, "-7.7", "-7.7", "-7.7", "+0.0"),
    ("keyword (39)", "0.4-0.5"): (13, "+0.0", "+15.4", "+0.0", "+0.0"),
    ("diff-in-means (46)", "0.3-0.4"): (10, "-10.0", "-10.0", "-10.0", "-20.0"),
    ("diff-in-means (46)", "0.4-0.5"): (11, "+0.0", "+9.1", "-9.1", "+0.0"),
    ("probe LR (46)", "0.3-0.4"): (14, "-7.1", "-7.1", "+0.0", "+0.0"),
    ("probe LR (46)", "0.4-0.5"): (10, "+0.0", "+0.0", "+0.0", "+0.0"),
}

needs_fixtures = pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason="real verdict CSVs absent; run scripts/build_concordance_results.py --fetch",
)


@pytest.fixture(scope="module")
def real():
    cfg = ct.load_sources(SOURCES)
    return cfg, ct.Store(FIXTURES, cfg["judges"])


# Published gate cells that the CURRENT volume data cannot reproduce, with the reason.
# F9 was committed 2026-08-29 22:38 IST (4880a1c). The vanilla Sonnet de-anchored
# verdicts it was built from were overwritten by a re-run on 2026-08-30 18:23 IST
# (volume mtime of vanilla_test_split/deanchored_eval/sonnet-4-6/). RESULTS.md
# §14.7c (committed 2026-09-01) already quotes the re-run -- 85% retained, which this
# builder reproduces -- so F9's row is stale, not the builder. Any cell NOT listed
# here that stops matching fails the test.
KNOWN_STALE = {
    ("F9_anchoring", ("SAE vanilla (380)", "Sonnet", "de-anchored (no r)"), "0.5-0.6"): (
        "26 (70)",
        "29 (70)",
    ),
    (
        "F9_anchoring",
        ("SAE vanilla (380)", "Sonnet", "de-anchored (no r)"),
        "anchored-YES retained",
    ): ("83%", "85%"),
    ("F9_anchoring", ("SAE vanilla (380)", "Sonnet", "de-anchored (no r)"), "UNKNOWN %"): (
        "4.5",
        "3.9",
    ),
}

# Non-gate published cells that do not reproduce. The published GemmaScope Sonnet
# concordance counts differ from the current gemmascope_test_split verdicts by two
# features at |r| 0.1-0.2 (19 vs 21 YES+PARTIAL of 97) and one at 0.5-0.6; F6's
# YES+PARTIAL gap moves by the same features. Source of the published counts
# unresolved (the CSV predates the F-table commit). Locked so any NEW drift fails.
KNOWN_NONGATE = {
    ("F2_yespartial", ("SAE gemmascope (154)", "general-purpose", "Sonnet"), "0.1-0.2"),
    ("F2_yespartial", ("SAE gemmascope (154)", "general-purpose", "Sonnet"), "0.5-0.6"),
    ("F3_no", ("SAE gemmascope (154)", "general-purpose", "Sonnet"), "0.1-0.2"),
    ("F3_no", ("SAE gemmascope (154)", "general-purpose", "Sonnet"), "0.5-0.6"),
    ("F6_interjudge", ("SAE gemmascope (113)", "YES+PARTIAL"), "0.1-0.2"),
    ("F6_interjudge", ("SAE gemmascope (113)", "YES+PARTIAL"), "0.5-0.6"),
}


@needs_fixtures
def test_reproduces_published_tables(real):
    cfg, _ = real
    tables = ct.build_all(cfg, FIXTURES)
    tally, mismatches = compare_published(tables)
    assert all(total > 0 for _, total in tally.values())
    gate_bad = {
        (m["table"], m["row"], m["col"]): (m["published"], m["computed"])
        for m in mismatches
        if m["gate"]
    }
    assert gate_bad == KNOWN_STALE, "\n".join(f"{k}: {v}" for k, v in gate_bad.items())
    n_ok = sum(ok for ok, _ in tally.values())
    n_total = sum(total for _, total in tally.values())
    assert n_ok == n_total - len(KNOWN_STALE)
    other = {(m["table"], m["row"], m["col"]) for m in mismatches if not m["gate"]}
    assert other == KNOWN_NONGATE, sorted(other ^ KNOWN_NONGATE)


@needs_fixtures
def test_reproduces_section14_four_judge_rows(real):
    cfg, store = real
    judges = ct.S14_JUDGES

    hit = ct.s14_hit1_same(cfg, store)
    for (label, band), (n, *vals) in S14_EXPECTED_HIT1.items():
        per = hit[label][band]
        assert {nn for _, nn in per.values()} == {n}, (label, band)
        assert tuple(pct(*per[j]) for j in judges) == tuple(vals), (label, band)

    for key, fn in [
        ("exact", ct.s14_exact_yes),
        ("yp", ct.s14_yes_partial),
        ("none", ct.s14_none_same),
    ]:
        table = fn(cfg, store)
        for label, cells in S14_EXPECTED_RANGES[key].items():
            assert [range_cell(table[label][b]) for b in BANDS3] == cells, (key, label)

    yp = ct.s14_yes_partial(cfg, store)
    for label, vals in S14_EXPECTED_YP_LOW.items():
        assert tuple(pct(*yp[label]["0.1-0.3"][j]) for j in judges) == vals, label

    fisher = ct.s14_fisher(cfg, store)
    for label, per in S14_EXPECTED_FISHER.items():
        for judge, (p_pct, p_str) in per.items():
            got_pct, got_p = fisher[label][judge]
            assert (got_pct, ct.format_p(got_p)) == (p_pct, p_str), (label, judge)

    anchor = ct.s14_anchoring(cfg, store)
    for label, (per, n) in S14_EXPECTED_ANCHOR.items():
        for judge, want in per.items():
            assert anchor[label][judge] == (want, n), (label, judge)

    binary = ct.s14_binary(cfg, store)
    for label, (per, n) in S14_EXPECTED_BINARY.items():
        for judge, want in per.items():
            b, ptoy, nn = binary[label][judge]
            assert (f"{b:.1f} / {ptoy:.1f}", nn) == (want, n), (label, judge)

    slate = {(r["source"], r["band"]): r for r in ct.s14_slate_delta(cfg, store)}
    for key, (n, *vals) in S14_EXPECTED_SLATE.items():
        row = slate[key]
        assert row["n"] == n, key
        assert tuple(f"{row[j]:+.1f}" for j in judges) == tuple(vals), key
