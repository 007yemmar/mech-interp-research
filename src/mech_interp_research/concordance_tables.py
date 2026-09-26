"""§14 concordance-validation tables, recomputed from per-judge verdict CSVs.

Rebuilds ``results/concordance_validation/F*.csv`` and the four-judge tables of
RESULTS.md §14 from the raw judge outputs that the auto_interp / arm0 / retrieval /
de-anchored / binary stages write under each ``auto_interp/<pool>/`` directory. The
point is to make those tables a function of committed code rather than a notebook,
so a new SAE can be added as one more source row (``configs/concordance_sources.yaml``)
and land next to the published rows under exactly the same definitions.

Definitions (each one verified against the published tables, see
``tests/test_concordance_tables.py::test_reproduces_published_tables``):

* **Bands** are disjoint and left-closed on ``|r|``: ``[0.1,0.2) … [0.5,0.6)``,
  ``[0.6, ∞)`` (six-band F-tables), and ``[0.1,0.3) [0.3,0.5) [0.5,∞)`` (§14 and
  F8-coarse). ``|r|`` is ``|concordance_r_pb|`` for concordance verdicts and the
  file's own ``abs_r_pb`` for every other arm.
* **Percentages** are ``round(100*k/n)`` with Python's round-half-to-even (the
  published tables show 52.5 → 52, 87.5 → 88). A cell with ``n == 0`` is ``–``.
* **Denominators** include every judged feature: UNKNOWN / unparseable verdicts
  count as "not YES" (and "not YES/PARTIAL", "not NO"), never dropped.
* **Multi-pool sources** (§14 SAE rows: published 380 + stratified 200) are the
  union by ``feature_idx``; a feature present in both keeps its **first** pool's
  verdict (the published pool). Every judge and arm follows the same rule.
* **Fisher** tests are two-sided ``scipy.stats.fisher_exact`` on the 2×2 of
  (hits, misses) for the source vs the random floor in the same band, condition
  and judge. Stars: ``***`` p<.001, ``**`` <.01, ``*`` <.05, else ``ns``.

No Modal or network access here; the CSVs are read from a local fixtures directory
that mirrors volume paths (see ``scripts/build_concordance_results.py --fetch``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mech_interp_research.config_vars import load_config

BANDS6 = ["0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6+"]
BANDS3 = ["0.1-0.3", "0.3-0.5", ">0.5"]
_EDGES6 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_EDGES3 = [0.1, 0.3, 0.5]

DASH = "–"

# arm directory, file name and the normalised columns each kind of verdict file yields
ARMS = {
    "concordance": ("arm0_eval", "concordance_results.csv"),
    "retrieval": ("retrieval_eval", "retrieval_verdicts.csv"),
    "retrieval_hardneg": ("retrieval_eval_hardneg", "retrieval_verdicts.csv"),
    "deanchored": ("deanchored_eval", "deanchored_verdicts.csv"),
    "binary": ("binary_eval", "binary_verdicts.csv"),
}
AUTO_INTERP = "auto_interp"  # judge slug meaning "<pool>/concordance_results.csv"

# F-tables carry two judges; §14 carries four.
F_JUDGES = ["Sonnet", "DeepSeek"]
S14_JUDGES = ["Sonnet", "DeepSeek", "GPT-5-mini", "Gemini"]
S14_ABBREV = {"Sonnet": "S", "DeepSeek": "D", "GPT-5-mini": "G", "Gemini": "M"}
CONDITIONS = {"cross-chapter": "retrieval", "same-chapter": "retrieval_hardneg"}


# --------------------------------------------------------------------------- primitives


def _band(abs_r: float, edges: list[float], labels: list[str]) -> str | None:
    a = abs(float(abs_r))
    if pd.isna(a) or a < edges[0]:
        return None
    label = labels[0]
    for edge, lab in zip(edges, labels, strict=True):
        if a >= edge:
            label = lab
    return label


def band_of(abs_r: float) -> str | None:
    """Six disjoint left-closed bands; None below 0.1."""
    return _band(abs_r, _EDGES6, BANDS6)


def band3_of(abs_r: float) -> str | None:
    """Three disjoint left-closed bands (§14 / F8-coarse); None below 0.1."""
    return _band(abs_r, _EDGES3, BANDS3)


def pct(k: int, n: int) -> int:
    """Rounded percentage, Python round-half-to-even (matches the published tables)."""
    return int(round(100 * k / n))


def cell(k: int, n: int) -> str:
    """``"pct (n)"`` or ``–`` when there is nothing to report."""
    return DASH if n == 0 else f"{pct(k, n)} ({n})"


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for hits k1/n1 vs k2/n2."""
    from scipy.stats import fisher_exact

    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def banded_counts(
    df: pd.DataFrame, flag: pd.Series, bands: str = "6"
) -> dict[str, tuple[int, int]]:
    """``{band: (k, n)}`` of a boolean ``flag`` over ``df['abs_r']`` bands."""
    fn, labels = (band_of, BANDS6) if bands == "6" else (band3_of, BANDS3)
    b = df["abs_r"].map(fn)
    flag = pd.Series(flag, index=df.index).astype(bool)
    return {lab: (int(flag[b == lab].sum()), int((b == lab).sum())) for lab in labels}


# ----------------------------------------------------------------------------- loading


def _volume_rel(path: str) -> str:
    path = str(path).strip()
    if path.startswith("/out/"):
        path = path[len("/out/") :]
    return path.strip("/")


def load_sources(path: str | Path) -> dict:
    """Load the sources config, resolving ``${var}`` and normalising pool paths."""
    cfg = load_config(path)
    for src in cfg["sources"]:
        src["pool_dirs"] = [_volume_rel(p) for p in src["pool_dirs"]]
    return cfg


def _normalise(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reduce a raw verdict CSV to the columns the metrics need (no free text)."""
    out = pd.DataFrame({"feature_idx": df["feature_idx"].astype(int)})
    if kind == "concordance":
        out["abs_r"] = df["concordance_r_pb"].astype(float).abs()
        out["verdict"] = df["concordance_verdict"].astype(str).str.strip().str.upper()
    elif kind.startswith("retrieval"):
        out["abs_r"] = df["abs_r_pb"].astype(float).abs()
        out["hit1"] = df["hit1"].astype(int)
        out["is_none"] = df["is_none"].astype(int)
    elif kind == "deanchored":
        out["abs_r"] = df["abs_r_pb"].astype(float).abs()
        out["anchored"] = df["anchored_verdict"].astype(str).str.strip().str.upper()
        out["deanchored"] = df["deanchored_verdict"].astype(str).str.strip().str.upper()
    elif kind == "binary":
        out["abs_r"] = df["abs_r_pb"].astype(float).abs()
        out["original"] = df["original_verdict"].astype(str).str.strip().str.upper()
        out["binary"] = df["binary_verdict"].astype(str).str.strip().str.upper()
    else:  # pragma: no cover - guarded by ARMS
        raise ValueError(kind)
    return out


def _pool_file(fixtures: Path, pool: str, kind: str, slug: str) -> Path:
    if kind == "concordance" and slug == AUTO_INTERP:
        return fixtures / pool / "concordance_results.csv"
    arm, fname = ARMS[kind]
    return fixtures / pool / arm / slug / fname


def merge_pools(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union by feature_idx; the first pool listed wins for a shared feature."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates("feature_idx", keep="first")


@dataclass
class Store:
    """Lazy, cached loader of normalised verdict frames from a fixtures dir."""

    fixtures: Path
    judges: dict
    _cache: dict = field(default_factory=dict)

    def pool(self, pool: str, kind: str, slugs: list[str]) -> pd.DataFrame | None:
        """First existing file among ``slugs`` (a preference list) for one pool."""
        for slug in slugs:
            key = (pool, kind, slug)
            if key not in self._cache:
                path = _pool_file(self.fixtures, pool, kind, slug)
                self._cache[key] = _normalise(pd.read_csv(path), kind) if path.is_file() else None
            if self._cache[key] is not None:
                return self._cache[key]
        return None

    def get(
        self, pool_dirs: list[str], kind: str, judge: str, table: str = "f"
    ) -> pd.DataFrame | None:
        """Merged frame for a source's pools, one judge, one arm.

        ``table`` selects the judge-slug mapping: ``"f"`` for the two-judge F-tables,
        ``"s14"`` for §14 (which may prefer a different Sonnet verdict, see config).
        """
        jcfg = self.judges[judge]
        akey = "concordance" if kind == "concordance" else kind.replace("_hardneg", "")
        slugs = jcfg.get(f"{akey}_s14", jcfg[akey]) if table == "s14" else jcfg[akey]
        slugs = [slugs] if isinstance(slugs, str) else list(slugs)
        return merge_pools([self.pool(p, kind, slugs) for p in pool_dirs])


def _in_table(src: dict, table: str) -> bool:
    return "tables" not in src or table in src["tables"]


# ------------------------------------------------------------------------ F-tables


def _rate_rows(cfg: dict, store: Store, table: str, kind: str, flag_fn) -> pd.DataFrame:
    rows = []
    for src in cfg["sources"]:
        if not _in_table(src, table):
            continue
        for judge in F_JUDGES:
            df = store.get(src["pool_dirs"], kind, judge)
            if df is None:
                continue
            counts = banded_counts(df, flag_fn(df))
            rows.append(
                {"Source": src["label"], "Type": src["type"], "Judge": judge}
                | {b: cell(*counts[b]) for b in BANDS6}
            )
    return pd.DataFrame(rows, columns=["Source", "Type", "Judge", *BANDS6])


def f1_exact_yes(cfg: dict, store: Store) -> pd.DataFrame:
    return _rate_rows(cfg, store, "F1", "concordance", lambda d: d["verdict"] == "YES")


def f2_yes_partial(cfg: dict, store: Store) -> pd.DataFrame:
    return _rate_rows(
        cfg, store, "F2", "concordance", lambda d: d["verdict"].isin(["YES", "PARTIAL"])
    )


def f3_no(cfg: dict, store: Store) -> pd.DataFrame:
    return _rate_rows(cfg, store, "F3", "concordance", lambda d: d["verdict"] == "NO")


def f4_hit1(cfg: dict, store: Store, same_chapter: bool = False) -> pd.DataFrame:
    table, kind = ("F4b", "retrieval_hardneg") if same_chapter else ("F4", "retrieval")
    return _rate_rows(cfg, store, table, kind, lambda d: d["hit1"] == 1)


def f5_none(cfg: dict, store: Store, same_chapter: bool = False) -> pd.DataFrame:
    table, kind = ("F5b", "retrieval_hardneg") if same_chapter else ("F5", "retrieval")
    return _rate_rows(cfg, store, table, kind, lambda d: d["is_none"] == 1)


def judge_gap(a: dict[str, tuple[int, int]], b: dict[str, tuple[int, int]]) -> dict[str, str]:
    """Per band ``|rate_a − rate_b|`` in points, rounded; ``–`` if either is empty.

    This is what F6 ("inter-judge") reports: the gap between the two judges' banded
    rates, taken on the unrounded rates, not a per-feature disagreement rate.
    """
    out = {}
    for band, (k1, n1) in a.items():
        k2, n2 = b[band]
        out[band] = (
            DASH if n1 == 0 or n2 == 0 else str(int(round(abs(100 * k1 / n1 - 100 * k2 / n2))))
        )
    return out


def f6_interjudge(cfg: dict, store: Store) -> pd.DataFrame:
    """Sonnet-vs-DeepSeek gap in points: concordance metrics first, then hit@1."""
    rows_c, rows_r = [], []
    conc_metrics = [
        ("exact-YES", lambda d: d["verdict"] == "YES"),
        ("YES+PARTIAL", lambda d: d["verdict"].isin(["YES", "PARTIAL"])),
    ]
    for src in cfg["sources"]:
        if not _in_table(src, "F6"):
            continue
        label = src.get("label_f6", src["label"])
        a = store.get(src["pool_dirs"], "concordance", "Sonnet")
        b = store.get(src["pool_dirs"], "concordance", "DeepSeek")
        if a is not None and b is not None:
            for metric, fn in conc_metrics:
                gap = judge_gap(banded_counts(a, fn(a)), banded_counts(b, fn(b)))
                rows_c.append({"Source": label, "Metric": metric} | gap)
        for cond, kind in [("cross-ch.", "retrieval"), ("same-ch.", "retrieval_hardneg")]:
            a = store.get(src["pool_dirs"], kind, "Sonnet")
            b = store.get(src["pool_dirs"], kind, "DeepSeek")
            if a is None or b is None:
                continue
            gap = judge_gap(banded_counts(a, a["hit1"] == 1), banded_counts(b, b["hit1"] == 1))
            rows_r.append({"Source": label, "Metric": f"hit@1 ({cond})"} | gap)
    return pd.DataFrame(rows_c + rows_r, columns=["Source", "Metric", *BANDS6])


def f7_binary(cfg: dict, store: Store) -> pd.DataFrame:
    """Forced-binary YES %, and of originally-PARTIAL features the % now YES."""
    rows = []
    for src in cfg["sources"]:
        if not _in_table(src, "F7"):
            continue
        for judge in F_JUDGES:
            df = store.get(src["pool_dirs"], "binary", judge)
            if df is None:
                continue
            label = src.get("label_short", src["label"])
            yes = banded_counts(df, df["binary"] == "YES")
            part = df[df["original"] == "PARTIAL"]
            pty = banded_counts(part, part["binary"] == "YES")
            base = {"Source": label, "Judge": judge}
            rows.append(base | {"Metric": "binary-YES"} | {b: cell(*yes[b]) for b in BANDS6})
            rows.append(base | {"Metric": "PARTIAL→YES"} | {b: cell(*pty[b]) for b in BANDS6})
    return pd.DataFrame(rows, columns=["Source", "Judge", "Metric", *BANDS6])


def _fisher_ref(cfg: dict) -> dict:
    refs = [s for s in cfg["sources"] if s.get("fisher_ref")]
    if len(refs) != 1:
        raise ValueError("exactly one source must set fisher_ref: true")
    return refs[0]


def f8_significance(cfg: dict, store: Store, coarse: bool = False) -> pd.DataFrame:
    """hit@1 difference vs the random floor, with two-sided Fisher stars.

    Fine: ``"{src% − ref%:+d} {stars}"`` per six-band cell (difference of the two
    already-rounded percentages). Coarse: ``"{src%} vs {ref%}  {diff:+d} {stars}
    (n={n_src})"`` per three-band cell. ``–`` whenever either side is empty.
    """
    ref = _fisher_ref(cfg)
    bands, bmode, table = (BANDS3, "3", "F8c") if coarse else (BANDS6, "6", "F8")
    rows = []
    for src in cfg["sources"]:
        if src is ref or not _in_table(src, table):
            continue
        label = src.get("label_f8c" if coarse else "label_f8", src["label"])
        for cond, kind in CONDITIONS.items():
            for judge in F_JUDGES:
                df = store.get(src["pool_dirs"], kind, judge)
                rf = store.get(ref["pool_dirs"], kind, judge)
                if df is None or rf is None:
                    continue
                c = banded_counts(df, df["hit1"] == 1, bmode)
                r = banded_counts(rf, rf["hit1"] == 1, bmode)
                row = {"Source": label, "Condition": cond, "Judge": judge}
                for b in bands:
                    (k1, n1), (k2, n2) = c[b], r[b]
                    if n1 == 0 or n2 == 0:
                        row[b] = DASH
                        continue
                    p = fisher_p(k1, n1, k2, n2)
                    diff = pct(k1, n1) - pct(k2, n2)
                    if coarse:
                        row[b] = f"{pct(k1, n1)} vs {pct(k2, n2)}  {diff:+d} {stars(p)} (n={n1})"
                    else:
                        row[b] = f"{diff:+d} {stars(p)}"
                rows.append(row)
    return pd.DataFrame(rows, columns=["Source", "Condition", "Judge", *bands])


def f9_anchoring(cfg: dict, store: Store) -> pd.DataFrame:
    """Anchored vs de-anchored exact-YES, % of anchored YES retained, % UNKNOWN."""
    rows = []
    for src in cfg["sources"]:
        if not _in_table(src, "F9"):
            continue
        for judge in F_JUDGES:
            df = store.get(src["pool_dirs"], "deanchored", judge)
            if df is None:
                continue
            label = src.get("label_short", src["label"])
            anc = banded_counts(df, df["anchored"] == "YES")
            dea = banded_counts(df, df["deanchored"] == "YES")
            base = {"Source": label, "Judge": judge}
            rows.append(
                base
                | {"Prompt": "anchored (r stated)"}
                | {b: cell(*anc[b]) for b in BANDS6}
                | {"anchored-YES retained": "", "UNKNOWN %": ""}
            )
            k, n = retention(df)
            unk = 100 * (df["deanchored"] == "UNKNOWN").sum() / len(df)
            rows.append(
                base
                | {"Prompt": "de-anchored (no r)"}
                | {b: cell(*dea[b]) for b in BANDS6}
                | {"anchored-YES retained": f"{pct(k, n)}%", "UNKNOWN %": f"{unk:.1f}"}
            )
    return pd.DataFrame(
        rows,
        columns=["Source", "Judge", "Prompt", *BANDS6, "anchored-YES retained", "UNKNOWN %"],
    )


def retention(df: pd.DataFrame) -> tuple[int, int]:
    """(# anchored-YES features still YES without r, # anchored-YES features)."""
    yes = df["anchored"] == "YES"
    return int((yes & (df["deanchored"] == "YES")).sum()), int(yes.sum())


TABLE_FILES = {
    "F1_exactYES": f1_exact_yes,
    "F2_yespartial": f2_yes_partial,
    "F3_no": f3_no,
    "F4_hit1_crosschapter": lambda c, s: f4_hit1(c, s, same_chapter=False),
    "F4b_hit1_samechapter": lambda c, s: f4_hit1(c, s, same_chapter=True),
    "F5_none_crosschapter": lambda c, s: f5_none(c, s, same_chapter=False),
    "F5b_none_samechapter": lambda c, s: f5_none(c, s, same_chapter=True),
    "F6_interjudge": f6_interjudge,
    "F7_binary": f7_binary,
    "F8_significance": lambda c, s: f8_significance(c, s, coarse=False),
    "F8_significance_coarse": lambda c, s: f8_significance(c, s, coarse=True),
    "F9_anchoring": f9_anchoring,
}


def build_all(cfg: dict, fixtures: str | Path) -> dict[str, pd.DataFrame]:
    store = Store(Path(fixtures), cfg["judges"])
    return {name: fn(cfg, store) for name, fn in TABLE_FILES.items()}


# ----------------------------------------------------------------------- §14 tables


def _group_pools(cfg: dict, group: dict) -> list[str]:
    by_key = {s["key"]: s for s in cfg["sources"]}
    return [p for key in group["members"] for p in by_key[key]["pool_dirs"]]


MIN_N_S14 = 5  # §14: "– means fewer than 5 features exist in that band"


def s14_rates(cfg: dict, store: Store, kind: str, flag_fn) -> dict:
    """``{group_label: {band3: {judge: (k, n)}}}`` for the §14 groups."""
    out = {}
    for group in cfg["groups"]:
        pools = _group_pools(cfg, group)
        per = {b: {} for b in BANDS3}
        for judge in S14_JUDGES:
            df = store.get(pools, kind, judge, table="s14")
            if df is None:
                continue
            counts = banded_counts(df, flag_fn(df), "3")
            for b in BANDS3:
                per[b][judge] = counts[b]
        out[group["label"]] = per
    return out


def s14_hit1_same(cfg: dict, store: Store) -> dict:
    """§14.1: hit@1 %, same-chapter distractors."""
    return s14_rates(cfg, store, "retrieval_hardneg", lambda d: d["hit1"] == 1)


def s14_exact_yes(cfg: dict, store: Store) -> dict:
    """§14.4: exact-YES %."""
    return s14_rates(cfg, store, "concordance", lambda d: d["verdict"] == "YES")


def s14_yes_partial(cfg: dict, store: Store) -> dict:
    """§14.5: YES+PARTIAL %."""
    return s14_rates(cfg, store, "concordance", lambda d: d["verdict"].isin(["YES", "PARTIAL"]))


def s14_none_same(cfg: dict, store: Store) -> dict:
    """§14.6: "none of these" %, same-chapter."""
    return s14_rates(cfg, store, "retrieval_hardneg", lambda d: d["is_none"] == 1)


def range_cell(per_judge: dict[str, tuple[int, int]]) -> str:
    """``min–max (n)`` across judges (single value when they agree); ``–`` if n < 5."""
    if not per_judge:
        return DASH
    n = max(n for _, n in per_judge.values())
    if n < MIN_N_S14:
        return DASH
    vals = [pct(k, n_) for k, n_ in per_judge.values()]
    lo, hi = min(vals), max(vals)
    return f"{lo} ({n})" if lo == hi else f"{lo}–{hi} ({n})"


def s14_fisher(cfg: dict, store: Store, band: str = "0.3-0.5") -> dict:
    """§14.2: ``{group: {judge: (pct, p)}}`` hit@1 same-chapter vs the random floor."""
    rates = s14_hit1_same(cfg, store)
    ref = [g for g in cfg["groups"] if g.get("fisher_ref")]
    if len(ref) != 1:
        raise ValueError("exactly one group must set fisher_ref: true")
    ref_rates = rates[ref[0]["label"]][band]
    out = {}
    for group in cfg["groups"]:
        if group is ref[0]:
            continue
        out[group["label"]] = {}
        for judge, (k, n) in rates[group["label"]][band].items():
            if judge not in ref_rates or n < MIN_N_S14 or ref_rates[judge][1] == 0:
                continue
            k2, n2 = ref_rates[judge]
            out[group["label"]][judge] = (pct(k, n), fisher_p(k, n, k2, n2))
    return out


def s14_anchoring(cfg: dict, store: Store) -> dict:
    """§14.7c: ``{source_label: {judge: (retained_pct, n_anchored_yes)}}``."""
    out = {}
    for src in cfg["sources"]:
        per = {}
        for judge in S14_JUDGES:
            df = store.get(src["pool_dirs"], "deanchored", judge, table="s14")
            if df is None:
                continue
            k, n = retention(df)
            per[judge] = (pct(k, n), n)
        if per:
            out[src["label"]] = per
    return out


def s14_binary(cfg: dict, store: Store) -> dict:
    """§14.7d: ``{source_label: {judge: (binary_yes_pct1, partial_to_yes_pct1, n)}}``."""
    out = {}
    for src in cfg["sources"]:
        per = {}
        for judge in S14_JUDGES:
            df = store.get(src["pool_dirs"], "binary", judge, table="s14")
            if df is None:
                continue
            part = df[df["original"] == "PARTIAL"]
            per[judge] = (
                100 * (df["binary"] == "YES").mean(),
                100 * (part["binary"] == "YES").mean() if len(part) else float("nan"),
                len(df),
            )
        if per:
            out[src["label"]] = per
    return out


def s14_slate_delta(
    cfg: dict, store: Store, bands: tuple[str, ...] = ("0.3-0.4", "0.4-0.5")
) -> list:
    """§14.7e: same-chapter − cross-chapter hit@1 (pp) per source × six-band.

    Only emitted where every §14 judge has both conditions for that source.
    """
    rows = []
    for src in cfg["sources"]:
        deltas = {}
        n_band = {}
        for judge in S14_JUDGES:
            x = store.get(src["pool_dirs"], "retrieval", judge, table="s14")
            h = store.get(src["pool_dirs"], "retrieval_hardneg", judge, table="s14")
            if x is None or h is None:
                break
            cx, ch = banded_counts(x, x["hit1"] == 1), banded_counts(h, h["hit1"] == 1)
            for b in bands:
                if cx[b][1] and ch[b][1]:
                    deltas.setdefault(b, {})[judge] = (
                        100 * ch[b][0] / ch[b][1] - 100 * cx[b][0] / cx[b][1]
                    )
                    n_band[b] = ch[b][1]
        else:
            for b in bands:
                if b in deltas and len(deltas[b]) == len(S14_JUDGES):
                    rows.append({"source": src["label"], "band": b, "n": n_band[b]} | deltas[b])
    return rows


def section14_markdown(cfg: dict, fixtures: str | Path) -> str:
    """Render the §14 four-judge tables as markdown (for pasting / diffing)."""
    store = Store(Path(fixtures), cfg["judges"])
    lines = ["# §14 four-judge tables (generated by build_concordance_results.py)", ""]
    jh = " | ".join(S14_ABBREV[j] for j in S14_JUDGES)

    hit = s14_hit1_same(cfg, store)
    lines += ["## 14.1 hit@1, same-chapter", ""]
    for b in BANDS3:
        lines += [
            f"**Band |r| {b}**",
            "",
            f"| source | n | {jh} | spread |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, per in hit.items():
            pj = per[b]
            n = max((n for _, n in pj.values()), default=0)
            if n < MIN_N_S14:
                lines.append(f"| {label} | {n} | – | – | – | – | – |")
                continue
            vals = [pct(*pj[j]) if j in pj else None for j in S14_JUDGES]
            got = [v for v in vals if v is not None]
            cells = " | ".join("" if v is None else str(v) for v in vals)
            lines.append(f"| {label} | {n} | {cells} | {max(got) - min(got)} pp |")
        lines.append("")

    lines += ["## 14.2 Fisher vs random floor, |r| 0.3-0.5, same-chapter", ""]
    lines += [f"| source | {' | '.join(S14_JUDGES)} |", "|---|---|---|---|---|"]
    for label, per in s14_fisher(cfg, store).items():
        cells = " | ".join(
            f"{per[j][0]}%, p = {format_p(per[j][1])}" if j in per else "" for j in S14_JUDGES
        )
        lines.append(f"| {label} | {cells} |")
    lines.append("")

    for title, table in [
        ("14.4 exact-YES (min–max across judges)", s14_exact_yes(cfg, store)),
        ("14.5 YES+PARTIAL (min–max across judges)", s14_yes_partial(cfg, store)),
        ("14.6 none-of-these, same-chapter (min–max)", s14_none_same(cfg, store)),
    ]:
        lines += [f"## {title}", "", f"| source | {' | '.join(BANDS3)} |", "|---|---|---|---|"]
        for label, per in table.items():
            lines.append(f"| {label} | " + " | ".join(range_cell(per[b]) for b in BANDS3) + " |")
        lines.append("")

    lines += [
        "## 14.7c anchored-YES retained without r",
        "",
        f"| source | {jh} | n |",
        "|---|---|---|---|---|---|",
    ]
    for label, per in s14_anchoring(cfg, store).items():
        n = max(v[1] for v in per.values())
        lines.append(
            f"| {label} | "
            + " | ".join(f"{per[j][0]}%" if j in per else "" for j in S14_JUDGES)
            + f" | {n} |"
        )
    lines.append("")

    lines += [
        "## 14.7d binary-YES % / PARTIAL→YES %",
        "",
        f"| source | n | {jh} |",
        "|---|---|---|---|---|---|",
    ]
    for label, per in s14_binary(cfg, store).items():
        n = max(v[2] for v in per.values())
        cells = " | ".join(
            f"{per[j][0]:.1f} / {per[j][1]:.1f}" if j in per else "" for j in S14_JUDGES
        )
        lines.append(f"| {label} | {n} | {cells} |")
    lines.append("")

    lines += [
        "## 14.7e same-chapter − cross-chapter hit@1 (pp)",
        "",
        f"| source | band | n | {jh} |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in s14_slate_delta(cfg, store):
        cells = " | ".join(f"{row[j]:+.1f}" for j in S14_JUDGES)
        lines.append(f"| {row['source']} | {row['band']} | {row['n']} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def format_p(p: float) -> str:
    """§14.2 style: four decimals below 0.1, three above."""
    return f"{p:.4f}" if p < 0.1 else f"{p:.3f}"
