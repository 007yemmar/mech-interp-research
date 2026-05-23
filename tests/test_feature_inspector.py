"""Tests for latent feature inspector pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sae(d_model: int = 64, d_sae: int = 32, seed: int = 0):
    """Create a minimal JumpReLUSAE for testing."""
    from mech_interp_research.icd_eval import JumpReLUSAE

    rng = np.random.default_rng(seed)
    return JumpReLUSAE(
        W_enc=rng.standard_normal((d_model, d_sae)).astype(np.float32),
        b_enc=np.zeros(d_sae, dtype=np.float32),
        b_dec=np.zeros(d_model, dtype=np.float32),
        threshold=np.zeros(d_sae, dtype=np.float32),
        d_model=d_model,
        d_sae=d_sae,
        W_dec=rng.standard_normal((d_sae, d_model)).astype(np.float32),
    )


def _make_shard_ckpt(
    tmp_path: Path, n_shards: int = 2, n_notes_per_shard: int = 3, d_sae: int = 32
) -> Path:
    """Create synthetic shard_ckpt/ directory with note-level vectors."""
    ckpt_dir = tmp_path / "shard_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    for s in range(n_shards):
        vecs = rng.standard_normal((n_notes_per_shard, d_sae)).astype(np.float32)
        np.save(ckpt_dir / f"shard_{s:04d}_vectors.npy", vecs)

    return ckpt_dir


# ---------------------------------------------------------------------------
# test_scan_shard_top_tokens
# ---------------------------------------------------------------------------


def test_scan_shard_top_tokens(synthetic_run_dir: Path) -> None:
    """scan_shard_for_top_tokens finds top-k tokens with correct positions."""
    from mech_interp_research.feature_inspector import scan_shard_for_top_tokens
    from mech_interp_research.icd_eval import load_metadata

    d_model, d_sae = 64, 32
    sae = _make_sae(d_model, d_sae)
    metadata = load_metadata(synthetic_run_dir)
    meta_shard0 = metadata[metadata["shard"] == 0]

    shard_path = synthetic_run_dir / "shard_0000.safetensors"
    target_indices = [0, 5, 10]
    top_k = 5

    hits, firing_stats = scan_shard_for_top_tokens(
        sae=sae,
        shard_path=shard_path,
        shard_idx=0,
        metadata_for_shard=meta_shard0,
        target_latent_indices=target_indices,
        top_k=top_k,
    )

    for lat_idx in target_indices:
        lat_hits = hits[lat_idx]
        assert len(lat_hits) <= top_k
        for h in lat_hits:
            assert h.shard == 0
            assert h.activation > 0
            assert h.note_idx >= 0
            assert h.position_in_note >= 0
        if len(lat_hits) > 1:
            activations = [h.activation for h in lat_hits]
            assert activations == sorted(activations, reverse=True)


def test_scan_shard_with_spike(tmp_path: Path) -> None:
    """A known spike at a specific position is found as the top hit."""
    from safetensors.numpy import save_file

    from mech_interp_research.feature_inspector import scan_shard_for_top_tokens

    d_model, d_sae = 16, 8
    sae = _make_sae(d_model, d_sae, seed=7)

    n_tokens = 100
    acts = np.zeros((n_tokens, d_model), dtype=np.float32)
    spike_pos = 42
    acts[spike_pos] = sae.W_enc[:, 3] * 100.0

    shard_path = tmp_path / "shard_0000.safetensors"
    save_file({"activations": acts.astype(np.float16)}, str(shard_path))

    meta = pd.DataFrame(
        [
            {"note_idx": 0, "shard": 0, "row_start": 0, "row_end": 50, "n_tokens": 50},
            {"note_idx": 1, "shard": 0, "row_start": 50, "row_end": 100, "n_tokens": 50},
        ]
    )

    hits, _ = scan_shard_for_top_tokens(
        sae=sae,
        shard_path=shard_path,
        shard_idx=0,
        metadata_for_shard=meta,
        target_latent_indices=[3],
        top_k=3,
    )

    assert 3 in hits
    assert len(hits[3]) >= 1
    top_hit = hits[3][0]
    assert top_hit.position_in_shard == spike_pos
    assert top_hit.note_idx == 0
    assert top_hit.position_in_note == 42


# ---------------------------------------------------------------------------
# test_merge_top_tokens
# ---------------------------------------------------------------------------


def test_merge_top_tokens() -> None:
    """merge_top_tokens keeps globally highest activations from two shards."""
    from mech_interp_research.feature_inspector import TokenHit, merge_top_tokens

    shard1 = {
        0: [
            TokenHit(
                shard=0, position_in_shard=10, note_idx=0, position_in_note=10, activation=5.0
            ),
            TokenHit(
                shard=0, position_in_shard=20, note_idx=0, position_in_note=20, activation=3.0
            ),
            TokenHit(shard=0, position_in_shard=30, note_idx=1, position_in_note=5, activation=1.0),
        ]
    }
    shard2 = {
        0: [
            TokenHit(
                shard=1, position_in_shard=15, note_idx=3, position_in_note=15, activation=4.0
            ),
            TokenHit(
                shard=1, position_in_shard=25, note_idx=4, position_in_note=25, activation=2.0
            ),
        ]
    }

    merged = merge_top_tokens([shard1, shard2], top_k=3)

    assert 0 in merged
    assert len(merged[0]) == 3
    assert merged[0][0].activation == 5.0
    assert merged[0][1].activation == 4.0
    assert merged[0][2].activation == 3.0


def test_merge_top_tokens_multiple_latents() -> None:
    """merge_top_tokens handles multiple latent indices independently."""
    from mech_interp_research.feature_inspector import TokenHit, merge_top_tokens

    shard1 = {
        0: [
            TokenHit(shard=0, position_in_shard=1, note_idx=0, position_in_note=1, activation=10.0)
        ],
        5: [TokenHit(shard=0, position_in_shard=2, note_idx=0, position_in_note=2, activation=8.0)],
    }
    shard2 = {
        0: [
            TokenHit(shard=1, position_in_shard=3, note_idx=1, position_in_note=3, activation=12.0)
        ],
    }

    merged = merge_top_tokens([shard1, shard2], top_k=5)

    assert 0 in merged and 5 in merged
    assert len(merged[0]) == 2
    assert merged[0][0].activation == 12.0
    assert len(merged[5]) == 1


# ---------------------------------------------------------------------------
# test_extract_contexts
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer substitute for testing context extraction."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {}
        self._inv = {v: k for k, v in self._vocab.items()}

    def __call__(self, text: str, **kwargs) -> dict:
        tokens = text.split()
        ids = [self._vocab.get(t, hash(t) % 10000) for t in tokens]
        if kwargs.get("max_length"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids}

    def decode(self, ids: list[int], **kwargs) -> str:
        return " ".join(self._inv.get(i, f"<{i}>") for i in ids)


def test_extract_contexts() -> None:
    """extract_token_contexts decodes correct token and context window."""
    from mech_interp_research.feature_inspector import TokenHit, extract_token_contexts

    vocab = {"The": 0, "patient": 1, "has": 2, "thyroid": 3, "disease": 4, "noted": 5}
    tok = _FakeTokenizer(vocab)

    hits = {
        99: [
            TokenHit(
                shard=0, position_in_shard=100, note_idx=0, position_in_note=3, activation=50.0
            ),
        ]
    }
    note_texts = {0: "The patient has thyroid disease noted"}

    extract_token_contexts(hits, note_texts, tok, context_window=2, max_length=8192)

    h = hits[99][0]
    assert h.token_str == "thyroid"
    assert h.token_id == 3
    assert len(h.context_tokens) == 5  # positions 1..5
    assert h.context_tokens[0] == "patient"
    assert h.context_tokens[2] == "thyroid"
    assert h.context_tokens[4] == "noted"


def test_context_boundary_start() -> None:
    """Hit at position 0 does not cause IndexError."""
    from mech_interp_research.feature_inspector import TokenHit, extract_token_contexts

    vocab = {"Hello": 0, "world": 1}
    tok = _FakeTokenizer(vocab)

    hits = {
        0: [TokenHit(shard=0, position_in_shard=0, note_idx=0, position_in_note=0, activation=1.0)]
    }
    note_texts = {0: "Hello world"}

    extract_token_contexts(hits, note_texts, tok, context_window=5)

    h = hits[0][0]
    assert h.token_str == "Hello"
    assert len(h.context_tokens) == 2


def test_context_boundary_end() -> None:
    """Hit at the last position does not cause IndexError."""
    from mech_interp_research.feature_inspector import TokenHit, extract_token_contexts

    vocab = {"a": 0, "b": 1, "c": 2}
    tok = _FakeTokenizer(vocab)

    hits = {
        0: [TokenHit(shard=0, position_in_shard=2, note_idx=0, position_in_note=2, activation=1.0)]
    }
    note_texts = {0: "a b c"}

    extract_token_contexts(hits, note_texts, tok, context_window=5)

    h = hits[0][0]
    assert h.token_str == "c"
    assert len(h.context_tokens) == 3


# ---------------------------------------------------------------------------
# test_firing_stats
# ---------------------------------------------------------------------------


def test_firing_stats(tmp_path: Path) -> None:
    """Firing stats show higher rate for ICD-positive notes."""
    from safetensors.numpy import save_file

    from mech_interp_research.feature_inspector import (
        aggregate_firing_statistics,
        scan_shard_for_top_tokens,
    )

    d_model, d_sae = 16, 8
    sae = _make_sae(d_model, d_sae, seed=99)

    n_notes = 4
    tokens_per_note = 20
    n_tokens = n_notes * tokens_per_note
    rng = np.random.default_rng(42)
    acts = rng.standard_normal((n_tokens, d_model)).astype(np.float32)

    for i in range(tokens_per_note):
        acts[i] += sae.W_enc[:, 2] * 5.0
        acts[tokens_per_note + i] += sae.W_enc[:, 2] * 5.0

    shard_path = tmp_path / "shard_0000.safetensors"
    save_file({"activations": acts.astype(np.float16)}, str(shard_path))

    meta = pd.DataFrame(
        [
            {
                "note_idx": i,
                "shard": 0,
                "row_start": i * tokens_per_note,
                "row_end": (i + 1) * tokens_per_note,
                "n_tokens": tokens_per_note,
            }
            for i in range(n_notes)
        ]
    )

    icd_labels = {
        0: {"icd9_test": 1},
        1: {"icd9_test": 1},
        2: {"icd9_test": 0},
        3: {"icd9_test": 0},
    }
    target_codes = {2: "icd9_test"}

    _, shard_stats = scan_shard_for_top_tokens(
        sae=sae,
        shard_path=shard_path,
        shard_idx=0,
        metadata_for_shard=meta,
        target_latent_indices=[2],
        top_k=5,
        icd_labels_by_note=icd_labels,
        target_codes=target_codes,
    )

    agg = aggregate_firing_statistics([shard_stats])
    stats = agg[2]
    assert stats["firing_rate_pos"] > stats["firing_rate_neg"]
    assert stats["n_tokens_pos"] == 2 * tokens_per_note
    assert stats["n_tokens_neg"] == 2 * tokens_per_note


# ---------------------------------------------------------------------------
# test_diversity
# ---------------------------------------------------------------------------


def test_diversity_all_same() -> None:
    """All same token → diversity_score = 1/k."""
    from mech_interp_research.feature_inspector import TokenHit, assess_context_diversity

    k = 10
    hits = {
        0: [
            TokenHit(
                shard=0,
                position_in_shard=i,
                note_idx=0,
                position_in_note=i,
                activation=float(k - i),
                token_str="thyroid",
            )
            for i in range(k)
        ]
    }

    div = assess_context_diversity(hits)
    assert div[0]["n_unique_tokens"] == 1
    assert abs(div[0]["diversity_score"] - 1.0 / k) < 1e-6
    assert div[0]["top_token_frequency"][0] == ("thyroid", k)


def test_diversity_all_unique() -> None:
    """All unique tokens → diversity_score = 1.0."""
    from mech_interp_research.feature_inspector import TokenHit, assess_context_diversity

    tokens = ["thyroid", "diabetes", "insulin", "heart", "lung"]
    hits = {
        0: [
            TokenHit(
                shard=0,
                position_in_shard=i,
                note_idx=0,
                position_in_note=i,
                activation=float(len(tokens) - i),
                token_str=t,
            )
            for i, t in enumerate(tokens)
        ]
    }

    div = assess_context_diversity(hits)
    assert div[0]["n_unique_tokens"] == len(tokens)
    assert abs(div[0]["diversity_score"] - 1.0) < 1e-6


def test_diversity_empty() -> None:
    """No filled tokens → diversity_score = 0."""
    from mech_interp_research.feature_inspector import TokenHit, assess_context_diversity

    hits = {
        0: [
            TokenHit(
                shard=0,
                position_in_shard=0,
                note_idx=0,
                position_in_note=0,
                activation=1.0,
                token_str=None,
            )
        ]
    }

    div = assess_context_diversity(hits)
    assert div[0]["diversity_score"] == 0.0
    assert div[0]["n_unique_tokens"] == 0


# ---------------------------------------------------------------------------
# test_select_target_shards
# ---------------------------------------------------------------------------


def test_select_target_shards(tmp_path: Path) -> None:
    """select_target_shards picks shards with highest max-pooled activations."""
    from mech_interp_research.feature_inspector import select_target_shards

    ckpt_dir = tmp_path / "shard_ckpt"
    ckpt_dir.mkdir()

    d_sae = 16
    vecs0 = np.zeros((3, d_sae), dtype=np.float32)
    vecs0[0, 5] = 100.0
    vecs1 = np.zeros((3, d_sae), dtype=np.float32)
    vecs1[1, 5] = 10.0
    vecs2 = np.zeros((3, d_sae), dtype=np.float32)
    vecs2[2, 5] = 50.0

    np.save(ckpt_dir / "shard_0000_vectors.npy", vecs0)
    np.save(ckpt_dir / "shard_0001_vectors.npy", vecs1)
    np.save(ckpt_dir / "shard_0002_vectors.npy", vecs2)

    selected = select_target_shards(ckpt_dir, [5], n_shards=2)
    assert len(selected) == 2
    assert 0 in selected
    assert 2 in selected


def test_select_target_shards_missing_raises(tmp_path: Path) -> None:
    """Missing shard_ckpt directory raises FileNotFoundError."""
    from mech_interp_research.feature_inspector import select_target_shards

    with pytest.raises(FileNotFoundError, match="shard_ckpt"):
        select_target_shards(tmp_path / "nonexistent", [0], n_shards=1)


# ---------------------------------------------------------------------------
# test_load_target_latents
# ---------------------------------------------------------------------------


def test_load_target_latents(tmp_path: Path) -> None:
    """load_target_latents deduplicates by latent, keeps highest |r_pb|."""
    from mech_interp_research.feature_inspector import load_target_latents

    df = pd.DataFrame(
        {
            "latent": [10, 10, 20, 30, 40],
            "code": ["icd9_A", "icd9_B", "icd9_C", "icd9_D", "icd9_E"],
            "r_pb": [0.8, 0.9, 0.7, 0.6, 0.5],
            "abs_r": [0.8, 0.9, 0.7, 0.6, 0.5],
        }
    )
    csv_path = tmp_path / "top_associations.csv"
    df.to_csv(csv_path, index=False)

    pairs = load_target_latents(csv_path, n_pairs=3)
    assert len(pairs) == 3
    latent_ids = [p["latent"] for p in pairs]
    assert len(set(latent_ids)) == 3
    # Latent 10 should keep code B (abs_r=0.9)
    lat10 = [p for p in pairs if p["latent"] == 10][0]
    assert lat10["code"] == "icd9_B"


# ---------------------------------------------------------------------------
# test_build_report + serialize_report
# ---------------------------------------------------------------------------


def test_serialize_report_roundtrip(tmp_path: Path) -> None:
    """serialize_report produces valid JSON and CSV files."""
    import json

    from mech_interp_research.feature_inspector import (
        LatentReport,
        TokenHit,
        serialize_report,
    )

    report = LatentReport(
        latent_idx=42,
        icd_code="icd9_4019",
        r_pb=0.85,
        top_tokens=[
            TokenHit(
                shard=0,
                position_in_shard=10,
                note_idx=0,
                position_in_note=10,
                activation=50.0,
                token_str="thyroid",
                token_id=123,
                context_tokens=["has", "thyroid", "disease"],
            ),
        ],
        firing_rate_icd_pos=0.5,
        firing_rate_icd_neg=0.01,
        firing_rate_ratio=50.0,
        n_unique_tokens=1,
        diversity_score=1.0,
        top_token_frequency=[("thyroid", 1)],
    )

    out_dir = tmp_path / "output"
    serialize_report([report], out_dir, config={"n_pairs": 1})

    json_path = out_dir / "feature_inspection_report.json"
    csv_path = out_dir / "feature_inspection_details.csv"
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert data["summary"]["n_latents_inspected"] == 1
    assert len(data["latents"]) == 1
    assert data["latents"][0]["latent_idx"] == 42
    assert len(data["latents"][0]["top_tokens"]) == 1

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == 1
    assert csv_df.iloc[0]["latent_idx"] == 42


# ---------------------------------------------------------------------------
# test_integration — end-to-end with synthetic data
# ---------------------------------------------------------------------------


def test_integration_end_to_end(synthetic_run_dir: Path, tmp_path: Path) -> None:
    """Full pipeline round-trip on synthetic data produces valid output."""
    import json

    from mech_interp_research.feature_inspector import (
        aggregate_firing_statistics,
        assess_context_diversity,
        build_report,
        merge_top_tokens,
        scan_shard_for_top_tokens,
        select_target_shards,
        serialize_report,
    )
    from mech_interp_research.icd_eval import encode_and_pool, load_metadata

    d_model, d_sae = 64, 32
    sae = _make_sae(d_model, d_sae)
    metadata = load_metadata(synthetic_run_dir)

    # Create shard_ckpt via encode_and_pool
    ckpt_dir = tmp_path / "shard_ckpt"
    note_vectors, note_meta = encode_and_pool(
        sae=sae,
        activations_dir=synthetic_run_dir,
        metadata=metadata,
        checkpoint_dir=ckpt_dir,
    )

    # Create top_associations.csv
    eval_dir = tmp_path / "eval_output"
    eval_dir.mkdir()
    top_assoc = pd.DataFrame(
        {
            "latent": [0, 5, 10],
            "code": ["icd9_A", "icd9_B", "icd9_C"],
            "r_pb": [0.8, 0.7, 0.6],
            "abs_r": [0.8, 0.7, 0.6],
        }
    )
    top_assoc.to_csv(eval_dir / "top_associations.csv", index=False)

    # Symlink shard_ckpt into eval_dir
    import shutil

    shutil.copytree(ckpt_dir, eval_dir / "shard_ckpt")

    # Select shards
    target_indices = [0, 5, 10]
    selected = select_target_shards(eval_dir / "shard_ckpt", target_indices, n_shards=2)
    assert len(selected) <= 2

    # ICD labels
    icd_labels = {i: {"icd9_A": 1 if i < 3 else 0, "icd9_B": 0, "icd9_C": 0} for i in range(5)}
    target_codes = {0: "icd9_A", 5: "icd9_B", 10: "icd9_C"}

    # Pass 1: scan shards
    all_hits = []
    all_stats = []
    for shard_idx in selected:
        shard_path = synthetic_run_dir / f"shard_{shard_idx:04d}.safetensors"
        if not shard_path.exists():
            continue
        meta_s = metadata[metadata["shard"] == shard_idx]
        hits, stats = scan_shard_for_top_tokens(
            sae=sae,
            shard_path=shard_path,
            shard_idx=shard_idx,
            metadata_for_shard=meta_s,
            target_latent_indices=target_indices,
            top_k=10,
            icd_labels_by_note=icd_labels,
            target_codes=target_codes,
        )
        all_hits.append(hits)
        all_stats.append(stats)

    global_top = merge_top_tokens(all_hits, top_k=10)
    agg_stats = aggregate_firing_statistics(all_stats)
    diversity = assess_context_diversity(global_top)

    target_pairs = [
        {"latent": 0, "code": "icd9_A", "r_pb": 0.8, "abs_r": 0.8},
        {"latent": 5, "code": "icd9_B", "r_pb": 0.7, "abs_r": 0.7},
        {"latent": 10, "code": "icd9_C", "r_pb": 0.6, "abs_r": 0.6},
    ]
    reports = build_report(target_pairs, global_top, agg_stats, diversity)
    assert len(reports) == 3

    output_dir = tmp_path / "inspection_output"
    serialize_report(reports, output_dir)

    json_path = output_dir / "feature_inspection_report.json"
    csv_path = output_dir / "feature_inspection_details.csv"
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path) as f:
        data = json.load(f)
    assert "summary" in data
    assert "latents" in data
    assert len(data["latents"]) == 3
    for lat in data["latents"]:
        assert "latent_idx" in lat
        assert "firing_stats" in lat
        assert "diversity" in lat
        assert "top_tokens" in lat
