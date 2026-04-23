import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_INPUT_CSV = "/Users/rishittyagi/Desktop/mimic-iv/pipeline_outputs/step1/train.csv"
DEFAULT_OUTPUT_DIR = "/Users/rishittyagi/Desktop/mimic-iv/pipeline_outputs/step2_activations"
MODEL_NAME = "google/gemma-2-2b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract layer-16 residual activations for a tiny batch of notes."
    )
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--num_notes", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument(
        "--tokens_per_shard",
        type=int,
        default=250000,
        help="Approx token rows per shard for SAE-ready storage.",
    )
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(input_csv: Path, text_col: str | None, num_notes: int) -> tuple[pd.DataFrame, str]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    candidate_cols = [text_col] if text_col else []
    candidate_cols.extend(["note_text", "text"])
    chosen_text_col = next((c for c in candidate_cols if c and c in df.columns), None)

    if chosen_text_col is None:
        raise ValueError(
            f"Could not find text column. Tried: {candidate_cols}. Available columns: {list(df.columns)}"
        )

    df = df[df[chosen_text_col].notna()].copy().head(num_notes).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No non-empty notes found after filtering.")

    return df, chosen_text_col


def load_model_and_tokenizer(device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise OSError("HF_TOKEN is not set. Export your Hugging Face token before running.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def extract_note_activation(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    layer: int,
    max_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)
    truncated_tokens = int(attn_mask.sum().item())

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # For HF decoder models, hidden_states[0] = embedding output.
    # Post-block residual for layer L maps to hidden_states[L + 1].
    hs_index = layer + 1
    if hs_index >= len(out.hidden_states):
        raise ValueError(
            f"Requested layer {layer} but model only has {len(out.hidden_states) - 1} layers."
        )

    activations = out.hidden_states[hs_index][0, :truncated_tokens, :].detach().cpu().float()

    meta = {
        "num_tokens_truncated": truncated_tokens,
        "d_model": int(activations.shape[1]),
        "shape": [int(activations.shape[0]), int(activations.shape[1])],
    }
    return activations, meta


def run_checks(acts: list[torch.Tensor], expected_d_model: int = 2304) -> dict[str, Any]:
    shape_ok = all(a.ndim == 2 and a.shape[1] == expected_d_model for a in acts)
    finite_ok = all(torch.isfinite(a).all().item() for a in acts)
    non_zero_ok = all((a.abs().sum() > 0).item() for a in acts)

    diversity_ok = None
    mean_abs_diff = None
    if len(acts) >= 2:
        a0 = acts[0]
        a1 = acts[1]
        t = min(a0.shape[0], a1.shape[0])
        mean_abs_diff = float((a0[:t] - a1[:t]).abs().mean().item())
        diversity_ok = mean_abs_diff > 0.0

    return {
        "shape_check_pass": bool(shape_ok),
        "finite_check_pass": bool(finite_ok),
        "non_zero_check_pass": bool(non_zero_ok),
        "diversity_check_pass": bool(diversity_ok) if diversity_ok is not None else None,
        "mean_abs_diff_note0_note1": mean_abs_diff,
        "expected_d_model": expected_d_model,
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    df, text_col = load_dataset(input_csv, args.text_col, args.num_notes)
    tokenizer, model = load_model_and_tokenizer(device)

    all_activations: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting activations"):
        text = str(row[text_col])
        acts, act_meta = extract_note_activation(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            layer=args.layer,
            max_length=args.max_length,
        )
        all_activations.append(acts)
        meta = {
            "note_idx": int(idx),
            "subject_id": int(row["subject_id"]) if "subject_id" in df.columns else None,
            "admission_id": int(row["admission_id"])
            if "admission_id" in df.columns
            else (int(row["hadm_id"]) if "hadm_id" in df.columns else None),
            "activation_shape": act_meta["shape"],
            "num_tokens_truncated": act_meta["num_tokens_truncated"],
            "d_model": act_meta["d_model"],
            "text_char_len": len(text),
        }
        metadata.append(meta)

    checks = run_checks(all_activations, expected_d_model=2304)

    activations_path = output_dir / "layer16_residual_activations.pt"
    metadata_path = output_dir / "layer16_metadata.json"
    checks_path = output_dir / "activation_checks.json"
    summary_path = output_dir / "run_summary.json"

    torch.save(
        {
            "model_name": MODEL_NAME,
            "layer": args.layer,
            "activations": all_activations,
        },
        activations_path,
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    summary = {
        "model_name": MODEL_NAME,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "num_notes_processed": len(metadata),
        "text_column_used": text_col,
        "max_length": args.max_length,
        "layer": args.layer,
        "device": device,
        "files": {
            "activations_pt": str(activations_path),
            "metadata_json": str(metadata_path),
            "checks_json": str(checks_path),
        },
        "checks": checks,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Activation extraction complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
