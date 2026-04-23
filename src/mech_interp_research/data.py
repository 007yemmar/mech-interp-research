"""Load clinical notes from a CSV into a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_notes(
    input_csv: str | Path,
    text_col: str | None,
    num_notes: int,
) -> tuple[pd.DataFrame, str]:
    """Load up to `num_notes` rows from `input_csv`, returning (df, text_col).

    Column auto-detection: if `text_col` is None, tries "note_text" then "text".
    Drops rows with null text. Raises if the CSV is missing or no text column
    can be resolved.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    candidates: list[str] = []
    if text_col is not None:
        candidates.append(text_col)
    candidates.extend(["note_text", "text"])

    chosen = next((c for c in candidates if c in df.columns), None)
    if chosen is None:
        raise ValueError(
            f"No text column found. Tried: {candidates}. Columns present: {list(df.columns)}"
        )

    df = df[df[chosen].notna()].copy().head(num_notes).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No non-empty notes after filtering.")

    return df, chosen
