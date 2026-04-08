#!/usr/bin/env python3
"""
Convert Alloy-Bench into BENCH3-style XLSX.

Supported inputs:
  1. HuggingFace dataset nn-tech/Alloy-Bench (default, requires `datasets`)
  2. Local parquet file
  3. Directory with parquet files (e.g. from `huggingface-cli download`)

Output columns:
  Вопрос, Вариант 1..5, Правильный ответ, Область знаний
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"question", "correct_answer", "knowledge_area"}


def _load_remote(split: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("nn-tech/Alloy-Bench")
    resolved = split if split in ds else next(iter(ds.keys()))
    print(f"HF split: {resolved}")
    return ds[resolved].to_pandas()


def _load_local(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _option_value(options: object, index: int) -> str | None:
    if isinstance(options, (list, tuple)) and len(options) > index:
        val = options[index]
        return None if pd.isna(val) else str(val)
    return None


def convert(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    has_options_list = "options" in df.columns
    has_option_cols = all(f"option_{i}" in df.columns for i in range(1, 6))
    if not has_options_list and not has_option_cols:
        raise ValueError("Need either 'options' list or 'option_1'..'option_5' columns")

    out = pd.DataFrame(
        {
            "Вопрос": df["question"],
            "Правильный ответ": df["correct_answer"],
            "Область знаний": df["knowledge_area"],
        }
    )

    if has_options_list:
        for i in range(5):
            out[f"Вариант {i + 1}"] = df["options"].apply(
                lambda x, idx=i: _option_value(x, idx)
            )
    else:
        for i in range(1, 6):
            col = f"option_{i}"
            out[f"Вариант {i}"] = df[col].where(df[col].notna(), None)

    return out[
        [
            "Вопрос",
            "Вариант 1",
            "Вариант 2",
            "Вариант 3",
            "Вариант 4",
            "Вариант 5",
            "Правильный ответ",
            "Область знаний",
        ]
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Alloy-Bench to BENCH3 XLSX")
    p.add_argument("--input", default=None, help="Local parquet file or directory")
    p.add_argument("--output", default="data/Alloy-Bench.xlsx", help="Output XLSX path")
    p.add_argument("--split", default="validation", help="HF split (if downloading)")
    args = p.parse_args()

    df = _load_local(args.input) if args.input else _load_remote(args.split)
    out = convert(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.output, index=False)
    print(f"rows:   {len(out)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
