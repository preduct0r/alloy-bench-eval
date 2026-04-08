#!/usr/bin/env python3
"""
Alloy-Bench evaluation runner.

Usage:
  # Evaluate a base model directly:
  python run_benchmark.py --model Qwen/Qwen3-32B --data-path data/Alloy-Bench.xlsx

  # Evaluate with LoRA adapter:
  python run_benchmark.py --model Qwen/Qwen3-32B --lora /path/to/adapter --data-path data/Alloy-Bench.xlsx

  # Use a config file (reads model_path from model_settings.model_path):
  python run_benchmark.py --config configs/pure_qwen3.json --data-path data/Alloy-Bench.xlsx
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bench_eval import (
    detect_area_column,
    evaluate_batch_vllm,
    merge_lora,
)


def _load_model(model_path: str, lora_path: Optional[str]) -> Tuple[str, Optional[str], Any]:
    from vllm import LLM

    merged_dir: Optional[str] = None
    used_path = model_path
    if lora_path:
        merged_dir = merge_lora(model_path, lora_path)
        used_path = merged_dir

    llm = LLM(
        model=used_path,
        tokenizer=used_path,
        trust_remote_code=True,
        enable_lora=False,
        max_model_len=4096,
        guided_decoding_backend="outlines",
    )
    return used_path, merged_dir, llm


def run_validation(
    model_path: str,
    data_path: str,
    lora_path: Optional[str] = None,
    batch_size: int = 50,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_new_tokens: int = 2048,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model_name = Path(lora_path).name if lora_path else Path(model_path).name

    print(f"\n{'=' * 80}")
    print(f"Model:   {model_path}")
    if lora_path:
        print(f"LoRA:    {lora_path}")
    print(f"Dataset: {data_path}")
    print(f"Params:  temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_new_tokens}")
    print(f"{'=' * 80}\n")

    df = pd.read_excel(data_path, engine="openpyxl")
    area_col = detect_area_column(df)
    if area_col is None:
        raise ValueError("Cannot detect knowledge area column in dataset")

    merged_dir: Optional[str] = None
    try:
        _, merged_dir, llm = _load_model(model_path, lora_path)

        totals: Dict[str, int] = {}
        corrects: Dict[str, int] = {}
        samples: List[Dict[str, Any]] = []

        rows_with_idx = list(df.iterrows())
        for i in range(0, len(rows_with_idx), batch_size):
            batch = rows_with_idx[i : i + batch_size]
            batch_rows = [row for _, row in batch]

            batch_areas: List[str] = []
            for row in batch_rows:
                area = row.get(area_col, "UNKNOWN")
                area = "UNKNOWN" if pd.isna(area) else str(area).strip() or "UNKNOWN"
                batch_areas.append(area)

            batch_results = evaluate_batch_vllm(
                rows=batch_rows,
                llm=llm,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )

            for (row_idx, row), area, res in zip(batch, batch_areas, batch_results):
                totals[area] = totals.get(area, 0) + 1
                if res.get("correct"):
                    corrects[area] = corrects.get(area, 0) + 1

                question = str(row.get("Вопрос", "") if pd.notna(row.get("Вопрос", "")) else "")
                samples.append(
                    {
                        "Model": model_name,
                        "RowIndex": row_idx,
                        "Area": area,
                        "Question": question,
                        "Correct": bool(res.get("correct")),
                        "ModelAnswerKey": res.get("model_answer_key"),
                        "ModelAnswerText": res.get("model_answer_text"),
                        "CorrectAnswerKey": res.get("correct_answer_key"),
                        "CorrectAnswerText": res.get("correct_answer_text"),
                    }
                )

        area_results: Dict[str, float] = {}
        for area, total in totals.items():
            area_results[area] = (corrects.get(area, 0) / total * 100.0) if total else 0.0

        total_correct = sum(corrects.values())
        total_questions = sum(totals.values())
        micro_avg = (total_correct / total_questions * 100.0) if total_questions else 0.0
        macro_avg = (
            sum(area_results.values()) / len(area_results) if area_results else 0.0
        )

        print(f"\n{'=' * 80}")
        print(f"Results: {model_name}")
        print(f"{'=' * 80}")
        for area, acc in sorted(area_results.items()):
            count = totals[area]
            print(f"  {area}: {acc:.2f}%  ({corrects.get(area, 0)}/{count})")
        print(f"\n  Macro avg (по областям): {macro_avg:.2f}%")
        print(f"  Micro avg (по вопросам): {micro_avg:.2f}%  ({total_correct}/{total_questions})")
        print(f"{'=' * 80}\n")

        return area_results, samples
    finally:
        if merged_dir:
            shutil.rmtree(merged_dir, ignore_errors=True)


def save_results(
    area_results: Dict[str, float],
    samples: List[Dict[str, Any]],
    model_path: str,
    data_path: str,
    output_path: str,
) -> None:
    all_areas = sorted(area_results.keys())
    row_values = [area_results.get(a) for a in all_areas]
    valid = [v for v in row_values if v is not None]
    macro_avg = sum(valid) / len(valid) if valid else None
    row_values.append(macro_avg)

    df_summary = pd.DataFrame(
        [row_values], columns=all_areas + ["Average"], index=[Path(model_path).name]
    )
    df_summary.index.name = "Model"

    df_samples = pd.DataFrame(samples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Results", float_format="%.2f")
        if not df_samples.empty:
            col_order = [
                "Model", "RowIndex", "Area", "Question",
                "Correct", "ModelAnswerKey", "ModelAnswerText",
                "CorrectAnswerKey", "CorrectAnswerText",
            ]
            cols = [c for c in col_order if c in df_samples.columns]
            df_samples[cols].to_excel(writer, sheet_name="Samples", index=False)
        pd.DataFrame(
            {
                "Parameter": ["Model", "Data", "Generated"],
                "Value": [model_path, data_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
        ).to_excel(writer, sheet_name="Metadata", index=False)

    print(f"Results saved to: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Alloy-Bench evaluation runner")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=str, help="HF model name or local path")
    source.add_argument("--config", type=str, help="JSON config with model_settings.model_path")
    p.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter")
    p.add_argument("--data-path", type=str, required=True, help="Path to BENCH3-style XLSX dataset")
    p.add_argument("--output", type=str, default=None, help="Output XLSX path (auto-generated if omitted)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_path = cfg.get("model_settings", {}).get("model_path")
        if not model_path:
            raise ValueError("model_path not found in config model_settings.model_path")
    else:
        model_path = args.model

    area_results, samples = run_validation(
        model_path=model_path,
        data_path=args.data_path,
        lora_path=args.lora,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).name.replace("/", "_")
        os.makedirs("results", exist_ok=True)
        output_path = f"results/results_{model_name}_{ts}.xlsx"

    save_results(area_results, samples, model_path, args.data_path, output_path)


if __name__ == "__main__":
    main()
