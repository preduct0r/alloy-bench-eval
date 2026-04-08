"""Core evaluation logic: prompt building, vLLM inference, answer checking."""

from __future__ import annotations

import json
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def build_prompt(row: pd.Series) -> Tuple[str, List[str]]:
    all_options = ["a", "b", "c", "d", "e"]
    variants: List[str] = []
    for i in range(1, 6):
        val = row.get(f"Вариант {i}", "")
        if pd.notna(val) and str(val).strip():
            variants.append(str(val).strip())
    available_options = all_options[: len(variants)]
    prompt = f"Вопрос: {row['Вопрос']}\n\nВарианты ответа:\n"
    for i, v in enumerate(variants):
        prompt += f"{available_options[i]}. {v}\n"
    options_str = "/".join(f'"{opt}"' for opt in available_options)
    prompt += (
        f'Ответьте в формате JSON: {{"reasoning": "Объяснение вашего выбора", "answer": "одна из букв: {options_str}"}}.'
        " Обязательно выберите один из вариантов."
    )
    return prompt, available_options


def parse_answer_json(raw_text: str) -> Dict[str, Optional[str]]:
    try:
        data = json.loads(raw_text)
    except Exception:
        data = {}
    reasoning = str(data.get("reasoning", "")).strip()
    answer = str(data.get("answer", "")).strip().lower()
    if answer not in ["a", "b", "c", "d", "e"]:
        answer = None
    return {"reasoning": reasoning, "answer": answer}


def detect_area_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {c: str(c).lower() for c in df.columns}
    preferred = [
        "область знаний",
        "область",
        "категория",
        "тематика",
        "area",
        "domain",
        "subject",
    ]
    for p in preferred:
        for col, low in lowered.items():
            if low == p:
                return col
    keywords = ["область", "знаний", "категор", "тем", "area", "domain", "subject"]
    for col, low in lowered.items():
        if any(k in low for k in keywords):
            return col
    return None


def _schema_for_options(options: List[str]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {"reasoning": {"type": "string"}}
    if options:
        properties["answer"] = {"type": "string", "enum": options}
    else:
        properties["answer"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    }


def _extract_variants(row: pd.Series) -> List[str]:
    variants: List[str] = []
    for i in range(1, 6):
        val = row.get(f"Вариант {i}", "")
        if pd.notna(val) and str(val).strip():
            variants.append(str(val).strip())
    return variants


def _group_by_options(
    rows: List[pd.Series],
) -> Tuple[List[List[int]], List[List[str]], List[str], List[Dict]]:
    prompts: List[str] = []
    options_list: List[List[str]] = []
    meta: List[Dict] = []

    for row in rows:
        prompt, available_options = build_prompt(row)
        prompts.append(prompt)
        options_list.append(available_options)

        variants = _extract_variants(row)
        correct_answer_raw = str(row["Правильный ответ"]).strip()
        correct_answer_texts = [ans.strip() for ans in correct_answer_raw.split("|")]

        option_keys = ["a", "b", "c", "d", "e"]
        correct_answer_keys: List[str] = []
        for correct_text in correct_answer_texts:
            try:
                correct_answer_keys.append(option_keys[variants.index(correct_text)])
            except ValueError:
                pass

        meta.append(
            {
                "variants": variants,
                "correct_answer_text": correct_answer_texts,
                "correct_answer_key": correct_answer_keys,
            }
        )

    key_to_indices: Dict[Tuple[str, ...], List[int]] = {}
    for idx, opts in enumerate(options_list):
        key_to_indices.setdefault(tuple(opts), []).append(idx)

    return list(key_to_indices.values()), options_list, prompts, meta


def _build_sampling_params(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    json_schema: Optional[Dict] = None,
):
    from vllm import SamplingParams

    if json_schema is None:
        return SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
        )

    attempts = []
    # A: GuidedDecodingParams(json_schema=...)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json_schema=json_schema),
        )
    except Exception as e:
        attempts.append(e)
    # B: GuidedDecodingParams(json=...)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json=json_schema),
        )
    except Exception as e:
        attempts.append(e)
    # C: guided_json kwarg
    try:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            guided_json=json_schema,
        )
    except Exception as e:
        attempts.append(e)

    raise RuntimeError(
        "Cannot configure JSON Schema guided decoding for installed vLLM version. "
        f"Tried {len(attempts)} methods. Last error: {attempts[-1]}"
    )


def evaluate_batch_vllm(
    rows: List[pd.Series],
    llm: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> List[Dict[str, Optional[str]]]:
    if not rows:
        return []

    groups_indices, options_list, prompts, meta = _group_by_options(rows)
    parsed_answers: Dict[int, Dict[str, Optional[str]]] = {}

    for indices in groups_indices:
        if not indices:
            continue
        group_options = options_list[indices[0]]
        json_schema = _schema_for_options(group_options)
        sampling_params = _build_sampling_params(
            temperature, top_p, top_k, max_new_tokens, json_schema
        )
        group_prompts = [prompts[i] for i in indices]

        try:
            outputs = llm.generate(group_prompts, sampling_params)
        except Exception as e_guided:
            print(
                f"WARNING: guided decoding failed, retrying without schema. Error: {e_guided}"
            )
            try:
                fallback = _build_sampling_params(
                    temperature, top_p, top_k, max_new_tokens
                )
                outputs = llm.generate(group_prompts, fallback)
            except Exception as e_fb:
                print(
                    f"WARNING: generation failed for {len(group_prompts)} samples, "
                    f"marking as incorrect. Error: {e_fb}"
                )
                outputs = []

        for local_idx, out in zip(indices, outputs):
            text = out.outputs[0].text if out.outputs else "{}"
            parsed_answers[local_idx] = parse_answer_json(text)

    results: List[Dict[str, Optional[str]]] = []
    option_keys = ["a", "b", "c", "d", "e"]
    for i in range(len(rows)):
        parsed = parsed_answers.get(i, {"reasoning": "", "answer": None})
        variants = meta[i]["variants"]
        model_answer_key = parsed.get("answer")
        idx_option = (
            option_keys.index(model_answer_key)
            if model_answer_key in option_keys
            else -1
        )
        model_answer_text = (
            variants[idx_option] if 0 <= idx_option < len(variants) else None
        )

        correct_answer_texts = meta[i]["correct_answer_text"]
        correct_answer_keys = meta[i]["correct_answer_key"]
        correct = (model_answer_text in correct_answer_texts) or (
            model_answer_key in correct_answer_keys
        )

        correct_answer_text_str = (
            " | ".join(correct_answer_texts) if correct_answer_texts else None
        )
        correct_answer_key_str = (
            ", ".join(correct_answer_keys) if correct_answer_keys else None
        )

        results.append(
            {
                "correct": correct,
                "model_answer_key": model_answer_key,
                "model_answer_text": model_answer_text,
                "correct_answer_key": correct_answer_key_str,
                "correct_answer_text": correct_answer_text_str,
            }
        )
    return results


def merge_lora(model_path: str, lora_path: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tmp_dir = tempfile.mkdtemp(prefix="vllm_merged_")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(
        base, lora_path, torch_dtype="auto", device_map="auto", is_trainable=False
    )
    merged = peft_model.merge_and_unload()
    if merged is None or isinstance(merged, PeftModel):
        raise RuntimeError("LoRA merge failed")
    merged.save_pretrained(tmp_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(tmp_dir)
    return tmp_dir
