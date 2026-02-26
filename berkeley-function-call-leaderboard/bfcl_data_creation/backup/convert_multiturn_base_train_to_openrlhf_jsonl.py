#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}") from exc
    return rows


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return load_jsonl(path)


def flatten_turn_messages(question_turns: List[List[Dict[str, Any]]], turn_id: int) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for idx in range(turn_id + 1):
        turn_messages = question_turns[idx]
        if not isinstance(turn_messages, list):
            continue
        for msg in turn_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(role, str) and isinstance(content, str):
                messages.append({"role": role, "content": content})
    return messages


def maybe_add_system_prompt(question_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
        from bfcl_eval.utils import populate_test_cases_with_predefined_functions
    except Exception as exc:
        print(f"Warning: could not import BFCL prompt helpers; continuing without system prompt. ({exc})")
        return question_rows

    enriched = populate_test_cases_with_predefined_functions(question_rows)
    updated: List[Dict[str, Any]] = []
    for row in enriched:
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        test_id = row.get("id")
        function_doc = row.get("function")
        if (
            isinstance(question, list)
            and question
            and isinstance(question[0], list)
            and isinstance(test_id, str)
        ):
            # Add BFCL-style system prompt into the first turn chat history.
            question[0] = system_prompt_pre_processing_chat_model(question[0], function_doc, test_id)
        updated.append(row)
    return updated


def load_system_prompt_map_from_sft(path: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for row in load_jsonl(path):
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        instruction = row.get("instruction")
        if not isinstance(row_id, str) or not isinstance(instruction, str):
            continue
        if "_turn_" not in row_id:
            continue
        task_id = row_id.rsplit("_turn_", 1)[0]
        if task_id not in prompts:
            prompts[task_id] = instruction
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BFCL v4 multi-turn base train task ids to OpenRLHF-compatible JSONL."
    )
    parser.add_argument(
        "--train",
        default="BFCL_v4_multi_turn_base_train.json",
        help="Path to train jsonl with fields id and ground_truth",
    )
    parser.add_argument(
        "--questions",
        default="../bfcl_eval/data/BFCL_v4_multi_turn_base.json",
        help="Path to BFCL multi-turn question JSON",
    )
    parser.add_argument(
        "--output",
        default="BFCL_v4_multi_turn_base_train_openrlhf.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Inject BFCL-generated system prompt into the first turn messages.",
    )
    parser.add_argument(
        "--sft-system-prompts",
        default="BFCL_v4_multi_turn_base_sft_qwen.jsonl",
        help="Optional SFT JSONL path to source system prompts from `instruction` field.",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    questions_path = Path(args.questions)
    output_path = Path(args.output)
    sft_prompt_path = Path(args.sft_system_prompts) if args.sft_system_prompts else None

    train_rows = load_jsonl(train_path)
    question_rows = load_json_or_jsonl(questions_path)
    if args.include_system_prompt:
        question_rows = maybe_add_system_prompt(question_rows)
    question_by_id = {row.get("id"): row for row in question_rows if isinstance(row, dict) and row.get("id")}
    system_prompt_by_task: Dict[str, str] = {}
    if sft_prompt_path and sft_prompt_path.exists():
        system_prompt_by_task = load_system_prompt_map_from_sft(sft_prompt_path)

    output_count = 0
    missing_questions: List[str] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in train_rows:
            if not isinstance(row, dict):
                continue
            task_id = row.get("id")
            gt_turns = row.get("ground_truth")

            if not isinstance(task_id, str) or not isinstance(gt_turns, list):
                continue

            question_entry = question_by_id.get(task_id)
            if not isinstance(question_entry, dict):
                missing_questions.append(task_id)
                continue

            question_turns = question_entry.get("question")
            if not isinstance(question_turns, list):
                continue

            max_turns = min(len(question_turns), len(gt_turns))
            for turn_id in range(max_turns):
                input_messages = flatten_turn_messages(question_turns, turn_id)
                system_prompt = system_prompt_by_task.get(task_id)
                if isinstance(system_prompt, str) and system_prompt.strip():
                    input_messages = [{"role": "system", "content": system_prompt}] + input_messages
                output_row = {
                    "input": input_messages,
                    "ground_truth": gt_turns[turn_id],
                    "task_id": task_id,
                    "turn_id": turn_id,
                }
                out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                output_count += 1

    print(f"Train tasks: {len(train_rows)}")
    print(f"Wrote rows: {output_count}")
    print(f"Output: {output_path}")
    if missing_questions:
        print(f"Missing question entries: {len(missing_questions)}")


if __name__ == "__main__":
    main()
