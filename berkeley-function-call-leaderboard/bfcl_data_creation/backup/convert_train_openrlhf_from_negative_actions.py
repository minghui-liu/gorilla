#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


BFCL_EVAL_HEADER = (
    "You are an expert in composing functions."
    "You are given a question and a set of possible functions. Based on the question, you will "
    "need to make one or more function/tool calls to achieve the purpose. If none of the functions "
    "can be used, point it out. If the given question lacks the parameters required by the function, "
    "also point it out.\n\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  "
    "You SHOULD NOT include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user within "
    "the current turn. Continue to output functions to call until you have fulfilled the user's "
    "request to the best of your ability. Once you have no more functions to call, the system will "
    "consider the current turn complete and proceed to the next turn or task.\n\n"
    "Here is a list of functions in json format that you can invoke.\n"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def train_task_ids(train_jsonl_path: Path) -> Set[str]:
    ids: Set[str] = set()
    for row in load_jsonl(train_jsonl_path):
        task_id = row.get("id")
        if isinstance(task_id, str):
            ids.add(task_id)
    return ids


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not isinstance(role, str):
            continue
        normalized.append(dict(msg))
    return normalized


def build_compact_functions_from_system_prompt(system_prompt: str) -> List[Dict[str, Any]]:
    lines = system_prompt.splitlines()
    i = 0
    functions: List[Dict[str, Any]] = []

    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("Function: "):
            i += 1
            continue

        name = line[len("Function: ") :].strip()
        i += 1
        params: List[Dict[str, Any]] = []

        while i < len(lines) and not lines[i].strip().startswith("Parameters:"):
            if lines[i].strip().startswith("Function: "):
                break
            i += 1

        if i < len(lines) and lines[i].strip().startswith("Parameters:"):
            i += 1

        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("Function: ") or s.startswith("When you need to call functions"):
                break
            if s.startswith("- "):
                match = re.match(r"-\s*([A-Za-z0-9_]+)\s*\(([^)]+)\):\s*(.*)", s)
                if match:
                    param_name = match.group(1)
                    param_type = match.group(2).strip()
                    tail = match.group(3)
                    required = "(required)" in tail
                    params.append(
                        {
                            "name": param_name,
                            "type": param_type,
                            "required": required,
                        }
                    )
            i += 1

        functions.append({"name": name, "parameters": params})

    return functions


def rewrite_system_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "system":
        return messages
    content = first.get("content")
    if not isinstance(content, str) or not content:
        return messages

    compact_functions = build_compact_functions_from_system_prompt(content)
    if not compact_functions:
        return messages

    new_first = dict(first)
    new_first["content"] = BFCL_EVAL_HEADER + json.dumps(compact_functions, ensure_ascii=False, indent=2)
    return [new_first] + messages[1:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OpenRLHF JSONL from BFCL_v4_negative_actions trajectories."
    )
    parser.add_argument(
        "--negative-actions",
        default="BFCL_v4_negative_actions.json",
        help="Path to BFCL negative actions JSON.",
    )
    parser.add_argument(
        "--train",
        default="BFCL_v4_train.json",
        help="Path to BFCL train JSONL (used to filter task ids).",
    )
    parser.add_argument(
        "--output",
        default="BFCL_v4_train_openrlhf_from_negative_actions.jsonl",
        help="Output OpenRLHF JSONL path.",
    )
    parser.add_argument(
        "--keep-original-system-prompt",
        action="store_true",
        help="Do not rewrite system prompt to compact BFCL-eval style.",
    )
    args = parser.parse_args()

    negative_actions_path = Path(args.negative_actions)
    train_path = Path(args.train)
    output_path = Path(args.output)

    train_ids = train_task_ids(train_path)
    data = load_json(negative_actions_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {negative_actions_path}")

    written = 0
    skipped_not_train = 0
    skipped_invalid = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in data:
            if not isinstance(row, dict):
                skipped_invalid += 1
                continue

            task_id = row.get("task_name")
            turn_id = row.get("prefix_len")
            positive_action = row.get("positive_action")
            negative_action = row.get("negative_action")
            prompt_info = row.get("prompt_info")
            messages = prompt_info.get("messages") if isinstance(prompt_info, dict) else None

            if not isinstance(task_id, str) or task_id not in train_ids:
                skipped_not_train += 1
                continue
            if (
                not isinstance(turn_id, int)
                or not isinstance(positive_action, list)
                or not isinstance(negative_action, list)
            ):
                skipped_invalid += 1
                continue

            normalized_messages = normalize_messages(messages)
            if not args.keep_original_system_prompt:
                normalized_messages = rewrite_system_prompt(normalized_messages)

            output_row = {
                "input": normalized_messages,
                "positive_action": positive_action,
                "negative_action": negative_action,
                "task_id": task_id,
                "turn_id": turn_id,
            }
            out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Train task ids: {len(train_ids)}")
    print(f"Input rows: {len(data)}")
    print(f"Wrote rows: {written}")
    print(f"Skipped non-train task rows: {skipped_not_train}")
    print(f"Skipped invalid rows: {skipped_invalid}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
