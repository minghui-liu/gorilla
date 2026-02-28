#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def find_first_json_array(content: str) -> str:
    anchor = "Here is a list of functions in json format that you can invoke."
    anchor_pos = content.find(anchor)
    search_start = anchor_pos if anchor_pos != -1 else 0

    start = content.find("[", search_start)
    if start == -1:
        raise ValueError("Could not find function JSON array start '[' in instruction.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(content)):
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return content[start : i + 1]

    raise ValueError("Could not find matching ']' for function JSON array in instruction.")


def build_compact_functions(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for func in functions:
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        params_obj = func.get("parameters", {})
        props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
        required = (
            set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
        )

        params: List[Dict[str, Any]] = []
        if isinstance(props, dict):
            for param_name, param_info in props.items():
                param_type = "string"
                if isinstance(param_info, dict):
                    p_type = param_info.get("type", "string")
                    if isinstance(p_type, list):
                        p_type = "|".join(str(x) for x in p_type)
                    param_type = p_type
                params.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "required": param_name in required,
                    }
                )

        compact.append({"name": name, "parameters": params})
    return compact


def rewrite_instruction(instruction: str) -> str:
    fn_json = find_first_json_array(instruction)
    functions = json.loads(fn_json)
    if not isinstance(functions, list):
        raise ValueError("Parsed function docs are not a JSON array.")
    compact = build_compact_functions(functions)
    return BFCL_EVAL_HEADER + json.dumps(compact, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BFCL_v4_sft_qwen.jsonl to short/compact system prompt format."
    )
    parser.add_argument(
        "--input",
        default="BFCL_v4_sft_qwen.jsonl",
        help="Input SFT JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="BFCL_v4_sft_qwen_short_prompt.jsonl",
        help="Output SFT JSONL path with compact system prompts.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(in_path)
    converted = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            instruction = row.get("instruction")
            if not isinstance(instruction, str):
                raise ValueError(
                    f"Row {idx} has no valid 'instruction' string; cannot rewrite."
                )
            new_row = dict(row)
            new_row["instruction"] = rewrite_instruction(instruction)
            out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Input rows: {len(rows)}")
    print(f"Converted rows: {converted}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
