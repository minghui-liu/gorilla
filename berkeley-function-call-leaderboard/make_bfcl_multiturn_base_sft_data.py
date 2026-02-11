import json
import random
from pathlib import Path

from bfcl_eval.utils import load_file, populate_test_cases_with_predefined_functions
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

DATA_Q = Path("bfcl_eval/data/BFCL_v4_multi_turn_base.json")
DATA_A = Path("bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json")
TRAIN_OUT = Path("BFCL_v4_multi_turn_base_train.json")
TEST_OUT = Path("./BFCL_v4_multi_turn_base_test.json")
ALPACA_OUT = Path("./BFCL_v4_multi_turn_base_sft_qwen.jsonl")
SPLIT_SEED = 42
TRAIN_RATIO = 0.75


def to_python_list_str(turn_calls):
    return "[" + ", ".join(turn_calls) + "]"


# 1) Load possible answers and split into train/test
all_answers = load_file(DATA_A)
random.Random(SPLIT_SEED).shuffle(all_answers)
train_size = int(len(all_answers) * TRAIN_RATIO)
train_answers = all_answers[:train_size]
test_answers = all_answers[train_size:]

with TRAIN_OUT.open("w", encoding="utf-8") as f:
    for row in train_answers:
        f.write(json.dumps(row) + "\n")

with TEST_OUT.open("w", encoding="utf-8") as f:
    for row in test_answers:
        f.write(json.dumps(row) + "\n")

train_ids = {x["id"] for x in train_answers}
answers = {x["id"]: x["ground_truth"] for x in train_answers}

# 2) Load and enrich questions with function docs, keep only train ids
questions = load_file(DATA_Q)
questions = [q for q in questions if q["id"] in train_ids]
questions = populate_test_cases_with_predefined_functions(questions)

# 3) Qwen prompt formatter
qwen = QwenHandler(
    model_name="Qwen/Qwen3-8B",
    temperature=0.0,
    registry_name="local",
    is_fc_model=False,
)

with ALPACA_OUT.open("w", encoding="utf-8") as f:
    for entry in questions:
        test_id = entry["id"]
        gt_turns = answers[test_id]
        messages = []

        # add system prompt to turn 0
        entry["question"][0] = system_prompt_pre_processing_chat_model(
            entry["question"][0], entry["function"], test_id
        )

        involved_classes = entry["involved_classes"]
        initial_config = entry["initial_config"]

        for t, user_msgs in enumerate(entry["question"]):
            messages.extend(user_msgs)

            turn_calls = gt_turns[t]
            output_str = to_python_list_str(turn_calls)

            # format prompt exactly as BFCL would for Qwen (includes <|im_start|>assistant\n)
            formatted_prompt = qwen._format_prompt(messages, entry["function"])

            f.write(
                json.dumps(
                    {
                        "instruction": formatted_prompt,
                        "input": "",
                        "output": output_str + "<|im_end|>",
                        "id": f"{test_id}_turn_{t}",
                    }
                )
                + "\n"
            )

            # execute tools and append tool outputs for next turn
            execution_results, _ = execute_multi_turn_func_call(
                func_call_list=turn_calls,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name="sft",
                test_entry_id=test_id,
                long_context=False,
                is_evaL_run=False,
            )

            for call_str, result in zip(turn_calls, execution_results):
                messages.append(
                    {
                        "role": "tool",
                        "name": call_str,
                        "content": result,
                    }
                )

print(f"Wrote train split: {TRAIN_OUT}")
print(f"Wrote test split: {TEST_OUT}")
print(f"Wrote alpaca train: {ALPACA_OUT}")
