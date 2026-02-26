#!/usr/bin/env python3
"""
Script to collect critic training data from multi-turn trajectories (base only).

For each state-action pair in expert trajectories:
- Extract state (observation + gold_prefix_history)
- Get positive_action from expert trajectory
- negative_action and negative_reasoning are set to null
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from bfcl_eval.constants.eval_config import *
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.constants.enums import ReturnFormat
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval._llm_response_generation import build_handler
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import (
    load_dataset_entry,
    load_ground_truth_entry,
    make_json_serializable,
)
from bfcl_eval.model_handler.utils import resolve_ast_call
from dotenv import load_dotenv


def format_functions_for_prompt(functions: List[Dict[str, Any]]) -> str:
    """
    Format function definitions for the system prompt (same as in convert_to_llamafactory_sft.py).
    
    Args:
        functions: List of function definition dictionaries
    
    Returns:
        Formatted string with function definitions
    """
    if not functions:
        return ""
    
    func_descriptions = []
    for func in functions:
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        
        # Format parameters
        params_str = ""
        if parameters and "properties" in parameters:
            props = parameters["properties"]
            required = parameters.get("required", [])
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required_marker = " (required)" if param_name in required else " (optional)"
                param_list.append(f"  - {param_name} ({param_type}): {param_desc}{required_marker}")
            params_str = "\n".join(param_list)
        
        func_descriptions.append(
            f"Function: {name}\n"
            f"Description: {description}\n"
            f"Parameters:\n{params_str}\n"
        )
    
    return "\n".join(func_descriptions)


def build_prompt_messages(
    message_history: List[Dict[str, Any]],
    functions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build prompt messages in the same format as SFT data.
    
    Args:
        message_history: List of messages (user, assistant, tool, etc.)
        functions: List of function definitions
    
    Returns:
        List of messages with system message containing formatted functions
    """
    # Build system message with function definitions
    system_content = "You are a helpful AI assistant that can call functions to help users complete tasks.\n\n"
    if functions:
        system_content += "Available functions:\n\n"
        system_content += format_functions_for_prompt(functions)
        system_content += "\n\n"
    system_content += (
        "When you need to call functions, output them in the following format:\n"
        "<tool_calls>\n"
        "[\n"
        '  {"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}},\n'
        "  ...\n"
        "]\n"
        "</tool_calls>\n"
        "You can call multiple functions in one turn if needed."
    )
    
    # Build messages array: system message + message_history
    messages = [
        {"role": "system", "content": system_content}
    ]
    
    # Add message history (filter out any existing system messages to avoid duplication)
    for msg in message_history:
        if msg.get("role") != "system":
            messages.append(msg)
    
    return messages




def build_state_from_instances(involved_instances: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build state (observation) from involved instances.
    This represents the current state of the environment.
    """
    state = {}
    for class_name, class_instance in involved_instances.items():
        # Skip stateless classes
        from bfcl_eval.constants.executable_backend_config import (
            STATELESS_CLASSES,
            OMIT_STATE_INFO_CLASSES,
        )
        if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
            continue
        # Get the state of the instance
        # Make it JSON serializable by converting complex objects to strings
        instance_state = {
            key: value
            for key, value in vars(class_instance).items()
            if not key.startswith("_")
        }
        # Make the state JSON serializable
        state[class_name] = make_json_serializable(instance_state)
    return state


def get_model_response(
    handler: BaseHandler,
    prompt_info: Dict[str, Any],
    test_entry_id: str,
    functions: List[Dict[str, Any]] = None,
) -> List[str]:
    """
    Get model response (negative action) for a given prompt.
    Uses the full conversation history from prompt_info to respect multi-turn context.
    
    Args:
        handler: Model handler instance
        prompt_info: Dictionary with 'messages' key (same format as SFT data)
                    Contains full conversation history: [system, user1, assistant1, tool1, user2, ...]
        test_entry_id: Test entry ID for context
        functions: Optional list of function definitions (if not provided, will try to extract from messages)
    
    Returns:
        List of function call strings (negative_action), or empty list if no function calls
    """
    try:
        # Extract messages from prompt_info (same format as SFT data)
        # This contains the full conversation history: system + all previous turns + current user message
        # Format: [system, user1, assistant1, tool1, user2, assistant2, tool2, user3, ...]
        # Note: prompt_info is built BEFORE adding the assistant response, so it already contains
        # the full conversation history up to the current user message (no assistant at the end)
        messages = deepcopy(prompt_info["messages"])
        
        # Remove any assistant message at the end if present (we're predicting it)
        # But keep all previous assistant/tool messages (they're part of the conversation history)
        # In normal flow, prompt_info shouldn't have an assistant at the end, but handle edge cases
        if messages and messages[-1].get("role") == "assistant":
            # Remove the last assistant message (we're predicting it)
            conversation_messages = messages[:-1]
        else:
            # No assistant at the end, use all messages
            conversation_messages = messages
        
        # Extract system message if present (handler will use its own system prompt, but we need functions)
        system_message = None
        non_system_messages = []
        for msg in conversation_messages:
            if msg.get("role") == "system":
                system_message = msg
            else:
                non_system_messages.append(msg)
        
        # If functions not provided, try to extract from system message content
        # (though handler will format them from test_entry["function"])
        if functions is None:
            functions = []
        
        # Create a test_entry for inference
        # Use a simple ID that won't trigger multi-turn handling
        # Must NOT contain "multi_turn" in the category part to avoid multi-turn detection
        simple_test_id = f"simple_{hash(test_entry_id) % 1000000}"
        
        # Pass the full conversation history (excluding system message, handler will add its own)
        # The handler's add_first_turn_message_prompting will add all these messages to inference_data
        test_entry_for_inference = {
            "id": simple_test_id,  # Simple ID to ensure single-turn inference
            "question": [non_system_messages],  # Full conversation history (system excluded, handler adds it)
            "function": functions,  # Function definitions for handler
            "initial_config": {},
            "involved_classes": [],
        }
        
        # Use single-turn inference methods directly
        # Note: QwenFCHandler is actually a prompting model (inherits from OSSHandler)
        # So we need to check the handler type, not just is_fc_model flag
        # OSSHandler always uses prompting methods, even for "FC" models
        if isinstance(handler, OSSHandler):
            # OSS models (including QwenFCHandler) use prompting inference
            model_responses, _ = handler.inference_single_turn_prompting(
                test_entry_for_inference,
                include_input_log=False,
            )
        elif handler.is_fc_model or "FC" in handler.registry_name:
            # True FC models (API-based) use FC inference
            model_responses, _ = handler.inference_single_turn_FC(
                test_entry_for_inference,
                include_input_log=False,
            )
        else:
            # Regular prompting models
            model_responses, _ = handler.inference_single_turn_prompting(
                test_entry_for_inference,
                include_input_log=False,
            )
        
        # Decode model responses to function call strings
        # model_responses format depends on model type:
        # - For FC models: list of function call dicts [{"function_name": {"arg1": "value1", ...}}, ...]
        # - For prompting models: string response that needs decoding
        from bfcl_eval.model_handler.utils import decoded_output_to_execution_list, default_decode_execute_prompting
        
        if isinstance(model_responses, list):
            # FC model response - already in dict format
            negative_action = decoded_output_to_execution_list(model_responses)
        elif isinstance(model_responses, str):
            # Prompting model response - need to decode
            if hasattr(handler, 'decode_execute'):
                try:
                    negative_action = handler.decode_execute(
                        model_responses,
                        has_tool_call_tag=True,
                    )
                except Exception as e:
                    # Fallback: try default decode
                    negative_action = default_decode_execute_prompting(
                        model_responses,
                        has_tool_call_tag=True,
                    )
            else:
                # Use default decode
                negative_action = default_decode_execute_prompting(
                    model_responses,
                    has_tool_call_tag=True,
                )
        else:
            # Unexpected format
            print(f"Warning: Unexpected model response format: {type(model_responses)}")
            negative_action = []
        
        return negative_action if negative_action else []
        
    except Exception as e:
        print(f"Error getting model response: {e}")
        import traceback
        traceback.print_exc()
        return []  # Return empty list on error


def collect_critic_data_for_task(
    test_entry: Dict[str, Any],
    ground_truth: List[List[str]],
    handler: Optional[BaseHandler] = None,
) -> List[Dict[str, Any]]:
    """
    Collect critic training data for a single task.
    
    Returns a list of data points, where each data point contains:
    - task_idx, task_name, variation_idx, prefix_len
    - state (observation + gold_prefix_history)
    - positive_action (from expert trajectory)
    - negative_action (null)
    - prompt_info (full prompt that will be fed to LLM)
    - negative_reasoning, positive_reasoning (null)
    """
    collected_data = []
    
    test_entry_id = test_entry["id"]
    initial_config = test_entry.get("initial_config", {})
    involved_classes = test_entry["involved_classes"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    
    # Extract task_idx and variation_idx from test_entry_id
    # Format: multi_turn_base_{task_idx} or multi_turn_base_{task_idx}_{variation_idx}
    parts = test_entry_id.split("_")
    task_idx = int(parts[-1]) if parts[-1].isdigit() else None
    variation_idx = 0  # Default, can be extracted if needed
    
    # Initialize message list for building prompts
    # We'll build the prompt structure manually without a handler
    message_history: List[Dict[str, Any]] = []
    
    # Initialize instances for state tracking
    # Use a simple model name for state tracking
    state_tracking_model_name = "critic_data_collection"
    _, involved_instances = execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        state_tracking_model_name,
        test_entry_id,
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )
    
    all_multi_turn_messages: List[List[Dict]] = test_entry["question"]
    
    # Track prefix length (number of previous turns)
    prefix_len = 0
    
    # Simulate expert trajectory turn by turn
    for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
        # Add user message to message history
        message_history.extend(current_turn_message)
        
        # Get ground truth actions for this turn
        turn_ground_truth = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
        
        # Process each step in the turn
        # Each turn_ground_truth is a list of function calls (strings) for that turn
        # We'll process all function calls in the turn as a single action
        if not turn_ground_truth:
            # No ground truth for this turn, skip
            prefix_len += 1
            continue
        
        # Get positive action from expert trajectory (all function calls in this turn)
        positive_action = turn_ground_truth  # This is a list of function call strings
        
        # Build state (observation + gold_prefix_history)
        # Observation: current state of instances
        observation = build_state_from_instances(involved_instances)
        
        # Gold prefix history: messages up to this point
        gold_prefix_history = deepcopy(message_history)
        
        state = {
            "observation": observation,
            "gold_prefix_history": gold_prefix_history,
        }
        
        # Build prompt_info (full prompt that will be fed to LLM)
        # Format it the same way as SFT data: messages array with system message containing functions
        functions = deepcopy(test_entry.get("function", []))
        prompt_messages = build_prompt_messages(message_history, functions)
        
        prompt_info = {
            "messages": prompt_messages,  # Same format as SFT data
        }
        
        # Get negative action from policy model (if handler is provided)
        if handler is not None:
            negative_action = get_model_response(
                handler,
                prompt_info,
                test_entry_id,
                functions=deepcopy(test_entry.get("function", [])),
            )
            # If no function calls were extracted, use empty list to match positive_action format
            if not negative_action:
                negative_action = []
        else:
            # Negative action is empty list if no handler provided (to match positive_action format)
            negative_action = []
        
        # Skip data point if negative_action and positive_action are the same
        # (no learning signal for critic if they match)
        if negative_action == positive_action:
            print(f"  Skipping data point: negative_action == positive_action for turn {turn_idx}")
            # Still need to execute positive action to update state for next turn
            # (so we continue with the expert trajectory)
        else:
            # Create data point only if actions are different
            data_point = {
                "task_idx": task_idx,
                "task_name": test_entry_id,
                "variation_idx": variation_idx,
                "prefix_len": prefix_len,
                "state": state,
                "positive_action": positive_action,
                "negative_action": negative_action,
                "negative_reasoning": None,
                "positive_reasoning": None,
                "prompt_info": prompt_info,
            }
            
            collected_data.append(data_point)
        
        # Execute positive action to update state for next turn (always, even if data point was skipped)
        execution_results, involved_instances = execute_multi_turn_func_call(
            positive_action,
            initial_config,
            involved_classes,
            state_tracking_model_name,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )
        
        # Add execution results to message history for next turn
        # Add assistant message with tool calls
        # Format tool calls as content string (same format as SFT data)
        # The handler expects 'content' field with formatted tool_calls
        import json
        import ast
        
        # Format function calls as tool_calls content (same as SFT data format)
        tool_calls_list = []
        for action in positive_action:
            try:
                # Parse function call string to extract name and arguments
                parsed = ast.parse(action, mode="eval")
                if isinstance(parsed.body, ast.Call):
                    func_dict = resolve_ast_call(parsed.body)
                    for func_name, args_dict in func_dict.items():
                        tool_calls_list.append({
                            "name": func_name,
                            "arguments": args_dict
                        })
            except Exception:
                # Fallback: simple parsing
                func_name = action.split("(")[0].split(".")[-1] if "(" in action else action.split(".")[-1] if "." in action else action
                tool_calls_list.append({
                    "name": func_name,
                    "arguments": {}
                })
        
        # Format as tool_calls XML tag (same format as SFT data)
        assistant_content = "<tool_calls>\n" + json.dumps(tool_calls_list, indent=2) + "\n</tool_calls>"
        
        assistant_message = {
            "role": "assistant",
            "content": assistant_content,  # Formatted tool_calls content (same as SFT data)
        }
        message_history.append(assistant_message)
        
        # Add tool responses
        for i, execution_result in enumerate(execution_results):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": f"call_{turn_idx}_{i}",
            }
            message_history.append(tool_message)
        
        prefix_len += 1
    
    return collected_data


def main():
    parser = argparse.ArgumentParser(description="Collect critic training data from multi-turn trajectories")
    parser.add_argument(
        "--output-file",
        type=str,
        default="critic_training_data.json",
        help="Output file path for collected data",
    )
    parser.add_argument(
        "--test-category",
        type=str,
        default="multi_turn_base",
        help="Test category to collect data from (should be multi_turn_base)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Optional: Limit the number of tasks to process (useful for testing)",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default=None,
        help="Policy model name (e.g., 'Qwen/Qwen3-8B-FC') to generate negative actions. If not provided, negative_action will be null. Uses same format as 'bfcl generate --model'.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="Disable thinking mode for policy model (if applicable)",
    )
    
    args = parser.parse_args()
    
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    
    # Initialize policy model handler if provided (same way as bfcl generate)
    handler = None
    if args.policy_model:
        print(f"Initializing policy model: {args.policy_model}")
        try:
            # Use the same build_handler function as bfcl generate command
            enable_thinking = not args.no_think
            handler = build_handler(
                model_name=args.policy_model,
                temperature=0.001,  # Same as bfcl generate default
                enable_thinking=enable_thinking,
            )
            
            # For OSS models, spin up the local server (same as bfcl generate)
            if isinstance(handler, OSSHandler):
                print(f"✓ Policy model handler initialized (OSS model)")
                print("Spinning up local server...")
                # Use default settings similar to bfcl generate
                handler.spin_up_local_server(
                    num_gpus=1,  # Default, can be made configurable
                    gpu_memory_utilization=0.9,  # Default
                    backend="vllm",  # Default, can be made configurable
                    skip_server_setup=False,  # Set up server
                    local_model_path=None,  # Use default HF_HOME
                )
                print("✓ Local server ready")
            else:
                print(f"✓ Policy model handler initialized (API model)")
        except Exception as e:
            print(f"Warning: Failed to initialize policy model: {e}")
            print("Continuing without policy model (negative_action will be empty list [])")
            import traceback
            traceback.print_exc()
            handler = None
    
    # Load test entries and ground truth
    print(f"Loading test entries for category: {args.test_category}")
    test_entries = load_dataset_entry(
        args.test_category,
        include_prereq=False,
        include_language_specific_hint=False,
    )
    ground_truth_entries = load_ground_truth_entry(args.test_category)
    
    # Create a mapping from test_entry_id to ground_truth
    ground_truth_map = {
        entry["id"]: entry["ground_truth"] for entry in ground_truth_entries
    }
    
    # Collect data for each task
    all_collected_data = []
    
    # Limit number of tasks if specified
    if args.num_tasks is not None:
        test_entries = test_entries[:args.num_tasks]
        print(f"Limited to processing {len(test_entries)} tasks (--num-tasks={args.num_tasks})")
    
    print(f"Collecting data from {len(test_entries)} tasks...")
    
    # Prepare output path
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process tasks and save periodically
    task_count = 0
    for test_entry in tqdm(test_entries, desc="Processing tasks"):
        test_entry_id = test_entry["id"]
        if test_entry_id not in ground_truth_map:
            print(f"Warning: No ground truth found for {test_entry_id}, skipping...")
            continue
        
        ground_truth = ground_truth_map[test_entry_id]
        
        try:
            collected_data = collect_critic_data_for_task(
                test_entry,
                ground_truth,
                handler=handler,
            )
            all_collected_data.extend(collected_data)
            task_count += 1
            print(f"Collected {len(collected_data)} data points from {test_entry_id}")
            
            # Save periodically after every 10 tasks
            if task_count % 10 == 0:
                # Make all data JSON serializable before saving
                serializable_data = [make_json_serializable(data_point) for data_point in all_collected_data]
                
                with open(output_path, "w") as f:
                    json.dump(serializable_data, f, indent=2)
                
                print(f"💾 Periodically saved {len(all_collected_data)} data points after {task_count} tasks")
        except Exception as e:
            print(f"Error processing {test_entry_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save of all collected data
    # Make all data JSON serializable before saving
    serializable_data = [make_json_serializable(data_point) for data_point in all_collected_data]
    
    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nCollected {len(all_collected_data)} data points")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

