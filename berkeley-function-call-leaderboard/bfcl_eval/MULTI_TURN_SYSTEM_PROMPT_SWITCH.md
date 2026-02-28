# Multi-turn System Prompt Switch

This repo now supports switching system-prompt style for **all `multi_turn_*` tasks** at runtime.

## What changed

- File changed: `bfcl_eval/model_handler/utils.py`
- `system_prompt_pre_processing_chat_model(...)` now checks whether the task category starts with `multi_turn_`.
- For `multi_turn_*` tasks, prompt style is selected by environment variable:
  - `original`: existing BFCL prompt construction (default behavior)
  - `short`: shorter BFCL-eval style prompt + compact function schema
- Non-`multi_turn_*` tasks are unchanged and always use the existing BFCL prompt flow.

## How to switch

### Recommended: per-run CLI flag

Use `bfcl generate` with an explicit flag so style does not depend on ambient shell env state:

```bash
bfcl generate \
  --model MODEL_NAME \
  --test-category multi_turn_base \
  --multi-turn-system-prompt-style short
```

or:

```bash
bfcl generate \
  --model MODEL_NAME \
  --test-category multi_turn_base \
  --multi-turn-system-prompt-style original
```

This also prints a startup log line:

```text
[BFCL_PROMPT_STYLE_CONFIG] source=cli BFCL_MULTI_TURN_SYSTEM_PROMPT_STYLE=short
```

### 1) Use original BFCL prompt (default)

```bash
unset BFCL_MULTI_TURN_SYSTEM_PROMPT_STYLE
```

or explicitly:

```bash
export BFCL_MULTI_TURN_SYSTEM_PROMPT_STYLE=original
```

### 2) Use shorter prompt for all `multi_turn_*` tasks

```bash
export BFCL_MULTI_TURN_SYSTEM_PROMPT_STYLE=short
```

Then run your normal BFCL eval/generation command in the same shell session.

## Notes

- If `BFCL_MULTI_TURN_SYSTEM_PROMPT_STYLE` is set to an unsupported value, code falls back to `original`.
- This switch is category-based (`multi_turn_*` prefix), so it applies to `multi_turn_base` and other multi-turn categories without extra code changes.
- Runtime verification logs for each multi-turn test entry:
  - `[BFCL_PROMPT_STYLE] test_entry_id=... selected_style=... short_prompt_applied=...`
