"""Prompt scaffolding for future model-assisted extraction.

The current implementation is fully deterministic and does not depend on an LLM.
These strings are kept as scaffolding so the project can later swap in a model
without changing the workflow contract.
"""

TASK_EXTRACTION_SYSTEM_PROMPT = (
    "You extract Tripletex task intent into strict JSON. "
    "Return only valid JSON with task_type, operation, entities, fields, language_hint, and confidence."
)

WORKFLOW_PLANNING_SYSTEM_PROMPT = (
    "You turn parsed Tripletex task intent into a deterministic workflow plan. "
    "Prefer the fewest API calls and keep prerequisite handling explicit."
)

