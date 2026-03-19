from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import AttachmentSummary


TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_type": {"type": "string"},
        "operation": {"type": "string"},
        "language_hint": {"type": "string"},
        "confidence": {"type": "number"},
        "fields": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task_type", "operation", "language_hint", "confidence", "fields", "notes"],
}


SYSTEM_PROMPT = """You extract Tripletex accounting tasks into strict JSON.
Choose a task_type from:
- customer_create
- customer_update
- employee_create
- employee_update
- product_create
- product_update
- project_create
- project_update
- department_create
- department_update
- invoice_create
- payment_register
- travel_expense_delete
- voucher_reverse
- credit_note_create
- unknown

Use the fields object for extracted values such as:
name, first_name, last_name, email, phone, number, organization_number,
customer_name, customer_email, customer_number, product_name, project_name,
department_name, role_template, amount, quantity, price, invoice_number,
voucher_id, travel_expense_id, payment_type, date, due_date, comment,
description, start_date, end_date.

Return only valid JSON that matches the schema.
"""


class OpenAIExtractor:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def extract(self, prompt: str, attachments: list[AttachmentSummary]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        attachment_lines = []
        for item in attachments:
            line = f"- {item.filename} ({item.mime_type or 'unknown'}, {item.size_bytes} bytes)"
            if item.text_excerpt:
                line += f"\n  Extracted text:\n{item.text_excerpt[:1500]}"
            attachment_lines.append(line)
        user_prompt = prompt.strip()
        if attachment_lines:
            user_prompt += "\n\nAttachments:\n" + "\n".join(attachment_lines)

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "tripletex_task",
                    "schema": TASK_SCHEMA,
                    "strict": True,
                }
            },
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/responses",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
        except URLError as exc:
            raise RuntimeError(str(exc)) from exc
        text = self._extract_text(data)
        if not text:
            return None
        return json.loads(text)

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        texts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(content["text"])
        return "".join(texts).strip()
