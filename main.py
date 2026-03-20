import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import os
import pickle
import re
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

load_dotenv()

APP_DIR = Path(__file__).resolve().parent


def resolve_app_path(value: Optional[str], default: Path) -> Path:
    path = Path(value) if value else default
    if path.is_absolute():
        return path
    return (APP_DIR / path).resolve()


def default_openapi_spec_path() -> Path:
    local_spec = APP_DIR / "openapi.json"
    if local_spec.exists():
        return local_spec
    return (APP_DIR.parent / "openapi.json").resolve()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def get_openai_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(api_key=api_key)

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAPI_SPEC_PATH = resolve_app_path(os.getenv("TRIPLETEX_OPENAPI_SPEC"), default_openapi_spec_path())
OPENAPI_CACHE_PATH = resolve_app_path(
    os.getenv("TRIPLETEX_OPENAPI_CACHE"),
    OPENAPI_SPEC_PATH.with_name(f"{OPENAPI_SPEC_PATH.stem}.registry-cache.pkl"),
)
APP_API_KEY = os.getenv("APP_API_KEY")
AUTO_VERIFY_WRITES = env_flag("AUTO_VERIFY_WRITES", default=False)
TRACE_LOGGING_ENABLED = env_flag("TRACE_LOGGING_ENABLED", default=True)
TRACE_LOG_PATH = resolve_app_path(
    os.getenv("TRACE_LOG_PATH"),
    APP_DIR / "logs" / "agent-trace.log",
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "12"))
MAX_AGENT_ATTEMPTS = int(os.getenv("MAX_AGENT_ATTEMPTS", "4"))
SOLVE_TIME_BUDGET_SECONDS = int(os.getenv("SOLVE_TIME_BUDGET_SECONDS", "285"))
MIN_SECONDS_LEFT_FOR_RETRY = int(os.getenv("MIN_SECONDS_LEFT_FOR_RETRY", "20"))
MAX_FILE_TEXT_CHARS = int(os.getenv("MAX_FILE_TEXT_CHARS", "12000"))
TRIPLETEX_TIMEOUT_SECONDS = int(os.getenv("TRIPLETEX_TIMEOUT_SECONDS", "60"))
OPENAPI_CACHE_VERSION = 2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripletex-agent")
trace_logger = logging.getLogger("tripletex-agent.trace")

if TRACE_LOGGING_ENABLED and not trace_logger.handlers:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    trace_handler = RotatingFileHandler(
        TRACE_LOG_PATH,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    trace_handler.setFormatter(logging.Formatter("%(message)s"))
    trace_logger.addHandler(trace_handler)
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False

app = FastAPI(title="Tripletex OpenAPI Agent")

_registry_cache = None
_registry_loading: asyncio.Future = None
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def strip_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_nones(v) for v in value if v is not None]
    return value


def parse_json_like_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    candidate = value.strip()
    if not candidate or candidate[0] not in "[{":
        return value

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return value


def truncate_for_log(value: Any, limit: int = 8000) -> Any:
    if isinstance(value, str):
        return truncate_text(value, limit)
    dumped = safe_json_dumps(value)
    if len(dumped) <= limit:
        return value
    return truncate_text(dumped, limit)


def redact_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, inner in value.items():
            lowered = str(key).lower()
            if lowered in {"authorization", "session_token", "password", "token", "api_key"}:
                sanitized[key] = redact_token(str(inner)) if inner is not None else None
            elif lowered == "content_base64":
                sanitized[key] = f"[base64 omitted, len={len(inner) if isinstance(inner, str) else 0}]"
            else:
                sanitized[key] = sanitize_for_log(inner)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    return truncate_for_log(value)


def log_trace(event: str, **payload: Any) -> None:
    if not TRACE_LOGGING_ENABLED:
        return

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        **{k: sanitize_for_log(v) for k, v in payload.items()},
    }
    trace_logger.info(safe_json_dumps(record))


def summarize_function_calls(response: Any) -> List[Dict[str, Any]]:
    calls = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        calls.append(
            {
                "name": getattr(item, "name", None),
                "call_id": getattr(item, "call_id", None),
                "arguments": parse_json_like_string(getattr(item, "arguments", "") or ""),
            }
        )
    return calls


VALID_AGENT_STATUSES = {"completed", "failed", "needs_input"}
NON_INPUT_TOP_LEVEL_FIELDS = {"id", "version", "changes", "url"}
QUESTION_PATTERNS = (
    "kan du",
    "could you",
    "please confirm",
    "bekrefte",
    "confirm",
    "trenger mer informasjon",
    "need more information",
)
UNCERTAINTY_PATTERNS = (
    "ikke sikker",
    "usikker",
    "fant ikke",
    "finner ingen",
    "klarte ikke",
    "kunne ikke",
    "could not",
    "unable to",
    "not found",
    "no matching",
    "no open",
    "ingen åpne",
    "ikke funnet",
    "unknown",
)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def guess_entity_from_operation(operation: Dict[str, Any]) -> str:
    path = str(operation.get("path", "")).strip("/")
    if not path:
        return "other"
    first = path.split("/", 1)[0]
    mapping = {
        "customer": "customer",
        "employee": "employee",
        "invoice": "invoice",
        "payment": "payment",
        "travelExpense": "travelExpense",
    }
    return mapping.get(first, "other")


def infer_created_or_changed_from_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inferred = []
    for entry in trace:
        if entry.get("tool") != "call_tripletex_operation":
            continue
        result = entry.get("result", {})
        operation = result.get("operation", {})
        if not result.get("ok") or not isinstance(operation, dict):
            continue

        method = operation.get("method")
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            continue

        entity_id = find_first_id(result.get("response"))
        status = {
            "POST": "created",
            "PUT": "updated",
            "PATCH": "updated",
            "DELETE": "deleted",
        }.get(method, "unknown")
        inferred.append(
            {
                "entity": guess_entity_from_operation(operation),
                "id": entity_id,
                "status": status,
            }
        )
    return inferred


def canonicalize_agent_result(result: Dict[str, Any], fallback_text: str, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = result.get("status")
    if status not in VALID_AGENT_STATUSES:
        status = "completed" if infer_created_or_changed_from_trace(trace) else "failed"

    summary = str(result.get("summary") or "").strip()
    if not summary:
        summary = fallback_text.strip() or ("Task completed." if status == "completed" else "Task failed.")

    created_or_changed = result.get("created_or_changed")
    if not isinstance(created_or_changed, list):
        created_or_changed = infer_created_or_changed_from_trace(trace)

    notes = result.get("notes")
    if not isinstance(notes, list):
        notes = []

    return {
        "status": status,
        "summary": summary,
        "created_or_changed": created_or_changed,
        "notes": [str(note) for note in notes],
    }


def normalize_final_agent_output(final_text: str, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = extract_json_object(final_text)
    if parsed is not None:
        return canonicalize_agent_result(parsed, final_text, trace)

    inferred = infer_created_or_changed_from_trace(trace)
    if inferred:
        return {
            "status": "completed",
            "summary": final_text.strip() or "Task completed.",
            "created_or_changed": inferred,
            "notes": [],
        }

    cleaned = " ".join((final_text or "").strip().split())
    return {
        "status": "failed",
        "summary": cleaned or "Agent could not complete the task.",
        "created_or_changed": [],
        "notes": [],
    }


def result_requests_clarification(result: Dict[str, Any], raw_text: str) -> bool:
    texts = [str(result.get("summary") or ""), raw_text]
    texts.extend(str(note) for note in result.get("notes", []) if note is not None)
    combined = normalize_text(" ".join(texts))
    return any(pattern in combined for pattern in QUESTION_PATTERNS) or "?" in " ".join(texts)


def apply_search_defaults(op: "OperationSpec", query_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(query_params)

    if op.method != "GET":
        return params

    if op.path == "/customer":
        if "name" in params and "customerName" not in params:
            params["customerName"] = params.pop("name")
        params.setdefault("count", 100)
        return params

    if op.path == "/invoice":
        today = date.today()
        params.setdefault("count", 100)
        params.setdefault("fields", "*")
        params.setdefault("sorting", "-invoiceDate")
        params.setdefault("invoiceDateTo", today.isoformat())
        params.setdefault("invoiceDateFrom", (today - timedelta(days=730)).isoformat())
        return params

    return params


def summarize_trace_for_retry(trace: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for entry in trace[-limit:]:
        tool = entry.get("tool")
        arguments = entry.get("arguments", {}) or {}
        result = entry.get("result", {}) or {}
        if tool == "call_tripletex_operation":
            summary = {
                "tool": tool,
                "operation_id": arguments.get("operation_id"),
                "ok": result.get("ok"),
                "status_code": result.get("status_code"),
            }
            if not result.get("ok"):
                summary["error"] = result.get("error") or result.get("response")
            summarized.append(summary)
        elif tool == "search_openapi_operations":
            matches = result.get("matches", []) if isinstance(result, dict) else []
            summarized.append(
                {
                    "tool": tool,
                    "query": arguments.get("query"),
                    "top_match_operation_id": matches[0].get("operation_id") if matches else None,
                }
            )
        else:
            summarized.append({"tool": tool, "arguments": arguments, "result": result})
    return summarized


def result_is_unsuccessful_or_uncertain(result: Dict[str, Any], raw_text: str, trace: List[Dict[str, Any]]) -> bool:
    if result.get("status") != "completed":
        return True
    if result_requests_clarification(result, raw_text):
        return True

    texts = [str(result.get("summary") or ""), raw_text]
    texts.extend(str(note) for note in result.get("notes", []) if note is not None)
    combined = normalize_text(" ".join(texts))
    if any(pattern in combined for pattern in UNCERTAINTY_PATTERNS):
        return True

    return False


def build_agent_input(
    prompt: str,
    file_context: Optional[str],
    attempt: int,
    previous_result: Optional[Dict[str, Any]] = None,
    previous_trace: Optional[List[Dict[str, Any]]] = None,
) -> str:
    parts = [
        "User prompt:",
        prompt,
        "",
        "Extracted file context:",
        file_context or "[no files]",
    ]

    if attempt > 1 and previous_result is not None:
        retry_context = {
            "attempt": attempt,
            "previous_result": previous_result,
            "recent_tool_outcomes": summarize_trace_for_retry(previous_trace or []),
        }
        parts.extend(
            [
                "",
                f"Retry attempt {attempt}.",
                "Previous attempt failed or was uncertain. Start over from scratch, but avoid repeating these mistakes:",
                safe_json_dumps(retry_context),
            ]
        )

    return "\n".join(parts)


def sanitize_filename(name: str) -> str:
    safe = Path(name or "unnamed").name
    if not safe or safe in (".", "..") or "\x00" in safe:
        safe = "unnamed"
    return safe


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_æøåÆØÅ-]+", normalize_text(text))


def resolve_json_pointer(doc: Any, pointer: str) -> Any:
    if not pointer.startswith("#/"):
        raise ValueError(f"Only local refs are supported, got: {pointer}")
    current = doc
    for part in pointer[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[part]
    return current


def deref_local_refs(
    obj: Any,
    root: Any,
    ref_cache: Optional[Dict[str, Any]] = None,
    seen: Optional[set] = None,
    depth: int = 0,
    max_depth: int = 8,
) -> Any:
    if depth >= max_depth:
        return obj
    if ref_cache is None:
        ref_cache = {}
    if seen is None:
        seen = set()

    if isinstance(obj, dict):
        if "$ref" in obj:
            ref = obj["$ref"]
            if ref in seen:
                return {}
            if ref in ref_cache:
                resolved = ref_cache[ref]
            else:
                target = resolve_json_pointer(root, ref)
                resolved = deref_local_refs(target, root, ref_cache, seen | {ref}, depth + 1, max_depth)
                ref_cache[ref] = resolved
            merged = {}
            if isinstance(resolved, dict):
                merged.update(resolved)
            for k, v in obj.items():
                if k != "$ref":
                    merged[k] = deref_local_refs(v, root, ref_cache, seen, depth + 1, max_depth)
            return merged

        return {k: deref_local_refs(v, root, ref_cache, seen, depth + 1, max_depth) for k, v in obj.items()}

    if isinstance(obj, list):
        return [deref_local_refs(v, root, ref_cache, seen, depth + 1, max_depth) for v in obj]

    return obj


def merge_object_schemas(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True,
    }
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "object" or "properties" in part:
            merged["properties"].update(part.get("properties", {}))
            merged["required"].extend(part.get("required", []))
            if part.get("additionalProperties") is False:
                merged["additionalProperties"] = False
    merged["required"] = sorted(set(merged["required"]))
    if not merged["required"]:
        merged.pop("required", None)
    return merged


def normalize_schema(schema: Optional[Dict[str, Any]], depth: int = 0) -> Dict[str, Any]:
    if not schema or depth > 6:
        return {"type": "object", "additionalProperties": True}

    if "allOf" in schema:
        parts = [normalize_schema(part, depth + 1) for part in schema["allOf"]]
        schema = merge_object_schemas(parts)

    if "oneOf" in schema:
        non_null = [s for s in schema["oneOf"] if s.get("type") != "null"]
        return normalize_schema(non_null[0] if non_null else {}, depth + 1)

    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return normalize_schema(non_null[0] if non_null else {}, depth + 1)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), "string")
    if not schema_type:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"
        else:
            return {"type": "object", "additionalProperties": True}

    out: Dict[str, Any] = {"type": schema_type}
    for key in (
        "description",
        "enum",
        "format",
        "pattern",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "readOnly",
        "writeOnly",
        "default",
    ):
        if key in schema:
            out[key] = schema[key]

    if schema_type == "object":
        props = {}
        for name, sub_schema in schema.get("properties", {}).items():
            props[name] = normalize_schema(sub_schema, depth + 1)
        out["properties"] = props
        required = [x for x in schema.get("required", []) if x in props]
        if required:
            out["required"] = required
        out["additionalProperties"] = schema.get("additionalProperties", True)
        return out

    if schema_type == "array":
        out["items"] = normalize_schema(schema.get("items", {}), depth + 1)
        return out

    if schema_type not in {"string", "integer", "number", "boolean"}:
        return {"type": "string"}

    return out


def required_fields(schema: Optional[Dict[str, Any]]) -> List[str]:
    if not schema or schema.get("type") != "object":
        return []
    return list(schema.get("required", []))


def property_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    if not schema or schema.get("type") != "object":
        return []
    return list(schema.get("properties", {}).keys())


def writable_property_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    if not schema or schema.get("type") != "object":
        return []
    return [
        name
        for name, sub_schema in schema.get("properties", {}).items()
        if name not in NON_INPUT_TOP_LEVEL_FIELDS and not sub_schema.get("readOnly")
    ]


def read_only_property_names(schema: Optional[Dict[str, Any]]) -> List[str]:
    if not schema or schema.get("type") != "object":
        return []
    return [
        name
        for name, sub_schema in schema.get("properties", {}).items()
        if name in NON_INPUT_TOP_LEVEL_FIELDS or sub_schema.get("readOnly")
    ]


def required_writable_fields(schema: Optional[Dict[str, Any]]) -> List[str]:
    writable = set(writable_property_names(schema))
    return [name for name in required_fields(schema) if name in writable]


def media_type_matches_json(media_type: str) -> bool:
    normalized = str(media_type or "").split(";", 1)[0].strip().lower()
    return normalized == "application/json" or normalized.endswith("+json") or normalized == "*/*"


def pick_json_content(content: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not isinstance(content, dict):
        return None, None

    preferred = (
        "application/json",
        "application/json; charset=utf-8",
        "application/*+json",
        "*/*",
    )
    for candidate in preferred:
        if candidate in content:
            entry = content[candidate]
            return candidate, entry if isinstance(entry, dict) else None

    for key, value in content.items():
        if media_type_matches_json(key):
            return key, value if isinstance(value, dict) else None

    return None, None


def preview_schema(schema: Optional[Dict[str, Any]], max_depth: int = 2, depth: int = 0) -> Optional[Dict[str, Any]]:
    if not schema:
        return None

    out: Dict[str, Any] = {"type": schema.get("type", "object")}
    for key in ("description", "enum", "format", "default", "readOnly", "writeOnly"):
        if key in schema:
            out[key] = schema[key]

    if schema.get("type") == "object":
        props = {}
        for name, sub_schema in schema.get("properties", {}).items():
            if depth == 0 and name in NON_INPUT_TOP_LEVEL_FIELDS:
                continue
            if depth >= max_depth:
                props[name] = {"type": sub_schema.get("type", "object")}
                if sub_schema.get("readOnly"):
                    props[name]["readOnly"] = True
            else:
                props[name] = preview_schema(sub_schema, max_depth=max_depth, depth=depth + 1)
        out["properties"] = props
        if "required" in schema:
            out["required"] = list(schema["required"])
        out["additionalProperties"] = schema.get("additionalProperties", True)
        return out

    if schema.get("type") == "array":
        item_schema = schema.get("items")
        if depth >= max_depth:
            out["items"] = {"type": item_schema.get("type", "object")} if isinstance(item_schema, dict) else {"type": "object"}
        else:
            out["items"] = preview_schema(item_schema, max_depth=max_depth, depth=depth + 1)
        return out

    return out


def operation_write_hints(op: "OperationSpec") -> List[str]:
    hints: List[str] = []

    if op.operation_id == "Invoice_post":
        hints.extend(
            [
                "Use body.customer, invoiceDate, invoiceDueDate, and orders.",
                "Do not send lines, invoiceLines, orderLines, customerId, or dueDate as top-level invoice fields.",
                "To invoice an existing order, use orders: [{id: <order_id>}] or the PUT /order/{id}/:invoice operation.",
            ]
        )
    elif op.operation_id in {"Order_post", "OrderList_postList"}:
        hints.extend(
            [
                "Use nested customer: {id: ...}, not customerId.",
                "Provide orderDate and deliveryDate when creating orders.",
                "New lines belong in orderLines.",
            ]
        )
    elif op.operation_id == "OrderOrderline_post":
        hints.extend(
            [
                "Use nested order: {id: ...}, not orderId.",
                "Use count, not quantity.",
                "Use unitPriceExcludingVatCurrency or unitPriceIncludingVatCurrency, not plain unitPrice.",
            ]
        )

    return hints


def normalize_order_line_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(body)

    if "count" not in normalized and "quantity" in normalized:
        normalized["count"] = normalized.pop("quantity")

    if "unitPriceExcludingVatCurrency" not in normalized and "unitPrice" in normalized and isinstance(normalized["unitPrice"], (int, float)):
        normalized["unitPriceExcludingVatCurrency"] = normalized.pop("unitPrice")

    if "vatType" not in normalized:
        if isinstance(normalized.get("vatTypeId"), int):
            normalized["vatType"] = {"id": normalized.pop("vatTypeId")}
        elif isinstance(normalized.get("vatId"), int):
            normalized["vatType"] = {"id": normalized.pop("vatId")}
        elif isinstance(normalized.get("vatPercent"), (int, float)):
            normalized["vatType"] = {"percentage": normalized.pop("vatPercent")}

    if "order" not in normalized and isinstance(normalized.get("orderId"), int):
        normalized["order"] = {"id": normalized.pop("orderId")}

    return normalized


def normalize_tripletex_body(op: "OperationSpec", body: Any) -> Any:
    if isinstance(body, list):
        item_schema = op.body_schema.get("items") if op.body_schema and op.body_schema.get("type") == "array" else None
        if not isinstance(item_schema, dict):
            return body

        item_operation_id = {
            "OrderList_postList": "Order_post",
            "InvoiceList_postList": "Invoice_post",
        }.get(op.operation_id, op.operation_id)
        item_op = OperationSpec(
            operation_id=item_operation_id,
            method=op.method,
            path=op.path,
            summary=op.summary,
            description=op.description,
            tags=op.tags,
            path_schema=op.path_schema,
            query_schema=op.query_schema,
            body_schema=item_schema,
            content_type=op.content_type,
        )
        return [normalize_tripletex_body(item_op, item) for item in body]

    if not isinstance(body, dict):
        return body

    normalized = dict(body)
    body_props = op.body_schema.get("properties", {}) if op.body_schema and op.body_schema.get("type") == "object" else {}

    if "customer" in body_props and "customer" not in normalized and isinstance(normalized.get("customerId"), int):
        normalized["customer"] = {"id": normalized.pop("customerId")}

    if "order" in body_props and "order" not in normalized and isinstance(normalized.get("orderId"), int):
        normalized["order"] = {"id": normalized.pop("orderId")}

    if "orderLines" in body_props:
        if "orderLines" not in normalized and "lines" in normalized:
            normalized["orderLines"] = normalized.pop("lines")
        if "orderLines" not in normalized and "invoiceLines" in normalized:
            normalized["orderLines"] = normalized.pop("invoiceLines")
        if isinstance(normalized.get("orderLines"), list):
            normalized["orderLines"] = [
                normalize_order_line_payload(item) if isinstance(item, dict) else item
                for item in normalized["orderLines"]
            ]

    if op.operation_id == "Invoice_post":
        if "invoiceDueDate" not in normalized and "dueDate" in normalized:
            normalized["invoiceDueDate"] = normalized.pop("dueDate")

        if "orders" not in normalized:
            derived_order: Dict[str, Any] = {}
            if isinstance(normalized.get("order"), dict):
                derived_order.update(normalized.pop("order"))

            for field in ("orderDate", "deliveryDate"):
                if field in normalized and field not in derived_order:
                    derived_order[field] = normalized.pop(field)

            if "orderLines" in normalized:
                derived_order.setdefault("orderLines", normalized.pop("orderLines"))
            elif "lines" in normalized:
                derived_order.setdefault("orderLines", normalized.pop("lines"))
            elif "invoiceLines" in normalized:
                derived_order.setdefault("orderLines", normalized.pop("invoiceLines"))

            if isinstance(derived_order.get("orderLines"), list):
                derived_order["orderLines"] = [
                    normalize_order_line_payload(item) if isinstance(item, dict) else item
                    for item in derived_order["orderLines"]
                ]

            if derived_order:
                today = date.today().isoformat()
                derived_order.setdefault("orderDate", today)
                derived_order.setdefault("deliveryDate", derived_order["orderDate"])
                if "customer" in normalized and "customer" not in derived_order:
                    derived_order["customer"] = normalized["customer"]
                normalized["orders"] = [derived_order]

    if op.operation_id in {"Order_post", "OrderList_postList"}:
        today = date.today().isoformat()
        normalized.setdefault("orderDate", today)
        normalized.setdefault("deliveryDate", normalized["orderDate"])

        if isinstance(normalized.get("orderLines"), list):
            normalized["orderLines"] = [
                normalize_order_line_payload(item) if isinstance(item, dict) else item
                for item in normalized["orderLines"]
            ]

    if op.operation_id == "OrderOrderline_post":
        normalized = normalize_order_line_payload(normalized)

    return normalized


def find_first_id(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        if "id" in value and isinstance(value["id"], int):
            return value["id"]
        for key in ("value", "data", "result"):
            if key in value:
                found = find_first_id(value[key])
                if found is not None:
                    return found
        for v in value.values():
            found = find_first_id(v)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_first_id(item)
            if found is not None:
                return found
    return None


def extract_file_text(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        if PdfReader is None:
            return "[PDF received but pypdf is not installed]"
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip()
        except Exception as exc:
            return f"[Failed to parse PDF: {exc}]"

    if suffix in {".txt", ".md", ".json", ".csv", ".xml", ".html"}:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return "[Text-like file received but could not decode]"

    return f"[Binary file received: {filename}]"


# -----------------------------------------------------------------------------
# OpenAPI registry
# -----------------------------------------------------------------------------

@dataclass
class OperationSpec:
    operation_id: str
    method: str
    path: str
    summary: str
    description: str
    tags: List[str]
    path_schema: Dict[str, Any]
    query_schema: Dict[str, Any]
    body_schema: Optional[Dict[str, Any]]
    content_type: Optional[str]


class OpenAPISpecRegistry:
    def __init__(self):
        self.operations: Dict[str, OperationSpec] = {}
        self.operations_by_path_method: Dict[Tuple[str, str], OperationSpec] = {}
        self._search_haystacks: Dict[str, str] = {}

    @classmethod
    def from_file(cls, path: Path) -> "OpenAPISpecRegistry":
        if not path.exists():
            raise FileNotFoundError(f"OpenAPI spec not found: {path}")
        raw = json.loads(path.read_bytes())
        return cls.from_spec(raw)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "OpenAPISpecRegistry":
        registry = cls()
        registry._build_index(spec)
        return registry

    @classmethod
    def from_cache_payload(cls, payload: Dict[str, Any]) -> "OpenAPISpecRegistry":
        registry = cls()
        for item in payload.get("operations", []):
            registry._register_operation(OperationSpec(**item))
        return registry

    def to_cache_payload(self) -> Dict[str, Any]:
        return {
            "operations": [asdict(op) for op in self.operations.values()],
        }

    def _build_search_haystack(self, op: OperationSpec) -> str:
        return normalize_text(
            " ".join(
                [
                    op.operation_id,
                    op.method,
                    op.path,
                    op.summary,
                    op.description,
                    " ".join(op.tags),
                    " ".join(property_names(op.path_schema)),
                    " ".join(property_names(op.query_schema)),
                    " ".join(property_names(op.body_schema)),
                    " ".join(operation_write_hints(op)),
                ]
            )
        )

    def _register_operation(self, op: OperationSpec) -> None:
        self.operations[op.operation_id] = op
        self.operations_by_path_method[(op.path, op.method)] = op
        self._search_haystacks[op.operation_id] = self._build_search_haystack(op)

    def _build_index(self, spec: Dict[str, Any]) -> None:
        paths = spec.get("paths", {})
        seen_names = set()
        ref_cache: Dict[str, Any] = {}

        for path, path_item in paths.items():
            path_level_params = path_item.get("parameters", []) if isinstance(path_item, dict) else []

            for method in ("get", "post", "put", "patch", "delete"):
                op = path_item.get(method) if isinstance(path_item, dict) else None
                if not isinstance(op, dict):
                    continue

                operation_id = op.get("operationId") or f"{method}_{path}"
                operation_id = re.sub(r"[^a-zA-Z0-9_]+", "_", operation_id).strip("_")
                if operation_id in seen_names:
                    suffix = 2
                    while f"{operation_id}_{suffix}" in seen_names:
                        suffix += 1
                    operation_id = f"{operation_id}_{suffix}"
                seen_names.add(operation_id)

                combined_params = {}
                for p in path_level_params + op.get("parameters", []):
                    if not isinstance(p, dict):
                        continue
                    combined_params[(p.get("in"), p.get("name"))] = p

                path_schema = {"type": "object", "properties": {}, "additionalProperties": True}
                query_schema = {"type": "object", "properties": {}, "additionalProperties": True}
                path_required = []
                query_required = []

                for (param_in, name), param in combined_params.items():
                    raw_schema = deref_local_refs(param.get("schema", {}), spec, ref_cache=ref_cache)
                    schema = normalize_schema(raw_schema)
                    if "description" not in schema and param.get("description"):
                        schema["description"] = param["description"]

                    if param_in == "path":
                        path_schema["properties"][name] = schema
                        if param.get("required", True):
                            path_required.append(name)
                    elif param_in == "query":
                        query_schema["properties"][name] = schema
                        if param.get("required", False):
                            query_required.append(name)

                if path_required:
                    path_schema["required"] = path_required
                if query_required:
                    query_schema["required"] = query_required

                body_schema = None
                content_type = None
                request_body = deref_local_refs(op.get("requestBody") or {}, spec, ref_cache=ref_cache)
                content = request_body.get("content", {}) if isinstance(request_body, dict) else {}
                content_type, content_entry = pick_json_content(content)
                if content_type and isinstance(content_entry, dict):
                    raw_body_schema = deref_local_refs(content_entry.get("schema", {}), spec, ref_cache=ref_cache)
                    body_schema = normalize_schema(raw_body_schema)
                else:
                    body_schema = None

                op_spec = OperationSpec(
                    operation_id=operation_id,
                    method=method.upper(),
                    path=path,
                    summary=op.get("summary", "") or "",
                    description=op.get("description", "") or "",
                    tags=op.get("tags", []) or [],
                    path_schema=path_schema,
                    query_schema=query_schema,
                    body_schema=body_schema,
                    content_type=content_type,
                )
                self._register_operation(op_spec)

    def search(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        top_k = max(1, min(top_k, 12))
        q = normalize_text(query)
        q_tokens = set(tokenize(query))

        scored = []
        for op in self.operations.values():
            haystack_norm = self._search_haystacks[op.operation_id]

            score = 0
            if q and q in haystack_norm:
                score += 10
            for token in q_tokens:
                if token in haystack_norm:
                    score += 2

            if "create" in q_tokens and op.method == "POST":
                score += 3
            if any(t in q_tokens for t in {"update", "modify", "change", "add"}) and op.method in {"PUT", "PATCH"}:
                score += 3
            if any(t in q_tokens for t in {"delete", "remove"}) and op.method == "DELETE":
                score += 3
            if any(t in q_tokens for t in {"find", "search", "get", "list"}) and op.method == "GET":
                score += 3

            scored.append((score, op))

        scored.sort(key=lambda x: (x[0], x[1].operation_id), reverse=True)

        matches = [self.preview(op) for score, op in scored[:top_k] if score > 0]
        if not matches:
            matches = [self.preview(op) for _, op in list(scored)[:top_k]]

        return {
            "query": query,
            "matches": matches,
            "count": len(matches),
        }

    def preview(self, op: OperationSpec) -> Dict[str, Any]:
        return {
            "operation_id": op.operation_id,
            "method": op.method,
            "path": op.path,
            "summary": op.summary,
            "description": truncate_text(op.description, 300),
            "tags": op.tags,
            "required_path_params": required_fields(op.path_schema),
            "required_query_params": required_fields(op.query_schema),
            "required_body_fields": required_writable_fields(op.body_schema),
            "body_fields": writable_property_names(op.body_schema),
            "read_only_body_fields": read_only_property_names(op.body_schema),
            "write_hints": operation_write_hints(op),
        }

    def get_operation_schema(self, operation_id: str) -> Dict[str, Any]:
        op = self.operations.get(operation_id)
        if not op:
            return {"ok": False, "error": f"Unknown operation_id: {operation_id}"}

        return {
            "ok": True,
            "operation": self.preview(op),
            "schemas": {
                "path_params": preview_schema(op.path_schema),
                "query_params": preview_schema(op.query_schema),
                "body": preview_schema(op.body_schema, max_depth=3),
            },
        }

    def _validate_required(self, schema: Optional[Dict[str, Any]], payload: Optional[Dict[str, Any]], scope: str) -> List[str]:
        missing = []
        if not schema or schema.get("type") != "object":
            return missing

        payload = payload or {}
        for field in schema.get("required", []):
            if schema.get("properties", {}).get(field, {}).get("readOnly"):
                continue
            if field not in payload or payload[field] in (None, ""):
                missing.append(f"{scope}.{field}")
        return missing

    def _build_url(self, base_url: str, path_template: str, path_params: Dict[str, Any]) -> str:
        url_path = path_template
        placeholders = re.findall(r"{([^}]+)}", path_template)
        for name in placeholders:
            if name not in path_params:
                raise ValueError(f"Missing path parameter: {name}")
            url_path = url_path.replace("{" + name + "}", quote(str(path_params[name]), safe=""))
        return base_url.rstrip("/") + url_path

    def _parse_response(self, response: requests.Response) -> Any:
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                return response.json()
            except Exception:
                pass
        return {"raw": response.text}

    def _find_verify_operation(self, op: OperationSpec, request_args: Dict[str, Any], response_payload: Any) -> Tuple[Optional[OperationSpec], Dict[str, Any]]:
        # POST /thing -> try GET /thing/{id}
        if op.method == "POST":
            new_id = find_first_id(response_payload)
            if new_id is not None:
                candidate_path = op.path.rstrip("/") + "/{id}"
                get_op = self.operations_by_path_method.get((candidate_path, "GET"))
                if get_op:
                    path_fields = required_fields(get_op.path_schema)
                    if len(path_fields) == 1:
                        return get_op, {"path_params": {path_fields[0]: new_id}, "query_params": {}, "body": None}

        # PUT/PATCH/DELETE /thing/{id} -> try GET same path
        if op.method in {"PUT", "PATCH", "DELETE"}:
            get_op = self.operations_by_path_method.get((op.path, "GET"))
            if get_op:
                return get_op, {
                    "path_params": request_args.get("path_params", {}) or {},
                    "query_params": {},
                    "body": None,
                }

        return None, {}

    def execute(
        self,
        base_url: str,
        auth: Tuple[str, str],
        operation_id: str,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        reason: str = "",
        allow_verify: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        op = self.operations.get(operation_id)
        if not op:
            return {"ok": False, "error": f"Unknown operation_id: {operation_id}"}

        path_params = strip_nones(path_params or {})
        query_params = strip_nones(query_params or {})
        body = strip_nones(parse_json_like_string(body))
        body = strip_nones(normalize_tripletex_body(op, body))

        # Rescue a common model mistake: sending GET filters as body instead of query params.
        if op.method == "GET" and isinstance(body, dict) and not query_params:
            query_params = body
            body = None
        query_params = apply_search_defaults(op, query_params)

        missing = []
        missing.extend(self._validate_required(op.path_schema, path_params, "path_params"))
        missing.extend(self._validate_required(op.query_schema, query_params, "query_params"))
        if op.body_schema and op.body_schema.get("type") == "object":
            missing.extend(self._validate_required(op.body_schema, body if isinstance(body, dict) else {}, "body"))

        if missing:
            return {
                "ok": False,
                "status_code": 0,
                "error": {
                    "message": "Missing required fields before request",
                    "missing": missing,
                },
                "operation": self.preview(op),
            }

        try:
            url = self._build_url(base_url, op.path, path_params)
        except ValueError as exc:
            return {
                "ok": False,
                "status_code": 0,
                "error": {"message": str(exc)},
                "operation": self.preview(op),
            }

        headers = {"Accept": "application/json"}
        request_kwargs: Dict[str, Any] = {
            "method": op.method,
            "url": url,
            "auth": auth,
            "params": query_params or None,
            "headers": headers,
            "timeout": TRIPLETEX_TIMEOUT_SECONDS,
        }

        if body is not None and op.method in {"POST", "PUT", "PATCH"}:
            request_kwargs["json"] = body
            headers["Content-Type"] = "application/json"

        logger.info("Tripletex %s %s", op.method, url)
        response = requests.request(**request_kwargs)
        payload = self._parse_response(response)
        log_trace(
            "tripletex_api_call",
            request_id=request_id,
            operation_id=operation_id,
            method=op.method,
            url=url,
            reason=reason,
            query_params=query_params,
            body=body,
            status_code=response.status_code,
            response=payload,
        )

        result = {
            "ok": response.status_code < 400,
            "status_code": response.status_code,
            "reason": reason,
            "operation": self.preview(op),
            "request": {
                "path_params": path_params,
                "query_params": query_params,
                "body": truncate_text(safe_json_dumps(body), 4000) if body is not None else None,
            },
            "response": payload,
        }

        if allow_verify and result["ok"] and op.method in {"POST", "PUT", "PATCH", "DELETE"}:
            verify_op, verify_args = self._find_verify_operation(
                op,
                {"path_params": path_params, "query_params": query_params, "body": body},
                payload,
            )
            if verify_op:
                verify_result = self.execute(
                    base_url=base_url,
                    auth=auth,
                    operation_id=verify_op.operation_id,
                    path_params=verify_args.get("path_params"),
                    query_params=verify_args.get("query_params"),
                    body=verify_args.get("body"),
                    reason="Automatic post-write verification",
                    allow_verify=False,
                    request_id=request_id,
                )
                result["verification"] = verify_result

        return result


def _registry_cache_meta(spec_path: Path) -> Dict[str, Any]:
    stat = spec_path.stat()
    return {
        "cache_version": OPENAPI_CACHE_VERSION,
        "python_version": list(sys.version_info[:2]),
        "source_path": str(spec_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _load_registry_from_cache(spec_path: Path, cache_path: Path) -> Optional[OpenAPISpecRegistry]:
    if not cache_path.exists():
        return None

    expected_meta = _registry_cache_meta(spec_path)
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        logger.warning("Ignoring unreadable OpenAPI cache %s: %s", cache_path, exc)
        return None

    if not isinstance(payload, dict) or payload.get("meta") != expected_meta:
        return None

    registry_payload = payload.get("registry")
    if not isinstance(registry_payload, dict):
        return None

    try:
        return OpenAPISpecRegistry.from_cache_payload(registry_payload)
    except Exception as exc:
        logger.warning("Ignoring incompatible OpenAPI cache %s: %s", cache_path, exc)
        return None


def _write_registry_cache(spec_path: Path, cache_path: Path, registry: OpenAPISpecRegistry) -> None:
    payload = {
        "meta": _registry_cache_meta(spec_path),
        "registry": registry.to_cache_payload(),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temp_path.replace(cache_path)


def _load_registry_sync() -> OpenAPISpecRegistry:
    started_at = time.perf_counter()

    reg = _load_registry_from_cache(OPENAPI_SPEC_PATH, OPENAPI_CACHE_PATH)
    if reg is not None:
        logger.info(
            "Loaded OpenAPI registry cache from %s in %.3fs (%d operations).",
            OPENAPI_CACHE_PATH,
            time.perf_counter() - started_at,
            len(reg.operations),
        )
        return reg

    logger.info("Building OpenAPI registry from %s ...", OPENAPI_SPEC_PATH)
    reg = OpenAPISpecRegistry.from_file(OPENAPI_SPEC_PATH)

    try:
        _write_registry_cache(OPENAPI_SPEC_PATH, OPENAPI_CACHE_PATH, reg)
    except Exception as exc:
        logger.warning("Failed to write OpenAPI cache %s: %s", OPENAPI_CACHE_PATH, exc)

    logger.info(
        "Built OpenAPI registry in %.3fs (%d operations).",
        time.perf_counter() - started_at,
        len(reg.operations),
    )
    return reg


async def get_registry() -> OpenAPISpecRegistry:
    global _registry_cache, _registry_loading
    if _registry_cache is not None:
        return _registry_cache
    if _registry_loading is None:
        loop = asyncio.get_running_loop()
        _registry_loading = loop.run_in_executor(_thread_pool, _load_registry_sync)

    try:
        _registry_cache = await _registry_loading
        return _registry_cache
    except Exception:
        _registry_loading = None
        raise
    finally:
        if _registry_cache is not None:
            _registry_loading = None


@app.on_event("startup")
async def _startup_load_registry():
    try:
        await get_registry()
    except Exception:
        logger.exception("Failed to load OpenAPI spec at startup (will retry on first request)")


# -----------------------------------------------------------------------------
# LLM tool layer
# -----------------------------------------------------------------------------

LLM_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_openapi_operations",
        "description": "Search the locally loaded Tripletex OpenAPI spec for relevant operations before choosing an endpoint.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language description of the task or entity to find."},
                "top_k": {"type": "integer", "description": "Number of matches to return.", "minimum": 1, "maximum": 12},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_operation_schema",
        "description": "Get the exact path/query/body schema for one Tripletex operation_id from the loaded OpenAPI spec.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
            },
            "required": ["operation_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "call_tripletex_operation",
        "description": "Execute a Tripletex API operation by operation_id using the loaded OpenAPI spec for routing and preflight validation.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation_id": {"type": "string"},
                "path_params": {"type": "object", "additionalProperties": True},
                "query_params": {"type": "object", "additionalProperties": True},
                "body": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": True},
                        {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "object", "additionalProperties": True},
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "null"},
                                ]
                            },
                        },
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "reason": {"type": "string"},
            },
            "required": ["operation_id"],
            "additionalProperties": False,
        },
    },
]


AGENT_INSTRUCTIONS = """
You are a Tripletex task agent.

Objectives:
- Understand prompts in Norwegian Bokmål, Nynorsk, English, Spanish, Portuguese, German, or French.
- Use the local OpenAPI-backed tools to find the correct Tripletex operations.
- Handle multi-step tasks, including prerequisite creation in an empty sandbox.
- Minimize API calls and avoid 4xx errors.
- Do not verify writes unless it is necessary to complete the task or resolve ambiguity.
- If a call fails, read the error carefully, correct the request, and retry at most once.
- Competition mode is active: never ask the user follow-up questions or for confirmation.

Rules:
- Do not invent endpoints or fields.
- Search the OpenAPI spec before the first write unless the exact operation_id is already certain.
- Prefer GET/search operations to identify the correct entity before update/delete.
- Use ?fields=* or similar only when needed for disambiguation.
- Avoid get_operation_schema if the endpoint and payload are already clear enough to act safely.
- If get_operation_schema returns read_only_body_fields, do not send those fields in write requests.
- For GET operations, send filters in query_params, not body.
- For POST/PUT/PATCH operations, send JSON objects or arrays, never JSON encoded as strings.
- If the first customer or invoice search does not identify the target, broaden the search before failing.
- For customer lookup, try strong identifiers first such as organizationNumber, then customerName/email, and increase count when needed.
- For invoice lookup, prefer broader searches with larger count and useful fields, then filter locally by customer, amount, date, description, or status.
- If the sandbox may be empty, create missing prerequisites instead of asking the user to confirm details.
- When file text includes invoice, contract, or expense details, extract and use those facts.
- Stop when the task is complete or when essential information is genuinely missing, but report that as JSON rather than a question.

Tripletex write patterns to prefer:
- Invoice_post uses customer, invoiceDate, invoiceDueDate, and orders. Do not use top-level lines, invoiceLines, orderLines, customerId, or dueDate.
- Order_post uses nested customer: {id: ...}, orderDate, deliveryDate, and optionally orderLines. Do not use customerId.
- OrderOrderline_post uses nested order: {id: ...}, count, and unitPriceExcludingVatCurrency or unitPriceIncludingVatCurrency. Do not use orderId, quantity, or plain unitPrice.
- To invoice an existing order, prefer PUT /order/{id}/:invoice or Invoice_post with orders: [{id: ...}].

When you finish, return ONLY JSON with this shape:
{
  "status": "completed" | "failed" | "needs_input",
  "summary": "short human-readable summary",
  "created_or_changed": [
    {
      "entity": "customer|employee|invoice|payment|travelExpense|other",
      "id": 123,
      "status": "created|updated|deleted|verified|unknown"
    }
  ],
  "notes": ["optional note 1", "optional note 2"]
}
"""


def execute_llm_tool(
    name: str,
    arguments: Dict[str, Any],
    registry: OpenAPISpecRegistry,
    base_url: str,
    auth: Tuple[str, str],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    if name == "search_openapi_operations":
        return registry.search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 8),
        )

    if name == "get_operation_schema":
        return registry.get_operation_schema(arguments["operation_id"])

    if name == "call_tripletex_operation":
        return registry.execute(
            base_url=base_url,
            auth=auth,
            operation_id=arguments["operation_id"],
            path_params=arguments.get("path_params"),
            query_params=arguments.get("query_params"),
            body=arguments.get("body"),
            reason=arguments.get("reason", ""),
            allow_verify=AUTO_VERIFY_WRITES,
            request_id=request_id,
        )

    return {"ok": False, "error": f"Unknown tool name: {name}"}


async def _run_agent_once(
    prompt: str,
    file_context: Optional[str],
    registry: OpenAPISpecRegistry,
    base_url: str,
    auth: Tuple[str, str],
    request_id: str,
    attempt: int,
    previous_result: Optional[Dict[str, Any]] = None,
    previous_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    client = get_openai_client()
    llm_input = build_agent_input(
        prompt=prompt,
        file_context=file_context,
        attempt=attempt,
        previous_result=previous_result,
        previous_trace=previous_trace,
    )
    log_trace(
        "openai_request",
        request_id=request_id,
        attempt=attempt,
        model=OPENAI_MODEL,
        prompt=prompt,
        file_context=file_context or "[no files]",
        input=llm_input,
    )

    response = await client.responses.create(
        model=OPENAI_MODEL,
        instructions=AGENT_INSTRUCTIONS,
        input=llm_input,
        tools=LLM_TOOLS,
        parallel_tool_calls=False,
    )
    log_trace(
        "openai_response",
        request_id=request_id,
        attempt=attempt,
        response_id=response.id,
        turn=0,
        function_calls=summarize_function_calls(response),
        output_text=response.output_text or "",
    )

    trace: List[Dict[str, Any]] = []
    repair_attempted = False

    for turn in range(MAX_AGENT_TURNS):
        function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]

        if not function_calls:
            final_text = response.output_text or ""
            final_json = normalize_final_agent_output(final_text, trace)

            if not repair_attempted and result_requests_clarification(final_json, final_text):
                repair_attempted = True
                repair_message = """
Competition mode correction:
- Never ask the user follow-up questions or ask for confirmation.
- Continue autonomously using tools if there is any plausible next step.
- If customer or invoice lookup failed, broaden the search before failing:
  - customer: use organizationNumber, customerName, email, count up to 100, and useful fields.
  - invoice: use broader filters, larger count, and useful fields, then filter locally.
- Return ONLY valid JSON in the required schema.
"""
                log_trace(
                    "openai_repair_request",
                    request_id=request_id,
                    attempt=attempt,
                    turn=turn,
                    reason="clarification_or_question",
                    repair_message=repair_message,
                    current_result=final_json,
                )
                response = await client.responses.create(
                    model=OPENAI_MODEL,
                    instructions=AGENT_INSTRUCTIONS,
                    previous_response_id=response.id,
                    input=repair_message,
                    tools=LLM_TOOLS,
                    parallel_tool_calls=False,
                )
                log_trace(
                    "openai_response",
                    request_id=request_id,
                    attempt=attempt,
                    response_id=response.id,
                    turn=turn + 1,
                    function_calls=summarize_function_calls(response),
                    output_text=response.output_text or "",
                )
                continue

            log_trace(
                "openai_final",
                request_id=request_id,
                attempt=attempt,
                response_id=response.id,
                turn=turn,
                output_text=final_text,
                final_json=final_json,
            )

            return {
                "final": final_json,
                "raw_output_text": final_text,
                "trace": trace,
                "response_id": response.id,
            }

        tool_outputs = []

        for call in function_calls:
            try:
                args = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            result = execute_llm_tool(
                name=call.name,
                arguments=args,
                registry=registry,
                base_url=base_url,
                auth=auth,
                request_id=request_id,
            )
            log_trace(
                "agent_tool_call",
                request_id=request_id,
                attempt=attempt,
                turn=turn,
                tool=call.name,
                arguments=args,
                result=result,
            )

            trace.append(
                {
                    "tool": call.name,
                    "arguments": to_jsonable(args),
                    "result": to_jsonable(result),
                }
            )

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": safe_json_dumps(result),
                }
            )

        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=AGENT_INSTRUCTIONS,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=LLM_TOOLS,
            parallel_tool_calls=False,
        )
        log_trace(
            "openai_response",
            request_id=request_id,
            attempt=attempt,
            response_id=response.id,
            turn=turn + 1,
            function_calls=summarize_function_calls(response),
            output_text=response.output_text or "",
        )

    return {
        "final": {
            "status": "failed",
            "summary": f"Agent stopped after {MAX_AGENT_TURNS} turns",
            "created_or_changed": [],
            "notes": ["Increase MAX_AGENT_TURNS if the workflow is legitimately longer."],
        },
        "raw_output_text": "",
        "trace": trace,
        "response_id": response.id,
    }


async def run_agent(
    prompt: str,
    file_context: Optional[str],
    registry: OpenAPISpecRegistry,
    base_url: str,
    auth: Tuple[str, str],
    request_id: str,
) -> Dict[str, Any]:
    deadline = time.monotonic() + SOLVE_TIME_BUDGET_SECONDS
    last_result: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
        seconds_left = deadline - time.monotonic()
        if last_result is not None and seconds_left < MIN_SECONDS_LEFT_FOR_RETRY:
            log_trace(
                "agent_retry_stopped",
                request_id=request_id,
                attempt=attempt,
                reason="time_budget_exhausted",
                seconds_left=seconds_left,
            )
            break

        previous_result = last_result["final"] if last_result else None
        previous_trace = last_result["trace"] if last_result else None
        log_trace(
            "agent_attempt_start",
            request_id=request_id,
            attempt=attempt,
            seconds_left=seconds_left,
            previous_result=previous_result,
        )

        result = await _run_agent_once(
            prompt=prompt,
            file_context=file_context,
            registry=registry,
            base_url=base_url,
            auth=auth,
            request_id=request_id,
            attempt=attempt,
            previous_result=previous_result,
            previous_trace=previous_trace,
        )
        last_result = result

        should_retry = result_is_unsuccessful_or_uncertain(
            result["final"],
            result.get("raw_output_text", ""),
            result.get("trace", []),
        )
        log_trace(
            "agent_attempt_end",
            request_id=request_id,
            attempt=attempt,
            should_retry=should_retry,
            result=result["final"],
        )

        if not should_retry:
            return result

    return last_result or {
        "final": {
            "status": "failed",
            "summary": "Agent could not complete the task within the retry budget.",
            "created_or_changed": [],
            "notes": [],
        },
        "raw_output_text": "",
        "trace": [],
        "response_id": None,
    }


# -----------------------------------------------------------------------------
# File handling
# -----------------------------------------------------------------------------

def save_and_extract_files(files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_files: List[Dict[str, Any]] = []
    chunks: List[str] = []

    for item in files:
        filename = sanitize_filename(item.get("filename", "unnamed"))
        content_base64 = item.get("content_base64", "")

        try:
            raw_bytes = base64.b64decode(content_base64)
        except Exception as exc:
            saved_files.append(
                {
                    "filename": filename,
                    "saved": False,
                    "error": f"base64 decode failed: {exc}",
                }
            )
            continue

        path = UPLOAD_DIR / filename
        path.write_bytes(raw_bytes)

        extracted = extract_file_text(filename, raw_bytes)
        extracted = truncate_text(extracted, MAX_FILE_TEXT_CHARS)

        saved_files.append(
            {
                "filename": filename,
                "path": str(path),
                "saved": True,
                "extracted_preview": extracted,
            }
        )

        chunks.append(f"FILE: {filename}\n{extracted}")

    return saved_files, "\n\n".join(chunks)


def authorize_request(request: Request) -> None:
    if not APP_API_KEY:
        return

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {APP_API_KEY}"
    if auth_header != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_solve_response(
    *,
    request_id: Optional[str],
    saved_files: List[Dict[str, Any]],
    agent_result: Optional[Dict[str, Any]] = None,
    raw_output_text: Optional[str] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
    response_id: Optional[str] = None,
    error: Optional[str] = None,
) -> JSONResponse:
    if error:
        logger.error("Solve failed: %s", error)
        log_trace("solve_error", request_id=request_id, error=error)
    if agent_result is not None:
        logger.info("Agent result: %s", safe_json_dumps(agent_result))
        log_trace("solve_result", request_id=request_id, agent_result=agent_result)
    if trace:
        logger.debug("Agent trace: %s", safe_json_dumps(trace))
        log_trace("solve_trace_summary", request_id=request_id, trace=trace)
    if response_id:
        logger.info("OpenAI response id: %s", response_id)
        log_trace("solve_response_id", request_id=request_id, response_id=response_id)
    return JSONResponse({"status": "completed"})


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    spec_exists = OPENAPI_SPEC_PATH.exists()
    return {
        "ok": True,
        "openapi_spec_path": str(OPENAPI_SPEC_PATH),
        "openapi_spec_exists": spec_exists,
        "model": OPENAI_MODEL,
    }


@app.post("/solve")
async def solve(request: Request):
    authorize_request(request)
    request_id = uuid.uuid4().hex

    try:
        body = await request.json()
    except Exception as exc:
        return build_solve_response(request_id=request_id, saved_files=[], error=f"Invalid JSON body: {exc}")

    prompt = body.get("prompt")
    creds = body.get("tripletex_credentials")
    files = body.get("files", [])
    log_trace(
        "solve_request",
        request_id=request_id,
        prompt=prompt,
        files=[{"filename": f.get("filename"), "mime_type": f.get("mime_type")} for f in files if isinstance(f, dict)],
        tripletex_credentials=creds,
        host=request.headers.get("host"),
        path=str(request.url.path),
    )

    if not prompt:
        return build_solve_response(request_id=request_id, saved_files=[], error="Missing 'prompt'")
    if not isinstance(creds, dict):
        return build_solve_response(request_id=request_id, saved_files=[], error="Missing 'tripletex_credentials'")

    base_url = creds.get("base_url")
    session_token = creds.get("session_token")

    if not base_url or not session_token:
        return build_solve_response(
            request_id=request_id,
            saved_files=[],
            error="tripletex_credentials must include base_url and session_token",
        )

    try:
        registry = await get_registry()
    except FileNotFoundError as exc:
        logger.exception("Failed to load OpenAPI spec")
        return build_solve_response(request_id=request_id, saved_files=[], error=str(exc))
    except Exception as exc:
        logger.exception("Failed to load OpenAPI spec")
        return build_solve_response(request_id=request_id, saved_files=[], error=f"Failed to load OpenAPI spec: {exc}")

    saved_files, file_context = save_and_extract_files(files)
    log_trace(
        "solve_files_processed",
        request_id=request_id,
        saved_files=saved_files,
        file_context=file_context,
    )
    auth = ("0", session_token)

    try:
        agent_result = await run_agent(
            prompt=prompt,
            file_context=file_context,
            registry=registry,
            base_url=base_url,
            auth=auth,
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("Agent execution failed")
        return build_solve_response(
            request_id=request_id,
            saved_files=saved_files,
            agent_result={"status": "failed", "summary": str(exc), "created_or_changed": [], "notes": []},
            error=str(exc),
        )

    return build_solve_response(
        request_id=request_id,
        saved_files=saved_files,
        agent_result=agent_result["final"],
        raw_output_text=agent_result.get("raw_output_text"),
        trace=agent_result.get("trace", []),
        response_id=agent_result.get("response_id"),
    )
