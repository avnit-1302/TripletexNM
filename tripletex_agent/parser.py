from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any

from .llm import OpenAIExtractor
from .models import AttachmentSummary


class TaskType(str, Enum):
    CUSTOMER_CREATE = "customer_create"
    CUSTOMER_UPDATE = "customer_update"
    EMPLOYEE_CREATE = "employee_create"
    EMPLOYEE_UPDATE = "employee_update"
    PRODUCT_CREATE = "product_create"
    PRODUCT_UPDATE = "product_update"
    PROJECT_CREATE = "project_create"
    PROJECT_UPDATE = "project_update"
    DEPARTMENT_CREATE = "department_create"
    DEPARTMENT_UPDATE = "department_update"
    INVOICE_CREATE = "invoice_create"
    PAYMENT_REGISTER = "payment_register"
    TRAVEL_EXPENSE_DELETE = "travel_expense_delete"
    VOUCHER_REVERSE = "voucher_reverse"
    CREDIT_NOTE_CREATE = "credit_note_create"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ParsedTask:
    raw_prompt: str
    normalized_prompt: str
    language_hint: str
    task_type: TaskType
    operation: str
    confidence: float
    fields: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def get_str(self, key: str) -> str | None:
        value = self.fields.get(key)
        if value is None:
            return None
        return str(value).strip() or None


ACTION_KEYWORDS = {
    "create": ["create", "opprett", "lag", "criar", "crear", "creer", "erstellen"],
    "update": ["update", "oppdater", "endre", "actualizar", "modifier"],
    "delete": ["delete", "slett", "remove", "supprimer", "eliminar", "löschen"],
    "register": ["register", "registrer", "registrar", "enregistrer"],
    "reverse": ["reverse", "reverser", "reversere", "storno", "ompost"],
}

RESOURCE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CUSTOMER_CREATE: ["customer", "kunde", "cliente", "client"],
    TaskType.CUSTOMER_UPDATE: ["customer", "kunde", "cliente", "client"],
    TaskType.EMPLOYEE_CREATE: ["employee", "ansatt", "tilsett", "mitarbeiter", "employe", "funcionario"],
    TaskType.EMPLOYEE_UPDATE: ["employee", "ansatt", "tilsett", "mitarbeiter", "employe", "funcionario"],
    TaskType.PRODUCT_CREATE: ["product", "produkt", "produit", "producto"],
    TaskType.PRODUCT_UPDATE: ["product", "produkt", "produit", "producto"],
    TaskType.PROJECT_CREATE: ["project", "prosjekt", "proyecto", "projet"],
    TaskType.PROJECT_UPDATE: ["project", "prosjekt", "proyecto", "projet"],
    TaskType.DEPARTMENT_CREATE: ["department", "avdeling", "departement"],
    TaskType.DEPARTMENT_UPDATE: ["department", "avdeling", "departement"],
    TaskType.INVOICE_CREATE: ["invoice", "faktura", "facture", "rechnung", "fatura", "factura"],
    TaskType.PAYMENT_REGISTER: ["payment", "betaling", "paiement", "pagamento", "zahlung"],
    TaskType.TRAVEL_EXPENSE_DELETE: ["travel expense", "reiseutgift", "reiseregning", "travel report"],
    TaskType.VOUCHER_REVERSE: ["voucher", "bilag", "ledger voucher"],
    TaskType.CREDIT_NOTE_CREATE: ["credit note", "kreditnota", "credit memo", "kreditnote"],
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s-]{6,}\d)")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})\b")
AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,9}(?:[.,]\d{1,2})?)(?!\d)")
ID_RE = re.compile(r"\b(?:id|nr|number|nummer|invoice number|fakturanummer|voucher id|bilag id)\s*[:#]?\s*([0-9]{1,12})\b", re.IGNORECASE)
ORG_RE = re.compile(r"\b(?:organization number|organisasjonsnummer|orgnr|org nr)\s*[:#]?\s*([0-9]{9,12})\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def detect_language(prompt: str) -> str:
    lowered = strip_diacritics(prompt.lower())
    scores = {
        "nb": sum(token in lowered for token in ["opprett", "kunde", "ansatt", "betaling", "faktura"]),
        "en": sum(token in lowered for token in ["create", "customer", "employee", "payment", "invoice"]),
        "es": sum(token in lowered for token in ["crear", "cliente", "empleado", "factura"]),
        "pt": sum(token in lowered for token in ["criar", "cliente", "funcionario", "fatura"]),
        "de": sum(token in lowered for token in ["erstellen", "kunde", "mitarbeiter", "rechnung"]),
        "fr": sum(token in lowered for token in ["creer", "client", "employe", "facture"]),
        "nn": sum(token in lowered for token in ["opprett", "kunde", "tilsett"]),
    }
    language = max(scores, key=scores.get)
    return language if scores[language] > 0 else "en"


def _match_action(prompt: str) -> str:
    lowered = strip_diacritics(prompt.lower())
    for action, keywords in ACTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return action
    return "unknown"


def _match_task_type(prompt: str, action: str) -> tuple[TaskType, float]:
    lowered = strip_diacritics(prompt.lower())
    if any(word in lowered for word in RESOURCE_KEYWORDS[TaskType.CREDIT_NOTE_CREATE]):
        return TaskType.CREDIT_NOTE_CREATE, 0.8
    if any(word in lowered for word in RESOURCE_KEYWORDS[TaskType.PAYMENT_REGISTER]):
        return TaskType.PAYMENT_REGISTER, 0.8 if action in {"register", "create"} else 0.6
    if any(word in lowered for word in RESOURCE_KEYWORDS[TaskType.INVOICE_CREATE]):
        return TaskType.INVOICE_CREATE, 0.8 if action in {"create", "register"} else 0.6
    if any(word in lowered for word in RESOURCE_KEYWORDS[TaskType.TRAVEL_EXPENSE_DELETE]) and action == "delete":
        return TaskType.TRAVEL_EXPENSE_DELETE, 0.8
    if any(word in lowered for word in RESOURCE_KEYWORDS[TaskType.VOUCHER_REVERSE]) and action == "reverse":
        return TaskType.VOUCHER_REVERSE, 0.8
    best_type = TaskType.UNKNOWN
    best_score = 0.0
    for task_type, keywords in RESOURCE_KEYWORDS.items():
        score = sum(keyword in lowered for keyword in keywords)
        if task_type.name.endswith(action.upper()):
            score += 1
        if action == "register" and task_type == TaskType.PAYMENT_REGISTER:
            score += 1
        if action == "reverse" and task_type in {TaskType.VOUCHER_REVERSE, TaskType.CREDIT_NOTE_CREATE}:
            score += 1
        if score > best_score:
            best_type = task_type
            best_score = float(score)
    if "credit" in lowered or "kredit" in lowered:
        return TaskType.CREDIT_NOTE_CREATE, max(best_score, 2.0) / 4.0
    if "travel" in lowered and action == "delete":
        return TaskType.TRAVEL_EXPENSE_DELETE, max(best_score, 2.0) / 4.0
    return best_type, min(best_score / 4.0, 1.0)


def _extract_named_value(prompt: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = re.compile(rf"{label}\s*[:=]?\s*[\"“]?([^,;\n]+)", re.IGNORECASE)
        match = pattern.search(prompt)
        if match:
            return _trim_value(match.group(1).strip(" \"”"))
    return None


def _extract_capitalized_name(prompt: str, resource_keywords: list[str]) -> str | None:
    for keyword in resource_keywords:
        pattern = re.compile(rf"{keyword}(?:\s+named|\s+med\s+navn|\s+with\s+name|\s+)([A-ZÆØÅ][^,;\n]+)", re.IGNORECASE)
        match = pattern.search(prompt)
        if match:
            candidate = normalize_text(match.group(1))
            if len(candidate.split()) <= 6:
                return _trim_value(candidate)
    return None


def _trim_value(value: str) -> str:
    normalized = normalize_text(value)
    for separator in (
        " with ",
        " med ",
        " for ",
        " og ",
        " and ",
        " som ",
        " where ",
    ):
        if separator in normalized.lower():
            index = normalized.lower().index(separator)
            normalized = normalized[:index]
            break
    return normalized.strip(" ,;.")


def _normalize_date(value: str) -> str:
    if "." in value:
        day, month, year = value.split(".")
        return f"{year}-{month}-{day}"
    return value


def _build_rule_based_task(prompt: str, attachments: list[AttachmentSummary]) -> ParsedTask:
    normalized_prompt = normalize_text(prompt)
    action = _match_action(normalized_prompt)
    task_type, confidence = _match_task_type(normalized_prompt, action)
    language_hint = detect_language(normalized_prompt)
    fields: dict[str, Any] = {}
    notes: list[str] = []

    email = EMAIL_RE.search(normalized_prompt)
    if email:
        fields["email"] = email.group(0)

    phone = PHONE_RE.search(normalized_prompt)
    if phone:
        fields["phone"] = normalize_text(phone.group(0))

    organization_number = ORG_RE.search(normalized_prompt)
    if organization_number:
        fields["organization_number"] = organization_number.group(1)

    dates = [_normalize_date(item) for item in DATE_RE.findall(normalized_prompt)]
    if dates:
        fields["date"] = dates[0]
        if len(dates) > 1:
            fields["due_date"] = dates[1]

    explicit_name = _extract_named_value(
        normalized_prompt,
        ["name", "navn", "nombre", "nom", "nome", "kunde", "customer", "client", "cliente", "ansatt", "employee"],
    )
    if explicit_name:
        fields["name"] = explicit_name

    if task_type in {TaskType.EMPLOYEE_CREATE, TaskType.EMPLOYEE_UPDATE} and not fields.get("name"):
        name = _extract_capitalized_name(normalized_prompt, RESOURCE_KEYWORDS[task_type])
        if name:
            fields["name"] = name
            parts = name.split()
            fields["first_name"] = parts[0]
            if len(parts) > 1:
                fields["last_name"] = " ".join(parts[1:])

    if task_type in {TaskType.CUSTOMER_CREATE, TaskType.CUSTOMER_UPDATE} and not fields.get("customer_name"):
        fields["customer_name"] = fields.get("name") or _extract_capitalized_name(normalized_prompt, RESOURCE_KEYWORDS[task_type])
        fields["customer_email"] = _extract_named_value(normalized_prompt, ["customer email", "kunde e-post", "invoice email"]) or fields.get("email")

    if task_type in {TaskType.PRODUCT_CREATE, TaskType.PRODUCT_UPDATE}:
        fields["product_name"] = fields.get("name") or _extract_capitalized_name(normalized_prompt, RESOURCE_KEYWORDS[task_type])

    if task_type in {TaskType.PROJECT_CREATE, TaskType.PROJECT_UPDATE}:
        fields["project_name"] = fields.get("name") or _extract_capitalized_name(normalized_prompt, RESOURCE_KEYWORDS[task_type])
        fields["customer_name"] = fields.get("customer_name") or _extract_named_value(normalized_prompt, ["customer", "kunde", "client", "cliente"])

    if task_type in {TaskType.DEPARTMENT_CREATE, TaskType.DEPARTMENT_UPDATE}:
        fields["department_name"] = fields.get("name") or _extract_capitalized_name(normalized_prompt, RESOURCE_KEYWORDS[task_type])

    if task_type == TaskType.INVOICE_CREATE:
        fields["customer_name"] = _extract_named_value(normalized_prompt, ["customer", "kunde", "client", "cliente"]) or fields.get("customer_name")
        fields["customer_email"] = _extract_named_value(normalized_prompt, ["customer email", "kunde e-post", "invoice email"]) or fields.get("customer_email") or fields.get("email")
        fields["product_name"] = _extract_named_value(normalized_prompt, ["product", "produkt", "item", "vare"]) or fields.get("product_name")
        fields["description"] = _extract_named_value(normalized_prompt, ["description", "beskrivelse", "comment", "kommentar"])
        fields["quantity"] = _extract_named_value(normalized_prompt, ["quantity", "antall", "qty"])
        fields["price"] = _extract_named_value(normalized_prompt, ["price", "pris", "amount", "beløp", "belop", "sum"])
        fields["project_name"] = fields.get("project_name") or _extract_named_value(normalized_prompt, ["project", "prosjekt"])

    if task_type == TaskType.PAYMENT_REGISTER:
        fields["invoice_number"] = _extract_named_value(normalized_prompt, ["invoice number", "fakturanummer", "invoice"]) or None
        fields["payment_type"] = _extract_named_value(normalized_prompt, ["payment type", "betalingstype"])
        fields["customer_name"] = fields.get("customer_name") or _extract_named_value(normalized_prompt, ["customer", "kunde", "client", "cliente"])
        fields["customer_email"] = fields.get("customer_email") or _extract_named_value(normalized_prompt, ["customer email", "kunde e-post", "invoice email"])
        if fields.get("invoice_number"):
            match = re.search(r"\d+", str(fields["invoice_number"]))
            if match:
                fields["invoice_number"] = match.group(0)

    if task_type == TaskType.TRAVEL_EXPENSE_DELETE:
        travel_expense_id = _extract_named_value(normalized_prompt, ["travel expense id", "reiseutgift id", "reiseregning id"])
        if travel_expense_id:
            fields["travel_expense_id"] = travel_expense_id

    if task_type == TaskType.VOUCHER_REVERSE:
        voucher_id = _extract_named_value(normalized_prompt, ["voucher id", "bilag id", "voucher", "bilag"])
        if voucher_id:
            fields["voucher_id"] = voucher_id

    if task_type == TaskType.CREDIT_NOTE_CREATE:
        invoice_number = _extract_named_value(normalized_prompt, ["invoice number", "fakturanummer", "invoice"])
        if invoice_number:
            fields["invoice_number"] = invoice_number

    lowered = normalized_prompt.lower()
    if "administrator" in lowered or "kontoadministrator" in lowered or "all privileges" in lowered:
        fields["role_template"] = "ALL_PRIVILEGES"
    elif "accountant" in lowered or "regnskapsforer" in lowered or "regnskapsfører" in lowered:
        fields["role_template"] = "ACCOUNTANT"
    elif "department leader" in lowered or "avdelingsleder" in lowered:
        fields["role_template"] = "DEPARTMENT_LEADER"

    amount_match = None
    for match in AMOUNT_RE.finditer(normalized_prompt):
        amount_match = match
    if amount_match:
        fields.setdefault("amount", amount_match.group(1).replace(",", "."))

    number_match = ID_RE.search(normalized_prompt)
    if number_match:
        fields.setdefault("number", number_match.group(1))

    if attachments:
        fields["attachments_present"] = True
        notes.extend(f"attachment:{item.filename}" for item in attachments)
        if attachments[0].text_excerpt and not fields.get("description"):
            fields["description"] = attachments[0].text_excerpt[:250]

    return ParsedTask(
        raw_prompt=prompt,
        normalized_prompt=normalized_prompt,
        language_hint=language_hint,
        task_type=task_type,
        operation=action,
        confidence=confidence,
        fields=fields,
        notes=notes,
    )


def parse_task(prompt: str, attachments: list[AttachmentSummary] | None = None) -> ParsedTask:
    attachment_summaries = attachments or []
    baseline = _build_rule_based_task(prompt, attachment_summaries)
    extractor = OpenAIExtractor()
    if not extractor.enabled:
        return baseline
    try:
        payload = extractor.extract(prompt, attachment_summaries)
    except Exception as exc:  # noqa: BLE001
        baseline.notes.append(f"llm_fallback:{type(exc).__name__}")
        return baseline
    if not payload:
        return baseline
    task_type_value = str(payload.get("task_type", baseline.task_type.value))
    try:
        task_type = TaskType(task_type_value)
    except ValueError:
        task_type = baseline.task_type
    merged_fields = dict(baseline.fields)
    merged_fields.update(payload.get("fields", {}))
    merged_notes = list(baseline.notes)
    merged_notes.extend(payload.get("notes", []))
    return ParsedTask(
        raw_prompt=prompt,
        normalized_prompt=baseline.normalized_prompt,
        language_hint=str(payload.get("language_hint", baseline.language_hint)),
        task_type=task_type,
        operation=str(payload.get("operation", baseline.operation)),
        confidence=float(payload.get("confidence", baseline.confidence)),
        fields=merged_fields,
        notes=merged_notes,
    )
