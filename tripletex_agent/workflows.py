from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .client import TripletexClient
from .models import AttachmentSummary
from .parser import ParsedTask, TaskType
from .verification import VerificationResult, verify_task


@dataclass(slots=True)
class ExecutionResult:
    task_type: TaskType
    verification: VerificationResult
    resource: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _split_name(task: ParsedTask) -> tuple[str | None, str | None]:
    first_name = task.get_str("first_name")
    last_name = task.get_str("last_name")
    if first_name or last_name:
        return first_name, last_name
    name = task.get_str("name")
    if not name:
        return None, None
    parts = name.split()
    return parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""


def _select_date(task: ParsedTask, key: str) -> str:
    explicit = task.get_str(key)
    return explicit or str(date.today())


def _customer_payload(task: ParsedTask) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": task.get_str("customer_name") or task.get_str("name")}
    if task.get_str("customer_email") or task.get_str("email"):
        payload["email"] = task.get_str("customer_email") or task.get_str("email")
    if task.get_str("organization_number"):
        payload["organizationNumber"] = task.get_str("organization_number")
    if task.get_str("phone"):
        payload["phoneNumberMobile"] = task.get_str("phone")
    if task.get_str("description"):
        payload["description"] = task.get_str("description")
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _create_or_update_customer(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    payload = _customer_payload(task)
    existing = client.find_customer(
        name=payload.get("name"),
        email=payload.get("email"),
        organization_number=payload.get("organizationNumber"),
    )
    if task.task_type == TaskType.CUSTOMER_UPDATE and existing:
        payload["id"] = existing["id"]
        payload["version"] = existing.get("version")
        return client.update(f"/customer/{existing['id']}", payload) or existing
    if existing:
        return existing
    return client.create("/customer", payload) or payload


def _employee_payload(task: ParsedTask) -> dict[str, Any]:
    first_name, last_name = _split_name(task)
    payload: dict[str, Any] = {"firstName": first_name, "lastName": last_name}
    if task.get_str("email"):
        payload["email"] = task.get_str("email")
    if task.get_str("phone"):
        payload["phoneNumberMobile"] = task.get_str("phone")
    if task.get_str("number"):
        payload["employeeNumber"] = task.get_str("number")
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _create_or_update_employee(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    first_name, last_name = _split_name(task)
    payload = _employee_payload(task)
    existing = client.find_employee(email=task.get_str("email"), first_name=first_name, last_name=last_name)
    if task.task_type == TaskType.EMPLOYEE_UPDATE and existing:
        payload["id"] = existing["id"]
        payload["version"] = existing.get("version")
        employee = client.update(f"/employee/{existing['id']}", payload) or existing
    elif existing:
        employee = existing
    else:
        employee = client.create("/employee", payload) or payload
    if task.get_str("role_template"):
        client.action(
            "/employee/entitlement/:grantEntitlementsByTemplate",
            params={"employeeId": employee["id"], "template": task.get_str("role_template")},
        )
    return employee


def _product_payload(task: ParsedTask) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": task.get_str("product_name") or task.get_str("name")}
    if task.get_str("number"):
        payload["number"] = task.get_str("number")
    if task.get_str("description"):
        payload["description"] = task.get_str("description")
    price = _to_float(task.get_str("price") or task.get_str("amount"))
    if price is not None:
        payload["priceExcludingVatCurrency"] = price
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _create_or_update_product(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    payload = _product_payload(task)
    existing = client.find_product(name=payload.get("name"), number=payload.get("number"))
    if task.task_type == TaskType.PRODUCT_UPDATE and existing:
        payload["id"] = existing["id"]
        payload["version"] = existing.get("version")
        return client.update(f"/product/{existing['id']}", payload) or existing
    if existing:
        return existing
    return client.create("/product", payload) or payload


def _department_payload(task: ParsedTask) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": task.get_str("department_name") or task.get_str("name")}
    if task.get_str("number"):
        payload["departmentNumber"] = task.get_str("number")
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _create_or_update_department(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    payload = _department_payload(task)
    existing = client.find_department(name=payload.get("name"), number=payload.get("departmentNumber"))
    if task.task_type == TaskType.DEPARTMENT_UPDATE and existing:
        payload["id"] = existing["id"]
        payload["version"] = existing.get("version")
        return client.update(f"/department/{existing['id']}", payload) or existing
    if existing:
        return existing
    return client.create("/department", payload) or payload


def _project_payload(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": task.get_str("project_name") or task.get_str("name")}
    if task.get_str("number"):
        payload["number"] = task.get_str("number")
    if task.get_str("description"):
        payload["description"] = task.get_str("description")
    if task.get_str("start_date"):
        payload["startDate"] = task.get_str("start_date")
    if task.get_str("end_date"):
        payload["endDate"] = task.get_str("end_date")
    customer_name = task.get_str("customer_name")
    customer_email = task.get_str("customer_email")
    if customer_name or customer_email:
        customer = client.find_customer(name=customer_name, email=customer_email)
        if customer:
            payload["customer"] = {"id": customer["id"]}
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _create_or_update_project(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    payload = _project_payload(client, task)
    existing = client.find_project(name=payload.get("name"), number=payload.get("number"))
    if task.task_type == TaskType.PROJECT_UPDATE and existing:
        payload["id"] = existing["id"]
        payload["version"] = existing.get("version")
        return client.update(f"/project/{existing['id']}", payload) or existing
    if existing:
        return existing
    return client.create("/project", payload) or payload


def _build_order_line(task: ParsedTask, client: TripletexClient) -> dict[str, Any]:
    line: dict[str, Any] = {}
    product_name = task.get_str("product_name")
    if product_name:
        product = client.find_product(name=product_name)
        if product:
            line["product"] = {"id": product["id"]}
    line["description"] = task.get_str("description") or product_name or "Generated order line"
    line["count"] = _to_float(task.get_str("quantity"), 1.0) or 1.0
    unit_price = _to_float(task.get_str("price") or task.get_str("amount"), 0.0) or 0.0
    if unit_price:
        line["unitPriceExcludingVatCurrency"] = unit_price
    return line


def _create_invoice(client: TripletexClient, task: ParsedTask, attachments: list[AttachmentSummary]) -> dict[str, Any]:
    customer = _create_or_update_customer(client, task)
    invoice_date = _select_date(task, "date")
    due_date = task.get_str("due_date") or invoice_date
    order_payload: dict[str, Any] = {
        "customer": {"id": customer["id"]},
        "orderDate": invoice_date,
        "orderLines": [_build_order_line(task, client)],
    }
    if task.get_str("project_name"):
        project = client.find_project(name=task.get_str("project_name"))
        if project:
            order_payload["project"] = {"id": project["id"]}
    if attachments and attachments[0].text_excerpt and not task.get_str("description"):
        order_payload["invoiceComment"] = attachments[0].text_excerpt[:500]
    payload = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "customer": {"id": customer["id"]},
        "orders": [order_payload],
    }
    if task.get_str("comment"):
        payload["comment"] = task.get_str("comment")
    return client.create("/invoice", payload, params={"sendToCustomer": False}) or payload


def _register_payment(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    invoice = None
    if task.get_str("invoice_number"):
        invoice = client.find_invoice(invoice_number=task.get_str("invoice_number"))
    if invoice is None:
        customer = client.find_customer(name=task.get_str("customer_name"), email=task.get_str("customer_email") or task.get_str("email"))
        if customer:
            invoice = client.find_invoice(customer_id=customer["id"])
    if invoice is None:
        raise ValueError("Could not identify invoice for payment registration")
    payment_type = client.find_invoice_payment_type(task.get_str("payment_type"))
    if payment_type is None:
        raise ValueError("Could not identify invoice payment type")
    amount = _to_float(task.get_str("amount"), invoice.get("amountCurrency") or invoice.get("amount"))
    return client.action(
        f"/invoice/{invoice['id']}/:payment",
        params={
            "paymentDate": _select_date(task, "date"),
            "paymentTypeId": payment_type["id"],
            "paidAmount": amount,
        },
    ) or invoice


def _delete_travel_expense(client: TripletexClient, task: ParsedTask) -> dict[str, Any] | None:
    expense = client.find_travel_expense(travel_expense_id=task.get_str("travel_expense_id"))
    if expense is None:
        raise ValueError("Could not identify travel expense to delete")
    client.delete(f"/travelExpense/{expense['id']}")
    return {"id": expense["id"], "deleted": True}


def _reverse_voucher(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    voucher_id = task.get_str("voucher_id") or task.get_str("number")
    if not voucher_id:
        raise ValueError("Voucher id is required to reverse a voucher")
    return client.action(
        f"/ledger/voucher/{voucher_id}/:reverse",
        params={"date": _select_date(task, "date")},
    ) or {"id": voucher_id}


def _create_credit_note(client: TripletexClient, task: ParsedTask) -> dict[str, Any]:
    invoice = None
    if task.get_str("invoice_number"):
        invoice = client.find_invoice(invoice_number=task.get_str("invoice_number"))
    if invoice is None:
        customer = client.find_customer(name=task.get_str("customer_name"), email=task.get_str("customer_email"))
        if customer:
            invoice = client.find_invoice(customer_id=customer["id"])
    if invoice is None:
        raise ValueError("Could not identify invoice for credit note")
    return client.action(
        f"/invoice/{invoice['id']}/:createCreditNote",
        params={
            "date": _select_date(task, "date"),
            "comment": task.get_str("comment") or "",
            "sendToCustomer": False,
        },
    ) or invoice


def execute_task(client: TripletexClient, task: ParsedTask, attachments: list[AttachmentSummary]) -> ExecutionResult:
    notes = list(task.notes)
    resource: dict[str, Any] | None
    if task.task_type in {TaskType.CUSTOMER_CREATE, TaskType.CUSTOMER_UPDATE}:
        resource = _create_or_update_customer(client, task)
    elif task.task_type in {TaskType.EMPLOYEE_CREATE, TaskType.EMPLOYEE_UPDATE}:
        resource = _create_or_update_employee(client, task)
    elif task.task_type in {TaskType.PRODUCT_CREATE, TaskType.PRODUCT_UPDATE}:
        resource = _create_or_update_product(client, task)
    elif task.task_type in {TaskType.PROJECT_CREATE, TaskType.PROJECT_UPDATE}:
        resource = _create_or_update_project(client, task)
    elif task.task_type in {TaskType.DEPARTMENT_CREATE, TaskType.DEPARTMENT_UPDATE}:
        resource = _create_or_update_department(client, task)
    elif task.task_type == TaskType.INVOICE_CREATE:
        resource = _create_invoice(client, task, attachments)
    elif task.task_type == TaskType.PAYMENT_REGISTER:
        resource = _register_payment(client, task)
    elif task.task_type == TaskType.TRAVEL_EXPENSE_DELETE:
        resource = _delete_travel_expense(client, task)
    elif task.task_type == TaskType.VOUCHER_REVERSE:
        resource = _reverse_voucher(client, task)
    elif task.task_type == TaskType.CREDIT_NOTE_CREATE:
        resource = _create_credit_note(client, task)
    else:
        resource = None
        notes.append("unknown_task_noop")
    verification = verify_task(task, resource)
    return ExecutionResult(task_type=task.task_type, verification=verification, resource=resource, notes=notes)
