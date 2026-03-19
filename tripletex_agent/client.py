from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, timedelta
import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import ApiCallRecord, ClientStats, TripletexApiError


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _normalize_path(path: str) -> str:
    return "/" + path.lstrip("/")


def _freeze_params(params: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not params:
        return ()
    items: list[tuple[str, str]] = []
    for key in sorted(params):
        items.append((key, str(params[key])))
    return tuple(items)


@dataclass(slots=True)
class CachedResponse:
    payload: Any
    status_code: int


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, *, timeout: float = 30.0) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout
        self.stats = ClientStats()
        basic_token = base64.b64encode(f"0:{session_token}".encode("utf-8")).decode("ascii")
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Basic {basic_token}",
        }
        self._get_cache: dict[tuple[str, tuple[tuple[str, str], ...]], CachedResponse] = {}

    def close(self) -> None:
        return None

    def __enter__(self) -> "TripletexClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,
        cacheable: bool | None = None,
    ) -> Any:
        method = method.upper()
        normalized_path = _normalize_path(path)
        query_string = f"?{urlencode(params)}" if params else ""
        url = f"{self.base_url}{normalized_path}{query_string}"
        if cacheable is None:
            cacheable = method == "GET"
        cache_key = (normalized_path, _freeze_params(params))
        if cacheable and method == "GET" and cache_key in self._get_cache:
            return self._get_cache[cache_key].payload

        body = None
        if json is not None:
            body = json_module_dumps(json).encode("utf-8")
        request = Request(url, data=body, method=method, headers=self._headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status_code = response.getcode()
                payload = self._parse_bytes(response.read())
        except HTTPError as exc:
            status_code = exc.code
            payload = self._parse_bytes(exc.read())
            self.stats.call_count += 1
            self.stats.error_count += 1
            self.stats.records.append(ApiCallRecord(method=method, path=normalized_path, status_code=status_code))
            raise TripletexApiError(
                method=method,
                url=url,
                status_code=status_code,
                message=self._extract_message(payload),
                payload=payload,
                response_text=json_module_dumps(payload) if not isinstance(payload, str) else payload,
            ) from exc
        except URLError as exc:
            self.stats.error_count += 1
            raise TripletexApiError(method=method, url=url, message=str(exc)) from exc

        self.stats.call_count += 1
        self.stats.records.append(ApiCallRecord(method=method, path=normalized_path, status_code=status_code))
        if cacheable and method == "GET":
            self._get_cache[cache_key] = CachedResponse(payload=payload, status_code=status_code)
        if method in {"POST", "PUT", "DELETE"}:
            self._invalidate_related_cache(normalized_path)
        return payload

    def list(self, path: str, *, params: Mapping[str, Any] | None = None, fields: str | None = None, count: int | None = None) -> list[dict[str, Any]]:
        query = dict(params or {})
        if fields:
            query["fields"] = fields
        if count is not None:
            query["count"] = count
        payload = self.request("GET", path, params=query)
        if isinstance(payload, Mapping) and isinstance(payload.get("values"), list):
            return [dict(item) for item in payload["values"] if isinstance(item, Mapping)]
        return []

    def get(self, path: str, *, params: Mapping[str, Any] | None = None, fields: str | None = None) -> dict[str, Any] | None:
        query = dict(params or {})
        if fields:
            query["fields"] = fields
        payload = self.request("GET", path, params=query)
        if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
            return dict(payload["value"])
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def create(self, path: str, body: Mapping[str, Any], *, params: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        payload = self.request("POST", path, params=params, json=body, cacheable=False)
        return self._unwrap_single(payload)

    def update(self, path: str, body: Mapping[str, Any], *, params: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        payload = self.request("PUT", path, params=params, json=body, cacheable=False)
        return self._unwrap_single(payload)

    def action(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        payload = self.request("PUT", path, params=params, cacheable=False)
        return self._unwrap_single(payload)

    def delete(self, path: str) -> None:
        self.request("DELETE", path, cacheable=False)

    def search_first(self, path: str, *, params: Mapping[str, Any], fields: str, count: int = 10) -> dict[str, Any] | None:
        values = self.list(path, params=params, fields=fields, count=count)
        return values[0] if values else None

    def find_customer(self, *, name: str | None = None, email: str | None = None, organization_number: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if email:
            params["email"] = email
        elif organization_number:
            params["organizationNumber"] = organization_number
        elif name:
            params["customerName"] = name
        else:
            return None
        return self.search_first("/customer", params=params, fields="id,name,email,organizationNumber,version")

    def find_employee(self, *, email: str | None = None, first_name: str | None = None, last_name: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if email:
            params["email"] = email
        if first_name:
            params["firstName"] = first_name
        if last_name:
            params["lastName"] = last_name
        if not params:
            return None
        return self.search_first("/employee", params=params, fields="id,firstName,lastName,email,userType,version")

    def find_product(self, *, name: str | None = None, number: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if number:
            params["productNumber"] = number
        elif name:
            params["name"] = name
        else:
            return None
        return self.search_first("/product", params=params, fields="id,name,number,priceExcludingVatCurrency,version")

    def find_project(self, *, name: str | None = None, number: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if number:
            params["number"] = number
        elif name:
            params["name"] = name
        else:
            return None
        return self.search_first("/project", params=params, fields="id,name,number,customer(id,name),version")

    def find_department(self, *, name: str | None = None, number: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if number:
            params["departmentNumber"] = number
        elif name:
            params["name"] = name
        else:
            return None
        return self.search_first("/department", params=params, fields="id,name,departmentNumber,version")

    def find_invoice(self, *, invoice_number: str | None = None, customer_id: int | None = None) -> dict[str, Any] | None:
        today = date.today()
        params: dict[str, Any] = {
            "invoiceDateFrom": str(today - timedelta(days=365 * 3)),
            "invoiceDateTo": str(today + timedelta(days=30)),
        }
        if invoice_number:
            params["invoiceNumber"] = invoice_number
        if customer_id:
            params["customerId"] = customer_id
        return self.search_first("/invoice", params=params, fields="id,invoiceNumber,invoiceDate,invoiceDueDate,amount,amountCurrency,customer(id,name)")

    def find_invoice_payment_type(self, description: str | None = None) -> dict[str, Any] | None:
        params = {"description": description} if description else {}
        result = self.search_first("/invoice/paymentType", params=params, fields="id,description")
        if result:
            return result
        return self.search_first("/invoice/paymentType", params={}, fields="id,description")

    def find_travel_expense(self, *, travel_expense_id: str | None = None) -> dict[str, Any] | None:
        if travel_expense_id:
            return self.get(f"/travelExpense/{travel_expense_id}", fields="id,departureDate,returnDate,state,version")
        values = self.list("/travelExpense", params={"state": "ALL"}, fields="id,departureDate,returnDate,state", count=10)
        return values[0] if values else None

    def _invalidate_related_cache(self, path: str) -> None:
        prefix = path.split("/{", 1)[0]
        self._get_cache = {key: value for key, value in self._get_cache.items() if not key[0].startswith(prefix)}

    @staticmethod
    def _parse_bytes(data: bytes) -> Any:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return data.decode("utf-8", errors="replace")

    @staticmethod
    def _unwrap_single(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, Mapping) and isinstance(payload.get("value"), Mapping):
            return dict(payload["value"])
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    @staticmethod
    def _extract_message(payload: Any) -> str | None:
        if isinstance(payload, Mapping):
            if isinstance(payload.get("message"), str):
                return payload["message"]
            if isinstance(payload.get("developerMessage"), str):
                return payload["developerMessage"]
            validations = payload.get("validationMessages")
            if isinstance(validations, list) and validations:
                first = validations[0]
                if isinstance(first, Mapping) and isinstance(first.get("message"), str):
                    return first["message"]
        if isinstance(payload, str):
            return payload
        return None


def json_module_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
