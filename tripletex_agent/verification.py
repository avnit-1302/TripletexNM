from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .parser import ParsedTask, TaskType


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class VerificationCheck:
    name: str
    status: VerificationStatus
    expected: Any = None
    actual: Any = None


@dataclass(slots=True)
class VerificationResult:
    status: VerificationStatus
    checks: list[VerificationCheck] = field(default_factory=list)


def verify_task(task: ParsedTask, resource: dict[str, Any] | None) -> VerificationResult:
    if resource is None:
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            checks=[VerificationCheck(name="resource_present", status=VerificationStatus.UNKNOWN)],
        )
    checks = [VerificationCheck(name="resource_present", status=VerificationStatus.PASS)]
    if task.task_type in {
        TaskType.CUSTOMER_CREATE,
        TaskType.CUSTOMER_UPDATE,
        TaskType.PRODUCT_CREATE,
        TaskType.PRODUCT_UPDATE,
        TaskType.PROJECT_CREATE,
        TaskType.PROJECT_UPDATE,
        TaskType.DEPARTMENT_CREATE,
        TaskType.DEPARTMENT_UPDATE,
    }:
        expected_name = (
            task.get_str("customer_name")
            or task.get_str("product_name")
            or task.get_str("project_name")
            or task.get_str("department_name")
            or task.get_str("name")
        )
        actual_name = resource.get("name") or resource.get("displayName")
        checks.append(
            VerificationCheck(
                name="name_matches",
                status=VerificationStatus.PASS if not expected_name or actual_name == expected_name else VerificationStatus.FAIL,
                expected=expected_name,
                actual=actual_name,
            )
        )
    if task.task_type in {TaskType.EMPLOYEE_CREATE, TaskType.EMPLOYEE_UPDATE}:
        expected_email = task.get_str("email")
        checks.append(
            VerificationCheck(
                name="email_matches",
                status=VerificationStatus.PASS if not expected_email or resource.get("email") == expected_email else VerificationStatus.FAIL,
                expected=expected_email,
                actual=resource.get("email"),
            )
        )
    status = VerificationStatus.PASS if all(check.status != VerificationStatus.FAIL for check in checks) else VerificationStatus.FAIL
    return VerificationResult(status=status, checks=checks)
