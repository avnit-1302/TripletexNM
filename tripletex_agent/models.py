from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class CompetitionCredentials:
    base_url: str
    session_token: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CompetitionCredentials":
        return cls(
            base_url=str(data["base_url"]),
            session_token=str(data["session_token"]),
        )


@dataclass(slots=True)
class CompetitionFile:
    filename: str
    content_base64: str
    mime_type: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CompetitionFile":
        return cls(
            filename=str(data["filename"]),
            content_base64=str(data["content_base64"]),
            mime_type=str(data["mime_type"]) if data.get("mime_type") else None,
        )


@dataclass(slots=True)
class CompetitionRequest:
    prompt: str
    files: list[CompetitionFile] = field(default_factory=list)
    tripletex_credentials: CompetitionCredentials | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CompetitionRequest":
        return cls(
            prompt=str(data["prompt"]),
            files=[CompetitionFile.from_mapping(item) for item in data.get("files", [])],
            tripletex_credentials=CompetitionCredentials.from_mapping(data["tripletex_credentials"]),
        )


@dataclass(slots=True)
class CompetitionResponse:
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status}


@dataclass(slots=True)
class AttachmentSummary:
    filename: str
    mime_type: str | None
    sha1: str
    size_bytes: int
    text_excerpt: str = ""
    materialized_path: Path | None = None


@dataclass(slots=True)
class ApiCallRecord:
    method: str
    path: str
    status_code: int


@dataclass(slots=True)
class ClientStats:
    call_count: int = 0
    error_count: int = 0
    records: list[ApiCallRecord] = field(default_factory=list)


@dataclass(slots=True)
class TripletexApiError(RuntimeError):
    method: str
    url: str
    status_code: int | None = None
    message: str | None = None
    payload: Any = None
    response_text: str | None = None

    def __str__(self) -> str:
        parts = [f"{self.method} {self.url}"]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.message:
            parts.append(self.message)
        return " | ".join(parts)
