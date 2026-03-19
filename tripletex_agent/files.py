from __future__ import annotations

import base64
import hashlib
import mimetypes
import tempfile
from pathlib import Path
from typing import Iterable

from .models import AttachmentSummary, CompetitionFile


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"}


def decode_attachment(attachment: CompetitionFile) -> bytes:
    return base64.b64decode(attachment.content_base64)


def materialize_attachment(attachment: CompetitionFile, base_dir: Path | None = None) -> Path:
    directory = base_dir or Path(tempfile.mkdtemp(prefix="tripletex_attachments_"))
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(attachment.filename).name or "attachment.bin"
    path = directory / safe_name
    path.write_bytes(decode_attachment(attachment))
    return path


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _extract_text(filename: str, mime_type: str | None, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        return _decode_text(data)
    if mime_type == "application/pdf" or suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            return ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)
        try:
            reader = PdfReader(str(temp_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        finally:
            temp_path.unlink(missing_ok=True)
    return ""


def summarize_attachments(attachments: Iterable[CompetitionFile]) -> list[AttachmentSummary]:
    summaries: list[AttachmentSummary] = []
    for attachment in attachments:
        data = decode_attachment(attachment)
        text = _extract_text(attachment.filename, attachment.mime_type, data).strip()
        summaries.append(
            AttachmentSummary(
                filename=attachment.filename,
                mime_type=attachment.mime_type or mimetypes.guess_type(attachment.filename)[0],
                sha1=hashlib.sha1(data).hexdigest(),
                size_bytes=len(data),
                text_excerpt=text[:4000],
            )
        )
    return summaries
