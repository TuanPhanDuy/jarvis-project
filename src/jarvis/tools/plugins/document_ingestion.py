"""Plugin: ingest_document — extract text from PDF/DOCX/TXT and index into memory.

Supported formats:
  .pdf  — requires: pip install pypdf
  .docx — requires: pip install python-docx
  .txt / .md — built-in, no extra deps

The extracted text is saved as a .md report (so it appears in search_memory)
and indexed into ChromaDB automatically.

SCHEMA / handle() follow the standard plugin interface.
"""
from __future__ import annotations

from pathlib import Path


SCHEMA: dict = {
    "name": "ingest_document",
    "description": (
        "Extract text from a local PDF, DOCX, or plain-text file and index it into "
        "JARVIS memory so it becomes searchable. Provide the absolute file path."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the document (PDF, DOCX, TXT, or MD).",
            },
            "title": {
                "type": "string",
                "description": "Optional title for the indexed document. Defaults to filename.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to extract (default 50000).",
                "default": 50000,
            },
        },
        "required": ["file_path"],
    },
}


def _extract_pdf(path: Path, max_chars: int) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        parts = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n".join(parts)[:max_chars]
    except ImportError:
        return "ERROR: pypdf not installed. Run: pip install pypdf"


def _extract_docx(path: Path, max_chars: int) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[:max_chars]
    except ImportError:
        return "ERROR: python-docx not installed. Run: pip install python-docx"


def _extract_text(path: Path, max_chars: int) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, max_chars)
    if suffix == ".docx":
        return _extract_docx(path, max_chars)
    if suffix in {".txt", ".md", ".rst", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    return f"ERROR: unsupported file type '{suffix}'. Supported: .pdf .docx .txt .md"


def handle(tool_input: dict) -> str:
    try:
        from jarvis.config import get_settings
        from jarvis.tools.memory import index_new_report

        settings = get_settings()
        reports_dir = Path(settings.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        file_path = Path(tool_input["file_path"])
        if not file_path.exists():
            return f"ERROR: file not found — {file_path}"

        max_chars = int(tool_input.get("max_chars", 50000))
        text = _extract_text(file_path, max_chars)
        if text.startswith("ERROR"):
            return text

        title = tool_input.get("title") or file_path.stem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        report_name = f"doc_{safe_name}.md"
        report_path = reports_dir / report_name

        report_path.write_text(
            f"# {title}\n\n*Ingested from: {file_path.name}*\n\n{text}",
            encoding="utf-8",
        )
        index_new_report(reports_dir, report_name)

        word_count = len(text.split())
        return (
            f"Document ingested: '{title}' ({word_count} words, {len(text)} chars). "
            f"Saved as '{report_name}' and indexed into memory."
        )
    except Exception as e:
        return f"ERROR: ingest_document failed — {e}"
