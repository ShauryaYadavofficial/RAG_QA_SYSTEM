import io
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def parse_file(file_path: Path) -> str:
    """
    Dispatch to the correct parser based on file extension.
    Returns the raw text content of the document.
    """
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return _parse_pdf(file_path)
    if ext in (".txt", ".md"):
        return _parse_text(file_path)
    if ext == ".docx":
        return _parse_docx(file_path)

    raise ValueError(f"No parser registered for '{ext}'")  # unreachable


def _parse_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


def _parse_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())