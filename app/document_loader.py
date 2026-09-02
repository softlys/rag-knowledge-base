from io import BytesIO

from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    pass


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(content)
    if lower.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    raise UnsupportedFileTypeError(f"Неподдерживаемый тип файла: {filename} (нужен .pdf или .txt)")


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages_text)
