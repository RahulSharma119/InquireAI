from typing import List
import io
import base64
from fastapi import UploadFile
from pypdf import PdfReader
import docx
import config
import time


CHUNK_SIZE = config.get_config("DEFAULT","CHUNK_SIZE", fallback="800")
CHUNK_OVERLAP = config.get_config("DEFAULT","CHUNK_OVERLAP", fallback="120")

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def parse_file(file: UploadFile) -> str:
    filename = file.filename or "uploaded_"+time.localtime()
    ext = filename.lower().split(".")[-1]
    content = file.file.read()

    if ext in ["txt", "md", "csv"]:
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")

    if ext in ["pdf"]:
        reader = PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(texts)

    if ext in ["docx"] and docx is not None:
        document = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs)

    b64 = base64.b64encode(content[:120]).decode()
    return f"[Unsupported file type: .{ext}] Base64 preview: {b64}"