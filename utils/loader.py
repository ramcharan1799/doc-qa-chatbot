from pypdf import PdfReader
import re

def load_pdf(file) -> list[dict]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    chunk_id = 0
    for p in pages:
        text = p["text"]
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append({"chunk_id": chunk_id, "page": p["page"], "text": chunk})
            chunk_id += 1
            start += chunk_size - overlap
    return chunks
