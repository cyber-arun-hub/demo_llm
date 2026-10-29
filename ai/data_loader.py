from pathlib import Path
import pymupdf
from app.core.security import get_file_hash

class DataPreprocessor:

    async def load_pdf(self, file_path: str | Path) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        if not path.is_file() or not path.name.lower().endswith(".pdf"):
            raise ValueError(f"Path must be an existing PDF file: {file_path}")

        file_bytes = path.read_bytes()
        file_hash = get_file_hash(file_bytes)
        file_size_bytes = len(file_bytes)

        pages = []
        with pymupdf.open(str(path)) as pdf:
            for page in pdf:
                page_text = page.get_text()
                if page_text and page_text.strip():
                    pages.append({
                        "page_number": page.number + 1,
                        "text": page_text
                    })

        return {
            "file_path": str(path),
            "filename": path.name,
            "file_bytes": file_bytes,
            "file_hash": file_hash,
            "file_size_bytes": file_size_bytes,
            "pages": pages
        }

    def chunk_pages(self, pages: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
        chunks = []
        chunk_index = 0

        for page in pages:
            words = page["text"].split()
            if not words:
                continue

            step = chunk_size - overlap
            for start in range(0, len(words), step):
                chunk_words = words[start:start + chunk_size]
                chunks.append({
                    "text": " ".join(chunk_words),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index
                })
                chunk_index += 1

                if start + chunk_size >= len(words):
                    break

        return chunks

data_preprocessor = DataPreprocessor()

async def process_file(file, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Convenience backward-compatible wrapper."""
    if isinstance(file, (str, Path)) and Path(file).exists() and Path(file).name.lower().endswith(".pdf"):
        doc = await data_preprocessor.load_pdf(file)
        chunks = data_preprocessor.chunk_pages(doc["pages"], chunk_size=chunk_size, overlap=overlap)
        return [c["text"] for c in chunks]
    pages = [{"page_number": 1, "text": str(file)}]
    chunks = data_preprocessor.chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return [c["text"] for c in chunks]