from pathlib import Path
import numpy as np
from ai.data_loader import data_preprocessor, process_file
from ai.embed import embedding
from ai.models import llm
from ai.vector_db import vector_db

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class RagPipeline:
    def __init__(self):
        self.chunks = []
        self.vectors = []
        self.vector_db = vector_db
        self.data_preprocessor = data_preprocessor

    async def data_ingestion(self, file, chunk_size: int = 200, overlap: int = 50):
        # 1. Save uploaded file to uploads/ directory if it's an UploadFile
        if hasattr(file, "filename") and hasattr(file, "read"):
            filename = file.filename or "uploaded_document.pdf"
            save_path = UPLOAD_DIR / filename
            if hasattr(file, "seek"):
                await file.seek(0)
            file_bytes = await file.read()
            save_path.write_bytes(file_bytes)
            file_target = save_path
        elif isinstance(file, (str, Path)):
            file_target = Path(file)
            if not file_target.exists():
                file_target = None
        else:
            file_target = None

        # 2. Extract pages using DataPreprocessor
        if file_target and file_target.exists() and file_target.name.lower().endswith(".pdf"):
            doc_info = await self.data_preprocessor.load_pdf(file_target)
            pages = doc_info["pages"]
            file_hash = doc_info["file_hash"]
            filename = doc_info["filename"]
        else:
            raw_text = str(file)
            filename = "raw_text"
            file_hash = "direct_input"
            pages = [{"page_number": 1, "text": raw_text}] if raw_text.strip() else []

        if not pages:
            return {
                "chunks": [],
                "total_chunks": 0,
                "embeddings": [],
                "embeddings_count": 0,
                "message": "No text extracted from input"
            }

        # 3. Chunk pages using DataPreprocessor.chunk_pages
        chunk_dicts = self.data_preprocessor.chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        if not chunk_dicts:
            return {
                "chunks": [],
                "total_chunks": 0,
                "embeddings": [],
                "embeddings_count": 0,
                "message": "No text extracted from input"
            }

        chunks = [c["text"] for c in chunk_dicts]
        metadatas = [
            {
                "filename": filename,
                "file_hash": file_hash,
                "page_number": c.get("page_number", 1),
                "chunk_index": c.get("chunk_index", i)
            }
            for i, c in enumerate(chunk_dicts)
        ]

        # 4. Generate embeddings
        vectors = await embedding.embed_chunks(chunks)

        # 5. Save to vector database with metadata and keep in memory for compatibility
        await self.vector_db.add_documents(chunks=chunks, embeddings=vectors, metadatas=metadatas)
        self.chunks = chunks
        self.vectors = vectors

        return {
            "filename": filename,
            "file_hash": file_hash,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "embeddings": vectors,
            "embeddings_count": len(vectors),
            "message": f"File '{filename}' saved to uploads, chunked, embedded, and stored in vector database successfully"
        }

    async def generation(self, query: str):
        if self.vector_db.count() == 0 and not self.chunks:
            return {"query": query, "answer": "No documents uploaded yet", "answer_from_file": ""}

        # 1. query -> vector
        query_vector = await embedding.embed_query(query)

        # 2. find best chunk from vector database (with fallback to in-memory cosine similarity)
        best_chunk, best_score = await self.vector_db.query_best(query_vector)
        if not best_chunk and self.chunks and self.vectors:
            for chunk, vec in zip(self.chunks, self.vectors):
                denom = np.linalg.norm(query_vector) * np.linalg.norm(vec)
                score = np.dot(query_vector, vec) / denom if denom > 0 else 0
                if score > best_score:
                    best_score = score
                    best_chunk = chunk

        # 3. generate answer using LLM
        answer = await llm.generate_answer(query, best_chunk)

        return {
            "query": query,
            "answer": answer,
            "answer_from_file": best_chunk,
            "score": round(float(best_score), 4)
        }

rag_pipeline = RagPipeline()