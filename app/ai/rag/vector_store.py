import asyncio
from pathlib import Path
import chromadb
from app.utils.ids import generate_uuid

# ChromaDB folder located in the project root directory
ROOT_DIR = Path(__file__).resolve().parents[3]
CHROMA_DIR = ROOT_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))


class VectorStore:
    def __init__(self):

        self.client = client

        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Collection tracking ingested documents"}
        )

        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"description": "Collection containing chunk text, embeddings, and metadata"}
        )

    async def get_document_by_hash(self, file_hash: str) -> dict | None:
        def _get():

            results = self.documents_collection.get(
                where={"file_hash": file_hash},
                limit=1
            )

            if results and results.get("ids") and len(results["ids"]) > 0:
                doc_id = results["ids"][0]
                metadata = results["metadatas"][0] if results.get("metadatas") else {}
                return {
                    "document_id": doc_id,
                    **metadata
                }

            return None

        return await asyncio.to_thread(_get)

    async def get_document_by_id(self, document_id: str) -> dict | None:
        def _get():

            results = self.documents_collection.get(
                ids=[document_id],
                limit=1
            )

            if results and results.get("ids") and len(results["ids"]) > 0:
                metadata = results["metadatas"][0] if results.get("metadatas") else {}
                return {
                    "document_id": document_id,
                    **metadata
                }
            return None

        return await asyncio.to_thread(_get)

    async def insert(
        self,
        document_record: dict,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> None:
        def _insert():

            # 1. Insert Document Record
            self.documents_collection.add(
                ids=[document_record["document_id"]],
                documents=[document_record["filename"]],
                metadatas=[{
                    "document_id": document_record["document_id"],
                    "filename": document_record["filename"],
                    "file_hash": document_record["file_hash"],
                    "file_size_bytes": int(document_record["file_size_bytes"]),
                    "uploaded_at": str(document_record["uploaded_at"]),
                    "ingested_at": str(document_record["ingested_at"]),
                    "ingestion_status": str(document_record["ingestion_status"]),
                    "chunk_count": int(document_record["chunk_count"]),
                    "chunking_version": str(document_record["chunking_version"]),
                    "embedding_model": str(document_record["embedding_model"]),
                }]
            )

            # 2. Insert Chunks with embeddings and metadata
            if chunks and embeddings:
                
                chunk_ids = [chunk.get("id") or generate_uuid() for chunk in chunks]
                chunk_docs = [chunk["text"] for chunk in chunks]
                chunk_metas = [{
                    "document_id": document_record["document_id"],
                    "filename": document_record["filename"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "page_number": int(chunk["page_number"])
                } for chunk in chunks]

                self.chunks_collection.add(
                    ids=chunk_ids,
                    documents=chunk_docs,
                    embeddings=embeddings,
                    metadatas=chunk_metas
                )

        await asyncio.to_thread(_insert)

    async def retrieve(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None
    ) -> list[dict]:

        def _retrieve():

            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }

            if document_id:
                kwargs["where"] = {"document_id": document_id}

            results = self.chunks_collection.query(**kwargs)

            formatted_results = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = (
                    results.get("distances", [[]])[0]
                    if "distances" in results and results["distances"]
                    else [None] * len(ids)
                )

                for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
                    formatted_results.append({
                        "id": chunk_id,
                        "document": doc,
                        "metadata": meta,
                        "distance": dist
                    })

            return formatted_results

        return await asyncio.to_thread(_retrieve)


vector_store = VectorStore()


