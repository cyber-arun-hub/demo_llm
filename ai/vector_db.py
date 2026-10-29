import uuid
from typing import List, Tuple, Optional, Dict, Any
import chromadb

class VectorDB:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "rag_documents"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Inserts text chunks along with their vector embeddings into ChromaDB."""
        if not chunks or not embeddings:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        if metadatas is None:
            metadatas = [{"chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return ids

    async def query_best(self, query_embedding: List[float]) -> Tuple[str, float]:
        """Queries the vector database for the single best matching chunk."""
        if self.count() == 0:
            return "", -1.0

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "distances"]
        )

        docs = results.get("documents", [[]])
        distances = results.get("distances", [[]])

        if docs and docs[0] and distances and distances[0]:
            best_chunk = docs[0][0]
            # In cosine space, cosine distance = 1 - cosine similarity
            # Therefore similarity = 1 - distance
            cosine_similarity = 1.0 - distances[0][0]
            return best_chunk, float(cosine_similarity)

        return "", -1.0

    async def query_top_k(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Queries the top k most relevant documents from the vector database."""
        count = self.count()
        if count == 0:
            return []

        n_results = min(k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )

        items = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(docs, distances, metas):
            items.append({
                "chunk": doc,
                "score": round(float(1.0 - dist), 4),
                "metadata": meta
            })
        return items

    def count(self) -> int:
        """Returns the total number of chunks stored in the vector database."""
        return self.collection.count()

    def clear(self):
        """Clears all records stored in the vector database collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

vector_db = VectorDB()
