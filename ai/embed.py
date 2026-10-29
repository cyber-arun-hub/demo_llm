from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def embed_chunks(self, chunks):
        # chunks = list of strings -> list of normalized vectors
        if not chunks:
            return []
        if isinstance(chunks, str):
            chunks = [chunks]
        return self.model.encode(chunks, normalize_embeddings=True).tolist()

    async def embed_query(self, prompt: str):
        if not prompt or not prompt.strip():
            return [0.0] * 384
        return self.model.encode([prompt], normalize_embeddings=True)[0].tolist()

embedding = Embedding()