from fastapi import APIRouter, File, UploadFile
from ai.rag_pipeline import rag_pipeline

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Ingest file, split chunks, generate embeddings, and return results
    return await rag_pipeline.data_ingestion(file)

@router.post("/query")
async def query_endpoint(query: str):
    # Search embeddings and generate answer from best chunk
    return await rag_pipeline.generation(query)