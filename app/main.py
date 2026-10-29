from fastapi import FastAPI
from app.routes import file
app = FastAPI()

@app.get("/")
def landing_page():
    return {"message": "vanga padinga"}


@app.get("/health")
def health():
    return {"status": "all good"}
  
    
app.include_router(file.router)

