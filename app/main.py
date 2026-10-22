from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def landing_page():
    return {"message": "vanga padinga "}


@app.get("/health")
def health():
    return {"status": "all good"}
    


