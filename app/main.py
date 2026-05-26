from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import ingest_pdf, ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    result = ingest_pdf(contents, file.filename)
    return {"message": result}

@app.post("/ask")
async def ask(q: Question):
    answer = ask_question(q.question)
    return {"answer": answer}
