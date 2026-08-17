# api.py
# Run with: uvicorn api:app --reload --port 8000
#
# This exposes the same RAG pipeline as a REST API, so any frontend
# (React, Vue, plain HTML, a mobile app, etc.) can talk to it over HTTP
# instead of using the Streamlit UI or the terminal.

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Day_4.src.settings import settings
from src.pdf_loader import read_pdf
from src.chunking import split_text_into_chunk
from src.vector_database import create_vectordb, load_vectordb
from src.rag import ask_question
# from src.schemas import ChatAnswer

app = FastAPI(title="PDF RAG Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")  #---- decorators(wrappers)
def health():
    """Simple endpoint so a frontend (or a load balancer) can check the API is up."""
    return {"status": "ok"}


@app.get("/status")
def status():
    """Tell the frontend whether a PDF has already been loaded."""
    return {"ready": vectorstore is not None}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, process it, and build the vector database."""
    global vectorstore

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    # Save the uploaded file to a temp path, since read_pdf() expects a file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_path = tmp_file.name

    pages = read_pdf(tmp_path)
    texts, metadatas = split_text_into_chunk(pages, pdf_file=file.filename)
    vectorstore = create_vectordb(texts, metadatas)

    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(texts),
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    """Ask a question about the uploaded PDF."""
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF loaded yet. Upload one first via /upload.",
        )

    return ask_question(vectorstore, request.question)


@app.on_event("startup")
def try_load_existing_db():
    """On startup, try to resume a previously saved vector database,
    same idea as the check we added to main.py earlier."""
    global vectorstore
    if os.path.exists(settings.persist_directory) and os.listdir(settings.persist_directory):
        vectorstore = load_vectordb()
        print("Resumed existing vector database on startup.")