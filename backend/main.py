import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import (
    load_pdf,
    split_docs,
    create_embeddings,
    store_in_faiss,
    load_faiss,
    ask_question
)

app = FastAPI(
    title="RAG Chatbot API",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for vector DBs
vector_store_cache = {}

# Shared embeddings model
embeddings_model = create_embeddings()


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG Chatbot API running"
    }


@app.get("/status")
def status(x_session_id: Optional[str] = Header(None)):

    if not x_session_id:
        return {
            "db_ready": False,
            "message": "No session ID provided"
        }

    db_ready = (
        x_session_id in vector_store_cache
        or load_faiss(x_session_id, embeddings_model) is not None
    )

    return {
        "db_ready": db_ready,
        "session_id": x_session_id
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None)
):

    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Session-ID header is required"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")

    temp_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join("temp", temp_filename)

    try:

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PDF
        docs = load_pdf(file_path)

        print("PAGES LOADED:", len(docs))

        # Split into chunks
        chunks = split_docs(docs)

        print("TOTAL CHUNKS:", len(chunks))

        if chunks:
            print(
                "FIRST CHUNK SAMPLE:\n",
                chunks[0].page_content[:500]
            )

        # Store in FAISS
        db = store_in_faiss(
            chunks,
            embeddings_model,
            x_session_id
        )

        print("FAISS STORED SUCCESSFULLY")

        # Cache DB
        vector_store_cache[x_session_id] = db

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "chunks": len(chunks),
            "session_id": x_session_id
        }

    except Exception as e:

        print("UPLOAD ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/ask")
def ask(
    query: str,
    x_session_id: Optional[str] = Header(None)
):

    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Session-ID header is required"
        )

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # Get vector DB
    db = vector_store_cache.get(x_session_id)

    if db is None:
        db = load_faiss(x_session_id, embeddings_model)

        if db:
            vector_store_cache[x_session_id] = db

    if db is None:
        raise HTTPException(
            status_code=400,
            detail="No document found for this session"
        )

    try:

        answer, sources = ask_question(db, query)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        print("ASK ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

