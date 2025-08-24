"""
FastAPI RAG Chat App — Single File
----------------------------------

✅ Not Streamlit — uses FastAPI + vanilla JS + Tailwind CDN
✅ Upload multiple files (PDF, TXT, MD, DOCX*)
✅ On-the-fly chunking + embeddings (sentence-transformers)
✅ Simple in-memory vector store per user session (cosine similarity)
✅ Chat endpoint does RAG over your uploaded content
✅ Works with OpenAI **or** local Ollama (config via env vars)

*DOCX requires optional dependency `python-docx`.

---
Quickstart
----------
1) Create a virtualenv, then install deps:

    pip install -U fastapi uvicorn pydantic "python-multipart" pypdf python-docx "sentence-transformers<3" numpy openai requests

2) Set your LLM provider (choose ONE):
   - OpenAI: set `OPENAI_API_KEY` in your env (and optional `OPENAI_MODEL`, default: gpt-4o-mini)
   - Ollama: set `USE_OLLAMA=1` (and optional `OLLAMA_MODEL`, default: llama3.1)

3) Run the server:

    uvicorn main:app --reload --port 8000

4) Open http://localhost:8000 in your browser.

Notes
-----
- This demo uses an in-memory store per-session (cookie). For production, swap for SQLite/pg + persistent vector DB (e.g., Chroma, FAISS, pgvector).
- For large files, consider background tasks and async parsers. This minimal app keeps things simple.
"""

import os
import time
import uuid
from typing import List, Dict, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


import sessionStore, llmCall, RAG, util


app = FastAPI(title="RAG Chat App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_session(request: Request, response: Response) -> str:
    sid = request.cookies.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        response.set_cookie("sid", sid, httponly=False, samesite="lax")
    if sessionStore.getDocStore(sid) is None:
        sessionStore.createDocStore(sid)
    return sid

def build_prompt(user_msg: str, contexts: List[Dict]) -> str:
    context_blocks = []
    for i, r in enumerate(contexts, 1):
        src = r["metadata"].get("source", "uploaded")
        context_blocks.append(f"[Doc {i} | {src} | score={r['score']:.3f}]\n{r['chunk']}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "(no documents uploaded)"
    system = (
        "You are a concise, accurate assistant. Use the provided context snippets to answer the "
        "user's question. If the answer isn't in the context, say so and optionally suggest where to look. "
        "Cite document numbers when helpful."
    )
    prompt = (
        f"System:\n{system}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User:\n{user_msg}\n\nAssistant:"
    )
    return prompt


#Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):

    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_file_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


class UploadResp(BaseModel):
    chunks_loaded: int
    total_tokens_est: int

class ChatReq(BaseModel):
    message: str

class ChatResp(BaseModel):
    answer: str
    citations: List[Dict]


@app.post("/upload", response_model=UploadResp)
async def upload(request: Request, response: Response, files: List[UploadFile] = File(...)):
    print(f"Upload: {[f.filename for f in files]}")
    sid = _ensure_session(request, response)
    store = sessionStore.getDocStore(sid)

    new_chunks = []
    new_metas = []
    for f in files:
        text = util.parse_file(f)
        chunks = util.chunk_text(text)
        new_chunks.extend(chunks)
        new_metas.extend({"source": f.filename, "chunk_index": i} for i in range(len(chunks)))

    if not new_chunks:
        return UploadResp(chunks_loaded=0, total_tokens_est=0)

    new_vecs = RAG.embed_texts(new_chunks)

    if store.embeddings is None:
        store.embeddings = new_vecs
        store.chunks = list(new_chunks)
        store.metadatas = list(new_metas)
    else:
        store.embeddings = np.vstack([store.embeddings, new_vecs])
        store.chunks.extend(new_chunks)
        store.metadatas.extend(new_metas)

    total_tokens_est = sum(len(c.split()) for c in store.chunks)
    return UploadResp(chunks_loaded=len(new_chunks), total_tokens_est=total_tokens_est)




@app.post("/chat", response_model=ChatResp)
async def chat(request: Request, response: Response, body: ChatReq):
    sid = _ensure_session(request, response)
    store = sessionStore.getDocStore(sid)
    print(f"Chat Reques: {body.message}")

    results = RAG.retrieve_relevant(body.message, store, k=5)
    prompt = build_prompt(body.message, results)

    return ChatResp(
        answer=llmCall.generate_response(prompt),
        citations=[{"source": r["metadata"].get("source"), "score": r["score"]} for r in results],
    )


@app.post("/reset")
async def reset(request: Request, response: Response):
    print("Reset")
    sid = _ensure_session(request, response)
    sessionStore.removeDocStore(sid)
    sessionStore.createDocStore(sid)
    return JSONResponse({"ok": True})
