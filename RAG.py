from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

import sessionStore



def get_embedder():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

_embedding_model: Optional[SentenceTransformer] = None

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def retrieve_relevant(query: str, store: sessionStore.DocStore, k: int = 5) -> List[Dict]:
    if not store.chunks:
        return []
    embedder = get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True)
    sims = cosine_sim_matrix(qvec, store.embeddings)
    idxs = np.argsort(-sims[0])[:k]
    results = []
    for i in idxs:
        results.append({
            "chunk": store.chunks[i],
            "metadata": store.metadatas[i],
            "score": float(sims[0, i]),
        })
    return results