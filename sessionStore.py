from typing import List, Dict, Optional
import numpy as np

class DocStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.metadatas: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

SESSION_STORES: Dict[str, DocStore] = {}

def getDocStore(sid: str) -> DocStore:
    if sid not in SESSION_STORES:
        return None
    return SESSION_STORES[sid]

def createDocStore(sid: str) -> DocStore:
    if sid in SESSION_STORES:
        raise ValueError(f"Session store with id {sid} already exists.")
    store = DocStore()
    SESSION_STORES[sid] = store
    return store

def removeDocStore(sid: str):
    if sid in SESSION_STORES:
        del SESSION_STORES[sid]
    else:
        raise ValueError(f"No session store found with id {sid}.")