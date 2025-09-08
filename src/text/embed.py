from __future__ import annotations
import numpy as np, requests
from typing import List
from src.config.defaults import LMSTUDIO_BASE, LMSTUDIO_EMB, HTTP_TIMEOUT, USER_AGENT, EMB_BATCH

HEADERS = {"Content-Type":"application/json","User-Agent":USER_AGENT}

def embed_texts(texts: List[str], batch: int = EMB_BATCH) -> np.ndarray:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/embeddings"
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        body = {"model": LMSTUDIO_EMB, "input": texts[i:i+batch]}
        r = requests.post(url, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()["data"]
        out.extend(d["embedding"] for d in data)
    arr = np.array(out, dtype="float32")
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr
