from __future__ import annotations
import os, time, json, math, subprocess, shutil, logging
from typing import List, Dict, Tuple, Optional
import numpy as np, requests

from src.config.defaults import (
    LMSTUDIO_BASE, LMSTUDIO_EMB, HTTP_TIMEOUT, USER_AGENT,
    EMB_BATCH, EMB_AUTO_BATCH, EMB_MAX_CHARS_PER_DOC,
    EMB_RETRY_BACKOFF_S, EMB_RETRY_MAX
)
from src.io.vecdb import VecDB

HEADERS = {"Content-Type":"application/json","User-Agent":USER_AGENT}
log = logging.getLogger("embed")

def _setup_logger_once():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOGLEVEL","INFO"),
            format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )

_setup_logger_once()

def _estimate_tokens(s: str) -> int:
    # Rough: ~ 1 token ≈ 4 chars (English-ish)
    return max(1, int(len(s) / 4))

def _truncate(s: str, max_chars: int) -> str:
    if max_chars and len(s) > max_chars:
        return s[:max_chars]
    return s

def _detect_vram_mb() -> Optional[int]:
    # Try torch first
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()  # bytes
            return int(free // (1024*1024))
    except Exception:
        pass
    # nvidia-smi
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.check_output(
                [nvsmi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=2.0
            )
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
            if vals:
                # if multi-GPU, take max free
                return max(vals)
        except Exception:
            pass
    return None

def _choose_batch_size(auto: bool, default_batch: int) -> int:
    if not auto:
        return default_batch
    free_mb = _detect_vram_mb()
    if free_mb is None:
        # unknown; be conservative
        return min(default_batch, 16)
    # Coarse heuristic: allow more if plenty free, otherwise tighten
    if free_mb < 2000:   return 8
    if free_mb < 4000:   return 12
    if free_mb < 8000:   return 16
    if free_mb < 12000:  return 20
    if free_mb < 20000:  return 24
    return min(32, default_batch)

def _post_embeddings(payload_inputs: List[str], model: str, timeout: int, retries: int) -> List[List[float]]:
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/embeddings"
    body = {"model": model, "input": payload_inputs}
    backoff = EMB_RETRY_BACKOFF_S
    for attempt in range(retries+1):
        try:
            r = requests.post(url, headers=HEADERS, json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()["data"]
            return [d["embedding"] for d in data]
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(backoff)
            backoff *= 1.8
    raise RuntimeError("unreachable")

def embed_texts(texts: List[str], batch: Optional[int] = None, model: Optional[str] = None) -> np.ndarray:
    """
    Generic embedders (no cache). Still auto-batches & truncates.
    """
    model = model or LMSTUDIO_EMB
    bsz = _choose_batch_size(EMB_AUTO_BATCH, batch if batch is not None else EMB_BATCH)
    work: List[str] = [_truncate(t or "", EMB_MAX_CHARS_PER_DOC) for t in texts]
    N = len(work)
    out: List[List[float]] = []
    t0 = time.time()
    log.info(f"Embedding {N} texts | model={model} | batch={bsz} | max_chars={EMB_MAX_CHARS_PER_DOC}")
    for i in range(0, N, bsz):
        sub = work[i:i+bsz]
        embs = _post_embeddings(sub, model=model, timeout=HTTP_TIMEOUT, retries=EMB_RETRY_MAX)
        out.extend(embs)
        if (i//bsz) % 5 == 0:
            done = i+len(sub)
            log.info(f"  progress {done}/{N} ({100.0*done/max(1,N):.1f}%)")
    arr = np.array(out, dtype="float32")
    # normalize
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    log.info(f"Embedding done in {time.time()-t0:.1f}s")
    return arr

def embed_docs_with_cache(pmids: List[str], texts: List[str], model: Optional[str] = None) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    pmids/texts aligned. Uses sqlite cache keyed by (pmid, model) + content hash.
    Auto-batches requests and logs progress. Returns matrix in pmids order.
    """
    assert len(pmids) == len(texts)
    model = model or LMSTUDIO_EMB
    bsz = _choose_batch_size(EMB_AUTO_BATCH, EMB_BATCH)

    # Precompute trunc, hash
    trunc = [_truncate(t or "", EMB_MAX_CHARS_PER_DOC) for t in texts]
    hashes = [VecDB.make_hash(model, t) for t in trunc]

    db = VecDB()
    have = db.get_many([(pmids[i], model) for i in range(len(pmids))])

    # Decide which need (missing or hash changed)
    to_do_idx: List[int] = []
    cached_vecs: Dict[int, np.ndarray] = {}
    need_hash_rows: List[Tuple[str,str]] = []
    for i, p in enumerate(pmids):
        key = (p, model)
        if key in have:
            h, dim, blob = have[key]
            if h == hashes[i]:
                cached_vecs[i] = VecDB.unpack_vec(blob, dim)
            else:
                to_do_idx.append(i)
        else:
            to_do_idx.append(i)

    log.info(f"Embeddings cache: hit={len(cached_vecs)} | miss={len(to_do_idx)} | model={model} | batch={bsz}")

    # Batch the misses
    rows_for_db: List[Tuple[str,str,str,int,bytes]] = []
    if to_do_idx:
        N = len(to_do_idx)
        t0 = time.time()
        for s in range(0, N, bsz):
            idxs = to_do_idx[s:s+bsz]
            payload = [trunc[i] for i in idxs]
            embs = _post_embeddings(payload, model=model, timeout=HTTP_TIMEOUT, retries=EMB_RETRY_MAX)
            for j, i in enumerate(idxs):
                vec = np.array(embs[j], dtype="float32")
                vec /= (np.linalg.norm(vec) + 1e-12)
                cached_vecs[i] = vec
                rows_for_db.append((pmids[i], model, hashes[i], int(vec.size), VecDB.pack_vec(vec)))
            if (s//bsz) % 5 == 0:
                done = s + len(idxs)
                log.info(f"  embed new {done}/{N} ({100.0*done/max(1,N):.1f}%)")
        db.upsert_many(rows_for_db)
        log.info(f"Embedded {N} new vecs in {time.time()-t0:.1f}s (cached saved to DB).")

    # Assemble output in pmids order
    dim = next(iter(cached_vecs.values())).size if cached_vecs else 0
    mat = np.zeros((len(pmids), dim), dtype="float32")
    for i, p in enumerate(pmids):
        mat[i,:] = cached_vecs[i]
    idx = {pmids[i]: i for i in range(len(pmids))}
    return mat, idx
