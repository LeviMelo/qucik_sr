from __future__ import annotations
import sqlite3, pathlib, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.config.defaults import VEC_DB_PATH

_PATH = pathlib.Path(VEC_DB_PATH)
_PATH.parent.mkdir(parents=True, exist_ok=True)

def _hash_text(model: str, text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(model.encode("utf-8", "ignore") + b"\x00" + text.encode("utf-8", "ignore"))
    return h.hexdigest()

class VecDB:
    def __init__(self, path: pathlib.Path = _PATH):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS embeds(
          pmid TEXT NOT NULL,
          model TEXT NOT NULL,
          hash TEXT NOT NULL,
          dim  INTEGER NOT NULL,
          vec  BLOB NOT NULL,
          PRIMARY KEY (pmid, model)
        )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_embeds_hash ON embeds(hash)")
        self.conn.commit()

    def get_many(self, keys: List[Tuple[str,str]]) -> Dict[Tuple[str,str], Tuple[str,int,bytes]]:
        if not keys: return {}
        q = ",".join(["(?,?)"]*len(keys))
        flat = []
        for pmid, model in keys: flat.extend([pmid, model])
        cur = self.conn.execute(f"SELECT pmid,model,hash,dim,vec FROM embeds WHERE (pmid,model) IN ({q})", flat)
        out = {}
        for pmid, model, h, dim, blob in cur.fetchall():
            out[(pmid, model)] = (h, int(dim), blob)
        return out

    def upsert_many(self, rows: List[Tuple[str,str,str,int,bytes]]) -> int:
        if not rows: return 0
        self.conn.executemany("INSERT OR REPLACE INTO embeds(pmid,model,hash,dim,vec) VALUES(?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    @staticmethod
    def make_hash(model: str, text: str) -> str:
        return _hash_text(model, text)

    @staticmethod
    def pack_vec(x: np.ndarray) -> bytes:
        assert x.dtype == np.float32
        return x.tobytes()

    @staticmethod
    def unpack_vec(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            raise ValueError(f"vec size {arr.size} != dim {dim}")
        return arr
