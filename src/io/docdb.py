from __future__ import annotations
import sqlite3, pathlib, json
from typing import Dict, List
from src.config.defaults import DATA_DIR

DB_PATH = pathlib.Path(DATA_DIR) / "cache" / "docs.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

class DocDB:
    def __init__(self, path: pathlib.Path = DB_PATH):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""CREATE TABLE IF NOT EXISTS docs(
            pmid TEXT PRIMARY KEY,
            json TEXT NOT NULL
        )""")
        self.conn.commit()

    def get_many(self, pmids: List[str]) -> Dict[str, dict]:
        if not pmids: return {}
        q = ",".join("?"*len(pmids))
        cur = self.conn.execute(f"SELECT pmid,json FROM docs WHERE pmid IN ({q})", pmids)
        out = {}
        for pmid, blob in cur.fetchall():
            try: out[pmid] = json.loads(blob)
            except Exception: pass
        return out

    def put_many(self, rows: Dict[str, dict]) -> int:
        if not rows: return 0
        data = [(pmid, json.dumps(rec)) for pmid, rec in rows.items()]
        self.conn.executemany("INSERT OR REPLACE INTO docs(pmid,json) VALUES(?,?)", data)
        self.conn.commit()
        return len(data)
