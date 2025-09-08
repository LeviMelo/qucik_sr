from __future__ import annotations
import requests, sqlite3, json, pathlib, time
from typing import Dict, List
from src.config.defaults import ICITE_BASE, HTTP_TIMEOUT, USER_AGENT, DATA_DIR

HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
CACHE_DIR = pathlib.Path(DATA_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "icite.sqlite3"

class ICiteCache:
    def __init__(self, path: pathlib.Path = DB_PATH):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("""CREATE TABLE IF NOT EXISTS pubs(
            pmid TEXT PRIMARY KEY, json TEXT NOT NULL
        )""")
        self._conn.commit()

    def get_many(self, pmids: List[str]) -> Dict[str, dict]:
        if not pmids: return {}
        q = ",".join("?"*len(pmids))
        cur = self._conn.execute(f"SELECT pmid,json FROM pubs WHERE pmid IN ({q})", pmids)
        out = {}
        for pmid, blob in cur.fetchall():
            try: out[pmid] = json.loads(blob)
            except Exception: pass
        return out

    def put_many(self, rows: List[dict]) -> int:
        data = []
        for rec in rows:
            pmid = str(rec.get("pmid") or rec.get("_id") or "")
            if not pmid: continue
            data.append((pmid, json.dumps(rec)))
        if not data: return 0
        self._conn.executemany("INSERT OR REPLACE INTO pubs(pmid,json) VALUES(?,?)", data)
        self._conn.commit()
        return len(data)

def icite_pubs_fetch(pmids: List[str]) -> List[dict]:
    out: List[dict] = []
    base = ICITE_BASE.rstrip("/")
    for i in range(0, len(pmids), 200):
        sub = pmids[i:i+200]
        params = {"pmids": ",".join(sub), "legacy": "false"}
        r = requests.get(base, headers=HEADERS, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json().get("data", r.json())
        if isinstance(data, list):
            out.extend(data)
        time.sleep(0.34)
    return out

def icite_neighbors_map(pmids: List[str]) -> Dict[str, set]:
    cache = ICiteCache()
    have = cache.get_many(pmids)
    need = [p for p in pmids if p not in have]
    if need:
        rows = icite_pubs_fetch(need)
        cache.put_many(rows)
        for rec in rows:
            have[str(rec.get("pmid") or rec.get("_id"))] = rec
    m = {}
    for p, rec in have.items():
        cited_by = rec.get("citedByPmids", []) or rec.get("cited_by") or []
        refs     = rec.get("citedPmids", [])   or rec.get("references") or []
        m[p] = set(int(x) for x in (cited_by + refs) if x)
    return m

def icite_degree_total(pmid: int) -> int:
    cache = ICiteCache()
    have = cache.get_many([str(pmid)])
    rec = have.get(str(pmid), {})
    cited_by = rec.get("citedByPmids", []) or rec.get("cited_by") or []
    refs     = rec.get("citedPmids", [])   or rec.get("references") or []
    return len(cited_by) + len(refs)
