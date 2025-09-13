# sr/retrieval/dedupe.py
from __future__ import annotations
from typing import List
from sr.config.schema import Record

def title_key(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum() or ch.isspace()).strip()

def dedupe(records: List[Record]) -> List[Record]:
    seen_pmid=set(); seen_doi=set(); seen_title=set(); out=[]
    for r in records:
        if r.pmid and r.pmid in seen_pmid: continue
        if r.doi and r.doi in seen_doi: continue
        tk = title_key(r.title or "")
        if tk and tk in seen_title: continue
        out.append(r)
        if r.pmid: seen_pmid.add(r.pmid)
        if r.doi: seen_doi.add(r.doi)
        if tk: seen_title.add(tk)
    return out
