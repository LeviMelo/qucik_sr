# sr/ranking/signals.py
from __future__ import annotations
import math
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sr.config.schema import Record, Protocol, Signals
from sr.llm.client import embed_texts

def _text(rec: Record) -> str:
    return f"{rec.title or ''}\n{rec.abstract or ''}".strip()

def build_tfidf_corpus(records: List[Record]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(ngram_range=(1,3), lowercase=True, max_features=150000)
    docs = [_text(r) for r in records]
    X = vec.fit_transform(docs)
    return vec, X

def tfidf_query_cos(vec: TfidfVectorizer, X: np.ndarray, query_text: str) -> np.ndarray:
    q = vec.transform([query_text])
    denom = (np.linalg.norm(X.toarray(), axis=1) * (np.linalg.norm(q.toarray()) + 1e-12) + 1e-12)
    sims = (X @ q.T).toarray().ravel() / denom
    return sims.astype(float)

def embed_cosine(records: List[Record], query: str) -> np.ndarray:
    # Title-weight the query by repeating title
    embs = embed_texts([query] + [ (r.title or "") + "\n" + (r.abstract or "") for r in records ])
    qv = np.array(embs[0], dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-12)
    M = []
    for i in range(1, len(embs)):
        v = np.array(embs[i], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        M.append(float(np.dot(qv, v)))
    return np.array(M, dtype=float)

def count_hits(text: str, syns: List[str]) -> int:
    tl = (text or "").lower()
    return sum(1 for s in syns if s.lower() in tl)

PRIMARY_HINTS = {"Randomized Controlled Trial","Clinical Trial","Controlled Clinical Trial","Prospective Studies","Cohort Studies","Case-Control Studies"}

def compute_signals(records: List[Record], proto: Protocol, tfidf_vec: TfidfVectorizer, X: np.ndarray, q_text: str) -> Dict[str, Signals]:
    tf = tfidf_query_cos(tfidf_vec, X, q_text)
    em = embed_cosine(records, q_text)

    # recency scaled: linear from year_min to (current_year ~ 2025)
    cur_year = 2025
    ymin = proto.picos.year_min or (cur_year - 15)

    out: Dict[str, Signals] = {}
    for i, r in enumerate(records):
        pit = count_hits(r.title or "", [proto.picos.population] + proto.picos.synonyms_population + [proto.picos.intervention] + proto.picos.synonyms_intervention)
        pia = count_hits(r.abstract or "", [proto.picos.population] + proto.picos.synonyms_population + [proto.picos.intervention] + proto.picos.synonyms_intervention)
        design_prior = 1.0 if (set(r.publication_types) & PRIMARY_HINTS) else 0.0
        recency = 0.0
        if r.year is not None:
            recency = max(0.0, min(1.0, (r.year - ymin) / max(1, cur_year - ymin)))
        out[r.pmid] = Signals(
            pi_hits_title=pit,
            pi_hits_abstract=pia,
            tfidf_cos=float(tf[i]),
            embed_cos=float(em[i]),
            design_prior=design_prior,
            recency_scaled=recency,
            abstract_missing=(len((r.abstract or "").strip())==0)
        )
    return out
