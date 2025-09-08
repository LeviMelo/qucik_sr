from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
from src.config.schema import Document, Signals
from src.text.embed import embed_texts
from src.net.icite import icite_neighbors_map

def _band_sem(x: float) -> str:
    if x >= 0.90: return "Very high"
    if x >= 0.85: return "High"
    if x >= 0.75: return "Medium"
    return "Low"

def abstract_len_bin(txt: str) -> str:
    if not txt: return "none"
    n = len(txt)
    if n < 400: return "short"
    if n < 2400: return "normal"
    return "long"

def build_embeddings(docs: List[Document]) -> Tuple[np.ndarray, Dict[str,int]]:
    texts = [ (d.title or "") + "\n" + (d.abstract or "") for d in docs ]
    mat = embed_texts(texts)
    idx = {docs[i].pmid: i for i in range(len(docs))}
    return mat, idx

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def compute_signals(
    docs: List[Document],
    emb: np.ndarray,
    idx: Dict[str,int],
    intent_vec: np.ndarray,
    seed_pmids: List[str],
    year_min: int | None
) -> Dict[str, Signals]:
    seed_idx = [idx[p] for p in seed_pmids if p in idx]
    if seed_idx:
        cent = emb[seed_idx].mean(axis=0); cent /= (np.linalg.norm(cent)+1e-12)
    else:
        cent = intent_vec
    pmids = [d.pmid for d in docs]
    neigh_map = icite_neighbors_map(pmids)
    G = nx.Graph()
    for i,p in enumerate(pmids):
        G.add_node(p)
    for p in pmids:
        ns = neigh_map.get(p, set())
        for q in ns:
            if str(q) in idx: G.add_edge(p, str(q))
    seeds = [p for p in seed_pmids if p in idx]
    if seeds:
        personalize = {p: 1.0/len(seeds) for p in seeds}
        ppr = nx.pagerank(G, alpha=0.85, personalization=personalize, max_iter=100)
        vals = np.array(list(ppr.values()), dtype="float32")
        ranks = {k: 100.0 * (float(np.sum(vals <= v)) / max(1,len(vals))) for k,v in ppr.items()}
    else:
        ppr = {p: 0.0 for p in pmids}
        ranks = {p: 0.0 for p in pmids}
    seeds_set = set(seeds)
    signals: Dict[str, Signals] = {}
    for d in docs:
        v = emb[idx[d.pmid]]
        sem_intent = cosine(v, intent_vec)
        sem_seed   = cosine(v, cent)
        neighbors = neigh_map.get(d.pmid, set())
        links_frac = 0.0
        if neighbors:
            links_frac = len(set(str(x) for x in neighbors) & seeds_set) / float(len(neighbors))
        ys = 0.0
        if year_min and d.year is not None:
            ys = max(0.0, min(1.0, (d.year - year_min) / (2025 - year_min)))
        signals[d.pmid] = Signals(
            sem_intent=sem_intent,
            sem_seed=sem_seed,
            graph_ppr_pct=float(ranks.get(d.pmid, 0.0)),
            graph_links_frac=float(links_frac),
            year_scaled=float(ys),
            abstract_len_bin=abstract_len_bin(d.abstract or "")
        )
    return signals

def signal_card(sig) -> str:
    return (f"Signals:\n"
            f"- Semantic match: {_band_sem(sig.sem_intent)} ({sig.sem_intent:.2f}) to intent; "
            f"seed-centroid {_band_sem(sig.sem_seed)} ({sig.sem_seed:.2f}).\n"
            f"- Graph locality: {'High' if sig.graph_ppr_pct>=90.0 else 'Medium' if sig.graph_ppr_pct>=60.0 else 'Low'} — "
            f"links_to_seeds {100.0*sig.graph_links_frac:.1f}%, PPR {sig.graph_ppr_pct:.0f}th pct.\n"
            f"- Abstract: {sig.abstract_len_bin}.")
