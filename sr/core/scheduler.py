# sr/core/scheduler.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sr.config.schema import Record, Protocol, Signals, RRFScore
from sr.ranking.rrf import rrf_fuse
from sr.ranking.ees import EESModel

def build_ranks(records: List[Record], signals: Dict[str, Signals]) -> Dict[str, Dict[str,int]]:
    # Build individual rank lists (lower rank = better); we store 1-based ranks
    pmids = [r.pmid for r in records]

    def rank_by(key):
        vals = [(pid, getattr(signals[pid], key)) for pid in pmids]
        # Descending sort (higher is better)
        vals.sort(key=lambda t: (-t[1], t[0]))
        return {pid: (i+1) for i,(pid,_) in enumerate(vals)}

    r_tfidf = rank_by("tfidf_cos")
    r_embed = rank_by("embed_cos")
    r_pi    = rank_by("pi_hits_title")
    r_design= rank_by("design_prior")
    r_recent= rank_by("recency_scaled")

    out: Dict[str, Dict[str,int]] = {}
    for pid in pmids:
        out[pid] = {
            "tfidf": r_tfidf.get(pid, 10**9),
            "embed": r_embed.get(pid, 10**9),
            "pi":    r_pi.get(pid, 10**9),
            "design":r_design.get(pid, 10**9),
            "recency":r_recent.get(pid, 10**9),
        }
    return out

def fuse_to_rrf(ranks: Dict[str, Dict[str,int]]) -> Dict[str, RRFScore]:
    fused = rrf_fuse(ranks)
    # Build back component ranks for audit
    out: Dict[str, RRFScore] = {}
    for pmid, score in fused.items():
        out[pmid] = RRFScore(score=float(score), components=ranks[pmid])
    return out

def pick_frontier(rrf_scores: Dict[str, RRFScore], size: int) -> List[str]:
    items = sorted(rrf_scores.items(), key=lambda kv: (-kv[1].score, kv[0]))
    return [pid for pid,_ in items[:size]]

def ees_within_frontier(ees: EESModel, signals: Dict[str, Signals], frontier: List[str], take: int) -> List[str]:
    preds = ees.predict({pid: signals[pid] for pid in frontier})
    # sort by EES descending
    ordered = sorted(frontier, key=lambda pid: (-preds.get(pid, 0.5), pid))
    return ordered[:take]
