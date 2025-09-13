# sr/ranking/rrf.py
from __future__ import annotations
from typing import Dict, List
from sr.config.defaults import RRF_K

def rrf_fuse(ranks: Dict[str, Dict[str, int]], k: int = RRF_K) -> Dict[str, float]:
    """
    ranks: pmid -> {list_name: rank_int (1-based)}
    Return: pmid -> fused score (higher is better)
    """
    out: Dict[str, float] = {}
    for pmid, comps in ranks.items():
        s = 0.0
        for _, r in comps.items():
            if r <= 0: continue
            s += 1.0 / (k + r)
        out[pmid] = s
    return out
