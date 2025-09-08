from __future__ import annotations
from typing import List, Dict, Set, Tuple
import random
from src.net.icite import icite_neighbors_map, icite_degree_total

def one_wave_expand(
    seeds_pos: List[int],
    H_existing: Set[int] | None,
    rel_gate_frac: float = 0.08,
    external_budget: int = 2500,
    max_accept: int = 600,
    hub_quarantine_external: bool = True,
    min_hub_soft: int = 450,
    rng_seed: int = 13
) -> Tuple[Set[int], Dict[str,int]]:
    random.seed(rng_seed)
    seeds_pos = [int(x) for x in seeds_pos]
    H_set = set(H_existing or set())
    neigh_map = icite_neighbors_map([str(s) for s in seeds_pos])
    neighborhood: Set[int] = set()
    for s in seeds_pos:
        neighborhood |= set(neigh_map.get(str(s), set()))
        neighborhood.add(s)
    candidates = list(neighborhood - H_set)
    accepted: List[Tuple[int,int]] = []
    total_ext = 0
    tmp = []
    for v in candidates:
        deg_tot = icite_degree_total(v)
        if deg_tot <= 0:
            continue
        vn = neigh_map.get(str(v))
        if vn is None:
            vn = icite_neighbors_map([str(v)]).get(str(v), set())
        rel = len(set(int(x) for x in vn) & set(seeds_pos)) / max(1, deg_tot)
        if rel < rel_gate_frac:
            continue
        ext = len(set(int(x) for x in vn) - H_set)
        if hub_quarantine_external and ext >= min_hub_soft:
            continue
        tmp.append((v, ext))
    tmp.sort(key=lambda t: (t[1], t[0]))
    for v, ext in tmp:
        if total_ext + ext > external_budget:
            continue
        accepted.append((v, ext))
        total_ext += ext
        if len(accepted) >= max_accept:
            break
    acc_nodes = set(v for v, _ in accepted)
    meta = {"H_prev": len(H_set), "neighborhood": len(neighborhood), "candidates": len(candidates),
            "accepted": len(acc_nodes), "sum_ext_after": total_ext}
    return (H_set | acc_nodes), meta
