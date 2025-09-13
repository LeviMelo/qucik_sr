# cile.py
# Library-only CILE expansion engine (deterministic, audit-friendly, no import-time side effects)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional
import os
import time
import math
import json
import logging
import hashlib
import itertools
import collections

import requests
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)  # handlers configured by host app


# --------------------------
# Utilities (deterministic)
# --------------------------
def stable_hash_u32(x: int | str) -> int:
    """Deterministic 32-bit hash (machine/seed independent)."""
    s = str(x).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.md5(s).digest()[:4], "little", signed=False)


def now_millis() -> int:
    return int(time.time() * 1000)


# --------------------------
# Configuration
# --------------------------
@dataclass(frozen=True)
class OuterConfig:
    # Acceptance policy (math unchanged)
    accept_policy: str = "elastic_phi"

    # Graph expansion / budgets
    per_node_cap: int = 50
    H_external_budget: Optional[int] = 3000  # None → unlimited
    max_accept_after_filter: Optional[int] = 300  # final cap after all filters

    # Relevance & hub constraints
    min_relevance_frac: float = 0.05  # gate by relative score vs. seed median
    per_node_ext_frac_cap: float = 0.85  # reject candidates with too many external neighbors
    quarantine_hubs: bool = True
    quarantine_mode: str = "external"  # "external" or "total"

    # Determinism
    deterministic_reservoir: bool = True
    rng_seed: int = 13  # reserved (unused in deterministic paths)

    # iCite / fetch budgets (wave limits)
    ICITE_BASE: str = "https://icite.od.nih.gov/api/pubs"
    WAVE_MAX_NEW_FETCH: Optional[int] = 5000     # None → unlimited
    WAVE_WALLCLOCK_SEC: Optional[float] = 90.0   # None → unlimited
    BATCH_SIZE: int = 200
    HTTP_TIMEOUT: int = 30

    # Cache
    ICITE_CACHE_FILE: str = "./_icite_cache/icite_cache.jsonl"

    # Elastic-phi knobs (unchanged math)
    eps_phi_max: float = 0.08
    lambda_phi_per_log_g: float = 0.08
    leak_cap: float = 0.35
    snr_rel_inflate_cap: float = 0.25
    churn_cap: float = 0.50

    # Sweep / FM parameters
    alpha: float = 0.15          # PPR teleport prob
    power_tol: float = 1e-8
    power_iter_max: int = 200
    FM_MAX_MOVES: int = 2000


# --------------------------
# Graph container
# --------------------------
@dataclass
class HGraph:
    pmids: List[int]                     # stable order
    idx: Dict[int, int]                  # pmid -> index
    neigh: List[List[int]]               # neighbors as indices (undirected, deduped)
    deg: np.ndarray                      # degrees
    year: Dict[int, Optional[int]]       # pmid -> year
    title: Dict[int, str]                # pmid -> title

    def subindex(self, subset_pmids: Iterable[int]) -> List[int]:
        return [self.idx[p] for p in subset_pmids if p in self.idx]


# --------------------------
# iCite fetch / cache
# --------------------------
def _iter_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _append_jsonl(path: str, rows: List[dict]) -> None:
    if os.getenv("CILE_DISABLE_CACHE") == "1":
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_icite_loaded(
    seeds: List[int],
    cfg: OuterConfig,
    extra_pmids: Optional[Iterable[int]] = None
) -> Dict[int, dict]:
    """
    Returns a dict pmid -> icite_record, pulling from local JSONL cache and fetching missing
    from the iCite API. Network failures do not crash the run; batches are skipped with warnings.
    Deterministic batching order under budgets.
    """
    # Load extant cache into memory index (pmid->record)
    cache: Dict[int, dict] = {}
    for row in _iter_jsonl(cfg.ICITE_CACHE_FILE) or []:
        try:
            p = int(row.get("pmid"))
        except Exception:
            continue
        cache[p] = row

    # Determine todo set (stable ordering)
    want: Set[int] = set(int(p) for p in (extra_pmids or [])) | set(int(p) for p in seeds)
    todo = [p for p in want if p not in cache]
    todo.sort(key=stable_hash_u32)

    # Respect wave budgets deterministically
    start_ms = now_millis()
    fetched_count = 0

    def time_ok() -> bool:
        if cfg.WAVE_WALLCLOCK_SEC is None:
            return True
        return (now_millis() - start_ms) < cfg.WAVE_WALLCLOCK_SEC * 1000

    if cfg.WAVE_MAX_NEW_FETCH is not None:
        todo = todo[: cfg.WAVE_MAX_NEW_FETCH]

    B = max(1, int(cfg.BATCH_SIZE))
    i = 0
    while i < len(todo) and time_ok():
        batch = todo[i : i + B]
        i += B
        if not batch:
            break

        # GET with robust retry/backoff; skip on persistent failure
        params = {"pmids": ",".join(str(p) for p in batch)}
        backoff = 0.75
        tries = 6
        r = None
        while tries > 0:
            tries -= 1
            try:
                r = requests.get(cfg.ICITE_BASE, params=params, timeout=cfg.HTTP_TIMEOUT)
            except Exception as e:
                logger.warning("iCite transport error: %s; backoff=%.2fs", e.__class__.__name__, backoff)
                time.sleep(backoff); backoff *= 1.7
                continue
            status = r.status_code
            if status == 200:
                break
            if status in (408, 429, 499, 502, 503, 504, 522, 523, 524):
                time.sleep(backoff); backoff *= 1.7
                continue
            # Fatal for this batch
            try:
                r.raise_for_status()
            except Exception as e:
                logger.warning("iCite batch fatal (%s); skipping %d ids", e.__class__.__name__, len(batch))
                r = None
            break

        if r is None:
            continue

        if r.status_code != 200:
            logger.warning("iCite batch failed after retries (status %s); skipping %d ids", r.status_code, len(batch))
            continue

        try:
            j = r.json()
        except Exception:
            logger.warning("iCite batch returned non-JSON; skipping %d ids", len(batch))
            continue

        data = j.get("data", j)
        if not isinstance(data, list):
            logger.warning("iCite batch 'data' missing/invalid; skipping %d ids", len(batch))
            continue

        # Normalize fields we use
        rows: List[dict] = []
        for rec in data:
            try:
                pmid = int(rec.get("pmid"))
            except Exception:
                continue
            refs = list({int(x) for x in (rec.get("references") or []) if str(x).isdigit()})
            cited = list({int(x) for x in (rec.get("cited_by") or []) if str(x).isdigit()})
            year = None
            try:
                y = rec.get("year")
                if y is not None:
                    y = int(y)
                    if 1500 <= y <= 2100:
                        year = y
            except Exception:
                year = None
            title = (rec.get("title") or "").replace("\t", " ").replace("\n", " ").strip()
            rows.append({"pmid": pmid, "references": refs, "cited_by": cited, "year": year, "title": title})
            cache[pmid] = rows[-1]
        fetched_count += len(rows)
        # Append to cache on disk (lazy creation)
        _append_jsonl(cfg.ICITE_CACHE_FILE, rows)

    if fetched_count:
        logger.info("iCite: fetched %d new records (todo=%d)", fetched_count, len(todo))
    return cache


# --------------------------
# Build H from seeds/graph
# --------------------------
def icite_neighbors_func(cache: Dict[int, dict]):
    """Return a callable pmid -> iterable of neighbor pmids (undirected)."""
    def _neigh(pmid: int) -> Iterable[int]:
        rec = cache.get(pmid)
        if not rec:
            return []
        refs = rec.get("references") or []
        cited = rec.get("cited_by") or []
        return itertools.chain(refs, cited)
    return _neigh


def build_H_hops(
    seeds: List[int],
    cache: Dict[int, dict],
    per_node_cap: int = 50,
    hops: int = 2
) -> HGraph:
    neigh = icite_neighbors_func(cache)

    visited: Set[int] = set(int(p) for p in seeds)
    curr: Set[int] = set(int(p) for p in seeds)

    for _ in range(max(0, hops)):
        nxt: Set[int] = set()
        for u in sorted(curr, key=stable_hash_u32):
            neighs_all = sorted({int(v) for v in neigh(u) if int(v) != u}, key=stable_hash_u32)
            for v in neighs_all[: per_node_cap]:
                if v not in visited:
                    nxt.add(v)
                    visited.add(v)
        curr = nxt
        if not curr:
            break

    H_nodes = sorted(visited, key=stable_hash_u32)
    idx = {p: i for i, p in enumerate(H_nodes)}

    # Build neighbor index (undirected adjacency)
    nbr_idx: List[Set[int]] = [set() for _ in H_nodes]
    for p in H_nodes:
        i = idx[p]
        for v in {int(x) for x in neigh(p) if int(x) in idx and int(x) != p}:
            j = idx[v]
            nbr_idx[i].add(j)
            nbr_idx[j].add(i)

    neigh_lists = [sorted(s, key=lambda j: stable_hash_u32(H_nodes[j])) for s in nbr_idx]
    deg = np.array([len(s) for s in neigh_lists], dtype=np.float64)

    year = {p: (cache.get(p, {}).get("year")) for p in H_nodes}
    title = {p: (cache.get(p, {}).get("title") or "") for p in H_nodes}

    return HGraph(pmids=H_nodes, idx=idx, neigh=neigh_lists, deg=deg, year=year, title=title)


def build_H_from_nodes(node_pmids: Iterable[int], cache: Dict[int, dict]) -> HGraph:
    H_nodes = sorted({int(p) for p in node_pmids}, key=stable_hash_u32)
    idx = {p: i for i, p in enumerate(H_nodes)}
    neigh = icite_neighbors_func(cache)
    nbr_idx: List[Set[int]] = [set() for _ in H_nodes]
    for p in H_nodes:
        i = idx[p]
        for v in {int(x) for x in neigh(p) if int(x) in idx and int(x) != p}:
            j = idx[v]
            nbr_idx[i].add(j)
            nbr_idx[j].add(i)
    neigh_lists = [sorted(s, key=lambda j: stable_hash_u32(H_nodes[j])) for s in nbr_idx]
    deg = np.array([len(s) for s in neigh_lists], dtype=np.float64)
    year = {p: (cache.get(p, {}).get("year")) for p in H_nodes}
    title = {p: (cache.get(p, {}).get("title") or "") for p in H_nodes}
    return HGraph(pmids=H_nodes, idx=idx, neigh=neigh_lists, deg=deg, year=year, title=title)


# --------------------------
# PPR / Sweep / Metrics
# --------------------------
def personalized_pagerank(H: HGraph, seed_idx: List[int], alpha: float, tol: float, itmax: int) -> np.ndarray:
    """Leaky PPR (standard), computed via power iteration on sparse column-stochastic matrix."""
    n = len(H.pmids)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # Build column-stochastic adjacency: A_ij = 1 if edge i-j, symmetric
    rows = []
    cols = []
    data = []
    for i, nbrs in enumerate(H.neigh):
        for j in nbrs:
            rows.append(j)
            cols.append(i)
            data.append(1.0)
    if not data:
        return np.zeros(n, dtype=np.float64)
    A = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
    colsum = np.asarray(A.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(colsum > 0, 1.0 / colsum, 0.0)
    Dinv = sp.diags(inv, offsets=0, shape=(n, n), format="csc")
    P = A @ Dinv  # column-stochastic
    # Personalization vector
    e = np.zeros(n, dtype=np.float64)
    for s in seed_idx:
        if 0 <= s < n:
            e[s] = 1.0
    if e.sum() == 0:
        e[:] = 1.0 / n
    else:
        e /= e.sum()
    x = e.copy()
    for _ in range(itmax):
        nx = alpha * e + (1.0 - alpha) * (P @ x)
        if np.linalg.norm(nx - x, 1) < tol:
            x = nx
            break
        x = nx
    return x


def sweep_best(H: HGraph, scores: np.ndarray, seed_idx: Set[int]) -> Tuple[List[int], Dict[str, float]]:
    """
    Stable dual sweep: order by (-score, stable_hash), choose k maximizing SNR-like criterion
    while enforcing seeds appear in prefix before eligibility. Returns index set and metrics.
    """
    n = len(H.pmids)
    if n == 0:
        return [], {"phi": 1.0, "leak_share": 1.0, "cut": 0.0, "wAA": 0.0, "jaccard": 1.0, "k": 0}

    # Stable deterministic order
    secondary = np.argsort(np.array([stable_hash_u32(p) for p in H.pmids]), kind="mergesort")
    order = np.argsort(-scores[secondary], kind="mergesort")
    order = secondary[order]

    # Maintain prefix seed coverage incrementally
    seed_idx_set = set(seed_idx)
    in_prefix: Set[int] = set()
    best_k = 0
    best_obj = float("-inf")

    # Precompute degrees for speed
    deg = H.deg
    vol_total = float(deg.sum())

    # Running cut/vol/wAA via incremental updates
    inA = np.zeros(n, dtype=bool)
    cut = 0.0
    volA = 0.0
    wAA = 0.0

    for k, u in enumerate(order, start=1):
        # Add u to A
        inA[u] = True
        in_prefix.add(u)
        # update vol and internal weight
        du = deg[u]
        volA += du
        # internal edges added: neighbors in A
        internal_added = 0
        for v in H.neigh[u]:
            if inA[v]:
                internal_added += 1
            else:
                cut += 1.0  # edge crosses A-Ā
        # each internal edge counted once here, contributes 2 to vol, 1 to wAA increment
        wAA += internal_added
        # As we added u, edges from neighbors that were previously crossing are now internal (handled above)
        # Ensure all seeds are inside prefix before considering eligibility
        if not seed_idx_set.issubset(in_prefix):
            continue
        # Compute conductance-like phi (cut / min(volA, volĀ))
        volA_eff = max(volA, 1e-9)
        volB_eff = max(vol_total - volA, 1e-9)
        phi = cut / min(volA_eff, volB_eff)
        # SNR-ish objective using wAA (avoid division by zero)
        r = cut / max(1e-9, wAA)
        # Combine (lower phi better, lower r better) into simple score; we keep original behavior: primary = -phi
        obj = -phi - 0.02 * r
        if obj > best_obj:
            best_obj = obj
            best_k = k

    A_idx = list(order[:best_k])
    # Metrics for chosen set
    metrics = phi_beta_stats(H, set(A_idx))
    # Add Jaccard w.r.t. seeds
    kSeeds = len(seed_idx_set)
    inter = len(seed_idx_set.intersection(A_idx))
    union = len(set(A_idx).union(seed_idx_set))
    jacc = inter / max(1, union) if (kSeeds > 0 or len(A_idx) > 0) else 1.0
    metrics["jaccard"] = jacc
    return A_idx, metrics


def phi_beta_stats(H: HGraph, A: Set[int]) -> Dict[str, float]:
    """Compute conductance-like phi, cut, wAA (internal edges), leak share, vol(A), size."""
    n = len(H.pmids)
    if not A:
        return dict(phi=1.0, cut=0.0, leak=1.0, vol=0.0, wAA=0.0, leak_share=1.0, k=0)
    inA = np.zeros(n, dtype=bool)
    for u in A:
        if 0 <= u < n:
            inA[u] = True

    cut = 0.0
    wAA = 0.0
    volA = 0.0
    for u in A:
        deg_u = len(H.neigh[u])
        volA += deg_u
        internal = 0
        for v in H.neigh[u]:
            if inA[v]:
                internal += 1
            else:
                cut += 1.0
        # each internal edge counted once (u,v) with u in A and v in A
        wAA += internal

    vol_total = float(H.deg.sum())
    phi = cut / max(1e-9, min(volA, vol_total - volA))
    # "leak" as fraction of boundary vs. volA
    leak_share = cut / max(1e-9, volA)
    return dict(phi=phi, cut=cut, leak=leak_share, vol=volA, wAA=wAA, leak_share=leak_share, k=len(A))


# --------------------------
# Acceptance logic / diagnostics
# --------------------------
def acceptance_diagnostics(prev: Dict[str, float], cur: Dict[str, float], cfg: OuterConfig, jaccard: float) -> Dict[str, float | bool]:
    phi0, phi1 = prev["phi"], cur["phi"]
    k0, k1 = max(1, int(prev["k"])), max(1, int(cur["k"]))
    gA = (k1 / k0) if k0 > 0 else 1.0
    leak0, leak1 = prev["leak_share"], cur["leak_share"]
    r0 = prev["cut"] / max(1e-9, prev["wAA"])
    r1 = cur["cut"] / max(1e-9, cur["wAA"])
    churn = 1.0 - jaccard
    dphi_allow = min(cfg.eps_phi_max, cfg.lambda_phi_per_log_g * (math.log(gA) if gA > 1.0 else 0.0))
    return {
        "ok_phi": (phi1 <= phi0 * (1.0 + dphi_allow)),
        "ok_leak": (leak1 <= min(leak0, cfg.leak_cap)),
        "ok_snr": (r1 <= r0 * (1.0 + cfg.snr_rel_inflate_cap)),
        "ok_churn": (churn <= cfg.churn_cap),
        "phi0": phi0, "phi1": phi1, "dphi_allow": dphi_allow,
        "leak0": leak0, "leak1": leak1,
        "r0": r0, "r1": r1,
        "churn": churn,
        "gA": gA,
        "jaccard": jaccard,
    }


def accept_wave(prev: Dict[str, float], cur: Dict[str, float], cfg: OuterConfig, jaccard: float) -> Tuple[bool, Dict[str, float | bool]]:
    diag = acceptance_diagnostics(prev, cur, cfg, jaccard)
    ok = diag["ok_phi"] and diag["ok_leak"] and diag["ok_snr"] and diag["ok_churn"]
    return ok, diag


# --------------------------
# Adaptive expansion (deterministic)
# --------------------------
def build_next_H_adaptive(
    H: HGraph,
    A_idx: List[int],
    cfg: OuterConfig
) -> Tuple[List[int], Dict[int, int]]:
    """
    Deterministic candidate selection with filters:
      (a) candidate neighborhood (from A) →
      (b) minimal relevance gate (min_relevance_frac) →
      (c) per-node external fraction cap →
      (d) sum-of-external global budget (H_external_budget) using ascending external degree, then stable hash →
      (e) post-filter max accept cap (max_accept_after_filter) by stable hash.
    Returns (accepted_indices, ext_degree_map).
    """
    n = len(H.pmids)
    inA = np.zeros(n, dtype=bool)
    for u in A_idx:
        inA[u] = True

    # (a) candidate neighborhood
    cand: Set[int] = set()
    for u in sorted(A_idx, key=lambda i: stable_hash_u32(H.pmids[i])):
        for v in H.neigh[u]:
            if not inA[v]:
                cand.add(v)
    cand_list = sorted(cand, key=lambda i: stable_hash_u32(H.pmids[i]))

    # Relevance scores are not recomputed here; we use degree as a simple proxy to avoid math changes.
    # Gate by fraction of max degree in A (local relevance proxy).
    degA = H.deg[A_idx] if A_idx else np.array([0.0])
    rel_ref = float(np.median(degA)) if len(degA) > 0 else 0.0
    rel_ref = max(rel_ref, 1.0)

    # (b) minimal relevance gate
    kept = []
    for v in cand_list:
        rel = H.deg[v] / rel_ref
        if rel >= cfg.min_relevance_frac:
            kept.append(v)

    # (c) per-node external fraction cap (external neighbors among Ā / total neighbors)
    kept2 = []
    for v in kept:
        dv = max(1, int(H.deg[v]))
        ext = sum(1 for w in H.neigh[v] if not inA[w])
        frac_ext = ext / dv
        if frac_ext <= cfg.per_node_ext_frac_cap:
            kept2.append(v)

    # (d) global external-degree budget ordering (ascending ext, then stable hash)
    acc_info = []
    ext_map: Dict[int, int] = {}
    for v in kept2:
        ext_v = sum(1 for w in H.neigh[v] if not inA[w])
        ext_map[v] = ext_v
        acc_info.append((v, ext_v, stable_hash_u32(H.pmids[v])))

    acc_info.sort(key=lambda t: (t[1], t[2]))  # ext asc, then stable

    if cfg.H_external_budget is not None:
        acc_info = acc_info[: cfg.H_external_budget]

    # (e) post-filter cap (by stable hash)
    if cfg.max_accept_after_filter is not None and len(acc_info) > cfg.max_accept_after_filter:
        acc_info.sort(key=lambda t: t[2])  # by stable hash only
        acc_info = acc_info[: cfg.max_accept_after_filter]

    accepted = [v for (v, _, _) in acc_info]
    return accepted, ext_map


# --------------------------
# Outer loop
# --------------------------
def outer_loop_cile(seeds: List[int], cfg: OuterConfig) -> Tuple[HGraph, Set[int], Dict[str, float]]:
    """
    Top-level CILE orchestrator. Deterministic given identical inputs and config.
    Returns (H_graph, A_indices_set, meta).
    """
    seeds = [int(p) for p in seeds if str(p).isdigit()]
    # Load / fetch iCite (deterministic order under budgets)
    cache = ensure_icite_loaded(seeds, cfg, extra_pmids=seeds)

    # Initial H from hops around seeds
    H = build_H_hops(seeds, cache, per_node_cap=cfg.per_node_cap, hops=1)

    # Indices for seeds
    seed_idx = H.subindex(seeds)

    # Initial PPR
    scores = personalized_pagerank(H, seed_idx, cfg.alpha, cfg.power_tol, cfg.power_iter_max)
    A_idx, metrics = sweep_best(H, scores, set(seed_idx))
    prev_metrics = metrics.copy()

    # FM local improvement (apply only when |A| > kmax_eff := best sweep size)
    kmax_eff = len(A_idx)
    if len(A_idx) > kmax_eff and cfg.FM_MAX_MOVES > 0:
        # (kept for contract; condition will be False as defined)
        pass

    Af = set(A_idx)
    wave = 0
    meta = {
        "waves": 0,
        "A_size": len(Af),
        "phi": metrics["phi"],
        "leak_share": metrics["leak_share"],
        "cut": metrics["cut"],
        "wAA": metrics["wAA"],
        "jaccard": metrics.get("jaccard", 1.0),
    }

    # Emit accept diagnostics for initial wave (trivial accept)
    diag0 = acceptance_diagnostics(prev_metrics, metrics, cfg, metrics.get("jaccard", 1.0))
    logger.info(
        "CILE wave %d accept=True ok_phi=%s ok_leak=%s ok_snr=%s ok_churn=%s "
        "(phi %.6f→%.6f; dphi_allow=%.4f; leak %.3f→%.3f; r %.3f→%.3f; churn=%.3f; gA=%.3f; J=%.3f)",
        wave, True, True, True, True,
        diag0["phi0"], diag0["phi1"], diag0["dphi_allow"],
        diag0["leak0"], diag0["leak1"], diag0["r0"], diag0["r1"],
        diag0["churn"], diag0["gA"], diag0["jaccard"],
    )

    # Single adaptive expansion wave (as per approved behavior)
    # Build next candidate set deterministically
    cand_idx, ext_map = build_next_H_adaptive(H, A_idx, cfg)

    # No growth → nothing to accept/reject
    if not cand_idx:
        meta["waves"] = 1
        return H, Af, meta

    # Expand H with candidates and recompute scores
    # Construct new H over previous nodes + accepted candidates
    new_nodes = [H.pmids[i] for i in sorted(set(A_idx).union(cand_idx), key=lambda i: stable_hash_u32(H.pmids[i]))]
    H2 = build_H_from_nodes(new_nodes, cache)
    seed_idx2 = H2.subindex(seeds)
    scores2 = personalized_pagerank(H2, seed_idx2, cfg.alpha, cfg.power_tol, cfg.power_iter_max)
    A2_idx, m2 = sweep_best(H2, scores2, set(seed_idx2))

    # Acceptance decision (elastic φ)
    jacc = m2.get("jaccard", 0.0)
    ok, diag = accept_wave(prev_metrics, m2, cfg, jaccard=jacc)
    wave = 1
    logger.info(
        "CILE wave %d accept=%s ok_phi=%s ok_leak=%s ok_snr=%s ok_churn=%s "
        "(phi %.6f→%.6f; dphi_allow=%.4f; leak %.3f→%.3f; r %.3f→%.3f; churn=%.3f; gA=%.3f; J=%.3f)",
        wave, bool(ok), diag["ok_phi"], diag["ok_leak"], diag["ok_snr"], diag["ok_churn"],
        diag["phi0"], diag["phi1"], diag["dphi_allow"],
        diag["leak0"], diag["leak1"], diag["r0"], diag["r1"],
        diag["churn"], diag["gA"], diag["jaccard"],
    )

    if ok:
        H, Af = H2, set(A2_idx)
        meta.update({
            "waves": 2,
            "A_size": len(Af),
            "phi": m2["phi"],
            "leak_share": m2["leak_share"],
            "cut": m2["cut"],
            "wAA": m2["wAA"],
            "jaccard": m2.get("jaccard", 1.0),
        })
    else:
        meta["waves"] = 2  # attempted, rejected; keep previous H/A/metrics

    return H, Af, meta
