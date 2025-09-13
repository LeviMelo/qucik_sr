#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CILE (compute-bounded, topic-agnostic expansion) with growth-aware acceptance.

Inner (fixed-H): leaky PPR -> dual sweep (x vs z) -> FM polish (size-bounded)
Outer (iterative expansion): build unbounded 1-hop from A, then apply *compute-only*
controls based on EXTERNAL degree (how many NEW nodes a candidate would open), not
content. Optional minimal relevance gate (can be 0 to disable). Optional hub quarantine
with *soft floor*.

Acceptance: default "elastic φ" — allow small φ increase if |A| grows a lot and
boundary hygiene (leak_share, cut/wAA) doesn’t worsen; softer churn cap.

Dependencies: requests, numpy, scipy, networkx
"""

import os, time, json, math, random, hashlib
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import requests
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

# =============================================================================
# Global config
# =============================================================================

SEED_PMIDS = [40720602, 40122203, 39522714, 37693640, 37364610, 32929487, 30935731]

ICITE_BASE = "https://icite.od.nih.gov/api/pubs"
CACHE_DIR  = "./_icite_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ICITE_CACHE_FILE = os.path.join(CACHE_DIR, "icite_cache.jsonl")
ICITE_SLEEP = 0.34
HTTP_TIMEOUT = 30

# Initial H (one-time BFS)
INIT_HOPS = 1
INIT_PER_NODE_CAP = 500

# Graph weighting knobs (off by default)
TRIANGLE_ALPHA = 0.0
RECIP_ETA      = 0.0

# PPR + sweep + FM
RW_ALPHA  = 0.85
FM_MAX_MOVES = 50
FM_SIZE_DRIFT = 0.15

# β continuation for later waves
BETA_RHO  = 2.0
BETA_REFINE_STEPS = 1

# Sweep admissibility / tie-breaks
SWEEP_LEAK_CLAMP = 0.33
SWEEP_TIE_EPS    = 0.02

# Fetch/time budgets
WAVE_MAX_NEW_FETCH = 1200
WAVE_WALLCLOCK_SEC = 240

# Randomness
RAND_SEED = 13
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

# Output
DUMP_DIR = "./_dump"
os.makedirs(DUMP_DIR, exist_ok=True)

# =============================================================================
# Cache
# =============================================================================

def _load_jsonl(path):
    d = {}
    if not os.path.exists(path): return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line); pmid=int(obj["pmid"]); d[pmid]=obj
            except Exception:
                pass
    return d

def _append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

ICITE_CACHE: Dict[int, dict] = _load_jsonl(ICITE_CACHE_FILE)

def _merge_icite_record(dst: dict, src: dict) -> dict:
    for k in ("citedByPmids","citedPmids"):
        if k in src and src[k]: dst[k]=[int(x) for x in src[k]]
    if src.get("title"): dst["title"]=src["title"]
    if src.get("year") is not None: dst["year"]=src["year"]
    return dst

def ensure_icite_loaded(pmids: Set[int]):
    if not pmids: return
    todo=[str(int(p)) for p in pmids]
    max_ids_per_call=200; max_url_len=5000
    def _batches(items):
        i=0; n=len(items)
        while i<n:
            batch=[]; url_len=len(ICITE_BASE)+len("?pmids=")+20
            while i<n and len(batch)<max_ids_per_call:
                cand=items[i]; extra=len(cand)+(1 if batch else 0)
                if (url_len+extra)>max_url_len: break
                batch.append(cand); url_len+=extra; i+=1
            if not batch: batch=[items[i]]; i+=1
            yield batch
    fetched=0; t0=time.monotonic()
    for batch in _batches(todo):
        if fetched>=WAVE_MAX_NEW_FETCH: break
        if time.monotonic()-t0>WAVE_WALLCLOCK_SEC: break
        params={"pmids":",".join(batch),"legacy":"false"}
        backoff=0.6
        for _ in range(6):
            r=requests.get(ICITE_BASE, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code==200: break
            if r.status_code in (429,502,503,504):
                time.sleep(backoff); backoff*=1.7; continue
            r.raise_for_status()
        data=r.json().get("data", r.json())
        recs=[]
        for rec in data:
            pmid=int(rec.get("pmid") or rec.get("_id"))
            obj=ICITE_CACHE.get(pmid, {"pmid":pmid,"citedByPmids":[],"citedPmids":[]})
            cited_by=rec.get("citedByPmids",[]) or rec.get("cited_by") or []
            refs    =rec.get("citedPmids",[])   or rec.get("references") or []
            merged=_merge_icite_record(obj,{
                "pmid":pmid,
                "citedByPmids":[int(x) for x in cited_by],
                "citedPmids":[int(x) for x in refs],
                "title":rec.get("title"),
                "year": rec.get("year"),
            })
            ICITE_CACHE[pmid]=merged; recs.append(merged)
        if recs: _append_jsonl(ICITE_CACHE_FILE, recs)
        fetched+=len(batch); time.sleep(ICITE_SLEEP)

def icite_neighbors(pmid:int)->Set[int]:
    rec=ICITE_CACHE.get(int(pmid)); 
    if not rec: return set()
    return set(rec.get("citedByPmids",[])) | set(rec.get("citedPmids",[]))

def icite_deg_tot(pmid:int)->int:
    rec=ICITE_CACHE.get(int(pmid),{})
    return len(rec.get("citedByPmids",[]))+len(rec.get("citedPmids",[]))

def icite_title(pmid:int)->str:
    return ICITE_CACHE.get(int(pmid),{}).get("title") or "(title unavailable)"

def icite_year(pmid:int):
    return ICITE_CACHE.get(int(pmid),{}).get("year")

def fmt_paper_line(pmid:int, extra_right:str="")->str:
    y=icite_year(pmid); ytxt=f"{y}" if y is not None else "----"
    t=(icite_title(pmid) or "").replace("\n"," ").strip()
    if len(t)>110: t=t[:107]+"..."
    return f"{pmid} ({ytxt}) | {t}{extra_right}"

# =============================================================================
# H-graph
# =============================================================================

@dataclass
class HGraph:
    pmids: List[int]
    index: Dict[int,int]
    edges: List[Tuple[int,int,float]]
    degH_wsum: np.ndarray
    outU: np.ndarray
    G: nx.Graph
    has_dir: Dict[Tuple[int,int], bool]

def _build_edges_and_degs(H_nodes: List[int],
                          reciprocation_eta: float,
                          triangle_alpha: float):
    ensure_icite_loaded(set(H_nodes))
    Hset=set(H_nodes); has_dir={}; undirected=set()
    for u in H_nodes:
        rec=ICITE_CACHE.get(u,{})
        for v in rec.get("citedPmids",[])+rec.get("citedByPmids",[]):
            if v==u: continue
            has_dir[(u,v)]=True
            if v in Hset:
                i,j=(u,v) if u<v else (v,u)
                if i!=j: undirected.add((i,j))
    idx={p:i for i,p in enumerate(H_nodes)}
    edges=[]; 
    for (a,b) in undirected:
        ia,ib=idx[a],idx[b]; w=1.0
        recip=(has_dir.get((a,b),False) and has_dir.get((b,a),False))
        if reciprocation_eta>0 and recip: w=1.0+reciprocation_eta
        edges.append((ia,ib,float(w)))
    G=nx.Graph(); G.add_nodes_from(range(len(H_nodes)))
    for i,j,w in edges:
        if i!=j: G.add_edge(i,j,weight=w)
    if triangle_alpha>0.0 and G.number_of_edges()>0:
        adj={i:set(G.neighbors(i)) for i in G.nodes}; degs={i:len(adj[i]) for i in adj}
        edges2=[]
        for (i,j,w) in edges:
            tri=len(adj[i]&adj[j]); denom=max(1, min(degs[i],degs[j])-1)
            norm=(tri/denom) if denom>0 else 0.0
            edges2.append((i,j,w*(1+triangle_alpha*norm)))
        edges=edges2
        G=nx.Graph(); G.add_nodes_from(range(len(H_nodes)))
        for i,j,w in edges:
            if i!=j: G.add_edge(i,j,weight=float(w))
    deg_w=np.zeros(len(H_nodes),dtype=float)
    for (i,j,w) in edges: deg_w[i]+=w; deg_w[j]+=w
    adj_unw_ct=np.zeros(len(H_nodes),dtype=int)
    for (i,j,w) in edges: adj_unw_ct[i]+=1; adj_unw_ct[j]+=1
    outU=np.zeros(len(H_nodes),dtype=float)
    for i,pmid in enumerate(H_nodes):
        outU[i]=max(0, icite_deg_tot(pmid)-int(adj_unw_ct[i]))
    return edges, deg_w, outU, G, has_dir

def build_H_hops(seeds: List[int], hops:int=1, per_node_cap:int=INIT_PER_NODE_CAP,
                 reciprocation_eta:float=RECIP_ETA, triangle_alpha:float=TRIANGLE_ALPHA,
                 verbose:bool=True)->HGraph:
    seeds=[int(p) for p in seeds]; ensure_icite_loaded(set(seeds))
    visited=set(seeds); frontier=[set(seeds)]
    for hop in range(1, hops+1):
        curr=frontier[-1]; nxt=set()
        preload=set()
        for u in curr: preload |= icite_neighbors(u)
        ensure_icite_loaded(preload)
        for u in curr:
            neighs_all=list(icite_neighbors(u)); random.shuffle(neighs_all)
            for v in neighs_all[:per_node_cap]:
                if v!=u and v not in visited: nxt.add(v)
        visited |= nxt; frontier.append(nxt)
        if verbose: print(f"[HOP {hop}] grew by +{len(nxt)} (H now {len(visited)})")
    H_nodes=list(visited)
    edges,deg_w,outU,G,has_dir=_build_edges_and_degs(H_nodes, reciprocation_eta, triangle_alpha)
    H=HGraph(pmids=H_nodes, index={p:i for i,p in enumerate(H_nodes)},
             edges=edges, degH_wsum=deg_w, outU=outU, G=G, has_dir=has_dir)
    if verbose: print(f"[H] Built H: |C|={len(seeds)}, |H|={len(H.pmids)}, |E|={len(H.edges)}")
    return H

def build_H_from_nodes(node_pmids: List[int],
                       reciprocation_eta:float=RECIP_ETA,
                       triangle_alpha:float=TRIANGLE_ALPHA)->HGraph:
    H_nodes=list(dict.fromkeys(int(p) for p in node_pmids))
    edges,deg_w,outU,G,has_dir=_build_edges_and_degs(H_nodes, reciprocation_eta, triangle_alpha)
    return HGraph(pmids=H_nodes, index={p:i for i,p in enumerate(H_nodes)},
                  edges=edges, degH_wsum=deg_w, outU=outU, G=G, has_dir=has_dir)

# =============================================================================
# Linear algebra
# =============================================================================

def build_W(H:HGraph)->sp.csr_matrix:
    n=len(H.pmids); rows=[]; cols=[]; data=[]
    for (i,j,w) in H.edges:
        rows+=[i,j]; cols+=[j,i]; data+=[w,w]
    return sp.csr_matrix((data,(rows,cols)), shape=(n,n))

def ppr_leaky(H:HGraph, seeds:List[int], beta:float,
              alpha_rw:float=RW_ALPHA, iters:int=2000, tol:float=1e-9,
              x0:Optional[np.ndarray]=None)->Tuple[np.ndarray,np.ndarray]:
    n=len(H.pmids)
    W=build_W(H)
    vol=H.degH_wsum + beta*H.outU
    inv_vol=np.zeros(n,dtype=float); pos=(vol>0); inv_vol[pos]=1.0/vol[pos]
    P=W @ sp.diags(inv_vol)
    b=np.zeros(n,dtype=float)
    for p in seeds:
        if p in H.index: b[H.index[p]]=1.0
    x = np.zeros(n,dtype=float) if (x0 is None or len(x0)!=n) else np.clip(x0,0.0,None)
    if not np.any(x): x=b.copy()
    last_norm=np.linalg.norm(x,1)+1e-12
    for _ in range(iters):
        x_new=(1.0-alpha_rw)*b + alpha_rw*(P @ x)
        if np.linalg.norm(x_new-x,1) < tol*max(1.0,last_norm):
            x=x_new; break
        x=x_new; last_norm=np.linalg.norm(x,1)+1e-12
    x=np.clip(x,0.0,None)
    z=np.zeros_like(x); z[pos]=x[pos]*inv_vol[pos]
    return x,z

# =============================================================================
# Stats, sweep, FM
# =============================================================================

def phi_beta_stats(H:HGraph, A:Set[int], beta:float)->Dict[str,float]:
    if not A:
        return dict(phi=float("inf"),cut=0.0,leak=0.0,vol=0.0,wAA=0.0,leak_share=0.0,k=0)
    n=len(H.pmids); A_mask=np.zeros(n,dtype=bool); idxs=sorted(list(A)); A_mask[idxs]=True
    cut=0.0; wAA=0.0
    for (i,j,w) in H.edges:
        ai=A_mask[i]; aj=A_mask[j]
        if ai and aj: wAA+=w
        elif ai ^ aj: cut+=w
    leak=float(np.sum(H.outU[A_mask]))
    volA=float(np.sum(H.degH_wsum[A_mask] + beta*H.outU[A_mask]))
    volH=float(np.sum(H.degH_wsum + beta*H.outU))
    den=min(volA, volH-volA) if volH>0 else 0.0
    phi=(cut + beta*leak)/den if den>0 else float("inf")
    ls=(beta*leak)/max(1e-12, cut+beta*leak)
    return dict(phi=phi, cut=cut, leak=leak, vol=volA, wAA=wAA, leak_share=ls, k=len(A))

def build_W_once(H:HGraph): return build_W(H)  # alias if you want to cache

def sweep_best(H:HGraph, scores:np.ndarray, beta:float, seeds:Set[int],
               kmin:Optional[int], kmax:Optional[int],
               leak_clamp:Optional[float]=SWEEP_LEAK_CLAMP,
               tie_eps:float=SWEEP_TIE_EPS):
    n=len(H.pmids); order=np.argsort(-scores); inA=np.zeros(n,dtype=bool)
    W=build_W(H); w_to_A=np.zeros(n,dtype=float)
    vol_total=float(np.sum(H.degH_wsum + beta*H.outU))
    vol=0.0; cut=0.0; leak=0.0
    best_k=0; best_stats=None
    seed_idx={H.index[p] for p in seeds if p in H.index}
    candidates=[]
    for k,u in enumerate(order, start=1):
        rs,re=W.indptr[u],W.indptr[u+1]; nbrs=W.indices[rs:re]; wts=W.data[rs:re]
        dH_u=H.degH_wsum[u]; w_uA=w_to_A[u]
        cut += (dH_u - 2.0*w_uA)
        leak+= H.outU[u]
        vol += dH_u + beta*H.outU[u]
        inA[u]=True
        for v,w in zip(nbrs,wts):
            if inA[v]: w_to_A[v]+=w
        if k<n:
            if not seed_idx.issubset(set(order[:k])): continue
            if kmin is not None and k<kmin: continue
            if kmax is not None and k>kmax: continue
            num=cut+beta*leak; den=min(vol, vol_total-vol)
            if den<=0: continue
            phi=num/den; leak_share=(beta*leak)/max(1e-12,num)
            if (leak_clamp is not None) and (leak_share>leak_clamp): continue
            st=dict(phi=phi,cut=cut,leak=leak,vol=vol,leak_share=leak_share,k=k)
            candidates.append((k,st))
    if not candidates:
        A=set(seed_idx); return A, phi_beta_stats(H,A,beta), 0
    min_phi=min(st["phi"] for _,st in candidates)
    near=[(k,st) for (k,st) in candidates if st["phi"] <= (1.0+tie_eps)*min_phi]
    near.sort(key=lambda kv: (kv[1]["phi"], kv[1]["leak_share"], kv[1]["k"]))
    best_k, best_stats = near[0]
    A_idx=set(order[:best_k]) | seed_idx
    return A_idx, phi_beta_stats(H,A_idx,beta), best_k

def dual_sweep(H:HGraph, x:np.ndarray, z:np.ndarray, beta:float, seeds:Set[int],
               kmin:Optional[int], kmax:Optional[int]):
    A1,st1,k1=sweep_best(H,x,beta,seeds,kmin,kmax)
    A2,st2,k2=sweep_best(H,z,beta,seeds,kmin,kmax)
    if st2["phi"]<st1["phi"]: return A2,st2,"z",k2
    else: return A1,st1,"x",k1

def fm_local_improve(H:HGraph, seeds:Set[int], A:Set[int], beta:float,
                     kmin:Optional[int], kmax:Optional[int],
                     max_moves:int=FM_MAX_MOVES):
    A=set(A); seed_idx={H.index[p] for p in seeds if p in H.index}
    n=len(H.pmids); W=build_W(H); inA=np.zeros(n,dtype=bool); inA[list(A)]=True
    def cur_stats(): return phi_beta_stats(H, set(np.where(inA)[0]), beta)
    w_to_A=np.zeros(n,dtype=float)
    for u in range(n):
        rs,re=W.indptr[u],W.indptr[u+1]; nbrs=W.indices[rs:re]; wts=W.data[rs:re]
        w_to_A[u]=np.sum(wts[inA[nbrs]])
    st=cur_stats(); K0=int(st["k"])
    kmin_eff=max(len(seed_idx), int((1.0-FM_SIZE_DRIFT)*K0))
    kmax_eff=int((1.0+FM_SIZE_DRIFT)*K0)
    if kmin is not None: kmin_eff=max(kmin_eff,kmin)
    if kmax is not None: kmax_eff=min(kmax_eff,kmax)
    moves=0
    while moves<max_moves:
        best_gain=0.0; best=None
        num_now=st["cut"]+beta*st["leak"]
        den_now=min(st["vol"], float(np.sum(H.degH_wsum + beta*H.outU)) - st["vol"])
        if den_now<=0: break
        phi_now=num_now/den_now; size_now=int(st["k"])
        if size_now<kmax_eff:
            for v in np.where(~inA)[0]:
                dH=H.degH_wsum[v]
                delta_cut = dH - 2.0*w_to_A[v]
                delta_leak= H.outU[v]
                delta_vol = dH + beta*H.outU[v]
                cut_new=st["cut"]+delta_cut
                leak_new=st["leak"]+delta_leak
                vol_new=st["vol"]+delta_vol
                den_new=min(vol_new, float(np.sum(H.degH_wsum + beta*H.outU)) - vol_new)
                if den_new<=0: continue
                phi_new=(cut_new + beta*leak_new)/den_new
                gain=phi_now - phi_new
                if gain>best_gain+1e-12: best_gain=gain; best=("add",v,cut_new,leak_new,vol_new,phi_new)
        if size_now>kmin_eff:
            for u in np.where(inA)[0]:
                if u in seed_idx: continue
                dH=H.degH_wsum[u]
                delta_cut = -(dH - 2.0*w_to_A[u])
                delta_leak= -H.outU[u]
                delta_vol = -(dH + beta*H.outU[u])
                cut_new=st["cut"]+delta_cut
                leak_new=st["leak"]+delta_leak
                vol_new=st["vol"]+delta_vol
                den_new=min(vol_new, float(np.sum(H.degH_wsum + beta*H.outU)) - vol_new)
                if den_new<=0: continue
                phi_new=(cut_new + beta*leak_new)/den_new
                gain=phi_now - phi_new
                if gain>best_gain+1e-12: best_gain=gain; best=("rem",u,cut_new,leak_new,vol_new,phi_new)
        if best is None: break
        kind,node,cut_new,leak_new,vol_new,phi_new=best
        if kind=="add":
            inA[node]=True
            rs,re=W.indptr[node],W.indptr[node+1]; nbrs=W.indices[rs:re]; wts=W.data[rs:re]
            for v,w in zip(nbrs,wts):
                if inA[v]: w_to_A[v]+=w
        else:
            inA[node]=False
            rs,re=W.indptr[node],W.indptr[node+1]; nbrs=W.indices[rs:re]; wts=W.data[rs:re]
            for v,w in zip(nbrs,wts):
                if inA[v]: w_to_A[v]-=w
        moves+=1
        st=dict(st); st["cut"],st["leak"],st["vol"],st["phi"]=cut_new,leak_new,vol_new,phi_new
        st["leak_share"]=(beta*st["leak"])/max(1e-12, st["cut"]+beta*st["leak"]); st["k"]=int(np.sum(inA))
    A_new=set(np.where(inA)[0]) | seed_idx
    return A_new, phi_beta_stats(H, A_new, beta)

# =============================================================================
# β tuning
# =============================================================================

def tune_beta_adaptive(H:HGraph, seeds:List[int],
                       kmin:Optional[int], kmax:Optional[int],
                       init_grid:Optional[np.ndarray]=None,
                       refine_steps:int=2,
                       local_span_rho:float=3.0,
                       points_per_grid:int=11):
    if init_grid is None:
        init_grid=np.logspace(-4, 1, 13)  # wide coarse
    best_beta=None; best_stats=None; best_x=None; best_z=None; best_order="x"; best_k=0
    center=None; x0=None
    grids=[np.array(init_grid,dtype=float)]
    for step in range(refine_steps):
        if center is not None:
            span = local_span_rho ** (refine_steps - step)
            mul  = np.logspace(-math.log10(span), math.log10(span), points_per_grid)
            grids.append(center * mul)
        grid=grids[-1]
        for beta in grid:
            beta=float(beta)
            x,z=ppr_leaky(H,seeds,beta,x0=x0)
            A,st,which,k=dual_sweep(H,x,z,beta,set(seeds),kmin,kmax)
            A,st=fm_local_improve(H,set(seeds),A,beta,kmin,kmax)
            if (best_stats is None) or (st["phi"]<best_stats["phi"]):
                best_stats=dict(st); best_beta=beta
                best_x, best_z, best_order, best_k = x, z, which, k
                center=best_beta; x0=x
    return best_beta, best_stats, best_x, best_z, best_order, best_k

def tune_beta_continue(H:HGraph, seeds:List[int], beta_prev:float,
                       kmin:Optional[int], kmax:Optional[int],
                       x0:Optional[np.ndarray]=None):
    center=math.log(beta_prev if beta_prev>0 else 1e-3)
    best_beta=None; best_stats=None; best_x=None; best_z=None; best_order="x"; best_k=0
    for _ in range(BETA_REFINE_STEPS):
        grid=np.exp(center + np.log(np.array([1/BETA_RHO, 1/math.sqrt(BETA_RHO), 1.0, math.sqrt(BETA_RHO), BETA_RHO])))
        for beta in grid:
            x,z=ppr_leaky(H,seeds,float(beta),x0=x0)
            A,st,which,k=dual_sweep(H,x,z,float(beta),set(seeds),kmin,kmax)
            A,st=fm_local_improve(H,set(seeds),A,float(beta),kmin,kmax)
            if (best_stats is None) or (st["phi"]<best_stats["phi"]):
                best_stats=dict(st); best_beta=float(beta)
                best_x, best_z, best_order, best_k = x, z, which, k
                center=math.log(best_beta)
    return best_beta, best_stats, best_x, best_z, best_order, best_k

# =============================================================================
# Utils
# =============================================================================

def top_deg_nodes(H:HGraph, A:Set[int], k:int=5):
    rows=[(H.pmids[i], float(H.degH_wsum[i])) for i in sorted(list(A))]
    rows.sort(key=lambda t:-t[1]); return rows[:k]

def boundary_offenders(H:HGraph, A:Set[int], beta:float, k:int=5):
    A_mask=np.zeros(len(H.pmids),dtype=bool); idxs=sorted(list(A)); A_mask[idxs]=True
    cut_i=np.zeros(len(H.pmids),dtype=float)
    for (i,j,w) in H.edges:
        if A_mask[i]^A_mask[j]:
            if A_mask[i]: cut_i[i]+=w
            else: cut_i[j]+=w
    rows=[]
    for i in idxs:
        rows.append((H.pmids[i], float(cut_i[i]), float(beta*H.outU[i])))
    rows.sort(key=lambda t:-(t[1]+t[2])); return rows[:k]

def print_run(label:str, H:HGraph, A:Set[int], st:Dict, beta:float):
    print(f"[{label}] |H|={len(H.pmids)} |E|={len(H.edges)} |A|={st['k']} | φ={st['phi']:.5f} | cut={st['cut']:.2f} | leak={st['leak']:.2f} | wAA={st['wAA']:.2f} | leak%={100*st['leak_share']:.1f}% | β={beta:.6g}")
    print("  center (top by deg_H_wsum in A):")
    for pmid,degw in top_deg_nodes(H,A,k=5):
        print("    "+fmt_paper_line(pmid, extra_right=f" | deg_w={degw:.2f}"))
    print("  boundary offenders (in A): pmid (year) | title | cut_i | beta*outU_i | approx_num_i")
    for pmid,ci,bl in boundary_offenders(H,A,beta,k=5):
        print("    "+fmt_paper_line(pmid, extra_right=f" | {ci:.2f} | {bl:.2f} | {(ci+bl):.2f}"))

def dump_A_tsv(path:str, H:HGraph, A:Set[int]):
    rows=[]
    for i in sorted(list(A)):
        p=H.pmids[i]; rows.append((p, icite_year(p), icite_title(p)))
    with open(path,"w",encoding="utf-8") as f:
        f.write("pmid\tyear\ttitle\n")
        for p,y,t in rows:
            ytxt="" if y is None else str(y)
            ttxt=(t or "").replace("\t"," ").replace("\n"," ").strip()
            f.write(f"{p}\t{ytxt}\t{ttxt}\n")
    print(f"[WRITE] {path}  ({len(rows)} rows)")

def jaccard_pmids(H1:HGraph, A1:Set[int], H2:HGraph, A2:Set[int])->float:
    P1={H1.pmids[i] for i in A1}; P2={H2.pmids[i] for i in A2}
    U=len(P1|P2); return (len(P1&P2)/U) if U>0 else 1.0

def stable_hash_u32(pmid:int)->int:
    h=hashlib.blake2b(str(pmid).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0xffffffff

# =============================================================================
# Configs
# =============================================================================

@dataclass
class OuterConfig:
    max_waves:int = 6
    patience:int  = 1

    # Acceptance policy: "elastic_phi" | "utility" | "probation"
    accept_policy:str = "elastic_phi"
    eps_phi_max:float = 0.08
    lambda_phi_per_log_g:float = 0.06
    leak_cap:float = 0.20
    snr_rel_inflate_cap:float = 0.10
    churn_cap:float = 0.60

    util_alpha_log_g:float = 0.12
    util_beta_leak:float   = 0.20
    util_gamma_snr:float   = 0.10
    probation_min_growth:float = 1.40

    # Expansion budgets (topic-agnostic)
    max_accept_after_filter: Optional[int] = 300
    H_external_budget: Optional[int] = 3000  # Σ |N(v)\H|
    per_node_ext_frac_cap: Optional[float] = 0.85
    deterministic_reservoir: bool = True

    # Hub controls (soft and neutral)
    hub_deg_percentile: float = 95.0
    hub_deg_multiplier: float = 3.0
    hub_hard_cap: int = 1200
    min_hub_degree_soft: int = 450
    quarantine_hubs: bool = True

    # Relevance gate (0.0 disables)
    min_relevance_frac: float = 0.05

    # Expansion-from A: quarantine mode — "external" | "total"
    quarantine_mode: str = "external"
    # For "external" mode (skip expanding from A-nodes that would open too many NEW nodes)
    A_expand_pernode_ext_cap: Optional[int] = None      # e.g., 800 (None = off)
    A_expand_pernode_ext_frac_cap: Optional[float] = None  # e.g., 0.9 (None = off)
    
    # >>> add this <<<
    label: str = "outer|cile"

# =============================================================================
# Acceptance logic
# =============================================================================

def accept_wave(prev:Dict, cur:Dict, cfg:OuterConfig)->bool:
    phi0,phi1 = prev["phi"], cur["phi"]
    k0,k1 = prev["k"], cur["k"]
    gA = (k1/max(1,k0)) if k0>0 else 1.0
    leak0,leak1 = prev["leak_share"], cur["leak_share"]
    r0 = prev["cut"]/max(1e-9, prev["wAA"])
    r1 = cur["cut"]/ max(1e-9, cur["wAA"])
    churn = 1.0 - cur["jaccard"]

    if cfg.accept_policy=="elastic_phi":
        dphi_allow = min(cfg.eps_phi_max, cfg.lambda_phi_per_log_g*(math.log(gA) if gA>1 else 0.0))
        ok_phi  = (phi1 <= phi0*(1.0 + dphi_allow))
        ok_leak = (leak1 <= min(leak0, cfg.leak_cap))
        ok_snr  = (r1 <= r0*(1.0 + cfg.snr_rel_inflate_cap))
        ok_churn= (churn <= cfg.churn_cap)
        return ok_phi and ok_leak and ok_snr and ok_churn

    if cfg.accept_policy=="utility":
        U1 = (-phi1
              + cfg.util_alpha_log_g * (math.log(gA) if gA>1 else 0.0)
              - cfg.util_beta_leak * leak1
              - cfg.util_gamma_snr * max(0.0, (r1/r0)-1.0))
        U0 = -phi0
        ok_churn = (churn <= cfg.churn_cap)
        return (U1 >= U0) and ok_churn

    if cfg.accept_policy=="probation":
        if phi1 <= phi0: return churn <= cfg.churn_cap
        big_growth = (gA >= cfg.probation_min_growth)
        better_leak = (leak1 <= min(leak0, cfg.leak_cap))
        stable_snr  = (r1 <= r0*(1.0 + cfg.snr_rel_inflate_cap))
        return big_growth and better_leak and stable_snr and (churn <= cfg.churn_cap)

    # default strict
    return (phi1 + 1e-9 < phi0) and (churn <= 0.40)

# =============================================================================
# Adaptive expansion (topic-agnostic, compute-bounded)
# =============================================================================

def _deg_distribution(pmids:Set[int])->List[int]:
    ensure_icite_loaded(pmids); return [icite_deg_tot(p) for p in pmids]

def _adaptive_hub_threshold(A_pmids:Set[int], cfg:OuterConfig)->int:
    degs=_deg_distribution(A_pmids) if A_pmids else [0]
    base=np.percentile(degs, cfg.hub_deg_percentile) if len(degs)>0 else 0.0
    thr=int(max(0,base) * max(1.0, cfg.hub_deg_multiplier))
    if cfg.hub_hard_cap and cfg.hub_hard_cap>0:
        thr=min(thr if thr>0 else cfg.hub_hard_cap, cfg.hub_hard_cap)
    thr=max(thr, cfg.min_hub_degree_soft)  # soft floor
    return max(1, thr)

def build_next_H_adaptive(H_prev:HGraph, A_prev:Set[int], cfg:OuterConfig)->Tuple[List[int], Dict[str,int]]:
    t0=time.monotonic()
    A_pmids={H_prev.pmids[i] for i in A_prev}; ensure_icite_loaded(A_pmids)
    H_set=set(H_prev.pmids)

    # Choose expansion sources from A
    expand_from=set(A_pmids)
    quarantined_A=0
    if cfg.quarantine_hubs:
        if cfg.quarantine_mode=="total":
            hub_thr=_adaptive_hub_threshold(A_pmids, cfg)
            keep=set()
            for p in expand_from:
                if icite_deg_tot(p)>hub_thr: quarantined_A+=1
                else: keep.add(p)
            expand_from=keep
        else:  # "external" (preferred): skip A nodes that would open too many NEW nodes
            keep=set()
            for p in expand_from:
                neighs=icite_neighbors(p)
                ext=len(neighs - H_set)
                frac= ext/max(1,len(neighs))
                too_many = (cfg.A_expand_pernode_ext_cap is not None and ext>cfg.A_expand_pernode_ext_cap)
                too_frac = (cfg.A_expand_pernode_ext_frac_cap is not None and frac>cfg.A_expand_pernode_ext_frac_cap)
                if too_many or too_frac:
                    quarantined_A+=1
                else:
                    keep.add(p)
            expand_from=keep
        hub_thr = _adaptive_hub_threshold(A_pmids, cfg) if cfg.quarantine_mode=="total" else -1
    else:
        hub_thr=-1

    # Neighborhood union from allowed A-sources
    preload=set()
    for u in expand_from: preload |= icite_neighbors(u)
    ensure_icite_loaded(preload)
    neighborhood=preload
    candidates=list(neighborhood - H_set)

    # Ensure candidate metadata present
    ensure_icite_loaded(set(candidates))

    # Optional minimal relevance gate (0.0 disables)
    accepted=[]
    dropped_rel=0
    for v in candidates:
        if cfg.min_relevance_frac and cfg.min_relevance_frac>0.0:
            neighs=icite_neighbors(v); dt=len(neighs)
            if dt<=0: continue
            links_to_A=len(neighs & A_pmids)
            rel=links_to_A/max(1,dt)
            if rel < cfg.min_relevance_frac:
                dropped_rel+=1
                continue
        accepted.append(v)

    # Compute external degree info (how many NEW nodes each candidate opens)
    acc_info=[]
    for v in accepted:
        neighs=icite_neighbors(v)
        ext=len(neighs - H_set)
        deg_tot=len(neighs)
        ext_frac=ext/max(1,deg_tot)
        acc_info.append((v, ext, ext_frac))

    # Per-node external frac cap
    if cfg.per_node_ext_frac_cap is not None:
        acc_info=[(v,ext,ef) for (v,ext,ef) in acc_info if ef <= cfg.per_node_ext_frac_cap]

    # Sum-of-external budget (greedy, neutral; small ext first; deterministic tiebreak)
    sum_ext_before = sum(ext for (_,ext,_) in acc_info)
    if cfg.H_external_budget is not None:
        acc_info.sort(key=lambda t:(t[1], stable_hash_u32(t[0])))
        picked=[]; sum_ext=0
        for (v,ext,ef) in acc_info:
            if sum_ext + ext > cfg.H_external_budget: continue
            picked.append((v,ext,ef)); sum_ext += ext
        acc_info=picked
    else:
        sum_ext=sum_ext_before

    # Absolute reservoir cap (deterministic if requested)
    if cfg.max_accept_after_filter is not None and len(acc_info)>cfg.max_accept_after_filter:
        if cfg.deterministic_reservoir:
            acc_info.sort(key=lambda t: stable_hash_u32(t[0]))
            acc_info=acc_info[:cfg.max_accept_after_filter]
        else:
            random.seed(RAND_SEED); acc_info=random.sample(acc_info, cfg.max_accept_after_filter)

    accepted=[v for (v,_,_) in acc_info]
    H_next_nodes=list(H_set | set(accepted))

    meta=dict(
        A=len(A_pmids),
        H_prev=len(H_prev.pmids),
        hub_thr=hub_thr,
        quarantineA=quarantined_A,
        neighborhood=len(neighborhood),
        candidates=len(candidates),
        accepted=len(accepted),
        dropped_relevance=dropped_rel,
        sum_ext_before=sum_ext_before,
        sum_ext_after=sum(ext for (_,ext,_) in acc_info),
        H_next=len(H_next_nodes),
        elapsed_ms=int(1000*(time.monotonic()-t0))
    )
    return H_next_nodes, meta

# =============================================================================
# Wave solve + acceptance
# =============================================================================

def solve_on_H(H:HGraph, seeds:List[int],
               beta_prev:Optional[float],
               A_prev_size:Optional[int]):
    n=len(H.pmids)
    if A_prev_size is None:
        kmin=max(len(seeds),1); kmax=int(0.45*n)
    else:
        A_REL_GROWTH=0.30; A_ABS_CAP=120
        kmin=max(len(seeds), int((1.0-A_REL_GROWTH)*A_prev_size))
        kmax=min(int(0.45*n), int((1.0+A_REL_GROWTH)*A_prev_size + A_ABS_CAP))
    if beta_prev is None:
        beta_star, st0, x0, z0, which0, k0 = tune_beta_adaptive(H,seeds,kmin,kmax,init_grid=None,refine_steps=2,local_span_rho=3.0,points_per_grid=11)
    else:
        beta_star, st0, x0, z0, which0, k0 = tune_beta_continue(H,seeds,beta_prev,kmin,kmax,x0=None)
    x,z=ppr_leaky(H,seeds,beta_star,x0=x0)
    A,st,which,k=dual_sweep(H,x,z,beta_star,set(seeds),kmin,kmax)
    A,st=fm_local_improve(H,set(seeds),A,beta_star,kmin,kmax)
    info=dict(kmin=kmin,kmax=kmax,beta_star=beta_star,order=which)
    return A,st,beta_star,x,z,which,info

# =============================================================================
# Outer loop
# =============================================================================

def outer_loop_cile(S0:List[int], cfg:OuterConfig):
    print("Seeds:", S0)
    ensure_icite_loaded(set(S0))
    H = build_H_hops(S0, hops=INIT_HOPS, per_node_cap=INIT_PER_NODE_CAP, verbose=True)
    A, st, beta, x, z, which, info = solve_on_H(H, S0, beta_prev=None, A_prev_size=None)
    print_run(f"BASE(PPR+dual+FM|min-φ|adaptive-β)", H, A, st, beta)
    dump_A_tsv(os.path.join(DUMP_DIR, "A_star.tsv"), H, A)

    best = dict(H=H, A=A, st=st, beta=beta)
    last = best

    for wave in range(1, cfg.max_waves+1):
        H_next_nodes, meta = build_next_H_adaptive(last["H"], last["A"], cfg)
        H_next = build_H_from_nodes(H_next_nodes, reciprocation_eta=RECIP_ETA, triangle_alpha=TRIANGLE_ALPHA)

        A_new, st_new, beta_new, x_new, z_new, which_new, info_new = solve_on_H(
            H_next, S0, beta_prev=last["beta"], A_prev_size=len(last["A"])
        )

        # Assemble prev/cur metrics for acceptance
        prev_metrics = dict(phi=last["st"]["phi"], k=last["st"]["k"],
                            leak_share=last["st"]["leak_share"], cut=last["st"]["cut"],
                            wAA=last["st"]["wAA"])
        cur_metrics  = dict(phi=st_new["phi"], k=st_new["k"],
                            leak_share=st_new["leak_share"], cut=st_new["cut"],
                            wAA=st_new["wAA"])
        cur_metrics["jaccard"] = jaccard_pmids(last["H"], last["A"], H_next, A_new)

        accept = accept_wave(prev_metrics, cur_metrics, cfg)

        tag = getattr(cfg, "label", "outer|cile")
        print_run(f"{tag}|wave{wave} (accept={accept})", H_next, A_new, st_new, beta_new)
        print(f"  β drift: {last['beta']:.6g} → {beta_new:.6g} | φ: {last['st']['phi']:.5f} → {st_new['phi']:.5f} | gA={cur_metrics['k']/max(1,prev_metrics['k']):.3f} | J={cur_metrics['jaccard']:.3f}")
        print(f"  expander meta: A={meta['A']} | H {meta['H_prev']}→{meta['H_next']} | quarantineA={meta['quarantineA']} | cand={meta['candidates']} | acc={meta['accepted']} | drop_rel={meta['dropped_relevance']} | Σext(before→after)={meta['sum_ext_before']}→{meta['sum_ext_after']} | {meta['elapsed_ms']} ms")

        if accept:
            last = dict(H=H_next, A=A_new, st=st_new, beta=beta_new)
            # Track best φ for optional probation/rollback policies if you enable them
            if st_new["phi"] < best["st"]["phi"]:
                best = last
        else:
            # Stop and return the best-so-far (greedy but safe)
            dump_A_tsv(os.path.join(DUMP_DIR, "A_final.tsv"), best["H"], best["A"])
            print(f"[SUMMARY] {len(best['A'])} nodes | φ={best['st']['phi']:.5f} | β={best['beta']:.6g} | waves={wave-1} | accepted=False")
            return best["H"], best["A"], dict(phi=best["st"]["phi"], beta=best["beta"], waves=wave-1, accepted=False)

    dump_A_tsv(os.path.join(DUMP_DIR, "A_final.tsv"), last["H"], last["A"])
    print(f"[SUMMARY] {len(last['A'])} nodes | φ={last['st']['phi']:.5f} | β={last['beta']:.6g} | waves={cfg.max_waves} | accepted=True")
    return last["H"], last["A"], dict(phi=last["st"]["phi"], beta=last["beta"], waves=cfg.max_waves, accepted=True)

# =============================================================================
# Main
# =============================================================================

def main():
    Hf, Af, meta = outer_loop_cile(SEED_PMIDS, OuterConfig(
        # feel free to tweak these on the fly:
        accept_policy="elastic_phi",
        max_accept_after_filter=300,
        H_external_budget=3000,
        per_node_ext_frac_cap=0.85,
        deterministic_reservoir=True,
        min_relevance_frac=0.05,  # set 0.0 to disable
        quarantine_hubs=True,
        quarantine_mode="external",   # "total" or "external"
        A_expand_pernode_ext_cap=None,
        A_expand_pernode_ext_frac_cap=None,
    ))
    # Final dump already printed

if __name__ == "__main__":
    main()
