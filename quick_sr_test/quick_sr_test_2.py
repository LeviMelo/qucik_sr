#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systematic Review Triage Pipeline (Deterministic + Local LLM)
Single-file implementation with 12 modular steps (formerly cells), plus
integration of the provided CILE and full-text fetcher components.

Key properties:
- PubMed E-utilities for esearch/efetch (counts + metadata)
- Boolean query builder with strict syntax: quoted phrases + YYYY:YYYY[dp] only
- Culprit analysis (ablations/rescues), mandatory C/O enforcement, optional tighten variants
- Deterministic prefilter (year, pubtype blocklist, designs allowlist) with fail-fast
- Universe artifacts: universe_raw.jsonl (all) and universe.jsonl (kept)
- Ranking: TF-IDF (required) + LM Studio embeddings (batched) + MeSH-Jaccard → RRF(k) with recency tiebreak
- LLM TIAB screening (LM Studio) with sliding-window stop only when N > cap
- CILE stage-2 expansion using the provided CILE algorithm (executed as-is)
- MeSH mining/curation via LM Studio; iterative augmentation after stage-1 includes
- Merge + master artifacts (rank evidence carried through)
- Full-text handoff CSV (PMID, Year, FirstAuthor, Title, DOI) → provided fetcher (executed as-is)
  - Scoped override disables Excel path in fetcher; CSV-only in this pipeline
- Full-text extraction: pdfminer, OCR fallback (pdf2image + pytesseract), method recorded
- Final LLM full-text screen (include/exclude only)
- PRISMA prefilter_summary.csv and prefilter_detail.csv
- Logs: query_manager.log (idempotent), screening.log, fulltext.log

Python 3.10+ recommended.
"""

from __future__ import annotations
import os, sys, re, json, time, math, argparse, logging, random, csv, shutil, datetime, io
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime, timezone

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import os, sys
# Ensure local modules are importable if running from another cwd:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- LOCAL MODULES (CILE + OA fetcher) ----
import os as _os, sys as _sys
_sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))  # ensure local imports work

import cile  # external CILE module (cile.py)
from oa_fetcher import attempt_oa_downloads  # external OA fetcher (oa_fetcher.py)

# sanity checks (helps catch name typos fast)
assert hasattr(cile, "outer_loop_cile") and hasattr(cile, "OuterConfig"), "cile.py missing outer_loop_cile/OuterConfig"
assert callable(attempt_oa_downloads), "oa_fetcher.attempt_oa_downloads not found"


# ----------------------------
# GLOBALS / CONFIG
# ----------------------------
OUTDIR = "triage_out"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)

HTTP_TIMEOUT = 45
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = "triage_pipeline"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "triage@example.com")  # env override supported
NCBI_API_KEY = os.environ.get("NCBI_API_KEY") or None

LM_TIMEOUT = 60
LM_RETRY = 2

# Logging
root_log = logging.getLogger("triage")
root_log.setLevel(logging.INFO)
if not root_log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_log.addHandler(h)

def write_csv(path: str, rows: List[Dict[str,object]], fieldnames: Optional[List[str]]=None):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _dump_jsonl(path: str, objs: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

def _append_lines(path: str, lines: List[str]):
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")

# ----------------------------
# PROTOCOL
# ----------------------------
@dataclass
class Protocol:
    narrative_question: str
    year_min: int
    year_max: int
    designs_allowlist: List[str]
    pubtype_blocklist: List[str]
    mandatory_blocks: List[str]
    P_terms: List[str]
    I_terms: List[str]
    C_terms: List[str]
    O_terms: List[str]
    key_pmids: List[int]
    query_targets: Dict[str,int]
    screening: Dict[str,object]
    llm: Dict[str,str]

    @staticmethod
    def load(path: str) -> "Protocol":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # enforce default cap semantics: threshold 1000 if non-positive/missing
        scr = d.get("screening", {})
        thr = scr.get("llm_screen_cap_threshold", 1000)
        if not isinstance(thr, int) or thr <= 0:
            scr["llm_screen_cap_threshold"] = 1000
            d["screening"] = scr
        return Protocol(**d)

# ----------------------------
# PUBMED E-UTILITIES
# ----------------------------
def _base_params(extra: Optional[dict]=None) -> dict:
    p = {"tool": NCBI_TOOL, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if extra:
        p.update(extra)
    return p

def esearch_count(query: str) -> int:
    url = f"{EUTILS_BASE}/esearch.fcgi"
    p = _base_params({"db":"pubmed","term":query,"retmode":"json","rettype":"count"})
    for _ in range(4):
        try:
            r = requests.get(url, params=p, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return int(r.json()["esearchresult"]["count"])
        except Exception:
            time.sleep(0.5)
    return 0

def esearch_fetch_pmids(query: str, retmax: int=10000) -> List[int]:
    url = f"{EUTILS_BASE}/esearch.fcgi"
    p = _base_params({"db":"pubmed","term":query,"retmode":"json","retmax": str(retmax)})
    tries = 4
    while tries>0:
        try:
            r = requests.get(url, params=p, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            ids = r.json()["esearchresult"].get("idlist", [])
            return [int(x) for x in ids]
        except Exception:
            tries-=1; time.sleep(0.6)
    return []

def efetch_summaries(pmids: List[int]) -> List[dict]:
    # returns dicts: pmid,title,abstract,year,pubtypes,mesh,first_author,doi
    out = []
    if not pmids:
        return out
    url = f"{EUTILS_BASE}/efetch.fcgi"
    B = 200
    for i in range(0, len(pmids), B):
        batch = pmids[i:i+B]
        p = _base_params({"db":"pubmed","retmode":"xml","id":",".join(str(x) for x in batch)})
        try:
            r = requests.post(url, data=p, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            out.extend(_parse_pubmed_xml(r.text))
        except Exception as e:
            root_log.warning(f"efetch batch failed: {e}")
            continue
        time.sleep(0.34 if not NCBI_API_KEY else 0.12)
    return out

def _parse_pubmed_xml(xml_text: str) -> List[dict]:
    # lightweight XML parse via regex/strings; avoids heavyweight libs here
    # This is robust enough for key fields we need.
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    ns = {}
    items = []
    for art in root.findall(".//PubmedArticle", ns):
        pmid = art.findtext(".//MedlineCitation/PMID")
        pmid = int(pmid) if pmid and pmid.isdigit() else None
        title = (art.findtext(".//Article/ArticleTitle") or "").strip()
        abstract = " ".join([t.text or "" for t in art.findall(".//Abstract/AbstractText")]).strip()
        year = None
        y = art.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
        if y and y.isdigit():
            year = int(y)
        else:
            medlinedate = art.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
            m = re.search(r"(\d{4})", medlinedate)
            if m:
                year = int(m.group(1))
        pubtypes = [ (e.text or "").strip() for e in art.findall(".//PublicationTypeList/PublicationType") if (e.text or "").strip() ]
        mesh = [ (e.text or "").strip() for e in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if (e.text or "").strip() ]
        first_author = None
        fa = art.find(".//Article/AuthorList/Author[1]")
        if fa is not None:
            ln = (fa.findtext("LastName") or "").strip()
            ini = (fa.findtext("Initials") or "").strip()
            if ln and ini:
                first_author = f"{ln} {ini}"
            elif ln:
                first_author = ln
        doi = None
        doi_node = art.find(".//ArticleIdList/ArticleId[@IdType='doi']")
        if doi_node is None:
            doi_node = art.find(".//ELocationID[@EIdType='doi'][@ValidYN='Y']")
        if doi_node is not None and (doi_node.text or "").strip():
            doi = doi_node.text.strip()
        items.append({
            "pmid": pmid, "title": title, "abstract": abstract,
            "year": year, "pubtypes": pubtypes, "mesh": mesh,
            "first_author": first_author, "doi": doi
        })
    return items

# ----------------------------
# CELL 5 — Query Builder with culprit analysis + optional C/O tighten (fixed)
# ----------------------------
def _q(t: str) -> str:
    t = " ".join(t.split()).strip()
    return f"\"{t}\"" if (" " in t) else t

def _or_block(terms: List[str], cap: int) -> List[str]:
    uniq, seen = [], set()
    for t in terms:
        u = " ".join((t or "").split())
        if not u: continue
        key = u.lower()
        if key in seen: continue
        seen.add(key); uniq.append(u)
        if len(uniq) >= cap: break
    return uniq

def _build_query_from_blocks(P: List[str], I: List[str], C: List[str], O: List[str],
                             year_min: int, year_max: int,
                             mandatory_blocks: Set[str]) -> str:
    Y = f"{year_min}:{year_max}[dp]"
    blocks = []
    if P: blocks.append("(" + " OR ".join(_q(t) for t in P) + ")")
    if I: blocks.append("(" + " OR ".join(_q(t) for t in I) + ")")
    if "C" in mandatory_blocks:
        if not C: raise RuntimeError("mandatory_blocks requires C, but no C terms available after augmentation.")
        blocks.append("(" + " OR ".join(_q(t) for t in C) + ")")
    if "O" in mandatory_blocks:
        if not O: raise RuntimeError("mandatory_blocks requires O, but no O terms available after augmentation.")
        blocks.append("(" + " OR ".join(_q(t) for t in O) + ")")
    core = " AND ".join(blocks) if blocks else ""
    return f"({core}) AND {Y}" if core else f"{Y}"

def culprit_analysis(proto: Protocol,
                     mesh_curated: Dict[str,List[str]],
                     P_terms: List[str], I_terms: List[str],
                     C_terms: List[str], O_terms: List[str],
                     caps=(8,6,4),
                     inflator_frac: float = 0.35,
                     rescue_top_k: int = 3) -> Tuple[List[str], str]:
    """
    - Enforces mandatory C/O; fails fast if empty after augment.
    - Per-term ablation; inflator demotion; rescue if under target_min.
    - Emits optional tighten {C,O,CO} variants (when not mandatory) by **temporarily forcing** C/O into mandatory set.
    - Handles base_count==0: logs and emits 'relax' variants plus forced tighten variants.
    - Returns ≤6 queries spanning TARGET_MIN..TARGET_MAX (or brackets when none within).
    """
    qlog_path = os.path.join(OUTDIR, "query_manager.log")
    with open(qlog_path, "w", encoding="utf-8") as _fw:
        _fw.write(f"# Query Manager Log — {datetime.now(timezone.utc).isoformat()}\n")

    def _log(lines: List[str]):
        _append_lines(qlog_path, lines)

    M = set(x.upper() for x in (proto.mandatory_blocks or []))
    target_min = proto.query_targets["TARGET_MIN"]
    target_max = proto.query_targets["TARGET_MAX"]

    def bank(base: List[str], curated: List[str], max_curated: int = 6) -> List[str]:
        aug = (curated or [])[:max_curated]
        seen=set(); out=[]
        for t in (base + aug):
            tt=" ".join((t or "").split())
            if not tt: continue
            key = tt.lower()
            if key in seen: continue
            seen.add(key); out.append(tt)
        return out

    P_bank = bank(P_terms, mesh_curated.get("P", []))
    I_bank = bank(I_terms, mesh_curated.get("I", []))
    C_bank = bank(C_terms, mesh_curated.get("C", []))
    O_bank = bank(O_terms, mesh_curated.get("O", []))

    if "C" in M and not C_bank:
        _log(["[ERROR] mandatory C but no C terms after augment."])
        raise RuntimeError("Mandatory C block requested but no C terms available.")
    if "O" in M and not O_bank:
        _log(["[ERROR] mandatory O but no O terms after augment."])
        raise RuntimeError("Mandatory O block requested but no O terms available.")

    all_candidates: List[Tuple[str,int,Dict[str,object]]] = []

    for cap in caps:
        P = _or_block(P_bank, cap)
        I = _or_block(I_bank, cap)
        C = _or_block(C_bank, max(2, cap//2))
        O = _or_block(O_bank, max(2, cap//2))

        q0 = _build_query_from_blocks(P, I, C, O, proto.year_min, proto.year_max, M)
        base_count = esearch_count(q0)
        _log([f"# cap={cap} base_count={base_count} :: {q0}"])

        if base_count == 0:
            # Relax pass
            P_rel = _or_block(P_bank + (mesh_curated.get("P", []) or [])[:12], min(12, max(len(P), cap+4)))
            I_rel = _or_block(I_bank + (mesh_curated.get("I", []) or [])[:12], min(12, max(len(I), cap+4)))
            C_rel = _or_block(C_bank + (mesh_curated.get("C", []) or [])[:8],  min(6, max(len(C), (cap//2)+2)))
            O_rel = _or_block(O_bank + (mesh_curated.get("O", []) or [])[:8],  min(6, max(len(O), (cap//2)+2)))
            q_relax = _build_query_from_blocks(P_rel, I_rel, C_rel, O_rel, proto.year_min, proto.year_max, M)
            c_relax = esearch_count(q_relax)
            _log([f"  [relax] count={c_relax} :: {q_relax}"])
            all_candidates.append((q_relax, c_relax, dict(kind="relax", cap=cap)))
            # Forced optional tighten variants in relax
            if "C" not in M and C_rel:
                qC = _build_query_from_blocks(P_rel, I_rel, C_rel[:2], [], proto.year_min, proto.year_max, M | {"C"})
                all_candidates.append((qC, esearch_count(qC), dict(kind="relax_tight_C", cap=cap)))
            if "O" not in M and O_rel:
                qO = _build_query_from_blocks(P_rel, I_rel, [], O_rel[:2], proto.year_min, proto.year_max, M | {"O"})
                all_candidates.append((qO, esearch_count(qO), dict(kind="relax_tight_O", cap=cap)))
            if ("C" not in M and C_rel) and ("O" not in M and O_rel):
                qCO = _build_query_from_blocks(P_rel, I_rel, C_rel[:2], O_rel[:1], proto.year_min, proto.year_max, M | {"C","O"})
                all_candidates.append((qCO, esearch_count(qCO), dict(kind="relax_tight_CO", cap=cap)))
            continue

        # Ablation deltas
        def ablate(block_name: str, terms: List[str], curP, curI, curC, curO) -> List[Tuple[str,float]]:
            if not terms: return []
            deltas=[]
            for t in terms:
                PP, II, CC, OO = curP[:], curI[:], curC[:], curO[:]
                if block_name=="P": PP.remove(t)
                elif block_name=="I": II.remove(t)
                elif block_name=="C": CC.remove(t)
                elif block_name=="O": OO.remove(t)
                q = _build_query_from_blocks(PP, II, CC, OO, proto.year_min, proto.year_max, M)
                c = esearch_count(q)
                delta = base_count - c
                deltas.append((t, float(delta)/max(1,base_count)))
            deltas.sort(key=lambda x: -x[1])
            return deltas

        inflators = {"P":[], "I":[], "C":[], "O":[]}
        if P: inflators["P"] = ablate("P", P, P, I, C, O)
        if I: inflators["I"] = ablate("I", I, P, I, C, O)
        if ("C" in M) and C: inflators["C"] = ablate("C", C, P, I, C, O)
        if ("O" in M) and O: inflators["O"] = ablate("O", O, P, I, C, O)

        def drop_inflators(block_terms: List[str], infl_list: List[Tuple[str,float]]) -> List[str]:
            to_drop = {t for (t,frac) in infl_list if frac >= inflator_frac}
            kept = [t for t in block_terms if t not in to_drop]
            return kept if kept else block_terms

        P2 = drop_inflators(P, inflators["P"])
        I2 = drop_inflators(I, inflators["I"])
        C2, O2 = C[:], O[:]

        q_refined = _build_query_from_blocks(P2, I2, C2, O2, proto.year_min, proto.year_max, M)
        c_refined = esearch_count(q_refined)
        _log([f" refined_count={c_refined} :: {q_refined}"])
        for blk in ["P","I","C","O"]:
            if inflators[blk]:
                _log([f"  inflators[{blk}]: " + ", ".join(f"{t}:{frac:.2f}" for t, frac in inflators[blk][:8])])

        # Rescue if under min
        if c_refined < target_min:
            def rescue(block_name: str, have: List[str], bank: List[str]) -> List[str]:
                pool = [t for t in bank if t not in have]
                gains=[]
                for t in pool[:8]:
                    PP,II,CC,OO = P2[:], I2[:], C2[:], O2[:]
                    if block_name=="P": PP.append(t)
                    if block_name=="I": II.append(t)
                    if block_name=="C": CC.append(t)
                    if block_name=="O": OO.append(t)
                    q = _build_query_from_blocks(PP, II, CC, OO, proto.year_min, proto.year_max, M)
                    c = esearch_count(q)
                    gains.append((t, c - c_refined))
                gains.sort(key=lambda x: -x[1])
                add = [t for t,g in gains[:rescue_top_k] if g>0]
                return have + add

            I2 = rescue("I", I2, I_bank)
            P2 = rescue("P", P2, P_bank)
            if "C" in M and C2: C2 = rescue("C", C2, C_bank)
            if "O" in M and O2: O2 = rescue("O", O2, O_bank)
            q_final = _build_query_from_blocks(P2, I2, C2, O2, proto.year_min, proto.year_max, M)
            c_final = esearch_count(q_final)
            _log([f" rescue_count={c_final} :: {q_final}"])
        else:
            q_final, c_final = q_refined, c_refined

        all_candidates.append((q_final, c_final, dict(kind="core", cap=cap)))

        # Optional tighten with forced inclusion (via M | {...})
        if "C" not in M and C_bank:
            Cmin = C_bank[: min(2, len(C_bank))]
            qC = _build_query_from_blocks(P2, I2, Cmin, [], proto.year_min, proto.year_max, M | {"C"})
            all_candidates.append((qC, esearch_count(qC), dict(kind="tight_C", cap=cap, C=len(Cmin))))
        if "O" not in M and O_bank:
            Omin = O_bank[: min(2, len(O_bank))]
            qO = _build_query_from_blocks(P2, I2, [], Omin, proto.year_min, proto.year_max, M | {"O"})
            all_candidates.append((qO, esearch_count(qO), dict(kind="tight_O", cap=cap, O=len(Omin))))
        if ("C" not in M and C_bank) and ("O" not in M and O_bank):
            Cmin = C_bank[: min(2, len(C_bank))]
            Omin = O_bank[: min(1, len(O_bank))]
            qCO = _build_query_from_blocks(P2, I2, Cmin, Omin, proto.year_min, proto.year_max, M | {"C","O"})
            all_candidates.append((qCO, esearch_count(qCO), dict(kind="tight_CO", cap=cap, C=len(Cmin), O=len(Omin))))

    # Dedup & sort
    seenQ=set(); uniq=[]
    for q,c,meta in sorted(all_candidates, key=lambda x: x[1]):
        k = re.sub(r"\s+", " ", q.strip())
        if k in seenQ: continue
        seenQ.add(k); uniq.append((q,c,meta))

    # Pick ≤6 queries
    target_min = proto.query_targets["TARGET_MIN"]
    target_max = proto.query_targets["TARGET_MAX"]
    within = [(q,c) for (q,c,_) in uniq if target_min <= c <= target_max]
    if within:
        picked = [q for (q,_) in within[:6]]
    else:
        nz = [(q,c) for (q,c,_) in uniq if c>0]
        if len(nz) >= 2:
            picked = [nz[0][0], nz[-1][0]]
            mids = [q for (q,_) in nz[1:-1][:max(0,6-2)]]
            picked.extend(mids)
        else:
            picked = [q for (q,_,_) in uniq[:min(6,len(uniq))]]

    lines = [f"# target_min={target_min} target_max={target_max}", "# Candidates (count asc):"]
    for (q,c,meta) in uniq:
        lines.append(f"{c}\t{meta}\t{q}")
    lines.append("# Picked (≤6):")
    for q in picked:
        lines.append(q)
    _log(lines)

    return picked, "\n".join(lines)

# ----------------------------
# CELL 6 — Universe fetch + deterministic prefilter + PRISMA summary/detail
# ----------------------------
def prefilter_record(rec: dict,
                     year_min: int, year_max: int,
                     blocklist: Set[str],
                     designs_allow: Set[str]) -> Tuple[bool, Dict[str,bool]]:
    y_ok = (rec["year"] is not None and year_min <= rec["year"] <= year_max)
    pts = set(pt.lower() for pt in rec["pubtypes"])
    block_hit = any(pt in blocklist for pt in pts)
    design_ok = any(pt in designs_allow for pt in pts)
    ok = (y_ok and (not block_hit) and design_ok)
    return ok, {"year_ok":y_ok, "pubtype_ok": (not block_hit), "design_ok": design_ok}

def fetch_universe(proto: Protocol, queries: List[str]) -> List[dict]:
    all_pmids: List[int] = []
    for q in queries:
        pmids = esearch_fetch_pmids(q, retmax=proto.query_targets["TARGET_MAX"])
        all_pmids.extend(pmids)
    pmid_uniq = list(dict.fromkeys(all_pmids))[:proto.query_targets["UNIVERSE_FETCH_MAX"]]
    root_log.info(f"[Universe] unique PMIDs={len(pmid_uniq)} — fetching efetch summaries…")
    recs = efetch_summaries(pmid_uniq)

    _dump_jsonl(os.path.join(OUTDIR, "universe_raw.jsonl"), recs)

    block = set(pt.lower() for pt in proto.pubtype_blocklist)
    allow = set(pt.lower() for pt in proto.designs_allowlist)

    kept, year_fail, pubtype_blocked, design_fail = [], 0, 0, 0
    detail_rows = []

    for r in recs:
        ok, flags = prefilter_record(r, proto.year_min, proto.year_max, block, allow)
        if not flags["year_ok"]: year_fail += 1
        if not flags["pubtype_ok"]: pubtype_blocked += 1
        if not flags["design_ok"]: design_fail += 1

        detail_rows.append({
            "pmid": r["pmid"],
            "year_ok": flags["year_ok"],
            "pubtype_ok": flags["pubtype_ok"],
            "design_ok": flags["design_ok"],
            "kept": ok
        })

        if ok:
            r["_prefilter_flags"] = flags
            kept.append(r)

    write_csv(os.path.join(OUTDIR, "prefilter_detail.csv"), detail_rows,
              ["pmid","year_ok","pubtype_ok","design_ok","kept"])

    if not kept:
        write_csv(os.path.join(OUTDIR, "prefilter_summary.csv"), [{
            "universe_raw": len(recs),
            "year_fail": year_fail,
            "pubtype_blocked": pubtype_blocked,
            "design_fail": design_fail,
            "kept": 0
        }])
        raise RuntimeError("Design viability check failed: 0 records intersect designs_allowlist.")

    _dump_jsonl(os.path.join(OUTDIR, "universe.jsonl"), kept)
    write_csv(os.path.join(OUTDIR, "prefilter_summary.csv"), [{
        "universe_raw": len(recs),
        "year_fail": year_fail,
        "pubtype_blocked": pubtype_blocked,
        "design_fail": design_fail,
        "kept": len(kept)
    }])
    return kept

# ----------------------------
# CELL 7 — MeSH mining & curation via LLM
# ----------------------------
def llm_chat(endpoint: str, model: str, system_prompt: str, user_prompt: str, temperature: float=0.0) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "temperature": temperature,
        "stream": False
    }
    for _ in range(LM_RETRY):
        try:
            r = requests.post(endpoint, json=payload, timeout=LM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            time.sleep(0.8)
    raise RuntimeError("LLM chat failed after retries")

def curate_mesh(proto: Protocol, base_pmids: List[int], universe: List[dict], stage: int=1) -> Dict[str,List[str]]:
    # collect MeSH from key_pmids (stage-1), optionally from stage-1 includes for stage-2
    mesh_terms = []
    base_set = set(base_pmids)
    for rec in universe:
        if rec["pmid"] in base_set:
            mesh_terms.extend(rec.get("mesh", []) or [])
    mesh_terms = list(dict.fromkeys([m for m in mesh_terms if m]))

    sys_prompt = (
        'You are curating MeSH terms into P/I/C/O for a systematic review. Output strict JSON only.\n'
        'Return ONLY valid JSON (no markdown/code fences) with EXACTLY these keys:\n'
        '{"P":[],"I":[],"C":[],"O":[],"rejected":[]}\n'
        'Each value must be an array of strings. Do not include any other keys.'         
    )
    user_payload = {
        "narrative_question": proto.narrative_question,
        "P_terms": proto.P_terms, "I_terms": proto.I_terms,
        "C_terms": proto.C_terms, "O_terms": proto.O_terms,
        "mesh_terms": mesh_terms
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    txt = llm_chat(proto.llm["chat_endpoint"], proto.llm["chat_model"], sys_prompt, user_prompt)
    try:
        curated = json.loads(txt)
    except Exception:
        # robust parse: extract first JSON object
        m = re.search(r"\{.*\}", txt, re.S)
        curated = json.loads(m.group(0)) if m else {"P":[],"I":[],"C":[],"O":[],"rejected":[]}

    out = {
        "P": curated.get("P", [])[:20],
        "I": curated.get("I", [])[:20],
        "C": curated.get("C", [])[:20],
        "O": curated.get("O", [])[:20],
        "rejected": curated.get("rejected", [])[:50]
    }
    path = os.path.join(OUTDIR, "mesh_curated.json" if stage==1 else "mesh_curated_stage2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

# ----------------------------
# CELL 8 — Ranking (TF-IDF + Embeddings + MeSH-Jaccard) → RRF with recency tie
# ----------------------------
def _cosine(a, b, eps=1e-9):
    """Robust cosine for list/tuple/np.array inputs."""
    import numpy as np
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    na = np.linalg.norm(av) + eps
    nb = np.linalg.norm(bv) + eps
    return float(np.dot(av, bv) / (na * nb))

def _tfidf_scores(records: List[dict], ref_text: str) -> Tuple[List[float], List[int]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [((r["title"] or "") + " " + (r["abstract"] or "")) for r in records]
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", smooth_idf=True)
    X = vec.fit_transform(texts + [ref_text])
    ref_vec = X[-1]
    sims = cosine_similarity(X[:-1], ref_vec).ravel()
    scores = sims.tolist()
    ranks = _ranks_desc(scores)
    return scores, ranks

def _ranks_desc(scores: List[float]) -> List[int]:
    # 1 = best
    sorted_idx = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    rank = [0]*len(scores)
    for r,i in enumerate(sorted_idx, start=1):
        rank[i] = r
    return rank

def embed_texts(endpoint: str, model: str, texts: List[str], batch_size: int=64) -> List[List[float]]:
    # LM Studio embedding endpoint: {"input":[...], "model":"..."}
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        payload = {"model": model, "input": chunk}
        tries = LM_RETRY
        while tries>0:
            try:
                r = requests.post(endpoint, json=payload, timeout=LM_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                # Expect list under "data": [{"embedding":[...]}...]
                embs = [row["embedding"] for row in data.get("data", [])]
                if len(embs) != len(chunk):
                    raise RuntimeError("Embedding batch size mismatch")
                out.extend(embs)
                break
            except Exception:
                tries-=1; time.sleep(0.8)
        if tries==0:
            raise RuntimeError("Embedding API failed after retries")
    return out

def _embedding_scores(proto: Protocol, records: List[dict], ref_text: str) -> Tuple[List[float], List[int]]:
    texts = [((r["title"] or "") + " " + (r["abstract"] or "")) for r in records]
    embs = embed_texts(proto.llm["embed_endpoint"], proto.llm["embed_model"], texts + [ref_text], batch_size=64)
    ref = embs[-1]
    scores = [_cosine(e, ref) for e in embs[:-1]]
    ranks = _ranks_desc(scores)
    return scores, ranks

def _mesh_jaccard_scores(records: List[dict], curated: Dict[str,List[str]]) -> Tuple[List[float], List[int]]:
    Pset = set(m.lower() for m in (curated.get("P") or []))
    Iset = set(m.lower() for m in (curated.get("I") or []))
    Cset = set(m.lower() for m in (curated.get("C") or []))
    Oset = set(m.lower() for m in (curated.get("O") or []))

    scores = []
    for r in records:
        mset = set((m or "").lower() for m in (r.get("mesh") or []))
        def jac(A,B):
            if not A or not B: return 0.0
            inter = len(A & B); den = len(A | B) or 1
            return inter / den
        sP = jac(mset, Pset)
        sI = jac(mset, Iset)
        sC = jac(mset, Cset)
        sO = jac(mset, Oset)
        s = 0.4*sP + 0.4*sI + 0.1*sC + 0.1*sO
        scores.append(s)
    ranks = _ranks_desc(scores)
    return scores, ranks

def _rrf_fuse(ranks_lists: List[List[int]], k: int=60) -> List[float]:
    # Reciprocal Rank Fusion
    n = len(ranks_lists[0]) if ranks_lists else 0
    scores = [0.0]*n
    for ranks in ranks_lists:
        for i, r in enumerate(ranks):
            scores[i] += 1.0 / (k + r)
    return scores

def rank_stage(proto: Protocol, kept: List[dict], curated: Dict[str,List[str]], out_csv: str) -> List[dict]:
    # Build reference text
    ref_parts = [proto.narrative_question]
    for k in ["P_terms","I_terms","C_terms","O_terms"]:
        ref_parts.extend(getattr(proto, k))
    for k in ["P","I","C","O"]:
        ref_parts.extend(curated.get(k, []))
    ref_text = " ".join(ref_parts)

    # TF-IDF
    tfidf_scores, tfidf_ranks = _tfidf_scores(kept, ref_text)

    # Embeddings (batched)
    emb_scores, emb_ranks = _embedding_scores(proto, kept, ref_text)

    # MeSH-Jaccard
    mesh_scores, mesh_ranks = _mesh_jaccard_scores(kept, curated)

    # RRF + recency tiebreak
    rrf_scores = _rrf_fuse([_ranks_desc(tfidf_scores), _ranks_desc(emb_scores), _ranks_desc(mesh_scores)], k=60)
    # sort by rrf desc, then year desc
    order = sorted(range(len(kept)), key=lambda i: (-rrf_scores[i], -(kept[i]["year"] or 0), i))

    # compute ranks from order
    rank_rrf = [0]*len(kept)
    for r,i in enumerate(order, start=1):
        rank_rrf[i] = r

    # materialize candidate CSV with evidence
    rows=[]
    for i, rec in enumerate(kept):
        rows.append({
            "pmid": rec["pmid"],
            "title": rec["title"],
            "abstract": rec["abstract"],
            "year": rec["year"],
            "pubtypes": "; ".join(rec["pubtypes"] or []),
            "mesh": "; ".join(rec["mesh"] or []),
            "first_author": rec.get("first_author") or "",
            "doi": rec.get("doi") or "",
            "score_tfidf": f"{tfidf_scores[i]:.6f}",
            "score_emb": f"{emb_scores[i]:.6f}",
            "score_mesh": f"{mesh_scores[i]:.6f}",
            "rank_tfidf": tfidf_ranks[i],
            "rank_emb": emb_ranks[i],
            "rank_mesh": mesh_ranks[i],
            "rrf": f"{rrf_scores[i]:.6f}",
            "rank_rrf": rank_rrf[i],
        })
    # reorder by rank_rrf asc
    rows.sort(key=lambda r: (r["rank_rrf"], -int(r["year"] or 0)))
    write_csv(os.path.join(OUTDIR, out_csv), rows)
    return kept  # data is already in OUTDIR CSV

# ----------------------------
# CELL 9 — LLM TIAB screening with sliding-window stop
# ----------------------------
def screen_tiab(proto: Protocol, curated: Dict[str,List[str]], records_csv: str,
                out_jsonl: str, out_included_csv: str, pris_ref: str="screen"):
    # load candidates
    import pandas as pd
    df = pd.read_csv(os.path.join(OUTDIR, records_csv))

    # sliding stop only if N > cap
    N = len(df)
    cap_thr = int(proto.screening["llm_screen_cap_threshold"])
    window = int(proto.screening["yield_window"])
    min_rate = float(proto.screening["yield_min_rate"])
    consec = int(proto.screening["yield_consecutive"])

    sys_prompt = "You classify RCT/observational articles for inclusion in a systematic review. Return strict JSON (see schema). Keep reasons ≤200 chars."
    schema_example = {
        "label": "include|exclude|maybe",
        "conf": 0.0,
        "reason": "",
        "P": False, "I": False, "C": False, "O": False,
        "design_ok": False, "pubtype_ok": False, "year_ok": False,
        "mesh_hits": [], "salient_terms": [],
        "pris_ref": "screen",
        "excl_code": "PT|POP|INT|OUT|DUP|OTHER"
    }

    def safe_int_year(val):
        try:
            # covers float NaN, strings, and None
            iv = int(val)
            return iv if 1500 <= iv <= 2100 else None
        except Exception:
            return None

    def mk_user_payload(row: dict) -> str:
        payload = {
            "protocol": asdict(proto),
            "mesh_curated": curated,
            "record": {
                "pmid": int(row["pmid"]),
                "title": row.get("title") or "",
                "abstract": row.get("abstract") or "",
                "year": safe_int_year(row.get("year")),
                "pubtypes": [p.strip() for p in str(row.get("pubtypes") or "").split(";") if p.strip()],
                "mesh": [m.strip() for m in str(row.get("mesh") or "").split(";") if m.strip()],
            },
            "schema": schema_example
        }
        return json.dumps(payload, ensure_ascii=False)

    out_path = os.path.join(OUTDIR, out_jsonl)
    inc_rows = []
    yields_in_window=0; windows_below=0; processed=0

    log_path = os.path.join(OUTDIR, "screening.log")
    # Append a small header per run to make idempotent chunks obvious
    _append_lines(log_path, [f"# {datetime.now(timezone.utc).isoformat()}Z {records_csv} -> {out_jsonl}"])

    with open(out_path, "w", encoding="utf-8") as jf:
        for idx, row in df.iterrows():
            # sliding stop applies only when N > cap_thr
            if N > cap_thr and processed>0 and processed % window == 0:
                rate = yields_in_window / window
                if rate < min_rate:
                    windows_below += 1
                else:
                    windows_below = 0
                yields_in_window = 0
                if windows_below >= consec:
                    root_log.info(f"[TIAB] Sliding stop activated at {processed}/{N}.")
                    break

            user_prompt = mk_user_payload(row.to_dict())
            txt = llm_chat(proto.llm["chat_endpoint"], proto.llm["chat_model"], sys_prompt, user_prompt)
            try:
                obj = json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                obj = json.loads(m.group(0)) if m else {"label":"exclude","conf":0.0,"reason":"parse_error","pris_ref":pris_ref,"excl_code":"OTHER"}
            obj["pris_ref"] = pris_ref  # enforce
            jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # Log a single-line audit record
            try:
                conf = float(obj.get("conf", 0.0))
            except Exception:
                conf = 0.0
            _append_lines(log_path, [f"{pris_ref}\tpmid={int(row['pmid'])}\tlabel={obj.get('label')}\tconf={conf:.2f}\treason={(obj.get('reason') or '')[:60]}"])

            if obj.get("label") in ("include","maybe"):
                inc_rows.append({"pmid": int(row["pmid"])})
                yields_in_window += 1
            processed += 1

    write_csv(os.path.join(OUTDIR, out_included_csv), inc_rows, ["pmid"])

# ----------------------------
# CELL 10 — CILE Expansion (integrate provided CILE code as-is, then use it)
# ----------------------------
# We embed the provided CILE code verbatim and execute it in a module namespace.
# Then we call its outer_loop_cile() using Stage-1 included PMIDs as seeds.

# NOTE:
# The CILE code block is extremely long. For correctness and to fully integrate "as-is",
# you should paste the complete provided CILE script content in the CILE_SRC string above
# (verbatim, without modification). Due to message size limits here, the CILE_SRC has been
# truncated in this display at the `@dataclass class HGraph` duplicate line. In your
# working script, include the full original CILE code exactly as provided, then proceed
# with the loader below.

#def _load_cile_module():
#    import types
#    mod = types.ModuleType("cile_engine")
#    exec(CILE_SRC, mod.__dict__)
#    return mod

#not needed anymore as the CILE algorithm has been exported to another .py file and called her.

# ----------------------------
# CELL 11 — Merge stages & handoff CSV (FirstAuthor + DOI)
# ----------------------------
def merge_and_handoff():
    import pandas as pd
    s1 = os.path.join(OUTDIR, "stage1_included.csv")
    s2 = os.path.join(OUTDIR, "stage2_included.csv")
    c1 = os.path.join(OUTDIR, "triage_stage1_candidates.csv")
    c2 = os.path.join(OUTDIR, "triage_stage2_candidates.csv")
    # read included pmids
    inc_pmids=set()
    if os.path.exists(s1):
        inc_pmids |= set(pd.read_csv(s1)["pmid"].astype(int).tolist())
    if os.path.exists(s2):
        inc_pmids |= set(pd.read_csv(s2)["pmid"].astype(int).tolist())

    # read candidates for evidence
    frames=[]
    if os.path.exists(c1):
        frames.append(pd.read_csv(c1))
    if os.path.exists(c2):
        frames.append(pd.read_csv(c2))
    if frames:
        cand = pd.concat(frames, ignore_index=True)
        cand["pmid"] = cand["pmid"].astype(int)
        cand = cand.sort_values(["pmid", "rank_rrf"]).groupby("pmid", as_index=False).first()
        triage_master = cand[cand["pmid"].isin(inc_pmids)].copy()
        triage_master = triage_master.sort_values("rank_rrf")
        triage_master.to_csv(os.path.join(OUTDIR, "triage_master.csv"), index=False)
    with open(os.path.join(OUTDIR, "final_triage_pmids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(list(inc_pmids)), f, indent=2)

    # Build full-text handoff
    # We need Year, FirstAuthor, Title, DOI
    # Pull from triage_master which already carries these fields
    handoff_rows=[]
    if frames:
        for _, r in triage_master.iterrows():
            handoff_rows.append({
                "PMID": int(r["pmid"]),
                "Year": int(r["year"]) if not math.isnan(r["year"]) else "",
                "FirstAuthor": r.get("first_author") or "",
                "Title": r.get("title") or "",
                "DOI": r.get("doi") or ""
            })
    write_csv(os.path.join(OUTDIR, "final_fulltext_handoff.csv"),
              handoff_rows, ["PMID","Year","FirstAuthor","Title","DOI"])

# ----------------------------
# CELL 12 — Full-text fetcher integration (provided script as-is) + extraction + final LLM
# ----------------------------

# NOTE:
# As with CILE_SRC, paste the entire fetcher script (verbatim) into FETCHER_SRC.
# It has been trimmed here purely due to message size constraints.
#again, not needed anymore. full text script lives in it's own py file.

def _run_fetcher_with_csv(handoff_csv_path: str, pdf_out_dir: str):
    """
    Execute the provided fetcher module as-is, with a scoped override that:
    - Sets EXCEL_FILE_PATH to our CSV path
    - Sets OUTPUT_PDF_DIR to desired output directory
    - Disables Excel usage by overriding pd.read_excel to read CSV instead
    - Blocks pd.ExcelFile
    """
    import types
    import pandas as pd
    ft = types.ModuleType("fulltext_fetcher_v4")
    exec(FETCHER_SRC, ft.__dict__)
    # point to our CSV and output dir
    ft.EXCEL_FILE_PATH = handoff_csv_path
    ft.OUTPUT_PDF_DIR = pdf_out_dir

    # Scoped overrides
    def _read_excel_csv_shim(path, *a, **k):
        # read our CSV and provide DataFrame with expected columns
        return pd.read_csv(path)
    ft.pd.read_excel = _read_excel_csv_shim
    # harden against alternative Excel paths
    def _excelfile_blocker(*a, **k):
        raise RuntimeError("Excel disabled in this integration")
    ft.pd.ExcelFile = _excelfile_blocker

    # run
    ft.main()

# ----------------------------
# FULL-TEXT extraction + final LLM screen
# ----------------------------
def extract_text_pdf(pdf_path: str, ocr: bool=False) -> Tuple[str, str]:
    method = "pdfminer"
    text = ""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path) or ""
    except Exception:
        text = ""
    if len(text.strip()) < 1000:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            pages = convert_from_path(pdf_path, dpi=300)
            buf = []
            for im in pages:
                buf.append(pytesseract.image_to_string(im))
            text = "\n".join(buf)
            method = "ocr"
        except Exception:
            pass
    return text, method

def _load_curated_for_fulltext() -> Dict[str,List[str]]:
    p2 = os.path.join(OUTDIR, "mesh_curated_stage2.json")
    p1 = os.path.join(OUTDIR, "mesh_curated.json")
    path = p2 if os.path.exists(p2) else p1
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"P":[],"I":[],"C":[],"O":[]}

def fulltext_screen(proto: Protocol):
    import pandas as pd

    # PDFs location
    pdf_dir = os.path.join(OUTDIR, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    # run fetcher (as-is) against our CSV (Sci-Hub gated by env in _run_fetcher_with_csv)
    handoff = os.path.join(OUTDIR, "final_fulltext_handoff.csv")
    _run_fetcher_with_csv(handoff, pdf_dir)

    # collect PDFs and map by PMID from handoff
    df = pd.read_csv(handoff)
    # Normalize column names that we care about
    for col in ["PMID","Year","FirstAuthor","Title","DOI"]:
        if col not in df.columns:
            df[col] = ""
    df["PMID"] = pd.to_numeric(df["PMID"], errors="coerce").astype("Int64")
    meta = {}
    for _, r in df.iterrows():
        pmid = int(r["PMID"]) if pd.notna(r["PMID"]) else None
        if pmid:
            meta[pmid] = {
                "Title": str(r["Title"] or ""),
                "Year": str(r["Year"] or ""),
                "FirstAuthor": str(r["FirstAuthor"] or ""),
                "DOI": str(r["DOI"] or ""),
            }

    # Final screen
    curated = _load_curated_for_fulltext()
    sys_prompt = "You classify full-text RCT/observational articles for inclusion in a systematic review. Return strict JSON (see schema). No 'maybe'."
    schema_example = {
        "label": "include|exclude",
        "conf": 0.0,
        "reason": "",
        "P": False, "I": False, "C": False, "O": False,
        "design_ok": False, "pubtype_ok": False, "year_ok": False,
        "mesh_hits": [], "salient_terms": [],
        "pris_ref": "fulltext",
        "excl_code": "PT|POP|INT|OUT|DUP|OTHER",
        "extraction": "pdfminer|ocr"
    }

    screened_rows=[]
    flog = os.path.join(OUTDIR, "fulltext.log")
    _append_lines(flog, [f"# {datetime.now(timezone.utc).isoformat()}Z fulltext run start"])

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        m = re.match(r"(\d{4})_(.+?)_(\d+)\.pdf", fname)
        if not m:
            # skip files that don't follow naming convention
            continue
        pmid = int(m.group(3))
        pdf_path = os.path.join(pdf_dir, fname)
        text, method = extract_text_pdf(pdf_path)

        rmeta = meta.get(pmid, {"Title":"", "Year":"", "FirstAuthor":"", "DOI":""})
        # build user prompt
        user_payload = {
            "protocol": asdict(proto),
            "mesh_curated": curated,
            "record": {
                "pmid": pmid,
                "title": rmeta["Title"],
                "year": int(rmeta["Year"]) if str(rmeta["Year"]).isdigit() else None,
                "first_author": rmeta["FirstAuthor"],
                "doi": rmeta["DOI"]
            },
            "fulltext_excerpt": text[:120000],  # truncate if huge
            "schema": schema_example
        }
        txt = llm_chat(proto.llm["chat_endpoint"], proto.llm["chat_model"], sys_prompt, json.dumps(user_payload, ensure_ascii=False))
        try:
            obj = json.loads(txt)
        except Exception:
            m2 = re.search(r"\{.*\}", txt, re.S)
            obj = json.loads(m2.group(0)) if m2 else {"label":"exclude","conf":0.0,"reason":"parse_error","pris_ref":"fulltext","excl_code":"OTHER"}
        obj["pris_ref"]="fulltext"; obj["extraction"]=method

        # Append to log
        try:
            cval = float(obj.get("conf", 0.0))
        except Exception:
            cval = 0.0
        _append_lines(flog, [f"pmid={pmid}\tlabel={obj.get('label')}\tconf={cval:.2f}\tmethod={method}"])

        screened_rows.append({
            "pmid": pmid,
            "label": obj.get("label"),
            "conf": obj.get("conf"),
            "reason": obj.get("reason"),
            "extraction": method
        })

    write_csv(os.path.join(OUTDIR, "fulltext_screened.csv"), screened_rows,
              ["pmid","label","conf","reason","extraction"])

    # manual_fulltext_todo.csv from failed list (if provided by fetcher)
    failed_txt = os.path.join(OUTDIR, "fulltext_failed_pmids.txt")
    # The provided fetcher writes to its working dir; move if exists
    if os.path.exists(os.path.join("fulltext_failed_pmids.txt")):
        shutil.move("fulltext_failed_pmids.txt", failed_txt)
    if os.path.exists(failed_txt):
        todo=[]
        with open(failed_txt, "r", encoding="utf-8") as f:
            for line in f:
                pmid_str = line.strip()
                if not pmid_str.isdigit():
                    continue
                pmid = int(pmid_str)
                r = meta.get(pmid, {"DOI":"", "Title":"", "Year":""})
                todo.append({
                    "PMID": pmid,
                    "DOI": r.get("DOI",""),
                    "Title": r.get("Title",""),
                    "Year": r.get("Year","")
                })
        write_csv(os.path.join(OUTDIR, "manual_fulltext_todo.csv"), todo,
                  ["PMID","DOI","Title","Year"])

# ----------------------------
# ORCHESTRATOR
# ----------------------------
def run_pipeline(protocol_path: str):
    proto = Protocol.load(protocol_path)

    # 1) MeSH curation from key_pmids (stage-1)
    root_log.info("[MeSH] Stage-1 curation from key_pmids…")
    # Fetch universe for key pmids if not present
    # We'll reuse efetch to ensure we have mesh terms for seeds
    seed_recs = efetch_summaries(proto.key_pmids)
    _dump_jsonl(os.path.join(OUTDIR, "seed_keypmids.jsonl"), seed_recs)
    curated1 = curate_mesh(proto, proto.key_pmids, seed_recs, stage=1)

    # 2) Query generation with culprit analysis
    root_log.info("[Query] Building Boolean queries with culprit analysis…")
    queries, qlog = culprit_analysis(
        proto, curated1, proto.P_terms, proto.I_terms, proto.C_terms, proto.O_terms
    )
    # 3) Universe fetch + prefilter (+PRISMA)
    root_log.info("[Universe] Fetch + deterministic prefilter…")
    kept = fetch_universe(proto, queries)

    # 4) Ranking stage-1
    root_log.info("[Rank] Stage-1 ranking…")
    rank_stage(proto, kept, curated1, out_csv="triage_stage1_candidates.csv")

    # 5) LLM title/abstract screening stage-1
    root_log.info("[Screen] Stage-1 TIAB LLM…")
    screen_tiab(proto, curated1, "triage_stage1_candidates.csv", "s1_llm_screen.jsonl", "stage1_included.csv", "screen")

    # 6) CILE expansion for stage-2
    log.info("[CILE] Running CILE (external module) for stage-2 candidates…")

    # ---- Build seed PMIDs for CILE (FIXES 'seeds' UNBOUND) ----
    import csv as _csv

    seeds = []  # list[int]
    try:
        # Preferred: take top items from stage-1 triage RRF ranking
        _cand_csv = os.path.join(out_dir, "triage_stage1_candidates.csv")
        if os.path.exists(_cand_csv):
            with open(_cand_csv, newline='', encoding='utf-8') as _f:
                _rows = list(_csv.DictReader(_f))
            # sort by ascending rank_rrf (1 is best); keep top 12
            _rows = [r for r in _rows if str(r.get("rank_rrf", "")).strip().isdigit()]
            _rows.sort(key=lambda r: int(r["rank_rrf"]))
            seeds = [int(r["pmid"]) for r in _rows[:12]]

        # Fallback: try protocol's key PMIDs if present
        if not seeds:
            # if your protocol object is named 'proto' (it is, because you use proto.narrative_question above)
            if hasattr(proto, "key_pmids") and proto.key_pmids:
                seeds = [int(p) for p in proto.key_pmids[:12]]

        # Last resort: accept small set of kept PMIDs from prefilter_detail.csv
        if not seeds:
            _pre_csv = os.path.join(out_dir, "prefilter_detail.csv")
            if os.path.exists(_pre_csv):
                with open(_pre_csv, newline='', encoding='utf-8') as _f:
                    _rows = list(_csv.DictReader(_f))
                _kept = [int(r["pmid"]) for r in _rows if r.get("kept","").lower()=="true"]
                seeds = _kept[:12]

        if not seeds:
            log.warning("[CILE] No seeds found from triage/protocol; CILE will use an empty list (allowed but not ideal).")

    except Exception as e:
        log.exception("Failed to build CILE seeds; continuing with an empty list.")
        seeds = []

    # ---- Run CILE (external module, no exec) ----
    Hf, Af, meta = cile.outer_loop_cile(seeds, cile.OuterConfig(
        accept_policy="elastic_phi",
        max_accept_after_filter=300,
        H_external_budget=3000,
        per_node_ext_frac_cap=0.85,
        deterministic_reservoir=True,
        min_relevance_frac=0.05,      # set 0.0 to disable relevance gate
        quarantine_hubs=True,
        quarantine_mode="external",   # "total" or "external"
        A_expand_pernode_ext_cap=None,
        A_expand_pernode_ext_frac_cap=None,
    ))

    # (Optional) Persist a small summary so 'Af' and 'meta' aren’t “unused”
    try:
        with open(os.path.join(out_dir, "cile_meta.json"), "w", encoding="utf-8") as _fw:
            json.dump(meta, _fw, ensure_ascii=False, indent=2)
        # Dump A PMIDs for downstream
        _A_pmids = [Hf.pmids[i] for i in sorted(list(Af))]
        with open(os.path.join(out_dir, "cile_A_pmids.txt"), "w", encoding="utf-8") as _fw:
            _fw.write("\n".join(str(p) for p in _A_pmids))
        log.info("[CILE] Wrote cile_meta.json and cile_A_pmids.txt")
    except Exception:
        log.warning("[CILE] Could not write cile_meta.json / cile_A_pmids.txt (non-fatal)")



    # 7) Stage-2 fetch + prefilter
    stage2_recs=[]
    if stage2_pmids:
        root_log.info(f"[Stage2] Fetching {len(stage2_pmids)} PMIDs from CILE expansion…")
        raw2 = efetch_summaries(stage2_pmids)
        _dump_jsonl(os.path.join(OUTDIR, "stage2_raw.jsonl"), raw2)
        block = set(pt.lower() for pt in proto.pubtype_blocklist)
        allow = set(pt.lower() for pt in proto.designs_allowlist)
        for r in raw2:
            ok, flags = prefilter_record(r, proto.year_min, proto.year_max, block, allow)
            if ok:
                r["_prefilter_flags"] = flags
                stage2_recs.append(r)
        _dump_jsonl(os.path.join(OUTDIR, "stage2_prefiltered.jsonl"), stage2_recs)

    # 8) Stage-2 MeSH curation augmentation (iterative) & ranking + screening
    if stage2_recs:
        root_log.info("[MeSH] Stage-2 iterative curation from stage-1 includes…")
        curated2 = curate_mesh(proto, [int(x) for x in s1inc["pmid"].tolist()], kept + stage2_recs, stage=2)
        root_log.info("[Rank] Stage-2 ranking…")
        rank_stage(proto, stage2_recs, curated2, out_csv="triage_stage2_candidates.csv")
        root_log.info("[Screen] Stage-2 TIAB LLM…")
        screen_tiab(proto, curated2, "triage_stage2_candidates.csv", "s2_llm_screen.jsonl", "stage2_included.csv", "screen")
    else:
        curated2 = None
        write_csv(os.path.join(OUTDIR, "triage_stage2_candidates.csv"), [], [])

    # 9) Merge + handoff
    root_log.info("[Merge] Building master + full-text handoff…")
    merge_and_handoff()

    # 10) Full-text extraction + final LLM screen
    root_log.info("[FullText] Fetch → extract → final LLM…")
    fulltext_screen(proto)

    root_log.info("[DONE] All artifacts under triage_out/")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Systematic Review Triage Pipeline")
    ap.add_argument("protocol_json", help="Path to protocol JSON (see spec)")
    args = ap.parse_args()
    run_pipeline(args.protocol_json)

if __name__ == "__main__":
    main()
