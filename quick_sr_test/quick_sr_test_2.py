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
        if doi_node is not None:
            doi_text = doi_node.text if hasattr(doi_node, "text") else None
            doi_clean = (doi_text or "").strip()
            doi = doi_clean or None
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

def llm_chat(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout_s: int = 60,
    max_retries: int = 4,
    api_key_env: str = "OPENAI_API_KEY",
    log_path: Optional[str] = None,
    stream: bool = True,              # <- new, default off (so nothing else changes) # Default True so that we dont cutoff midgen
    idle_timeout_s: int = 5,          # <- only used when stream=True
    continuation_retry: int = 1,       # <- try to finish a cut-off JSON once
) -> str:
    import os, requests, time, json, re
    headers = {"Content-Type": "application/json", "Connection": "keep-alive", "Accept": "text/event-stream"}
    api_key = os.getenv(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(payload, stream_flag):
        # Short read timeout so we never block forever on SSE.
        # Connect: 10s. Read: max(idle_timeout_s, 30s) for slow models.
        tout = (10, max(idle_timeout_s, 30)) if stream_flag else timeout_s
        return requests.post(endpoint, json=payload, headers=headers, timeout=tout, stream=stream_flag)

    def _assemble_stream(payload) -> str:
        import requests, time, json
        r = _post(payload, True)
        r.raise_for_status()
        parts = []
        last = time.monotonic()
        try:
            for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                now = time.monotonic()
                if line:
                    last = now
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            j = json.loads(data)
                            ch = (j.get("choices") or [{}])[0]
                            delta = (ch.get("delta") or {}).get("content")
                            if delta:
                                parts.append(delta)
                            else:
                                msg = (ch.get("message") or {}).get("content")
                                if msg:
                                    parts.append(msg)
                            # also break if finish_reason explicitly says stop
                            if str(ch.get("finish_reason") or "").lower() == "stop":
                                break
                        except Exception:
                            pass
                # idle watchdog is just a backstop now
                if idle_timeout_s and (now - last) > idle_timeout_s:
                    break
        except requests.exceptions.ReadTimeout:
            # treat as graceful end-of-stream; return what we buffered
            pass
        finally:
            try:
                r.close()
            except Exception:
                pass
        return "".join(parts).strip()

    def _finish_suffix(prefix_text: str) -> str:
        # Ask model to output ONLY the remaining suffix to complete the JSON object.
        # We include the tail of what we’ve already got for alignment.
        tail = prefix_text[-400:] if prefix_text else ""
        cont_user = json.dumps({
            "instruction": (
                "Continue emitting EXACTLY the remaining characters to complete the same JSON object. "
                "Do NOT repeat prior content. Do NOT add code fences or commentary. "
                "Start immediately with the next character."
            ),
            "already_emitted_tail": tail
        }, ensure_ascii=False)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cont_user}
            ],
            "temperature": temperature,
            "stream": True
        }
        suffix = _assemble_stream(payload)
        return prefix_text + suffix

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": temperature,
        "stream": bool(stream),
    }

    last_status, last_text, last_err = None, None, None
    backoff = 1.0
    for _ in range(max_retries):
        try:
            if stream:
                text = _assemble_stream(payload)
                # If we seem cut off and a continuation is allowed, try to finish once
                if continuation_retry > 0:
                    # quick check: do we already have a {} JSON object somewhere?
                    if not re.search(r"\{.*\}", text, re.S):
                        text2 = _finish_suffix(text)
                        # prefer the longer candidate
                        if len(text2) > len(text):
                            text = text2
                if text:
                    return text
                last_text = "(empty stream)"
            else:
                r = _post(payload, False)
                last_status, last_text = r.status_code, r.text[:3000]
                if r.status_code == 200:
                    j = r.json()
                    return j["choices"][0]["message"]["content"]

            if last_status in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 1.7; continue
            break
        except Exception as e:
            last_err = f"{e.__class__.__name__}: {e}"
            time.sleep(backoff); backoff *= 1.7

    if log_path:
        _append_lines(log_path, [f"LLM_FAIL status={last_status} err={last_err} body={last_text}"])
    raise RuntimeError("LLM chat failed after retries")



EXCL_CODES = ["PT","POP","INT","OUT","DUP","OTHER"]  # publication type/design, population, intervention, outcome, duplicate, other

STRICT_SCREEN_SCHEMA = {
    "label": ["include","exclude","maybe"],
    "conf": float,
    "reason": str,
    "P": bool, "I": bool, "C": bool, "O": bool,
    "design_ok": bool, "pubtype_ok": bool, "year_ok": bool,
    "mesh_hits": list, "salient_terms": list,
    "pris_ref": str,
    "excl_code": str,   # pipe-joined codes e.g. "PT|POP"
}

def _ensure_schema(obj: dict, pris_ref: str, allow_maybe: bool=True) -> dict:
    # defaults
    base = {
        "label": "exclude",
        "conf": 0.0,
        "reason": "unspecified",
        "P": False, "I": False, "C": False, "O": False,
        "design_ok": False, "pubtype_ok": False, "year_ok": False,
        "mesh_hits": [], "salient_terms": [],
        "pris_ref": pris_ref,
        "excl_code": "OTHER",
    }
    base.update({k: obj.get(k, base[k]) for k in base.keys()})
    # clamp fields
    if base["label"] not in (["include","exclude","maybe"] if allow_maybe else ["include","exclude"]):
        base["label"] = "exclude"
    try:
        base["conf"] = float(base["conf"])
    except Exception:
        base["conf"] = 0.0
    # normalize excl_code
    codes = []
    for tok in str(base.get("excl_code","OTHER")).upper().split("|"):
        tok = tok.strip()
        if tok in EXCL_CODES and tok not in codes:
            codes.append(tok)
    if not codes:
        codes = ["OTHER"]
    base["excl_code"] = "|".join(codes)
    return base

def _heuristic_excl_code(reason: str) -> str:
    r = (reason or "").lower()
    hits=[]
    if any(k in r for k in ["case report","protocol","review","letter","editorial","animal","cadaver","simulation","in vitro"]):
        hits.append("PT")
    if any(k in r for k in ["pediatric","neonate","rat","dog","non-thoracic","non-vats","urology","orthopedic"]):
        hits.append("POP")
    if any(k in r for k in ["not espb","wrong block","no block","no comparator","no sapb","not tpvb","epidural only"]):
        hits.append("INT")
    if any(k in r for k in ["no pain","no opioid","outcome not","no ponv","no complication data"]):
        hits.append("OUT")
    if any(k in r for k in ["duplicate","duplication","already included","same cohort"]):
        hits.append("DUP")
    return "|".join(hits) if hits else "OTHER"

def _extract_first_json_object(txt: str):
    import json, re
    if not txt:
        return None
    s = txt.strip()
    # strip code fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    # grab the first {...} block
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    cand = m.group(0)
    # remove trailing commas like  "foo": 1,}
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    try:
        return json.loads(cand)
    except Exception:
        return None

def curate_mesh(proto: Protocol, base_pmids: List[int], universe: List[dict], stage: int=1) -> Dict[str,List[str]]:
    # collect MeSH from key_pmids (stage-1), optionally from stage-1 includes for stage-2
    mesh_terms = []
    base_set = set(base_pmids)
    for rec in universe:
        if rec["pmid"] in base_set:
            mesh_terms.extend(rec.get("mesh", []) or [])
    mesh_terms = list(dict.fromkeys([m for m in mesh_terms if m]))
    
    # If there are no observed MeSH terms to curate, return a deterministic set
    # straight from protocol PICO (no LLM call).
    if not mesh_terms:
        curated = {
            "P": list(dict.fromkeys([t for t in proto.P_terms if t]))[:20],
            "I": list(dict.fromkeys([t for t in proto.I_terms if t]))[:20],
            "C": list(dict.fromkeys([t for t in proto.C_terms if t]))[:20],
            "O": list(dict.fromkeys([t for t in proto.O_terms if t]))[:20],
            "rejected": []
        }
        path = os.path.join(OUTDIR, "mesh_curated.json" if stage==1 else "mesh_curated_stage2.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(curated, f, ensure_ascii=False, indent=2)
        return curated


    sys_prompt = (
        "ROLE: You curate MeSH-like terms into P/I/C/O bins for a systematic review.\n"
        "INPUTS: A base PICO term set and a flat list of MeSH terms collected from key PMIDs.\n"
        "TASK:\n"
        "  1) Normalize, deduplicate, and assign each candidate term to exactly one bin among P, I, C, O.\n"
        "  2) If a term is irrelevant/too generic, put it in 'rejected'.\n"
        "  3) Preserve case and spelling; do not invent novel terms.\n"
        "  4) Return at most 20 terms per P/I/C/O and at most 50 in 'rejected'.\n"
        "OUTPUT:\n"
        "  Return STRICT JSON (no markdown, no commentary) with EXACTLY these keys:\n"
        "'  {\"P\":[],\"I\":[],\"C\":[],\"O\":[],\"rejected\":[]}\n"
        "  Each value is an array of strings.\n"
        "CONSTRAINTS: No extra keys. No trailing comments. JSON must parse."
    )
    user_payload = {
        "narrative_question": proto.narrative_question,
        "protocol_year_range": [proto.year_min, proto.year_max],
        "designs_allowlist": proto.designs_allowlist,
        "pubtype_blocklist": proto.pubtype_blocklist,
        "base_terms": {"P": proto.P_terms, "I": proto.I_terms, "C": proto.C_terms, "O": proto.O_terms},
        "mesh_terms": mesh_terms
    }
    user_prompt = json.dumps(user_payload, ensure_ascii=False)

    txt = llm_chat(
        proto.llm["chat_endpoint"], proto.llm["chat_model"],
        sys_prompt, user_prompt,
        timeout_s=60,               # leave as-is
        max_retries=4,
        log_path=os.path.join(OUTDIR, "screening.log"),
        stream=True,                # <— prevents mid-gen cutoff
        idle_timeout_s=10,          # <— bails only if no bytes for 45s (stalled)
        continuation_retry=1        # <— finish JSON once if cut mid-stream
    )

    # robust parse + sanitize
    curated = {"P": [], "I": [], "C": [], "O": [], "rejected": []}
    parsed = _extract_first_json_object(txt) or {"P":[],"I":[],"C":[],"O":[],"rejected":[]}

    for k in ["P","I","C","O","rejected"]:
        vals = parsed.get(k, [])
        if not isinstance(vals, list):
            vals = []
        # de-dup while preserving order
        seen = set(); kept = []
        for v in vals:
            s = str(v).strip()
            if not s or s in seen: continue
            seen.add(s); kept.append(s)
        curated[k] = kept[:20] if k in ["P","I","C","O"] else kept[:50]

    path = os.path.join(OUTDIR, "mesh_curated.json" if stage==1 else "mesh_curated_stage2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curated, f, ensure_ascii=False, indent=2)
    return curated

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

    sys_prompt = (
        "ROLE: You screen titles/abstracts for a systematic review.\n"
        "DECISION SPACE: label ∈ {include, exclude, maybe}.\n"
        "CRITERIA:\n"
        "  • Apply the protocol details in the provided JSON payload (year range, allowed designs, blocked pubtypes, PICO terms).\n"
        "  • INCLUDE if the abstract clearly matches PICO and allowed study designs.\n"
        "  • EXCLUDE if it clearly violates population/intervention/comparator/outcome/design/pubtype/year.\n"
        "  • MAYBE only if insufficient information is present in TIAB to decide.\n"
        "OUTPUT JSON (STRICT, single object, no extra text):\n"
        "  {\n"
        "    \"label\": \"include|exclude|maybe\",\n"
        "    \"conf\": <float 0..1>,\n"
        "    \"reason\": \"<=180 chars (concise justification)\",\n"
        "    \"P\": <bool>, \"I\": <bool>, \"C\": <bool>, \"O\": <bool>,\n"
        "    \"design_ok\": <bool>, \"pubtype_ok\": <bool>, \"year_ok\": <bool>,\n"
        "    \"mesh_hits\": [], \"salient_terms\": [],\n"
        "    \"pris_ref\": \"screen\",\n"
        "    \"excl_code\": \"PT|POP|INT|OUT|DUP|OTHER\"  // one or pipe-joined; use PT for wrong pubtype/design\n"
        "  }\n"
        "CONSTRAINTS:\n"
        "  • JSON must parse. No markdown. No commentary outside JSON.\n"
        "  • If excluding, set an appropriate excl_code (PT/POP/INT/OUT/DUP/OTHER).\n"
    )
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
            obj = _extract_first_json_object(txt)
            if obj is None:
                obj = {"label":"exclude","conf":0.0,"reason":"parse_error","pris_ref":pris_ref,"excl_code":"OTHER"}

            # enforce schema & backfill excl_code if missing
            if "excl_code" not in obj or not obj["excl_code"]:
                obj["excl_code"] = _heuristic_excl_code(obj.get("reason",""))

            obj = _ensure_schema(obj, pris_ref=pris_ref, allow_maybe=True)
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
    if os.path.exists(c1) and os.path.getsize(c1) > 0:
        frames.append(pd.read_csv(c1))
    if os.path.exists(c2) and os.path.getsize(c2) > 0:
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
                "Year": (int(r["year"]) if (str(r["year"]).isdigit() and 1500 <= int(r["year"]) <= 2100) else ""),
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

def _split_text_for_llm(txt: str, max_chars: int = 3000, overlap: int = 250) -> List[str]:
    txt = (txt or "").strip()
    if len(txt) <= max_chars:
        return [txt]
    chunks = []
    i = 0
    n = len(txt)
    while i < n:
        j = min(i + max_chars, n)
        # try to end at sentence boundary
        k = txt.rfind(".", i+int(0.6*max_chars), j)
        if k == -1: k = j
        chunks.append(txt[i:k].strip())
        i = max(k - overlap, i + max_chars - overlap)
    return [c for c in chunks if c]

def _ft_schema() -> dict:
    return {
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

def _fulltext_chunk_vote(proto: Protocol, record_meta: dict, chunk_text: str, sys_prompt: str, mesh_curated: Optional[dict]=None) -> dict:
    schema = _ft_schema()
    user_payload = {
        "protocol": {
            "year_min": proto.year_min, "year_max": proto.year_max,
            "designs_allowlist": proto.designs_allowlist, "pubtype_blocklist": proto.pubtype_blocklist
        },
        "mesh_curated": (mesh_curated or {"P":[],"I":[],"C":[],"O":[]}),
        "record": record_meta,
        "fulltext_chunk": chunk_text[:3000],  # hard cap
        "schema": schema
    }
    txt = llm_chat(
        proto.llm["chat_endpoint"], proto.llm["chat_model"],
        sys_prompt, json.dumps(user_payload, ensure_ascii=False),
        timeout_s=60, max_retries=4, log_path=os.path.join(OUTDIR, "fulltext.log")
    )
    obj = _extract_first_json_object(txt) or {"label":"exclude","conf":0.0,"reason":"parse_error","excl_code":"OTHER"}
    if "excl_code" not in obj or not obj["excl_code"]:
        obj["excl_code"] = _heuristic_excl_code(obj.get("reason",""))
    return _ensure_schema(obj, pris_ref="fulltext", allow_maybe=False)

def fulltext_screen(proto: Protocol):
    import pandas as pd, glob

    pdf_dir = os.path.join(OUTDIR, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    # unified handoff filename
    handoff = os.path.join(OUTDIR, "final_fulltext_handoff.csv")
    # ensure we call the fetcher exactly once (kept as you had)
    try:
        attempt_oa_downloads(
            handoff_csv_path=handoff,
            output_dir=pdf_dir,
            min_pdf_bytes=1000,
            ncbi_api_key=NCBI_API_KEY,
            contact_email=NCBI_EMAIL
        )
    except TypeError:
        attempt_oa_downloads(handoff, pdf_dir, ncbi_api_key=NCBI_API_KEY)

    # load meta
    df = pd.read_csv(handoff) if os.path.exists(handoff) else pd.DataFrame(columns=["PMID","Year","FirstAuthor","Title","DOI"])
    df["PMID"] = pd.to_numeric(df.get("PMID"), errors="coerce").astype("Int64")
    meta = {
        int(r.PMID): {
            "pmid": int(r.PMID),
            "title": str(r.Title or ""),
            "year": int(r.Year) if str(r.Year).isdigit() else None,
            "first_author": str(r.FirstAuthor or ""),
            "doi": str(r.DOI or "")
        }
        for _, r in df.iterrows() if pd.notna(r.PMID)
    }

    curated = _load_curated_for_fulltext()
    sys_prompt = (
        "ROLE: You judge ONE full-text chunk for SR eligibility.\n"
        "DECISION SPACE: label ∈ {include, exclude}. No 'maybe' at full text.\n"
        "INPUT JSON includes protocol constraints (years, designs, pubtype blocklist), curated P/I/C/O terms, record metadata, and a single chunk of full text.\n"
        "CRITERIA:\n"
        "  • INCLUDE only if the chunk provides enough evidence that the full article matches PICO and allowed designs (or clearly indicates trial/observational with correct outcomes).\n"
        "  • EXCLUDE if the chunk unambiguously shows a violation (wrong population/intervention/comparator/outcome/design/pubtype/year), or if it shows it's a non-eligible pubtype (review/protocol/case, etc.).\n"
        "  • If the chunk is insufficient and prior chunks are unknown, be conservative; prefer EXCLUDE unless inclusion is well supported.\n"
        "OUTPUT JSON (STRICT, single object):\n"
        "  {\n"
        "    \"label\": \"include|exclude\",\n"
        "    \"conf\": <float 0..1>,\n"
        "    \"reason\": \"<=160 chars (concise)\",\n"
        "    \"P\": <bool>, \"I\": <bool>, \"C\": <bool>, \"O\": <bool>,\n"
        "    \"design_ok\": <bool>, \"pubtype_ok\": <bool>, \"year_ok\": <bool>,\n"
        "    \"mesh_hits\": [], \"salient_terms\": [],\n"
        "    \"pris_ref\": \"fulltext\",\n"
        "    \"excl_code\": \"PT|POP|INT|OUT|DUP|OTHER\"\n"
        "  }\n"
        "CONSTRAINTS: JSON only, no extra text; set excl_code when excluding (PT for non-eligible pubtype/design).\n"
    )

    flog = os.path.join(OUTDIR, "fulltext.log")
    _append_lines(flog, [f"# {datetime.now(timezone.utc).isoformat()}Z fulltext run start"])

    results_jsonl = os.path.join(OUTDIR, "fulltext_llm_screen.jsonl")
    out_rows = []

    # extract texts
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    for pdf_path in pdf_paths:
        # try to locate pmid in filename first; else fallback: parse from handoff mapping against Title substring
        fname = os.path.basename(pdf_path)
        pmid = None
        m = re.search(r"(\d{7,9})", fname)
        if m: pmid = int(m.group(1))
        if pmid is None:
            # weak fallback: skip if no pmid found
            continue

        text, method = extract_text_pdf(pdf_path)
        record_meta = meta.get(pmid, {"pmid": pmid, "title": "", "year": None, "first_author": "", "doi": ""})

        chunks = _split_text_for_llm(text, max_chars=3000, overlap=250)
        votes = []
        include_hits = 0
        exclude_hits = 0

        for idx, ch in enumerate(chunks):
            v = _fulltext_chunk_vote(proto, record_meta, ch, sys_prompt, curated)
            # early stop rules (single-article, chunk-wise)
            if v["label"] == "exclude" and v["conf"] >= 0.80 and v["excl_code"] != "OTHER":
                votes.append(v); exclude_hits += 1
                _append_lines(flog, [f"pmid={pmid}\tchunk={idx}\tearly=exclude\tconf={v['conf']:.2f}\tcode={v['excl_code']}"])
                break
            votes.append(v)
            if v["label"] == "include" and v["conf"] >= 0.85:
                include_hits += 1
                if include_hits >= 2:  # require confirmation in another chunk
                    _append_lines(flog, [f"pmid={pmid}\tchunk={idx}\tearly=include\tconf={v['conf']:.2f}"])
                    break
            # hard cap on chunks to avoid stressing small LLMs
            if idx >= 6:  # analyze ≤7 chunks/article
                break

        # aggregate chunk votes
        inc = sum(1 for v in votes if v["label"] == "include")
        exc = sum(1 for v in votes if v["label"] == "exclude")
        if inc > 0 and exc == 0:
            final = max((v for v in votes if v["label"]=="include"), key=lambda x: x["conf"])
        elif exc > 0 and inc == 0:
            final = max((v for v in votes if v["label"]=="exclude"), key=lambda x: x["conf"])
        else:
            # tie-breaker: prefer exclude unless include has higher conf by ≥0.10
            best_inc = max((v for v in votes if v["label"]=="include"), default=None, key=lambda x: x["conf"])
            best_exc = max((v for v in votes if v["label"]=="exclude"), default=None, key=lambda x: x["conf"])
            if best_inc and (not best_exc or best_inc["conf"] >= (best_exc["conf"] + 0.10)):
                final = best_inc
            else:
                final = best_exc or {"label":"exclude","conf":0.0,"reason":"inconclusive","excl_code":"OTHER"}

        # ensure schema & attach extraction method
        final = _ensure_schema(final, pris_ref="fulltext", allow_maybe=False)
        if not final.get("excl_code"):
            final["excl_code"] = _heuristic_excl_code(final.get("reason",""))
        final["extraction"] = method

        # write JSONL (detailed)
        with open(results_jsonl, "a", encoding="utf-8") as jf:
            jf.write(json.dumps({"pmid": pmid, **final}, ensure_ascii=False) + "\n")

        # write tabular summary
        out_rows.append({
            "pmid": pmid,
            "label": final["label"],
            "conf": final["conf"],
            "reason": final["reason"],
            "excl_code": final["excl_code"],
            "extraction": method
        })

        _append_lines(flog, [f"pmid={pmid}\tfinal={final['label']}\tconf={final['conf']:.2f}\tcode={final['excl_code']}\tmethod={method}"])

    write_csv(os.path.join(OUTDIR, "fulltext_screened.csv"), out_rows,
              ["pmid","label","conf","reason","excl_code","extraction"])
    
def _collect_reasons(jsonl_paths: List[str]) -> List[str]:
    reasons = []
    for p in jsonl_paths:
        if not os.path.exists(p): continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if str(obj.get("label","")).lower()=="exclude":
                        reasons.append(str(obj.get("reason","")).strip())
                except Exception:
                    continue
    # dedup while preserving order
    seen=set(); out=[]
    for r in reasons:
        if r and r not in seen:
            seen.add(r); out.append(r)
    return out

def _llm_derive_bins(proto: Protocol, reasons_sample: List[str], max_chars: int=6000) -> Dict[str,str]:
    """
    Phase 1: ask LLM to propose dataset-specific bins and map them to EXCL_CODES.
    Returns a dict like {"Wrong population":"POP", "Unwanted design":"PT", ...}
    """
    sys_prompt = (
        "ROLE: You derive dataset-specific exclusion reason bins and map them to canonical codes.\n"
        "INPUT: A sample list of short exclusion reasons from an SR triage.\n"
        "TASK:\n"
        "  1) Propose a small set (5–15) of dataset-specific category names that cover these reasons.\n"
        "  2) Map each category to ONE canonical code: PT (pubtype/design), POP (population), INT (intervention), OUT (outcome), DUP (duplicate), OTHER.\n"
        "OUTPUT JSON (STRICT): {\"bins\":[{\"name\":\"...\",\"code\":\"PT|POP|INT|OUT|DUP|OTHER\"}, ...]}\n"
        "CONSTRAINTS: JSON only; names concise; codes from the allowed set only."
    )
    buf = []
    used = 0
    for r in reasons_sample:
        s = f"- {r}\n"
        if used + len(s) > max_chars: break
        buf.append(s); used += len(s)
    user = "Sample exclusion reasons:\n" + "".join(buf)

    try:
        txt = llm_chat(proto.llm["chat_endpoint"], proto.llm["chat_model"], sys_prompt, user,
                       timeout_s=45, max_retries=3, log_path=os.path.join(OUTDIR,"screening.log"))
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        j = json.loads(m.group(0))
        mapping = {}
        for b in j.get("bins", []):
            name = (b.get("name") or "").strip()
            code = (b.get("code") or "").strip().upper()
            if name and code in EXCL_CODES:
                mapping[name] = code
        return mapping
    except Exception:
        return {}

def _classify_reason_with_bins(reason: str, name2code: Dict[str,str]) -> str:
    for name, code in name2code.items():
        if name and name.lower() in (reason or "").lower():
            return code
    return _heuristic_excl_code(reason)

def _load_screen_jsonl(path: str) -> List[dict]:
    out=[]
    if not os.path.exists(path): return out
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

def _parse_query_log_counts(path: str) -> Tuple[List[str], List[int]]:
    qs, cs = [], []
    if not os.path.exists(path): return qs, cs
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            if "\t" in ln and "::" in ln and "count=" in ln:
                # lines like: " refined_count=123 :: (query...)"
                m = re.search(r"(?:base_count|refined_count|rescue_count|count)=(\d+)\s+::\s+(.*)$", ln.strip())
                if m:
                    cs.append(int(m.group(1))); qs.append(m.group(2))
    return qs, cs

def prisma_aggregate_and_report(proto: Protocol):
    import pandas as pd, glob
    report = {}

    # Queries and counts
    qlog = os.path.join(OUTDIR, "query_manager.log")
    queries, counts = _parse_query_log_counts(qlog)
    report["queries"] = [{"query": q, "count": c} for q,c in zip(queries, counts)]
    report["queries_total"] = sum(counts)

    # Universe + prefilter
    pre_sum = os.path.join(OUTDIR, "prefilter_summary.csv")
    pre_det = os.path.join(OUTDIR, "prefilter_detail.csv")
    if os.path.exists(pre_sum):
        report["prefilter_summary"] = pd.read_csv(pre_sum).to_dict(orient="records")[0]
    if os.path.exists(pre_det):
        dfpd = pd.read_csv(pre_det)
        report["prefilter_kept"] = int(dfpd["kept"].sum())

    # Stage-1/2 TIAB screens
    s1 = os.path.join(OUTDIR, "s1_llm_screen.jsonl")
    s2 = os.path.join(OUTDIR, "s2_llm_screen.jsonl")
    s1o = _load_screen_jsonl(s1)
    s2o = _load_screen_jsonl(s2)
    def _count_screen(objs):
        inc = sum(1 for o in objs if str(o.get("label","")).lower() in ("include","maybe"))
        exc = sum(1 for o in objs if str(o.get("label","")).lower() == "exclude")
        return inc, exc, len(objs)
    s1_inc, s1_exc, s1_n = _count_screen(s1o)
    s2_inc, s2_exc, s2_n = _count_screen(s2o)
    report["stage1_screen"] = {"n": s1_n, "include_maybe": s1_inc, "exclude": s1_exc}
    report["stage2_screen"] = {"n": s2_n, "include_maybe": s2_inc, "exclude": s2_exc}

    # Citation-reference snowballing (CILE)
    st2_sum = os.path.join(OUTDIR, "stage2_prefilter_summary.json")
    if os.path.exists(st2_sum):
        with open(st2_sum,"r",encoding="utf-8") as f:
            js = json.load(f)
        report["snowballing_identified_raw"] = js.get("n_raw", 0)
        report["snowballing_pass_prefilter"] = js.get("n_pass", 0)

    # Full text fetch & screen
    pdf_count = len(glob.glob(os.path.join(OUTDIR,"pdfs","*.pdf")))
    report["fulltext_pdfs_fetched"] = pdf_count
    ft_csv = os.path.join(OUTDIR, "fulltext_screened.csv")
    if os.path.exists(ft_csv):
        dfft = pd.read_csv(ft_csv)
        report["fulltext_screen_included"] = int((dfft["label"].str.lower()=="include").sum())
        # reason bins
        reasons_all = _collect_reasons([os.path.join(OUTDIR,"fulltext_llm_screen.jsonl")])
        # build bins in ≤6k context
        sample = reasons_all[:200]  # usually fits ≤6k with short lines
        bins_map = _llm_derive_bins(proto, sample, max_chars=6000)
        dfft["bin_code"] = dfft.apply(
            lambda r: (r.get("excl_code") if str(r.get("label","")).lower()=="exclude"
                       else ""), axis=1)
        # fill missing or OTHER by refined mapping
        def _fix(code, reason):
            code = str(code or "").strip().upper()
            if not code or code=="OTHER":
                return _classify_reason_with_bins(reason, bins_map)
            return code
        dfft["bin_code"] = dfft.apply(lambda r: _fix(r["bin_code"], r.get("reason","")), axis=1)
        rep = dfft[dfft["label"].str.lower()=="exclude"]["bin_code"].value_counts().to_dict()
        report["fulltext_excluded_by_reason"] = rep

    # PRISMA core tallies
    report["identification_total"] = report.get("queries_total", 0)
    report["screened_tiab"] = (report.get("stage1_screen",{}).get("n",0)
                               + report.get("stage2_screen",{}).get("n",0))
    report["included_fulltext"] = report.get("fulltext_screen_included", 0)

    # write artifacts
    with open(os.path.join(OUTDIR,"prisma_report.json"),"w",encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # simple markdown summary for humans
    md = []
    md.append("# PRISMA Aggregation\n")
    md.append("## Identification\n")
    md.append(f"- Database queries: {len(report.get('queries',[]))} (total hits sum={report.get('queries_total',0)})\n")
    md.append("## Screening\n")
    md.append(f"- Stage-1 TIAB: n={s1_n}, include/maybe={s1_inc}, exclude={s1_exc}\n")
    md.append(f"- Stage-2 TIAB (snowballing): n={s2_n}, include/maybe={s2_inc}, exclude={s2_exc}\n")
    md.append("## Eligibility (full text)\n")
    md.append(f"- PDFs fetched: {pdf_count}\n")
    md.append(f"- Included after full-text: {report.get('fulltext_screen_included',0)}\n")
    if report.get("fulltext_excluded_by_reason"):
        md.append("### Excluded by reason (full-text)\n")
        for k,v in report["fulltext_excluded_by_reason"].items():
            md.append(f"- {k}: {v}\n")
    md.append("## Snowballing (Citation-Reference)\n")
    if "snowballing_identified_raw" in report:
        md.append(f"- Identified via CILE: {report['snowballing_identified_raw']} (prefilter pass={report.get('snowballing_pass_prefilter',0)})\n")
    with open(os.path.join(OUTDIR,"prisma_report.md"),"w",encoding="utf-8") as f:
        f.write("\n".join(md))

def glm_similarity_vs_inclusion():
    """
    Fits GLMs (logit) for Stage-1 and Stage-2 TIAB:
      outcome = 1 if included/maybe, 0 if exclude
      predictors = z(score_tfidf), z(score_emb), z(score_mesh), z(rrf)
    Writes triage_stats.json with coefficients & (if available) p-values.
    """
    import pandas as pd, numpy as np
    try:
        import statsmodels.api as sm
        have_sm = True
    except Exception:
        have_sm = False
        from sklearn.linear_model import LogisticRegression

    stats = {}

    def _one(stage_csv: str, screen_jsonl: str, key="stage1"):
        p_csv = os.path.join(OUTDIR, stage_csv)
        p_jsn = os.path.join(OUTDIR, screen_jsonl)
        if not (os.path.exists(p_csv) and os.path.exists(p_jsn)):
            return
        X = pd.read_csv(p_csv)
        # build labels
        y_map = {}
        with open(p_jsn,"r",encoding="utf-8") as f:
            for ln in f:
                try:
                    o = json.loads(ln)
                    pmid = int(o.get("pmid", o.get("record",{}).get("pmid", -1)))
                    lab = str(o.get("label","")).lower()
                    y_map[pmid] = 1 if lab in ("include","maybe") else 0
                except:
                    pass
        X["y"] = X["pmid"].map(y_map)
        X = X.dropna(subset=["y"])
        if X.empty: return

        # numeric features
        for c in ["score_tfidf","score_emb","score_mesh","rrf"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.dropna(subset=["score_tfidf","score_emb","score_mesh","rrf"])
        if X.empty: return
        # z-scale
        for c in ["score_tfidf","score_emb","score_mesh","rrf"]:
            m, s = X[c].mean(), X[c].std() or 1.0
            X[c+"_z"] = (X[c]-m)/s

        feats = ["score_tfidf_z","score_emb_z","score_mesh_z","rrf_z"]

        if have_sm:
            Xmat = sm.add_constant(X[feats])
            model = sm.GLM(X["y"], Xmat, family=sm.families.Binomial())
            res = model.fit()
            stats[key] = {
                "n": int(X.shape[0]),
                "coef": {k: float(res.params.get(k, float("nan"))) for k in ["const"]+feats},
                "pval": {k: float(res.pvalues.get(k, float("nan"))) for k in ["const"]+feats},
                "library": "statsmodels"
            }
        else:
            # fallback: sklearn logistic regression (no p-values)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(X[feats], X["y"])
            coefs = dict(zip(feats, [float(x) for x in lr.coef_[0]]))
            stats[key] = {
                "n": int(X.shape[0]),
                "coef": {"const": float(lr.intercept_[0]), **coefs},
                "pval": {k: None for k in ["const"]+feats},
                "library": "sklearn_logistic"
            }

    _one("triage_stage1_candidates.csv","s1_llm_screen.jsonl","stage1")
    _one("triage_stage2_candidates.csv","s2_llm_screen.jsonl","stage2")

    with open(os.path.join(OUTDIR,"triage_stats.json"),"w",encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


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
    root_log.info("[CILE] Running CILE (external module) for stage-2 candidates…")

    # ---- Build seed PMIDs for CILE (robust + deterministic) ----
    import csv as _csv

    seeds: List[int] = []
    try:
        # Preferred: take top items from stage-1 triage RRF ranking
        _cand_csv = os.path.join(OUTDIR, "triage_stage1_candidates.csv")
        if os.path.exists(_cand_csv):
            with open(_cand_csv, newline='', encoding='utf-8') as _f:
                _rows = list(_csv.DictReader(_f))
            # sort by ascending rank_rrf (1 is best); allow floaty strings like "1.0"
            def _rank_int_safe(v) -> int:
                s = str(v).strip()
                if s.replace(".", "", 1).isdigit():
                    try:
                        return int(float(s))
                    except Exception:
                        return 10**9
                return 10**9
            _rows = [r for r in _rows if r.get("pmid") and str(r["pmid"]).isdigit()]
            _rows.sort(key=lambda r: (_rank_int_safe(r.get("rank_rrf")), -int(r.get("year") or 0)))
            seeds = [int(r["pmid"]) for r in _rows[:1200]]

        # Fallback: protocol key_pmids
        if not seeds and getattr(proto, "key_pmids", None):
            seeds = [int(p) for p in proto.key_pmids[:1200] if str(p).isdigit()]

        # Last resort: first kept PMIDs from prefilter detail
        if not seeds:
            _pre_csv = os.path.join(OUTDIR, "prefilter_detail.csv")
            if os.path.exists(_pre_csv):
                with open(_pre_csv, newline='', encoding="utf-8") as _f:
                    _rows = list(_csv.DictReader(_f))
                _kept = [int(r["pmid"]) for r in _rows if str(r.get("pmid","")).isdigit() and str(r.get("kept","")).lower()=="true"]
                seeds = _kept[:1200]

        if not seeds:
            root_log.warning("[CILE] No seeds found from triage/protocol; proceeding with an empty seed list (allowed but not ideal).")
    except Exception:
        root_log.exception("[CILE] Failed to build seeds; continuing with an empty list.")
        seeds = []

    # ---- Run CILE (external module) ----
    Hf, Af, meta = cile.outer_loop_cile(seeds, cile.OuterConfig(
        accept_policy="elastic_phi",
        # make the H neighborhood MUCH bigger (match “original PPR” feel)
        per_node_cap=500,
        # broaden/disable topic-agnostic gates to avoid tiny A:
        min_relevance_frac=0.00,
        per_node_ext_frac_cap=0.85,
        H_external_budget=None,           # or a larger number if you want a cap
        max_accept_after_filter=1200,     # since you seed with up to 1200 now
        deterministic_reservoir=True,
        quarantine_hubs=True,
        quarantine_mode="external",
    ))



    # Persist small artifacts
    try:
        with open(os.path.join(OUTDIR, "cile_meta.json"), "w", encoding="utf-8") as _fw:
            json.dump(meta, _fw, ensure_ascii=False, indent=2)
        _A_pmids = [Hf.pmids[i] for i in sorted(list(Af))]
        with open(os.path.join(OUTDIR, "cile_A_pmids.txt"), "w", encoding="utf-8") as _fw:
            _fw.write("\n".join(str(p) for p in _A_pmids))
        root_log.info("[CILE] Wrote cile_meta.json and cile_A_pmids.txt")
    except Exception:
        root_log.warning("[CILE] Could not write cile_meta.json / cile_A_pmids.txt (non-fatal)")

    # --- Define stage2_pmids deterministically (A-set minus stage-1 universe/kept) ---
    stage1_kept_pmids = {int(r["pmid"]) for r in (kept or []) if r.get("pmid") is not None}
    stage2_all_pmids = [Hf.pmids[i] for i in sorted(list(Af))]
    stage2_pmids = [p for p in stage2_all_pmids if p not in stage1_kept_pmids]


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
        
    # Write a detail CSV for every raw stage-2 candidate with pass/fail reasons
    detail_rows = []
    for r in raw2:
        # use the same flags the code already attaches for kept ones
        ok, flags = prefilter_record(r, proto.year_min, proto.year_max,
                                    set(pt.lower() for pt in proto.pubtype_blocklist),
                                    set(pt.lower() for pt in proto.designs_allowlist))
        detail_rows.append({
            "pmid": r.get("pmid"),
            "year": r.get("year"),
            "pubtypes": ";".join(r.get("pubtypes", [])),
            "ok": bool(ok),
            "year_ok": bool(flags.get("year_ok")),
            "pubtype_ok": bool(flags.get("pubtype_ok")),
            "design_ok": bool(flags.get("design_ok")),
        })

    # write CSV (header inferred)
    if detail_rows:
        write_csv(os.path.join(OUTDIR, "stage2_prefilter_detail.csv"),
                detail_rows,
                list(detail_rows[0].keys()))

    # tiny JSON summary
    from collections import Counter
    summary = {
        "n_raw": len(raw2),
        "n_pass": sum(1 for d in detail_rows if d["ok"]),
        "fail_breakdown": dict(Counter(
            reason
            for d in detail_rows if not d["ok"]
            for reason in (["year"] if not d["year_ok"] else [])
                    + (["pubtype"] if not d["pubtype_ok"] else [])
                    + (["design"] if not d["design_ok"] else [])
        ))
    }
    with open(os.path.join(OUTDIR, "stage2_prefilter_summary.json"), "w", encoding="utf-8") as _fw:
        json.dump(summary, _fw, ensure_ascii=False, indent=2)


    # 8) Stage-2 MeSH curation augmentation (iterative) & ranking + screening
    if stage2_recs:
        root_log.info("[MeSH] Stage-2 iterative curation from stage-1 includes…")
        import pandas as _pd
        s1inc_csv = os.path.join(OUTDIR, "stage1_included.csv")
        _s1_pmids = []

        try:
            if os.path.exists(s1inc_csv) and os.path.getsize(s1inc_csv) > 0:
                _s1 = _pd.read_csv(s1inc_csv)
                if "pmid" in _s1.columns:
                    _s1_pmids = [int(x) for x in _s1["pmid"].astype(str) if x.isdigit()]
                elif len(_s1.columns) > 0:
                    # very defensive: take first column if header changed
                    _s1_pmids = [int(x) for x in _s1.iloc[:,0].astype(str) if x.isdigit()]
        except Exception as e:
            root_log.warning(f"[Stage2] Could not read {s1inc_csv}: {e!s} — proceeding without S1 includes.")

        if not _s1_pmids:
            root_log.info("[Stage2] No stage-1 includes — using protocol base P/I/C/O for stage-2 curation.")

        curated2 = curate_mesh(proto, _s1_pmids, kept + stage2_recs, stage=2)

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

    # 11) PRISMA aggregation + stats
    prisma_aggregate_and_report(proto)
    glm_similarity_vs_inclusion()

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
