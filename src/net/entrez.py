from __future__ import annotations
import requests, re, time, math, concurrent.futures, xml.etree.ElementTree as ET
from typing import List, Dict, Any, Iterable, Optional, Tuple
import logging
from src.config.defaults import ENTREZ_EMAIL, ENTREZ_API_KEY, HTTP_TIMEOUT, USER_AGENT
from src.io.docdb import DocDB

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
log = logging.getLogger("entrez")

def esearch(query: str, db: str = "pubmed", retmax: int = 10000, mindate: Optional[int]=None, maxdate: Optional[int]=None, sort: str="date") -> List[str]:
    params = {"db": db, "term": query, "retmode": "json", "retmax": retmax, "sort": sort, "email": ENTREZ_EMAIL}
    if ENTREZ_API_KEY: params["api_key"] = ENTREZ_API_KEY
    if mindate: params["mindate"] = str(mindate)
    if maxdate: params["maxdate"] = str(maxdate)
    r = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])

def _parse_pubmed_xml(xml_text: str) -> Dict[str, Dict[str,Any]]:
    out: Dict[str,Dict[str,Any]] = {}
    root = ET.fromstring(xml_text)
    def _join(node) -> str:
        if node is None: return ""
        try: return "".join(node.itertext())
        except Exception: return (getattr(node, "text", None) or "")
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = _join(art.find(".//ArticleTitle")).strip()
        abs_nodes = art.findall(".//Abstract/AbstractText")
        abstract = " ".join(_join(n).strip() for n in abs_nodes) if abs_nodes else ""
        year = None
        for path in (".//ArticleDate/Year",".//PubDate/Year",".//DateCreated/Year",".//PubDate/MedlineDate"):
            s = art.findtext(path)
            if s:
                m = re.search(r"\d{4}", s)
                if m: year = int(m.group(0)); break
        journal = art.findtext(".//Journal/Title") or ""
        pubtypes = [pt.text for pt in art.findall(".//PublicationTypeList/PublicationType") if pt.text]
        doi = None
        for idn in art.findall(".//ArticleIdList/ArticleId"):
            if (idn.attrib.get("IdType","").lower()=="doi") and idn.text:
                doi = idn.text.strip().lower()
        lang = art.findtext(".//Language") or None
        out[pmid] = {"pmid": pmid, "title": title, "abstract": abstract, "year": year,
                     "journal": journal, "pub_types": pubtypes, "doi": doi, "language": lang}
    return out

def _efetch_chunk(chunk: List[str]) -> Dict[str, Dict[str,Any]]:
    params = {"db":"pubmed", "retmode":"xml", "rettype":"abstract", "id":",".join(chunk), "email":ENTREZ_EMAIL}
    if ENTREZ_API_KEY: params["api_key"] = ENTREZ_API_KEY
    r = requests.get(f"{EUTILS}/efetch.fcgi", headers={"User-Agent":USER_AGENT}, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return _parse_pubmed_xml(r.text)

def efetch_abstracts(pmids: Iterable[str], chunk_size: int = 200, workers: int = 3, use_cache: bool = True) -> Dict[str, Dict[str,Any]]:
    """Polite parallel efetch with caching + progress logs."""
    pmids = [str(p) for p in pmids]
    if not pmids: return {}

    cache = DocDB() if use_cache else None
    have: Dict[str,Dict[str,Any]] = {}
    todo = pmids

    if cache:
        cached = cache.get_many(pmids)
        if cached:
            for k, v in cached.items():
                have[k] = v
            todo = [p for p in pmids if p not in cached]

    log.info(f"efetch: total={len(pmids)} (cache hit={len(have)}, miss={len(todo)}), chunk={chunk_size}, workers={workers}")

    if not todo:
        return have

    chunks = [todo[i:i+chunk_size] for i in range(0, len(todo), chunk_size)]
    # Courtesy: limit concurrency; add slight pacing between submissions to avoid burst.
    results: Dict[str,Dict[str,Any]] = {}
    submitted = 0
    t0 = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for ch in chunks:
            futs.append(ex.submit(_efetch_chunk, ch))
            submitted += 1
            time.sleep(0.10)  # tiny stagger

        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                part = fut.result()
                results.update(part)
            except Exception as e:
                log.warning(f"efetch chunk failed: {e}")
            done += 1
            if done % 3 == 0 or done == len(chunks):
                dt = time.monotonic() - t0
                log.info(f"efetch progress {done}/{len(chunks)} chunks | items={sum(len(c) for c in chunks[:done])}/{len(todo)} | {dt:.1f}s")

            # polite global pacing (aim ~3-5 req/s overall)
            time.sleep(0.05)

    # merge + persist
    have.update(results)
    if cache and results:
        cache.put_many(results)

    return have
