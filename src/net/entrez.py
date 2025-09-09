from __future__ import annotations
import requests, re, time, concurrent.futures, xml.etree.ElementTree as ET
from typing import List, Dict, Any, Iterable, Optional
import logging
from src.config.defaults import ENTREZ_EMAIL, ENTREZ_API_KEY, HTTP_TIMEOUT, USER_AGENT

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
log = logging.getLogger("entrez")

_LANG_MAP = {
    "eng": "English", "en": "English", "english": "English",
    "spa": "Spanish", "es": "Spanish", "spanish": "Spanish",
    "por": "Portuguese", "pt": "Portuguese", "portuguese": "Portuguese",
    "fra": "French", "fre": "French", "fr": "French", "french": "French",
    "deu": "German", "ger":"German", "de": "German", "german": "German",
}

def _norm_lang_name(s: str | None) -> str | None:
    if not s: return None
    key = s.strip().lower()
    return _LANG_MAP.get(key, s)

def esearch(query: str, db: str = "pubmed", retmax: int = 10000, mindate: Optional[int]=None, maxdate: Optional[int]=None, sort: str="date") -> Dict[str, Any]:
    params = {"db": db, "term": query, "retmode": "json", "retmax": retmax, "sort": sort, "email": ENTREZ_EMAIL}
    if ENTREZ_API_KEY: params["api_key"] = ENTREZ_API_KEY
    if mindate: params["mindate"] = str(mindate)
    if maxdate: params["maxdate"] = str(maxdate)
    r = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json().get("esearchresult", {})
    ids = js.get("idlist", [])
    count = int(js.get("count", "0"))
    translation = js.get("querytranslation") or ""
    return {"ids": ids, "count": count, "translation": translation}

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
        lang_code = art.findtext(".//Language")
        lang = _norm_lang_name(lang_code)
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
    pmids = [str(p) for p in pmids if p]
    if not pmids: return {}
    results: Dict[str,Dict[str,Any]] = {}
    # simple polite parallel fetch
    chunks = [pmids[i:i+chunk_size] for i in range(0, len(pmids), chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_efetch_chunk, ch) for ch in chunks]
        for fut in concurrent.futures.as_completed(futs):
            try:
                results.update(fut.result())
            except Exception as e:
                log.warning(f"efetch chunk failed: {e}")
            time.sleep(0.08)
    return results
