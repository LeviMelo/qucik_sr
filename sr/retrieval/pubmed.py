# sr/retrieval/pubmed.py
from __future__ import annotations
import requests, time, xml.etree.ElementTree as ET, re
from typing import Dict, List, Optional, Any, Iterable
from sr.config.defaults import ENTREZ_EMAIL, ENTREZ_API_KEY, HTTP_TIMEOUT, USER_AGENT, RETRIEVAL_PAGE_SIZE, RETRIEVAL_MAX_PAGES
from sr.config.schema import Record

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

def esearch_paged(query: str, mindate: Optional[int]=None, maxdate: Optional[int]=None, db: str="pubmed") -> List[str]:
    """Return a deduped list of PMIDs across pages. Gracefully handle 0 results and non-JSON error bodies."""
    ids: List[str] = []
    retmax = RETRIEVAL_PAGE_SIZE
    params = {"db": db, "retmode": "json", "term": query, "retmax": retmax, "email": ENTREZ_EMAIL, "usehistory": "y"}
    if ENTREZ_API_KEY: params["api_key"] = ENTREZ_API_KEY
    if mindate: params["mindate"] = str(mindate)
    if maxdate: params["maxdate"] = str(maxdate)

    r = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    try:
        js = r.json().get("esearchresult", {})
    except Exception:
        # Non-JSON, likely an HTML error page due to malformed query
        return []
    count = int(js.get("count", "0"))
    webenv = js.get("webenv", None)
    query_key = js.get("querykey", None)

    if count == 0 or not webenv or not query_key:
        return []

    fetched = 0
    for page in range(RETRIEVAL_MAX_PAGES):
        retstart = page * retmax
        if retstart >= count:
            break
        p = {
            "db": db, "retmode": "json", "retmax": retmax, "retstart": retstart,
            "email": ENTREZ_EMAIL, "query_key": query_key, "WebEnv": webenv
        }
        if ENTREZ_API_KEY: p["api_key"] = ENTREZ_API_KEY
        r2 = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params=p, timeout=HTTP_TIMEOUT)
        r2.raise_for_status()
        try:
            js2 = r2.json().get("esearchresult", {})
        except Exception:
            # If a subsequent page returns non-JSON, stop paging and use what we have.
            break
        batch = js2.get("idlist", [])
        ids.extend([str(x) for x in batch if x])
        fetched += len(batch)
        time.sleep(0.08)
        if fetched >= count:
            break

    # dedupe, preserve order
    seen=set(); out=[]
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

# in sr/retrieval/pubmed.py — replace _parse_pubmed_xml

def _parse_pubmed_xml(xml_text: str) -> Dict[str, Dict[str,Any]]:
    out: Dict[str,Dict[str,Any]] = {}
    root = ET.fromstring(xml_text)

    def _join(node) -> str:
        if node is None: return ""
        try:
            return "".join(node.itertext())
        except Exception:
            return (getattr(node, "text", None) or "")

    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = _join(art.find(".//ArticleTitle")).strip()
        abs_nodes = art.findall(".//Abstract/AbstractText")
        abstract = " ".join(_join(n).strip() for n in abs_nodes) if abs_nodes else ""

        year = None
        for path in (".//ArticleDate/Year", ".//PubDate/Year", ".//DateCreated/Year", ".//PubDate/MedlineDate"):
            s = art.findtext(path)
            if s:
                m = re.search(r"\d{4}", s)
                if m:
                    year = int(m.group(0))
                    break

        lang = art.findtext(".//Language") or None
        pubtypes = [pt.text for pt in art.findall(".//PublicationTypeList/PublicationType") if pt.text]

        # DOI
        doi = None
        for idn in art.findall(".//ArticleIdList/ArticleId"):
            if (idn.attrib.get("IdType","").lower() == "doi") and idn.text:
                doi = idn.text.strip().lower()

        # MeSH
        mesh_headings: list[str] = []
        mesh_major: list[str] = []
        for mh in art.findall(".//MeshHeadingList/MeshHeading"):
            desc = mh.find("DescriptorName")
            desc_txt = (desc.text or "").strip() if desc is not None and desc.text else ""
            desc_major = (desc.attrib.get("MajorTopicYN","N") == "Y") if desc is not None else False
            if desc_txt:
                mesh_headings.append(desc_txt)
                if desc_major:
                    mesh_major.append(desc_txt)
            # qualifiers (append as "Descriptor/Qualifier" if descriptor exists; otherwise just qualifier)
            for q in mh.findall("QualifierName"):
                q_txt = (q.text or "").strip()
                if not q_txt:
                    continue
                combo = f"{desc_txt}/{q_txt}" if desc_txt else q_txt
                mesh_headings.append(combo)
                if q.attrib.get("MajorTopicYN","N") == "Y":
                    mesh_major.append(combo)

        out[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "language": lang,
            "publication_types": pubtypes,
            "doi": doi,
            "source": "retrieval",
            "mesh_headings": mesh_headings,
            "mesh_major": mesh_major,
        }
    return out

def efetch_abstracts(pmids: Iterable[str], chunk: int = 200, workers: int = 1) -> Dict[str, Dict[str,Any]]:
    # Simple sequential to reduce edge cases; you can parallelize later.
    pmids = [str(p) for p in pmids if p]
    if not pmids: return {}
    results: Dict[str,Dict[str,Any]] = {}
    for i in range(0, len(pmids), chunk):
        sub = pmids[i:i+chunk]
        params = {"db":"pubmed", "retmode":"xml", "rettype":"abstract", "id":",".join(sub), "email":ENTREZ_EMAIL}
        if ENTREZ_API_KEY: params["api_key"] = ENTREZ_API_KEY
        r = requests.get(f"{EUTILS}/efetch.fcgi", headers={"User-Agent": USER_AGENT}, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        results.update(_parse_pubmed_xml(r.text))
        time.sleep(0.08)
    return results

def to_records(raw: Dict[str, Dict[str,Any]]) -> List[Record]:
    out: List[Record] = []
    for p, rec in raw.items():
        try:
            out.append(Record(**rec))
        except Exception:
            # skip malformed
            continue
    return out
