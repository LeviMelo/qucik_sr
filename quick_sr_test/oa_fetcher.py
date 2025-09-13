# oa_fetcher.py
# Library-only OA PDF downloader (Unpaywall + PMC fallback). No import-time side effects.
from __future__ import annotations

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import os
import re
import io
import csv
import json
import time
import logging
import hashlib

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)  # handlers configured by host app

# Env defaults (read at import; callers can pass overrides at runtime)
MY_EMAIL_FOR_APIS = os.getenv("NCBI_EMAIL") or None
MY_NCBI_API_KEY = os.getenv("NCBI_API_KEY") or None

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HTTP_TIMEOUT = 30


# --------------------------
# Helpers
# --------------------------
def _make_session(email_for_ua: Optional[str], product_tag: str) -> requests.Session:
    s = requests.Session()
    ua_email = (email_for_ua or "unknown@example.org")
    s.headers.update({
        "User-Agent": f"{product_tag} (+mailto:{ua_email})",
        "Accept": "*/*",
    })
    s.max_redirects = 5
    return s


def sanitize_filename_component(s: str, maxlen: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    if not s:
        s = "NA"
    s = s[:maxlen]
    # Windows reserved basenames guard
    reserved = {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}
    if s.upper() in reserved:
        s = s + "_"
    return s


def deterministic_pdf_name(year: Optional[int], first_author: str, pmid: int) -> str:
    yr = ""
    try:
        if year is not None:
            y = int(year)
            if 1500 <= y <= 2100:
                yr = f"{y:04d}"
    except Exception:
        yr = ""
    author_stub = sanitize_filename_component((first_author or "NA").split()[0] or "NA", maxlen=40)
    return f"{yr}_{author_stub}_{pmid}.pdf"


def _read_handoff_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _is_likely_pdf(resp: requests.Response, url: str, allow_octet_stream: bool = True) -> bool:
    ct = resp.headers.get("Content-Type", "").lower()
    if "application/pdf" in ct:
        return True
    if allow_octet_stream and "application/octet-stream" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return False


def _first_k_bytes_is_pdf(buf: bytes) -> bool:
    # '%PDF' header
    return len(buf) >= 4 and buf[:4] == b"%PDF"


# --------------------------
# Unpaywall + PMC
# --------------------------
def get_unpaywall_json(doi: str, session: requests.Session, contact_email: Optional[str]) -> Optional[dict]:
    if not doi:
        return None
    email = contact_email or MY_EMAIL_FOR_APIS or "unknown@example.org"
    url = f"{UNPAYWALL_BASE}/{requests.utils.quote(doi)}"
    params = {"email": email}
    backoff = 0.8
    for _ in range(5):
        try:
            r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
        except Exception as e:
            logger.warning("Unpaywall transport error: %s; backoff=%.2fs", e.__class__.__name__, backoff)
            time.sleep(backoff); backoff *= 1.7
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff); backoff *= 1.7
            continue
        return None
    return None


def try_direct_pdf(url: str, session: requests.Session, min_bytes: int) -> Tuple[bool, Optional[bytes], str, str]:
    if not url:
        return (False, None, "", "no_url")
    backoff = 0.6
    for _ in range(4):
        try:
            r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=True)
        except Exception as e:
            logger.debug("Direct PDF transport: %s", e)
            time.sleep(backoff); backoff *= 1.7
            continue
        final_url = r.url or url
        if r.status_code != 200:
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 1.7
                continue
            return (False, None, final_url, f"status_{r.status_code}")
        if not _is_likely_pdf(r, final_url, allow_octet_stream=True):
            return (False, None, final_url, "not_pdf_mime")
        try:
            # Read first chunk to validate header; then full if small
            buf = r.content
            if not _first_k_bytes_is_pdf(buf[:8]):
                return (False, None, final_url, "no_pdf_magic")
            if len(buf) < min_bytes:
                return (False, None, final_url, "too_small")
            return (True, buf, final_url, "ok")
        except Exception as e:
            return (False, None, final_url, f"error_{e.__class__.__name__}")
    return (False, None, url, "retry_exhausted")


def pmid_to_pmcid(pmid: int, session: requests.Session, ncbi_api_key: Optional[str], contact_email: Optional[str]) -> Optional[str]:
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": str(int(pmid)),
        "retmode": "json",
        "tool": "oa_fetcher",
        "email": (contact_email or MY_EMAIL_FOR_APIS or "unknown@example.org"),
    }
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key
    url = f"{EUTILS_BASE}/elink.fcgi"
    backoff = 0.7
    for _ in range(5):
        try:
            r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
        except Exception as e:
            logger.debug("eLink transport: %s", e)
            time.sleep(backoff); backoff *= 1.7
            continue
        if r.status_code == 200:
            try:
                j = r.json()
            except Exception:
                return None
            try:
                linksets = j.get("linksets") or j.get("linkset") or []
                if not linksets:
                    return None
                linkset = linksets[0]
                links = linkset.get("linksetdbs", [])
                for ls in links:
                    if ls.get("linkname") == "pubmed_pmc" or ls.get("dbto") == "pmc":
                        ids = ls.get("links") or ls.get("link") or []
                        if ids:
                            pmcid = ids[0].get("id") or ids[0]
                            if isinstance(pmcid, str) and pmcid:
                                if pmcid.upper().startswith("PMC"):
                                    return pmcid.upper()
                                return "PMC" + str(pmcid)
            except Exception:
                return None
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff); backoff *= 1.7
            continue
        return None
    return None


def download_from_pmc(pmcid: str, session: requests.Session, min_bytes: int) -> Tuple[bool, Optional[bytes], str, str]:
    """Try a few canonical PMC PDF URLs, fallback to parse HTML for a pdf link."""
    candidates = []
    base = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}"
    candidates.append(base + "/pdf")
    candidates.append(base + "/pdf/")
    candidates.append(base + "/pdf/" + pmcid + ".pdf")

    backoff = 0.6
    for _ in range(3):
        for url in candidates:
            try:
                r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=True)
            except Exception as e:
                logger.debug("PMC transport: %s", e)
                continue
            if r.status_code != 200:
                continue
            final_url = r.url or url
            if _is_likely_pdf(r, final_url, allow_octet_stream=True):
                buf = r.content
                if not _first_k_bytes_is_pdf(buf[:8]):
                    continue
                if len(buf) < min_bytes:
                    return (False, None, final_url, "too_small")
                return (True, buf, final_url, "ok_pdf")
        time.sleep(backoff); backoff *= 1.6

    # Fallback: parse HTML and search for links with "pdf"
    try:
        r = session.get(base, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "pdf" in href.lower():
                    url = href if href.startswith("http") else ("https://www.ncbi.nlm.nih.gov" + href)
                    ok, buf, final, msg = try_direct_pdf(url, session, min_bytes)
                    if ok:
                        return (True, buf, final, "ok_pdf_link")
    except Exception:
        pass
    return (False, None, base, "no_pdf_found")


# --------------------------
# Public API
# --------------------------
def attempt_oa_downloads(
    handoff_csv_path: str,
    output_dir: str,
    min_pdf_bytes: int = 1000,
    ncbi_api_key: Optional[str] = None,
    contact_email: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Read a handoff CSV (PMID, Year, FirstAuthor, Title, DOI) and download PDFs deterministically.
    - Tries Unpaywall best_oa_location first (url_for_pdf / url).
    - Falls back to PMC via eLink (PMID→PMCID) and PMC PDF endpoints.
    - Writes files as YEAR_FirstAuthorStub_PMID.pdf
    - Returns a per-PMID status map {pmid: {"status": "...", "source": "unpaywall|pmc|none"}}
    """
    # Run-time override for API key/email
    global MY_NCBI_API_KEY, MY_EMAIL_FOR_APIS
    if ncbi_api_key:
        MY_NCBI_API_KEY = ncbi_api_key
    if contact_email:
        MY_EMAIL_FOR_APIS = contact_email

    os.makedirs(output_dir, exist_ok=True)

    http_session = _make_session(MY_EMAIL_FOR_APIS, "OA-Fetch/1.0")
    pmc_session = _make_session(MY_EMAIL_FOR_APIS, "OA-Fetch/PMC/1.0")
    ncbi_session = _make_session(MY_EMAIL_FOR_APIS, "OA-Fetch/NCBI/1.0")

    rows = _read_handoff_csv(handoff_csv_path)
    results: Dict[str, Dict[str, str]] = {}

    for row in rows:
        try:
            pmid = int(row.get("PMID"))
        except Exception:
            continue
        title = (row.get("Title") or "").replace("\t", " ").replace("\n", " ").strip()
        fa = (row.get("FirstAuthor") or "").strip()
        year = None
        yraw = row.get("Year")
        try:
            yv = int(yraw)
            if 1500 <= yv <= 2100:
                year = yv
        except Exception:
            year = None

        doi = row.get("DOI")
        if isinstance(doi, str) and doi.strip().lower() in {"", "nan", "none", "null"}:
            doi = None

        pdf_name = deterministic_pdf_name(year, fa, pmid)
        pdf_path = os.path.join(output_dir, pdf_name)

        # Skip if already present and large enough
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) >= min_pdf_bytes:
            results[str(pmid)] = {"status": "exists", "source": "existing"}
            continue

        # ---------- Try Unpaywall ----------
        source = "none"
        status = "init"
        if doi:
            j = get_unpaywall_json(doi, http_session, MY_EMAIL_FOR_APIS)
            if j:
                loc = j.get("best_oa_location") or {}
                url_pdf = loc.get("url_for_pdf") or loc.get("url") or ""
                if url_pdf:
                    ok, buf, final_url, msg = try_direct_pdf(url_pdf, http_session, min_pdf_bytes)
                    if ok and buf:
                        with open(pdf_path, "wb") as f:
                            f.write(buf)
                        results[str(pmid)] = {"status": "ok", "source": "unpaywall", "url": final_url}
                        continue
                    else:
                        status = f"unpaywall_{msg}"

        # ---------- Fallback: PMC ----------
        pmcid = pmid_to_pmcid(pmid, ncbi_session, MY_NCBI_API_KEY, MY_EMAIL_FOR_APIS)
        if pmcid:
            ok, buf, final_url, msg = download_from_pmc(pmcid, pmc_session, min_pdf_bytes)
            if ok and buf:
                with open(pdf_path, "wb") as f:
                    f.write(buf)
                results[str(pmid)] = {"status": "ok", "source": "pmc", "url": final_url}
                continue
            else:
                source = "pmc"
                status = f"pmc_{msg}"

        # ---------- Give up ----------
        results[str(pmid)] = {"status": status, "source": source}
    return results
