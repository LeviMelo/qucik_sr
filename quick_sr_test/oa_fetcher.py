# oa_fetcher.py
# Open Access PDF downloader (Unpaywall → PMC with PoW) with streaming writes.
# Drop-in replacement for your existing module. No import-time side effects.

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import os
import re
import csv
import time
import logging
import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("oa_fetcher")  # parent script can add handlers/levels

# --------------------------
# Environment + constants
# --------------------------
MY_EMAIL_FOR_APIS = os.getenv("NCBI_EMAIL") or None
MY_NCBI_API_KEY   = os.getenv("NCBI_API_KEY") or None

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
EUTILS_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HTTP_TIMEOUT   = 30

# Browser-like headers for PMC
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SCIHUB_DOMAINS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru"
]

# --------------------------
# Helpers
# --------------------------
def _make_session(user_agent: str, product_tag: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    ua_email = (MY_EMAIL_FOR_APIS or "unknown@example.org")
    # If product_tag is given, prepend; otherwise treat as a browser-like UA
    if product_tag:
        ua = f"{product_tag} (+mailto:{ua_email})"
    else:
        ua = user_agent
    s.headers.update({
        "User-Agent": ua,
        "Accept": "*/*",
    })
    s.max_redirects = 5
    return s

def _read_handoff_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

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
    # Keep this EXACT pattern so fulltext_screen() can parse with its regex
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
    return len(buf) >= 4 and buf[:4] == b"%PDF"

# --------------------------
# Streaming download
# --------------------------
def _download_streaming(url: str, session: requests.Session, out_path: str,
                        min_bytes: int, referer: Optional[str] = None,
                        extra_headers: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """
    Stream a PDF to disk. File appears immediately and grows while downloading.
    Validates '%PDF' magic on first chunk; deletes file if too small / not a PDF.
    Returns (ok, final_url_or_reason).
    """
    tmp_path = out_path + ".part"
    hdrs = (extra_headers or {}).copy()
    if referer:
        hdrs["Referer"] = referer
    # Be generous with Accept to allow server mislabels; validate by magic bytes
    hdrs.setdefault("Accept", "application/pdf,application/octet-stream,*/*;q=0.8")

    try:
        with session.get(url, headers=hdrs, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=True) as r:
            if r.status_code != 200:
                return (False, f"status_{r.status_code}")
            final_url = r.url or url

            total = 0
            first_chunk = True
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=131072):
                    if not chunk:
                        continue
                    if first_chunk:
                        if not _first_k_bytes_is_pdf(chunk[:8]):
                            f.close()
                            try: os.remove(tmp_path)
                            except Exception: pass
                            return (False, "no_pdf_magic")
                        first_chunk = False
                    f.write(chunk)
                    total += len(chunk)
            if total < min_bytes:
                try: os.remove(tmp_path)
                except Exception: pass
                return (False, "too_small")
            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_path, out_path)
            return (True, final_url)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return (False, f"error_{e.__class__.__name__}")

# --------------------------
# Unpaywall
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

# --------------------------
# PMC: PMCID resolution
# --------------------------
def pmid_to_pmcid(pmid: int, session: requests.Session,
                  ncbi_api_key: Optional[str], contact_email: Optional[str]) -> Optional[str]:
    """Resolve PMID→PMCID using JSON; fall back to XML linkname lookups."""
    def _json_try() -> Optional[str]:
        params = {
            "dbfrom": "pubmed", "db": "pmc", "id": str(int(pmid)),
            "retmode": "json", "tool": "oa_fetcher",
            "email": (contact_email or MY_EMAIL_FOR_APIS or "unknown@example.org"),
        }
        if ncbi_api_key: params["api_key"] = ncbi_api_key
        url = f"{EUTILS_BASE}/elink.fcgi"
        backoff = 0.7
        for _ in range(5):
            try:
                r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
            except Exception:
                time.sleep(backoff); backoff *= 1.7; continue
            if r.status_code == 200:
                try:
                    j = r.json()
                    linksets = j.get("linksets") or j.get("linkset") or []
                    if not linksets: return None
                    linkset = linksets[0]
                    for ls in (linkset.get("linksetdbs") or []):
                        if ls.get("linkname") == "pubmed_pmc" or ls.get("dbto") == "pmc":
                            ids = ls.get("links") or ls.get("link") or []
                            if not ids: continue
                            pmcid = ids[0].get("id") if isinstance(ids[0], dict) else ids[0]
                            if isinstance(pmcid, str) and pmcid:
                                return pmcid.upper() if pmcid.upper().startswith("PMC") else "PMC"+pmcid
                except Exception:
                    return None
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 1.7; continue
            return None
        return None

    def _xml_try(linkname: str) -> Optional[str]:
        params = {
            "dbfrom": "pubmed", "db": "pmc", "id": str(int(pmid)),
            "linkname": linkname, "retmode": "xml",
            "tool": "oa_fetcher", "email": (contact_email or MY_EMAIL_FOR_APIS or "unknown@example.org"),
        }
        if ncbi_api_key: params["api_key"] = ncbi_api_key
        url = f"{EUTILS_BASE}/elink.fcgi"
        try:
            r = session.post(url, data=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200 or not r.content:
                return None
            root = ET.fromstring(r.content)
            link_ids = root.findall(".//LinkSet/LinkSetDb/Link/Id")
            if link_ids:
                t = (link_ids[0].text or "").strip()
                if t:
                    return t if t.upper().startswith("PMC") else "PMC" + t
        except Exception:
            return None
        return None

    pmcid = _json_try()
    if pmcid:
        return pmcid
    for ln in ("pubmed_pmc", "pubmed_pmc_refs"):
        pmcid = _xml_try(ln)
        if pmcid:
            return pmcid
    return None

# --------------------------
# PMC Proof-of-Work helpers
# --------------------------
def _extract_pow_params(html: str) -> Optional[Tuple[str, int, str, str]]:
    """
    Parse PMC PoW parameters from challenge HTML.
    Returns (challenge_string, difficulty, cookie_name, cookie_path).
    """
    # Typical patterns seen in PMC challenge pages
    m_ch = re.search(r'POW_CHALLENGE\s*=\s*"([^"]+)"', html)
    m_df = re.search(r'POW_DIFFICULTY\s*=\s*"(\d+)"', html)
    m_cn = re.search(r'POW_COOKIE_NAME\s*=\s*"([^"]+)"', html)
    m_cp = re.search(r'POW_COOKIE_PATH\s*=\s*"([^"]+)"', html)
    if m_ch and m_df and m_cn:
        challenge = m_ch.group(1)
        try:
            diff = int(m_df.group(1))
        except Exception:
            diff = 5
        cookie_name = m_cn.group(1)
        cookie_path = m_cp.group(1) if m_cp else "/"
        return (challenge, diff, cookie_name, cookie_path)
    return None

def _solve_pow(challenge: str, difficulty: int) -> Optional[Tuple[int, str]]:
    """Brute-force SHA-256(challenge+nonce) with required leading zeros."""
    target_prefix = "0" * max(1, difficulty)
    nonce = 0
    # Reasonable safety caps
    cap = {4: 2_000_000, 5: 35_000_000, 6: 500_000_000}.get(difficulty, 100_000_000)
    start = time.time()
    while nonce <= cap:
        h = hashlib.sha256((challenge + str(nonce)).encode("utf-8")).hexdigest()
        if h.startswith(target_prefix):
            logger.info("PMC PoW solved: diff=%d nonce=%d time=%.2fs", difficulty, nonce, time.time()-start)
            return (nonce, h)
        nonce += 1
        if nonce % 1_000_000 == 0:
            logger.debug("PMC PoW progress nonce=%d…", nonce)
    logger.error("PMC PoW failed: exceeded cap=%d for difficulty=%d", cap, difficulty)
    return None

# --------------------------
# PMC download orchestration (streaming + PoW)
# --------------------------
def _pmc_try_endpoints(pmcid: str, session: requests.Session) -> List[str]:
    """Return candidate PDF endpoints across both PMC hosts."""
    bases = [
        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}",
    ]
    pdf_suffixes = ["/pdf", "/pdf/", f"/pdf/{pmcid}.pdf"]
    urls = []
    for base in bases:
        urls.append(base)  # article landing (HTML)
        for suf in pdf_suffixes:
            urls.append(base + suf)
    return urls

def _download_from_pmc_with_pow(pmcid: str, session: requests.Session, out_path: str,
                                min_bytes: int, contact_email: Optional[str]) -> Tuple[bool, str]:
    """
    Attempt PMC download with:
      1) direct /pdf endpoints (HEAD/GET),
      2) HTML landing parse (meta, anchors, iframe/embed),
      3) PoW challenge solving if encountered.
    Writes by streaming; returns (ok, final_url_or_reason).
    """
    cand_urls = _pmc_try_endpoints(pmcid, session)

    # First, try direct endpoints that already look like PDFs
    for url in [u for u in cand_urls if "/pdf" in u.lower() or u.lower().endswith(".pdf")]:
        try:
            r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, stream=True, headers=BROWSER_HEADERS)
        except Exception:
            continue
        final_url = r.url or url
        if r.status_code == 200 and _is_likely_pdf(r, final_url, allow_octet_stream=True):
            # Stream-save using final_url
            ok, fin = _download_streaming(final_url, session, out_path, min_bytes,
                                          referer=url, extra_headers=BROWSER_HEADERS)
            if ok:
                return (True, fin)

    # Next, parse HTML landings for explicit PDF links or PoW
    for url in [u for u in cand_urls if "/articles/" in u]:
        try:
            r = session.get(url, timeout=HTTP_TIMEOUT, allow_redirects=True, headers=BROWSER_HEADERS)
        except Exception:
            continue

        if r.status_code != 200:
            continue

        ct = r.headers.get("Content-Type", "").lower()
        final_landing = r.url or url

        # PoW branch: HTML with challenge content
        if "text/html" in ct and ("POW_CHALLENGE" in r.text or "POW_DIFFICULTY" in r.text):
            params = _extract_pow_params(r.text)
            if not params:
                logger.warning("PMC PoW page detected but parameters not parsed for %s", final_landing)
                continue
            challenge, diff, cookie_name, cookie_path = params
            sol = _solve_pow(challenge, diff)
            if not sol:
                continue
            nonce, _hex = sol
            # Set cookie and retry landing as same-origin
            parsed = urlparse(final_landing)
            session.cookies.set(name=cookie_name, value=f"{challenge},{nonce}",
                                domain=parsed.hostname, path=cookie_path)
            # Retry GET now that the cookie is set; expect a PDF or a page with PDF links
            try:
                r2 = session.get(final_landing, timeout=HTTP_TIMEOUT, allow_redirects=True,
                                 headers={**BROWSER_HEADERS, "Accept": "application/pdf,*/*;q=0.8"})
            except Exception:
                continue
            final2 = r2.url or final_landing
            if r2.status_code == 200 and _is_likely_pdf(r2, final2, allow_octet_stream=True):
                ok, fin = _download_streaming(final2, session, out_path, min_bytes,
                                              referer=final_landing, extra_headers=BROWSER_HEADERS)
                if ok:
                    return (True, fin)
            # If still HTML, try to discover a PDF link within the page
            if r2.status_code == 200 and "text/html" in r2.headers.get("Content-Type", "").lower():
                soup2 = BeautifulSoup(r2.text, "html.parser")
                meta = soup2.find("meta", attrs={"name": "citation_pdf_url"})
                if meta and meta.get("content"):
                    pdf_url = meta["content"]
                    if not pdf_url.startswith("http"):
                        pdf_url = urljoin(final2, pdf_url)
                    ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                                  referer=final2, extra_headers=BROWSER_HEADERS)
                    if ok:
                        return (True, fin)
                for a in soup2.find_all("a", href=True):
                    href = a["href"]
                    if ".pdf" in href.lower() or "/pdf" in href.lower():
                        pdf_url = href if href.startswith("http") else urljoin(final2, href)
                        ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                                      referer=final2, extra_headers=BROWSER_HEADERS)
                        if ok:
                            return (True, fin)
                iframe = soup2.find("iframe", id="pdf") or soup2.find("iframe", id="article") or soup2.find("iframe")
                if iframe and iframe.get("src") and "pdf" in iframe["src"].lower():
                    pdf_url = iframe["src"] if iframe["src"].startswith("http") else urljoin(final2, iframe["src"])
                    ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                                  referer=final2, extra_headers=BROWSER_HEADERS)
                    if ok:
                        return (True, fin)
                embed = soup2.find("embed", attrs={"type": "application/pdf"})
                if embed and embed.get("src"):
                    pdf_url = embed["src"] if embed["src"].startswith("http") else urljoin(final2, embed["src"])
                    ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                                  referer=final2, extra_headers=BROWSER_HEADERS)
                    if ok:
                        return (True, fin)
            continue  # try next landing

        # Non-PoW HTML: look for meta/anchors/iframe PDF links
        if "text/html" in ct:
            soup = BeautifulSoup(r.text, "html.parser")
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                pdf_url = meta["content"]
                if not pdf_url.startswith("http"):
                    pdf_url = urljoin(final_landing, pdf_url)
                ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                              referer=final_landing, extra_headers=BROWSER_HEADERS)
                if ok:
                    return (True, fin)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower() or "/pdf" in href.lower():
                    pdf_url = href if href.startswith("http") else urljoin(final_landing, href)
                    ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                                  referer=final_landing, extra_headers=BROWSER_HEADERS)
                    if ok:
                        return (True, fin)
            iframe = soup.find("iframe", id="pdf") or soup.find("iframe", id="article") or soup.find("iframe")
            if iframe and iframe.get("src") and "pdf" in iframe["src"].lower():
                pdf_url = iframe["src"] if iframe["src"].startswith("http") else urljoin(final_landing, iframe["src"])
                ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                              referer=final_landing, extra_headers=BROWSER_HEADERS)
                if ok:
                    return (True, fin)
            embed = soup.find("embed", attrs={"type": "application/pdf"})
            if embed and embed.get("src"):
                pdf_url = embed["src"] if embed["src"].startswith("http") else urljoin(final_landing, embed["src"])
                ok, fin = _download_streaming(pdf_url, session, out_path, min_bytes,
                                              referer=final_landing, extra_headers=BROWSER_HEADERS)
                if ok:
                    return (True, fin)

    return (False, "no_pdf_found")

# --------------------------
# Sci-Hub Fallback
# --------------------------
def _download_from_scihub(
    identifier: str,
    session: requests.Session,
    out_path: str,
    min_bytes: int,
    referer: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Attempt to download a PDF from Sci-Hub using a DOI or PMID.
    It tries multiple domains and parses the page to find the real PDF URL.
    Returns (ok, final_url_or_reason).
    """
    if not identifier:
        return (False, "no_identifier")

    for domain in SCIHUB_DOMAINS:
        try:
            # Construct the Sci-Hub URL for the given identifier (DOI or PMID)
            scihub_url = f"{domain}/{identifier}"
            logger.info(f"Sci-Hub → Trying domain {domain} for identifier {identifier}")

            # Get the Sci-Hub page that embeds the PDF
            r = session.get(scihub_url, timeout=HTTP_TIMEOUT, headers=BROWSER_HEADERS, allow_redirects=True)
            r.raise_for_status()

            # Check if we landed on a CAPTCHA or error page
            if "captcha" in r.text.lower() or "not found" in r.text.lower():
                logger.warning(f"Sci-Hub ~ {identifier}: Encountered CAPTCHA or 'not found' page at {domain}.")
                continue

            # Parse the HTML to find the PDF link
            soup = BeautifulSoup(r.text, "html.parser")
            pdf_url = None

            # Sci-Hub often embeds the PDF in an <iframe> or <embed> tag
            iframe = soup.find("iframe", id="pdf")
            if iframe and iframe.get("src"):
                pdf_url = iframe["src"]
            else:
                embed = soup.find("embed", attrs={"type": "application/pdf"})
                if embed and embed.get("src"):
                    pdf_url = embed["src"]

            if not pdf_url:
                logger.warning(f"Sci-Hub ~ {identifier}: Could not find PDF embed link at {domain}.")
                continue

            # Ensure the URL is absolute
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif not pdf_url.startswith("http"):
                # Sometimes the URL is relative to the Sci-Hub domain
                pdf_url = urljoin(domain, pdf_url)

            # Download the actual PDF file
            logger.info(f"Sci-Hub ✓ {identifier}: Found PDF URL {pdf_url}. Attempting download.")
            ok, final_url_or_reason = _download_streaming(
                pdf_url, session, out_path, min_bytes,
                referer=scihub_url, extra_headers=BROWSER_HEADERS
            )

            if ok:
                return (True, final_url_or_reason)
            else:
                logger.warning(f"Sci-Hub ✗ {identifier}: Streaming failed from {pdf_url} with reason: {final_url_or_reason}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Sci-Hub ✗ {identifier}: Request exception for domain {domain}: {e}")
            continue # Try next domain
        except Exception as e:
            logger.error(f"Sci-Hub ✗ {identifier}: An unexpected error occurred for domain {domain}: {e}", exc_info=True)
            continue # Try next domain

    return (False, "scihub_failed_all_domains")

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
      1) Tries Unpaywall best_oa_location (url_for_pdf / url) with streaming
      2) Falls back to PMC: PMID→PMCID (JSON + XML), then PDF endpoints/HTML parsing,
         handling PMC Proof-of-Work if present.
    Files are named as YEAR_FirstAuthorStub_PMID.pdf (to match downstream parser).
    Returns {pmid: {"status": "...", "source": "unpaywall|pmc|existing|none", "url": "<final_url_optional>"}}
    """
    # Run-time overrides
    global MY_NCBI_API_KEY, MY_EMAIL_FOR_APIS
    if ncbi_api_key:
        MY_NCBI_API_KEY = ncbi_api_key
    if contact_email:
        MY_EMAIL_FOR_APIS = contact_email

    os.makedirs(output_dir, exist_ok=True)

    # Sessions:
    # - api_session for Unpaywall (product tag UA)
    # - web_session for PMC (browser UA + browsery headers when needed)
    api_session = _make_session(user_agent="", product_tag="OA-Fetch/1.2")
    web_session = _make_session(user_agent=CHROME_UA, product_tag=None)

    ncbi_session = _make_session(user_agent="", product_tag="OA-Fetch/NCBI/1.2")

    rows = _read_handoff_csv(handoff_csv_path)
    results: Dict[str, Dict[str, str]] = {}

    for row in rows:
        # Read metadata
        try:
            pmid = int(str(row.get("PMID") or "").strip())
        except Exception:
            # tolerate lowercase fieldnames too
            try:
                pmid = int(str(row.get("pmid") or "").strip())
            except Exception:
                continue

        title = (row.get("Title") or row.get("title") or "").replace("\t", " ").replace("\n", " ").strip()
        fa    = (row.get("FirstAuthor") or row.get("first_author") or "").strip()

        year = None
        yraw = row.get("Year") or row.get("year")
        try:
            yv = int(yraw)
            if 1500 <= yv <= 2100:
                year = yv
        except Exception:
            year = None

        doi = row.get("DOI") or row.get("doi")
        if isinstance(doi, str) and doi.strip().lower() in {"", "nan", "none", "null"}:
            doi = None

        pdf_name = deterministic_pdf_name(year, fa, pmid)
        pdf_path = os.path.join(output_dir, pdf_name)

        # Skip if present and large enough
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) >= min_pdf_bytes:
            results[str(pmid)] = {"status": "exists", "source": "existing"}
            continue

        # -------- Unpaywall ----------
        source = "none"
        status = "init"
        if doi:
            j = get_unpaywall_json(doi, api_session, MY_EMAIL_FOR_APIS)
            if j:
                loc = j.get("best_oa_location") or {}
                url_pdf = loc.get("url_for_pdf") or loc.get("url") or ""
                if url_pdf:
                    ok, fin = _download_streaming(url_pdf, web_session, pdf_path, min_pdf_bytes,
                                                  referer=f"https://doi.org/{doi}", extra_headers=BROWSER_HEADERS)
                    if ok:
                        results[str(pmid)] = {"status": "ok", "source": "unpaywall", "url": fin}
                        continue
                    else:
                        status = f"unpaywall_{fin}"

        # -------- PMC fallback ----------
        pmcid = pmid_to_pmcid(pmid, ncbi_session, MY_NCBI_API_KEY, MY_EMAIL_FOR_APIS)
        if pmcid:
            ok, fin = _download_from_pmc_with_pow(pmcid, web_session, pdf_path, min_pdf_bytes, MY_EMAIL_FOR_APIS)
            if ok:
                results[str(pmid)] = {"status": "ok", "source": "pmc", "url": fin}
                continue
            else:
                source = "pmc"
                status = f"pmc_{fin}"

        # -------- Sci-Hub fallback ----------
        # Try with DOI first, as it's more reliable, then fall back to PMID
        identifier_to_try = doi or str(pmid)
        ok, fin = _download_from_scihub(identifier_to_try, web_session, pdf_path, min_pdf_bytes, referer=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        if ok:
            results[str(pmid)] = {"status": "ok", "source": "scihub", "url": fin}
            continue
        else:
            # If DOI failed, and we haven't already tried PMID, try it now
            if doi and str(pmid) != identifier_to_try:
                ok_pmid, fin_pmid = _download_from_scihub(str(pmid), web_session, pdf_path, min_pdf_bytes, referer=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                if ok_pmid:
                    results[str(pmid)] = {"status": "ok", "source": "scihub", "url": fin_pmid}
                    continue

            source = "scihub"
            status = f"scihub_{fin}"

        # -------- Give up ----------
        results[str(pmid)] = {"status": status, "source": source}

    return results
