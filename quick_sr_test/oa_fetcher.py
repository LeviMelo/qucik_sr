import pandas as pd
import requests
import os
import time
import re
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from collections import Counter
from bs4 import BeautifulSoup

# --- Configuration ---
MY_NCBI_API_KEY = "YOUR_API_KEY_HERE"
MY_EMAIL_FOR_APIS = "levi4328@gmail.com"

# --- Logging Setup ---
if not logging.getLogger(__name__).hasHandlers():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    # fh = logging.FileHandler('oa_downloader_v1.3_debug.log', mode='w')
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(log_formatter)
    # logger.addHandler(fh)
    # logger.setLevel(logging.DEBUG)
else:
    logger = logging.getLogger(__name__)

# --- HTTP Sessions ---
def create_http_session(email_for_ua, tool_name="GenericTool/1.0"):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=frozenset(['GET', 'POST'])) # Increased backoff slightly
    adapter = HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=30) # Increased pool
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': f'{tool_name} (mailto:{email_for_ua}; for academic research)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf,*/*;q=0.8', # Added application/pdf
        'Accept-Language': 'en-US,en;q=0.5'
    })
    return session

session_http = create_http_session(MY_EMAIL_FOR_APIS, "OAFinderHTTP/1.3")
session_ncbi_alt = create_http_session(MY_EMAIL_FOR_APIS, "OAFinderNCBI/1.3")

def _get_ncbi_params_alt(api_key_for_ncbi, extra=None):
    params = {"tool": "oa_pdf_finder_v4", "email": MY_EMAIL_FOR_APIS} # Hardcoded email now
    if api_key_for_ncbi and api_key_for_ncbi != "YOUR_API_KEY_HERE":
        params["api_key"] = api_key_for_ncbi
    if extra: params.update(extra)
    return params

def sanitize_filename_alt(filename_str):
    if not isinstance(filename_str, str): filename_str = str(filename_str)
    s = re.sub(r'[\\/*?:"<>|]', "", filename_str)
    s = s.replace("\n", " ").replace("\r", "").replace("/", "_").replace(":", "_")
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:180] # Slightly longer filenames allowed


def download_pdf_from_url(pdf_url, filepath, pmid_for_log="N/A", source_description="Source", current_session=None):
    downloader_session = current_session if current_session else session_http
    logger.info(f"PMID {pmid_for_log}: Attempting download ({source_description}): {pdf_url}")
    try:
        response = downloader_session.get(pdf_url, timeout=(10, 60), stream=True, allow_redirects=True)
        response.raise_for_status()
        # Log final URL and content type *before* checking it
        final_url_for_download = response.url
        final_content_type = response.headers.get('Content-Type', '').lower()
        logger.debug(f"PMID {pmid_for_log}: Final download URL: {final_url_for_download}, Content-Type: {final_content_type}")

        # Even if initial headers were misleading, if the final URL ends with .pdf, and we got a 200,
        # it's highly likely a PDF. The actual check for 'application/pdf' is good,
        # but we also need to trust the URL structure for direct PDF links from PMC.
        if 'application/pdf' not in final_content_type and final_url_for_download.lower().endswith('.pdf'):
            logger.warning(f"PMID {pmid_for_log}: Content-Type is '{final_content_type}' but URL '{final_url_for_download}' ends with .pdf. Proceeding with download attempt.")
            # Proceed to download, the binary content will be the ultimate decider.
        elif 'application/pdf' not in final_content_type:
            logger.error(f"PMID {pmid_for_log}: URL {final_url_for_download} ({source_description}) not direct PDF. Content-Type: {final_content_type}.")
            return False, "Not a direct PDF stream"

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536): # Larger chunk
                f.write(chunk)
        
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        if file_size > 1000: # Greater than 1KB
            logger.info(f"PMID {pmid_for_log}: SUCCESS! Downloaded to {filepath} from {source_description} (Size: {file_size} B).")
            return True, f"Downloaded from {source_description}"
        else:
            logger.error(f"PMID {pmid_for_log}: Downloaded file {filepath} from {source_description} is empty or too small (Size: {file_size} B). Deleting.")
            if os.path.exists(filepath): os.remove(filepath)
            return False, "Empty or too small file"

    except requests.exceptions.HTTPError as e:
        logger.error(f"PMID {pmid_for_log}: HTTP Error {e.response.status_code} from {pdf_url} ({source_description})")
        return False, f"HTTP Error {e.response.status_code}"
    # Add other specific exceptions if needed (Timeout, ConnectionError etc.)
    except requests.exceptions.RequestException as e:
        logger.error(f"PMID {pmid_for_log}: Request Exception for {pdf_url} ({source_description}): {e}")
        return False, f"Request Exception"
    except Exception as e:
        logger.error(f"PMID {pmid_for_log}: Unexpected error downloading {pdf_url} ({source_description}): {e}", exc_info=True)
        return False, "Unexpected download error"


def get_oa_pdf_from_unpaywall(doi):
    if not doi: logger.warning("Unpaywall: No DOI provided."); return None
    api_url = f"https://api.unpaywall.org/v2/{quote_plus(doi)}?email={MY_EMAIL_FOR_APIS}"
    logger.debug(f"Unpaywall: Querying {api_url}")
    try:
        response = session_http.get(api_url, timeout=20)
        if response.status_code == 404: logger.info(f"Unpaywall: DOI {doi} not found."); return None
        if response.status_code == 422: logger.error(f"Unpaywall: HTTP 422 Unprocessable Entity for DOI {doi}."); return None
        response.raise_for_status() # For other HTTP errors
        data = response.json()
        if data and data.get("is_oa"):
            pdf_url_to_try = None
            best_oa_location = data.get("best_oa_location")
            if best_oa_location and best_oa_location.get("url_for_pdf"):
                pdf_url_to_try = best_oa_location.get("url_for_pdf")
                logger.info(f"Unpaywall: Found PDF URL (best_oa) for DOI {doi}: {pdf_url_to_try}")
            else: # Check all oa_locations
                for loc in data.get("oa_locations", []):
                    if loc.get("url_for_pdf"):
                        pdf_url_to_try = loc.get("url_for_pdf")
                        logger.info(f"Unpaywall: Found PDF URL (oa_locations) for DOI {doi}: {pdf_url_to_try}")
                        break # Take the first one from oa_locations
            if pdf_url_to_try: return pdf_url_to_try
            logger.info(f"Unpaywall: DOI {doi} is OA, but no PDF URL in best_oa or oa_locations.")
        else: logger.info(f"Unpaywall: No OA or no PDF URL for DOI {doi}. Free text: {data.get('free_fulltext_url', 'N/A')}")
    except Exception as e: logger.error(f"Unpaywall: Error for DOI {doi}: {e}", exc_info=False)
    return None


def get_pmcid_from_pmid(pmid, ncbi_api_key):
    # (This function is generally okay, ensure it logs appropriately)
    if not pmid: return None
    logger.debug(f"PMC: Getting PMCID for PMID {pmid}.")
    params = _get_ncbi_params_alt(ncbi_api_key, {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "cmd": "neighbor_score"})
    linknames_to_try = ["pubmed_pmc_refs", "pubmed_pmc"]
    for linkname in linknames_to_try:
        params["linkname"] = linkname
        try:
            response = session_ncbi_alt.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi", data=params, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            link_ids = root.findall("./LinkSet/LinkSetDb/Link/Id")
            if link_ids:
                pmcid_text = link_ids[0].text
                if pmcid_text:
                    pmcid = pmcid_text if pmcid_text.upper().startswith("PMC") else "PMC" + pmcid_text.strip()
                    logger.info(f"PMC: Found PMCID {pmcid} for PMID {pmid} (linkname: {linkname}).")
                    return pmcid
        except Exception as e:
            logger.warning(f"PMC: Error getting PMCID for PMID {pmid} (linkname {linkname}): {e}", exc_info=False)
            if isinstance(e, requests.exceptions.HTTPError) and response and "Cannot ELink from db" in response.text: break
    logger.info(f"PMC: No PMCID found for PMID {pmid} via elink.")
    return None


def download_from_pmc(pmcid, filepath, pmid_for_log):
    if not pmcid: return False, "No PMCID provided"
    base_article_page_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    logger.info(f"PMC: Attempting access for {pmcid} via initial URL: {base_article_page_url}")

    try:
        pmc_session = create_http_session(MY_EMAIL_FOR_APIS, "OAFinderPMC/1.3_PMC")
        response = pmc_session.get(base_article_page_url, timeout=(10, 30), allow_redirects=True)
        response.raise_for_status()
        
        final_url_after_redirects = response.url
        final_content_type = response.headers.get('Content-Type', '').lower()
        logger.info(f"PMC: For {pmcid}, initial URL led to final URL: {final_url_after_redirects} (Content-Type: {final_content_type})")

        # CRITICAL CHANGE: If the final URL *is* a PDF link (ends with .pdf), try to download it directly.
        if final_url_after_redirects.lower().endswith('.pdf'):
            logger.info(f"PMC: Final URL for {pmcid} appears to be a direct PDF link. Attempting download.")
            return download_pdf_from_url(final_url_after_redirects, filepath, pmid_for_log, f"PubMed Central (Direct from {pmcid} redirect)", current_session=pmc_session)
        
        # If not a direct PDF link after redirects, then check if it's HTML to parse
        elif 'text/html' in final_content_type:
            logger.info(f"PMC: Final URL {final_url_after_redirects} is HTML for {pmcid}. Parsing for specific PDF link.")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # More targeted selectors for links within the PMC HTML page
            link_selectors = [
                'ul.format-menu li a[href$=".pdf"]',            # Standard format menu
                'div.fm-sec-pdf a[href$=".pdf"]',                # PDF section in format menu
                'div.format-dropdown ul li a[href$=".pdf"]',     # Dropdown format menu
                'a.pdf-link[href$=".pdf"]',                      # Explicit class
                'a[title*="Download PDF"][href$=".pdf"]',        # Title attribute
                f'a[href*="/pdf/"][href*="{pmcid.replace("PMC","")}"][href$=".pdf"]', # Contains PMCID num & /pdf/
                'a[href*="/articles/{pmcid}/pdf/"]' # A link that itself contains the full base path again + filename
            ]
            actual_pdf_link_on_page = None
            for selector in link_selectors:
                link_tag = soup.select_one(selector)
                if link_tag and link_tag.get('href'):
                    href_val = link_tag.get('href')
                    # Ensure it's a valid-looking link
                    if href_val and not href_val.startswith("javascript:"):
                        resolved_url = urljoin(final_url_after_redirects, href_val)
                        if resolved_url.lower().endswith(".pdf"): # Double check
                             logger.info(f"PMC: Found PDF link on landing page for {pmcid} via selector '{selector}': {resolved_url}")
                             actual_pdf_link_on_page = resolved_url
                             break
            
            if actual_pdf_link_on_page:
                return download_pdf_from_url(actual_pdf_link_on_page, filepath, pmid_for_log, f"PubMed Central (Parsed from {pmcid} landing)", current_session=pmc_session)
            else:
                logger.warning(f"PMC: No definitive PDF link found on HTML landing page for {pmcid} ({final_url_after_redirects}).")
                # Optional: save HTML
                # debug_html_path = os.path.join(os.getcwd(), f"debug_pmc_{pmcid}_landing.html")
                # with open(debug_html_path, "w", encoding="utf-8") as f_html: f_html.write(soup.prettify())
                # logger.debug(f"Saved PMC HTML for {pmcid} to {debug_html_path}")
                return False, "PMC: PDF link not found in HTML"
        else:
            logger.warning(f"PMC: Unexpected Content-Type '{final_content_type}' for {pmcid} at {final_url_after_redirects} (not PDF or HTML).")
            return False, f"PMC: Unexpected Content-Type {final_content_type}"

    except requests.exceptions.HTTPError as e:
        logger.error(f"PMC: HTTP Error {e.response.status_code} for {pmcid} (Initial URL: {base_article_page_url})")
        return False, f"PMC HTTP Error {e.response.status_code}"
    except Exception as e:
        logger.error(f"PMC: Error processing {pmcid}: {e}", exc_info=True)
        return False, "PMC processing error"


# --- Main Orchestration ---
def attempt_oa_downloads(articles_info, download_dir, ncbi_api_key):
    # (Main logic structure of this function remains similar, calls the updated download_from_pmc)
    if not os.path.exists(download_dir): os.makedirs(download_dir)
    successful_downloads, already_existed_count = 0, 0
    failed_oalookup_items = [] # Store dicts of failed items for more info
    download_sources = Counter()

    for i, article in enumerate(articles_info):
        pmid = article.get("pmid")
        doi = article.get("doi")
        title_for_file = article.get("title_for_file", doi or pmid or f"unknown_article_{i+1}")
        filename_stem = sanitize_filename_alt(title_for_file)
        pdf_filepath = os.path.join(download_dir, filename_stem + ".pdf")

        logger.info(f"\n--- OA Check {i+1}/{len(articles_info)} for PMID: {pmid}, DOI: {doi or 'N/A'} ---")
        if os.path.exists(pdf_filepath):
            logger.info(f"PDF already exists: {pdf_filepath}. Skipping."); already_existed_count += 1; continue

        downloaded_this_article = False; source_of_download = "Unknown"; status_msg = "Not found"

        if doi: # Try Unpaywall first if DOI exists
            unpaywall_pdf_url = get_oa_pdf_from_unpaywall(doi)
            if unpaywall_pdf_url:
                success, status_msg = download_pdf_from_url(unpaywall_pdf_url, pdf_filepath, pmid, "Unpaywall")
                if success: downloaded_this_article = True; source_of_download = "Unpaywall"
        
        if not downloaded_this_article and pmid: # Then try PMC if PMID exists
            pmcid = get_pmcid_from_pmid(pmid, ncbi_api_key)
            if pmcid:
                success, status_msg = download_from_pmc(pmcid, pdf_filepath, pmid)
                if success: downloaded_this_article = True; source_of_download = "PubMed Central"
        
        if downloaded_this_article:
            successful_downloads += 1
            download_sources[source_of_download] += 1
        else:
            logger.info(f"PMID {pmid}: No OA PDF found/downloaded. Last status: {status_msg}")
            failed_oalookup_items.append({'pmid': pmid, 'doi': doi, 'status': status_msg})

    logger.info("\n--- Open Access Download Summary ---")
    logger.info(f"Total articles for OA check: {len(articles_info)}")
    logger.info(f"Successfully downloaded via OA: {successful_downloads}")
    logger.info(f"Already existed (skipped): {already_existed_count}")
    if download_sources:
        logger.info("Downloads by OA source:"); [logger.info(f"  {s}: {c}") for s,c in download_sources.items()]
    if failed_oalookup_items:
        logger.info(f"Items for which no OA PDF was found/downloaded ({len(failed_oalookup_items)}):")
        for item in failed_oalookup_items: logger.info(f"  PMID: {item['pmid']}, DOI: {item['doi']}, Status: {item['status']}")
    return successful_downloads, already_existed_count, failed_oalookup_items


if __name__ == "__main__":
    logger.info(f"Using email: {MY_EMAIL_FOR_APIS} for API calls.")
    if MY_NCBI_API_KEY == "YOUR_API_KEY_HERE": logger.warning("NCBI_API_KEY is placeholder.")

    example_failed_pmids = [
        "39955421", "40340819", "39520824", "39489669", "39384309", "39185540", 
        "39068053", "38673038", "37802689", "39083294", "36788057", "39342249", 
        "37062759", "36969299", "35790215", "31274269", "33401363", "26888001"
    ]
    example_pmid_to_doi_map = {
        "39955421": "10.1007/s00383-025-05977-0", "40340819": "10.1136/bmjpo-2024-003280",
        "39520824": "10.1016/j.jpedsurg.2024.162046", "39489669": "10.1053/j.jvca.2024.10.005",
        "39384309": "10.1136/bmjpo-2024-002824", "39185540": "10.14309/crj.0000000000001469",
        "39068053": "10.1016/j.jpedsurg.2024.06.021", 
        "38673038": "10.3390/jpm14040411", # This one (PMC11051180) should work now
        "36788057": None, # This one (PMC11680611) should work now
        "37802689": "10.1053/j.jvca.2023.09.006", # Test DOI if available, example here
        # Fill other DOIs if available, otherwise None
        "39083294": None, "39342249": "10.1186/s12887-024-05083-5", # Example
        "37062759": "10.1089/lap.2022.0537", # Example
        "36969299": "10.3389/fped.2023.933158", # Example
        "35790215": "10.4097/kja.22279", # Example
        "31274269": "10.23736/S0375-9393.19.13880-1", # Example
        "33401363": "10.2196/10996", # Example
        "26888001": None
    }
    example_pmid_to_title_map = {
        "39955421": "Li_S_2025_Pneumonectomies", "40340819": "Pentz_B_2025_ERAS_Pain",
        "39520824": "Mansfield_SA_2025_ERAS_Neonates", "39489669": "Meier_S_2025_JVCA_ERAS",
        "39384309": "Pilkington_M_2024_BMJPO_Pain", "39185540": "Pilkington_M_2024_CRJ_Esophageal",
        "39068053": "Pilkington_M_2024_JPedSurg_Gastroschisis", "38673038": "Zacha_S_2024_JPM_Anesthesia",
        "37802689": "Alfaras-Melainis_K_2024_JVCA_Postoperative", "39083294": "Wishahi_M_2024_JAMA_Surg_AI",
        "36788057": "Downing_L_2023_JPedSurg_Telemedicine", "39342249": "Guo_R_2024_BMC_Peds_Appendicitis",
        "37062759": "Martynov_I_2023_LAP_Hernia", "36969299": "Huang_J_2023_Front_Peds_Pyloromyotomy",
        "35790215": "Lucente_M_2022_KJA_Analgesia", "31274269": "Paladini_G_2019_Minerva_Anestesiol_ERAS",
        "33401363": "Wildemeersch_D_2018_JMIR_Mobile_Health", "26888001": "Wang_Y_2015_ERAS_Thoracic"
    }

    articles_to_check_oa = []
    for pmid in example_failed_pmids:
        doi = example_pmid_to_doi_map.get(pmid)
        title_stub = example_pmid_to_title_map.get(pmid, pmid) 
        articles_to_check_oa.append({"pmid": pmid, "doi": doi, "title_for_file": title_stub})

    OA_DOWNLOAD_DIR = os.path.join(os.getcwd(), "oa_downloads_v4_final_pmc_test") # New dir for test
    attempt_oa_downloads(
        articles_info=articles_to_check_oa,
        download_dir=OA_DOWNLOAD_DIR,
        ncbi_api_key=MY_NCBI_API_KEY
    )
    logger.info("OA download process finished.")