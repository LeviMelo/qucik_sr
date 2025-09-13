# sniff_validation_engine_v3_1.py
# Refactored "Sniff Validation Engine" (state-machine architecture)
# - Single "universe" query + validated filters (no legacy BROAD/FOCUSED)
# - Deterministic prefilter (language/year/design) BEFORE LLM
# - PICO-weighted TF-IDF re-ranker (true TF-IDF × role weights)
# - Strict screener with Ask-Validate-Retry JSON
# - Senior plausibility check (second LLM pass) to prevent topic drift
# - Ground-truth vocabulary (MeSH) only from confirmed INCLUDEs
# - Robust model switching with idle TTL and conservative waits to avoid CPU fallback
#
# Requirements:
#   - LM Studio running at LMSTUDIO_BASE (default http://127.0.0.1:1234)
#   - Two local models served by LM Studio:
#       QWEN_MODEL  (for protocol, remediation, plausibility)
#       SCREENER_MODEL (fast model for checklist screening)
#   - Internet for NCBI E-utilities
#
# Usage:
#   1) Edit USER_NLQ below (natural language RQ; you may include notes for screening).
#   2) Optionally edit constants (MODEL names, thresholds) or set via env vars.
#   3) Run as a single cell/script. See printed summary + output files in OUT_DIR.
#
# Outputs:
#   - sniff_report.txt  : human-readable report of states, warnings, and final strategy
#   - sniff_artifacts.json : machine-readable details (locked protocol, universe query,
#                            recommended filters, ground truth PMIDs, MeSH vernaculum,
#                            research_question_string_for_embedding, warnings)
#
# NOTE: We DO NOT hard-filter by "Humans" or MeSH age bands deterministically.
#       Deterministic gates: year, language, and publication type (design allowlist).
#       The LLM uses PubTypes + MeSH contextually during screening.

import os, json, time, re, textwrap, pathlib, random, math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Callable, Optional
import requests
from xml.etree import ElementTree as ET

# ----------------------------
# Config / Constants
# ----------------------------
LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234")
QWEN_MODEL    = os.getenv("QWEN_MODEL", "unsloth/qwen3-4b")
SCREENER_MODEL= os.getenv("SCREENER_MODEL", "gemma-3n-e4b-it@q4_k_s")  # fast checklist screener

ENTREZ_EMAIL   = os.getenv("ENTREZ_EMAIL", "you@example.com")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

HTTP_TIMEOUT   = int(os.getenv("HTTP_TIMEOUT", "300"))
MODEL_TTL_SEC  = float(os.getenv("MODEL_TTL_SEC", "5.0"))  # idle TTL hint (LM Studio)
MODEL_SWAP_WAIT= float(os.getenv("MODEL_SWAP_WAIT", "10.0"))  # conservative wait between model swaps

OUT_DIR = pathlib.Path("sniff_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Universe sizing thresholds
UNIVERSE_TARGET = (50, 10000)   # ideal window
UNIVERSE_HARD_MIN = 25          # if final count < 25 after remediation -> terminate

# Rerank / screening sizes
UNIVERSE_FETCH_MAX = 800        # number of PubMed records to fetch for rerank (cap)
SCREEN_TOP_K       = 60         # how many (after rerank+prefilter) to send to screener

# Screener rules
SCREENER_RETRY_MAX = 3
PLAUSIBILITY_MIN_INCLUDES = 3   # need ≥ this many includes, post-plausibility, or terminate

# Role weights for PICO TF-IDF reranker
WEIGHTS = {
    "P": 1.5,
    "I": 1.75,
    "C": 1.0,
    "O": 1.0,
    "ANCHOR": 2.0,
    "AVOID": -2.5
}

# PubMed E-utilities base + headers
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "sniff-validation-engine/3.1 (+local)", "Accept": "application/json"}

random.seed(42)

# ----------------------------
# Lightweight logging helpers
# ----------------------------
from datetime import datetime
LOG_VERBOSE = True  # set False to quiet detailed logs

def log(tag: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts}  [{tag}] {msg}")

def log_json(tag: str, obj):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        j = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        j = str(obj)
    print(f"{ts}  [{tag}] {j}")


# ----------------------------
# Utilities: LM Studio model management
# ----------------------------
class ModelManager:
    def __init__(self, base: str, idle_ttl_sec: float = MODEL_TTL_SEC, swap_wait: float = MODEL_SWAP_WAIT):
        self.base = base.rstrip("/")
        self.idle_ttl = idle_ttl_sec
        self.swap_wait = swap_wait
        self.current_model = None
        self.last_used_ts = 0.0

    def _maybe_wait_for_idle_eviction(self):
        now = time.time()
        idle = now - self.last_used_ts
        if idle < self.idle_ttl:
            time.sleep(self.idle_ttl - idle)
        # conservative extra wait to allow LM Studio to evict models
        time.sleep(max(0.0, self.swap_wait - self.idle_ttl))

    def _best_effort_unload_all(self):
        # LM Studio does not officially document unload; try likely endpoints, ignore errors.
        for path in ["/v1/models/unload_all", "/v1/engines/unload_all", "/v1/models/unload"]:
            try:
                requests.post(self.base + path, timeout=3.0)
            except Exception:
                pass

    def switch(self, model: str):
        if self.current_model and self.current_model != model:
            # allow time for the previous model to be evicted
            self._maybe_wait_for_idle_eviction()
            self._best_effort_unload_all()
        self.current_model = model
        self.last_used_ts = time.time()

    def mark_used(self):
        self.last_used_ts = time.time()

MM = ModelManager(LMSTUDIO_BASE)

def lm_chat(model: str, system: str, user: str, temperature=0.0, max_tokens=2048) -> str:
    MM.switch(model)
    url = f"{LMSTUDIO_BASE.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False
    }
    r = requests.post(url, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    MM.mark_used()
    return r.json()["choices"][0]["message"]["content"]

# ----------------------------
# JSON extraction & Ask-Validate-Retry
# ----------------------------
_BEGIN = re.compile(r"BEGIN_JSON\s*", re.I)
_END   = re.compile(r"\s*END_JSON", re.I)
FENCE  = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

def _sanitize_json_str(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018","'").replace("\u2019","'")
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    return s.strip()

def extract_json_block_or_fence(txt: str) -> str:
    blocks = []
    pos=0
    while True:
        m1 = _BEGIN.search(txt, pos)
        if not m1: break
        m2 = _END.search(txt, m1.end())
        if not m2: break
        blocks.append(txt[m1.end():m2.start()])
        pos = m2.end()
    if blocks:
        return _sanitize_json_str(blocks[-1])

    fences = FENCE.findall(txt)
    if fences:
        return _sanitize_json_str(fences[-1])

    # last balanced {...}
    s = txt
    last_obj=None; stack=0; start=None
    for i,ch in enumerate(s):
        if ch=='{':
            if stack==0: start=i
            stack+=1
        elif ch=='}':
            if stack>0:
                stack-=1
                if stack==0 and start is not None:
                    last_obj = s[start:i+1]
    if last_obj:
        return _sanitize_json_str(last_obj)
    raise ValueError("No JSON-like content found")

STRICT_JSON_RULES = (
  "Return ONLY one JSON object. No analysis, no preface, no notes. "
  "Wrap it EXACTLY with:\nBEGIN_JSON\n{...}\nEND_JSON"
)

def get_validated_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[Dict[str,Any]], Tuple[bool,str]],
    retries: int = 3,
    max_tokens: int = 2048
) -> Dict[str,Any]:
    history_user = user_prompt
    for i in range(retries):
        raw = lm_chat(model, system_prompt, history_user + "\n\n" + STRICT_JSON_RULES, max_tokens=max_tokens)
        try:
            js = json.loads(extract_json_block_or_fence(raw))
        except Exception as e:
            err = f"malformed JSON: {e}"
            if i == retries-1:
                raise SystemExit(f"Fatal: LLM failed to produce valid JSON after retries. Last error: {err}")
            history_user += f"\n\nYour previous output was invalid due to: {err}\nPlease fix and return a single valid JSON object."
            continue
        ok, why = validator(js)
        if ok:
            return js
        if i == retries-1:
            raise SystemExit(f"Fatal: LLM JSON schema invalid after retries: {why}")
        history_user += f"\n\nYour previous JSON failed validation: {why}\nPlease correct your output and adhere to the required schema."

# ----------------------------
# KB defaults (designs, languages, pubtype map)
# ----------------------------
KB_PATH = pathlib.Path("system_knowledge_base.json")
KB_DEFAULT = {
    "publication_types_allowable": [
        "Randomized Controlled Trial",
        "Controlled Clinical Trial",
        "Clinical Trial",
        "Comparative Study",
        "Cohort Studies",
        "Case-Control Studies",
        "Observational Study",
        "Multicenter Study",
        "Cross-Sectional Studies",
        "Clinical Trial Protocol",
        "Evaluation Study"
    ],
    "languages": ["english","spanish","portuguese","french","german","italian","chinese","japanese","korean"],
    "designs_primary": ["Randomized Controlled Trial","Controlled Clinical Trial","Clinical Trial"],
    "designs_secondary": ["Comparative Study","Cohort Studies","Case-Control Studies","Observational Study","Multicenter Study","Evaluation Study"],
    "pubtype_aliases": {
        "Randomized Controlled Trial": ["Randomized Controlled Trial"],
        "Controlled Clinical Trial": ["Controlled Clinical Trial"],
        "Clinical Trial": ["Clinical Trial"],
        "Comparative Study": ["Comparative Study"],
        "Cohort Studies": ["Cohort Studies","Prospective Studies","Retrospective Studies"],
        "Case-Control Studies": ["Case-Control Studies"],
        "Observational Study": ["Observational Study"],
        "Multicenter Study": ["Multicenter Study"],
        "Cross-Sectional Studies": ["Cross-Sectional Studies"],
        "Clinical Trial Protocol": ["Clinical Trial Protocol","Study Protocols"],
        "Evaluation Study": ["Evaluation Study"]
    }
}

def load_or_init_kb() -> Dict[str,Any]:
    if KB_PATH.exists():
        try:
            on_disk = json.loads(KB_PATH.read_text(encoding="utf-8"))
        except Exception:
            # if corrupted, reset to defaults
            KB_PATH.write_text(json.dumps(KB_DEFAULT, indent=2), encoding="utf-8")
            return KB_DEFAULT

        # Merge defaults → fill any missing keys from KB_DEFAULT
        merged = dict(KB_DEFAULT)
        for k, v in on_disk.items():
            merged[k] = v

        # Persist the merged file so future runs are stable
        KB_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return merged

    KB_PATH.write_text(json.dumps(KB_DEFAULT, indent=2), encoding="utf-8")
    return KB_DEFAULT


KB = load_or_init_kb()

# ----------------------------
# State 1: Protocol Lockdown
# ----------------------------
PROTO_SYSTEM = """You are designing a structured, search-ready SR protocol from a natural-language question.

Produce a protocol that includes BOTH narrative fields for LLMs and structured fields for code.

Rules:
- Use concise search tokens for P/I/C/O (each token ≤ 3-4 words). Avoid overlong phrases.
- Populate 'designs_preference' by selecting ONE from the provided KB 'designs_primary'.
- 'deterministic_filters' MUST include: languages (subset of KB.languages) and year_min (from user or a reasonable default).
- Do not hallucinate comparators or outcomes not implied; it's OK to leave lists empty if not provided.
- If the question is incoherent or underspecified, set "needs_clarification"=true and write a short "clarification_request".

Return ONLY JSON as requested."""

def proto_user(nlq: str, kb: Dict[str,Any]) -> str:
    kb_view = {
        "designs_primary": kb.get("designs_primary", KB_DEFAULT["designs_primary"]),
        "languages": kb.get("languages", KB_DEFAULT["languages"])
    }
    return f"""Natural-Language Question:
<<<
{nlq.strip()}
>>>

Knowledge Base (valid choices):
{json.dumps({"designs_primary":KB["designs_primary"], "languages":KB["languages"]}, indent=2)}

Output schema:
{{
  "narrative_question": "<1 paragraph restatement>",
  "inclusion_criteria": ["...","..."],
  "exclusion_criteria": ["..."],
  "screening_rules_note": {{
    "user_notes": "<verbatim any adjunct/instructions embedded in NLQ>",
    "llm_guidance": "<short additional instructions inferred>"
  }},
  "pico_tokens": {{
    "P": ["..."],
    "I": ["..."],
    "C": ["..."],
    "O": ["..."]
  }},
  "anchors_must_have": ["..."],   // topical anchors to enforce (e.g., MIRPE, Nuss)
  "avoid_terms": ["..."],
  "designs_preference": "<ONE of designs_primary>",
  "deterministic_filters": {{
     "languages": ["..."],  // subset of KB.languages
     "year_min": 2015
  }},
  "needs_clarification": false,
  "clarification_request": ""
}}"""

def validate_protocol(js: Dict[str,Any]) -> Tuple[bool,str]:
    try:
        # minimal schema checks
        req_top = ["narrative_question","inclusion_criteria","exclusion_criteria",
                   "screening_rules_note","pico_tokens","anchors_must_have",
                   "avoid_terms","designs_preference","deterministic_filters",
                   "needs_clarification","clarification_request"]
        for k in req_top:
            if k not in js: return False, f"missing key: {k}"
        if not isinstance(js["pico_tokens"], dict): return False, "pico_tokens must be object"
        for k in ["P","I","C","O"]:
            if k not in js["pico_tokens"]: return False, f"pico_tokens missing {k}"
            if not isinstance(js["pico_tokens"][k], list): return False, f"pico_tokens[{k}] must be list"
        df = js["deterministic_filters"]
        if not isinstance(df.get("languages",[]), list) or not df.get("languages"):
            return False, "languages must be non-empty list"

        y = df.get("year_min", 0)
        if isinstance(y, str) and y.isdigit():
            df["year_min"] = int(y)
        elif not isinstance(y, int):
            return False, "year_min must be int or numeric string"

        if js["designs_preference"] not in KB["designs_primary"]:
            return False, "designs_preference must be one of KB.designs_primary"

        # keep the short-token safeguard
        long_bad = [t for t in (js["pico_tokens"]["P"]+js["pico_tokens"]["I"]+js["pico_tokens"]["C"]+js["pico_tokens"]["O"]) if len(t.split())>5]
        if long_bad:
            return False, f"tokens too long: {long_bad[:3]}"
        return True, ""
    except Exception as e:
        return False, f"exception in protocol validation: {e}"

# ----------------------------
# PubMed: search & fetch
# ----------------------------
def esearch_ids(term: str, mindate: Optional[int], retmax: int = 5000) -> Tuple[int, List[str]]:
    p = {"db":"pubmed","retmode":"json","term":term,"retmax":retmax,"email":ENTREZ_EMAIL,"usehistory":"y"}
    if ENTREZ_API_KEY: p["api_key"]=ENTREZ_API_KEY
    if mindate: p["mindate"]=str(mindate)
    r = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params=p, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json().get("esearchresult", {})
    count = int(js.get("count","0"))
    webenv = js.get("webenv"); qk = js.get("querykey")
    if not count or not webenv or not qk:
        return 0, []
    r2 = requests.get(f"{EUTILS}/esearch.fcgi", headers=HEADERS, params={
        "db":"pubmed","retmode":"json","retmax":retmax,"retstart":0,"email":ENTREZ_EMAIL,"WebEnv":webenv,"query_key":qk,
        **({"api_key":ENTREZ_API_KEY} if ENTREZ_API_KEY else {})
    }, timeout=HTTP_TIMEOUT)
    r2.raise_for_status()
    ids = r2.json().get("esearchresult",{}).get("idlist",[])
    return count, [str(x) for x in ids]

def efetch_xml(pmids: List[str]) -> str:
    if not pmids: return ""
    params = {"db":"pubmed","retmode":"xml","rettype":"abstract","id":",".join(pmids),"email":ENTREZ_EMAIL}
    if ENTREZ_API_KEY: params["api_key"]=ENTREZ_API_KEY
    r = requests.get(f"{EUTILS}/efetch.fcgi", headers={"User-Agent":"sniff-validation-engine/3.1"}, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str) -> List[Dict[str,Any]]:
    out = []
    if not xml_text.strip(): return out
    root = ET.fromstring(xml_text)
    def _join(node):
        if node is None: return ""
        try: return "".join(node.itertext())
        except Exception: return node.text or ""
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
        lang = art.findtext(".//Language") or None
        pubtypes = [pt.text for pt in art.findall(".//PublicationTypeList/PublicationType") if pt.text]
        mesh = [mh.findtext("./DescriptorName") for mh in art.findall(".//MeshHeadingList/MeshHeading") if mh.findtext("./DescriptorName")]
        out.append({"pmid":pmid,"title":title,"abstract":abstract,"year":year,"language":lang,"pubtypes":pubtypes,"mesh":mesh})
    return out

# ----------------------------
# Query assembly, remediation
# ----------------------------
def or_block(terms: List[str], field="tiab") -> str:
    toks=[]
    for t in terms:
        t=t.strip()
        if not t: continue
        if " " in t or "-" in t:
            toks.append(f"\"{t}\"[{field}]")
        else:
            toks.append(f"{t}[{field}]")
    if not toks: return ""
    return "(" + " OR ".join(toks) + ")"

def build_universe_query(P: List[str], I: List[str], anchors: List[str]) -> str:
    Pq = or_block(P, "tiab"); Iq = or_block(I, "tiab")
    Aq = or_block(anchors, "tiab") if anchors else ""
    parts = [x for x in [Pq, Iq, Aq] if x]
    return " AND ".join(parts)

REM_SYS = """You are a search strategy repair assistant. The current query is underperforming (too few hits).

Constraints (do NOT violate):
- Keep the core topic: population and intervention must remain faithful to the protocol.
- Only operate on the P/I token lists: REMOVE_TERM, SIMPLIFY_TERM (shorten phrase), or ADD_ALTERNATE (synonym).
- Return at most 2 operations.
- Do NOT introduce terms that contradict population or intervention focus.
Return JSON only."""

def rem_user(query: str, count: int, protocol: Dict[str,Any]) -> str:
    return f"""Current universe query (hits={count}):
{query}

Protocol (brief):
P tokens: {protocol["pico_tokens"]["P"]}
I tokens: {protocol["pico_tokens"]["I"]}
Anchors: {protocol["anchors_must_have"]}
Avoid: {protocol["avoid_terms"]}
Design preference: {protocol["designs_preference"]}

Allowed ops (array of steps):
[{{"op":"REMOVE_TERM","where":"P|I","term":"..."}}, {{"op":"SIMPLIFY_TERM","where":"P|I","term":"full phrase","simplified":"short term"}}, {{"op":"ADD_ALTERNATE","where":"P|I","term":"root","alternate":"synonym"}}]

BEGIN_JSON
{{"ops":[]}}
END_JSON"""

def validate_remediation(js: Dict[str,Any]) -> Tuple[bool,str]:
    if "ops" not in js or not isinstance(js["ops"], list): return False, "missing ops[]"
    if len(js["ops"])>2: return False, "too many ops"
    for op in js["ops"]:
        if op.get("op") not in ["REMOVE_TERM","SIMPLIFY_TERM","ADD_ALTERNATE"]:
            return False, f"bad op: {op.get('op')}"
        if op.get("where") not in ["P","I"]:
            return False, "where must be P or I"
    return True, ""

def apply_remediation(P: List[str], I: List[str], ops: List[Dict[str,str]]) -> Tuple[List[str], List[str]]:
    Pn = P[:]; In = I[:]
    def _apply(lst, op):
        if op["op"]=="REMOVE_TERM":
            lst = [t for t in lst if t.lower()!=op.get("term","").lower()]
        elif op["op"]=="SIMPLIFY_TERM":
            t = op.get("term",""); s=op.get("simplified","")
            lst = [s if x.lower()==t.lower() and s else x for x in lst]
        elif op["op"]=="ADD_ALTERNATE":
            alt = op.get("alternate","")
            if alt and alt.lower() not in [x.lower() for x in lst]:
                lst.append(alt)
        return lst
    for op in ops:
        if op["where"]=="P":
            Pn = _apply(Pn, op)
        else:
            In = _apply(In, op)
    return Pn, In

# ----------------------------
# Deterministic prefilter (language/year/design only)
# ----------------------------
def passes_prefilter(rec: Dict[str,Any], languages: List[str], year_min: int, design_allowlist: List[str], pubtype_alias: Dict[str,List[str]]) -> bool:
    if rec.get("year") and rec["year"] < year_min:
        return False
    if rec.get("language") and rec["language"].lower() not in [x.lower() for x in languages]:
        return False
    if design_allowlist:
        # Any intersection between aliases for allowed designs and rec.pubtypes
        rpts = set(rec.get("pubtypes") or [])
        for design in design_allowlist:
            aliases = set(pubtype_alias.get(design, [design]))
            if rpts & aliases:
                return True
        # allow if no pubtypes present (unknown design) -> keep for LLM
        if not rpts:
            return True
        return False
    return True

# ----------------------------
# TF-IDF PICO reranker
# ----------------------------
def build_tfidf_and_score(records: List[Dict[str,Any]], protocol: Dict[str,Any]) -> List[Tuple[float,Dict[str,Any]]]:
    texts = []
    for r in records:
        t = (r.get("title","") + " " + r.get("abstract","")).strip()
        texts.append(t if t else r.get("title",""))
    # import TF-IDF with fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", max_features=50000)
        X = vec.fit_transform(texts)
        vocab = vec.vocabulary_
        idf_diag = None  # scikit handles internally
        def tfidf(term, row_idx):
            j = vocab.get(term.lower())
            if j is None: return 0.0
            return X[row_idx, j]
    except Exception:
        # very simple fallback: case-insensitive term frequency proxy
        vocab = {}
        def tfidf(term, row_idx):
            low = texts[row_idx].lower()
            return float(low.count(term.lower()))

    # compile weighted term list
    wt_terms = []
    for t in protocol["pico_tokens"]["P"]:
        wt_terms.append( (t, WEIGHTS["P"]) )
    for t in protocol["pico_tokens"]["I"]:
        wt_terms.append( (t, WEIGHTS["I"]) )
    for t in protocol["pico_tokens"]["C"]:
        wt_terms.append( (t, WEIGHTS["C"]) )
    for t in protocol["pico_tokens"]["O"]:
        wt_terms.append( (t, WEIGHTS["O"]) )
    for t in protocol["anchors_must_have"]:
        wt_terms.append( (t, WEIGHTS["ANCHOR"]) )
    for t in protocol["avoid_terms"]:
        wt_terms.append( (t, WEIGHTS["AVOID"]) )

    scored = []
    for i, rec in enumerate(records):
        s = 0.0
        for term, w in wt_terms:
            if not term: continue
            s += float(tfidf(term, i)) * w
        scored.append( (s, rec) )
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# ----------------------------
# Screener prompts & validators
# ----------------------------
SCREEN_SYS = """You are a strict but realistic title+abstract screener for an evidence scan.

Checklist logic (INCLUDE requires P & I true AND (O OR D) true):
- P (Population/Context): study matches the target clinical context; synonyms acceptable.
- I (Intervention): intercostal nerve cryoablation / cryoanalgesia used intraoperatively for the target surgery; synonyms acceptable.
- O (Outcomes): any acute postoperative analgesia outcomes acceptable (pain, opioid use, LOS, early complications). Do NOT require exact day windows at abstract level unless protocol explicitly demands it.
- D (Design): randomized/comparative preferred; strong cohorts acceptable if protocol allows. Use PubTypes if available; otherwise infer from abstract.

Return ONLY JSON with schema below; be conservative but do not nitpick details that require full-text.
If the record is clearly pediatric while protocol is adults-only (or vice-versa), you may EXCLUDE for population mismatch."""

def screen_user(protocol: Dict[str,Any], record: Dict[str,Any]) -> str:
    return f"""Protocol (narrative):
{protocol["narrative_question"]}

Key lists:
P: {protocol["pico_tokens"]["P"]}
I: {protocol["pico_tokens"]["I"]}
C: {protocol["pico_tokens"]["C"]}
O: {protocol["pico_tokens"]["O"]}
Design preference: {protocol["designs_preference"]}
Anchors: {protocol["anchors_must_have"]}
Avoid: {protocol["avoid_terms"]}
Inclusion criteria: {protocol["inclusion_criteria"]}
Exclusion criteria: {protocol["exclusion_criteria"]}
Screening notes: {protocol["screening_rules_note"]}

Record:
PMID: {record['pmid']}
Title: {record['title']}
PubTypes: {record.get('pubtypes',[])}
MeSH: {record.get('mesh',[])}
Abstract:
{record.get('abstract','')}

Return schema:
{{
  "pmid": "{record['pmid']}",
  "decision": "INCLUDE|BORDERLINE|EXCLUDE",
  "why": "<one concise reason>",
  "checklist": {{"P": true|false, "I": true|false, "O": true|false, "D": true|false}},
  "mesh_roles": [{{"mesh":"...","role":"P|I|C|O|G"}}]
}}"""

def validate_screen(js: Dict[str,Any]) -> Tuple[bool,str]:
    try:
        if js.get("decision") not in ["INCLUDE","BORDERLINE","EXCLUDE"]:
            return False, "bad decision"
        ch = js.get("checklist",{})
        for k in ["P","I","O","D"]:
            if not isinstance(ch.get(k), bool):
                return False, f"checklist.{k} must be bool"
        m = js.get("mesh_roles",[])
        if not isinstance(m, list):
            return False, "mesh_roles must be list"
        for it in m:
            if not isinstance(it, dict): return False, "mesh_roles items must be dict"
            if "mesh" not in it or "role" not in it: return False, "mesh_roles items need mesh & role"
        return True, ""
    except Exception as e:
        return False, f"exception in screen validation: {e}"

# Senior plausibility check
PLAUS_SYS = """You are a senior reviewer validating junior screening decisions to prevent topic drift.
Given the protocol and an already-INCLUDED record, answer PASS if the record’s core topic clearly matches the protocol’s core P+I context; otherwise FAIL.
Be brief and conservative. Return JSON only with {"pmid":"...","verdict":"PASS|FAIL","why":"..."}"""

def plaus_user(protocol: Dict[str,Any], record: Dict[str,Any]) -> str:
    core = f"P core terms: {protocol['pico_tokens']['P']} ; I core terms: {protocol['pico_tokens']['I']} ; Anchors: {protocol['anchors_must_have']}"
    return f"""Protocol core:
{core}

Record:
PMID: {record['pmid']}
Title: {record['title']}
PubTypes: {record.get('pubtypes',[])}
MeSH: {record.get('mesh',[])}
Abstract:
{record.get('abstract','')}

BEGIN_JSON
{{"pmid":"{record['pmid']}", "verdict":"PASS", "why":""}}
END_JSON"""

def validate_plaus(js: Dict[str,Any]) -> Tuple[bool,str]:
    v = js.get("verdict")
    if v not in ["PASS","FAIL"]: return False, "verdict must be PASS|FAIL"
    if "pmid" not in js: return False, "missing pmid"
    return True, ""

# ----------------------------
# State machine
# ----------------------------
def state1_protocol_lockdown(nlq: str) -> Dict[str,Any]:
    print("[S1] Protocol lockdown...")
    system = PROTO_SYSTEM
    # Add explicit guardrails/examples to discourage long tokens
    user = proto_user(nlq, KB) + """

Guidance:
- BAD token example: "intraoperative intercostal nerve cryoablation for analgesia"
- GOOD tokens: ["intercostal nerve","cryoablation","cryoanalgesia","INC","analgesia"]
"""
    proto = get_validated_json(QWEN_MODEL, system, user, validate_protocol, retries=3, max_tokens=2048)
    if proto.get("needs_clarification"):
        raise SystemExit("Protocol needs clarification: " + proto.get("clarification_request",""))
    print("  [S1] Locked protocol:")
    print("   ", json.dumps(proto, ensure_ascii=False))
    return proto

def state2_universe(protocol, window=(50, 10000), max_tries=3):
    """
    Build the Universe Query from P/I anchors and must_have.
    Try remediate if count not in window. Logs query each attempt.
    Returns (universe_query, hit_count, id_list)
    """
    P = protocol["pico_tokens"]["P"]
    I = protocol["pico_tokens"]["I"]
    anchors = protocol.get("anchors_must_have", [])

    # Build initial query
    def tiab_or(terms):
        toks=[]
        for t in terms:
            t=t.strip()
            if not t: 
                continue
            toks.append(f"\"{t}\"[tiab]" if (" " in t or "-" in t) else f"{t}[tiab]")
        return "(" + " OR ".join(toks) + ")" if toks else ""

    Pq = tiab_or(P)
    Iq = tiab_or(I)
    Aq = tiab_or(anchors) if anchors else ""

    core = f"{Pq} AND {Iq}" if Pq and Iq else (Pq or Iq)
    if Aq:
        core = f"{core} AND {Aq}" if core else Aq
    if not core:
        raise SystemExit("Fatal: could not build universe core from protocol P/I/anchors.")

    # Try window sizing
    query = core
    for t in range(max_tries):
        cnt, ids = esearch_count_and_ids(query, protocol["deterministic_filters"]["year_min"])
        log("S2", f"[Universe] try={t} count={cnt} window={window} query={query}")
        if window[0] <= cnt <= window[1]:
            return query, cnt, ids

        # remediate scope (broaden or anchor) with guardrails
        direction = "broaden" if cnt < window[0] else "shrink"
        query = remediate_scope(query, cnt, protocol, direction=direction)

    # Last attempt counts anyway (to avoid hidden state)
    cnt, ids = esearch_count_and_ids(query, protocol["deterministic_filters"]["year_min"])
    log("S2", f"[Universe] final count={cnt} window={window} query={query}")
    return query, cnt, ids


# ----------------------------
# Deterministic prefilter for LLM screening input
# ----------------------------
PUBTYPE_ALLOWLIST_BASE = {
    # Narrow core
    "Randomized Controlled Trial",
    "Controlled Clinical Trial",
    "Clinical Trial",
    "Clinical Trial, Phase II",
    "Clinical Trial, Phase III",
    "Comparative Study",
    # Broader comparative/observational
    "Prospective Studies",
    "Retrospective Studies",
    "Cohort Studies",
    "Case-Control Studies",
    "Observational Study",
}

def deterministic_prefilter(records, protocol, extra_allow_pubtypes=None, tag="Prefilter"):
    """
    Apply deterministic filters BEFORE sending to the LLM screener:
      - language ∈ protocol.deterministic_filters.languages
      - year >= protocol.deterministic_filters.year_min
      - publication_types ∩ allowlist != ∅
    Logs a full diagnostic: distributions & exclusion reasons.

    Returns: (kept_records, diag)
      diag = {"before":N, "after":M, "reasons":{reason:count}, "pubtypes_dist":{...}, "lang_dist":{...}, "year_min":..., "allow_pubtypes":[...]}
    """
    # Pull filters from protocol
    df = protocol.get("deterministic_filters", {})
    allowed_langs = set(x.lower() for x in df.get("languages", []))
    year_min = int(df.get("year_min", 0))

    # Build allowlist for pubtypes
    allow_pubtypes = set(PUBTYPE_ALLOWLIST_BASE)
    # If designs_preference is in KB, we keep base as a minimum; you can expand if you want:
    # e.g., if preference is RCT, the base already contains RCTs + reasonable comparators.
    if extra_allow_pubtypes:
        allow_pubtypes |= set(extra_allow_pubtypes)

    # --- Diagnostics on the raw corpus
    from collections import Counter
    dist_pub = Counter()
    dist_lang = Counter()
    years = []

    for r in records:
        for pt in (r.get("publication_types") or []):
            dist_pub[pt] += 1
        lang = (r.get("language") or "").lower()
        if lang:
            dist_lang[lang] += 1
        y = r.get("year")
        if isinstance(y, int):
            years.append(y)

    diag = {
        "before": len(records),
        "pubtypes_dist": dict(dist_pub.most_common(25)),
        "lang_dist": dict(dist_lang.most_common()),
        "year_min": year_min,
        "allow_pubtypes": sorted(list(allow_pubtypes)),
    }

    if LOG_VERBOSE:
        log_json(tag, {
            "corpus_overview": {
                "n": len(records),
                "year_range": (min(years) if years else None, max(years) if years else None),
                "pubtypes_top": diag["pubtypes_dist"],
                "languages": diag["lang_dist"],
            },
            "filters": {
                "languages": sorted(list(allowed_langs)),
                "year_min": year_min,
                "allow_pubtypes": sorted(list(allow_pubtypes)),
            }
        })

    # --- Apply filters with reason tracking
    kept = []
    reasons = Counter()

    for r in records:
        # language
        lang = (r.get("language") or "").lower()
        if allowed_langs and lang not in allowed_langs:
            reasons["language"] += 1
            continue
        # year
        y = r.get("year")
        if isinstance(y, int) and y < year_min:
            reasons["year"] += 1
            continue
        # pubtype allowlist (intersection)
        pts = set(r.get("publication_types") or [])
        if pts.isdisjoint(allow_pubtypes):
            reasons["pubtype"] += 1
            continue
        kept.append(r)

    diag["after"] = len(kept)
    diag["reasons"] = dict(reasons)

    log(tag, f"Deterministic prefilter: before={diag['before']} after={diag['after']}")
    if reasons:
        log_json(tag, {"exclusion_reasons": diag["reasons"]})

    # If everything got filtered, emit a more verbose hint
    if not kept:
        # Show top 10 pmids from original with their pubtypes to help debug
        sample_meta = []
        for r in records[:10]:
            sample_meta.append({
                "pmid": r.get("pmid"),
                "year": r.get("year"),
                "lang": r.get("language"),
                "pubtypes": r.get("publication_types"),
                "title": (r.get("title") or "")[:140]
            })
        log_json(tag, {"no_records_after_filter_hint": sample_meta})

    return kept, diag


def state2_5_rerank_universe(universe_query, universe_ids, protocol, fetch_n=600):
    """
    Fetch a larger sample, log corpus stats, apply deterministic prefilter,
    then build TF-IDF and compute PICO-weighted relevance scores.
    Returns: (reranked_records, prefilter_diag)
    """
    # Fetch
    ids_sample = universe_ids[:min(fetch_n, len(universe_ids))]
    xml = efetch_xml(ids_sample)
    records = parse_pubmed_xml(xml)

    log("S2.5", f"Fetched corpus for rerank: n={len(records)} (requested up to {fetch_n})")

    # Deterministic prefilter (language/year/pubtypes)
    # You can pass extra allowed pubtypes if you want; here we stick to base allowlist
    kept, pre_diag = deterministic_prefilter(records, protocol, extra_allow_pubtypes=None, tag="Prefilter")

    if not kept:
        # Make the failure self-explanatory
        log("S2.5", "No records passed deterministic prefilter.")
        raise SystemExit("Fatal: no records after deterministic prefilter.")

    # Build TF-IDF on kept corpus
    texts = []
    for r in kept:
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        texts.append((title + " " + abstract).strip())

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), lowercase=True)
    X = vec.fit_transform(texts)
    vocab = vec.vocabulary_

    # Prepare weighted token list
    proto = protocol["pico_tokens"]
    weights = []
    def tok_weight(t):
        t = t.lower().strip()
        # P/I higher, O medium, C lower, avoid negative
        if t in [x.lower() for x in (proto.get("P") or [])]: return 1.5
        if t in [x.lower() for x in (proto.get("I") or [])]: return 1.5
        if t in [x.lower() for x in (proto.get("O") or [])]: return 1.0
        if t in [x.lower() for x in (proto.get("C") or [])]: return 0.6
        return 0.0

    # Build a sparse query vector as sum of term columns * weights
    import numpy as np
    q = np.zeros(X.shape[1], dtype=float)
    all_terms = (proto.get("P") or []) + (proto.get("I") or []) + (proto.get("O") or []) + (proto.get("C") or [])
    present_terms = []
    for t in all_terms:
        w = tok_weight(t)
        if w <= 0: 
            continue
        idx = vocab.get(t.lower())
        if idx is not None:
            q[idx] += w
            present_terms.append((t, w))

    if LOG_VERBOSE:
        log_json("S2.5", {"tfidf_query_terms_used": present_terms[:20]})

    # Score = X dot q
    scores = X.dot(q)
    scores = np.asarray(scores).ravel()

    # Attach scores and sort
    for r, s in zip(kept, scores):
        r["_score"] = float(s)

    kept.sort(key=lambda r: r.get("_score", 0.0), reverse=True)

    # Preview top 10 after rerank
    preview = []
    for r in kept[:10]:
        preview.append({
            "pmid": r.get("pmid"),
            "score": round(r.get("_score", 0.0), 4),
            "year": r.get("year"),
            "lang": r.get("language"),
            "pubtypes": r.get("publication_types"),
            "title": (r.get("title") or "")[:140]
        })
    log_json("S2.5", {"rerank_preview_top10": preview})

    return kept, pre_diag


def state3_ground_truth(reranked_records: List[Dict[str,Any]], protocol: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    print("[S3] Ground-truth discovery & vocabulary mining...")
    to_screen = reranked_records[:SCREEN_TOP_K]
    includes=[]; borderlines=[]
    for r in to_screen:
        
        js = get_validated_json(SCREENER_MODEL, SCREEN_SYS, screen_user(protocol, r), validate_screen, retries=SCREENER_RETRY_MAX, max_tokens=1536)
        d = js.get("decision")
        why = js.get("why","")
        chk = js.get("checklist",{})
        # concise logging
        short_abs = (r.get("abstract","")[:320] + "…") if r.get("abstract") and len(r["abstract"])>320 else (r.get("abstract","") or "")
        print(f"  [Screen] PMID {r['pmid']} -> decision={d} checklist={chk} why={why}")
        print(f"    Title: {r['title']}")
        print(f"    Abstract: {short_abs}")
        if d=="INCLUDE":
            # attach mesh_roles if any
            r["_mesh_roles"] = js.get("mesh_roles",[])
            includes.append(r)
        elif d=="BORDERLINE":
            r["_mesh_roles"] = js.get("mesh_roles",[])
            borderlines.append(r)
    return includes, borderlines

def state3_5_plausibility(includes: List[Dict[str,Any]], protocol: Dict[str,Any]) -> List[Dict[str,Any]]:
    print("[S3.5] Senior plausibility check (guard against topic drift)...")
    confirmed=[]
    for r in includes:
        js = get_validated_json(QWEN_MODEL, PLAUS_SYS, plaus_user(protocol, r), validate_plaus, retries=2, max_tokens=768)
        if js.get("verdict")=="PASS":
            confirmed.append(r)
        else:
            print(f"   [Plausibility] DROP PMID {r['pmid']} — {js.get('why','')}")
    return confirmed

def mesh_vernaculum_from(includes: List[Dict[str,Any]]) -> Dict[str,List[str]]:
    roles = {"P":set(),"I":set(),"C":set(),"O":set(),"G":set()}
    for r in includes:
        for mr in r.get("_mesh_roles",[]):
            m = mr.get("mesh"); role = mr.get("role","G")
            if m and role in roles:
                roles[role].add(m)
    return {k:sorted(v) for k,v in roles.items()}

def state4_validate_strategy(universe_query: str, confirmed_includes: List[Dict[str,Any]], protocol: Dict[str,Any], vernac: Dict[str,List[str]]) -> Dict[str,str]:
    print("[S4] Search-strategy validation & refinement...")
    # Build "topic_filter" deterministically from vernaculum (use P+I+O meshes as TIAB surface tokens)
    topic_tokens = list(dict.fromkeys(vernac.get("P",[]) + vernac.get("I",[]) + vernac.get("O",[])))
    topic_filter = or_block(topic_tokens, "tiab") if topic_tokens else ""
    # Build design filter deterministically from protocol preference (map to aliases)
    pref = protocol["designs_preference"]
    aliases = KB["pubtype_aliases"].get(pref, [pref])
    # recommended_filters are strings meant to be combined during Harvest:
    #   final_query := (universe_query) AND (topic_filter)  then apply 'design_filter' at execution time
    recommended = {
        "topic_filter": topic_filter,
        "design_filter": " OR ".join(f'"{a}"[Publication Type]' for a in aliases)
    }

    # Validation: recall of includes
    # We check that each include is still retrievable with (universe AND topic_filter)
    recall_ok = True
    for r in confirmed_includes:
        # cheap check: topic_filter tokens appear in title/abstract (proxy for final execution)
        if topic_filter:
            any_tok = False
            low = (r.get("title","") + " " + r.get("abstract","")).lower()
            # parse tokens out of the topic_filter string approximately
            toks = re.findall(r'"([^"]+)"\[tiab\]|(\w+)\[tiab\]', topic_filter)
            flat = [a or b for a,b in toks if (a or b)]
            for t in flat:
                if t.lower() in low:
                    any_tok = True; break
            if not any_tok:
                recall_ok = False
                print(f"   [S4] Recall risk: topic_filter might drop PMID {r['pmid']}")

    if not recall_ok:
        print("   [S4] Relaxing topic_filter (drop vernaculum; rely on universe_query only).")
        recommended["topic_filter"] = ""  # fall back to universe-only; design filter still applied downstream

    return recommended

def state5_finalize(protocol: Dict[str,Any], universe_query: str, recommended_filters: Dict[str,str],
                    confirmed_includes: List[Dict[str,Any]], vernac: Dict[str,List[str]], warnings: List[str]) -> None:
    print("[S5] Finalization & handoff...")
    # embedding string: concise, validated question
    rq_embed = f"{protocol['narrative_question']} | P:{', '.join(protocol['pico_tokens']['P'])} I:{', '.join(protocol['pico_tokens']['I'])} O:{', '.join(protocol['pico_tokens']['O'])} Anchors:{', '.join(protocol['anchors_must_have'])}"

    artifacts = {
        "locked_protocol": protocol,
        "universe_query": universe_query,
        "recommended_filters": recommended_filters,
        "ground_truth_pmids": [r["pmid"] for r in confirmed_includes],
        "mesh_vernaculum": vernac,
        "research_question_string_for_embedding": rq_embed,
        "warnings": warnings
    }
    (OUT_DIR/"sniff_artifacts.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable report
    lines=[]
    lines.append("========= SNIFF VALIDATION ENGINE REPORT (v3.1) =========\n")
    lines.append("Protocol (narrative):\n" + textwrap.fill(protocol["narrative_question"], 100) + "\n")
    lines.append("Deterministic filters: languages=" + ", ".join(protocol["deterministic_filters"]["languages"]) +
                 f" ; year_min={protocol['deterministic_filters']['year_min']}\n")
    lines.append("Universe query:\n" + universe_query + "\n")
    lines.append("Recommended filters:\n  topic_filter=" + (recommended_filters["topic_filter"] or "<none>") +
                 "\n  design_filter=" + recommended_filters["design_filter"] + "\n")
    lines.append(f"Ground truth includes (n={len(artifacts['ground_truth_pmids'])}): " + ", ".join(artifacts["ground_truth_pmids"]) + "\n")
    lines.append("MeSH vernaculum (from includes only):\n" + json.dumps(vernac, indent=2, ensure_ascii=False) + "\n")
    if warnings:
        lines.append("WARNINGS:\n- " + "\n- ".join(warnings) + "\n")
    (OUT_DIR/"sniff_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote:", OUT_DIR/"sniff_artifacts.json", "and", OUT_DIR/"sniff_report.txt")

# ----------------------------
# Orchestration
# ----------------------------
def sniff_engine_run(USER_NLQ: str):
    warnings=[]
    # S1
    protocol = state1_protocol_lockdown(USER_NLQ)

    # S2
    universe_query, universe_count, universe_ids = state2_universe(protocol)

    # S2.5
    reranked = state2_5_rerank_universe(universe_query, universe_ids, protocol)
    if not reranked:
        raise SystemExit("Fatal: no records after deterministic prefilter.")

    # S3
    includes, borderlines = state3_ground_truth(reranked, protocol)
    if not includes:
        raise SystemExit("Fatal: no includes after screening. Revisit protocol or universe scope.")

    # S3.5
    confirmed = state3_5_plausibility(includes, protocol)
    if len(confirmed) < PLAUSIBILITY_MIN_INCLUDES:
        raise SystemExit(f"Fatal: insufficient confirmed includes after plausibility ({len(confirmed)}<{PLAUSIBILITY_MIN_INCLUDES}).")

    # vernaculum strictly from confirmed includes
    vernac = mesh_vernaculum_from(confirmed)

    # S4
    recommended_filters = state4_validate_strategy(universe_query, confirmed, protocol, vernac)

    # S5
    state5_finalize(protocol, universe_query, recommended_filters, confirmed, vernac, warnings)

# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    USER_NLQ = """
Population = children/adolescents undergoing minimally invasive repair of pectus excavatum (Nuss/MIRPE).
Intervention = intercostal nerve cryoablation (INC) used intraoperatively for analgesia during Nuss/MIRPE.
Comparators = thoracic epidural, paravertebral block, intercostal nerve block, erector spinae plane block, or systemic multimodal analgesia.
Outcomes = postoperative opioid consumption (in-hospital and at discharge) and pain scores within 0–7 days (abstract-level timing not strictly required).
Study designs = RCTs preferred; if absent, include comparative cohorts/case-control/observational.
Year_min = 2015.
Languages = English, Portuguese, Spanish.
Screening notes: Be conservative; INCLUDE if P & I present and (O or D) is present; do not exclude for lack of exact day window if acute postop outcomes are clearly reported.
"""
    sniff_engine_run(USER_NLQ.strip())
