from __future__ import annotations
import os

# ---- External services ----
LMSTUDIO_BASE    = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234")
LMSTUDIO_EMB     = os.getenv("LMSTUDIO_EMB_MODEL", "text-embedding-qwen3-embedding-0.6b")
LMSTUDIO_CHAT    = os.getenv("LMSTUDIO_CHAT_MODEL", "gemma-3n-e2b-it")

ENTREZ_EMAIL     = os.getenv("ENTREZ_EMAIL", "you@example.com")
ENTREZ_API_KEY   = os.getenv("ENTREZ_API_KEY", "")
HTTP_TIMEOUT     = int(os.getenv("HTTP_TIMEOUT", "30"))
USER_AGENT       = os.getenv("USER_AGENT", "sr-screener/0.1 (+local)")

ICITE_BASE       = os.getenv("ICITE_BASE", "https://icite.od.nih.gov/api/pubs")

# ---- Retrieval ----
ESEARCH_RETMAX_PER_QUERY = int(os.getenv("ESEARCH_RETMAX_PER_QUERY", "2000"))

# ---- Retrieval curation knobs (hit-count band to keep a query) ----
ESEARCH_KEEP_MIN = int(os.getenv("ESEARCH_KEEP_MIN", "2"))       # drop brittle (0–1) queries
ESEARCH_KEEP_MAX = int(os.getenv("ESEARCH_KEEP_MAX", "50000"))   # quarantine gigantic queries
QUERY_VARIANT_CAP = int(os.getenv("QUERY_VARIANT_CAP", "60"))    # cap auto-generated variants

# ---- FS (define early; used by VEC_DB_PATH) ----
DATA_DIR         = os.getenv("DATA_DIR", "data")
RUNS_DIR         = os.getenv("RUNS_DIR", "runs")

# ---- Embeddings ----
EMB_BATCH        = int(os.getenv("EMB_BATCH", "48"))

# ---- Embedding controls / cache ----
EMB_AUTO_BATCH        = os.getenv("EMB_AUTO_BATCH", "1") == "1"
EMB_MAX_CHARS_PER_DOC = int(os.getenv("EMB_MAX_CHARS_PER_DOC", "12000"))
EMB_RETRY_BACKOFF_S   = float(os.getenv("EMB_RETRY_BACKOFF_S", "1.2"))
EMB_RETRY_MAX         = int(os.getenv("EMB_RETRY_MAX", "4"))

# ---- Vector DB (sqlite) ----
VEC_DB_PATH      = os.getenv("VEC_DB_PATH", os.path.join(DATA_DIR, "cache", "vectors.sqlite3"))

# ---- Seeds & thresholds ----
SEED_SEM_TAU_HI  = float(os.getenv("SEED_SEM_TAU_HI", "0.92"))
SEED_MIN_COUNT   = int(os.getenv("SEED_MIN_COUNT", "8"))
SEED_RELAX_STEP  = float(os.getenv("SEED_RELAX_STEP", "0.02"))

# ---- CILE knobs ----
CILE_REL_GATE_FRAC      = float(os.getenv("CILE_REL_GATE_FRAC", "0.08"))
CILE_EXT_BUDGET         = int(os.getenv("CILE_EXT_BUDGET", "2500"))
CILE_MAX_ACCEPT         = int(os.getenv("CILE_MAX_ACCEPT", "600"))
CILE_HUB_QUARANTINE     = os.getenv("CILE_HUB_QUARANTINE", "1") == "1"
CILE_MIN_HUB_SOFT       = int(os.getenv("CILE_MIN_HUB_SOFT", "450"))

# ---- Regressor thresholds ----
REG_P_HI         = float(os.getenv("REG_P_HI", "0.85"))
REG_P_LO         = float(os.getenv("REG_P_LO", "0.15"))

# ---- LLM budget ----
LLM_BUDGET       = int(os.getenv("LLM_BUDGET", "150"))

# ---- Languages & year defaults ----
DEFAULT_LANGS    = os.getenv("DEFAULT_LANGS", "English,Portuguese,Spanish").split(",")
DEFAULT_YEAR_MIN = int(os.getenv("DEFAULT_YEAR_MIN", "2000"))

# ---- FS ----
DATA_DIR         = os.getenv("DATA_DIR", "data")
RUNS_DIR         = os.getenv("RUNS_DIR", "runs")
