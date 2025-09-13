# sr/config/defaults.py
from __future__ import annotations
import os

# LM Studio
LMSTUDIO_BASE  = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234")
LMSTUDIO_CHAT  = os.getenv("LMSTUDIO_CHAT_MODEL", "gemma-3n-e4b-it")
LMSTUDIO_EMB   = os.getenv("LMSTUDIO_EMB_MODEL", "text-embedding-qwen3-embedding-0.6b")
HTTP_TIMEOUT   = int(os.getenv("HTTP_TIMEOUT", "30"))
USER_AGENT     = os.getenv("USER_AGENT", "sr-engine/0.1 (+local)")

# PubMed / E-utilities
ENTREZ_EMAIL   = os.getenv("ENTREZ_EMAIL", "you@example.com")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

# Retrieval
DEFAULT_YEAR_MIN         = int(os.getenv("DEFAULT_YEAR_MIN", "2015"))
RETRIEVAL_PAGE_SIZE      = int(os.getenv("RETRIEVAL_PAGE_SIZE", "10000"))  # eutils retmax per page
RETRIEVAL_MAX_PAGES      = int(os.getenv("RETRIEVAL_MAX_PAGES", "10"))     # hard cap for safety

# Screening parameters
FRONTIER_SIZE            = int(os.getenv("FRONTIER_SIZE", "100"))
PASS_A_BATCH             = int(os.getenv("PASS_A_BATCH", "50"))
INCLUDE_CONFIDENCE_TAU   = float(os.getenv("INCLUDE_CONFIDENCE_TAU", "0.62"))

# RRF
RRF_K                    = int(os.getenv("RRF_K", "60"))

# Filesystem
DATA_DIR                 = os.getenv("DATA_DIR", "data")
RUNS_DIR                 = os.getenv("RUNS_DIR", "runs")

# Languages
DEFAULT_LANGUAGES        = os.getenv("DEFAULT_LANGUAGES", "English,Portuguese,Spanish").split(",")
