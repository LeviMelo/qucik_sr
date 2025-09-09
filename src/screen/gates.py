from __future__ import annotations
from typing import Optional, Tuple
from src.config.schema import Document, PICOS, Reason

PRIMARY_TYPES = {"Randomized Controlled Trial","Clinical Trial","Cohort Studies","Case-Control Studies","Observational Study","Controlled Clinical Trial","Prospective Studies"}
NON_PRIMARY_TYPES = {"Review","Meta-Analysis","Editorial","Letter","Comment","News","Case Reports","Guideline"}
LANG_MAP = {
    "eng": "English", "en": "English", "english": "English",
    "spa": "Spanish", "es": "Spanish", "spanish": "Spanish",
    "por": "Portuguese", "pt": "Portuguese", "portuguese": "Portuguese",
}
REV_LANG = {}

for code_or_name, pretty in LANG_MAP.items():
    REV_LANG.setdefault(pretty, set()).add(code_or_name)

def _norm_lang(s: str | None) -> str | None:
    if not s:
        return None
    return LANG_MAP.get(s.strip().lower(), s)

def _lang_ok(doc_lang: str | None, allowed: list[str]) -> bool:
    if not doc_lang or not allowed:
        return True
    dl = _norm_lang(doc_lang)
    allowed_norm = {_norm_lang(a) or a for a in allowed}
    if dl in allowed_norm:
        return True
    # also accept if doc code matches one of the allowed names
    for a in allowed_norm:
        for alias in REV_LANG.get(a, set()):
            if doc_lang.strip().lower() == alias:
                return True
    return False

def objective_gate(doc: Document, picos: PICOS) -> Optional[Tuple[str, Reason]]:
    if picos.year_min and doc.year is not None and doc.year < int(picos.year_min):
        return ("year", "year")
    if picos.languages and doc.language and not _lang_ok(doc.language, picos.languages):
        return ("language", "language")
    if (not (doc.title or "").strip()) and (not (doc.abstract or "").strip()):
        return ("insufficient_info", "insufficient_info")
    if set(doc.pub_types) & NON_PRIMARY_TYPES:
        return ("design_mismatch", "design_mismatch")
    return None

def is_primary_design(doc: Document) -> bool:
    return bool(set(doc.pub_types) & PRIMARY_TYPES)
