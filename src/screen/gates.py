from __future__ import annotations
from typing import Optional, Tuple
from src.config.schema import Document, PICOS, Reason

# Primary designs (hinting only elsewhere; NOT used to auto-exclude)
PRIMARY_TYPES = {
    "Randomized Controlled Trial","Clinical Trial","Cohort Studies",
    "Case-Control Studies","Observational Study","Controlled Clinical Trial",
    "Prospective Studies"
}

# Only true non-research / admin junk is auto-excluded here
HARD_EXCLUDE_TYPES = {
    "Editorial","Letter","Comment","News","Newspaper Article","Interview",
    "Published Erratum","Retraction of Publication","Retracted Publication",
    "Expression of Concern","Congresses"
}
# NOTE: We do NOT auto-exclude Review/Meta-Analysis/Guideline/Case Reports.

LANG_MAP = {
    "eng": "English", "en": "English", "english": "English",
    "spa": "Spanish", "es": "Spanish", "spanish": "Spanish",
    "por": "Portuguese", "pt": "Portuguese", "portuguese": "Portuguese",
}

REV_LANG = {}
for code_or_name, pretty in LANG_MAP.items():
    REV_LANG.setdefault(pretty, set()).add(code_or_name)

def _norm_lang(s: str | None) -> str | None:
    if not s: return None
    return LANG_MAP.get(s.strip().lower(), s)

def _lang_ok(doc_lang: str | None, allowed: list[str]) -> bool:
    if not doc_lang or not allowed:
        return True
    dl = _norm_lang(doc_lang)
    allowed_norm = {_norm_lang(a) or a for a in allowed}
    if dl in allowed_norm:
        return True
    for a in allowed_norm:
        for alias in REV_LANG.get(a, set()):
            if doc_lang.strip().lower() == alias:
                return True
    return False

def objective_gate(doc: Document, picos: PICOS) -> Optional[Tuple[str, Reason]]:
    # Year
    if picos.year_min and doc.year is not None and doc.year < int(picos.year_min):
        return ("year", "year")
    # Language
    if picos.languages and doc.language and not _lang_ok(doc.language, picos.languages):
        return ("language", "language")
    # Missing both title and abstract
    if (not (doc.title or "").strip()) and (not (doc.abstract or "").strip()):
        return ("insufficient_info", "insufficient_info")
    # Hard non-research types only
    if set(doc.pub_types) & HARD_EXCLUDE_TYPES:
        return ("design_mismatch", "design_mismatch")
    return None

def is_primary_design(doc: Document) -> bool:
    return bool(set(doc.pub_types) & PRIMARY_TYPES)
