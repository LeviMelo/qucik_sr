from __future__ import annotations
from typing import Optional, Tuple
from src.config.schema import Document, PICOS, Reason

PRIMARY_TYPES = {"Randomized Controlled Trial","Clinical Trial","Cohort Studies","Case-Control Studies","Observational Study","Controlled Clinical Trial","Prospective Studies"}
NON_PRIMARY_TYPES = {"Review","Meta-Analysis","Editorial","Letter","Comment","News","Case Reports","Guideline"}

def objective_gate(doc: Document, picos: PICOS) -> Optional[Tuple[str, Reason]]:
    if picos.year_min and doc.year is not None and doc.year < int(picos.year_min):
        return ("year", "year")
    if picos.languages and doc.language and doc.language not in picos.languages:
        return ("language", "language")
    if (not (doc.title or "").strip()) and (not (doc.abstract or "").strip()):
        return ("insufficient_info", "insufficient_info")
    if set(doc.pub_types) & NON_PRIMARY_TYPES:
        return ("design_mismatch", "design_mismatch")
    return None

def is_primary_design(doc: Document) -> bool:
    return bool(set(doc.pub_types) & PRIMARY_TYPES)
