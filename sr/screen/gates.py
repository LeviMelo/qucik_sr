# sr/screen/gates.py
from __future__ import annotations
from typing import Optional, Tuple
from sr.config.schema import Record, Protocol, Reason

ADMIN = {"Editorial","Letter","Comment","News","Interview","Published Erratum","Retraction of Publication","Retracted Publication","Expression of Concern","Newspaper Article","Congresses"}
ADJUNCT = {"Review","Meta-Analysis","Practice Guideline","Guideline","Case Reports"}
ANIMAL_HINTS = {"rat","mouse","mice","murine","canine","porcine","pig","rabbit","zebrafish","in vitro","rodent"}

def language_ok(rec_lang: Optional[str], allowed: list[str]) -> bool:
    if not rec_lang or not allowed: return True
    rl= (rec_lang or "").strip().lower()
    return any(rl==a.strip().lower() for a in allowed)

def title_animal_offtopic(title: str) -> Optional[Reason]:
    tl= (title or "").lower()
    if any(tok in tl for tok in ANIMAL_HINTS): return "animal_preclinical"
    return None

def apply_gates(rec: Record, proto: Protocol) -> Optional[Tuple[Reason,str]]:
    # Year (already gated at retrieval typically, but keep)
    if proto.picos.year_min and rec.year is not None and rec.year < int(proto.picos.year_min):
        return ("year","year below protocol")
    # Language
    if proto.picos.languages and rec.language and not language_ok(rec.language, proto.picos.languages):
        return ("language","language outside protocol")
    # Missing both title and abstract
    if not ((rec.title or "").strip() or (rec.abstract or "").strip()):
        return ("insufficient_info","missing title and abstract")
    # Admin
    if set(rec.publication_types) & ADMIN:
        return ("admin","administrative/non-research type")
    # Adjunct families dropped for effects triage
    if proto.drop_adjuncts and (set(rec.publication_types) & ADJUNCT):
        return ("design_ineligible","adjunct family (review/meta/guideline/case report)")
    # Animals/obvious off-topic (title-only cautious)
    t_reason = title_animal_offtopic(rec.title or "")
    if t_reason:
        return (t_reason, "animal/preclinical hint in title")
    return None
