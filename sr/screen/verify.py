# sr/screen/verify.py
from __future__ import annotations
from sr.config.schema import Record, PassAResult, Protocol

def _verbatim(hay: str, needle: str) -> bool:
    if not needle.strip(): return False
    return needle.strip() in (hay or "")

def quotes_ok(rec: Record, a: PassAResult) -> bool:
    hay = f"{rec.title or ''}\n{rec.abstract or ''}"
    ok_p = _verbatim(hay, a.population_quote)
    ok_i = _verbatim(hay, a.intervention_quote)
    return ok_p and ok_i

def design_ok(rec: Record, proto: Protocol) -> bool:
    if not proto.allowed_designs:
        return True
    return bool(set(rec.publication_types) & set(proto.allowed_designs))
