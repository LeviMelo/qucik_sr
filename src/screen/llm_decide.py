from __future__ import annotations
from typing import List, Dict, Any
import math
from src.config.schema import Criteria, Document, DecisionLLM, TopicRelevance
from src.text.llm import chat_json
from src.text.prompts import P1_SYSTEM
from src.screen.features import signal_card

_ALLOWED = {
    "design_mismatch","population_mismatch","intervention_mismatch",
    "language","year","insufficient_info","off_topic"
}

_TOPIC = {
    "primary_rct","primary_observational",
    "adjacent_meta_analysis","adjacent_review","adjacent_guideline","adjacent_case_report",
    "background","off_topic","unknown"
}

DECISION_SCHEMA: Dict[str, Any] = {
    "name": "decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pmid": {"type": "string"},
            "decision": {"type": "string", "enum": ["include","exclude","borderline"]},
            "primary_reason": {"type": "string", "enum": sorted(list(_ALLOWED))},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "topic_relevance": {"type": "string", "enum": sorted(list(_TOPIC))},
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "population_quote": {"type": "string"},
                    "intervention_quote": {"type": "string"},
                    "design_evidence": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["population_quote","intervention_quote","design_evidence","notes"]
            }
        },
        "required": ["pmid","decision","primary_reason","confidence","topic_relevance","evidence"]
    }
}

def _norm_str(x: Any) -> str:
    if x is None: return ""
    s = str(x)
    return s.strip()

def _canonical_reason(txt: str) -> str:
    t = _norm_str(txt).lower()
    if t in _ALLOWED:
        return t
    if any(k in t for k in ["observational", "cohort", "case-control", "registry"]):
        return "design_mismatch"
    if any(k in t for k in ["randomized", "rct"]):
        return "design_mismatch"
    if any(k in t for k in ["population", "pediatric", "child", "children", "adult", "adolescent"]):
        return "population_mismatch"
    if any(k in t for k in ["intervention", "cryo", "neurolys", "nerve"]):
        return "intervention_mismatch"
    if "language" in t:
        return "language"
    if "year" in t or "date" in t or "older than" in t:
        return "year"
    if any(k in t for k in ["insufficient", "no abstract", "missing", "unclear"]):
        return "insufficient_info"
    if any(k in t for k in ["off-topic", "not relevant", "unrelated", "irrelevant"]):
        return "off_topic"
    return "off_topic"

def _canonical_topic(txt: str) -> TopicRelevance:
    t = _norm_str(txt).lower()
    for k in _TOPIC:
        if t == k:
            return k  # type: ignore
    # light heuristics
    if "meta" in t or "systematic" in t: return "adjacent_meta_analysis"  # type: ignore
    if "review" in t: return "adjacent_review"  # type: ignore
    if "guideline" in t: return "adjacent_guideline"  # type: ignore
    if "case report" in t: return "adjacent_case_report"  # type: ignore
    if "rct" in t or "random" in t: return "primary_rct"  # type: ignore
    if "observational" in t or "cohort" in t or "case-control" in t: return "primary_observational"  # type: ignore
    if "off" in t: return "off_topic"  # type: ignore
    if "background" in t: return "background"  # type: ignore
    return "unknown"  # type: ignore

def _normalize_evidence(e: Any) -> Dict[str,str]:
    if not isinstance(e, dict): e = {}
    return {
        "population_quote": _norm_str(e.get("population_quote", "")),
        "intervention_quote": _norm_str(e.get("intervention_quote", "")),
        "design_evidence": _norm_str(e.get("design_evidence", "")),
        "notes": _norm_str(e.get("notes", "")),
    }

def _coerce_decision(js: Dict[str, Any]) -> DecisionLLM:
    pmid = _norm_str(js.get("pmid", ""))
    decision = _norm_str(js.get("decision", "")).lower()
    if decision not in {"include","exclude","borderline"}:
        decision = "borderline"
    reason = _canonical_reason(js.get("primary_reason",""))
    try:
        conf = float(js.get("confidence", 0.5))
        if not math.isfinite(conf): conf = 0.5
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    topic = _canonical_topic(js.get("topic_relevance","unknown"))
    ev = _normalize_evidence(js.get("evidence", {}))
    return DecisionLLM(pmid=pmid, decision=decision, primary_reason=reason, confidence=conf, topic_relevance=topic, evidence=ev)

def llm_decide_batch(criteria: Criteria, docs: List[Document], signals_map: dict, temperature: float = 0.1) -> List[DecisionLLM]:
    out: List[DecisionLLM] = []
    for d in docs:
        sig = signals_map[d.pmid]
        user = (
            f"CRITERIA_JSON:\n{criteria.model_dump_json()}\n\n"
            f"RECORD:\n{d.model_dump_json()}\n\n"
            f"{signal_card(sig)}\n"
            "Return the JSON now."
        )
        try:
            raw = chat_json(P1_SYSTEM, user, temperature=temperature, max_tokens=700, schema=DECISION_SCHEMA)
        except Exception:
            raw = chat_json(P1_SYSTEM, user, temperature=0.0, max_tokens=700, schema=None)
        out.append(_coerce_decision(raw))
    return out
