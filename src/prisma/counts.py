from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter
from src.config.schema import LedgerRow

def prisma_counts(ledger: List[LedgerRow]) -> Dict[str, Any]:
    N = len(ledger)
    decided = Counter(r.final_decision for r in ledger)
    reasons = Counter(r.final_reason for r in ledger)
    topic = Counter((r.topic_relevance or "unknown") for r in ledger)
    return {
        "records_screened": N,
        "included": decided.get("include", 0),
        "excluded": decided.get("exclude", 0),
        "borderline": decided.get("borderline", 0),
        "exclusions_by_reason": dict(reasons),
        "by_topic_relevance": dict(topic),
    }
