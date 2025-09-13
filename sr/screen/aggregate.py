# sr/screen/aggregate.py
from __future__ import annotations
from typing import List
from sr.config.schema import PassAResult, PassBResult, PassCResult, FinalDecision, Record, Protocol, RRFScore, Reason
from sr.screen.verify import quotes_ok as _quotes_ok, design_ok as _design_ok

def dissonant(a: PassAResult, rrf_score: float, ees_score: float | None, low_rrf_thresh: float = 0.05) -> bool:
    # Example: Include but bottom RRF decile -> suspicious
    if a.decision == "include" and rrf_score < low_rrf_thresh:
        return True
    return False

def aggregate(proto: Protocol, rec: Record, rrf: RRFScore, a: PassAResult, b: PassBResult | None, c: PassCResult | None) -> FinalDecision:
    q_ok = _quotes_ok(rec, a)
    d_ok = _design_ok(rec, proto)
    passes = ["A"] + (["B"] if b else []) + (["C"] if c else [])

    # Consensus logic
    final_decision = a.decision
    final_conf = a.confidence
    final_reason: Reason = a.reason

    if b and b.stance == "challenge":
        if not c:
            # Without C, downgrade to borderline unless hard mismatch
            if a.reason in ("admin","language","year","animal_preclinical","design_ineligible"):
                final_decision = "exclude"
            else:
                final_decision = "borderline"
        else:
            final_decision = c.decision
            final_reason = c.reasons[0] if c.reasons else a.reason
            final_conf = c.confidence

    # Acceptance contract
    if final_decision == "include":
        if not (q_ok and d_ok and final_conf >= proto.accept_confidence_tau):
            final = "borderline"
        else:
            final = "include_for_full_text"
    elif final_decision == "exclude":
        final = "exclude"
    else:
        final = "borderline"

    # Justification policy
    justification = a.justification_short
    if final == "exclude" and a.reason in ("admin","language","year","animal_preclinical"):
        # templated is fine, already provided
        pass

    return FinalDecision(
        pmid=rec.pmid, final=final, reason=final_reason, justification=justification,
        quotes_ok=q_ok, design_ok=d_ok, passes_triggered=passes, rrf=rrf
    )
