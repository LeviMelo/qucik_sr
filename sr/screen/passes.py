# sr/screen/passes.py
from __future__ import annotations
import json
from typing import Optional
from sr.llm.client import chat_json
from sr.llm.prompts import PASS_A_SYSTEM, pass_a_user, PASS_B_SYSTEM, pass_b_user, PASS_C_SYSTEM, pass_c_user
from sr.config.schema import Protocol, Record, PassAResult, PassBResult, PassCResult

_PASS_A_TEMPLATE = """{
  "pmid": "",
  "decision": "borderline",
  "confidence": 0.0,
  "reason": "off_topic",
  "population_quote": "",
  "intervention_quote": "",
  "design_evidence": "",
  "justification_short": ""
}"""

def pass_a(proto: Protocol, rec: Record) -> PassAResult:
    pj = proto.model_dump_json()
    rj = rec.model_dump_json()
    res = chat_json(
        PASS_A_SYSTEM,
        pass_a_user(pj, rj),
        schema_model=PassAResult,
        temperature=0.0,
        max_tokens=700,
        template_for_repair=_PASS_A_TEMPLATE,   # <— repair if shapes drift
    )
    assert isinstance(res, PassAResult)
    if not res.pmid:
        res.pmid = rec.pmid
    return res

def pass_b(proto: Protocol, rec: Record, a: PassAResult, trigger_note: str) -> PassBResult:
    pj = proto.model_dump_json()
    rj = rec.model_dump_json()
    aj = a.model_dump_json()
    res = chat_json(PASS_B_SYSTEM, pass_b_user(pj, rj, aj, trigger_note), schema_model=PassBResult, temperature=0.0, max_tokens=600)
    assert isinstance(res, PassBResult)
    if not res.pmid:
        res.pmid = rec.pmid
    return res

def pass_c(proto: Protocol, rec: Record, a: PassAResult, b: PassBResult) -> PassCResult:
    pj = proto.model_dump_json()
    rj = rec.model_dump_json()
    aj = a.model_dump_json()
    bj = b.model_dump_json()
    res = chat_json(PASS_C_SYSTEM, pass_c_user(pj, rj, aj, bj), schema_model=PassCResult, temperature=0.0, max_tokens=700)
    assert isinstance(res, PassCResult)
    if not res.pmid:
        res.pmid = rec.pmid
    return res
