from __future__ import annotations
from typing import List
from src.config.schema import Criteria, Document, DecisionLLM
from src.text.llm import chat_json
from src.text.prompts import P1_SYSTEM
from src.screen.features import signal_card

def llm_decide_batch(criteria: Criteria, docs: List[Document], signals_map: dict, temperature: float = 0.1) -> List[DecisionLLM]:
    out = []
    for d in docs:
        sig = signals_map[d.pmid]
        user = (
            f"CRITERIA_JSON:\n{criteria.model_dump_json()}\n\n"
            f"RECORD:\n{d.model_dump_json()}\n\n"
            f"{signal_card(sig)}\n"
            "Return the JSON now."
        )
        resp = chat_json(P1_SYSTEM, user, temperature=temperature, max_tokens=600)
        out.append(DecisionLLM(**resp))
    return out
