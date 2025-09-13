# sr/core/engine.py
from __future__ import annotations
from typing import Tuple, List
from sr.core.sniff_orchestrator import infer_protocol, sniff
from sr.core.commit_orchestrator import commit
from sr.io.runs import Runs
from sr.config.schema import Protocol, Record

def run_end_to_end(nl_prompt: str, out_dir: str):
    runs = Runs(out_dir)
    draft = infer_protocol(nl_prompt)
    if getattr(draft, "needs_reprompt", False):
        raise RuntimeError(f"Protocol requires user clarification: {draft.reprompt_reason}")

    proto: Protocol = Protocol(**draft.model_dump())
    proto, seeds = sniff(proto, runs)
    runs.save_json("sniff_seeds.json", [s.model_dump() for s in seeds])

    # Freeze protocol here (already concrete). Commit triage:
    ledger, diary = commit(proto, nl_prompt, out_dir=runs.root)
    runs.save_json("search_diary.json", diary.snapshot().model_dump())
    return ledger
