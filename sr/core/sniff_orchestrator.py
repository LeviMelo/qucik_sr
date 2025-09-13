# sr/core/sniff_orchestrator.py
from __future__ import annotations
from typing import Tuple, List
from sr.config.schema import Protocol, PICOS, Record
from sr.llm.client import chat_json
from sr.llm.prompts import PROTOCOL_SYSTEM, protocol_user, _protocol_template_json
from sr.retrieval.pubmed import esearch_paged, efetch_abstracts, to_records
from sr.retrieval.dedupe import dedupe
from sr.io.runs import Runs
from sr.config.defaults import DEFAULT_YEAR_MIN, DEFAULT_LANGUAGES
from sr.screen.passes import pass_a
import logging
from collections import Counter

log = logging.getLogger("sniff")


class ProtocolDraft(Protocol):  # pydantic inheritance OK
    needs_reprompt: bool = False
    reprompt_reason: str = ""

def infer_protocol(nl: str) -> ProtocolDraft:
    # Supply the template to the repair step so tiny models can be nudged into shape
    res = chat_json(
        PROTOCOL_SYSTEM,
        protocol_user(nl),
        schema_model=ProtocolDraft,
        temperature=0.0,
        max_tokens=900,
        template_for_repair=_protocol_template_json(),
    )
    assert isinstance(res, ProtocolDraft)

    # Minimal defaults
    if not res.picos.year_min:
        res.picos.year_min = DEFAULT_YEAR_MIN
    if not res.picos.languages:
        res.picos.languages = DEFAULT_LANGUAGES

    # Enforce: retrieval_plan must not be empty (no silent auto-fill)
    if not res.retrieval_plan or len(res.retrieval_plan) == 0:
        res.needs_reprompt = True
        res.reprompt_reason = "retrieval_plan is empty; provide at least 'broad' and 'focused' PubMed queries built from Population and Intervention title/abstract terms."

    # Also check if the two required keys exist but are empty strings
    if not res.needs_reprompt:
        rp = res.retrieval_plan or {}
        missing = []
        for key in ("broad", "focused"):
            if key not in rp or not isinstance(rp[key], str) or not rp[key].strip():
                missing.append(key)
        if missing:
            res.needs_reprompt = True
            res.reprompt_reason = f"retrieval_plan missing or empty for: {', '.join(missing)}."

    return res


def sniff(proto: Protocol, runs: Runs, pilot_cap: int = 1200, top_k: int = 60, min_primaries: int = 3) -> Tuple[Protocol, List[Record]]:
    # --- Retrieval across all queries (no fallback) ---
    qmap = proto.retrieval_plan or {}
    if not qmap:
        log.error("[sniff] retrieval_plan is empty -> 0 seeds is expected. Aborting sniff early.")
        runs.save_json("retrieval_debug.json", {
            "error": "empty_retrieval_plan",
            "protocol_snapshot": proto.model_dump()
        })
        return proto, []

    ids_all: List[str] = []
    per_query_hits = []
    for name, q in qmap.items():
        try:
            ids = esearch_paged(q, mindate=proto.picos.year_min)
        except Exception as e:
            log.exception(f"[sniff] esearch failed for query '{name}': {e}")
            ids = []
        per_query_hits.append({"name": name, "hits": len(ids)})
        ids_all.extend(ids)

    # Dedupe IDs and cap
    ids_all = list(dict.fromkeys(ids_all))
    log.info(f"[sniff] queries={len(qmap)} | total_ids={len(ids_all)} | per_query_hits={per_query_hits}")

    runs.save_json("retrieval_debug.json", {
        "queries": qmap,
        "per_query_hits": per_query_hits,
        "total_ids": len(ids_all),
        "pilot_cap": pilot_cap
    })

    if not ids_all:
        log.warning("[sniff] 0 IDs after retrieval. Check retrieval_debug.json and your protocol prompt.")
        return proto, []

    ids_all = ids_all[:pilot_cap]

    # --- Fetch + dedupe ---
    raw = efetch_abstracts(ids_all)
    recs = dedupe(to_records(raw))
    log.info(f"[sniff] efetch raw={len(raw)} | unique_records={len(recs)}")

    if not recs:
        log.warning("[sniff] 0 records after efetch/dedupe.")
        return proto, []

    # --- Candidate ordering: simple P&I presence over title+abstract (for visibility only) ---
    def _hits(text: str, terms: list[str]) -> int:
        tl = (text or "").lower()
        return sum(1 for s in terms if s and s.strip() and s.lower() in tl)

    pop_terms = [proto.picos.population] + (proto.picos.synonyms_population or [])
    int_terms = [proto.picos.intervention] + (proto.picos.synonyms_intervention or [])

    def score_ta(r: Record) -> int:
        t = (r.title or "") + "\n" + (r.abstract or "")
        return _hits(t, pop_terms) + _hits(t, int_terms)

    recs_sorted = sorted(recs, key=lambda r: (-score_ta(r), r.pmid))
    candidates = recs_sorted[:top_k]
    log.info(f"[sniff] candidates_for_passA={len(candidates)} (top_k={top_k})")

    # Dump candidates TSV for quick eyeballing
    try:
        p = runs.path("sniff_candidates.tsv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("pmid\tyear\tpubtypes\tscore_ta\ttitle\n")
            for r in candidates:
                st = score_ta(r)
                t = (r.title or "").replace("\t", " ").replace("\n", " ")
                f.write(f"{r.pmid}\t{r.year or ''}\t{';'.join(r.publication_types)}\t{st}\t{t[:160]}\n")
        log.info(f"[sniff] wrote {p}")
    except Exception as e:
        log.warning(f"[sniff] failed to write sniff_candidates.tsv: {e}")

    # --- Pass A over candidates (no behavior changes) ---
    includes: List[Record] = []
    pass_a_rows = []
    reason_counter = Counter()
    decision_counter = Counter()

    for r in candidates:
        try:
            a = pass_a(proto, r)
        except Exception as e:
            log.exception(f"[sniff] Pass-A crashed for pmid={r.pmid}: {e}")
            continue

        decision_counter[a.decision] += 1
        reason_counter[a.reason] += 1

        pass_a_rows.append({
            "pmid": r.pmid,
            "decision": a.decision,
            "confidence": a.confidence,
            "reason": a.reason,
            "title": (r.title or "")[:160]
        })

        if a.decision == "include":
            includes.append(r)
            if len(includes) >= min_primaries:
                break

    log.info(f"[sniff] Pass-A decisions: {dict(decision_counter)} | reasons_top={reason_counter.most_common(6)}")
    log.info(f"[sniff] seeds_selected={len(includes)} (min_primaries={min_primaries})")

    # Dump pass-a debug TSV
    try:
        p = runs.path("pass_a_debug.tsv")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write("pmid\tdecision\tconfidence\treason\ttitle\n")
            for row in pass_a_rows:
                f.write(f"{row['pmid']}\t{row['decision']}\t{row['confidence']:.3f}\t{row['reason']}\t{row['title'].replace('\t',' ')}\n")
        log.info(f"[sniff] wrote {p}")
    except Exception as e:
        log.warning(f"[sniff] failed to write pass_a_debug.tsv: {e}")

    if not includes:
        log.warning("[sniff] 0 seeds after Pass-A. Inspect pass_a_debug.tsv and sniff_candidates.tsv to see decisions and reasons.")

    return proto, includes
