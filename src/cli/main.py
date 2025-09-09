from __future__ import annotations
import argparse, logging, time, re
from typing import Dict, List, Tuple
from collections import defaultdict

from src.config.defaults import (
    DEFAULT_LANGS, DEFAULT_YEAR_MIN, ESEARCH_RETMAX_PER_QUERY,
)
from src.config.schema import Criteria, PICOS, Document, LedgerRow
from src.io.store import ensure_run_dir, write_json, write_tsv, write_csv
from src.net.entrez import esearch, efetch_abstracts
from src.text.prompts import P0_SYSTEM, p0_user_prompt
from src.text.llm import chat_json
from src.text.embed import embed_texts
from src.screen.features import build_embeddings, compute_signals
from src.screen.gates import objective_gate
from src.screen.regressor import OnlineRegressor, featurize_row
from src.screen.llm_decide import llm_decide_batch
from src.prisma.counts import prisma_counts

log = logging.getLogger("cli")
logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

# --- Heuristic PICOS parse if P0 fails (just enough to keep the run alive) ---
def _naive_picos_from_prompt(prompt: str, langs: List[str], year_min: int) -> PICOS:
    p = "pectus excavatum OR Nuss procedure"
    # crude intervention detection
    itv = "intercostal nerve cryoablation"
    if re.search(r"\bcryo", prompt, re.I) and re.search(r"\bintercostal", prompt, re.I):
        itv = "intercostal nerve cryoablation"
    elif re.search(r"\bcryo", prompt, re.I):
        itv = "cryoablation"
    outcomes = ["LOS","opioid","pain","neuropathic"]
    study_design = ["RCT","clinical trials","cohort","case-control","observational"]
    return PICOS(
        population=p,
        intervention=itv,
        comparison=None,
        outcomes=outcomes,
        study_design=study_design,
        year_min=year_min,
        languages=langs or DEFAULT_LANGS
    )

def _fallback_pi_queries(criteria: Criteria) -> Dict[str, str]:
    pop = criteria.picos.population.strip() if criteria.picos.population else ""
    itv = criteria.picos.intervention.strip() if criteria.picos.intervention else ""
    q = {}
    if pop and itv:
        q["pi_nl"] = f"\"{pop}\" \"{itv}\""
        q["pi_and"] = f"({pop}) AND ({itv})"
    elif itv:
        q["i_only"] = f"\"{itv}\""
    elif pop:
        q["p_only"] = f"\"{pop}\""
    return q

def _p0_or_fallback(prompt: str, languages: List[str], year_min: int) -> Criteria:
    # Try P0 once; on error, synthesize a minimal Criteria so the run can proceed.
    try:
        log.info("Phase P0: generating criteria via LLM…")
        p0 = chat_json(P0_SYSTEM, p0_user_prompt(prompt), temperature=0.0, max_tokens=1200, schema=None)
        crit = Criteria.model_validate(p0)
    except Exception as e:
        log.warning(f"P0 LLM failed ({e}); using heuristic PI fallback.")
        picos = _naive_picos_from_prompt(prompt, languages or DEFAULT_LANGS, year_min or DEFAULT_YEAR_MIN)
        crit = Criteria(
            picos=picos,
            inclusion_criteria={},
            exclusion_criteria={},
            reason_taxonomy=[],
            boolean_queries={}
        )
    # apply CLI overrides
    if languages:
        crit.picos.languages = languages
    if year_min:
        crit.picos.year_min = year_min
    if not crit.picos.year_min:
        crit.picos.year_min = DEFAULT_YEAR_MIN
    log.info("Phase P0 done")
    return crit

def run(prompt: str,
        languages: List[str] | None,
        year_min: int | None,
        out_dir: str,
        max_records: int):

    t0 = time.time()
    outp = ensure_run_dir(out_dir)

    # ---------- P0 (robust) ----------
    crit = _p0_or_fallback(prompt, languages or DEFAULT_LANGS, year_min or DEFAULT_YEAR_MIN)
    write_json(outp / "criteria.json", crit.model_dump())

    # ---------- Retrieval ----------
    log.info("Retrieval: esearch…")
    queries: Dict[str, str] = dict(crit.boolean_queries or {})
    if not queries:
        log.warning("P0 produced no boolean_queries. Using PI fallback queries.")
        queries = _fallback_pi_queries(crit)

    per_query_hits: List[Dict[str, int]] = []
    per_query_trans: Dict[str, str] = {}
    all_pmids: List[str] = []
    for name, q in queries.items():
        rs = esearch(q, retmax=ESEARCH_RETMAX_PER_QUERY, mindate=crit.picos.year_min or DEFAULT_YEAR_MIN)
        per_query_trans[name] = rs.get("translation","")
        ids = rs["ids"]
        per_query_hits.append({"name": name, "hits": len(ids)})
        all_pmids.extend(ids)

    # Union, cap, fetch
    seen = set()
    pmids = []
    for p in all_pmids:
        if p not in seen:
            pmids.append(p); seen.add(p)
        if len(pmids) >= max_records:
            log.info(f"Dev cap: limiting to first {max_records} records")
            break

    docs_raw = efetch_abstracts(pmids, chunk_size=200, workers=3, use_cache=True)
    docs: List[Document] = []
    for p in pmids:
        rec = docs_raw.get(p)
        if not rec: continue
        docs.append(Document(**rec))

    write_json(outp / "identification.json", {
        "queries_from_llm": crit.boolean_queries,
        "queries_used": queries,
        "query_translations": per_query_trans,
        "query_stats": per_query_hits,
        "hits": len(docs),
        "year_min": crit.picos.year_min,
        "languages": crit.picos.languages,
    })

    if not docs:
        log.info("No documents retrieved. Done.")
        return

    # ---------- Embeddings & signals ----------
    log.info(f"Embedding {len(docs)} documents…")
    emb, idx = build_embeddings(docs)
    intent_vec = embed_texts([prompt])[0]
    signals = compute_signals(
        docs=docs, emb=emb, idx=idx, intent_vec=intent_vec,
        seed_pmids=[], year_min=crit.picos.year_min
    )

    # ---------- Gates (hard objective only) ----------
    lanes: Dict[str, str] = {}
    gate_reason: Dict[str, str] = {}
    eligible_for_llm: List[Document] = []
    for d in docs:
        g = objective_gate(d, crit.picos)
        if g:
            lanes[d.pmid] = "auto_exclude"
            gate_reason[d.pmid] = g[1]
        else:
            lanes[d.pmid] = "uncertain"
            eligible_for_llm.append(d)

    # ---------- Regressor (ranking only; NEVER excludes) ----------
    X = [featurize_row(signals[d.pmid], d) for d in docs]
    try:
        import numpy as np
        reg = OnlineRegressor()
        y0 = np.array([0 if lanes[d.pmid]=="auto_exclude" else 1 for d in docs], dtype="int32")
        reg.fit_bootstrap(np.stack(X), y0)
        p_all = reg.predict_proba(np.stack(X))
        model_p = {docs[i].pmid: float(p_all[i]) for i in range(len(docs))}
    except Exception:
        model_p = {d.pmid: 0.5 for d in docs}

    # ---------- LLM screening (ALL eligible docs go) ----------
    log.info(f"Sending {len(eligible_for_llm)} docs to LLM screening…")
    decisions = {dec.pmid: dec for dec in llm_decide_batch(crit, eligible_for_llm, signals_map=signals, temperature=0.1)}

    # ---------- Finalize ledger ----------
    ledger: List[LedgerRow] = []
    for d in docs:
        pmid = d.pmid
        lane = lanes[pmid]
        llm_dec = decisions.get(pmid)

        if lane == "auto_exclude":
            final_decision = "exclude"
            final_reason = gate_reason.get(pmid, "off_topic")
            topic_rel = "off_topic"
        else:
            if llm_dec is None:
                final_decision = "borderline"
                final_reason = "insufficient_info"
                topic_rel = "unknown"
                lane = "sent_to_llm"  # still mark it
            else:
                final_decision = llm_dec.decision
                final_reason = llm_dec.primary_reason
                topic_rel = llm_dec.topic_relevance
                lane = "sent_to_llm"

        ledger.append(LedgerRow(
            pmid=pmid,
            lane_before_llm=lane,
            gate_reason=gate_reason.get(pmid),
            model_p=model_p.get(pmid, 0.5),
            llm=llm_dec,
            final_decision=final_decision,
            final_reason=final_reason,
            topic_relevance=topic_rel,
            signals=signals[pmid],
            pub_types=d.pub_types,
            year=d.year,
            title=d.title or "",
            abstract=d.abstract or "",
        ))

    # ---------- Outputs ----------
    fieldnames = [
        "pmid","lane_before_llm","gate_reason","model_p",
        "final_decision","final_reason","topic_relevance",
        "sem_intent","sem_seed","graph_ppr_pct","graph_links_frac","year_scaled","abstract_len_bin",
        "pub_types","year","title","abstract"
    ]
    rows = []
    for r in ledger:
        rows.append({
            "pmid": r.pmid,
            "lane_before_llm": r.lane_before_llm,
            "gate_reason": r.gate_reason or "",
            "model_p": f"{r.model_p:.4f}" if r.model_p is not None else "",
            "final_decision": r.final_decision,
            "final_reason": r.final_reason,
            "topic_relevance": r.topic_relevance or "unknown",
            "sem_intent": f"{r.signals.sem_intent:.4f}",
            "sem_seed": f"{r.signals.sem_seed:.4f}",
            "graph_ppr_pct": f"{r.signals.graph_ppr_pct:.1f}",
            "graph_links_frac": f"{r.signals.graph_links_frac:.4f}",
            "year_scaled": f"{r.signals.year_scaled:.4f}",
            "abstract_len_bin": r.signals.abstract_len_bin,
            "pub_types": ";".join(r.pub_types),
            "year": r.year or "",
            "title": r.title,
            "abstract": r.abstract
        })
    write_tsv(outp / "screening.tsv", rows, fieldnames)

    adjunct_rows = []
    for r in ledger:
        if r.topic_relevance in {"adjacent_meta_analysis","adjacent_review","adjacent_guideline","adjacent_case_report"}:
            adjunct_rows.append({
                "pmid": r.pmid,
                "topic_relevance": r.topic_relevance,
                "final_decision": r.final_decision,
                "pub_types": ";".join(r.pub_types),
                "year": r.year or "",
                "title": r.title
            })
    if adjunct_rows:
        write_csv(outp / "adjunct.csv", adjunct_rows, ["pmid","topic_relevance","final_decision","pub_types","year","title"])

    pc = prisma_counts(ledger)
    write_json(outp / "prisma.json", pc)

    gate_counts = defaultdict(int)
    for r in ledger:
        if r.gate_reason:
            gate_counts[r.gate_reason] += 1
    log.info(f"Gates (auto_exclude reasons): {dict(gate_counts)}")
    lane_summary = defaultdict(int)
    for r in ledger: lane_summary[r.lane_before_llm] += 1
    log.info(f"Lanes summary: {dict(lane_summary)}")
    dec_summary = defaultdict(int)
    for r in ledger: dec_summary[r.final_decision] += 1
    log.info(f"Decisions: {dict(dec_summary)}")
    log.info(f"Done in {time.time()-t0:.1f}s. Records: {len(docs)} | Ledger: {len(ledger)} | Out: {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str, help="Intent/preferences paragraph")
    ap.add_argument("--languages", type=str, default=",".join(DEFAULT_LANGS))
    ap.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--max-records", type=int, default=300)
    args = ap.parse_args()

    langs = [x.strip() for x in (args.languages or "").split(",") if x.strip()]
    run(
        prompt=args.prompt,
        languages=langs,
        year_min=args.year_min,
        out_dir=args.out,
        max_records=args.max_records,
    )

if __name__ == "__main__":
    main()
