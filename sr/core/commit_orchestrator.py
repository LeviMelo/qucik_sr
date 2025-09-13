# sr/core/commit_orchestrator.py
from __future__ import annotations
from typing import List, Dict, Tuple
from sr.config.schema import Protocol, Record, Signals, LedgerRow, FinalDecision
from sr.retrieval.pubmed import esearch_paged, efetch_abstracts, to_records
from sr.retrieval.dedupe import dedupe
from sr.retrieval.diary import Diary
from sr.screen.gates import apply_gates
from sr.ranking.signals import build_tfidf_corpus, compute_signals
from sr.core.scheduler import build_ranks, fuse_to_rrf, pick_frontier, ees_within_frontier
from sr.ranking.ees import EESModel
from sr.screen.passes import pass_a, pass_b, pass_c
from sr.screen.aggregate import aggregate, dissonant
from sr.io.export import write_ledger_and_prisma
from sr.config.defaults import FRONTIER_SIZE, PASS_A_BATCH

def commit(proto: Protocol, question_text: str, out_dir: str) -> Tuple[List[LedgerRow], Diary]:
    diary = Diary()
    # Retrieval
    ids_all: List[str] = []
    for name, q in (proto.retrieval_plan or {}).items():
        diary.log_query(name, q)
        ids = esearch_paged(q, mindate=proto.picos.year_min)
        ids_all.extend(ids)
    ids_all = list(dict.fromkeys(ids_all))
    diary.set_total(len(ids_all))

    raw = efetch_abstracts(ids_all)
    recs = dedupe(to_records(raw))

    # Gates
    pool: List[Record] = []
    ledger: List[LedgerRow] = []
    for r in recs:
        gate = apply_gates(r, proto)
        if gate:
            # deterministic exclude; log minimal row if desired
            continue
        pool.append(r)

    if not pool:
        return ledger, diary

    # Signals + RRF
    vec, X = build_tfidf_corpus(pool)
    sigs: Dict[str, Signals] = compute_signals(pool, proto, vec, X, question_text)
    ranks = build_ranks(pool, sigs)
    rrf = fuse_to_rrf(ranks)
    frontier_ids = pick_frontier(rrf, FRONTIER_SIZE)

    # EES within frontier (epochal learning: cold start -> identity)
    ees = EESModel()
    batch_ids = ees_within_frontier(ees, sigs, frontier_ids, PASS_A_BATCH)

    # Passes + aggregation
    for pid in batch_ids:
        rec = next(r for r in pool if r.pmid == pid)
        a = pass_a(proto, rec)
        # dissonance triggers
        rrfs = rrf[pid].score
        b=None; c=None
        if dissonant(a, rrfs, None):
            b = pass_b(proto, rec, a, "include@low_RRF")
            if b.stance == "challenge":
                c = pass_c(proto, rec, a, b)
        final = aggregate(proto, rec, rrf[pid], a, b, c)
        ledger.append(LedgerRow(
            record=rec,
            signals=sigs[pid],
            rrf=rrf[pid],
            ees=None,
            pass_a=a,
            pass_b=b,
            pass_c=c,
            final=final
        ))

    # Export
    write_ledger_and_prisma(ledger, out_dir)
    return ledger, diary
