from __future__ import annotations
import sys, json, typer
from typing import List, Dict
from src.config.defaults import (ESEARCH_RETMAX_PER_QUERY, SEED_SEM_TAU_HI, SEED_MIN_COUNT, SEED_RELAX_STEP,
                              CILE_REL_GATE_FRAC, CILE_EXT_BUDGET, CILE_MAX_ACCEPT, CILE_HUB_QUARANTINE, CILE_MIN_HUB_SOFT,
                              REG_P_HI, REG_P_LO, LLM_BUDGET, DEFAULT_LANGS, DEFAULT_YEAR_MIN)
from src.config.schema import Criteria, Document, LedgerRow
from src.text.prompts import P0_SYSTEM, p0_user_prompt
from src.text.llm import chat_json
from src.net.entrez import esearch, efetch_abstracts
from src.text.embed import embed_texts
from src.screen.gates import objective_gate, is_primary_design
from src.screen.features import build_embeddings, compute_signals
from src.screen.regressor import OnlineRegressor, featurize_row
from src.screen.llm_decide import llm_decide_batch
from src.graph.cile import one_wave_expand
from src.io.store import ensure_run_dir, write_json, write_tsv

app = typer.Typer(add_completion=False)

def _to_docs(meta_map: Dict[str,dict]) -> List[Document]:
    docs = []
    for pmid, m in meta_map.items():
        docs.append(Document(
            pmid=str(pmid), title=m.get("title") or "", abstract=m.get("abstract") or "",
            year=m.get("year"), journal=m.get("journal"), language=m.get("language"),
            pub_types=m.get("pub_types") or [], doi=m.get("doi")
        ))
    return docs

@app.command()
def run(prompt: str = typer.Argument(..., help="Topic intent / preferences paragraph"),
        languages: str = typer.Option(",".join(DEFAULT_LANGS), "--languages"),
        year_min: int = typer.Option(DEFAULT_YEAR_MIN, "--year-min"),
        out_dir: str = typer.Option("runs/demo", "--out"),
        llm_budget: int = typer.Option(LLM_BUDGET, "--llm-budget"),
        ):
    # ---- 1) P0 criteria ----
    criteria_js = chat_json(P0_SYSTEM, p0_user_prompt(prompt))
    # sanitize missing fields
    if "picos" not in criteria_js:
        criteria_js["picos"] = {"population":"","intervention":"","comparison":None,"outcomes":[],"study_design":[],
                                "year_min":year_min,"languages":languages.split(",")}
    else:
        criteria_js["picos"].setdefault("year_min", year_min)
        criteria_js["picos"].setdefault("languages", languages.split(","))
    criteria = Criteria(**criteria_js)

    run_dir = ensure_run_dir(out_dir)
    write_json(run_dir/"criteria.json", criteria.model_dump())

    # ---- 2) Retrieval ----
    pmid_set = set()
    for name, q in criteria.boolean_queries.items():
        ids = esearch(q, retmax=ESEARCH_RETMAX_PER_QUERY, mindate=criteria.picos.year_min)
        pmid_set.update(ids)
    meta = efetch_abstracts(list(pmid_set))
    docs = _to_docs(meta)
    write_json(run_dir/"identification.json", {"queries": list(criteria.boolean_queries.items()), "hits": len(docs)})

    if not docs:
        print("No records retrieved.", file=sys.stderr)
        raise typer.Exit(code=1)

    # ---- 3) Embeddings & intent vector ----
    emb, idx = build_embeddings(docs)
    intent_vec = embed_texts([prompt])[0]

    # ---- 4) Seeds S+ ----
    tau = SEED_SEM_TAU_HI
    seeds = [d.pmid for d in docs if is_primary_design(d) and (float(emb[idx[d.pmid]] @ intent_vec) >= tau)]
    while len(seeds) < SEED_MIN_COUNT and tau > 0.80:
        tau -= SEED_RELAX_STEP
        seeds = [d.pmid for d in docs if is_primary_design(d) and (float(emb[idx[d.pmid]] @ intent_vec) >= tau)]

    # ---- 5) CILE expansion (1 wave) ----
    H0 = set(int(x) for x in seeds)
    H1, meta_cile = one_wave_expand(
        seeds_pos=[int(x) for x in seeds],
        H_existing=H0,
        rel_gate_frac=CILE_REL_GATE_FRAC,
        external_budget=CILE_EXT_BUDGET,
        max_accept=CILE_MAX_ACCEPT,
        hub_quarantine_external=CILE_HUB_QUARANTINE,
        min_hub_soft=CILE_MIN_HUB_SOFT
    )
    # add new pmids to pool (fetch if missing)
    new_pmids = [str(x) for x in list(H1 - H0)]
    if new_pmids:
        extra = efetch_abstracts(new_pmids)
        meta.update(extra)
        docs = _to_docs(meta)
        emb, idx = build_embeddings(docs)  # extend embedding matrix

    # ---- 6) Signals ----
    signals = compute_signals(docs, emb, idx, intent_vec, seeds, criteria.picos.year_min or year_min)

    # ---- 7) Objective gates ----
    ledger: List[LedgerRow] = []
    pool_for_model = []
    for d in docs:
        g = objective_gate(d, criteria.picos)
        if g is not None:
            reason_code, reason = g
            row = LedgerRow(
                pmid=d.pmid, lane_before_llm="auto_exclude", gate_reason=reason,
                model_p=None, llm=None,
                final_decision="exclude", final_reason=reason,
                signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                title=d.title, abstract=d.abstract
            )
            ledger.append(row)
        else:
            # possible auto-include (slam-dunk)
            sig = signals[d.pmid]
            if (is_primary_design(d) and sig.sem_intent >= 0.92 and (sig.graph_ppr_pct >= 90.0 or sig.graph_links_frac >= 0.20)):
                row = LedgerRow(
                    pmid=d.pmid, lane_before_llm="auto_include", gate_reason=None,
                    model_p=1.0, llm=None,
                    final_decision="include", final_reason="insufficient_info",  # reason not used for includes
                    signals=sig, pub_types=d.pub_types, year=d.year,
                    title=d.title, abstract=d.abstract
                )
                ledger.append(row)
            else:
                pool_for_model.append(d)

    # ---- 8) Regressor bootstrap ----
    # positives: seeds; negatives: auto_excludes + low semantic tail
    X_boot, y_boot = [], []
    seed_set = set(seeds)
    for d in docs:
        sig = signals[d.pmid]
        X_boot.append(featurize_row(sig, d))
        y_boot.append(1 if d.pmid in seed_set else 0)
    import numpy as np
    X_boot = np.stack(X_boot, axis=0); y_boot = np.array(y_boot, dtype="int64")
    reg = OnlineRegressor()
    reg.fit_bootstrap(X_boot, y_boot)

    # ---- 9) Model triage ----
    uncertain_docs = []
    for d in pool_for_model:
        p = float(reg.predict_proba(np.stack([featurize_row(signals[d.pmid], d)], axis=0))[0])
        if p >= REG_P_HI:
            ledger.append(LedgerRow(pmid=d.pmid, lane_before_llm="model_include", gate_reason=None, model_p=p, llm=None,
                                    final_decision="include", final_reason="insufficient_info",  # not used
                                    signals=signals[d.pmid], pub_types=d.pub_types, year=d.year, title=d.title, abstract=d.abstract))
        elif p <= REG_P_LO:
            ledger.append(LedgerRow(pmid=d.pmid, lane_before_llm="model_exclude", gate_reason="off_topic", model_p=p, llm=None,
                                    final_decision="exclude", final_reason="off_topic",
                                    signals=signals[d.pmid], pub_types=d.pub_types, year=d.year, title=d.title, abstract=d.abstract))
        else:
            uncertain_docs.append(d)

    # ---- 10) LLM budget on most uncertain ----
    # rank by closeness to 0.5 using current regressor
    if uncertain_docs:
        import numpy as np
        ps = []
        for d in uncertain_docs:
            p = float(reg.predict_proba(np.stack([featurize_row(signals[d.pmid], d)], axis=0))[0])
            ps.append((d, abs(p - 0.5), p))
        ps.sort(key=lambda t: t[1])  # smallest distance first
        to_llm = [d for d,_,_ in ps[:llm_budget]]
        llm_out = llm_decide_batch(criteria, to_llm, signals)
        # update model with LLM labels (include=1, exclude=0; borderline -> skip now)
        X_upd, y_upd = [], []
        for dec in llm_out:
            d = next(dd for dd in to_llm if dd.pmid == dec.pmid)
            row = LedgerRow(
                pmid=d.pmid, lane_before_llm="sent_to_llm", gate_reason=None, model_p=None,
                llm=dec, final_decision=dec.decision,
                final_reason=dec.primary_reason,
                signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                title=d.title, abstract=d.abstract
            )
            ledger.append(row)
            if dec.decision == "include":
                X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(1)
            elif dec.decision == "exclude":
                X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(0)
        if X_upd:
            reg.partial_update(np.stack(X_upd, axis=0), np.array(y_upd, dtype="int64"))

        # any remaining uncertain not sent to LLM -> borderline (for next layer P2)
        skipped = set(dd.pmid for dd,_,_ in ps[llm_budget:])
        for d in uncertain_docs:
            if d.pmid in skipped:
                ledger.append(LedgerRow(
                    pmid=d.pmid, lane_before_llm="uncertain", gate_reason=None, model_p=None, llm=None,
                    final_decision="borderline", final_reason="insufficient_info",
                    signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                    title=d.title, abstract=d.abstract
                ))

    # ---- 11) Write outputs ----
    # screening.tsv (ledger)
    fieldnames = ["pmid","lane_before_llm","gate_reason","model_p","final_decision","final_reason","year","pub_types","title","abstract"]
    rows = []
    for r in ledger:
        rows.append({
            "pmid": r.pmid,
            "lane_before_llm": r.lane_before_llm,
            "gate_reason": r.gate_reason or "",
            "model_p": "" if r.model_p is None else f"{r.model_p:.3f}",
            "final_decision": r.final_decision,
            "final_reason": r.final_reason,
            "year": r.year or "",
            "pub_types": ";".join(r.pub_types),
            "title": r.title.replace("\t"," ").replace("\n"," ").strip(),
            "abstract": (r.abstract or "").replace("\t"," ").replace("\n"," ").strip(),
        })
    write_tsv(run_dir/"screening.tsv", rows, fieldnames)

    # prisma counts
    from src.prisma.counts import prisma_counts
    write_json(run_dir/"prisma.json", prisma_counts([r for r in ledger]))

    # cile meta
    write_json(run_dir/"cile.json", meta_cile)

    print(f"Done. Records: {len(docs)} | Ledger: {len(ledger)} | Out: {run_dir}")

if __name__ == "__main__":
    app()
