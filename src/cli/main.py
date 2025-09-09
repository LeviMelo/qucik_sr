from __future__ import annotations
import sys, os, json, typer, time, logging
from typing import List, Dict
from collections import Counter

from src.config.defaults import (
    ESEARCH_RETMAX_PER_QUERY, SEED_SEM_TAU_HI, SEED_MIN_COUNT, SEED_RELAX_STEP,
    CILE_REL_GATE_FRAC, CILE_EXT_BUDGET, CILE_MAX_ACCEPT, CILE_HUB_QUARANTINE, CILE_MIN_HUB_SOFT,
    REG_P_HI, REG_P_LO, LLM_BUDGET, DEFAULT_LANGS, DEFAULT_YEAR_MIN
)
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
import re
from itertools import combinations, product
from src.config.defaults import (
    ESEARCH_KEEP_MIN, ESEARCH_KEEP_MAX, QUERY_VARIANT_CAP,
)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("cli")

# --- Helpers for robust retrieval ---
def _empty_or_blank_queries(qdict: Dict[str, str] | None) -> bool:
    if not qdict:
        return True
    for v in qdict.values():
        if isinstance(v, str) and v.strip():
            return False
    return True

# ---------- NEW: generic query generation from PICOS (no MeSH, NL-friendly) ----------


_STOP = {
    "the","a","an","of","and","or","to","for","in","on","with","without","by","vs","versus",
    "during","after","before","at","from","as","within","between","under","over",
    "study","trial","randomized","randomised","cohort","case","control","observational",
    "adult","adults","adolescent","adolescents","children","child","pediatric","paediatric"
}

def _split_phrases(text: str) -> list[str]:
    if not text: return []
    s = text.replace("/", " ").replace(";", " ").replace(",", " ").replace(":", " ").replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def _clean_tokens(words: list[str]) -> list[str]:
    out = []
    for w in words:
        w = re.sub(r"[^A-Za-z0-9\-]+", "", w)
        if not w: continue
        wl = w.lower()
        if wl in _STOP: continue
        if len(wl) <= 2: continue
        out.append(wl)
    return out

def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _unique_keep_order(seq: list[str]) -> list[str]:
    seen = set(); out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _extract_key_terms(picos) -> tuple[list[str], list[str]]:
    """
    From PICOS, pull P and I terms. We do not force MeSH or fields—keep it NL-friendly.
    Strategy:
      - tokenize population & intervention
      - keep unigrams/bigrams/trigrams (longer phrases first)
      - drop common stopwords
    """
    p_text = (picos.population or "") or ""
    i_text = (picos.intervention or "") or ""
    p_tok  = _clean_tokens(_split_phrases(p_text))
    i_tok  = _clean_tokens(_split_phrases(i_text))

    p_grams = _unique_keep_order(_ngrams(p_tok,3) + _ngrams(p_tok,2) + p_tok)
    i_grams = _unique_keep_order(_ngrams(i_tok,3) + _ngrams(i_tok,2) + i_tok)

    # favor longer phrases: sort by token count desc then alphabetically
    p_grams.sort(key=lambda s: (-len(s.split()), s))
    i_grams.sort(key=lambda s: (-len(s.split()), s))

    # cap each side to avoid explosion; we still build many query variants below
    return p_grams[:10], i_grams[:12]

def _quote(t: str) -> str:
    return f"\"{t}\"" if " " in t else t

def _build_query_variants(p_terms: list[str], i_terms: list[str], cap: int) -> dict[str,str]:
    """
    Produce a *portfolio* of queries:
      1) Natural-language “bag” (P + I).
      2) Phrase-pairs (quoted) in AND.
      3) OR-packs on each side (P) AND (I) with small rotations.
    No field tags; let PubMed ATM expand freely. Dedup outputs.
    """
    variants: dict[str,str] = {}
    def add(name: str, q: str):
        if len(variants) < cap and q.strip():
            variants[name] = q

    # 1) NL bags (prefer top phrases)
    bag1 = " ".join((_quote(x) for x in (p_terms[:2] + i_terms[:2])))
    if bag1: add("nl_bag_1", bag1)
    bag2 = " ".join((_quote(x) for x in (p_terms[:3] + i_terms[:3])))
    if bag2: add("nl_bag_2", bag2)

    # 2) Pairwise P-I AND combinations (limit)
    for p, i in list(product(p_terms[:5], i_terms[:6]))[:20]:
        add(f"pi_pair_{p}_{i}", f"{_quote(p)} AND {_quote(i)}")

    # 3) OR-packs (small rotations)
    def pack(xs: list[str], k: int) -> str:
        xsq = [_quote(x) for x in xs[:k]]
        return "(" + " OR ".join(xsq) + ")"
    # a few sizes to diversify
    for kP, kI in [(3,3),(4,4),(5,3)]:
        add(f"pack_{kP}_{kI}", f"{pack(p_terms, kP)} AND {pack(i_terms, kI)}")

    # 4) Minimal backoffs if still empty later (we’ll call generator again)
    return variants


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
        max_records: int = typer.Option(None, "--max-records", help="Dev cap: limit number of records processed"),
        ):
    t_start = time.monotonic()
    run_dir = ensure_run_dir(out_dir)

    try:
        # ---- 1) P0 criteria ----
        log.info("Phase P0: generating criteria via LLM…")
        criteria_js = chat_json(P0_SYSTEM, p0_user_prompt(prompt))
        # sanitize missing fields
        if "picos" not in criteria_js:
            criteria_js["picos"] = {"population":"","intervention":"","comparison":None,"outcomes":[],"study_design":[],
                                    "year_min":year_min,"languages":languages.split(",")}
        else:
            criteria_js["picos"].setdefault("year_min", year_min)
            # normalize languages to names, not codes
            langs = languages.split(",")
            criteria_js["picos"].setdefault("languages", langs)
            if isinstance(criteria_js["picos"]["languages"], list) and all(len(x)<=3 for x in criteria_js["picos"]["languages"]):
                criteria_js["picos"]["languages"] = langs
        criteria = Criteria(**criteria_js)
        write_json(run_dir/"criteria.json", criteria.model_dump())
        log.info(f"Phase P0 done in {time.monotonic()-t_start:.1f}s")

        # ---- 2) Retrieval ----
        log.info("Retrieval: esearch…")

        # Harvest P0 queries (may be empty or brittle)
        p0_q = dict(criteria.boolean_queries or {})
        if not p0_q:
            log.warning("P0 returned no boolean_queries.")

        # Always augment with generic P∧I variants
        p_terms, i_terms = _extract_key_terms(criteria.picos)
        auto_q = _build_query_variants(p_terms, i_terms, QUERY_VARIANT_CAP)

        # Merge and probe everything
        candidate_q = {**p0_q, **auto_q}

        log.info(f"Candidate queries to probe: {len(candidate_q)}")
        for name, q in candidate_q.items():
            snip = (q or "").strip().replace("\n", " ")
            if len(snip) > 160: snip = snip[:160] + " …"
            log.info(f"  {name}: {snip}")

        # Probe counts and keep a healthy band
        from src.net.entrez import esearch, efetch_abstracts, esearch_count  # <-- use new helper

        kept: dict[str,str] = {}
        q_stats = []
        total_expected = 0
        for name, q in candidate_q.items():
            try:
                cnt = esearch_count(q, mindate=criteria.picos.year_min)
            except Exception as e:
                log.warning(f"esearch_count failed for {name}: {e}")
                continue
            q_stats.append({"name": name, "hits": cnt})
            if cnt >= ESEARCH_KEEP_MIN and cnt <= ESEARCH_KEEP_MAX:
                kept[name] = q
                total_expected += cnt

        # If nothing in band, try minimal backoffs (P top-1 & I top-1; and plain bag)
        if not kept:
            log.warning("No query fell in the keep band; trying minimal backoffs.")
            if p_terms and i_terms:
                kept["pi_min"] = f"{_quote(p_terms[0])} AND {_quote(i_terms[0])}"
            bag_min = " ".join((_quote(x) for x in (p_terms[:1] + i_terms[:2])))
            if bag_min:
                kept["nl_min"] = bag_min

        # Execute kept queries → union+dedupe
        pmid_set = set()
        for name, q in kept.items():
            try:
                ids = esearch(q, retmax=ESEARCH_RETMAX_PER_QUERY, mindate=criteria.picos.year_min)
                pmid_set.update(ids)
            except Exception as e:
                log.warning(f"esearch failed for {name}: {e}")

        pmids = list(pmid_set)
        # Dev cap if requested
        if max_records and len(pmids) > max_records:
            pmids = pmids[:max_records]
            log.info(f"Dev cap: limiting to first {len(pmids)} records")

        log.info(f"Retrieval: efetch abstracts for {len(pmids)} PMIDs…")
        meta = efetch_abstracts(pmids, workers=3, use_cache=True)

        write_json(run_dir/"retrieval.json", {
            "queries_from_llm": p0_q,
            "queries_auto_generated": auto_q,
            "queries_kept": kept,
            "query_stats": q_stats,
            "pmids": pmids,
            "hits": len(meta)
        })
        log.info(f"Query stats: {q_stats}")

        docs = _to_docs(meta)
        if not docs:
            log.error("No records retrieved.")
            raise typer.Exit(code=1)


        # ---- 3) Embeddings & intent vector ----
        log.info(f"Embedding {len(docs)} documents…")
        emb, idx = build_embeddings(docs)
        intent_vec = embed_texts([prompt])[0]

        # ---- 4) Seeds S+ (relax tau, then one-time fallback if still too few) ----
        tau = SEED_SEM_TAU_HI
        seeds = [d.pmid for d in docs if is_primary_design(d) and (float(emb[idx[d.pmid]] @ intent_vec) >= tau)]
        while len(seeds) < SEED_MIN_COUNT and tau > 0.80:
            tau -= SEED_RELAX_STEP
            seeds = [d.pmid for d in docs if is_primary_design(d) and (float(emb[idx[d.pmid]] @ intent_vec) >= tau)]
        if len(seeds) < SEED_MIN_COUNT:
            primaries = [d for d in docs if is_primary_design(d)]
            primaries.sort(key=lambda d: float(emb[idx[d.pmid]] @ intent_vec), reverse=True)
            k = min(max(5, SEED_MIN_COUNT), len(primaries))
            seeds = [d.pmid for d in primaries[:k]]
            log.info(f"Seed fallback → using top-{len(seeds)} semantic primaries")
        log.info(f"Seeds established: {len(seeds)} (tau_final={tau:.2f})")

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
        new_pmids = [str(x) for x in list(H1 - H0)]
        if new_pmids:
            log.info(f"CILE added {len(new_pmids)} new PMIDs; fetching their abstracts…")
            extra = efetch_abstracts(new_pmids, workers=3, use_cache=True)
            meta.update(extra)
            docs = _to_docs(meta)
            emb, idx = build_embeddings(docs)  # extend embedding matrix

        # ---- 6) Signals ----
        log.info("Computing signals…")
        signals = compute_signals(docs, emb, idx, intent_vec, seeds, criteria.picos.year_min or year_min)

        # ---- 7–10) Screening pipeline ----
        log.info("Screening…")
        ledger: List[LedgerRow] = []
        pool_for_model: List[Document] = []
        gate_counts = Counter()

        # 7) Objective gates (+ auto_include path)
        for d in docs:
            g = objective_gate(d, criteria.picos)
            if g is not None:
                reason_code, reason = g
                gate_counts[reason] += 1
                ledger.append(LedgerRow(
                    pmid=d.pmid, lane_before_llm="auto_exclude", gate_reason=reason,
                    model_p=None, llm=None,
                    final_decision="exclude", final_reason=reason,
                    signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                    title=d.title, abstract=d.abstract
                ))
            else:
                sig = signals[d.pmid]
                if (is_primary_design(d) and sig.sem_intent >= 0.92 and (sig.graph_ppr_pct >= 90.0 or sig.graph_links_frac >= 0.20)):
                    ledger.append(LedgerRow(
                        pmid=d.pmid, lane_before_llm="auto_include", gate_reason=None,
                        model_p=1.0, llm=None,
                        final_decision="include", final_reason="insufficient_info",
                        signals=sig, pub_types=d.pub_types, year=d.year,
                        title=d.title, abstract=d.abstract
                    ))
                else:
                    pool_for_model.append(d)

        # Safety valve: if gates consumed everything, still send some to LLM
        if not pool_for_model:
            survivors = [d for d in docs if all(r.pmid != d.pmid for r in ledger)]
            if survivors:
                survivors.sort(key=lambda d: float(emb[idx[d.pmid]] @ intent_vec), reverse=True)
                to_llm = survivors[:min(llm_budget, len(survivors))]
                log.warning(f"No pool_for_model after gates; forcing LLM pass on {len(to_llm)} best semantic candidates.")
                llm_out = llm_decide_batch(criteria, to_llm, signals)
                for dec in llm_out:
                    d = next(dd for dd in to_llm if dd.pmid == dec.pmid)
                    ledger.append(LedgerRow(
                        pmid=d.pmid, lane_before_llm="sent_to_llm", gate_reason=None, model_p=None,
                        llm=dec, final_decision=dec.decision, final_reason=dec.primary_reason,
                        signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                        title=d.title, abstract=d.abstract
                    ))

        # 8) Regressor bootstrap (guard single-class)
        import numpy as np
        X_boot, y_boot = [], []
        seed_set = set(seeds)
        for d in docs:
            sig = signals[d.pmid]
            X_boot.append(featurize_row(sig, d))
            y_boot.append(1 if d.pmid in seed_set else 0)
        X_boot = np.stack(X_boot, axis=0); y_boot = np.array(y_boot, dtype="int64")
        reg = OnlineRegressor()

        unique = set(y_boot.tolist())
        if len(unique) < 2:
            log.warning("Bootstrap labels have a single class; warm-starting with synthetic pos/neg from semantic ranks.")
            scores = [(d, float(emb[idx[d.pmid]] @ intent_vec)) for d in docs]
            scores.sort(key=lambda t: t[1], reverse=True)
            k = max(20, len(scores) // 10)
            pos = [featurize_row(signals[d.pmid], d) for d,_ in scores[:k]]
            neg = [featurize_row(signals[d.pmid], d) for d,_ in scores[-k:]]
            X_boot = np.stack(pos + neg, axis=0)
            y_boot = np.array([1]*len(pos) + [0]*len(neg), dtype="int64")

        reg.fit_bootstrap(X_boot, y_boot)

        # 9) Model triage → hi/lo auto, mid → uncertain
        uncertain_docs: List[Document] = []
        for d in pool_for_model:
            p = float(reg.predict_proba(np.stack([featurize_row(signals[d.pmid], d)], axis=0))[0])
            if p >= REG_P_HI:
                ledger.append(LedgerRow(pmid=d.pmid, lane_before_llm="model_include", gate_reason=None, model_p=p, llm=None,
                                        final_decision="include", final_reason="insufficient_info",
                                        signals=signals[d.pmid], pub_types=d.pub_types, year=d.year, title=d.title, abstract=d.abstract))
            elif p <= REG_P_LO:
                ledger.append(LedgerRow(pmid=d.pmid, lane_before_llm="model_exclude", gate_reason="off_topic", model_p=p, llm=None,
                                        final_decision="exclude", final_reason="off_topic",
                                        signals=signals[d.pmid], pub_types=d.pub_types, year=d.year, title=d.title, abstract=d.abstract))
            else:
                uncertain_docs.append(d)

        # 10) Spend LLM budget: uncertain first, else top candidates
        if uncertain_docs:
            ps = []
            for d in uncertain_docs:
                p = float(reg.predict_proba(np.stack([featurize_row(signals[d.pmid], d)], axis=0))[0])
                ps.append((d, abs(p - 0.5), p))
            ps.sort(key=lambda t: t[1])
            to_llm = [d for d,_,_ in ps[:llm_budget]]
            llm_out = llm_decide_batch(criteria, to_llm, signals)
            X_upd, y_upd = [], []
            for dec in llm_out:
                d = next(dd for dd in to_llm if dd.pmid == dec.pmid)
                ledger.append(LedgerRow(
                    pmid=d.pmid, lane_before_llm="sent_to_llm", gate_reason=None, model_p=None,
                    llm=dec, final_decision=dec.decision,
                    final_reason=dec.primary_reason,
                    signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                    title=d.title, abstract=d.abstract
                ))
                if dec.decision == "include":
                    X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(1)
                elif dec.decision == "exclude":
                    X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(0)
            if X_upd:
                reg.partial_update(np.stack(X_upd, axis=0), np.array(y_upd, dtype="int64"))
        else:
            # No uncertain docs → still spend LLM budget on highest-priority candidates
            candidates = [d for d in pool_for_model if is_primary_design(d)] or pool_for_model
            candidates.sort(key=lambda d: (signals[d.pmid].sem_intent + 0.2*signals[d.pmid].graph_links_frac), reverse=True)
            to_llm = candidates[:min(llm_budget, len(candidates))]
            if to_llm:
                log.info(f"Forcing LLM pass on top-{len(to_llm)} candidates (no uncertain docs).")
                llm_out = llm_decide_batch(criteria, to_llm, signals)
                X_upd, y_upd = [], []
                for dec in llm_out:
                    d = next(dd for dd in to_llm if dd.pmid == dec.pmid)
                    ledger.append(LedgerRow(
                        pmid=d.pmid, lane_before_llm="sent_to_llm", gate_reason=None, model_p=None,
                        llm=dec, final_decision=dec.decision,
                        final_reason=dec.primary_reason,
                        signals=signals[d.pmid], pub_types=d.pub_types, year=d.year,
                        title=d.title, abstract=d.abstract
                    ))
                    if dec.decision == "include":
                        X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(1)
                    elif dec.decision == "exclude":
                        X_upd.append(featurize_row(signals[d.pmid], d)); y_upd.append(0)
                if X_upd:
                    reg.partial_update(np.stack(X_upd, axis=0), np.array(y_upd, dtype="int64"))

        # ---- 11) Outputs ----
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

        # prisma + cile meta
        from src.prisma.counts import prisma_counts
        write_json(run_dir/"prisma.json", prisma_counts([r for r in ledger]))
        write_json(run_dir/"cile.json", meta_cile)

        # audit logs
        lanes = Counter(r.lane_before_llm for r in ledger)
        decisions = Counter(r.final_decision for r in ledger)
        log.info(f"Gates (auto_exclude reasons): {dict(gate_counts)}")
        log.info(f"Lanes summary: {dict(lanes)}")
        log.info(f"Decisions: {dict(decisions)}")
        log.info(f"Done in {time.monotonic()-t_start:.1f}s. Records: {len(docs)} | Ledger: {len(ledger)} | Out: {run_dir}")

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl-C). Partial results saved where available.")
        raise

if __name__ == "__main__":
    # compatibility shim: allow "… run <args>" when we only have one command
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "run":
        _sys.argv.pop(1)
    app()
