# sr/io/export.py
from __future__ import annotations
from typing import List, Dict, Any
import csv, json, pathlib
from collections import Counter
from sr.config.schema import LedgerRow

def prisma_counts(ledger: List[LedgerRow]) -> Dict[str, Any]:
    N = len(ledger)
    finals = [r.final.final if r.final else "unknown" for r in ledger]
    reasons = [r.final.reason for r in ledger if r.final]
    return {
        "records_screened": N,
        "included": finals.count("include_for_full_text"),
        "excluded": finals.count("exclude"),
        "borderline": finals.count("borderline"),
        "exclusions_by_reason": dict(Counter(reasons)),
    }

def write_ledger_and_prisma(ledger: List[LedgerRow], out_dir: str):
    root = pathlib.Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # ledger.tsv
    with open(root/"ledger.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pmid","final","reason","rrf","tfidf_cos","embed_cos","pi_hits_t","pi_hits_a","design_prior","recency","title"])
        for row in ledger:
            r = row.record; s = row.signals; fin = row.final
            w.writerow([r.pmid, fin.final if fin else "", fin.reason if fin else "", f"{row.rrf.score:.6f}",
                        f"{s.tfidf_cos:.4f}", f"{s.embed_cos:.4f}", s.pi_hits_title, s.pi_hits_abstract, f"{s.design_prior:.1f}", f"{s.recency_scaled:.3f}",
                        (r.title or "").replace("\t"," ").replace("\n"," ")[:160]])
    # prisma
    with open(root/"prisma_triage.json", "w", encoding="utf-8") as f:
        json.dump(prisma_counts(ledger), f, ensure_ascii=False, indent=2)
