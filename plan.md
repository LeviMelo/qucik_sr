# Directed SR Screener — Codebase Plan

**Goal:** Local, deterministic tool that takes a **prompt + preferences**, derives **PICOS criteria** with an LLM, retrieves PubMed records, performs **bounded iCite-graph expansion (CILE)**, and produces **PRISMA-compliant title/abstract screening** with `{include | exclude | borderline}` decisions plus a complete audit trail.

**Non-Goals:** web service, full-text screening, citation managers, continuous background jobs.
**Constraints:** Only **iCite** is used for references/citations; Entrez is used for PubMed retrieval. Everything runs **locally**.

---

## 1) Repository Layout

```
sr-screener/
├─ plan.md                        # this document
├─ README.md
├─ pyproject.toml                 # or requirements.txt
├─ .env.example                   # environment knobs (keys, timeouts)
├─ src/
│  ├─ cli/
│  │  └─ main.py                  # single entrypoint CLI (Fire/Typer)
│  ├─ config/
│  │  ├─ defaults.py              # default constants & thresholds
│  │  └─ schema.py                # pydantic models for config & outputs
│  ├─ io/
│  │  ├─ store.py                 # run directory creation, artifact writers
│  │  ├─ cache_fs.py              # file caches (embeddings, LLM jsonl logs)
│  │  └─ cache_sqlite.py          # sqlite caches (icite, efetch)
│  ├─ net/
│  │  ├─ entrez.py                # esearch/efetch (requests)
│  │  └─ icite.py                 # pubs endpoint (iCite only)
│  ├─ text/
│  │  ├─ embed.py                 # LM Studio embeddings (title+abstract)
│  │  ├─ prompts.py               # P0/P1/P2 prompt templates
│  │  └─ llm.py                   # LM Studio chat wrapper (JSON guardrails)
│  ├─ graph/
│  │  ├─ cile.py                  # bounded expansion (1 wave typical)
│  │  └─ qa_cluster.py            # Leiden/HDBSCAN for QA/triage only
│  ├─ screen/
│  │  ├─ criteria.py              # P0: derive PICOS & reason enum (LLM)
│  │  ├─ features.py              # semantic/graph/design features
│  │  ├─ gates.py                 # objective auto-exclude/include rules
│  │  ├─ regressor.py             # online logistic + calibration
│  │  ├─ llm_decide.py            # P1/P2/P3 decisions & borderline loop
│  │  └─ pipeline.py              # orchestration A→D with budgets
│  ├─ prisma/
│  │  ├─ counts.py                # flow tallies + exclusion breakdown
│  │  └─ diagram.py               # Mermaid .mmd or DOT export
│  └─ util/
│     ├─ rng.py                   # seeded RNG, hash helpers
│     ├─ log.py                   # structured logging
│     └─ timeit.py                # timers & budget accounting
├─ runs/                          # output runs (ignored by VCS)
└─ tests/
   ├─ mini_corpus/                # 20–40 labeled abstracts for smoke tests
   └─ test_end_to_end.py
```

---

## 2) End-to-End Flow (Stages)

### A) Retrieval & Seeding (cheap)

1. **P0 Criteria (LLM):** From prompt + preferences → strict JSON PICOS, inclusion/exclusion enum, reason taxonomy, and PubMed boolean queries.
2. **Entrez `esearch`:** Run each boolean variant; cap per-variant hits; union PMIDs; **dedupe**.
3. **Entrez `efetch`:** Title, abstract, pub\_types, year, journal, language, doi → `Document`.
4. **Embeddings:** Title+abstract → vector; cache.
5. **Seeds `S⁺` (sure-positives):** `pub_types` in primary set AND semantic to intent ≥ τ\_hi (e.g., 0.92). If <10 seeds, relax to 0.90.

### B) Graph Expansion (CILE; iCite only)

1. Build H from **iCite** refs∪citers around `S⁺`.
2. Run one **CILE** wave (compute-bounded): relevance gate (`links_to_S⁺/deg ≥ 0.08`), external-degree budget, hub quarantine (external), elastic-φ, semantic cohesion guard.
3. Accept ΔA → add to candidates; `efetch`+embed for Δ only.

### C) Screening Cascade (minimize LLM)

**Lane 1: Auto-exclude (no LLM)**

* Year < year\_min → **year**
* Language not allowed → **language**
* `pub_types` contains {Review, Meta-Analysis, Editorial, Letter, Conference Abstract} → **design\_mismatch**
* Missing title **and** abstract → **insufficient\_info**

**Lane 2: Auto-include (no LLM)**

* `pub_types` in primary set **AND** semantic ≥ τ\_hi **AND** graph locality high (e.g., PPR ≥ 90th pct or links\_to\_S⁺/deg ≥ 0.20).

**Lane 3: Regressor (teacher-student)**

* Features: semantic (to intent & seed centroid), graph (PPR, links\_frac), design one-hots, recency, abstract length bin, language one-hots.
* Online logistic regression + calibration.
* Initialize with `S⁺` as positives and auto-excludes (non-primary) + low-semantic tails as negatives.
* Predict `p`. Thresholds:

  * `p ≥ p_hi` → **model\_include**
  * `p ≤ p_lo` → **model\_exclude**
  * else **uncertain**.

**Lane 4: LLM (budgeted)**

* Send **most uncertain** up to budget **B** to **P1** (document + SignalCard).
* Update regressor with returned labels.
* Remaining uncertain → **borderline** (first pass).
* **Borderline escalation (P2)** with nearest included neighbors (k=3) + iCite included neighbors; ask model to commit if no clear violation.
* **Committee (P3)** for still-borderline: 2–3 prompt variants at T=0.2; majority vote; unresolved remain **borderline**.

> **Outcome rule**: never exclude solely for missing outcomes in abstract. If P/I match but outcomes unclear → **borderline**, not exclude.

### D) PRISMA Packaging

* Tally counts (identified, deduped, screened, excluded by primary reason, borderline, to-full-text).
* Write artifacts (see §7).

---

## 3) Data Models (Pydantic)

```python
# src/config/schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

Reason = Literal[
  "design_mismatch","population_mismatch","intervention_mismatch",
  "language","year","insufficient_info","off_topic"
]

class PICOS(BaseModel):
    population: str
    intervention: str
    comparison: Optional[str] = None
    outcomes: List[str]
    study_design: List[str]
    year_min: Optional[int] = None
    languages: List[str] = ["English"]

class Criteria(BaseModel):
    picos: PICOS
    inclusion_criteria: Dict[str, str]     # free-text per facet
    exclusion_criteria: Dict[str, str]
    reason_taxonomy: List[Reason]
    boolean_queries: Dict[str, str]        # named PubMed strings

class Document(BaseModel):
    pmid: str
    title: Optional[str] = ""
    abstract: Optional[str] = ""
    year: Optional[int] = None
    journal: Optional[str] = None
    language: Optional[str] = None
    pub_types: List[str] = []
    doi: Optional[str] = None

class Signals(BaseModel):
    sem_intent: float
    sem_seed: float
    graph_ppr_pct: float
    graph_links_frac: float
    year_scaled: float
    abstract_len_bin: Literal["none","short","normal","long"]

class DecisionLLM(BaseModel):
    pmid: str
    decision: Literal["include","exclude","borderline"]
    primary_reason: Reason
    confidence: float
    evidence: Dict[str, str]               # population_quote, intervention_quote, design_evidence, notes

class LedgerRow(BaseModel):
    pmid: str
    lane_before_llm: Literal["auto_exclude","auto_include","sent_to_llm","model_exclude","model_include","uncertain"]
    gate_reason: Optional[Reason] = None
    model_p: Optional[float] = None
    llm: Optional[DecisionLLM] = None
    final_decision: Literal["include","exclude","borderline"]
    final_reason: Reason
    signals: Signals
    pub_types: List[str]
    year: Optional[int]
    title: str
    abstract: str
```

---

## 4) Prompt Templates (tight JSON contracts)

### P0 — Criteria Derivation

**System**

* You are configuring a PRISMA title/abstract screening. From the user's description and preferences, produce strict, codeable criteria and PubMed boolean queries.
* **Return JSON ONLY** validating the `Criteria` schema: `picos` (object with keys), `inclusion_criteria` (object), `exclusion_criteria` (object), `reason_taxonomy` (array of enums), `boolean_queries` (object of strings).
* Do not add prose. Use double quotes. If unsure, leave a field empty; do not invent.

**User**

* Provide intent paragraph + any hard preferences (languages, year\_min).
* (We log your example outputs for model benchmarking.)

### P1 — Record Decision (first pass)

**System**

* You are a PRISMA title/abstract screener. Decide **INCLUDE/EXCLUDE/BORDERLINE** strictly from `CRITERIA_JSON` and the record.
* Use **pub\_types** only for design. Never infer design from title words.
* Do **NOT** exclude solely because outcomes are not stated; mark **BORDERLINE** instead.
* Treat Signals as hints: they are **not** sufficient reasons.
* **Return JSON ONLY** validating `DecisionLLM`.

**User**

* `CRITERIA_JSON: {...}`
* `RECORD: {pmid, title, abstract, pub_types, year, language}`
* `SIGNALS (SignalCard):`

  * Semantic match: {Very high|High|Medium|Low} ({sem\_intent:.2f})
  * Graph locality: {High|Medium|Low} — links\_to\_seeds {X%}, PPR {pct}
  * Design (pub\_types): \[...]
  * Year: #### (within scope | below year\_min | unknown)

### P2 — Borderline Escalation

**System**

* Second-pass adjudication for **BORDERLINE**. You will receive: the borderline record, K nearest **INCLUDED** neighbors (title + 1-sentence “why included”), and iCite included neighbors. Use this extra context to revisit the decision. Prefer **INCLUDE** over **EXCLUDE** unless a clear criterion violation exists.
* **Return JSON ONLY** validating `DecisionLLM`.

**User**

* `CRITERIA_JSON: {...}`
* `BORDERLINE_RECORD: {...}`
* `NEAREST_INCLUDED: [{pmid,title,why_included},...]`
* `GRAPH_INCLUDED_NEIGHBORS: [pmid,...]`
* `PRIOR_DECISION: {...}`

---

## 5) Features & Regressor

* **Features:**

  * `sem_intent`, `sem_seed` (cosine)
  * `graph_ppr_pct`, `graph_links_frac`
  * `design_primary` (1/0), `design_nonprimary` (1/0), one-hots for common pub\_types
  * `year_scaled` ((year−year\_min)/range), `abstract_len_bin` (ordinal)
  * `language_en`, `language_pt`, `language_es` one-hots

* **Model:** scikit-learn `SGDClassifier(loss='log', penalty='l2')` with `StandardScaler` on continuous features; `CalibratedClassifierCV` (isotonic) refreshed periodically.

* **Bootstrapping labels:** positives = `S⁺`; negatives = non-primary auto-excludes + lowest semantic decile.

* **Thresholds:** `p_hi=0.85`, `p_lo=0.15` (configurable).

* **Active learning:** send top-B by uncertainty (|p−0.5| ascending) to LLM.

---

## 6) Graph & Clustering

* **iCite only** for citations/references. SQLite cache of `pubs` JSON by pmid.
* **CILE** wave: relevance gate, external budget, hub quarantine (external), elastic-φ + **semantic cohesion guard** (centroid drift ≤ ε).
* **Leiden** (or HDBSCAN fallback) on candidate pool for **QA only** (cluster summaries: % include/exclude/borderline, median year, top terms). No cluster drives inclusion.

---

## 7) Outputs (Run Bundle)

Under `runs/YYYYMMDD-HHMM-topic/`:

* `criteria.json` — P0 output.
* `identification.json` — per-query hits, run timestamps, Entrez params.
* `candidates.csv` — pmid, pub\_types, year, language, embedding ids, feature columns.
* `screening.tsv` — **ledger** (see `LedgerRow`).
* `exclusions_by_reason.csv` — counts per primary reason.
* `prisma.json` — flow counts and legend; plus `prisma.mmd` (Mermaid) for a diagram.
* `cile.json` — metrics for wave(s).
* `clusters.csv` — QA stats per cluster.
* `intent.txt` — the paragraph that was embedded.
* `logs.jsonl` — structured log lines (timestamps, phases, decisions).

Everything is **deterministic**: RNG seeds, model versions, thresholds are committed to `run_meta.json`.

---

## 8) CLI (Typer/Fire)

```
sr-screen run \
  --prompt "pectus excavatum cryoablation during Nuss..." \
  --languages en,pt,es \
  --year-min 2000 \
  --max-variant-hits 2000 \
  --cile-wave 1 \
  --llm-budget 150 \
  --out runs/2025-09-08-pectus
```

**Subcommands (optional):**

* `sr-screen inspect --run <path>` (open ledger summaries, clusters)
* `sr-screen replay --run <path> --llm-pass borderline` (re-adjudicate)
* `sr-screen bench --model gemma-3n-e2b-it` (mini test set)

---

## 9) Caching & Persistence

* **SQLite:**

  * `icite.sqlite3(pmid TEXT PRIMARY KEY, json TEXT NOT NULL)`
  * `efetch.sqlite3(pmid TEXT PRIMARY KEY, json TEXT NOT NULL)`
* **FS cache:**

  * `data/cache/emb/<model>/<pmid>.npy`
  * `runs/<id>/llm_calls.jsonl` (raw LLM I/O for audit)
* **Retry & backoff:** Entrez and iCite with polite backoff; respect API etiquette.
* **Time budgets:** per phase timers; early abort on runaway stages.

---

## 10) Determinism & Reproducibility

* Fix RNG seeds (NumPy, Python, model init).
* Freeze thresholds in `config/defaults.py`; record overrides in `run_meta.json`.
* Log model version, embedding model, LM Studio model, exact prompts.
* Every LLM output stored raw; JSON schema validated; on parse error → retry once → else mark record as **borderline** with reason `insufficient_info`.

---

## 11) Testing Strategy

* **Unit tests:** criteria JSON validation, signal banding, gates, regressor fit/predict, CILE acceptance logic.
* **Mini E2E:** small topic with 20–40 labeled abstracts (manual gold), asserting ≥ target recall for `include+borderline`.
* **Non-regression:** snapshot of PRISMA counts on the mini corpus.
* **Speed checks:** per-stage timings; assert total completes under N minutes on local hardware.

---

## 12) Default Thresholds (tunable, then freeze)

* **Embedding:** intent vector from P0 paragraph; cosine on (title+abstract).
* **Seeds `S⁺`:** primary pub\_type ∧ sem\_intent ≥ **0.92**.
* **CILE relevance gate ρ:** **0.08**; ext-budget: **2500**; hub quarantine: **on (external)**.
* **Regressor:** `p_hi=0.85`, `p_lo=0.15`; calibration refresh every 200 LLM labels.
* **LLM Budget:** B ≈ 10% of candidate pool per batch (cap by CLI arg).
* **Borderline P2 neighbors:** k=3.
* **Committee size:** 3 (P3) at T=0.2.

---

## 13) Implementation Notes

* **LM Studio**: `src/text/llm.py` provides `chat_json(system, user, schema_name)` with robust JSON extraction (brace matching, `json.loads`, fallback repair).
* **SignalCard**: numeric → qualitative bands (Very high/High/Medium/Low) with the number in parentheses.
* **Quotes**: P1 requires short evidence quotes for P/I (if present); if absent, P2 may still include based on strong P/I paraphrase plus neighbors.

---

## 14) Minimal Dependencies

* `requests`, `numpy`, `pandas`, `scipy`, `networkx`, `scikit-learn`, `pydantic`, `orjson`, `typer` (or `fire`)
* Optional for clustering: `igraph`, `leidenalg` (fallback to HDBSCAN if missing)

---

## 15) Milestones

**MVP (week 1–2):**

* P0 criteria → retrieval → embeddings → S⁺ → CILE(1) → features → gates → simple regressor → P1 LLM (single pass) → PRISMA bundle.

**M1 (week 3):**

* Borderline P2 escalation, committee P3, calibration loop, clustering QA, Mermaid diagram.

**M2 (week 4):**

* Bench harness, speed knobs, richer logs, replay tools.

---

### Done Right, You Get

* High-recall candidate pool from Entrez + CILE (iCite only).
* **LLM where it matters** (uncertain boundary), with an online model shrinking that boundary over time.
* Deterministic, PRISMA-ready **screening.tsv** with reasons and quotes.
* Complete, local, auditable runs you can reproduce and defend.
