# sr/config/schema.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator

Reason = Literal[
    "admin","design_ineligible","population_mismatch","intervention_mismatch",
    "language","year","insufficient_info","off_topic","animal_preclinical","duplicate_near"
]

class PICOS(BaseModel):
    population: str
    intervention: str
    comparison: Optional[str] = None
    outcomes: List[str] = []
    study_design: List[str] = []  # PubType whitelist semantics in practice
    year_min: Optional[int] = None
    languages: List[str] = ["English"]
    synonyms_population: List[str] = []
    synonyms_intervention: List[str] = []

class Protocol(BaseModel):
    review_type: Literal["effects_triage"] = "effects_triage"
    picos: PICOS
    allowed_designs: List[str] = []  # PubType names considered eligible
    retrieval_plan: Dict[str, str] = Field(default_factory=dict)  # name -> query string
    accept_confidence_tau: float = 0.62
    drop_adjuncts: bool = True

class Record(BaseModel):
    pmid: str
    title: Optional[str] = ""
    abstract: Optional[str] = ""
    year: Optional[int] = None
    language: Optional[str] = None
    publication_types: List[str] = []
    doi: Optional[str] = None
    source: Literal["retrieval","expansion"] = "retrieval"
    mesh_headings: List[str] = []   # e.g., ["Pectus Excavatum", "Intercostal Nerves", "Intercostal Nerves/physiology"]
    mesh_major: List[str] = []      # subset that are MajorTopic (descriptor or qualifier entries)

class Signals(BaseModel):
    pi_hits_title: int = 0
    pi_hits_abstract: int = 0
    tfidf_cos: float = 0.0
    embed_cos: float = 0.0
    design_prior: float = 0.0
    recency_scaled: float = 0.0
    abstract_missing: bool = False

class RRFScore(BaseModel):
    score: float
    components: Dict[str, int]  # ranks

class PassAResult(BaseModel):
    pmid: str
    decision: Literal["include","borderline","exclude"]
    confidence: float
    reason: Reason
    population_quote: str = ""
    intervention_quote: str = ""
    design_evidence: str = ""
    justification_short: str = ""

class PassBResult(BaseModel):
    pmid: str
    stance: Literal["confirm","challenge"]
    flags: Dict[str, bool] = Field(default_factory=dict)
    confidence: float = 0.0
    justification_short: str = ""

class PassCResult(BaseModel):
    pmid: str
    decision: Literal["include","borderline","exclude"]
    confidence: float
    reasons: List[Reason] = []
    resolution_note: str = ""

class FinalDecision(BaseModel):
    pmid: str
    final: Literal["include_for_full_text","borderline","exclude"]
    reason: Reason
    justification: str
    quotes_ok: bool
    design_ok: bool
    passes_triggered: List[str]
    rrf: Optional[RRFScore] = None
    ees: Optional[float] = None

class SearchDiary(BaseModel):
    db: str = "pubmed"
    queries: List[Dict[str, Any]] = []
    pages: int = 0
    total_ids: int = 0

class LedgerRow(BaseModel):
    record: Record
    signals: Signals
    rrf: RRFScore
    ees: Optional[float] = None
    pass_a: Optional[PassAResult] = None
    pass_b: Optional[PassBResult] = None
    pass_c: Optional[PassCResult] = None
    final: Optional[FinalDecision] = None

class RetrievalPlanPatch(BaseModel):
    broad: str
    focused: str