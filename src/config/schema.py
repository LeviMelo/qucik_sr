# src/config/schema.py
from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

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
    # <-- be permissive; the LLM may return lists/ints/bools etc.
    inclusion_criteria: Dict[str, Any] = Field(default_factory=dict)
    exclusion_criteria: Dict[str, Any] = Field(default_factory=dict)
    # be flexible here too
    reason_taxonomy: List[str] = Field(default_factory=list)
    boolean_queries: Dict[str, str] = Field(default_factory=dict)

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
    evidence: Dict[str, str]

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
