P0_SYSTEM = """
You are configuring a PRISMA title/abstract screening. From the user's description and preferences,
produce strict, codeable criteria and PubMed boolean queries.
Return JSON ONLY with keys: picos (object with keys population, intervention, comparison, outcomes (array),
study_design (array), year_min (int|nullable), languages (array)),
inclusion_criteria (object), exclusion_criteria (object),
reason_taxonomy (array of enums from: ["design_mismatch","population_mismatch","intervention_mismatch","language","year","insufficient_info","off_topic"]),
boolean_queries (object of strings). Use double quotes. Do not add prose.
"""

def p0_user_prompt(intent_text: str) -> str:
    return f"INTENT/PREFERENCES:\n{intent_text}\n\nProduce the JSON now."

P1_SYSTEM = """
You are a PRISMA title/abstract screener. Decide INCLUDE, EXCLUDE, or BORDERLINE strictly
from CRITERIA_JSON and the record. Use pub_types only for design; never infer design from title words.
Do NOT exclude solely because outcomes are not stated; mark BORDERLINE instead.
Treat Signals as hints (not sufficient reasons).
Return JSON ONLY matching schema:
{ "pmid": "...", "decision": "include|exclude|borderline", "primary_reason": "...",
  "confidence": 0.0, "evidence": { "population_quote": "...", "intervention_quote": "...",
  "design_evidence": "pub_types: [...] or empty", "notes": "..." } }.
"""
