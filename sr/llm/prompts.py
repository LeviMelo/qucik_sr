# sr/llm/prompts.py — full replacement

from __future__ import annotations
import json

# =========================
# Protocol inference prompts
# =========================

PROTOCOL_SYSTEM = """
You will return strict JSON for a PRISMA title/abstract triage protocol.

REQUIREMENTS
- Produce ONE JSON object only, wrapped between literal markers:
  BEGIN_JSON
  { ...json... }
  END_JSON
- Do NOT add any extra prose outside the JSON block.
- Arrays must be arrays even for singletons. No null strings. Use null only where allowed.

FIELDS (schema)
- review_type: must be "effects_triage"
- picos: object with:
  - population: string (concise, concrete)
  - intervention: string (concise, concrete)
  - comparison: string or null
  - outcomes: array of strings (0+)
  - study_design: array of strings (0+) (e.g., ["Randomized Controlled Trial","Cohort"])
  - year_min: integer year or null
  - languages: array of strings (0+)
  - synonyms_population: array of strings (0+)
  - synonyms_intervention: array of strings (0+)
- allowed_designs: array of PubMed Publication Type names (may be empty)
- retrieval_plan: OBJECT **REQUIRED** with AT LEAST:
    "broad": PubMed Boolean string ready to run,
    "focused": same as broad PLUS an RCT hedge.
  • Build each using BOTH Title/Abstract tags **[tiab]** and MeSH tags **[MeSH Terms]**.
  • Expand Population and Intervention with generous synonyms via OR blocks.
  • **No placeholders or macros** (e.g., do NOT emit P_TA_OR_BLOCK, I_TA_OR_BLOCK, BROAD_CORE).
  • Do **not** include language, date, or age filters inside the query (year/language are separate fields).

FORMAT & EXAMPLES
- Use parentheses and field tags. Examples of style (different topic):
  Population_TA = ("acute myocardial infarction"[tiab] OR "heart attack"[tiab] OR STEMI[tiab])
  Population_MESH = ("Myocardial Infarction"[MeSH Terms])
  Intervention_TA = ("beta blocker"[tiab] OR "beta-blocker"[tiab] OR "beta adrenergic antagonist"[tiab])
  Intervention_MESH = ("Adrenergic beta-Antagonists"[MeSH Terms])
  broad = ( (Population_TA OR Population_MESH) AND (Intervention_TA OR Intervention_MESH) )
  focused = ( broad ) AND ("randomized controlled trial"[Publication Type] OR randomized[tiab] OR randomised[tiab] OR "random allocation"[MeSH Terms])

IF YOU CANNOT CONSTRUCT a meaningful retrieval_plan from the user's intent:
- Set "needs_reprompt": true
- Set "reprompt_reason": a short, concrete sentence explaining exactly what is missing.
- Still return a valid JSON object with all required keys; retrieval_plan may be {} in that case.

Return only one JSON object between BEGIN_JSON and END_JSON.
"""

def _protocol_template_json() -> str:
    tmpl = {
        "review_type": "effects_triage",
        "picos": {
            "population": "",
            "intervention": "",
            "comparison": None,
            "outcomes": [],
            "study_design": [],
            "year_min": None,
            "languages": [],
            "synonyms_population": [],
            "synonyms_intervention": []
        },
        "allowed_designs": [],
        "retrieval_plan": {
            "broad": "",
            "focused": ""
        },
        "accept_confidence_tau": 0.62,
        "drop_adjuncts": True,
        "needs_reprompt": False,
        "reprompt_reason": ""
    }
    return json.dumps(tmpl, ensure_ascii=False, indent=2)

def protocol_user(nl: str) -> str:
    return f"""NATURAL_LANGUAGE_INTENT:
<<<
{nl}
>>>

Instructions:
- Fill the JSON template below with concrete values.
- Build 'broad' and 'focused' PubMed queries with explicit [tiab] and [MeSH Terms] blocks.
- Expand Population and Intervention with OR lists (tiab + MeSH). DO NOT use macros like P_TA_OR_BLOCK, BROAD_CORE, etc.
- Do NOT include language/date/age filters inside the queries.
- If you truly cannot build a retrieval_plan, set needs_reprompt=true and give reprompt_reason.
- Return ONLY one JSON object enclosed by BEGIN_JSON/END_JSON.

BEGIN_JSON
{_protocol_template_json()}
END_JSON
"""

# =========================
# JSON repair prompts
# =========================

REPAIR_SYSTEM = """
You repair malformed JSON to match a provided template structure.

Rules:
- Return exactly ONE JSON object, wrapped between BEGIN_JSON/END_JSON.
- No commentary. No backticks. No extra keys. No trailing commas.
- Preserve the schema and key names from the template.
- Coerce singleton strings to arrays where the template shows arrays.
- If a required field cannot be sensibly filled, set needs_reprompt=true and provide a clear reprompt_reason.
"""

def repair_user(template_json: str, bad_output: str) -> str:
    return f"""TEMPLATE_JSON:
{template_json}

BAD_OUTPUT:
{bad_output}

TASK:
- Produce valid JSON that matches TEMPLATE_JSON’s structure and keys.
- If retrieval_plan cannot be constructed, set needs_reprompt=true with a concrete reprompt_reason.

Return only:

BEGIN_JSON
{{...}}
END_JSON
"""

# =========================
# Screening passes prompts
# =========================

PASS_A_SYSTEM = """You are a PRISMA TITLE/ABSTRACT screener for effects triage.
Decide INCLUDE, BORDERLINE, or EXCLUDE strictly from the protocol and the record.

What you get in RECORD_JSON:
- title, abstract, year, language
- publication_types (PubMed PublicationTypeList)
- mesh_headings (all MeSH descriptor/qualifier strings)
- mesh_major (subset that are MajorTopic)

Rules:
- Use publication types as design evidence; do not infer design solely from title without pubtypes.
- Use MeSH (mesh_headings/mesh_major) as supportive evidence for Population/Intervention presence when T/A is weak.
- If P or I is missing from Title/Abstract AND MeSH does NOT clearly support it, BORDERLINE unless there is a clear mismatch -> EXCLUDE.
- Admin/adjunct (review/meta/guideline/case report) are excluded for effects triage (if protocol drop_adjuncts=true).
- Provide population_quote and intervention_quote as exact verbatim excerpts from title/abstract when possible; empty if absent.
- Return ONLY JSON as requested by the caller's schema, wrapped in BEGIN_JSON/END_JSON.
"""

def pass_a_user(protocol_json: str, record_json: str) -> str:
    return f"""PROTOCOL_JSON:
{protocol_json}

RECORD_JSON:
{record_json}

Return ONLY one JSON object in BEGIN_JSON/END_JSON.
"""

PASS_B_SYSTEM = """You are a critical auditor for a previous Pass A decision.
Given the protocol and record, either confirm or challenge the Pass A decision.

Return fields:
- stance: "confirm" or "challenge"
- flags: object with booleans (e.g., {"p_missing":true,"i_missing":false})
- confidence: float
- justification_short: one sentence

Return ONLY JSON in BEGIN_JSON/END_JSON.
"""

def pass_b_user(protocol_json: str, record_json: str, pass_a_json: str, trigger_note: str) -> str:
    return f"""PROTOCOL_JSON:
{protocol_json}

RECORD_JSON:
{record_json}

PASS_A_JSON:
{pass_a_json}

TRIGGER_NOTE: {trigger_note}

Return ONLY one JSON object in BEGIN_JSON/END_JSON.
"""

PASS_C_SYSTEM = """Consensus tie-break.
Resolve to include|borderline|exclude with reasons list and a short resolution note.
Return ONLY JSON in BEGIN_JSON/END_JSON.
"""

def pass_c_user(protocol_json: str, record_json: str, pass_a_json: str, pass_b_json: str) -> str:
    return f"""PROTOCOL_JSON:
{protocol_json}

RECORD_JSON:
{record_json}

PASS_A_JSON:
{pass_a_json}

PASS_B_JSON:
{pass_b_json}

Return ONLY one JSON object in BEGIN_JSON/END_JSON.
"""
