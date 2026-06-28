"""
AGENT 3: ATS REVIEW
Analyzes the resume from an ATS (Applicant Tracking System) perspective.

NOTE: Real ATS systems (Workday, Taleo, Greenhouse, etc.) use proprietary algorithms
that are not publicly available. The score returned here is a HEURISTIC ESTIMATE,
not a guarantee of what any real ATS system would produce.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are an ATS (Applicant Tracking System) Compatibility Reviewer.
Evaluate the resume based on keyword optimization, structure, readability, section
hierarchy, ATS compatibility, and formatting.

IMPORTANT: Your score is an estimate/heuristic because real ATS algorithms
(Workday, Taleo, etc.) are proprietary and not publicly known. Communicate this clearly."""


def run(resume_text: str) -> dict:
    user_prompt = (
        f"Resume text:\n{resume_text}\n\n"
        "Evaluate and return in this JSON format:\n"
        '{"ats_score": 0-100, "ats_issues": [], "ats_improvement_suggestions": [], '
        '"disclaimer": "This is an estimated score. Actual ATS results may vary."}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)