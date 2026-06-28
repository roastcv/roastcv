"""
AGENT 2: RESUME AUDIT AGENT
Evaluates overall resume quality — missing sections, weak summary,
formatting issues, grammar issues, repeated content, low-impact bullets.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Resume Audit Agent — a senior resume reviewer.
Evaluate the overall quality of the resume: check for missing sections, weak summary,
weak project descriptions, formatting issues, grammar issues, repeated content,
incomplete information, and low-impact bullet points.
Be honest and specific — no generic advice. For every weakness, give the exact
example from the resume that demonstrates the problem."""


def run(resume_text: str, resume_data: dict) -> dict:
    user_prompt = (
        f"Structured resume data:\n{resume_data}\n\nRaw resume text:\n{resume_text}\n\n"
        "Evaluate and return in this JSON format:\n"
        '{"health_score": 0-100, "strengths": [], "weaknesses": [], '
        '"improvement_suggestions": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)