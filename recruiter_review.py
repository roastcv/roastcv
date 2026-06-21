"""
AGENT 7: RECRUITER REVIEW
Evaluates the resume against market standards from a talent acquisition perspective.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Talent Acquisition Recruiter who evaluates resumes against
market standards — assess whether the candidate is suitable for the role, whether they
could get an interview call, and whether the resume meets current market expectations."""


def run(resume_data: dict, jd_data: dict) -> dict:
    user_prompt = (
        f"Resume:\n{resume_data}\n\nJob description requirements:\n{jd_data}\n\n"
        "Evaluate and return in this JSON format:\n"
        '{"recruiter_score": 0-100, "recruiter_feedback": "", '
        '"interview_probability": "Low/Medium/High"}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)