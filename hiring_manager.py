"""
AGENT 10: HIRING MANAGER
Evaluates the candidate from a hiring manager's perspective — team fit and practical skills.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Hiring Manager who directly evaluates candidates who will
work on your team. Look through a practical lens: will this candidate add value to the
team, are their projects relevant, do their skills seem practical, should they be
considered for an interview?"""


def run(resume_data: dict, jd_data: dict) -> dict:
    user_prompt = (
        f"Resume:\n{resume_data}\n\nJob description:\n{jd_data}\n\n"
        "Evaluate and return in this JSON format:\n"
        '{"hiring_manager_score": 0-100, "hiring_recommendation": "", '
        '"team_fit_notes": "", "practical_skill_assessment": ""}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)