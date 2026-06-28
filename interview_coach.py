"""
AGENT 11: INTERVIEW COACH
Generates candidate-specific interview questions based on the resume and job description.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are an Interview Coach. Generate candidate-specific interview
questions based on the resume and job description — not generic questions, but ones
tied to the candidate's actual projects and experience. Provide 4-6 questions per category."""


def run(resume_data: dict, jd_data: dict) -> dict:
    user_prompt = (
        f"Resume:\n{resume_data}\n\nJob description:\n{jd_data}\n\n"
        "Return questions in this JSON format:\n"
        '{"hr_questions": [], "technical_questions": [], "project_questions": [], '
        '"scenario_based_questions": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)