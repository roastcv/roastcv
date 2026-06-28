"""
AGENT 4: JOB DESCRIPTION ANALYZER
Analyzes the job description and extracts structured requirements.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Job Description Analyzer. Carefully read the job description
and extract its key requirements in a structured format. Clearly differentiate between
required and preferred skills."""


def run(jd_text: str) -> dict:
    user_prompt = (
        f"Job Description:\n{jd_text}\n\n"
        "Extract and return in this JSON format:\n"
        '{"required_skills": [], "preferred_skills": [], "tools_technologies": [], '
        '"responsibilities": [], "experience_required": "", "important_keywords": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)