"""
AGENT 5: GAP ANALYSIS
Compares resume against the job description to identify skill gaps, keyword gaps, and matches.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Gap Analysis Agent. Compare the resume data against the job
description data and identify where the candidate is strong and where there are gaps.
Be specific — avoid generic advice like "improve skills". Name the exact missing skill
or keyword."""


def run(resume_data: dict, jd_data: dict) -> dict:
    user_prompt = (
        f"Resume data:\n{resume_data}\n\nJob description data:\n{jd_data}\n\n"
        "Compare and return in this JSON format:\n"
        '{"match_percentage": 0-100, "missing_skills": [], "missing_keywords": [], '
        '"strong_matches": [], "weak_matches": [], "recommendations": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)