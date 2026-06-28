"""
AGENT 6: HR ROAST
Provides a brutally honest review from the perspective of an experienced HR recruiter.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are an experienced HR Recruiter with 10+ years of experience giving
brutally honest resume reviews. Do not sugar-coat. Answer the real questions a recruiter asks:
Would you shortlist this candidate? Is the resume impressive? Are projects clearly explained?
Are achievements measurable? Does this resume deserve an interview?
Be honest but constructive — not just critical, give reasons too."""


def run(resume_data: dict, jd_data: dict) -> dict:
    user_prompt = (
        f"Resume:\n{resume_data}\n\nJob Description requirements:\n{jd_data}\n\n"
        "Provide a brutally honest review in this JSON format:\n"
        '{"hr_score": 0-100, "hr_feedback": "", "shortlist_decision": "Yes/No/Maybe", '
        '"critical_issues": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)