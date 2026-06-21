"""
AGENT 1: RESUME READER
Converts raw resume text into structured JSON data.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Resume Parsing Agent. Your job is to convert raw resume text
into structured JSON data. Extract only what is present in the resume — do not assume
or invent anything. If a field is not found, leave it empty ("" or [])."""

SCHEMA_HINT = """{
  "name": "",
  "contact": {"email": "", "phone": "", "linkedin": "", "github": "", "portfolio": "", "location": ""},
  "summary": "",
  "skills": [],
  "education": [{"degree": "", "institution": "", "year": ""}],
  "experience": [{"role": "", "company": "", "duration": "", "description": []}],
  "projects": [{"title": "", "description": "", "tech_used": []}],
  "certifications": [],
  "achievements": []
}"""


def run(resume_text: str) -> dict:
    user_prompt = (
        f"Raw resume text:\n\n{resume_text}\n\n"
        f"Structure this into the following exact JSON schema:\n{SCHEMA_HINT}"
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)