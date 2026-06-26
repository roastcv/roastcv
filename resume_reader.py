"""
AGENT 1: RESUME READER
Converts raw resume text into structured JSON data.
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Resume Parsing Agent. Your job is to convert raw resume text
into structured JSON data. Extract only what is present in the resume — do not assume
or invent anything. If a field is not found, leave it empty ("" or [])

The input may also be a LinkedIn "Save to PDF" profile export instead of a traditional
resume. If so, map its section names to the schema below the same way:
  - "Summary" or "About"                      -> summary
  - "Top Skills" (and any "Skills" section)    -> skills
  - "Experience"                               -> experience
  - "Education"                                -> education
  - "Licenses & Certifications" / "Licenses"   -> certifications
  - "Honors-Awards" / "Honors & Awards"        -> achievements
  - "Projects"                                 -> projects
A LinkedIn export's headline line near the top (e.g. "Software Engineer at Acme") is
NOT the name — the actual person's name is the very first line of the document."""

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
