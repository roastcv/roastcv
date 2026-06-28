"""
AGENT 12: COVER LETTER
Generates a customized cover letter based on the resume and job description.

FIX: Now accepts humanized_resume text as optional input.
If humanized text is available, it is preferred over raw resume_data
because it sounds more natural and human — which makes the cover letter
sound better too.
"""

from llm_client import call_llm

SYSTEM_PROMPT = """You are a Cover Letter Writer. Write a personalized, natural-sounding
(not AI-generated) cover letter based on the candidate's resume and the job description.
Use only real resume information — do not invent anything.
The tone should match the resume — professional but human, not robotic."""


def run(
    resume_data: dict,
    jd_data: dict,
    company_name: str = "",
    humanized_resume: str = "",
) -> str:
    """
    Args:
        resume_data:      Structured dict from resume_reader agent.
        jd_data:          Structured dict from jd_analyzer agent.
        company_name:     Optional company name for personalization.
        humanized_resume: Plain text from humanizer agent (preferred if available).
    """
    company_line = f"Company: {company_name}\n" if company_name else ""

    # Use humanized plain text if available — it sounds more natural
    if humanized_resume and humanized_resume.strip():
        resume_section = f"Candidate's resume (humanized, natural tone):\n{humanized_resume}"
    else:
        resume_section = f"Candidate's structured resume data:\n{resume_data}"

    user_prompt = (
        f"{company_line}"
        f"{resume_section}\n\n"
        f"Job description requirements:\n{jd_data}\n\n"
        "Write a concise (under 300 words), professional, natural-sounding cover letter. "
        "Do NOT use generic AI phrases like 'I am excited to apply' or 'passionate professional'. "
        "Make it specific to the candidate's actual experience and the job requirements."
    )
    return call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=1000)