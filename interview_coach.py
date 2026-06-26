"""
AGENT 9: HUMANIZER
Removes AI-sounding language from the resume and gives it a natural, human tone.

FIXES:
- Explicitly told to preserve all URLs (LinkedIn, GitHub, portfolio)
- Explicitly told to remove emojis if any slipped through rewrite
- Explicitly told NOT to remove contact details
- Better instructions for natural tone without losing ATS keywords

Input: plain text resume string (output from resume_rewrite agent).
"""

from llm_client import call_llm_json

SYSTEM_PROMPT = """You are a Resume Humanizer Agent. Your job is to make the resume sound
less AI-generated and more like a real person wrote it.

WHAT TO REMOVE — AI buzzwords and clichés:
- "results-driven professional"
- "highly motivated individual"  
- "passionate professional"
- "dynamic team player"
- "leverage synergies"
- "detail-oriented self-starter"
- "proven track record"
- "strong communication skills"
- "go-getter" / "thought leader" / "guru"
- Any phrase that sounds like a job ad wrote it

WHAT TO KEEP — always preserve these exactly:
- All URLs: LinkedIn, GitHub, portfolio — never shorten or remove
- Email address and phone number
- All technical skill names (SQL, Python, Power BI, etc.)
- ATS keywords from the job description
- All section headers: SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS, ACHIEVEMENTS
- All bullet points starting with "- "
- All numbers and metrics

WHAT TO FIX:
- Remove any remaining emojis (🏆 ✅ etc.) — ATS cannot parse them
- Rewrite in natural first-person or third-person tone consistently
- Make bullet points start with strong action verbs (Built, Analyzed, Developed, Created, Designed)
- Keep sentences concise — max 2 lines per bullet
- Summary should sound like the candidate wrote it, not a robot

NOTE: The "ai_detection_risk_score" is a rough estimate only. AI detection tools have
a high false positive rate — treat this as a directional guide, not exact science."""


def run(resume_text: str) -> dict:
    """
    Args:
        resume_text: Plain text resume string (from resume_rewrite agent).
    Returns:
        dict with humanized_resume, humanization_score, ai_detection_risk_score, changes_made.
    """
    user_prompt = (
        f"Resume:\n{resume_text}\n\n"
        "Humanize this resume — make it sound like a real person wrote it.\n"
        "CRITICAL: Keep all URLs, contact info, technical keywords, and section headers intact.\n"
        "Remove all emojis. Use strong action verbs for bullets.\n\n"
        "Return ONLY this JSON format (no extra text):\n"
        '{"humanized_resume": "", "humanization_score": 0-100, '
        '"ai_detection_risk_score": 0-100, "changes_made": []}'
    )
    return call_llm_json(SYSTEM_PROMPT, user_prompt)
