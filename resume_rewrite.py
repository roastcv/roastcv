"""
AGENT 8: RESUME REWRITE
Improves the resume — better summary, better project descriptions, better wording,
ATS optimization. STRICT RULE: never generates fake information.

FIXES:
- Projects treated as Experience if no formal work experience (ATS-safe)
- LinkedIn/GitHub full URLs preserved always
- Emojis removed from all sections
- Metrics extracted from existing context (not invented)
- Standard ATS section headers enforced
"""

from llm_client import call_llm

SYSTEM_PROMPT = """You are a Resume Rewrite Agent. Your goal is to improve the resume —
better summary, better project descriptions, stronger achievements, better wording,
and ATS optimization.

STRICT RULES (never break these):
- Do NOT add any fake experience, fake projects, fake achievements, or fake skills.
- Only present the information already in the resume in a better way.
- Enhance and clarify existing facts — do not invent new facts.
- Only add numbers/metrics if they are already implied or mentioned in the original resume.
- NEVER remove LinkedIn URL, GitHub URL, portfolio URL, or email — always keep full URLs.
- Remove ALL emojis from every section — ATS systems cannot parse emojis correctly.

ATS FORMATTING RULES (always follow):
- Use ONLY these section headers (exact spelling): SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS, ACHIEVEMENTS
- Bullet points must start with "- " (dash space) — no special characters
- Contact line format: email | phone | linkedin.com/in/username | github.com/username
- No tables, no columns, no text boxes — plain single-column text only
- Skills must be listed as individual bullet points, not comma-separated in one line

EXPERIENCE SECTION RULES:
- If the candidate has NO formal work experience, do NOT leave the section blank or write "no experience"
- Instead, list their strongest projects under EXPERIENCE with this format:
    Project Name | Personal Project | Year
    - bullet about what they built
    - bullet about tools used
    - bullet about outcome or impact
- This is ATS-safe and shows practical ability honestly

METRICS RULES:
- If the resume mentions a dataset, estimate a reasonable scale (e.g. "ride-booking data" → "10,000+ records")
- If dashboards are mentioned, note how many KPIs or metrics were tracked
- If SQL analysis is mentioned, note the type of business decisions it supported
- Only use numbers that are reasonable based on what is already described — never invent"""


def run(resume_data: dict, audit: dict, gap_analysis: dict) -> str:
    # Extract contact info to explicitly remind the agent to preserve it
    contact = resume_data.get("contact", {})
    contact_reminder = []
    if contact.get("linkedin"):
        contact_reminder.append(f"LinkedIn URL: {contact['linkedin']}")
    if contact.get("github"):
        contact_reminder.append(f"GitHub URL: {contact['github']}")
    if contact.get("portfolio"):
        contact_reminder.append(f"Portfolio URL: {contact['portfolio']}")
    if contact.get("email"):
        contact_reminder.append(f"Email: {contact['email']}")
    if contact.get("phone"):
        contact_reminder.append(f"Phone: {contact['phone']}")

    contact_block = (
        "IMPORTANT — Preserve these contact details exactly as given:\n" +
        "\n".join(contact_reminder) + "\n\n"
    ) if contact_reminder else ""

    has_experience = bool(resume_data.get("experience"))
    experience_hint = (
        ""
        if has_experience
        else (
            "NOTE: This candidate has NO formal work experience. "
            "Under the EXPERIENCE section, list their top 2-3 projects "
            "formatted as project-based experience entries (Project Name | Personal Project | Year). "
            "Do NOT write 'no experience' or leave it blank.\n\n"
        )
    )

    user_prompt = (
        f"{contact_block}"
        f"{experience_hint}"
        f"Original resume data:\n{resume_data}\n\n"
        f"Audit findings (weaknesses to fix):\n{audit.get('weaknesses', [])}\n"
        f"Suggestions:\n{audit.get('improvement_suggestions', [])}\n\n"
        f"Gap analysis recommendations:\n{gap_analysis.get('recommendations', [])}\n"
        f"Missing keywords (weave in naturally ONLY IF the candidate genuinely has that skill):\n"
        f"{gap_analysis.get('missing_keywords', [])}\n\n"
        "Now write an improved, complete resume in plain text format.\n"
        "Sections in this order: SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS, ACHIEVEMENTS\n"
        "Rules:\n"
        "- Remove ALL emojis\n"
        "- Keep all URLs exactly as provided\n"
        "- Use '- ' bullet points only\n"
        "- Use only real information\n"
        "- Add reasonable metrics where context supports them"
    )
    return call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=3000)