"""
resume_pdf.py — ATS-friendly PDF resume generator with professional template.
Generates a clean, formatted PDF from humanized resume text.
Links from original resume are preserved if present.

FIX (this revision):
- Section parsing now keys sections by their canonical name directly
  (e.g. "skills") instead of the first word of the raw heading text
  (e.g. "technical" for "TECHNICAL SKILLS"). The previous approach silently
  dropped any section whose heading wasn't a single word, because the
  lookup at render time (sections.get(key) / sections.get(key+'s') / ...)
  never matched the stored key. The new _SECTION_DEFS list maps each regex
  pattern straight to its canonical key, so parsing and lookup can never
  disagree.
"""

import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Color palette ────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1a2e4a")
ACCENT_BLUE = colors.HexColor("#2563eb")
LIGHT_GRAY  = colors.HexColor("#f1f5f9")
MID_GRAY    = colors.HexColor("#64748b")
BLACK       = colors.HexColor("#0f172a")
WHITE       = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


# ── Styles ───────────────────────────────────────────────────
def _styles():
    return {
        "name": ParagraphStyle(
            "name", fontName="Helvetica-Bold", fontSize=22,
            textColor=DARK_BLUE, spaceAfter=2, alignment=TA_CENTER,
        ),
        "contact": ParagraphStyle(
            "contact", fontName="Helvetica", fontSize=9,
            textColor=MID_GRAY, spaceAfter=6, alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=11,
            textColor=ACCENT_BLUE, spaceBefore=10, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9.5,
            textColor=BLACK, spaceAfter=3, leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName="Helvetica", fontSize=9.5,
            textColor=BLACK, spaceAfter=2, leading=13,
            leftIndent=12, bulletIndent=0,
        ),
        "job_title": ParagraphStyle(
            "job_title", fontName="Helvetica-Bold", fontSize=10,
            textColor=BLACK, spaceAfter=1, spaceBefore=4,
        ),
        "job_meta": ParagraphStyle(
            "job_meta", fontName="Helvetica-Oblique", fontSize=9,
            textColor=MID_GRAY, spaceAfter=2,
        ),
    }


def _extract_links(resume_data: dict) -> dict:
    """Pull links from original resume_data so we can embed them."""
    links = {}
    contact = resume_data.get("contact", {})
    if contact.get("linkedin"):
        links["linkedin"] = contact["linkedin"]
    if contact.get("github"):
        links["github"] = contact.get("github", "")
    if contact.get("portfolio"):
        links["portfolio"] = contact.get("portfolio", "")
    # Also scan raw fields for URLs
    for key, val in contact.items():
        if isinstance(val, str) and val.startswith("http"):
            links.setdefault(key, val)
    return links


def _make_contact_line(resume_data: dict, links: dict) -> str:
    contact = resume_data.get("contact", {})
    parts = []
    if contact.get("email"):
        parts.append(f'<a href="mailto:{contact["email"]}" color="#2563eb">{contact["email"]}</a>')
    if contact.get("phone"):
        parts.append(contact["phone"])
    if contact.get("location"):
        parts.append(contact["location"])
    if links.get("linkedin"):
        url = links["linkedin"]
        if not url.startswith("http"):
            url = "https://" + url
        parts.append(f'<a href="{url}" color="#2563eb">LinkedIn</a>')
    if links.get("github"):
        url = links["github"]
        if not url.startswith("http"):
            url = "https://" + url
        parts.append(f'<a href="{url}" color="#2563eb">GitHub</a>')
    if links.get("portfolio"):
        url = links["portfolio"]
        if not url.startswith("http"):
            url = "https://" + url
        parts.append(f'<a href="{url}" color="#2563eb">Portfolio</a>')
    return "  |  ".join(parts)


def _section_divider():
    return HRFlowable(width="100%", thickness=1, color=ACCENT_BLUE, spaceAfter=4, spaceBefore=2)


# ── Section parser ────────────────────────────────────────────
# FIX: each entry maps directly to the canonical key used later for lookup,
# instead of deriving the key from the first word of the matched heading
# (e.g. "TECHNICAL SKILLS" used to become key "technical", which never
# matched any lookup variant tried at render time — the section just
# silently disappeared from the PDF).
_SECTION_DEFS = [
    ("summary",        r"^(summary|profile|objective)"),
    ("skills",         r"^(skills|technical skills|core competencies|key skills)"),
    ("experience",     r"^(experience|work experience|employment|professional experience)"),
    ("projects",       r"^(projects|key projects|personal projects)"),
    ("education",      r"^(education|academic background|qualifications)"),
    ("certifications", r"^(certifications?|licenses?|courses?)"),
    ("achievements",   r"^(achievements?|accomplishments?|awards?)"),
]


def _match_section_key(line: str):
    """Returns the canonical section key if `line` looks like a section
    heading, else None."""
    lowered = line.lower()
    for key, pattern in _SECTION_DEFS:
        if re.match(pattern, lowered):
            return key
    return None


def _parse_resume_text(text: str) -> dict:
    """
    Parse plain-text resume into a sections dict, keyed by canonical
    section name ("summary", "skills", "experience", ...), plus "header"
    for any preamble before the first recognized heading.
    """
    sections = {}
    current_section = "header"
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        matched_key = _match_section_key(stripped)
        if matched_key:
            sections[current_section] = current_lines
            current_section = matched_key
            current_lines = []
        else:
            current_lines.append(stripped)

    sections[current_section] = current_lines
    return sections


def generate_resume_pdf(
    humanized_text: str,
    resume_data: dict,
    output_path: str,
):
    """
    Generate a professional ATS-friendly PDF resume.

    Args:
        humanized_text: Plain text from humanizer agent
        resume_data:    Structured dict from resume_reader agent (for name/contact/links)
        output_path:    Where to save the PDF
    """
    S = _styles()
    links = _extract_links(resume_data)
    story = []

    # ── Header: Name ─────────────────────────────────────────
    name = resume_data.get("name", "") or _extract_name_from_text(humanized_text)
    if name:
        story.append(Paragraph(name.upper(), S["name"]))

    # ── Header: Contact line ──────────────────────────────────
    contact_line = _make_contact_line(resume_data, links)
    if contact_line:
        story.append(Paragraph(contact_line, S["contact"]))

    story.append(_section_divider())

    # ── Parse resume text into sections ──────────────────────
    sections = _parse_resume_text(humanized_text)

    section_order = [
        ("summary",        "PROFESSIONAL SUMMARY"),
        ("skills",         "SKILLS"),
        ("experience",     "EXPERIENCE"),
        ("projects",       "PROJECTS"),
        ("education",      "EDUCATION"),
        ("certifications", "CERTIFICATIONS"),
        ("achievements",   "ACHIEVEMENTS"),
    ]

    for key, heading in section_order:
        # FIX: sections are now stored under their canonical key directly,
        # so a plain lookup is enough — no more guessing variants like
        # "technical " + key or "work " + key.
        content_lines = [l for l in sections.get(key, []) if l.strip()]
        if not content_lines:
            continue

        story.append(Paragraph(heading, S["section"]))
        story.append(_section_divider())

        for line in content_lines:
            line = line.strip()
            if not line:
                continue
            # Bullet points
            if line.startswith(("- ", "• ", "* ", "· ")):
                text = line[2:].strip()
                story.append(Paragraph(f"• {_linkify(text, links)}", S["bullet"]))
            # Role/company lines (bold if all caps or title-case short line)
            elif len(line) < 60 and (line.isupper() or _looks_like_heading(line)):
                story.append(Paragraph(_linkify(line, links), S["job_title"]))
            # Date / meta lines
            elif re.search(r"\b(20\d{2}|present|current)\b", line.lower()):
                story.append(Paragraph(line, S["job_meta"]))
            else:
                story.append(Paragraph(_linkify(line, links), S["body"]))

        story.append(Spacer(1, 4))

    # ── Build PDF ─────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"{name} — Resume",
        author=name,
    )
    doc.build(story)
    return output_path


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    if len(words) > 8:
        return False
    cap_words = sum(1 for w in words if w and w[0].isupper())
    return cap_words / max(len(words), 1) > 0.6


def _extract_name_from_text(text: str) -> str:
    """Fallback: first non-empty line is usually the name."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _linkify(text: str, links: dict) -> str:
    """Replace raw URLs in text with clickable PDF links."""
    def replacer(m):
        url = m.group(0)
        display = url.replace("https://", "").replace("http://", "").rstrip("/")
        return f'<a href="{url}" color="#2563eb">{display}</a>'
    return re.sub(r"https?://[^\s,)>\"']+", replacer, text)


def generate_cover_letter_pdf(
    cover_letter_text: str,
    resume_data: dict,
    company_name: str = "",
    output_path: str = "cover_letter.pdf",
):
    """
    Generate a professional PDF cover letter.

    Args:
        cover_letter_text: Plain text cover letter from cover_letter agent
        resume_data:       Structured dict from resume_reader agent (for name/contact)
        company_name:      Target company name
        output_path:       Where to save the PDF
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from datetime import date

    DARK_BLUE   = colors.HexColor("#1a2e4a")
    ACCENT_BLUE = colors.HexColor("#2563eb")
    MID_GRAY    = colors.HexColor("#64748b")
    BLACK       = colors.HexColor("#0f172a")
    MARGIN      = 20 * mm

    name_style = ParagraphStyle(
        "cl_name", fontName="Helvetica-Bold", fontSize=18,
        textColor=DARK_BLUE, spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "cl_contact", fontName="Helvetica", fontSize=9,
        textColor=MID_GRAY, spaceAfter=4,
    )
    date_style = ParagraphStyle(
        "cl_date", fontName="Helvetica", fontSize=10,
        textColor=MID_GRAY, spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "cl_body", fontName="Helvetica", fontSize=10,
        textColor=BLACK, spaceAfter=8, leading=16,
    )

    story = []

    # Header: name
    name = resume_data.get("name", "")
    if name:
        story.append(Paragraph(name, name_style))

    # Header: contact
    contact = resume_data.get("contact", {})
    contact_parts = []
    if contact.get("email"):
        contact_parts.append(contact["email"])
    if contact.get("phone"):
        contact_parts.append(contact["phone"])
    if contact.get("location"):
        contact_parts.append(contact["location"])
    if contact_parts:
        story.append(Paragraph("  |  ".join(contact_parts), contact_style))

    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT_BLUE, spaceAfter=10, spaceBefore=4))

    # Date
    story.append(Paragraph(date.today().strftime("%B %d, %Y"), date_style))

    # Company name if provided
    if company_name:
        story.append(Paragraph(f"Hiring Team, {company_name}", body_style))
        story.append(Spacer(1, 6))

    # Cover letter body — split by paragraphs
    for para in cover_letter_text.strip().split("\n\n"):
        para = para.strip()
        if para:
            # Replace single newlines with spaces within a paragraph
            para = para.replace("\n", " ")
            story.append(Paragraph(para, body_style))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"{name} — Cover Letter",
        author=name,
    )
    doc.build(story)
    return output_path