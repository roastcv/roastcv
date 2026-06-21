"""
resume_templates.py
====================
A library of ready-made PDF resume templates, built by combining a small
number of distinct LAYOUT engines with several COLOR PALETTES. This avoids
hand-drawing 20+ totally separate designs while still producing visually
distinct, genuinely different-looking resumes.

    4 layouts  x  6 palettes  =  24 templates

LAYOUTS:
    classic  - centered header, single column (traditional, ATS-safe)
    minimal  - no color in body text, pure typography (safest for ATS)
    band     - full-width colored header banner, single column body
    sidebar  - two-column layout with a colored sidebar (skills/contact/education)

PALETTES:
    navy, teal, charcoal, maroon, forest, slate

PUBLIC API (this is what app.py / a website backend should import):
    list_templates() -> list[dict]
        Metadata for every template (id, name, layout, palette, ats_safe,
        description) — enough to render a template-picker UI.

    generate_resume_pdf(template_id, humanized_text, resume_data, output_path) -> str
        Renders the chosen template to `output_path` and returns that path.
        `humanized_text` / `resume_data` are the SAME inputs already produced
        by your pipeline (humanizer output text + resume_reader structured dict).

Adding a NEW layout later: write one `_build_<name>(output_path, resume_data,
humanized_text, palette)` function and register it in LAYOUT_META + LAYOUT_BUILDERS.
Every palette will automatically apply to it — no extra work per color.

Adding a NEW palette later: add one entry to PALETTE_META/PALETTES — it will
automatically apply to every existing layout.
"""

import re

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, BaseDocTemplate, PageTemplate, Frame, FrameBreak,
    Paragraph, Spacer, HRFlowable, Table, TableStyle,
)

# Reuse the existing, already-working text-parsing helpers instead of
# duplicating them — they are layout-agnostic.
from resume_pdf import (
    _extract_links,
    _parse_resume_text,
    _linkify,
    _looks_like_heading,
    _extract_name_from_text,
)

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
SIDEBAR_W = 62 * mm  # width of the colored sidebar in the "sidebar" layout

SECTION_ORDER = [
    ("summary", "PROFILE"),
    ("skills", "SKILLS"),
    ("experience", "EXPERIENCE"),
    ("projects", "PROJECTS"),
    ("education", "EDUCATION"),
    ("certifications", "CERTIFICATIONS"),
    ("achievements", "ACHIEVEMENTS"),
]

# Which sections go in the sidebar vs the main column for the "sidebar" layout
SIDEBAR_KEYS = {"skills", "education", "certifications"}
MAIN_KEYS = {"summary", "experience", "projects", "achievements"}


# ---------------------------------------------------------------------------
# 1. COLOR PALETTES
# ---------------------------------------------------------------------------
PALETTE_META = {
    "navy": "Navy Blue",
    "teal": "Teal",
    "charcoal": "Charcoal",
    "maroon": "Maroon",
    "forest": "Forest Green",
    "slate": "Slate Purple",
}

PALETTES = {
    "navy":     {"primary": "#1a2e4a", "accent": "#2563eb", "muted": "#64748b", "text": "#0f172a"},
    "teal":     {"primary": "#0f3d3e", "accent": "#0d9488", "muted": "#5e6a69", "text": "#102322"},
    "charcoal": {"primary": "#1f2125", "accent": "#3f4654", "muted": "#6b7280", "text": "#111111"},
    "maroon":   {"primary": "#4a1024", "accent": "#9f1239", "muted": "#7a6066", "text": "#1f1012"},
    "forest":   {"primary": "#1e3a26", "accent": "#15803d", "muted": "#5f6f63", "text": "#10180f"},
    "slate":    {"primary": "#2e2a4a", "accent": "#6d28d9", "muted": "#69647f", "text": "#15131f"},
}


def _c(hex_str):
    return colors.HexColor(hex_str)


# ---------------------------------------------------------------------------
# 2. SHARED HELPERS
# ---------------------------------------------------------------------------
def _get_section_lines(sections, key):
    # FIX: use `is not None` instead of truthiness check.
    # Previously, if sections[key] was an empty list [],
    # the `or` chain fell through to wrong fallback keys because [] is falsy.
    val = sections.get(key)
    if val is not None:
        return [l for l in val if l.strip()]
    return []


def _contact_line(resume_data, links, accent_hex, white=False):
    c = resume_data.get("contact", {})
    link_color = "#ffffff" if white else accent_hex
    parts = []
    if c.get("email"):
        parts.append(f'<a href="mailto:{c["email"]}" color="{link_color}">{c["email"]}</a>')
    if c.get("phone"):
        parts.append(c["phone"])
    if c.get("location"):
        parts.append(c["location"])
    for key in ("linkedin", "github", "portfolio"):
        url = links.get(key)
        if url:
            full = url if url.startswith("http") else "https://" + url
            label = "Portfolio" if key == "portfolio" else key.title()
            parts.append(f'<a href="{full}" color="{link_color}">{label}</a>')
    return "  |  ".join(parts)


def _styles(p, section_color_hex=None, body_color_hex=None):
    """Build the paragraph styles used by every layout, parameterized by palette."""
    primary, accent, muted, text = _c(p["primary"]), _c(p["accent"]), _c(p["muted"]), _c(p["text"])
    section_color = _c(section_color_hex) if section_color_hex else accent
    body_color = _c(body_color_hex) if body_color_hex else text
    return {
        "name": ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=21, leading=25,
                                textColor=primary, alignment=TA_CENTER, spaceAfter=4),
        "name_white": ParagraphStyle("name_white", fontName="Helvetica-Bold", fontSize=22, leading=26,
                                      textColor=colors.white, alignment=TA_CENTER, spaceAfter=4),
        "contact": ParagraphStyle("contact", fontName="Helvetica", fontSize=9,
                                   textColor=muted, alignment=TA_CENTER, spaceAfter=6),
        "contact_white": ParagraphStyle("contact_white", fontName="Helvetica", fontSize=9,
                                         textColor=colors.whitesmoke, alignment=TA_CENTER, spaceAfter=6),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10.5,
                                   textColor=section_color, spaceBefore=10, spaceAfter=3),
        "section_side": ParagraphStyle("section_side", fontName="Helvetica-Bold", fontSize=10,
                                        textColor=colors.white, spaceBefore=10, spaceAfter=3),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                                textColor=body_color, spaceAfter=3, leading=13.5),
        "body_white": ParagraphStyle("body_white", fontName="Helvetica", fontSize=9,
                                      textColor=colors.whitesmoke, spaceAfter=3, leading=13),
        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.5,
                                  textColor=body_color, spaceAfter=2, leading=13, leftIndent=10),
        "job_title": ParagraphStyle("job_title", fontName="Helvetica-Bold", fontSize=10,
                                     textColor=text, spaceAfter=1, spaceBefore=4),
        "job_meta": ParagraphStyle("job_meta", fontName="Helvetica-Oblique", fontSize=8.7,
                                    textColor=muted, spaceAfter=2),
    }


def _render_section(heading, lines, S, links, divider_color=None, white=False):
    """Turns one section's raw text lines into flowables. Reused by all layouts."""
    flow = [Paragraph(heading, S["section_side"] if white else S["section"])]
    if divider_color is not None:
        flow.append(HRFlowable(width="100%", thickness=0.8, color=divider_color, spaceAfter=4, spaceBefore=1))
    body_style = S["body_white"] if white else S["body"]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "• ", "* ", "· ")):
            text = line[2:].strip()
            shown = text if white else _linkify(text, links)
            flow.append(Paragraph(f"• {shown}", body_style))
        elif len(line) < 60 and (line.isupper() or _looks_like_heading(line)) and not white:
            flow.append(Paragraph(_linkify(line, links), S["job_title"]))
        elif re.search(r"\b(20\d{2}|present|current)\b", line.lower()) and not white:
            flow.append(Paragraph(line, S["job_meta"]))
        else:
            shown = line if white else _linkify(line, links)
            flow.append(Paragraph(shown, body_style))
    flow.append(Spacer(1, 4))
    return flow


def _doc_meta(resume_data, humanized_text, suffix):
    name = resume_data.get("name") or _extract_name_from_text(humanized_text) or "Resume"
    return {"title": f"{name} — {suffix}", "author": name}, name


# ---------------------------------------------------------------------------
# 3. LAYOUT 1 — CLASSIC (centered header, single column)
# ---------------------------------------------------------------------------
def _build_classic(output_path, resume_data, humanized_text, palette):
    p = PALETTES[palette]
    S = _styles(p)
    links = _extract_links(resume_data)
    sections = _parse_resume_text(humanized_text)
    meta, name = _doc_meta(resume_data, humanized_text, "Resume")

    story = []
    if name:
        story.append(Paragraph(name.upper(), S["name"]))
    contact_line = _contact_line(resume_data, links, p["accent"])
    if contact_line:
        story.append(Paragraph(contact_line, S["contact"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=_c(p["accent"]), spaceAfter=6, spaceBefore=2))

    for key, heading in SECTION_ORDER:
        lines = _get_section_lines(sections, key)
        if lines:
            story += _render_section(heading, lines, S, links, divider_color=_c(p["accent"]))

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN, **meta)
    doc.build(story)


# ---------------------------------------------------------------------------
# 4. LAYOUT 2 — MINIMAL (no color in body, safest for ATS parsers)
# ---------------------------------------------------------------------------
def _build_minimal(output_path, resume_data, humanized_text, palette):
    p = PALETTES[palette]
    # Section headings stay plain black/text-colored (not the bright accent) —
    # only the name keeps a faint trace of the chosen palette.
    S = _styles(p, section_color_hex=p["text"])
    links = _extract_links(resume_data)
    sections = _parse_resume_text(humanized_text)
    meta, name = _doc_meta(resume_data, humanized_text, "Resume")

    story = []
    if name:
        story.append(Paragraph(name.upper(), S["name"]))
    contact_line = _contact_line(resume_data, links, p["primary"])
    if contact_line:
        story.append(Paragraph(contact_line, S["contact"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=_c(p["muted"]), spaceAfter=8, spaceBefore=2))

    for key, heading in SECTION_ORDER:
        lines = _get_section_lines(sections, key)
        if lines:
            # No colored divider line under each heading — keeps it plain.
            story += _render_section(heading, lines, S, links, divider_color=None)

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN, **meta)
    doc.build(story)


# ---------------------------------------------------------------------------
# 5. LAYOUT 3 — BAND (full-width colored header banner)
# ---------------------------------------------------------------------------
def _build_band(output_path, resume_data, humanized_text, palette):
    p = PALETTES[palette]
    S = _styles(p)
    links = _extract_links(resume_data)
    sections = _parse_resume_text(humanized_text)
    meta, name = _doc_meta(resume_data, humanized_text, "Resume")

    header_inner = []
    if name:
        header_inner.append(Paragraph(name.upper(), S["name_white"]))
    contact_line = _contact_line(resume_data, links, p["accent"], white=True)
    if contact_line:
        header_inner.append(Paragraph(contact_line, S["contact_white"]))

    band = Table([[header_inner]], colWidths=[PAGE_W - 2 * MARGIN])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _c(p["primary"])),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    story = [band, Spacer(1, 10)]
    for key, heading in SECTION_ORDER:
        lines = _get_section_lines(sections, key)
        if lines:
            story += _render_section(heading, lines, S, links, divider_color=_c(p["accent"]))

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN, **meta)
    doc.build(story)


# ---------------------------------------------------------------------------
# 6. LAYOUT 4 — SIDEBAR (two-column: colored sidebar + main column)
# ---------------------------------------------------------------------------
def _sidebar_background(primary_hex):
    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(_c(primary_hex))
        canvas.rect(0, 0, SIDEBAR_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()
    return _draw


def _build_sidebar(output_path, resume_data, humanized_text, palette):
    p = PALETTES[palette]
    S = _styles(p)
    links = _extract_links(resume_data)
    sections = _parse_resume_text(humanized_text)
    meta, name = _doc_meta(resume_data, humanized_text, "Resume")

    # --- sidebar (colored) content: name, contact, skills, education, certs ---
    side_flow = []
    if name:
        side_flow.append(Paragraph(name.upper(), S["name_white"]))
        side_flow.append(Spacer(1, 4))
    contact_line = _contact_line(resume_data, links, p["accent"], white=True)
    if contact_line:
        # Contact info stacked (sidebar is narrow), not pipe-joined
        for part in contact_line.split("  |  "):
            side_flow.append(Paragraph(part, S["contact_white"]))
        side_flow.append(Spacer(1, 6))
    for key in ("skills", "education", "certifications"):
        heading = dict(SECTION_ORDER)[key]
        lines = _get_section_lines(sections, key)
        if lines:
            side_flow += _render_section(heading, lines, S, links, divider_color=None, white=True)

    # --- main column (white) content: summary, experience, projects, achievements ---
    main_flow = []
    for key in ("summary", "experience", "projects", "achievements"):
        heading = dict(SECTION_ORDER)[key]
        lines = _get_section_lines(sections, key)
        if lines:
            main_flow += _render_section(heading, lines, S, links, divider_color=_c(p["accent"]))

    story = side_flow + [FrameBreak()] + main_flow

    left_frame = Frame(6 * mm, MARGIN, SIDEBAR_W - 12 * mm, PAGE_H - 2 * MARGIN, id="left")
    right_frame = Frame(SIDEBAR_W + 8 * mm, MARGIN, PAGE_W - SIDEBAR_W - 8 * mm - MARGIN,
                         PAGE_H - 2 * MARGIN, id="right")

    doc = BaseDocTemplate(output_path, pagesize=A4, **meta)
    doc.addPageTemplates([
        PageTemplate(id="sidebar_template", frames=[left_frame, right_frame],
                     onPage=_sidebar_background(p["primary"]))
    ])
    doc.build(story)


# ---------------------------------------------------------------------------
# 7. REGISTRY — combine every layout with every palette
# ---------------------------------------------------------------------------
LAYOUT_META = {
    "classic": {
        "label": "Classic",
        "description": "Centered header, single column — clean and traditional.",
        "ats_safe": True,
    },
    "minimal": {
        "label": "Minimal",
        "description": "No color in the body text, pure typography — safest choice for ATS parsers.",
        "ats_safe": True,
    },
    "band": {
        "label": "Header Band",
        "description": "Bold full-width colored header banner, single column body below it.",
        "ats_safe": True,
    },
    "sidebar": {
        "label": "Sidebar",
        "description": "Two-column layout with a colored sidebar for contact/skills/education.",
        "ats_safe": False,  # some ATS parsers struggle with multi-column text order
    },
}

LAYOUT_BUILDERS = {
    "classic": _build_classic,
    "minimal": _build_minimal,
    "band": _build_band,
    "sidebar": _build_sidebar,
}

TEMPLATES = []
for _layout_id, _lmeta in LAYOUT_META.items():
    for _pal_id, _pal_label in PALETTE_META.items():
        TEMPLATES.append({
            "id": f"{_layout_id}_{_pal_id}",
            "name": f"{_lmeta['label']} — {_pal_label}",
            "layout": _layout_id,
            "palette": _pal_id,
            "ats_safe": _lmeta["ats_safe"],
            "description": _lmeta["description"],
        })

_TEMPLATES_BY_ID = {t["id"]: t for t in TEMPLATES}


# ---------------------------------------------------------------------------
# 8. PUBLIC API
# ---------------------------------------------------------------------------
def list_templates():
    """Returns metadata for every available template — use this to build a picker UI."""
    return TEMPLATES


def generate_resume_pdf(template_id: str, humanized_text: str, resume_data: dict, output_path: str) -> str:
    """
    Renders the resume using the chosen template.

    Args:
        template_id:    one of the ids returned by list_templates(), e.g. "sidebar_teal"
        humanized_text: plain text resume (output of the humanizer agent)
        resume_data:    structured dict (output of the resume_reader agent)
        output_path:    where to save the PDF
    """
    template = _TEMPLATES_BY_ID.get(template_id)
    if template is None:
        valid = ", ".join(sorted(_TEMPLATES_BY_ID.keys()))
        raise ValueError(f"Unknown template_id '{template_id}'. Valid ids: {valid}")
    builder = LAYOUT_BUILDERS[template["layout"]]
    builder(output_path, resume_data, humanized_text, template["palette"])
    return output_path