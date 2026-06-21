"""
RESUME AI AGENT — Improved Web Interface (Streamlit)
Features:
 - Professional landing page with hero section
 - Engaging step-by-step progress with tips shown during analysis
 - Strategic ad placements (Google AdSense ready)
 - Score visualization with color-coded badges
 - Better UX with tabs, expandable sections, and download options

RUN:
    streamlit run app.py
"""

import os
import time
import tempfile
import contextlib
import streamlit as st

from main import run_pipeline
from resume_templates import generate_resume_pdf, list_templates
from resume_pdf import generate_cover_letter_pdf
from resume_docx import generate_resume_docx, generate_cover_letter_docx


@contextlib.contextmanager
def temp_output_path(suffix: str):
    """
    Creates a temp file path and GUARANTEES cleanup, even if the generator
    function raises partway through (e.g. PDF/DOCX build fails after the
    file was created but before content was written).

    FIX: previously each download button created a NamedTemporaryFile with
    delete=False and called os.unlink() manually right after use — but if
    generate_resume_pdf()/generate_resume_docx() raised before reaching the
    unlink line, the temp file was silently left behind on disk forever.
    Wrapping creation + cleanup in a try/finally closes that leak.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    try:
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass  # best-effort cleanup — don't crash the UI over this

# ── Page Config ───────────────────────────────────────────────────────────────
# Browser tab icon uses the actual logo file instead of an emoji. Falls back
# to the emoji only if logo.png isn't present (e.g. running before the asset
# is copied in), so set_page_config never crashes.
_page_icon = "logo.png" if os.path.exists("logo.png") else "🔥"

st.set_page_config(
    page_title="RoastCV — Free AI Resume Analyzer",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "mailto:support@roastcv.io",
        "About": "RoastCV — Powered by AI agents to help you land your dream job.",
    },
)


# ── Inject tags into the real page <head> ───────────────────────────────────
# Streamlit serves one static index.html shell for the whole app, and there's
# no first-class API to add things to its <head> (a sandboxed components.html()
# iframe can't reach window.parent.document either — Streamlit denies it
# `allow-same-origin`). So this patches the snippets straight into Streamlit's
# own static/index.html on disk, once, at process startup.
#
# Each tag lives in a named "slot" wrapped in start/end marker comments that
# DON'T contain the tag's value (e.g. the GA Measurement ID) — only the slot
# name. That's what makes this safe to re-run: if the GA ID is ever changed
# (like just now, switching accounts), the new snippet replaces whatever was
# between the old slot's markers instead of sitting next to it as a stray
# duplicate. A slot that's not there yet just gets inserted after <head>.
def _patch_streamlit_head(slots: dict):
    import pathlib
    import re

    logs = []  # collected here instead of print()'d — the Cloud "Logs" viewer
               # doesn't reliably surface per-visit stdout, so this gets shown
               # directly on the page instead (see the debug expander below).

    try:
        index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
        logs.append(f"index.html path: {index_path}")
        logs.append(f"exists: {index_path.exists()}")

        html = index_path.read_text(encoding="utf-8")
        logs.append(f"read {len(html)} chars — contains '<head>': {'<head>' in html}")
        changed = False

        for slot_name, snippet in slots.items():
            start = f"<!-- {slot_name}:start -->"
            end = f"<!-- {slot_name}:end -->"
            block = f"{start}\n{snippet}\n{end}"
            pattern = re.escape(start) + r".*?" + re.escape(end)

            if re.search(pattern, html, flags=re.DOTALL):
                new_html = re.sub(pattern, block, html, flags=re.DOTALL)
                logs.append(f"slot '{slot_name}': found existing block, replaced")
            else:
                new_html = html.replace("<head>", "<head>\n" + block + "\n", 1)
                logs.append(f"slot '{slot_name}': no existing block, inserted fresh")

            if new_html != html:
                html = new_html
                changed = True

        if changed:
            index_path.write_text(html, encoding="utf-8")
            logs.append("wrote updated index.html successfully")
        else:
            logs.append("no changes needed, file already up to date")
    except Exception as e:
        # Tags are non-critical — never let a patch failure break the app —
        # but DO surface it (in the debug expander) so it's not invisible.
        logs.append(f"FAILED: {type(e).__name__}: {e}")

    return logs


_GA_MEASUREMENT_ID = "G-P0VRFWVQ9T"

_head_patch_logs = _patch_streamlit_head({
    "google-analytics": f"""<script async src="https://www.googletagmanager.com/gtag/js?id={_GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{ dataLayer.push(arguments); }}
  gtag('js', new Date());
  gtag('config', '{_GA_MEASUREMENT_ID}');
</script>""",

    "google-site-verification": (
        '<meta name="google-site-verification" '
        'content="_qo1PUczRxCQ8jxIjllvlFqrJmrMolPLlDgZwtDT4oU" />'
    ),
})

# TEMPORARY — visible right on the page so it doesn't depend on the Cloud
# logs viewer at all. Remove this expander once the tags are confirmed
# working (search this comment to find it again).
with st.expander("🔧 Debug: GA / verification tag patch status", expanded=True):
    for _line in _head_patch_logs:
        st.code(_line, language=None)


MAX_RESUME_SIZE_MB = 5


@st.cache_data
def _logo_data_uri() -> str:
    """
    Reads logo.png once and returns it as a base64 data URI so it can be
    embedded directly inside HTML/markdown strings (e.g. <img src="...">
    inside a badge) — st.image() alone can't be placed inline next to text.
    Returns "" if the logo file isn't found, so callers can fall back cleanly.
    """
    import base64
    if not os.path.exists("logo.png"):
        return ""
    with open("logo.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _show_friendly_error(e: Exception):
    """Smart error handler — shows user-friendly message based on error type."""
    msg = str(e).lower()
    if any(x in msg for x in ("429", "resource_exhausted", "quota", "rate limit", "ratelimit", "throttl")):
        st.error("⏳ **API Rate Limit Reached**")
        st.warning("Your AI API quota has been temporarily exceeded. Please wait 1–2 minutes and try again.")
    elif any(x in msg for x in ("api key", "authentication", "unauthorized", "invalid key", "api_key")):
        st.error("🔑 **Invalid API Key**")
        st.warning("Your API key appears to be invalid or missing. Check your `.env` file.")
    elif any(x in msg for x in ("scanned", "image-based", "ocr", "no text")):
        st.error("🖼️ **Scanned PDF Detected**")
        st.warning("Your PDF appears to be a scanned image. Please use a text-based PDF or DOCX instead.")
    elif any(x in msg for x in ("corrupted", "password", "not a valid", "cannot read")):
        st.error("📄 **File Could Not Be Read**")
        st.warning("Your resume file appears to be corrupted or password-protected. Try re-uploading.")
    elif any(x in msg for x in ("too large", "file size")):
        st.error("📦 **File Too Large**")
        st.warning(f"Maximum file size is {MAX_RESUME_SIZE_MB} MB. Please compress your resume.")
    elif any(x in msg for x in ("connection", "timeout", "network", "ssl", "socket")):
        st.error("🌐 **Network Error**")
        st.warning("Could not connect to the AI service. Check your internet connection and try again.")
    elif any(x in msg for x in ("busy", "too many", "concurrent")):
        st.error("⏳ **Server Busy**")
        st.warning("Too many resumes are being processed right now. Please try again in a minute.")
    elif any(x in msg for x in ("json", "parse", "decode")):
        st.error("⚠️ **AI Response Error**")
        st.warning("The AI returned an unexpected response. This usually fixes itself — please try again.")
    else:
        st.error(f"❌ **Something went wrong:** {e}")
        st.info("Check your `.env` — `LLM_PROVIDER`, API key, and model name.")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── RESPONSIVE: Mobile first ── */
/* Make Streamlit main block full width on mobile */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}
@media (min-width: 768px) {
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
}

/* ── Split hero (Enhancv-style layout — left: pitch, right: live preview) ──
   Same RoastCV palette as before (#0f172a / #1a2e4a / #2563eb / #60a5fa),
   no new colors introduced. */
.hero-split {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.75rem;
    align-items: center;
    margin-bottom: 1.5rem;
}
@media (min-width: 900px) {
    .hero-split {
        grid-template-columns: 1.05fr 0.95fr;
        gap: 2.5rem;
        margin-bottom: 2rem;
    }
}

.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60a5fa;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    margin-bottom: 0.9rem;
}

.hero-headline {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.18;
    letter-spacing: -0.5px;
    margin: 0 0 0.85rem 0;
}
@media (min-width: 768px) {
    .hero-headline { font-size: 2.6rem; }
}
.hero-headline .accent { color: #60a5fa; }

.hero-desc {
    font-size: 0.92rem;
    color: rgba(248,250,252,0.65);
    line-height: 1.6;
    margin: 0 0 1.2rem 0;
    max-width: 480px;
}
@media (min-width: 768px) {
    .hero-desc { font-size: 1.02rem; }
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem 0.35rem 0.2rem 0;
}
@media (min-width: 768px) {
    .hero-badge { padding: 0.3rem 1rem; font-size: 0.8rem; }
}

.hero-cta-hint {
    font-size: 0.8rem;
    color: rgba(248,250,252,0.45);
    margin-top: 0.9rem;
    font-style: italic;
}

/* ── Page sections (Enhancv-style full page layout, RoastCV colors) ──
   Generic wrapper used for every section below the hero: consistent
   max-width, vertical rhythm and heading sizes matching Enhancv's own
   spacing scale, but built entirely with the existing dark navy/blue
   palette — no new colors. */
.page-section {
    margin: 3.5rem 0;
}
@media (min-width: 768px) {
    .page-section { margin: 5rem 0; }
}
.section-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 0.6rem;
    text-align: center;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    text-align: center;
    line-height: 1.25;
    margin: 0 auto 0.75rem auto;
    max-width: 700px;
}
@media (min-width: 768px) {
    .section-title { font-size: 2.1rem; }
}
.section-sub {
    font-size: 0.92rem;
    color: rgba(248,250,252,0.6);
    text-align: center;
    line-height: 1.65;
    max-width: 680px;
    margin: 0 auto 2.5rem auto;
}
@media (min-width: 768px) {
    .section-sub { font-size: 1rem; }
}

/* ── "How we score" numbered steps (mirrors Enhancv's two-tier explainer) ── */
.steps-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.25rem;
}
@media (min-width: 768px) {
    .steps-grid { grid-template-columns: 1fr 1fr; gap: 1.75rem; }
}
.step-card {
    background: linear-gradient(160deg, #1a2e4a 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 16px;
    padding: 1.5rem;
}
.step-card-num {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(37,99,235,0.18);
    border: 1px solid rgba(96,165,250,0.4);
    color: #60a5fa;
    font-weight: 700;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.9rem;
}
.step-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.5rem;
}
.step-card-desc {
    font-size: 0.86rem;
    color: rgba(248,250,252,0.6);
    line-height: 1.65;
}

/* ── 27 checks grid (mirrors Enhancv's 7-category checklist) ── */
.checks-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
}
@media (min-width: 640px) {
    .checks-grid { grid-template-columns: 1fr 1fr; }
}
@media (min-width: 1000px) {
    .checks-grid { grid-template-columns: repeat(3, 1fr); }
}
.checks-category {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
}
.checks-category-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #60a5fa;
    margin-bottom: 0.7rem;
}
.checks-category-item {
    font-size: 0.84rem;
    color: rgba(248,250,252,0.78);
    padding: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.checks-category-item::before {
    content: "✓";
    color: #4ade80;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Tools showcase (mirrors "Put your score to work" cards) ── */
.tools-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.25rem;
}
@media (min-width: 768px) {
    .tools-grid { grid-template-columns: 1fr 1fr; gap: 1.75rem; }
}
.tool-card {
    background: linear-gradient(160deg, #1a2e4a 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 16px;
    padding: 1.5rem;
}
.tool-card-icon { font-size: 1.6rem; margin-bottom: 0.7rem; }
.tool-card-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.5rem;
}
.tool-card-desc {
    font-size: 0.85rem;
    color: rgba(248,250,252,0.6);
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.tool-card-bullet {
    font-size: 0.8rem;
    color: rgba(248,250,252,0.55);
    padding: 0.2rem 0;
    padding-left: 1rem;
    position: relative;
}
.tool-card-bullet::before {
    content: "—";
    position: absolute;
    left: 0;
    color: #60a5fa;
}

/* ── FAQ accordion (mirrors Enhancv's FAQ list) ── */
.faq-wrap { max-width: 760px; margin: 0 auto; }

/* ── Final CTA band before the upload form ── */
.cta-band {
    background: linear-gradient(135deg, #1a2e4a, #2563eb);
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: white;
    margin: 3rem 0 2rem 0;
}
.cta-band-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
@media (min-width: 768px) {
    .cta-band-title { font-size: 1.6rem; }
}
.cta-band-sub {
    font-size: 0.88rem;
    opacity: 0.85;
    max-width: 480px;
    margin: 0 auto;
}

/* Preview card — the "what you'll get" panel on the right, mirrors the
   live-preview pattern but uses RoastCV's own dark/blue gradient instead
   of swapping to a light theme. */
.preview-card {
    background: linear-gradient(160deg, #1a2e4a 0%, #0f172a 100%);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    color: white;
}
.preview-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(255,255,255,0.5);
    margin-bottom: 1rem;
}
.gauge-row {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin-bottom: 1.1rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.gauge-caption-title { font-weight: 600; font-size: 0.95rem; }
.gauge-caption-desc { font-size: 0.76rem; color: rgba(255,255,255,0.5); margin-top: 0.15rem; }

.check-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 0.85rem;
}
.check-row:last-child { border-bottom: none; }
.check-left { display: flex; align-items: center; gap: 0.55rem; color: rgba(255,255,255,0.9); }
.check-icon { color: #4ade80; font-weight: 700; }
.check-tag {
    font-size: 0.66rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background: rgba(96,165,250,0.15);
    color: #93c5fd;
    white-space: nowrap;
}

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}
.score-green  { background: #dcfce7; color: #15803d; }
.score-yellow { background: #fef9c3; color: #a16207; }
.score-red    { background: #fee2e2; color: #dc2626; }

/* Ad container styling */
.ad-container {
    background: #f1f5f9;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin: 1rem 0;
}

/* Progress tip box */
.tip-box {
    background: linear-gradient(90deg, #eff6ff, #f0fdf4);
    border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #1e3a5f;
}

/* Section heading */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

/* Overall score card */
.overall-score-card {
    background: linear-gradient(135deg, #1a2e4a, #2563eb);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}
@media (min-width: 768px) {
    .overall-score-card { padding: 1.5rem; }
}
.overall-number { font-size: 2.8rem; font-weight: 700; line-height: 1; }
@media (min-width: 768px) {
    .overall-number { font-size: 3.5rem; }
}
.overall-label  { font-size: 0.85rem; opacity: 0.85; margin-top: 0.25rem; }

/* Recommendation banner */
.rec-banner {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    margin: 1rem 0;
}
.rec-strong  { background: #dcfce7; color: #15803d; border-left: 5px solid #15803d; }
.rec-moderate{ background: #fef9c3; color: #92400e; border-left: 5px solid #d97706; }
.rec-weak    { background: #fee2e2; color: #dc2626; border-left: 5px solid #dc2626; }

/* Step progress indicator */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    font-size: 0.88rem;
}
.step-dot-done    { width:10px; height:10px; border-radius:50%; background:#22c55e; flex-shrink:0; }
.step-dot-active  { width:10px; height:10px; border-radius:50%; background:#2563eb; flex-shrink:0; animation: pulse 1s infinite; }
.step-dot-pending { width:10px; height:10px; border-radius:50%; background:#e2e8f0; flex-shrink:0; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Hide streamlit default header */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}

/* ── COLORFUL FOOTER ── */
.footer-wrapper {
    background: linear-gradient(160deg, #0f172a 0%, #1a2e4a 50%, #1e3a8a 100%);
    border-radius: 20px 20px 0 0;
    margin-top: 3rem;
    padding: 0 0 0 0;
    overflow: hidden;
}

/* Stats bar */
.footer-stats {
    background: linear-gradient(90deg, #2563eb, #7c3aed, #0891b2);
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0;
}
@media (min-width: 768px) {
    .footer-stats { grid-template-columns: repeat(4, 1fr); }
}
.footer-stat {
    text-align: center;
    padding: 1.2rem 0.5rem;
    border-right: 1px solid rgba(255,255,255,0.15);
}
.footer-stat:last-child { border-right: none; }
.footer-stat-number {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    line-height: 1;
}
.footer-stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.75);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Footer columns */
.footer-cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    padding: 2rem 1.5rem;
}
@media (min-width: 768px) {
    .footer-cols {
        grid-template-columns: 2fr 1fr 1fr 1fr;
        padding: 2.5rem 3rem;
        gap: 2rem;
    }
}

.footer-brand {
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.6rem;
}
.footer-brand-desc {
    color: rgba(255,255,255,0.6);
    font-size: 0.82rem;
    line-height: 1.6;
}
.footer-col-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #60a5fa;
    margin-bottom: 0.75rem;
}
.footer-link {
    color: rgba(255,255,255,0.7);
    font-size: 0.82rem;
    padding: 0.2rem 0;
    line-height: 1.8;
}

/* Bottom bar */
.footer-bottom {
    border-top: 1px solid rgba(255,255,255,0.1);
    padding: 1rem 1.5rem;
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.78rem;
}
@media (min-width: 768px) {
    .footer-bottom { padding: 1rem 3rem; }
}

/* ── RESPONSIVE: Hide columns on mobile, stack them ── */
@media (max-width: 640px) {
    /* Stack score grid to 2 columns on mobile */
    div[data-testid="column"] {
        min-width: 45% !important;
    }
    .rec-banner {
        font-size: 0.88rem;
        padding: 0.8rem 1rem;
    }
    .tip-box { font-size: 0.82rem; }
    .ad-container { font-size: 0.72rem; padding: 0.75rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("running", False),
    ("report", None),
    ("report_company_name", ""),
    ("current_step", 0),
    ("current_label", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper: Score color ───────────────────────────────────────────────────────
def score_color(val):
    if not isinstance(val, (int, float)):
        return "score-yellow"
    if val >= 70:
        return "score-green"
    if val >= 45:
        return "score-yellow"
    return "score-red"


def score_badge(val, suffix=""):
    if isinstance(val, (int, float)):
        cls = score_color(val)
        return f'<span class="score-badge {cls}">{val}{suffix}</span>'
    return f'<span class="score-badge score-yellow">N/A</span>'


def rec_class(score):
    if isinstance(score, (int, float)):
        if score >= 70:
            return "rec-strong"
        if score >= 45:
            return "rec-moderate"
    return "rec-weak"


# ── AD PLACEHOLDER (replace with real AdSense code) ──────────────────────────
# Toggle: set to True later if you add real Google AdSense code to show_ad().
# Left as a single flag (rather than deleting every show_ad() call) so ads can
# be turned back on in one place without re-touching every call site.
ADS_ENABLED = False


def show_ad(label="Advertisement"):
    if not ADS_ENABLED:
        return
    st.markdown(f"""
    <div class="ad-container">
        📢 {label}<br>
        <small>[ Google AdSense — Replace this div with your ad unit code ]</small>
    </div>
    """, unsafe_allow_html=True)


# ── LANDING PAGE (shown before analysis) ─────────────────────────────────────
def show_landing():
    # Logo + split hero: pitch on the left, "what you'll get" preview on the right.
    # Layout pattern inspired by resume-checker landing pages — pitch + live
    # preview side by side — but built entirely with RoastCV's own copy,
    # own 13-agent feature set, and the existing dark navy/blue palette.
    col_logo, col_hero = st.columns([1, 5], gap="small")
    with col_logo:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=110)
        else:
            st.markdown("🔥", unsafe_allow_html=False)
    with col_hero:
        _logo_uri = _logo_data_uri()
        _badge_icon = (
            f'<img src="{_logo_uri}" alt="logo" style="height:14px;width:14px;'
            f'vertical-align:-2px;border-radius:3px;margin-right:2px;">'
            if _logo_uri else "🔥"
        )
        st.markdown(f"""
        <div class="hero-split">
            <div class="hero-left">
                <div class="hero-eyebrow">{_badge_icon} Free · AI-Powered · 13 Agents</div>
                <div class="hero-headline">Is your resume <span class="accent">actually</span> working?</div>
                <div class="hero-desc">
                    13 AI agents — recruiter, hiring manager, ATS bot, HR reviewer and more —
                    tear into your resume and tell you exactly what's wrong, in plain language.
                    No sugar-coating.
                </div>
                <div class="hero-badges-row">
                    <span class="hero-badge">✅ Free</span>
                    <span class="hero-badge">{_badge_icon} 13 AI Agents</span>
                    <span class="hero-badge">📄 PDF & DOCX Output</span>
                    <span class="hero-badge">🔒 Private & Secure</span>
                </div>
                <div class="hero-cta-hint">↓ Upload your resume below to get started</div>
            </div>
            <div class="hero-right">
                <div class="preview-card">
                    <div class="preview-label">What You'll Get</div>
                    <div class="gauge-row">
                        <svg width="80" height="80" viewBox="0 0 120 120" style="flex-shrink:0;">
                            <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="10"/>
                            <circle cx="60" cy="60" r="52" fill="none" stroke="#4ade80" stroke-width="10"
                                    stroke-dasharray="326.7" stroke-dashoffset="19.6"
                                    stroke-linecap="round" transform="rotate(-90 60 60)"/>
                            <text x="60" y="57" text-anchor="middle" font-size="30" font-weight="700" fill="#f8fafc">94</text>
                            <text x="60" y="78" text-anchor="middle" font-size="12" fill="rgba(255,255,255,0.5)">/100</text>
                        </svg>
                        <div>
                            <div class="gauge-caption-title">Overall Resume Score</div>
                            <div class="gauge-caption-desc">Calculated from 7 AI scoring agents</div>
                        </div>
                    </div>
                    <div class="check-row">
                        <div class="check-left"><span class="check-icon">✓</span> ATS Score</div>
                        <span class="check-tag">Estimate</span>
                    </div>
                    <div class="check-row">
                        <div class="check-left"><span class="check-icon">✓</span> Gap Analysis</div>
                        <span class="check-tag">vs JD</span>
                    </div>
                    <div class="check-row">
                        <div class="check-left"><span class="check-icon">✓</span> AI Rewrite</div>
                        <span class="check-tag">Human Tone</span>
                    </div>
                    <div class="check-row">
                        <div class="check-left"><span class="check-icon">✓</span> Cover Letter</div>
                        <span class="check-tag">Personalized</span>
                    </div>
                    <div class="check-row">
                        <div class="check-left"><span class="check-icon">✓</span> Interview Prep</div>
                        <span class="check-tag">Custom Qs</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ad slot — top of page (before form)
    show_ad("Sponsored")

    st.markdown("---")
    st.markdown("### 📂 Upload Your Resume & Job Description")


# ── MORE INFO (shown BELOW the upload form, only before a report exists) ──────
def show_more_info():
    # ── SECTION: How RoastCV scores your resume (two-tier explainer) ──────
    st.markdown("""
    <div class="page-section">
        <div class="section-kicker">How It Works</div>
        <div class="section-title">RoastCV's score comes from two layers, not a guess</div>
        <div class="section-sub">
            We don't show a single made-up number. Your score is built from how well your
            resume can be parsed and understood, plus how strong the actual content is —
            judged the way a recruiter and an ATS each would.
        </div>
        <div class="steps-grid">
            <div class="step-card">
                <div class="step-card-num">1</div>
                <div class="step-card-title">The proportion of content we can interpret</div>
                <div class="step-card-desc">
                    Like a real ATS, our agents parse your resume's structure, sections, and
                    contact details — checking against signals we know from how systems such
                    as Workday, Greenhouse, and Taleo actually behave. If our agents can
                    clearly understand your skills, experience, and sections, a company's ATS
                    likely can too.
                </div>
            </div>
            <div class="step-card">
                <div class="step-card-num">2</div>
                <div class="step-card-title">What a recruiter would actually think</div>
                <div class="step-card-desc">
                    An ATS doesn't catch weak bullet points or vague claims — but our HR
                    Roast, Recruiter Review, and Hiring Manager agents do. They look for
                    quantifiable achievements, clear impact, and whether your resume would
                    survive a brutally honest human read.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION: Full checks grid (what the 13 agents actually check) ─────
    st.markdown("""
    <div class="page-section">
        <div class="section-kicker">27+ Checks · 13 Agents</div>
        <div class="section-title">Every part of your resume, checked from a different angle</div>
        <div class="section-sub">
            Each category below is handled by one or more of our 13 AI agents — no single
            model trying to do everything at once.
        </div>
        <div class="checks-grid">
            <div class="checks-category">
                <div class="checks-category-title">ATS Essentials</div>
                <div class="checks-category-item">Keyword optimization</div>
                <div class="checks-category-item">Section hierarchy</div>
                <div class="checks-category-item">ATS-safe formatting</div>
                <div class="checks-category-item">Readability & structure</div>
            </div>
            <div class="checks-category">
                <div class="checks-category-title">Resume Quality</div>
                <div class="checks-category-item">Missing sections</div>
                <div class="checks-category-item">Weak summary detection</div>
                <div class="checks-category-item">Grammar & repetition</div>
                <div class="checks-category-item">Low-impact bullet points</div>
            </div>
            <div class="checks-category">
                <div class="checks-category-title">Job Tailoring</div>
                <div class="checks-category-item">Skill gap analysis</div>
                <div class="checks-category-item">Missing keywords vs JD</div>
                <div class="checks-category-item">Strong & weak matches</div>
                <div class="checks-category-item">Match percentage score</div>
            </div>
            <div class="checks-category">
                <div class="checks-category-title">Recruiter Red Flags</div>
                <div class="checks-category-item">Brutally honest HR roast</div>
                <div class="checks-category-item">Shortlist likelihood</div>
                <div class="checks-category-item">Critical issues called out</div>
                <div class="checks-category-item">Interview probability</div>
            </div>
            <div class="checks-category">
                <div class="checks-category-title">Hiring Manager Lens</div>
                <div class="checks-category-item">Team fit assessment</div>
                <div class="checks-category-item">Practical skill evaluation</div>
                <div class="checks-category-item">Project relevance</div>
                <div class="checks-category-item">Hiring recommendation</div>
            </div>
            <div class="checks-category">
                <div class="checks-category-title">AI Rewrite & Tone</div>
                <div class="checks-category-item">Human-sounding rewrite</div>
                <div class="checks-category-item">AI buzzword removal</div>
                <div class="checks-category-item">AI-detection risk score</div>
                <div class="checks-category-item">Action-verb bullet points</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION: Put your score to work (tools showcase) ──────────────────
    st.markdown("""
    <div class="page-section">
        <div class="section-kicker">Put Your Score To Work</div>
        <div class="section-title">Checking is step one — RoastCV covers the rest</div>
        <div class="section-sub">
            The same upload also gets you a rewritten resume, a tailored cover letter, and
            interview prep — generated from your actual experience, nothing invented.
        </div>
        <div class="tools-grid">
            <div class="tool-card">
                <div class="tool-card-icon">📝</div>
                <div class="tool-card-title">AI Resume Rewrite</div>
                <div class="tool-card-desc">
                    Stronger summary, sharper bullets, ATS-safe section headers — built only
                    from facts already in your resume.
                </div>
                <div class="tool-card-bullet">Strict no-fake-experience rule</div>
                <div class="tool-card-bullet">Keeps your real URLs and contact info intact</div>
            </div>
            <div class="tool-card">
                <div class="tool-card-icon">🧑‍💼</div>
                <div class="tool-card-title">Humanizer</div>
                <div class="tool-card-desc">
                    Strips AI-sounding buzzwords like "results-driven professional" so your
                    resume reads like a real person wrote it.
                </div>
                <div class="tool-card-bullet">Removes emojis ATS can't parse</div>
                <div class="tool-card-bullet">Comes with an AI-detection risk estimate</div>
            </div>
            <div class="tool-card">
                <div class="tool-card-icon">💌</div>
                <div class="tool-card-title">Cover Letter Generator</div>
                <div class="tool-card-desc">
                    A personalized, natural-sounding cover letter built from your resume and
                    the job description — under 300 words, no generic filler.
                </div>
                <div class="tool-card-bullet">Matches your resume's actual tone</div>
                <div class="tool-card-bullet">Download as PDF or DOCX</div>
            </div>
            <div class="tool-card">
                <div class="tool-card-icon">🎯</div>
                <div class="tool-card-title">Interview Prep</div>
                <div class="tool-card-desc">
                    Candidate-specific questions tied to your actual projects — HR,
                    technical, project deep-dives, and scenario-based.
                </div>
                <div class="tool-card-bullet">4-6 questions per category</div>
                <div class="tool-card-bullet">Built from your real experience, not generic banks</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION: FAQ ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-section">
        <div class="section-kicker">FAQ</div>
        <div class="section-title">Frequently asked questions</div>
    </div>
    """, unsafe_allow_html=True)

    _faq_items = [
        ("What is a resume checker?",
         "A resume checker evaluates your resume's formatting, keyword usage, grammar, "
         "and content relevance. RoastCV goes further with 13 specialized AI agents that "
         "each look at a different angle — ATS compatibility, recruiter perspective, "
         "hiring manager fit, and more — instead of one generic check."),
        ("How is the overall score calculated?",
         "Your overall score is the average of the numeric scores from each of our 13 "
         "agents — resume health, ATS compatibility, job-description match, HR review, "
         "recruiter review, hiring manager review, and humanization. If any agent fails "
         "to respond, it's excluded from the average rather than counted as zero."),
        ("Is the ATS score a guarantee?",
         "No. Real ATS platforms like Workday, Taleo, and Greenhouse use proprietary "
         "algorithms that aren't publicly documented. Our ATS score is a heuristic "
         "estimate based on keyword optimization, structure, and formatting best "
         "practices — not a guarantee of what any specific ATS will produce."),
        ("Is my resume data stored anywhere?",
         "No. When you use RoastCV through this app, nothing is saved to disk — your "
         "resume and report are held only in memory for your session and disappear once "
         "you close the tab."),
        ("What file formats are supported?",
         "PDF and DOCX, up to 5 MB. Your file must be text-based — scanned image PDFs "
         "can't be read without OCR, so a text-based PDF or DOCX will give you the most "
         "accurate results."),
        ("Will the rewritten resume include made-up experience?",
         "No. Our Resume Rewrite agent has a strict rule against inventing experience, "
         "projects, or metrics. It only rephrases and strengthens what's already in your "
         "resume — if you have no formal work experience, it presents your strongest "
         "projects as experience instead, honestly labeled."),
    ]
    for _q, _a in _faq_items:
        with st.expander(f"**{_q}**"):
            st.write(_a)


# ── PROGRESS TIPS (shown during analysis to keep users engaged) ───────────────
AGENT_TIPS = {
    0:  "📖 Reading your resume structure and content...",
    1:  "💡 Tip: Quantify your achievements (e.g., 'Increased sales by 30%') to stand out.",
    2:  "🔍 Checking resume health — missing sections, weak bullets, formatting issues...",
    3:  "🤖 ATS systems scan for keywords. Most resumes get rejected before a human sees them!",
    4:  "📋 Analyzing the job description for required vs preferred skills...",
    5:  "📊 Matching your skills to the job requirements — finding gaps...",
    6:  "💼 HR reviewers spend just 6 seconds on a resume. First impression matters!",
    7:  "👔 Recruiters compare you against market standards. Let's see where you stand...",
    8:  "✍️ Rewriting your resume with stronger language and better structure...",
    9:  "🧠 Removing AI-sounding phrases to make your resume sound genuinely human...",
    10: "🏢 Evaluating team fit and practical skill match for the hiring manager...",
    11: "🎯 Generating interview questions specific to YOUR projects and experience...",
    12: "💌 Writing a personalized cover letter for this role...",
    13: "📊 Calculating your final score and recommendation...",
}

AGENT_STEPS = [
    "Parsing Resume",
    "Resume Reader",
    "Resume Audit",
    "ATS Review",
    "JD Analyzer",
    "Gap Analysis",
    "HR Roast",
    "Recruiter Review",
    "Resume Rewrite",
    "Humanizer",
    "Hiring Manager",
    "Interview Coach",
    "Cover Letter",
    "Final Decision",
]


# ── MAIN INPUT FORM ───────────────────────────────────────────────────────────
show_landing()

# ── Input Form: hamesha dikhta hai ──────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    resume_file = st.file_uploader(
        "📎 Resume upload karein (PDF ya DOCX) — Max 5 MB",
        type=["pdf", "docx"],
        disabled=st.session_state.running,
        help="✅ Sirf 5 MB tak. Text-based PDF ya DOCX chahiye — scanned image PDF kaam nahi karega.",
    )

with col_right:
    jd_text = st.text_area(
        "📋 Paste Job Description",
        height=200,
        placeholder="Paste the full job description here — the more complete, the better the analysis...",
        disabled=st.session_state.running,
    )

company_name = st.text_input(
    "🏢 Company Name (optional)",
    placeholder="e.g. Google, Infosys, Zomato — used in your cover letter",
    disabled=st.session_state.running,
)

btn_col, reset_col = st.columns([4, 1])
run_clicked = btn_col.button(
    "🚀 Analyze My Resume — Free",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.running,
)

if st.session_state.report is not None:
    if reset_col.button("🔄 New", use_container_width=True, disabled=st.session_state.running):
        st.session_state.report = None
        st.session_state.report_company_name = ""
        st.session_state.current_step = 0
        st.rerun()

# ── Analysis placeholder: button ke JUST NEECHE, hamesha yahi ────────────────
# Jab analysis chal rahi ho to progress yahan dikhegi — form upar wahi rahega.
# Jab kuch nahi chal raha to ye empty rahega (koi space nahi leta).
analysis_placeholder = st.empty()

# Extra landing-page info — sirf tab jab koi report nahi aur analysis nahi chal rahi
if st.session_state.report is None and not st.session_state.running:
    show_more_info()


# ── RUN PIPELINE ──────────────────────────────────────────────────────────────

if run_clicked:
    # ── Validation: pehle sab check karo, kuch bhi missing ho toh rok do ──
    if not resume_file and not jd_text.strip():
        st.error("⬆️ Pehle resume upload karein aur job description paste karein.")
    elif not resume_file:
        st.error("⬆️ Pehle apna resume upload karein (PDF ya DOCX).")
    elif resume_file.size > MAX_RESUME_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File bahut badi hai. Maximum size {MAX_RESUME_SIZE_MB} MB hai.")
    elif not jd_text.strip():
        st.error("📋 Job description paste karein.")
    elif len(jd_text.strip()) < 50:
        st.error("📋 Job description bahut chhoti hai. Poori JD paste karein.")
    else:
        st.session_state.running = True
        tmp_path = None

        try:
            suffix = os.path.splitext(resume_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name

            # Saari progress UI ek hi container ke andar — page scroll nahi hoga
            with analysis_placeholder.container():
                st.markdown("---")
                st.markdown("### ⚙️ Analysis in Progress")
                st.caption("Sit tight — 13 AI agents are reviewing your resume. This takes 1–3 minutes.")

                # Progress bar
                progress_bar = st.progress(0)

                # Step indicators container
                step_container = st.empty()
                tip_container  = st.empty()

                completed_steps = []

                def update_progress(step, total, label):
                    progress_bar.progress(step / total)
                    completed_steps.append(label)
                    st.session_state.current_step = step

                    # Render all steps with status
                    steps_html = ""
                    for i, s in enumerate(AGENT_STEPS):
                        if i < step:
                            steps_html += f'<div class="step-indicator"><div class="step-dot-done"></div> <span style="color:#15803d;text-decoration:line-through;opacity:0.6">{s}</span></div>'
                        elif i == step:
                            steps_html += f'<div class="step-indicator"><div class="step-dot-active"></div> <span style="color:#2563eb;font-weight:600">{s}</span></div>'
                        else:
                            steps_html += f'<div class="step-indicator"><div class="step-dot-pending"></div> <span style="color:#94a3b8">{s}</span></div>'

                    step_container.markdown(steps_html, unsafe_allow_html=True)

                    # Tip box
                    tip = AGENT_TIPS.get(step, "")
                    if tip:
                        tip_container.markdown(f'<div class="tip-box">💡 {tip}</div>', unsafe_allow_html=True)

                report = run_pipeline(
                    tmp_path,
                    jd_text=jd_text,
                    company_name=company_name,
                    progress_callback=update_progress,
                    save_to_disk=False,
                )

                progress_bar.progress(1.0)
                tip_container.empty()
                step_container.empty()
                st.success("✅ Analysis complete! Results neeche dekh sakte hain.")

            st.session_state.report = report
            st.session_state.report_company_name = company_name
            st.session_state.running = False
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Analysis done — ab rerun karo taaki form hide ho aur results dikhen
            st.rerun()

        except Exception as e:
            _show_friendly_error(e)
            st.session_state.running = False
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.report is not None:
    report      = st.session_state.report
    company_name = st.session_state.report_company_name
    final       = report["final_decision"]
    scores      = final["scores"]
    meta        = final.get("meta", {})
    failed      = final.get("failed_agents", [])
    agent_errors = report.get("_agent_errors", {})
    overall     = final["overall_score"]

    # ── Ad — top of results (high visibility) ────────────────
    show_ad("Sponsored — Resume Writing Services")

    # ── Agent errors warning ──────────────────────────────────
    if agent_errors:
        with st.expander(f"⚠️ {len(agent_errors)} agent(s) had issues — click to see details"):
            for agent_name, error_msg in agent_errors.items():
                st.error(f"**{agent_name}:** {error_msg}")
            st.info("Check your `.env` — `LLM_PROVIDER`, API key, and model name.")
    else:
        st.success("✅ All 13 agents completed successfully!")

    st.markdown("---")

    # ── Overall Score Card + Recommendation ──────────────────
    st.markdown("## 📊 Your Resume Score")

    ov_col, rec_col = st.columns([1, 2], gap="large")

    with ov_col:
        st.markdown(f"""
        <div class="overall-score-card">
            <div class="overall-number">{overall}</div>
            <div class="overall-label">Overall Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with rec_col:
        rec_text = final["final_recommendation"]
        css_cls  = rec_class(overall)
        st.markdown(f'<div class="rec-banner {css_cls}">📋 {rec_text}</div>', unsafe_allow_html=True)

        prob  = meta.get("interview_probability", "-")
        short = meta.get("shortlist_decision", "-")
        hire  = meta.get("hiring_recommendation", "-")

        m1, m2 = st.columns(2)
        m1.metric("📞 Interview Probability", prob)
        m2.metric("✅ Shortlist Decision", short)
        if hire:
            st.caption(f"🏢 Hiring Manager: {hire}")

    st.markdown("---")

    # ── Score Grid ────────────────────────────────────────────
    st.markdown("### 🎯 Detailed Scores")
    score_items = [
        ("Resume Health",   scores.get("resume_health_score"),   "/100"),
        ("ATS Score",       scores.get("ats_score"),             "/100"),
        ("JD Match",        scores.get("jd_match_score"),        "%"),
        ("HR Score",        scores.get("hr_score"),              "/100"),
        ("Recruiter Score", scores.get("recruiter_score"),       "/100"),
        ("Hiring Manager",  scores.get("hiring_manager_score"),  "/100"),
        ("Humanization",    scores.get("humanization_score"),    "/100"),
    ]

    cols = st.columns(4)
    for i, (label, val, suffix) in enumerate(score_items):
        with cols[i % 4]:
            display = f"{val}{suffix}" if isinstance(val, (int, float)) else "N/A"
            st.metric(label, display)

    # ── Download Report ───────────────────────────────────────
    st.markdown("---")
    report_content = report.get("_report_content", {})
    dl1, dl2 = st.columns(2)
    if report_content.get("md"):
        dl1.download_button(
            "📄 Download Full Report (Markdown)",
            report_content["md"],
            file_name="resume_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_md_report",
        )
    if report_content.get("json"):
        dl2.download_button(
            "📦 Download Full Report (JSON)",
            report_content["json"],
            file_name="resume_report.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_report",
        )

    # ── Ad — between scores and tabs ─────────────────────────
    show_ad("Sponsored — Interview Coaching & Career Services")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "🔍 Audit",
        "🤖 ATS",
        "📋 Gap Analysis",
        "🔥 HR Roast",
        "👔 Recruiter",
        "🏢 Hiring Manager",
        "✨ Improved Resume",
        "📝 Cover Letter",
        "🎯 Interview Prep",
    ])

    # ── Tab 0: Audit ──────────────────────────────────────────
    with tabs[0]:
        audit = report.get("audit", {})
        if not audit:
            st.warning("Resume Audit agent failed.")
        else:
            health = audit.get("health_score", "N/A")
            st.markdown(f"**Resume Health Score:** {score_badge(health, '/100')}", unsafe_allow_html=True)
            st.markdown("")

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("#### ✅ Strengths")
                for s in audit.get("strengths", []):
                    st.success(s)
            with col_w:
                st.markdown("#### ❌ Weaknesses")
                for w in audit.get("weaknesses", []):
                    st.error(w)

            st.markdown("#### 💡 Improvement Suggestions")
            for s in audit.get("improvement_suggestions", []):
                st.info(f"→ {s}")

    # ── Tab 1: ATS ────────────────────────────────────────────
    with tabs[1]:
        ats = report.get("ats_review", {})
        if not ats:
            st.warning("ATS Review agent failed.")
        else:
            ats_score = ats.get("ats_score", "N/A")
            st.markdown(f"**ATS Compatibility Score:** {score_badge(ats_score, '/100')}", unsafe_allow_html=True)
            st.caption(f"⚠️ {ats.get('disclaimer', 'This is an estimated score. Actual ATS results may vary.')}")
            st.markdown("")

            col_i, col_s = st.columns(2)
            with col_i:
                st.markdown("#### 🚨 ATS Issues")
                for i in ats.get("ats_issues", []):
                    st.error(i)
            with col_s:
                st.markdown("#### 💡 ATS Suggestions")
                for s in ats.get("ats_improvement_suggestions", []):
                    st.info(f"→ {s}")

    # ── Tab 2: Gap Analysis ───────────────────────────────────
    with tabs[2]:
        gap = report.get("gap_analysis", {})
        if not gap:
            st.warning("Gap Analysis agent failed.")
        else:
            match_pct = gap.get("match_percentage", "N/A")
            st.markdown(f"**JD Match:** {score_badge(match_pct, '%')}", unsafe_allow_html=True)
            st.markdown("")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### ✅ Strong Matches")
                for s in gap.get("strong_matches", []):
                    st.success(s)
            with c2:
                st.markdown("#### ❌ Missing Skills")
                for s in gap.get("missing_skills", []):
                    st.error(s)
                st.markdown("#### 🔑 Missing Keywords")
                for k in gap.get("missing_keywords", []):
                    st.warning(f"`{k}`")
            with c3:
                st.markdown("#### 💡 Recommendations")
                for r in gap.get("recommendations", []):
                    st.info(f"→ {r}")

    # ── Tab 3: HR Roast ───────────────────────────────────────
    with tabs[3]:
        hr = report.get("hr_roast", {})
        if not hr:
            st.warning("HR Roast agent failed.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**HR Score:** {score_badge(hr.get('hr_score'), '/100')}", unsafe_allow_html=True)
            shortlist = hr.get("shortlist_decision", "N/A")
            color = "✅" if shortlist == "Yes" else ("⚠️" if shortlist == "Maybe" else "❌")
            c2.markdown(f"**Shortlist Decision:** {color} **{shortlist}**")
            st.markdown("")

            st.markdown("#### 💬 HR Feedback")
            st.write(hr.get("hr_feedback", ""))

            if hr.get("critical_issues"):
                st.markdown("#### 🚨 Critical Issues")
                for i in hr.get("critical_issues", []):
                    st.error(f"→ {i}")

    # ── Tab 4: Recruiter ──────────────────────────────────────
    with tabs[4]:
        rec = report.get("recruiter_review", {})
        if not rec:
            st.warning("Recruiter Review agent failed.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Recruiter Score:** {score_badge(rec.get('recruiter_score'), '/100')}", unsafe_allow_html=True)
            prob = rec.get("interview_probability", "N/A")
            prob_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(prob, "⚪")
            c2.markdown(f"**Interview Probability:** {prob_icon} **{prob}**")
            st.markdown("")
            st.markdown("#### 💬 Recruiter Feedback")
            st.write(rec.get("recruiter_feedback", ""))

    # ── Tab 5: Hiring Manager ─────────────────────────────────
    with tabs[5]:
        hm = report.get("hiring_manager", {})
        if not hm:
            st.warning("Hiring Manager agent failed.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Hiring Manager Score:** {score_badge(hm.get('hiring_manager_score'), '/100')}", unsafe_allow_html=True)
            c2.info(f"**Recommendation:** {hm.get('hiring_recommendation', 'N/A')}")
            st.markdown("")

            if hm.get("team_fit_notes"):
                st.markdown("#### 👥 Team Fit Notes")
                st.write(hm["team_fit_notes"])
            if hm.get("practical_skill_assessment"):
                st.markdown("#### 🔧 Practical Skill Assessment")
                st.write(hm["practical_skill_assessment"])

    # ── Tab 6: Improved Resume ────────────────────────────────
    with tabs[6]:
        humanized   = report.get("humanized_resume", {})
        rewritten   = report.get("rewritten_resume", "")
        resume_data = report.get("resume_data", {})

        if not humanized and not rewritten:
            st.warning("Resume Rewrite/Humanizer agents failed.")
        else:
            resume_text_out = ""

            if humanized:
                c1, c2 = st.columns(2)
                c1.markdown(f"**Humanization Score:** {score_badge(humanized.get('humanization_score'), '/100')}", unsafe_allow_html=True)
                ai_risk = humanized.get("ai_detection_risk_score", "N/A")
                ai_icon = "🟢" if isinstance(ai_risk, (int, float)) and ai_risk < 40 else "🔴"
                c2.markdown(f"**AI Detection Risk:** {ai_icon} {score_badge(ai_risk, '/100')}", unsafe_allow_html=True)

                changes = humanized.get("changes_made", [])
                if changes:
                    with st.expander(f"📝 {len(changes)} Changes Made"):
                        for c in changes:
                            st.write(f"• {c}")
                resume_text_out = humanized.get("humanized_resume", "")

            if not resume_text_out and rewritten:
                resume_text_out = rewritten

            if resume_text_out:
                st.markdown("---")
                st.markdown("#### ✨ Your Improved Resume")
                st.text_area(
                    "Improved Resume",
                    resume_text_out,
                    height=400,
                    key="resume_area",
                    label_visibility="collapsed",
                )

                # ── Ad inside resume tab ──────────────────────
                show_ad("Sponsored — Professional Resume Review Services")

                candidate_name = (resume_data.get("name") or "resume").replace(" ", "_")

                st.markdown("#### 🎨 Choose a Template & Download")
                templates = list_templates()

                fc1, fc2 = st.columns([2, 1])
                with fc1:
                    layout_options = ["All", "Classic", "Minimal", "Header Band", "Sidebar"]
                    chosen_layout  = st.selectbox("Filter by Layout", layout_options, key="layout_filter")
                with fc2:
                    ats_only = st.checkbox("ATS-safe only", value=True, key="ats_filter")

                filtered = [
                    t for t in templates
                    if (not ats_only or t["ats_safe"])
                    and (chosen_layout == "All" or t["name"].startswith(chosen_layout))
                ]

                if not filtered:
                    st.info("No templates match. Try unchecking 'ATS-safe only'.")
                else:
                    template_options = {t["name"]: t["id"] for t in filtered}
                    chosen_name = st.selectbox("Select Template", list(template_options.keys()), key="template_select")
                    chosen_id   = template_options[chosen_name]
                    chosen_tmpl = next(t for t in filtered if t["id"] == chosen_id)
                    st.caption(f"ℹ️ {chosen_tmpl['description']}")

                    if not chosen_tmpl["ats_safe"]:
                        st.warning("⚠️ **ATS Warning:** Sidebar layout may not parse correctly in Workday, Taleo, or Greenhouse. Use for direct email only.")
                    else:
                        st.success("✅ ATS Safe — works with all major ATS parsers.")

                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button("📄 Generate PDF", key="gen_pdf_btn", use_container_width=True):
                            try:
                                with temp_output_path(".pdf") as pdf_path:
                                    generate_resume_pdf(
                                        template_id=chosen_id,
                                        humanized_text=resume_text_out,
                                        resume_data=resume_data,
                                        output_path=pdf_path,
                                    )
                                    with open(pdf_path, "rb") as pf:
                                        pdf_bytes = pf.read()
                                st.download_button(
                                    "⬇️ Download Resume PDF", pdf_bytes,
                                    file_name=f"{candidate_name}_resume_{chosen_id}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    key="dl_resume_pdf",
                                )
                            except Exception as pdf_err:
                                st.error(f"PDF generation failed: {pdf_err}")

                    with btn_col2:
                        if st.button("📝 Generate DOCX", key="gen_docx_btn", use_container_width=True):
                            try:
                                with temp_output_path(".docx") as docx_path:
                                    generate_resume_docx(
                                        humanized_text=resume_text_out,
                                        resume_data=resume_data,
                                        output_path=docx_path,
                                    )
                                    with open(docx_path, "rb") as df:
                                        docx_bytes = df.read()
                                st.download_button(
                                    "⬇️ Download Resume DOCX", docx_bytes,
                                    file_name=f"{candidate_name}_resume.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True,
                                    key="dl_resume_docx",
                                )
                            except Exception as docx_err:
                                st.error(f"DOCX generation failed: {docx_err}")

    # ── Tab 7: Cover Letter ───────────────────────────────────
    with tabs[7]:
        cl          = report.get("cover_letter", "")
        resume_data = report.get("resume_data", {})

        if not cl:
            st.warning("Cover Letter agent failed.")
        else:
            st.markdown("#### 💌 Your Personalized Cover Letter")
            st.text_area(
                "Cover Letter",
                cl,
                height=350,
                key="cover_letter_area",
                label_visibility="collapsed",
            )

            candidate_name = (resume_data.get("name") or "candidate").replace(" ", "_")
            cl_col1, cl_col2 = st.columns(2)

            with cl_col1:
                try:
                    with temp_output_path(".pdf") as cl_pdf_path:
                        generate_cover_letter_pdf(
                            cover_letter_text=cl,
                            resume_data=resume_data,
                            company_name=company_name,
                            output_path=cl_pdf_path,
                        )
                        with open(cl_pdf_path, "rb") as pf:
                            cl_pdf_bytes = pf.read()
                    st.download_button(
                        "⬇️ Cover Letter PDF", cl_pdf_bytes,
                        file_name=f"{candidate_name}_cover_letter.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_cl_pdf",
                    )
                except Exception as e:
                    st.warning(f"Cover letter PDF failed: {e}")

            with cl_col2:
                try:
                    with temp_output_path(".docx") as cl_docx_path:
                        generate_cover_letter_docx(
                            cover_letter_text=cl,
                            resume_data=resume_data,
                            company_name=company_name,
                            output_path=cl_docx_path,
                        )
                        with open(cl_docx_path, "rb") as df:
                            cl_docx_bytes = df.read()
                    st.download_button(
                        "⬇️ Cover Letter DOCX", cl_docx_bytes,
                        file_name=f"{candidate_name}_cover_letter.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key="dl_cl_docx",
                    )
                except Exception as e:
                    st.warning(f"Cover letter DOCX failed: {e}")

    # ── Tab 8: Interview Prep ─────────────────────────────────
    with tabs[8]:
        prep = report.get("interview_prep", {})
        if not any(prep.values()):
            st.warning("Interview Coach agent failed.")
        else:
            category_config = {
                "hr_questions":             ("👥 HR Questions",             "These assess your personality, motivation, and cultural fit."),
                "technical_questions":      ("💻 Technical Questions",      "Be ready to explain your technical choices and trade-offs."),
                "project_questions":        ("🚀 Project Deep-Dives",       "You'll be asked to walk through your projects in detail."),
                "scenario_based_questions": ("🎯 Scenario / Behavioural",   "Use the STAR method: Situation, Task, Action, Result."),
            }

            for cat_key, (cat_label, cat_tip) in category_config.items():
                questions = prep.get(cat_key, [])
                if not questions:
                    continue
                st.markdown(f"#### {cat_label}")
                st.caption(f"💡 {cat_tip}")
                for i, q in enumerate(questions, 1):
                    with st.expander(f"Q{i}. {q}"):
                        st.info("Prepare a specific example from your experience to answer this.")
                st.markdown("")

    # ── Bottom Ad ─────────────────────────────────────────────
    st.markdown("---")
    show_ad("Sponsored — Job Boards & Career Opportunities")

    # ── Colorful Footer ───────────────────────────────────────
    _footer_logo_uri = _logo_data_uri()
    _footer_icon = (
        f'<img src="{_footer_logo_uri}" alt="logo" style="height:20px;width:20px;'
        f'vertical-align:-4px;border-radius:4px;margin-right:4px;">'
        if _footer_logo_uri else "🔥"
    )
    st.markdown(f"""
    <div class="footer-wrapper">
        <div class="footer-stats">
            <div class="footer-stat">
                <div class="footer-stat-number">13</div>
                <div class="footer-stat-label">AI Agents</div>
            </div>
            <div class="footer-stat">
                <div class="footer-stat-number">24</div>
                <div class="footer-stat-label">Resume Templates</div>
            </div>
            <div class="footer-stat">
                <div class="footer-stat-number">100%</div>
                <div class="footer-stat-label">Free to Use</div>
            </div>
            <div class="footer-stat">
                <div class="footer-stat-number">PDF+DOCX</div>
                <div class="footer-stat-label">Download Formats</div>
            </div>
        </div>
        <div class="footer-cols">
            <div class="footer-col">
                <div class="footer-brand">{_footer_icon} RoastCV</div>
                <div class="footer-brand-desc">
                    Brutally honest AI-powered resume analysis to help you land your dream job.
                    13 expert agents roast every aspect of your resume.
                </div>
            </div>
            <div class="footer-col">
                <div class="footer-col-title">Features</div>
                <div class="footer-link">✅ ATS Score Check</div>
                <div class="footer-link">✅ Gap Analysis</div>
                <div class="footer-link">✅ AI Resume Rewrite</div>
                <div class="footer-link">✅ Cover Letter Generator</div>
                <div class="footer-link">✅ Interview Preparation</div>
            </div>
            <div class="footer-col">
                <div class="footer-col-title">Supported Formats</div>
                <div class="footer-link">📄 PDF Resume Upload</div>
                <div class="footer-link">📝 DOCX Resume Upload</div>
                <div class="footer-link">🎨 24 PDF Templates</div>
                <div class="footer-link">📦 Download as PDF/DOCX</div>
            </div>
            <div class="footer-col">
                <div class="footer-col-title">Note</div>
                <div class="footer-link" style="color:rgba(255,255,255,0.55);font-size:0.78rem;line-height:1.6;">
                    ATS scores are heuristic estimates. Real ATS algorithms (Workday, Taleo, Greenhouse) are proprietary and may differ.
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            <span>© 2025 RoastCV — All rights reserved</span>
            <span style="opacity:0.5; margin: 0 0.75rem;">|</span>
            <span>Powered by AI · Built for job seekers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
