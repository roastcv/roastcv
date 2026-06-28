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

import requests
import wall_of_roasts
from resume_templates import generate_resume_pdf, list_templates, generate_blank_template
from resume_pdf import generate_cover_letter_pdf
from resume_docx import generate_resume_docx, generate_cover_letter_docx
import jd_analyzer

# ── Backend API Config ────────────────────────────────────────
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
POLL_INTERVAL = 2  # seconds

def _submit_to_backend(resume_file, jd_text: str, company_name: str):
    """Submit resume to FastAPI backend — returns task_id."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/v1/analyze",
            files={"resume_file": (resume_file.name, resume_file.read(), "application/octet-stream")},
            data={"jd_text": jd_text, "company_name": company_name},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("task_id")
        else:
            raise RuntimeError(r.json().get("detail", "Backend error"))
    except requests.ConnectionError:
        raise RuntimeError("Could not connect to backend. Is the server running?")

def _fetch_task_status(task_id: str) -> dict:
    """Fetch current task status."""
    try:
        r = requests.get(f"{BACKEND_URL}/api/v1/status/{task_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception:
        return {}


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
# Logo sits in the same folder as this script (everything lives together
# inside the "frontend" folder). Resolved relative to this script's own
# location (not the process's current working directory) so it keeps working
# no matter which directory Streamlit/Render launches the app from.
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

# Browser tab icon uses the actual logo file instead of an emoji. Falls back
# to the emoji only if logo.png isn't present (e.g. running before the asset
# is copied in), so set_page_config never crashes.
_page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else "🔥"

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


# ── Google Search Console Verification ───────────────────────────────────────
st.markdown("""
<meta name="google-site-verification" content="R6_IkqknJi3i8_JRs7Ie5IYhF3vYfiJjTCo0r44CdAk" />
""", unsafe_allow_html=True)

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
    if not os.path.exists(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
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

/* ── Hamburger: hide default collapse arrow, show ☰ icon ── */
button[data-testid="baseButton-headerNoPadding"] {
    display: none !important;
}
/* The actual sidebar toggle button Streamlit renders */
[data-testid="collapsedControl"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    background: rgba(37,99,235,0.15) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    position: fixed !important;
    top: 0.75rem !important;
    left: 0.75rem !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"] svg {
    display: none !important;
}
[data-testid="collapsedControl"]::after {
    content: "☰" !important;
    font-size: 1.2rem !important;
    color: #f8fafc !important;
    line-height: 1 !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(37,99,235,0.3) !important;
    border-color: rgba(96,165,250,0.6) !important;
}
/* Hide the default < > toggle inside open sidebar */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(37,99,235,0.1) !important;
    border: 1px solid rgba(96,165,250,0.2) !important;
    border-radius: 6px !important;
    color: #60a5fa !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(37,99,235,0.25) !important;
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

/* ── HIDE NAV TRIGGER BUTTONS GLOBALLY ── */
/* These are the hidden Streamlit buttons used by the JS navbar.
   We target them by their data-testid + position in DOM.
   The JS in show_navbar() also hides their container at runtime. */
#nav-hidden-btns,
#nav-hidden-btns * {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ── SIDEBAR NAVIGATION ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2e4a 100%) !important;
    border-right: 1px solid rgba(96,165,250,0.15) !important;
    min-width: 230px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(96,165,250,0.15);
}
.sidebar-logo-text {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
}
.sidebar-section-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: rgba(148,163,184,0.7);
    margin: 1.25rem 0 0.4rem 0.25rem;
}
/* Style Streamlit sidebar buttons to look like nav items */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: rgba(248,250,252,0.75) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: background 0.15s, color 0.15s !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(37,99,235,0.18) !important;
    color: #f8fafc !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
/* Active sidebar nav item — the Python code sets type="primary" on
   whichever button matches the current page, so targeting Streamlit's own
   button[kind="primary"] attribute (scoped to the sidebar only) is all
   that's needed. A clean, friendly red — not a dark/muddy red. */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(239, 68, 68, 0.35);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #fca5a5 0%, #f87171 100%) !important;
    color: #ffffff !important;
}
.sidebar-active-btn button {
    background: rgba(37,99,235,0.25) !important;
    color: #93c5fd !important;
    font-weight: 600 !important;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(96,165,250,0.12);
    margin: 1rem 0;
}

/* Hide Streamlit default "200MB per file" text — use aggressive selectors
   because Streamlit injects this text deep inside shadow-like divs and
   the exact element path changes between versions. We target every known
   location so at least one rule always fires. */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] > div > div > span:not(:first-child) {
    display: none !important;
}
/* Show our own size hint below the upload button */
[data-testid="stFileUploader"] > label + div::after {
    content: "PDF or DOCX · Max 5 MB";
    display: block;
    font-size: 0.78rem;
    color: rgba(250,250,250,0.4);
    margin-top: 6px;
    padding-left: 4px;
}

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

/* ── SIDEBAR STYLING ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2e4a 100%) !important;
    border-right: 1px solid rgba(96,165,250,0.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.2rem 0.8rem !important;
}

/* Sidebar nav section label */
.sb-section-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: rgba(96,165,250,0.6);
    padding: 0.9rem 0.6rem 0.35rem 0.6rem;
    margin-top: 0.4rem;
}

/* Sidebar nav item */
.sb-nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.52rem 0.75rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: rgba(248,250,252,0.72);
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    margin-bottom: 2px;
    text-decoration: none;
    border: none;
    background: transparent;
    width: 100%;
    text-align: left;
}
.sb-nav-item:hover {
    background: rgba(96,165,250,0.12);
    color: #f8fafc;
}
.sb-nav-item.active {
    background: rgba(37,99,235,0.22);
    color: #93c5fd;
    font-weight: 700;
    border-left: 3px solid #2563eb;
    padding-left: calc(0.75rem - 3px);
}
.sb-nav-icon { font-size: 1rem; flex-shrink: 0; }

/* Sidebar divider */
.sb-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 0.6rem 0;
}

/* Sidebar brand */
.sb-brand {
    font-size: 1.1rem;
    font-weight: 800;
    color: #f8fafc;
    padding: 0.4rem 0.6rem 0.8rem 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.5rem;
    cursor: pointer;
    border-radius: 8px;
    transition: background 0.15s, color 0.15s;
    position: relative;
}
.sb-brand:hover {
    background: rgba(96,165,250,0.08);
    color: #60a5fa;
}
.sb-score-pill {
    display: inline-block;
    background: rgba(37,99,235,0.25);
    border: 1px solid rgba(96,165,250,0.35);
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #93c5fd;
    margin-top: 0.3rem;
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
    ("active_tab", 0),
    ("nav_section", "upload"),   # tracks which section user navigated to
    # Two-Stage UI Loading state
    ("stage1_report", None),     # partial report after Stage 1 completes
    ("stage2_running", False),   # True while Stage 2 is loading in background
    ("stage2_complete", False),  # True once Stage 2 finishes
    ("task_id", None),           # FastAPI backend task ID
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── TOP NAVBAR ────────────────────────────────────────────────────────────────

def show_navbar():
    """
    Streamlit native sidebar navigation.
    Stays open until user manually closes it (default_sidebar_state=expanded).
    """
    has_report = st.session_state.report is not None
    _nav       = st.session_state.nav_section
    _logo_uri  = _logo_data_uri()

    # Score to show in sidebar header
    _score_text = ""
    if has_report:
        _ov  = st.session_state.report["final_decision"]["overall_score"]
        _col = "#4ade80" if _ov >= 70 else ("#fbbf24" if _ov >= 45 else "#f87171")
        _score_text = (
            f'<div style="display:inline-block;background:rgba(37,99,235,0.2);'
            f'border:1px solid {_col}55;border-radius:999px;padding:0.2rem 0.75rem;'
            f'font-size:0.82rem;font-weight:700;color:{_col};margin-top:0.4rem;">'
            f'Score: {_ov}/100</div>'
        )

    _icon_html = (
        f'<img src="{_logo_uri}" alt="RoastCV" '
        f'style="height:26px;width:26px;border-radius:6px;vertical-align:-5px;margin-right:6px;">'
        if _logo_uri else "🔥 "
    )

    with st.sidebar:
        # ── Brand header ─────────────────────────────────────────────────
        st.markdown(f"""
        <div style="padding:0.5rem 0 1rem 0;border-bottom:1px solid rgba(96,165,250,0.15);margin-bottom:0.75rem;">
            <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;display:flex;align-items:center;">
                {_icon_html}RoastCV
            </div>
            {_score_text}
        </div>
        """, unsafe_allow_html=True)

        # Clickable "Home" button right under the brand. This is a plain,
        # fully native st.button — same pattern as every other sidebar nav
        # button below — so it can NEVER leak styling onto other buttons.
        # (An earlier version tried to overlay an invisible button directly
        # on top of the logo using a CSS :has() selector; that selector
        # accidentally matched the sidebar's outer container too, which
        # stripped the background/border off every other nav button. This
        # version avoids that risk entirely.)
        if st.button(" Home", key="sb_logo_home", use_container_width=True):
            st.session_state.nav_section = "upload"
            st.rerun()


        # ── Main nav ─────────────────────────────────────────────────────
        st.markdown('<div style="font-size:0.62rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.7);margin-bottom:0.4rem;">TOOLS</div>', unsafe_allow_html=True)

        _nav_items = [
            ("upload",    "🚀 Analyze Resume"),
            ("keywords",  "🔑 JD Keywords"),
            ("templates", "🎨 Free Templates"),
            ("roasts",    "🔥 Wall of Roasts"),
            ("about",     "ℹ️ About"),
            ("contact",   "✉️ Contact"),
        ]

        for _key, _label in _nav_items:
            _active = _nav == _key
            if st.button(
                _label,
                key=f"sb_{_key}",
                use_container_width=True,
                type="primary" if _active else "secondary",
            ):
                st.session_state.nav_section = _key
                st.rerun()

        # ── Results nav (only after analysis) ────────────────────────────
        if has_report:
            st.markdown("---")
            st.markdown('<div style="font-size:0.62rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:rgba(148,163,184,0.7);margin-bottom:0.4rem;margin-top:0.5rem;">RESULTS</div>', unsafe_allow_html=True)

            _result_items = [
                ("overview",       "📊 Score Overview"),
                ("audit",          "🔍 Resume Audit"),
                ("ats",            "🤖 ATS Review"),
                ("gap",            "📋 Gap Analysis"),
                ("hr",             "🔥 HR Roast"),
                ("recruiter",      "👔 Recruiter"),
                ("hiring_manager", "🏢 Hiring Manager"),
                ("rewrite",        "✨ ATS Resume"),
                ("cover_letter",   "📝 Cover Letter"),
                ("interview",      "🎯 Interview Prep"),
            ]

            for _key, _label in _result_items:
                _active = _nav == _key
                if st.button(
                    _label,
                    key=f"sb_{_key}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    st.session_state.nav_section = _key
                    st.rerun()

        # ── New Analysis button ───────────────────────────────────────────
        if has_report:
            st.markdown("---")
            if st.button("🔄 New Analysis", key="sb_new", use_container_width=True, type="primary"):
                st.session_state.report = None
                st.session_state.report_company_name = ""
                st.session_state.current_step = 0
                st.session_state.nav_section = "upload"
                st.rerun()

        # ── Footer note ───────────────────────────────────────────────────
        st.markdown("""
        <div style="position:absolute;bottom:1rem;left:1rem;right:1rem;
             font-size:0.72rem;color:rgba(148,163,184,0.5);text-align:center;line-height:1.5;">
            Free · AI-Powered · 13 Agents
        </div>
        """, unsafe_allow_html=True)

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


def _looks_like_raw_dict(text: str) -> bool:
    """
    FIX: the Humanizer/Rewrite agents are supposed to return a plain-text
    resume (SUMMARY / SKILLS / EXPERIENCE ...). Occasionally the LLM instead
    echoes back the structured resume_data dict as a stringified Python/JSON
    object (e.g. "{'contact': {...}, 'skills': [...], ...}"). Showing that
    raw in the resume box looks broken to the user, so we detect it here.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    # Cheap signal: structured resume_data always has these literal keys
    markers = ("'contact':", '"contact":', "'skills':", '"skills":', "'experience':", '"experience":')
    return any(m in stripped[:400] for m in markers)


def _dict_to_plain_resume(resume_data: dict) -> str:
    """Builds a simple plain-text resume from structured resume_data —
    used only as a last-resort fallback when the AI agents return malformed
    text, so the user always sees a real resume instead of raw dict text."""
    if not isinstance(resume_data, dict) or not resume_data:
        return ""

    lines = []
    name = resume_data.get("name", "")
    if name:
        lines.append(name)

    contact = resume_data.get("contact", {}) or {}
    contact_bits = [contact.get(k) for k in ("email", "phone", "location", "linkedin", "github", "portfolio") if contact.get(k)]
    if contact_bits:
        lines.append(" | ".join(contact_bits))

    if resume_data.get("summary"):
        lines += ["", "SUMMARY", resume_data["summary"]]

    skills = resume_data.get("skills") or []
    if skills:
        lines += ["", "SKILLS"] + [f"- {s}" for s in skills]

    experience = resume_data.get("experience") or []
    if experience:
        lines += ["", "EXPERIENCE"]
        for exp in experience:
            header = " | ".join(x for x in [exp.get("role", ""), exp.get("company", ""), exp.get("duration", "")] if x)
            if header:
                lines.append(header)
            for d in exp.get("description", []) or []:
                lines.append(f"- {d}")

    projects = resume_data.get("projects") or []
    if projects:
        lines += ["", "PROJECTS"]
        for proj in projects:
            title = proj.get("title", "")
            if title:
                lines.append(title)
            if proj.get("description"):
                lines.append(f"- {proj['description']}")
            tech = proj.get("tech_used") or []
            if tech:
                lines.append(f"- Tech used: {', '.join(tech)}")

    education = resume_data.get("education") or []
    if education:
        lines += ["", "EDUCATION"]
        for edu in education:
            line = " | ".join(x for x in [edu.get("degree", ""), edu.get("institution", ""), edu.get("year", "")] if x)
            if line:
                lines.append(line)

    if resume_data.get("certifications"):
        lines += ["", "CERTIFICATIONS"] + [f"- {c}" for c in resume_data["certifications"]]

    if resume_data.get("achievements"):
        lines += ["", "ACHIEVEMENTS"] + [f"- {a}" for a in resume_data["achievements"]]

    return "\n".join(lines).strip()


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
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=110)
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

    # ── Landing page footer ───────────────────────────────────────────────
    _lp_logo_uri = _logo_data_uri()
    _lp_icon = (
        f'<img src="{_lp_logo_uri}" alt="logo" style="height:18px;width:18px;'
        f'vertical-align:-3px;border-radius:3px;margin-right:4px;">' 
        if _lp_logo_uri else "🔥"
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
                <div class="footer-brand">{_lp_icon} RoastCV</div>
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
            <span>© 2026 RoastCV — All rights reserved</span>
            <span style="opacity:0.5; margin: 0 0.75rem;">|</span>
            <span>Powered by AI · Built for job seekers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


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


# ── STANDALONE KEYWORD SUGGESTER (no resume needed) ───────────────────────────
for _kw_key, _kw_default in [
    ("kw_result", None),
    ("kw_running", False),
]:
    if _kw_key not in st.session_state:
        st.session_state[_kw_key] = _kw_default


def show_keyword_suggester():
    st.markdown("""
    <div class="page-section" style="margin:2rem 0 1rem 0;">
        <div class="section-kicker">Free Instant Tool</div>
        <div class="section-title" style="font-size:1.5rem;">🔑 JD Keyword Extractor</div>
        <div class="section-sub" style="margin-bottom:1rem;">
            No resume needed. Paste any job description and instantly get the
            top keywords, required skills, and tools — before you even apply.
        </div>
    </div>
    """, unsafe_allow_html=True)

    kw_col1, kw_col2 = st.columns([3, 2], gap="large")

    with kw_col1:
        st.markdown("""
        <div style="background:linear-gradient(160deg,#1a2e4a 0%,#0f172a 100%);
             border:1px solid rgba(96,165,250,0.2); border-radius:16px; padding:1.5rem;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#60a5fa;margin-bottom:0.75rem;">
                STEP 1 — Paste Job Description
            </div>
        """, unsafe_allow_html=True)

        kw_jd_text = st.text_area(
            "Paste Job Description",
            height=180,
            key="kw_jd_text",
            placeholder="Paste the full job description here — the more complete, the better...",
            disabled=st.session_state.kw_running,
            label_visibility="collapsed",
        )

        kw_clicked = st.button(
            "🔍 Extract Keywords — Instant & Free",
            key="kw_find_btn",
            disabled=st.session_state.kw_running,
            use_container_width=True,
            type="primary",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.kw_running:
            st.markdown("""
            <div style="text-align:center;padding:0.75rem;color:#60a5fa;font-size:0.88rem;">
                ⚡ Analyzing job description...
            </div>""", unsafe_allow_html=True)

    with kw_col2:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(96,165,250,0.15);
             border-radius:16px;padding:1.4rem;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#60a5fa;margin-bottom:1rem;">WHY USE THIS</div>
            <div style="font-size:0.88rem;color:rgba(248,250,252,0.75);line-height:1.8;">
                ✅ Know exactly which keywords are missing from your resume<br>
                ✅ Tailor your resume for each job in minutes<br>
                ✅ See required vs preferred skills clearly<br>
                ✅ Works before you upload anything<br>
                ✅ 100% free — 1 AI call, instant results
            </div>
        </div>
        """, unsafe_allow_html=True)

    if kw_clicked:
        if not kw_jd_text.strip():
            st.error("📋 Please paste a job description first.")
        elif len(kw_jd_text.strip()) < 50:
            st.error("📋 Job description looks too short — paste the full JD for accurate results.")
        else:
            st.session_state.kw_running = True
            with st.spinner("⚡ Extracting keywords..."):
                try:
                    st.session_state.kw_result = jd_analyzer.run(kw_jd_text)
                except Exception as e:
                    st.session_state.kw_result = None
                    st.error(f"Couldn't analyze the JD right now: {e}")
            st.session_state.kw_running = False

    result = st.session_state.kw_result
    if result:
        required = result.get("required_skills", [])
        preferred = result.get("preferred_skills", [])
        keywords = result.get("important_keywords", [])
        tools = result.get("tools_technologies", [])
        exp_req = result.get("experience_required", "")

        if not any([required, preferred, keywords, tools]):
            st.warning("Couldn't extract structured keywords from this JD — try pasting more of the text.")
        else:
            st.markdown("""
            <div style="margin:1.5rem 0 0.5rem 0;padding:1.5rem;
                 background:linear-gradient(160deg,#1a2e4a 0%,#0f172a 100%);
                 border:1px solid rgba(96,165,250,0.2);border-radius:16px;">
                <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:1px;color:#60a5fa;margin-bottom:1rem;">
                     ✅ RESULTS — Keywords Extracted
                </div>
            """, unsafe_allow_html=True)

            if exp_req:
                st.markdown(
                    f'<div style="display:inline-block;background:rgba(37,99,235,0.15);'
                    f'border:1px solid rgba(96,165,250,0.3);border-radius:8px;padding:0.4rem 0.9rem;'
                    f'font-size:0.85rem;color:#93c5fd;margin-bottom:1rem;">'
                    f'📌 Experience Required: <strong>{exp_req}</strong></div>',
                    unsafe_allow_html=True
                )

            top_keywords = (keywords or required)[:12]
            if top_keywords:
                st.markdown(
                    '<div style="font-size:0.82rem;font-weight:700;color:#f8fafc;'
                    'margin-bottom:0.6rem;">🎯 Must-Have Keywords for Your Resume:</div>',
                    unsafe_allow_html=True
                )
                chips = "".join(
                    f'<span style="display:inline-block;background:rgba(37,99,235,0.2);'
                    f'border:1px solid rgba(96,165,250,0.35);border-radius:999px;'
                    f'padding:0.3rem 0.85rem;font-size:0.8rem;color:#93c5fd;'
                    f'margin:0.2rem 0.3rem 0.2rem 0;font-weight:600;">{k}</span>'
                    for k in top_keywords
                )
                st.markdown(f'<div style="margin-bottom:1.2rem;">{chips}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            kc1, kc2, kc3 = st.columns(3)
            with kc1:
                if required:
                    st.markdown("**✅ Required Skills**")
                    for s in required:
                        st.markdown(f"- {s}")
            with kc2:
                if tools:
                    st.markdown("**🛠️ Tools & Technologies**")
                    for t in tools:
                        st.markdown(f"- {t}")
            with kc3:
                if preferred:
                    st.markdown("**➕ Preferred Skills**")
                    for s in preferred:
                        st.markdown(f"- {s}")

            st.markdown("""
            <div style="margin-top:1.2rem;padding:0.9rem 1.2rem;
                 background:linear-gradient(90deg,rgba(37,99,235,0.12),rgba(124,58,237,0.08));
                 border-left:3px solid #2563eb;border-radius:0 10px 10px 0;
                 font-size:0.88rem;color:rgba(248,250,252,0.8);">
                💡 <strong>Next step:</strong> Upload your resume below and get a full
                match % analysis, ATS score, rewrite + cover letter — all free.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:2rem 0;'>", unsafe_allow_html=True)


# ── FREE BLANK ATS TEMPLATES (no resume, no LLM call needed) ──────────────────
def show_blank_templates():
    st.markdown("""
    <div class="page-section" style="margin:1rem 0;">
        <div class="section-kicker">100% Free Download</div>
        <div class="section-title" style="font-size:1.5rem;">🎨 Free ATS Resume Templates</div>
        <div class="section-sub" style="margin-bottom:1.25rem;">
            24 professional, ATS-safe layouts — download blank and fill them yourself.
            Same templates used to generate your analyzed resume.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature badges row
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:1.5rem;">
        <span style="background:rgba(37,99,235,0.15);border:1px solid rgba(96,165,250,0.3);
             border-radius:999px;padding:0.3rem 0.9rem;font-size:0.8rem;color:#93c5fd;font-weight:600;">
             ✅ ATS-Safe Formatting</span>
        <span style="background:rgba(37,99,235,0.15);border:1px solid rgba(96,165,250,0.3);
             border-radius:999px;padding:0.3rem 0.9rem;font-size:0.8rem;color:#93c5fd;font-weight:600;">
             📄 Single Column Layout</span>
        <span style="background:rgba(37,99,235,0.15);border:1px solid rgba(96,165,250,0.3);
             border-radius:999px;padding:0.3rem 0.9rem;font-size:0.8rem;color:#93c5fd;font-weight:600;">
             🆓 Free PDF Download</span>
        <span style="background:rgba(37,99,235,0.15);border:1px solid rgba(96,165,250,0.3);
             border-radius:999px;padding:0.3rem 0.9rem;font-size:0.8rem;color:#93c5fd;font-weight:600;">
             24 Templates Available</span>
    </div>
    """, unsafe_allow_html=True)

    templates = list_templates()
    layout_options = ["All", "Classic", "Minimal", "Header Band", "Sidebar"]

    tf1, tf2, tf3 = st.columns([2, 1, 1])
    with tf1:
        chosen_layout = st.selectbox("Filter by Layout Style", layout_options, key="blank_layout_filter")
    with tf2:
        ats_only = st.checkbox("ATS-safe only", value=True, key="blank_ats_filter")
    with tf3:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

    filtered = [
        t for t in templates
        if (not ats_only or t["ats_safe"])
        and (chosen_layout == "All" or t["name"].startswith(chosen_layout))
    ]

    if not filtered:
        st.info("No templates match. Try unchecking 'ATS-safe only'.")
        return

    template_options = {t["name"]: t["id"] for t in filtered}

    ts1, ts2 = st.columns([3, 2], gap="large")
    with ts1:
        chosen_name = st.selectbox(
            "Choose Template",
            list(template_options.keys()),
            key="blank_template_select",
        )
    chosen_id   = template_options[chosen_name]
    chosen_tmpl = next(t for t in filtered if t["id"] == chosen_id)

    with ts2:
        ats_tag = (
            '<span style="background:rgba(74,222,128,0.15);border:1px solid rgba(74,222,128,0.35);'
            'border-radius:999px;padding:0.2rem 0.7rem;font-size:0.75rem;color:#4ade80;font-weight:700;">'
            '✅ ATS Safe</span>'
            if chosen_tmpl["ats_safe"] else
            '<span style="background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.35);'
            'border-radius:999px;padding:0.2rem 0.7rem;font-size:0.75rem;color:#fbbf24;font-weight:700;">'
            '⚠️ Not ATS Safe</span>'
        )
        st.markdown(
            f'<div style="padding-top:1.9rem;">'
            f'{ats_tag}'
            f'<div style="font-size:0.8rem;color:rgba(248,250,252,0.55);margin-top:0.4rem;">'
            f'{chosen_tmpl["description"]}</div></div>',
            unsafe_allow_html=True
        )

    if not chosen_tmpl["ats_safe"]:
        st.warning("⚠️ Sidebar layouts may not parse in Workday/Taleo. Best for emailing directly to a hiring manager.")

    if st.button("⬇️ Generate & Download Free Template", key="gen_blank_btn", use_container_width=True, type="primary"):
        try:
            with temp_output_path(".pdf") as blank_path:
                generate_blank_template(chosen_id, blank_path)
                with open(blank_path, "rb") as bf:
                    blank_bytes = bf.read()
            st.download_button(
                f"📄 Download  {chosen_name}  (PDF)",
                blank_bytes,
                file_name=f"blank_resume_{chosen_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_blank_pdf",
            )
        except Exception as e:
            st.error(f"Template generation failed: {e}")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:2rem 0;'>", unsafe_allow_html=True)


# ── WALL OF ROASTS PAGE ───────────────────────────────────────────────────────
def show_wall_of_roasts():
    total = wall_of_roasts.roast_count()
    st.markdown(f"""
    <div class="page-section" style="margin:1rem 0 1.5rem 0;">
        <div class="section-kicker">Community</div>
        <div class="section-title" style="font-size:1.6rem;">🔥 Wall of Roasts</div>
        <div class="section-sub" style="margin-bottom:0;">
            Brutally honest AI feedback — anonymized and shared by the community.
            {total} roast{"s" if total != 1 else ""} so far.
        </div>
    </div>
    """, unsafe_allow_html=True)

    roasts = wall_of_roasts.get_roasts(limit=30)

    if not roasts:
        st.info("No roasts yet — be the first! Analyze your resume and check the share box on the overview page.")
        return

    # Score filter
    _filter = st.select_slider(
        "Filter by score range",
        options=["All", "0–40 (Needs Work)", "41–70 (Moderate)", "71–100 (Strong)"],
        value="All",
        key="roast_filter",
    )

    def _passes_filter(r):
        s = r.get("overall", 0)
        if _filter == "0–40 (Needs Work)":   return s <= 40
        if _filter == "41–70 (Moderate)":    return 41 <= s <= 70
        if _filter == "71–100 (Strong)":     return s >= 71
        return True

    filtered = [r for r in roasts if _passes_filter(r)]

    if not filtered:
        st.info("No roasts match this filter.")
        return

    for r in filtered:
        s = r.get("overall", 0)
        hr = r.get("hr_score", 0)

        if s >= 71:
            score_css = "background:#dcfce7;color:#15803d;"
        elif s >= 41:
            score_css = "background:#fef9c3;color:#92400e;"
        else:
            score_css = "background:#fee2e2;color:#dc2626;"

        shortlist = r.get("shortlist", "")
        shortlist_tag = ""
        if shortlist == "Yes":
            shortlist_tag = '<span style="background:#dcfce7;color:#15803d;font-size:0.72rem;font-weight:700;padding:2px 8px;border-radius:999px;margin-left:0.5rem;">Shortlisted</span>'
        elif shortlist == "No":
            shortlist_tag = '<span style="background:#fee2e2;color:#dc2626;font-size:0.72rem;font-weight:700;padding:2px 8px;border-radius:999px;margin-left:0.5rem;">Rejected</span>'
        elif shortlist == "Maybe":
            shortlist_tag = '<span style="background:#fef9c3;color:#92400e;font-size:0.72rem;font-weight:700;padding:2px 8px;border-radius:999px;margin-left:0.5rem;">Maybe</span>'

        st.markdown(f"""
        <div style="background:linear-gradient(160deg,#1a2e4a 0%,#0f172a 100%);
             border:1px solid rgba(96,165,250,0.15);border-radius:14px;
             padding:1.1rem 1.4rem;margin-bottom:0.75rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;
                 flex-wrap:wrap;gap:0.5rem;margin-bottom:0.6rem;">
                <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">
                    <span style="font-size:0.9rem;font-weight:600;color:#f8fafc;">
                        {r.get("role","Professional")}
                    </span>
                    <span style="font-size:0.78rem;color:rgba(248,250,252,0.5);">
                        · {r.get("experience","?")} · {r.get("date","")}
                    </span>
                    {shortlist_tag}
                </div>
                <div style="display:flex;gap:0.5rem;align-items:center;">
                    <span style="font-size:0.75rem;color:rgba(248,250,252,0.5);">
                        HR {hr}/100
                    </span>
                    <span style="{score_css}font-size:0.8rem;font-weight:700;
                         padding:3px 10px;border-radius:999px;">
                        {s}/100
                    </span>
                </div>
            </div>
            <div style="font-size:0.88rem;color:rgba(248,250,252,0.72);
                 line-height:1.65;font-style:italic;">
                "{r.get("feedback","")}"
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0;font-size:0.82rem;
         color:rgba(248,250,252,0.4);">
        Showing {len(filtered)} of {total} roasts · Analyze your resume to add yours
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Analyze My Resume", type="primary", use_container_width=False, key="roast_wall_cta"):
        st.session_state.nav_section = "upload"
        st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:2rem 0;'>", unsafe_allow_html=True)


# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
def show_about():
    st.markdown("""
    <div class="page-section" style="margin:2rem 0 1.5rem 0;">
        <div class="section-kicker">Our Story</div>
        <div class="section-title" style="font-size:1.8rem;">About RoastCV</div>
        <div class="section-sub">Built by job seekers, for job seekers.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(160deg,#1a2e4a 0%,#0f172a 100%);
         border:1px solid rgba(96,165,250,0.15);border-radius:16px;padding:2rem;margin-bottom:1.5rem;">
        <p style="font-size:1rem;color:rgba(248,250,252,0.85);line-height:1.8;margin:0 0 1rem 0;">
            <strong style="color:#f8fafc;">RoastCV</strong> is a free AI-powered resume analyzer that gives you
            the kind of honest, direct feedback that a real recruiter or hiring manager would give —
            not the sugar-coated advice you get from generic tools.
        </p>
        <p style="font-size:1rem;color:rgba(248,250,252,0.85);line-height:1.8;margin:0;">
            We built RoastCV because we were tired of sending resumes into a black hole. Most resume
            checkers tell you "add more action verbs" — we run <strong style="color:#60a5fa;">13 specialized AI agents</strong>
            that each look at your resume from a different angle: ATS systems, HR recruiters,
            hiring managers, gap analysis, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3, gap="large")
    steps = [
        ("📤", "Upload Your Resume", "Upload a PDF or DOCX resume, or even a LinkedIn profile PDF export. We support both."),
        ("🤖", "13 Agents Analyze It", "Our AI pipeline runs 13 specialized agents — ATS review, gap analysis, HR roast, hiring manager view, and more."),
        ("📊", "Get Actionable Results", "You receive a score, a rewritten resume, a cover letter, interview prep questions, and a final hire/no-hire verdict."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        col.markdown(f"""
        <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(96,165,250,0.15);
             border-radius:12px;padding:1.4rem;text-align:center;">
            <div style="font-size:2rem;margin-bottom:0.75rem;">{icon}</div>
            <div style="font-weight:700;color:#f8fafc;margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.87rem;color:rgba(248,250,252,0.65);line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### What Makes RoastCV Different")
    diffs = [
        ("🆓", "Completely Free", "No sign-up, no credit card, no limits. Upload and analyze instantly."),
        ("🔒", "Private by Design", "Your resume is never stored on our servers. Data lives only in your browser session."),
        ("🎯", "13 Specialized Agents", "Each agent has a unique role — not one LLM doing everything generically."),
        ("📄", "Download Everything", "Get your rewritten resume as PDF or DOCX, plus a cover letter, all formatted and ready."),
        ("🔑", "ATS Keyword Analysis", "Know exactly which keywords your resume is missing before you apply."),
        ("💬", "Honest Feedback", "We tell you what a real recruiter thinks — not what you want to hear."),
    ]
    d1, d2 = st.columns(2, gap="large")
    for i, (icon, title, desc) in enumerate(diffs):
        col = d1 if i % 2 == 0 else d2
        col.markdown(f"""
        <div style="display:flex;gap:1rem;align-items:flex-start;padding:1rem;
             border-bottom:1px solid rgba(255,255,255,0.05);">
            <div style="font-size:1.4rem;flex-shrink:0;">{icon}</div>
            <div>
                <div style="font-weight:600;color:#f8fafc;margin-bottom:0.2rem;">{title}</div>
                <div style="font-size:0.85rem;color:rgba(248,250,252,0.6);line-height:1.5;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### The Technology Behind RoastCV")
    st.markdown("""
    <div style="background:rgba(37,99,235,0.06);border:1px solid rgba(96,165,250,0.12);
         border-radius:12px;padding:1.5rem;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;">
    """ + "".join([
        f'<span style="background:rgba(37,99,235,0.18);border:1px solid rgba(96,165,250,0.3);'
        f'border-radius:999px;padding:0.3rem 0.9rem;font-size:0.82rem;color:#93c5fd;font-weight:600;">{t}</span>'
        for t in ["Python", "Streamlit", "Google Gemini", "Groq", "Cerebras", "Mistral AI",
                  "ReportLab", "python-docx", "pdfplumber", "Multi-Agent Pipeline"]
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Try RoastCV Now — It's Free", type="primary", use_container_width=True, key="about_cta"):
        st.session_state.nav_section = "upload"
        st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:2rem 0;'>", unsafe_allow_html=True)


# ── CONTACT US PAGE ───────────────────────────────────────────────────────────
def show_contact():
    st.markdown("""
    <div class="page-section" style="margin:2rem 0 1.5rem 0;">
        <div class="section-kicker">Get In Touch</div>
        <div class="section-title" style="font-size:1.8rem;">Contact Us</div>
        <div class="section-sub">We read every message. Usually reply within 24–48 hours.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("""
        <div style="background:linear-gradient(160deg,#1a2e4a 0%,#0f172a 100%);
             border:1px solid rgba(96,165,250,0.2);border-radius:16px;padding:1.75rem;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#60a5fa;margin-bottom:1.25rem;">Send Us a Message</div>
        """, unsafe_allow_html=True)

        contact_name  = st.text_input("Your Name", placeholder="John Smith", key="contact_name")
        contact_email = st.text_input("Your Email", placeholder="john@example.com", key="contact_email")
        contact_topic = st.selectbox(
            "Topic",
            ["General Feedback", "Bug Report", "Feature Request", "Partnership / Business", "Other"],
            key="contact_topic",
        )
        contact_msg = st.text_area(
            "Message",
            placeholder="Tell us what's on your mind — feedback, bugs, ideas, anything...",
            height=150,
            key="contact_msg",
        )

        if st.button("📨 Send Message", type="primary", use_container_width=True, key="contact_send"):
            if not contact_name.strip():
                st.error("Please enter your name.")
            elif not contact_email.strip() or "@" not in contact_email:
                st.error("Please enter a valid email address.")
            elif not contact_msg.strip() or len(contact_msg.strip()) < 10:
                st.error("Please write a message (at least 10 characters).")
            else:
                st.success(
                    f"✅ Thanks, **{contact_name.strip()}**! We've received your message and "
                    f"will reply to **{contact_email.strip()}** within 24–48 hours."
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(96,165,250,0.15);
             border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#60a5fa;margin-bottom:1rem;">Direct Contact</div>
            <div style="font-size:0.9rem;color:rgba(248,250,252,0.8);line-height:2;">
                📧 <a href="mailto:support@roastcv.in"
                    style="color:#60a5fa;text-decoration:none;">support@roastcv.in</a><br>
                🌐 <a href="https://roastcv.in"
                    style="color:#60a5fa;text-decoration:none;">roastcv.in</a><br>
                🐦 <a href="https://twitter.com/roastcv"
                    style="color:#60a5fa;text-decoration:none;">@roastcv</a>
            </div>
        </div>
        <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(96,165,250,0.15);
             border-radius:16px;padding:1.5rem;">
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:1px;color:#60a5fa;margin-bottom:1rem;">Common Questions</div>
            <div style="font-size:0.85rem;color:rgba(248,250,252,0.7);line-height:1.9;">
                <strong style="color:#f8fafc;">Is RoastCV really free?</strong><br>
                Yes — always free, no hidden paywalls.<br><br>
                <strong style="color:#f8fafc;">Is my resume stored anywhere?</strong><br>
                No. Data is cleared when you close the tab.<br><br>
                <strong style="color:#f8fafc;">Which file types are supported?</strong><br>
                PDF and DOCX resumes, plus LinkedIn Profile PDFs.<br><br>
                <strong style="color:#f8fafc;">How long does analysis take?</strong><br>
                Typically 1–3 minutes for all 13 agents.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:2rem 0;'>", unsafe_allow_html=True)


# ── NAVBAR ───────────────────────────────────────────────────────────────────
show_navbar()

# ── MAIN CONTENT ROUTING ──────────────────────────────────────────────────────
# Determine which section to show based on sidebar navigation state.
# When a report exists, default to "overview" unless user clicked something else.
_nav = st.session_state.nav_section
has_report = st.session_state.report is not None

# If report just arrived and nav is still on a tool section, switch to overview
if has_report and _nav in ("upload", "keywords", "templates", "roasts", "about", "contact"):
    st.session_state.nav_section = "overview"
    _nav = "overview"

# ── SECTION: About ───────────────────────────────────────────────────────────
if _nav == "about":
    show_about()

# ── SECTION: Contact Us ──────────────────────────────────────────────────────
elif _nav == "contact":
    show_contact()

# ── SECTION: Wall of Roasts ──────────────────────────────────────────────────
elif _nav == "roasts":
    show_wall_of_roasts()

# ── TOOLS: always show landing hero at top ───────────────────────────────────
elif not has_report and _nav == "upload":
    show_landing()

# ── SECTION: JD Keyword Extractor ────────────────────────────────────────────
elif _nav == "keywords":
    show_keyword_suggester()

# ── SECTION: Free ATS Templates ──────────────────────────────────────────────
elif _nav == "templates":
    show_blank_templates()

# ── SECTION: Upload / Analyze (shown on landing page only) ───────────────────
if not has_report and _nav not in ("keywords", "templates", "roasts", "about", "contact"):
    st.markdown("### 📂 Upload Your Resume & Job Description")

    upload_mode = st.radio(
        "What are you uploading?",
        options=["📄 Resume (PDF or DOCX)", "🔗 LinkedIn Profile PDF"],
        horizontal=True,
        disabled=st.session_state.running,
        key="upload_mode",
    )

    is_linkedin_mode = upload_mode == "🔗 LinkedIn Profile PDF"

    if is_linkedin_mode:
        st.info(
            "**How to export your LinkedIn PDF:** Go to your LinkedIn profile → "
            "click the **More** button → **Save to PDF**. Upload that file here. "
            "Our AI will convert it into a full ATS-optimized resume automatically."
        )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        if is_linkedin_mode:
            resume_file = st.file_uploader(
                "🔗 Upload LinkedIn Profile PDF — Max 5 MB",
                type=["pdf"],
                disabled=st.session_state.running,
                help="Export your LinkedIn profile as a PDF and upload it here.",
                label_visibility="visible",
            )
            st.caption("📁 Accepted: PDF only (LinkedIn Save to PDF) · Maximum size: **5 MB**")
        else:
            resume_file = st.file_uploader(
                "📎 Upload Resume — PDF or DOCX, Max 5 MB",
                type=["pdf", "docx"],
                disabled=st.session_state.running,
                help="Text-based PDF or DOCX required — scanned/image PDFs cannot be read.",
                label_visibility="visible",
            )
            st.caption("📁 Accepted: PDF, DOCX · Maximum size: **5 MB**")

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

    _btn_label = (
        "🔗 Build Resume from LinkedIn & Analyze — Free"
        if is_linkedin_mode
        else "🚀 Analyze My Resume — Free"
    )

    btn_col, reset_col = st.columns([4, 1])
    run_clicked = btn_col.button(
        _btn_label,
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )

    if st.session_state.report is not None:
        if reset_col.button("🔄 New", use_container_width=True, disabled=st.session_state.running):
            st.session_state.report = None
            st.session_state.report_company_name = ""
            st.session_state.current_step = 0
            st.session_state.nav_section = "upload"
            st.session_state.stage1_report = None
            st.session_state.stage2_running = False
            st.session_state.stage2_complete = False
            st.rerun()

    # Progress UI renders here while the pipeline runs so the form stays in place
    analysis_placeholder = st.empty()

    # Show landing page info only when no report and nothing is running
    if st.session_state.report is None and not st.session_state.running:
        show_more_info()

    # ── RUN PIPELINE — FastAPI Backend ───────────────────────────────────────
    if run_clicked:
        if not resume_file and not jd_text.strip():
            st.error("⬆️ Please upload your resume and paste a job description to continue.")
        elif not resume_file:
            if is_linkedin_mode:
                st.error("⬆️ Please upload your LinkedIn PDF to continue.")
            else:
                st.error("⬆️ Please upload your resume (PDF or DOCX) to continue.")
        elif resume_file.size > MAX_RESUME_SIZE_MB * 1024 * 1024:
            st.error(f"❌ File too large. Maximum size is {MAX_RESUME_SIZE_MB} MB.")
        elif not jd_text.strip():
            st.error("📋 Please paste the job description to continue.")
        elif len(jd_text.strip()) < 50:
            st.error("📋 Job description is too short — please paste the full job description.")
        else:
            try:
                task_id = _submit_to_backend(resume_file, jd_text, company_name)
                st.session_state.task_id = task_id
                st.session_state.running = True
                st.session_state.stage1_report = None
                st.session_state.stage2_complete = False
                st.session_state.report_company_name = company_name
                st.rerun()
            except Exception as e:
                _show_friendly_error(e)

    # ── POLLING LOOP — check task status while running ──────────────
    if st.session_state.running and st.session_state.task_id:
        task_id = st.session_state.task_id

        with analysis_placeholder.container():
            st.markdown("---")
            st.markdown("### ⚙️ Analysis in progress...")
            st.caption("⚡ ATS Score ready in ~20 sec. Rewrite & Cover Letter will follow.")

            progress_bar   = st.progress(st.session_state.get("_poll_progress", 0))
            step_container = st.empty()
            tip_container  = st.empty()

            task = _fetch_task_status(task_id)
            status   = task.get("status", "queued")
            progress = task.get("progress", 0)
            message  = task.get("message", "")

            progress_bar.progress(progress / 100)
            tip_container.markdown(
                f'<div class="tip-box">⚙️ {message}</div>',
                unsafe_allow_html=True,
            )
            st.session_state["_poll_progress"] = progress

            # Stage 1 complete — show partial results
            if status == "stage1_complete" and task.get("partial_report"):
                partial = task["partial_report"]
                st.session_state.stage1_report = partial
                st.session_state.report = partial
                st.session_state.stage2_running = True
                st.session_state.nav_section = "overview"
                step_container.markdown(
                    '<div style="color:#22c55e;font-weight:600;padding:0.5rem 0;">'
                    '✅ ATS Score + Resume Health ready! Loading Rewrite...</div>',
                    unsafe_allow_html=True,
                )

            # Fully complete
            elif status == "complete" and task.get("report"):
                st.session_state.report = task["report"]
                st.session_state.stage2_complete = True
                st.session_state.stage2_running = False
                st.session_state.running = False
                st.session_state.task_id = None
                st.session_state.nav_section = "overview"
                st.session_state["roast_share_asked"] = False
                st.rerun()

            # Error
            elif status == "error":
                st.session_state.running = False
                st.session_state.task_id = None
                _show_friendly_error(Exception(task.get("error", "Analysis failed.")))

        # Auto-refresh every 2 seconds
        if st.session_state.running:
            time.sleep(POLL_INTERVAL)
            st.rerun()

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
    _is_stage1_only = report.get("_stage", 2) == 1  # True = Stage 2 still running

    # ── Stage 2 loading banner (shown while rewrite/cover letter loads) ──
    if _is_stage1_only or st.session_state.stage2_running:
        st.markdown("""
        <div style="background:linear-gradient(90deg,#1a2e4a,#2563eb22);
             border:1px solid rgba(96,165,250,0.35);border-radius:12px;
             padding:0.9rem 1.25rem;margin-bottom:1rem;display:flex;
             align-items:center;gap:0.75rem;">
            <div style="font-size:1.3rem;">⚙️</div>
            <div>
                <div style="font-weight:700;color:#f8fafc;font-size:0.95rem;">
                    Resume Rewrite &amp; Cover Letter loading...
                </div>
                <div style="font-size:0.82rem;color:rgba(248,250,252,0.6);margin-top:0.2rem;">
                    ATS Score &amp; Resume Health are ready below.
                    Rewrite, Humanizer &amp; Cover Letter tabs will unlock shortly — stay on this page.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

    # ── LinkedIn Shareable Score Badge ───────────────────────
    # FIX: this card claims the resume was rewritten and a cover letter was
    # generated. Those only exist after Stage 2 finishes — showing this
    # during Stage 1 (while "Resume Rewrite & Cover Letter loading..." is
    # still up) made false claims and showed up before Detailed Scores.
    if not _is_stage1_only and not st.session_state.stage2_running:
        _candidate_name = report.get("resume_data", {}).get("name", "I")
        _ats_score      = scores.get("ats_score", "N/A")
        _jd_match       = scores.get("jd_match_score", "N/A")
        _rec_short      = rec_text.split("—")[0].strip() if "—" in rec_text else rec_text

        # Score color label for the badge
        if isinstance(overall, (int, float)):
            if overall >= 80:
                _score_color = "#15803d"
                _score_bg    = "#dcfce7"
                _score_label = "Strong Match"
            elif overall >= 60:
                _score_color = "#92400e"
                _score_bg    = "#fef9c3"
                _score_label = "Moderate Match"
            else:
                _score_color = "#dc2626"
                _score_bg    = "#fee2e2"
                _score_label = "Needs Work"
        else:
            _score_color = "#64748b"
            _score_bg    = "#f1f5f9"
            _score_label = "Analyzed"

        # Build the shareable LinkedIn post text
        _linkedin_post = (
            f"I just got my resume roasted by 13 AI agents on RoastCV \U0001f525\n\n"
            f"Here's what they found:\n"
            f"\U0001f3af Overall Score: {overall}/100 ({_score_label})\n"
            f"\U0001f916 ATS Compatibility: {_ats_score}/100\n"
            f"\U0001f4ca Job Match: {_jd_match}%\n"
            f"\U0001f4cb Verdict: {_rec_short}\n\n"
            f"It gave me a brutally honest review - highlighted gaps I didn't even know existed, "
            f"rewrote my resume, and generated a cover letter. All free.\n\n"
            f"Try it yourself:\nroastcv.in"
        )

        # LinkedIn share URL — only the site URL is passed (LinkedIn removed text pre-fill support)
        _linkedin_url = "https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Froastcv.in"

        # ── LinkedIn Share Card ───────────────────────────────────────────────────
        # Header (pure HTML — no textarea inside, so it renders cleanly)
        st.markdown(
            f"""
    <div style="
        background: linear-gradient(135deg, #0a66c2 0%, #004182 100%);
        border-radius: 16px 16px 0 0;
        padding: 1.25rem 1.75rem 0.75rem 1.75rem;
    ">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
             flex-wrap:wrap;gap:1rem;">
            <div>
                <div style="font-size:0.7rem;font-weight:700;letter-spacing:1.2px;
                     text-transform:uppercase;color:rgba(255,255,255,0.55);margin-bottom:0.35rem;">
                    Share on LinkedIn
                </div>
                <div style="font-size:1.05rem;font-weight:700;color:white;margin-bottom:0.2rem;">
                    Let your network know you're job-ready 💼
                </div>
                <div style="font-size:0.82rem;color:rgba(255,255,255,0.6);">
                    1. Edit the post below &nbsp;·&nbsp; 2. Click Copy &nbsp;·&nbsp; 3. Paste on LinkedIn
                </div>
            </div>
            <div style="background:{_score_bg};color:{_score_color};border-radius:12px;
                 padding:0.5rem 1rem;text-align:center;min-width:76px;flex-shrink:0;">
                <div style="font-size:1.8rem;font-weight:800;line-height:1;">{overall}</div>
                <div style="font-size:0.67rem;font-weight:700;opacity:0.75;">/ 100</div>
            </div>
        </div>
    </div>
    """,
            unsafe_allow_html=True,
        )

        # Editable post text — Streamlit native widget (no HTML textarea = no rendering bug)
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0a66c2,#004182);'
            'padding:0 1.75rem 0.5rem 1.75rem;">',
            unsafe_allow_html=True,
        )
        _edited_post = st.text_area(
            "Edit your LinkedIn post",
            value=_linkedin_post,
            height=180,
            key="li_post_edit",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Buttons row
        _btn_col1, _btn_col2, _btn_spacer = st.columns([1, 1, 2])
        with _btn_col1:
            if st.button("📋 Copy Post", key="li_copy_btn", use_container_width=True, type="primary"):
                st.session_state["li_copied"] = True
        with _btn_col2:
            st.link_button("🔗 Open LinkedIn", url=_linkedin_url, use_container_width=True)

        if st.session_state.get("li_copied"):
            st.success("✅ Post copied! Now open LinkedIn and paste it.")
            st.session_state["li_copied"] = False

        st.markdown("---")

    # ── Detailed sections — rendered based on sidebar nav_section ────────────
    # "overview" shows the score grid + download report (default after analysis)
    # All other keys show their specific section content only.

    if _nav in ("overview", "audit", "ats", "gap", "hr", "recruiter",
                "hiring_manager", "rewrite", "cover_letter", "interview"):

        # ── OVERVIEW: Score grid + downloads (always shown in overview) ──────
        if _nav == "overview":
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
            row1 = score_items[:4]
            row2 = score_items[4:]
            cols1 = st.columns(4)
            for i, (label, val, suffix) in enumerate(row1):
                with cols1[i]:
                    display = f"{val}{suffix}" if isinstance(val, (int, float)) else "N/A"
                    st.metric(label, display)
            if row2:
                cols2 = st.columns(len(row2))
                for i, (label, val, suffix) in enumerate(row2):
                    with cols2[i]:
                        display = f"{val}{suffix}" if isinstance(val, (int, float)) else "N/A"
                        st.metric(label, display)

            st.markdown("---")

            # ── Wall of Roasts share checkbox ───────────────────────────
            if not _is_stage1_only and not st.session_state.get("roast_saved"):
                _share = st.checkbox(
                    "🔥 Share my roast anonymously on the Wall of Roasts "
                    "(no personal info — only role, score & HR feedback)",
                    key="roast_share_checkbox",
                )
                if _share:
                    _saved = wall_of_roasts.add_roast(
                        resume_data=report.get("resume_data", {}),
                        hr_roast=report.get("hr_roast", {}),
                        overall_score=overall,
                    )
                    if _saved:
                        st.session_state["roast_saved"] = True
                        st.success("✅ Your roast has been added to the Wall!")
                    else:
                        st.warning("Couldn't save — HR feedback may be missing.")

            report_content = report.get("_report_content", {})
            dl1, dl2 = st.columns(2)
            if report_content.get("md"):
                dl1.download_button(
                    "📄 Download Full Report (Markdown)",
                    report_content["md"],
                    file_name="resume_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_md_report_overview",
                )
            if report_content.get("json"):
                dl2.download_button(
                    "📦 Download Full Report (JSON)",
                    report_content["json"],
                    file_name="resume_report.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_json_report_overview",
                )

        # ── RESUME AUDIT ─────────────────────────────────────────────────────
        if _nav == "audit":
            st.markdown("## 🔍 Resume Audit")
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

        # ── ATS REVIEW ───────────────────────────────────────────────────────
        if _nav == "ats":
            st.markdown("## 🤖 ATS Review")
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

        # ── GAP ANALYSIS ─────────────────────────────────────────────────────
        if _nav == "gap":
            st.markdown("## 📋 Gap Analysis")
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

        # ── HR ROAST ─────────────────────────────────────────────────────────
        if _nav == "hr":
            st.markdown("## 🔥 HR Roast")
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

        # ── RECRUITER REVIEW ─────────────────────────────────────────────────
        if _nav == "recruiter":
            st.markdown("## 👔 Recruiter Review")
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

        # ── HIRING MANAGER ───────────────────────────────────────────────────
        if _nav == "hiring_manager":
            st.markdown("## 🏢 Hiring Manager")
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

        # ── IMPROVED RESUME ──────────────────────────────────────────────────
        if _nav == "rewrite":
            st.markdown("## ✨ ATS Resume")
            humanized   = report.get("humanized_resume", {})
            rewritten   = report.get("rewritten_resume", "")
            resume_data = report.get("resume_data", {})

            # FIX: previously checked `if not humanized and not rewritten` —
            # but `humanized` can be a non-empty dict that's still missing
            # the "humanized_resume" text field (e.g. truncated/partial LLM
            # JSON). That made the whole section render blank with no
            # resume AND no warning. Now we resolve the actual text first,
            # then decide based on whether we ended up with real content.
            resume_text_out = ""
            if isinstance(humanized, dict) and humanized.get("humanized_resume"):
                candidate = humanized["humanized_resume"]
                # FIX: candidate can be an actual dict/list object (not a
                # stringified one) when the LLM returns structured JSON
                # instead of plain text. _looks_like_raw_dict() only catches
                # stringified dicts, so non-string types must be rejected
                # here too — otherwise a dict slips through and crashes
                # PDF/DOCX generation later with "'dict' object has no
                # attribute 'splitlines'".
                if isinstance(candidate, str) and not _looks_like_raw_dict(candidate):
                    resume_text_out = candidate

            if not resume_text_out and isinstance(rewritten, str) and rewritten and not _looks_like_raw_dict(rewritten):
                resume_text_out = rewritten

            # Last resort: agents returned malformed/raw structured data —
            # build a clean plain-text resume from resume_data instead of
            # ever showing the raw dict to the user.
            _used_fallback_format = False
            if not resume_text_out and resume_data:
                resume_text_out = _dict_to_plain_resume(resume_data)
                _used_fallback_format = bool(resume_text_out)

            if not resume_text_out:
                st.warning(
                    "Resume Rewrite/Humanizer agents failed to produce a usable resume. "
                    "Check the agent errors panel above for details, or try again."
                )
            else:
                if _used_fallback_format:
                    st.info(
                        "The AI rewrite/humanizer step returned malformed text, so this is "
                        "a plain version built directly from your original resume data instead."
                    )
                if humanized and not _used_fallback_format:
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

                st.markdown("---")
                st.markdown("#### ✨ ATS Resume")
                st.text_area(
                    "Improved Resume",
                    resume_text_out,
                    height=400,
                    key="resume_area",
                    label_visibility="collapsed",
                )

                candidate_name = (resume_data.get("name") or "candidate").replace(" ", "_")

                # FIX: PDF generation was crashing with
                # "Unknown template_id 'classic_blue'" — that id doesn't
                # exist. The 24 real templates are 4 styles (Classic, Band,
                # Minimal, Sidebar) x 6 colors (Charcoal, Forest, Maroon,
                # Navy, Slate, Teal). Let the user pick instead of guessing.
                _TEMPLATE_STYLES = ["Classic", "Band", "Minimal", "Sidebar"]
                _TEMPLATE_COLORS = ["Navy", "Charcoal", "Forest", "Maroon", "Slate", "Teal"]
                st.markdown("**Choose a resume template (24 available):**")
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    _style_choice = st.selectbox("Style", _TEMPLATE_STYLES, key="tpl_style")
                with tcol2:
                    _color_choice = st.selectbox("Color", _TEMPLATE_COLORS, key="tpl_color")
                chosen_id = f"{_style_choice.lower()}_{_color_choice.lower()}"

                st.markdown("**Download as:**")
                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("📄 Generate PDF", key="gen_pdf_btn", use_container_width=True):
                        try:
                            with temp_output_path(".pdf") as pdf_path:
                                generate_resume_pdf(
                                    humanized_text=resume_text_out,
                                    resume_data=resume_data,
                                    output_path=pdf_path,
                                    template_id=chosen_id,
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

        # ── COVER LETTER ─────────────────────────────────────────────────────
        if _nav == "cover_letter":
            st.markdown("## 📝 Cover Letter")
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

        # ── INTERVIEW PREP ───────────────────────────────────────────────────
        if _nav == "interview":
            st.markdown("## 🎯 Interview Preparation")
            prep = report.get("interview_prep", {})
            if not prep or not any(prep.values()):
                st.warning("Interview Coach agent failed.")
            else:
                category_config = {
                    "hr_questions":             ("👥 HR Questions",           "These assess your personality, motivation, and cultural fit."),
                    "technical_questions":      ("💻 Technical Questions",    "Be ready to explain your technical choices and trade-offs."),
                    "project_questions":        ("🚀 Project Deep-Dives",     "You'll be asked to walk through your projects in detail."),
                    "scenario_based_questions": ("🎯 Scenario / Behavioural", "Use the STAR method: Situation, Task, Action, Result."),
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

    # ── Footer ────────────────────────────────────────────────
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
            <span>© 2026 RoastCV — All rights reserved</span>
            <span style="opacity:0.5; margin: 0 0.75rem;">|</span>
            <span>Powered by AI · Built for job seekers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)