"""
Patches Streamlit's own static index.html to inject:
  1. Google Analytics (GA4) — gtag.js
  2. Google AdSense — pagead2.googlesyndication.com

WHY THIS IS NEEDED:
Streamlit apps are rendered client-side via JavaScript/WebSocket — the raw
HTML the server returns on first request is just an empty shell. Anything
added via st.markdown(..., unsafe_allow_html=True) only appears in the DOM
AFTER JavaScript runs. Google's crawlers (AdSense verification + Analytics)
do NOT execute JavaScript — they only read the raw server HTML. This script
solves that by editing Streamlit's actual static/index.html so both tags
are present from the very first byte of HTML — no JS required.

RUN THIS AS PART OF YOUR BUILD STEP (every deploy), e.g. in Render's
Build Command:

    pip install -r requirements.txt && python patch_streamlit_html.py

It is safe to run multiple times — it checks if each tag is already
present before adding it again.
"""

import os
import sys

GA4_SCRIPT = """<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-FF6541C3YW"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-FF6541C3YW');
</script>"""

ADSENSE_SCRIPT = (
    '<script async src="https://pagead2.googlesyndication.com/pagead/js/'
    'adsbygoogle.js?client=ca-pub-7537620467950326" crossorigin="anonymous"></script>'
)

SEO_TAGS = """<meta name="description" content="Roast your resume with AI. Get ATS score, cover letter, rewrite suggestions &amp; interview prep — 100% free. Powered by 13 AI agents." />
<meta name="keywords" content="resume analyzer, ATS checker, AI resume review, cover letter generator, resume roast, free resume checker" />
<meta name="robots" content="index, follow" />
<meta property="og:title" content="RoastCV — Free AI Resume Analyzer &amp; ATS Checker" />
<meta property="og:description" content="Roast your resume with AI. Get ATS score, cover letter, rewrite suggestions &amp; interview prep — 100% free." />
<meta property="og:url" content="https://roastcv.in" />
<meta property="og:type" content="website" />
<meta property="og:image" content="https://roastcv.in/logo.png" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="RoastCV — Free AI Resume Analyzer &amp; ATS Checker" />
<meta name="twitter:description" content="Roast your resume with AI. Get ATS score, cover letter, rewrite &amp; interview prep — 100% free." />
<link rel="canonical" href="https://roastcv.in" />
<title>RoastCV — Free AI Resume Analyzer &amp; ATS Checker</title>"""


def main():
    try:
        import streamlit
    except ImportError:
        print("ERROR: streamlit is not installed yet — run this AFTER pip install.")
        sys.exit(1)

    streamlit_dir = os.path.dirname(os.path.abspath(streamlit.__file__))
    index_path = os.path.join(streamlit_dir, "static", "index.html")

    if not os.path.exists(index_path):
        print(f"ERROR: could not find {index_path} — Streamlit's static "
              f"folder structure may have changed in this version.")
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    if "</head>" not in html:
        print(f"ERROR: no </head> tag found in {index_path} — cannot patch safely.")
        sys.exit(1)

    changed = False

    # ── Inject SEO tags ───────────────────────────────────────
    if "roastcv — free ai resume" in html.lower():
        print("SEO tags already present — skipping.")
    else:
        html = html.replace("</head>", SEO_TAGS + "\n</head>")
        print("✓ SEO meta tags injected.")
        changed = True

    # ── Inject GA4 ────────────────────────────────────────────
    if "G-FF6541C3YW" in html:
        print("GA4 script already present — skipping.")
    else:
        html = html.replace("</head>", GA4_SCRIPT + "\n</head>")
        print("✓ GA4 script injected.")
        changed = True

    # ── Inject AdSense ────────────────────────────────────────
    if "googlesyndication" in html:
        print("AdSense script already present — skipping.")
    else:
        html = html.replace("</head>", ADSENSE_SCRIPT + "\n</head>")
        print("✓ AdSense script injected.")
        changed = True

    # ── Write back only if something changed ──────────────────
    if changed:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nDone — patched: {index_path}")
    else:
        print("\nNothing to do — both scripts already present.")


if __name__ == "__main__":
    main()
