"""
Patches Streamlit's own static index.html to inject the Google AdSense
loader script directly into <head>.

WHY THIS IS NEEDED:
Streamlit apps are rendered client-side via JavaScript/WebSocket — the raw
HTML the server returns on first request is just an empty shell. Anything
added via st.markdown(..., unsafe_allow_html=True) only appears in the DOM
AFTER JavaScript runs and Streamlit's frontend renders it.

Google's AdSense site-verification crawler does NOT execute JavaScript —
it only reads the raw server HTML. So a script/meta tag injected via
st.markdown will never be visible to it, even though it's visible in the
browser. This script solves that by editing Streamlit's actual
static/index.html file (the real file the server sends on first load),
so the tag is present from the very first byte of HTML — no JS required.

RUN THIS AS PART OF YOUR BUILD STEP (every deploy), e.g. in Render's
Build Command:

    pip install -r requirements.txt && python patch_streamlit_html.py

It is safe to run multiple times — it checks if the tag is already
present before adding it again.
"""

import os
import sys

ADSENSE_SCRIPT = (
    '<script async src="https://pagead2.googlesyndication.com/pagead/js/'
    'adsbygoogle.js?client=ca-pub-7537620467950326" crossorigin="anonymous"></script>'
)


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

    if "googlesyndication" in html:
        print(f"AdSense script already present in {index_path} — skipping.")
        return

    if "</head>" not in html:
        print(f"ERROR: no </head> tag found in {index_path} — cannot patch safely.")
        sys.exit(1)

    patched_html = html.replace("</head>", ADSENSE_SCRIPT + "\n</head>")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(patched_html)

    print(f"AdSense script successfully injected into {index_path}")


if __name__ == "__main__":
    main()
