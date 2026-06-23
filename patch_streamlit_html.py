"""
Patches Streamlit's own static index.html to inject Google Analytics
directly into <head>.

RUN THIS AS PART OF YOUR BUILD STEP (every deploy), e.g. in Render's
Build Command:

    pip install -r requirements.txt && python patch_streamlit_html.py

It is safe to run multiple times — it checks if the tag is already
present before adding it again.
"""

import os
import sys

ANALYTICS_SCRIPT = """<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-FF6541C3YW"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-FF6541C3YW');
</script>"""


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

    if "G-FF6541C3YW" in html:
        print(f"Analytics script already present in {index_path} — skipping.")
        return

    if "</head>" not in html:
        print(f"ERROR: no </head> tag found in {index_path} — cannot patch safely.")
        sys.exit(1)

    patched_html = html.replace("</head>", ANALYTICS_SCRIPT + "\n</head>")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(patched_html)

    print(f"Analytics script successfully injected into {index_path}")


if __name__ == "__main__":
    main()
