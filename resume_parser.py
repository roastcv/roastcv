"""
Extracts raw text from a resume file (.pdf or .docx).
"""

import os
import pdfplumber
import docx


def extract_text_from_pdf(path: str) -> str:
    text_chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
    except Exception as e:
        raise ValueError(
            f"Could not read the PDF file. It may be corrupted or password-protected. "
            f"Original error: {e}"
        )
    return "\n".join(text_chunks)


def extract_text_from_docx(path: str) -> str:
    try:
        document = docx.Document(path)
    except Exception as e:
        raise ValueError(
            f"Could not read the DOCX file. It may be corrupted or not a valid Word file. "
            f"Original error: {e}"
        )
    return "\n".join(para.text for para in document.paragraphs if para.text.strip())


def extract_resume_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext == ".docx":
        text = extract_text_from_docx(path)
    elif ext == ".doc":
        # FIX: python-docx can only read the modern .docx (XML) format, not
        # the legacy binary .doc format. Previously .doc was routed into
        # extract_text_from_docx(), which always failed there with a
        # confusing "corrupted or not a valid Word file" message even for
        # perfectly valid old-style .doc files. Tell the user clearly what
        # to do instead.
        raise ValueError(
            "The legacy .doc format (old Word 97-2003) is not supported. "
            "Please save/export your resume as .docx or .pdf and upload that instead."
        )
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Only .pdf and .docx are supported."
        )

    if not text.strip():
        raise ValueError(
            "Could not extract text from the resume. "
            "The file may be a scanned image-based PDF (OCR required). "
            "Please use a text-based PDF or DOCX file instead."
        )
    return text