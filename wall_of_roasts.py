"""
wall_of_roasts.py — Anonymous public roast storage.

Storage: JSON file (roasts.json) in the project root.
Data saved: role hint, experience, hr_score, hr_feedback excerpt, overall_score.
Data NOT saved: name, email, phone, LinkedIn, GitHub, resume text — nothing identifiable.

Thread-safe: file lock via threading.Lock() so concurrent Streamlit
sessions don't corrupt the JSON.
"""

import json
import os
import threading
import re
from datetime import datetime

ROASTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roasts.json")
MAX_ROASTS  = 100   # keep only latest N roasts
_lock       = threading.Lock()


def _load() -> list:
    if not os.path.exists(ROASTS_FILE):
        return []
    try:
        with open(ROASTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(roasts: list):
    with open(ROASTS_FILE, "w", encoding="utf-8") as f:
        json.dump(roasts, f, ensure_ascii=False, indent=2)


def _guess_role(resume_data: dict) -> str:
    """
    Guess a generic role label from resume_data — no real name is used.
    Falls back to 'Professional' if nothing useful found.
    """
    experience = resume_data.get("experience", [])
    if experience and isinstance(experience, list):
        role = experience[0].get("role", "") if experience[0] else ""
        if role and len(role) < 60:
            # Strip company name if present (e.g. "Engineer at Google" → "Engineer")
            role = re.split(r"\bat\b|\@", role, flags=re.IGNORECASE)[0].strip()
            return role or "Professional"
    skills = resume_data.get("skills", [])
    if skills:
        return "Tech Professional"
    return "Professional"


def _guess_experience(resume_data: dict) -> str:
    """Returns a rough experience label like '2 yrs exp' or 'Fresher'."""
    experience = resume_data.get("experience", [])
    if not experience:
        return "Fresher"
    edu = resume_data.get("education", [])
    # Count number of jobs as a proxy
    count = len([e for e in experience if isinstance(e, dict) and e.get("company")])
    if count == 0:
        return "Fresher"
    if count == 1:
        return "1 yr exp"
    return f"{count}+ yrs exp"


def _truncate(text: str, max_chars: int = 220) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def add_roast(resume_data: dict, hr_roast: dict, overall_score) -> bool:
    """
    Save an anonymized roast entry.
    Returns True on success, False on failure.
    """
    try:
        role       = _guess_role(resume_data)
        experience = _guess_experience(resume_data)
        hr_score   = hr_roast.get("hr_score", 0)
        feedback   = _truncate(hr_roast.get("hr_feedback", ""), 220)
        shortlist  = hr_roast.get("shortlist_decision", "")

        if not feedback:
            return False  # nothing useful to show

        entry = {
            "role":       role,
            "experience": experience,
            "hr_score":   hr_score if isinstance(hr_score, (int, float)) else 0,
            "overall":    overall_score if isinstance(overall_score, (int, float)) else 0,
            "feedback":   feedback,
            "shortlist":  shortlist,
            "date":       datetime.utcnow().strftime("%b %Y"),
        }

        with _lock:
            roasts = _load()
            roasts.append(entry)
            # Keep only the latest MAX_ROASTS
            if len(roasts) > MAX_ROASTS:
                roasts = roasts[-MAX_ROASTS:]
            _save(roasts)
        return True

    except Exception as e:
        print(f"  wall_of_roasts: failed to save — {e}")
        return False


def get_roasts(limit: int = 30) -> list:
    """Return latest `limit` roasts, newest first."""
    with _lock:
        roasts = _load()
    return list(reversed(roasts))[:limit]


def roast_count() -> int:
    with _lock:
        return len(_load())