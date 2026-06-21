"""
AGENT 13: FINAL DECISION
Combines all agent outputs and produces a final verdict.
Pure math on existing scores — no LLM calls (fast, free, deterministic).

FIX: Scores from failed agents (which return 0 as fallback) are excluded
from the average so they don't drag down the overall score incorrectly.
"""

_SCORE_KEYS = {
    "resume_health_score": ("audit", "health_score"),
    "ats_score": ("ats", "ats_score"),
    "jd_match_score": ("gap_analysis", "match_percentage"),
    "hr_score": ("hr_roast", "hr_score"),
    "recruiter_score": ("recruiter_review", "recruiter_score"),
    "hiring_manager_score": ("hiring_manager", "hiring_manager_score"),
    "humanization_score": ("humanizer_data", "humanization_score"),
}


def run(audit: dict, ats: dict, gap_analysis: dict, hr_roast: dict,
        recruiter_review: dict, hiring_manager: dict, humanizer_data: dict) -> dict:

    sources = {
        "audit": audit,
        "ats": ats,
        "gap_analysis": gap_analysis,
        "hr_roast": hr_roast,
        "recruiter_review": recruiter_review,
        "hiring_manager": hiring_manager,
        "humanizer_data": humanizer_data,
    }

    numeric_scores = {}
    missing_scores = []

    def _is_valid_score(raw) -> bool:
        # FIX: bool is a subclass of int in Python, so isinstance(True, (int, float))
        # is True. Without the explicit bool check, a stray `true`/`false` from a
        # malformed LLM JSON response would silently be treated as a score of 1 or 0
        # instead of being flagged as invalid.
        if isinstance(raw, bool):
            return False
        if not isinstance(raw, (int, float)):
            return False
        # FIX: scores are documented as 0-100 everywhere (see each agent's prompt).
        # An out-of-range value (LLM hallucinating e.g. 150, or -5) used to be
        # accepted as-is and would silently skew the overall average.
        return 0 <= raw <= 100

    for score_key, (source_name, field_name) in _SCORE_KEYS.items():
        source = sources[source_name]
        # FIX: was `source == {}` which fails for non-empty dicts with missing keys
        if not source:
            numeric_scores[score_key] = "N/A (agent failed)"
            missing_scores.append(score_key)
            continue

        raw = source.get(field_name)
        if raw is None:
            numeric_scores[score_key] = "N/A (agent failed)"
            missing_scores.append(score_key)
        elif not _is_valid_score(raw):
            numeric_scores[score_key] = "N/A"
            missing_scores.append(score_key)
        else:
            numeric_scores[score_key] = raw

    meta = {
        "interview_probability": recruiter_review.get("interview_probability", "Unknown"),
        "shortlist_decision": hr_roast.get("shortlist_decision", "Unknown"),
        "hiring_recommendation": hiring_manager.get("hiring_recommendation", ""),
    }

    valid_values = [v for v in numeric_scores.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    overall = round(sum(valid_values) / len(valid_values), 1) if valid_values else 0

    if missing_scores:
        recommendation_suffix = f" (Note: {len(missing_scores)} agent(s) failed — scores may be incomplete)"
    else:
        recommendation_suffix = ""

    if overall >= 80:
        recommendation = "Strong Match — Apply Now"
    elif overall >= 60:
        recommendation = "Moderate Match — Apply, but improvements recommended"
    elif overall >= 40:
        recommendation = "Weak Match — Improve Resume First"
    elif overall == 0 and missing_scores:
        recommendation = "Could not evaluate — all agents failed. Check your API key in .env"
    else:
        recommendation = "Not Ready — Major Improvements Needed Before Applying"

    return {
        "scores": numeric_scores,
        "meta": meta,
        "overall_score": overall,
        "final_recommendation": recommendation + recommendation_suffix,
        "failed_agents": missing_scores,
    }