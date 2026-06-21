"""
RESUME AI AGENT — Main Orchestrator
Runs all agents in sequence and generates the final report.

USAGE:
    python main.py --resume path/to/resume.pdf --jd-file path/to/jd.txt
    python main.py --resume path/to/resume.docx --jd-text "Paste job description here..."
"""

import os
import json
import glob
import argparse
from datetime import datetime
from typing import Callable, Optional

from config import OUTPUT_DIR
from resume_parser import extract_resume_text
import resume_reader
import resume_audit
import ats_review
import jd_analyzer
import gap_analysis
import hr_roast
import recruiter_review
import resume_rewrite
import humanizer
import hiring_manager
import interview_coach
import cover_letter
import final_decision

TOTAL_STEPS = 13
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep only the N most recent reports in the output folder (per format)
MAX_REPORTS_TO_KEEP = 10

agent_errors: dict = {}


def log(step: str):
    print(f"\n>>> {step}")


def safe_run(label: str, fn, *args, fallback=None):
    try:
        result = fn(*args)
        if result == {} or result == "":
            agent_errors[label] = "Agent returned empty result (possible JSON parse failure)"
            return fallback if fallback is not None else {}
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"  WARNING: {label} failed — {error_msg}")
        agent_errors[label] = error_msg
        return fallback if fallback is not None else {}


def _resolve_output_dir() -> str:
    if os.path.isabs(OUTPUT_DIR):
        return OUTPUT_DIR
    return os.path.join(PROJECT_ROOT, OUTPUT_DIR)


def _cleanup_old_reports(output_dir: str, keep: int = MAX_REPORTS_TO_KEEP):
    """
    FIX: Delete oldest reports when folder exceeds `keep` count.
    Removes both the .json and .md pair together.
    """
    json_files = sorted(glob.glob(os.path.join(output_dir, "report_*.json")))
    # If we have more than `keep`, delete the oldest ones
    to_delete = json_files[:-keep] if len(json_files) > keep else []
    for json_path in to_delete:
        md_path = json_path.replace(".json", ".md")
        try:
            os.remove(json_path)
            if os.path.exists(md_path):
                os.remove(md_path)
            print(f"  Cleaned up old report: {os.path.basename(json_path)}")
        except OSError:
            pass  # Don't crash if file is locked/missing


def run_pipeline(
    resume_path: str,
    jd_path: str = None,
    jd_text: str = None,
    company_name: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    save_to_disk: bool = True,
):
    """
    Args:
        save_to_disk: If True (CLI default), report JSON/MD files are written to
            OUTPUT_DIR on disk — useful when running locally for yourself.
            If False (used by the web app), NOTHING is written to disk. The
            report content is only ever held in memory (report["_report_content"]),
            so once the browser session/tab is closed, the data is gone —
            no resume data sits on the server after the request finishes.
    """
    global agent_errors
    agent_errors = {}

    if not jd_path and not jd_text:
        raise ValueError("Provide either a JD file path or JD text — one is required.")

    def report_progress(step: int, label: str):
        log(label)
        if progress_callback:
            try:
                progress_callback(step, TOTAL_STEPS, label)
            except Exception:
                pass

    report_progress(0, "Parsing resume...")
    resume_text = extract_resume_text(resume_path)

    if jd_path:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    report_progress(1, "Agent 1: Resume Reader")
    resume_data = safe_run("Resume Reader", resume_reader.run, resume_text)

    report_progress(2, "Agent 2: Resume Audit")
    audit = safe_run("Resume Audit", resume_audit.run, resume_text, resume_data)

    report_progress(3, "Agent 3: ATS Review")
    ats = safe_run("ATS Review", ats_review.run, resume_text)

    report_progress(4, "Agent 4: JD Analyzer")
    jd_data = safe_run("JD Analyzer", jd_analyzer.run, jd_text)

    report_progress(5, "Agent 5: Gap Analysis")
    gap = safe_run("Gap Analysis", gap_analysis.run, resume_data, jd_data)

    report_progress(6, "Agent 6: HR Roast")
    hr = safe_run("HR Roast", hr_roast.run, resume_data, jd_data)

    report_progress(7, "Agent 7: Recruiter Review")
    recruiter = safe_run("Recruiter Review", recruiter_review.run, resume_data, jd_data)

    report_progress(8, "Agent 8: Resume Rewrite")
    rewrite_source = resume_data if resume_data else {"raw_text": resume_text}
    rewritten_resume = safe_run(
        "Resume Rewrite", resume_rewrite.run, rewrite_source, audit, gap, fallback=""
    )

    report_progress(9, "Agent 9: Humanizer")
    humanizer_input = rewritten_resume if rewritten_resume.strip() else resume_text
    humanized = safe_run("Humanizer", humanizer.run, humanizer_input)

    report_progress(10, "Agent 10: Hiring Manager")
    hiring_mgr = safe_run("Hiring Manager", hiring_manager.run, resume_data, jd_data)

    report_progress(11, "Agent 11: Interview Coach")
    interview_prep = safe_run(
        "Interview Coach", interview_coach.run, resume_data, jd_data,
        fallback={"hr_questions": [], "technical_questions": [],
                  "project_questions": [], "scenario_based_questions": []}
    )

    report_progress(12, "Agent 12: Cover Letter")
    # FIX: Pass humanized resume text so cover letter sounds more natural
    humanized_text_for_cl = humanized.get("humanized_resume", "") if humanized else ""
    cover_letter_text = safe_run(
        "Cover Letter", cover_letter.run,
        resume_data, jd_data, company_name, humanized_text_for_cl,
        fallback=""
    )

    report_progress(13, "Agent 13: Final Decision")
    final = final_decision.run(audit, ats, gap, hr, recruiter, hiring_mgr, humanized)

    report = {
        "generated_at": datetime.now().isoformat(),
        "resume_data": resume_data,
        "audit": audit,
        "ats_review": ats,
        "jd_data": jd_data,
        "gap_analysis": gap,
        "hr_roast": hr,
        "recruiter_review": recruiter,
        "rewritten_resume": rewritten_resume,
        "humanized_resume": humanized,
        "hiring_manager": hiring_mgr,
        "interview_prep": interview_prep,
        "cover_letter": cover_letter_text,
        "final_decision": final,
        "_agent_errors": dict(agent_errors),
    }

    # PRIVACY: build the report content purely in memory first. This is always
    # available via report["_report_content"], regardless of save_to_disk —
    # the web app uses THIS for download buttons, never a path on disk.
    json_content = json.dumps(report, indent=2, ensure_ascii=False)
    md_content = build_markdown_report(report)
    report["_report_content"] = {"json": json_content, "md": md_content}
    report["_report_files"] = {}

    if save_to_disk:
        output_dir = _resolve_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        _cleanup_old_reports(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"report_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_content)

        md_path = json_path.replace(".json", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        report["_report_files"] = {"json": json_path, "md": md_path}
        log(f"Done! Reports saved:\n  - {json_path}\n  - {md_path}")
    else:
        log("Done! (save_to_disk=False — nothing written to disk, report kept in memory only)")

    if agent_errors:
        log(f"Agents with errors: {list(agent_errors.keys())}")
    return report


def build_markdown_report(report: dict) -> str:
    """Builds the Markdown report as a string — no disk I/O. Used both for the
    on-disk .md file (CLI mode) and for the in-memory download button (web mode)."""
    final = report["final_decision"]
    lines = [
        "# Resume Analysis Report",
        f"_Generated: {report['generated_at']}_\n",
        "## Final Scores",
    ]
    for key, value in final["scores"].items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

    lines.append(f"\n**Overall Score**: {final['overall_score']}/100")
    lines.append(f"\n**Final Recommendation**: {final['final_recommendation']}\n")

    meta = final.get("meta", {})
    lines.append(f"**Interview Probability**: {meta.get('interview_probability', '-')}")
    lines.append(f"\n**Shortlist Decision**: {meta.get('shortlist_decision', '-')}")
    lines.append(f"\n**Hiring Recommendation**: {meta.get('hiring_recommendation', '-')}\n")

    lines.append("## HR Roast Feedback")
    lines.append(report["hr_roast"].get("hr_feedback", ""))

    lines.append("\n## Recruiter Feedback")
    lines.append(report["recruiter_review"].get("recruiter_feedback", ""))

    lines.append("\n## Critical Issues (per HR)")
    for issue in report["hr_roast"].get("critical_issues", []):
        lines.append(f"- {issue}")

    lines.append("\n## Hiring Manager Notes")
    lines.append(report["hiring_manager"].get("team_fit_notes", ""))
    lines.append(report["hiring_manager"].get("practical_skill_assessment", ""))

    lines.append("\n## Humanized Resume (Final Improved Version)")
    lines.append(report["humanized_resume"].get("humanized_resume", ""))

    lines.append("\n## Cover Letter")
    lines.append(report["cover_letter"])

    lines.append("\n## Interview Preparation")
    for category, questions in report["interview_prep"].items():
        lines.append(f"\n### {category.replace('_', ' ').title()}")
        for q in questions:
            lines.append(f"- {q}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume AI Agent Pipeline")
    parser.add_argument("--resume", required=True, help="Resume file path (.pdf or .docx)")
    parser.add_argument("--jd-file", help="Job description text file path")
    parser.add_argument("--jd-text", help="Job description text passed directly")
    parser.add_argument("--company", default="", help="Company name (for cover letter)")
    args = parser.parse_args()

    # CLI usage: save_to_disk stays True (default) — you're running this on your
    # own machine, so saving local report files is expected and convenient.
    run_pipeline(args.resume, jd_path=args.jd_file, jd_text=args.jd_text, company_name=args.company)