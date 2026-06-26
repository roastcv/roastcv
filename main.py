"""
RESUME AI AGENT — Main Orchestrator (PARALLEL VERSION)
Agents jo ek dusre pe depend nahi karte, unhe ek saath chalaya jata hai.
Wait-time ~60% kam ho jaata hai compared to sequential version.

PARALLEL EXECUTION STAGES:
  Stage 1 (parallel): Resume Reader + JD Analyzer
  Stage 2 (parallel): Resume Audit, ATS Review, Gap Analysis,
                      HR Roast, Recruiter Review, Hiring Manager, Interview Coach
  Stage 3 (chain):    Resume Rewrite → Humanizer → Cover Letter
  Stage 4:            Final Decision

TWO-STAGE UI LOADING (web app):
  run_pipeline_stage1() → Fast agents only (~20 sec)
                          Returns: ATS score, Health score, Gap, HR, Recruiter,
                                   Hiring Manager, Interview Coach + partial final decision
  run_pipeline_stage2() → Slow chain (Rewrite → Humanizer → Cover Letter → Final Decision)
                          Returns: complete report with rewritten resume + cover letter

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
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

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

TOTAL_STEPS = 7   # Now 7 stages instead of 13 sequential steps
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_REPORTS_TO_KEEP = 10

# Max workers for parallel stage — 7 agents launched but llm_client.py's
# MAX_CONCURRENT=2 semaphore throttles actual API calls for Gemini free tier.
# Keeping workers=7 so threads are ready; the semaphore controls real concurrency.
_STAGE2_WORKERS = 7

agent_errors: dict = {}


def log(step: str):
    print(f"\n>>> {step}")


def safe_run(label: str, fn, *args, fallback=None):
    """Run an agent safely — catch errors, store them, return fallback."""
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
    """Delete oldest reports when folder exceeds `keep` count."""
    json_files = sorted(glob.glob(os.path.join(output_dir, "report_*.json")))
    to_delete = json_files[:-keep] if len(json_files) > keep else []
    for json_path in to_delete:
        md_path = json_path.replace(".json", ".md")
        try:
            os.remove(json_path)
            if os.path.exists(md_path):
                os.remove(md_path)
            print(f"  Cleaned up old report: {os.path.basename(json_path)}")
        except OSError:
            pass


def _run_parallel(tasks: dict, max_workers: int) -> dict:
    """
    Run multiple agents in parallel using ThreadPoolExecutor.

    Args:
        tasks: { "label": (fn, *args, fallback=None) }
               Each value is a tuple: (function, arg1, arg2, ...) 
               Optionally last item can be {"_fallback": value}
    Returns:
        { "label": result }
    """
    results = {}
    fallbacks = {}

    # Extract fallbacks from tasks if provided
    clean_tasks = {}
    for label, task in tasks.items():
        if task and isinstance(task[-1], dict) and "_fallback" in task[-1]:
            fallbacks[label] = task[-1]["_fallback"]
            clean_tasks[label] = task[:-1]
        else:
            fallbacks[label] = None
            clean_tasks[label] = task

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(safe_run, label, *task, fallback=fallbacks.get(label)): label
            for label, task in clean_tasks.items()
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                results[label] = future.result()
            except Exception as e:
                print(f"  WARNING: {label} future failed — {e}")
                agent_errors[label] = str(e)
                results[label] = fallbacks.get(label) or {}

    return results


def run_pipeline(
    resume_path: str,
    jd_path: str = None,
    jd_text: str = None,
    company_name: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    save_to_disk: bool = True,
):
    """
    Parallel pipeline — runs independent agents concurrently.

    Args:
        save_to_disk: If True (CLI default), saves JSON/MD to OUTPUT_DIR.
                      If False (web app), keeps everything in memory only.
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

    # ── STAGE 0: Parse files (no LLM, fast) ──────────────────
    report_progress(0, "Parsing resume & JD files...")
    resume_text = extract_resume_text(resume_path)
    if jd_path:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()

    # ── STAGE 1: Resume Reader + JD Analyzer (parallel) ──────
    # These two are fully independent — run them together
    report_progress(1, "Stage 1 (parallel): Resume Reader + JD Analyzer...")
    stage1 = _run_parallel(
        {
            "Resume Reader": (resume_reader.run, resume_text),
            "JD Analyzer":   (jd_analyzer.run, jd_text),
        },
        max_workers=2,
    )
    resume_data = stage1.get("Resume Reader") or {}
    jd_data     = stage1.get("JD Analyzer") or {}

    # ── STAGE 2: All analysis agents (parallel) ───────────────
    # These all depend on resume_data + jd_data but not on each other
    report_progress(2, "Stage 2 (parallel): Audit, ATS, Gap, HR, Recruiter, Hiring Manager, Interview Coach...")
    stage2 = _run_parallel(
        {
            "Resume Audit":    (resume_audit.run, resume_text, resume_data),
            "ATS Review":      (ats_review.run, resume_text),
            "Gap Analysis":    (gap_analysis.run, resume_data, jd_data),
            "HR Roast":        (hr_roast.run, resume_data, jd_data),
            "Recruiter Review":(recruiter_review.run, resume_data, jd_data),
            "Hiring Manager":  (hiring_manager.run, resume_data, jd_data),
            "Interview Coach": (
                interview_coach.run, resume_data, jd_data,
                {
                    "_fallback": {
                        "hr_questions": [], "technical_questions": [],
                        "project_questions": [], "scenario_based_questions": [],
                    }
                }
            ),
        },
        max_workers=_STAGE2_WORKERS,
    )
    audit          = stage2.get("Resume Audit") or {}
    ats            = stage2.get("ATS Review") or {}
    gap            = stage2.get("Gap Analysis") or {}
    hr             = stage2.get("HR Roast") or {}
    recruiter      = stage2.get("Recruiter Review") or {}
    hiring_mgr     = stage2.get("Hiring Manager") or {}
    interview_prep = stage2.get("Interview Coach") or {
        "hr_questions": [], "technical_questions": [],
        "project_questions": [], "scenario_based_questions": [],
    }

    # ── STAGE 3: Rewrite chain (sequential — each depends on previous) ──
    report_progress(3, "Stage 3: Resume Rewrite...")
    rewrite_source = resume_data if resume_data else {"raw_text": resume_text}
    rewritten_resume = safe_run(
        "Resume Rewrite", resume_rewrite.run, rewrite_source, audit, gap, fallback=""
    )

    report_progress(4, "Stage 4: Humanizer...")
    humanizer_input = rewritten_resume if rewritten_resume and rewritten_resume.strip() else resume_text
    humanized = safe_run("Humanizer", humanizer.run, humanizer_input, fallback={})

    report_progress(5, "Stage 5: Cover Letter...")
    humanized_raw = humanized.get("humanized_resume", "") if isinstance(humanized, dict) else ""
    humanized_text_for_cl = humanized_raw if isinstance(humanized_raw, str) else ""
    cover_letter_text = safe_run(
        "Cover Letter", cover_letter.run,
        resume_data, jd_data, company_name, humanized_text_for_cl,
        fallback=""
    )

    # ── STAGE 4: Final Decision (pure math, no LLM) ───────────
    report_progress(6, "Stage 6: Final Decision...")
    final = final_decision.run(audit, ats, gap, hr, recruiter, hiring_mgr, humanized)

    # ── Build report ──────────────────────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "resume_data":     resume_data,
        "audit":           audit,
        "ats_review":      ats,
        "jd_data":         jd_data,
        "gap_analysis":    gap,
        "hr_roast":        hr,
        "recruiter_review":recruiter,
        "rewritten_resume":rewritten_resume,
        "humanized_resume":humanized,
        "hiring_manager":  hiring_mgr,
        "interview_prep":  interview_prep,
        "cover_letter":    cover_letter_text,
        "final_decision":  final,
        "_agent_errors":   dict(agent_errors),
    }

    # In-memory content (always built — web app uses this for downloads)
    json_content = json.dumps(report, indent=2, ensure_ascii=False)
    md_content   = build_markdown_report(report)
    report["_report_content"] = {"json": json_content, "md": md_content}
    report["_report_files"]   = {}

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
        log("Done! (save_to_disk=False — nothing written to disk)")

    if agent_errors:
        log(f"Agents with errors: {list(agent_errors.keys())}")

    report_progress(7, "All done!")
    return report


def build_markdown_report(report: dict) -> str:
    """Builds the Markdown report as a string — no disk I/O."""
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
    lines.append(str(report["hr_roast"].get("hr_feedback", "") or ""))

    lines.append("\n## Recruiter Feedback")
    lines.append(str(report["recruiter_review"].get("recruiter_feedback", "") or ""))

    lines.append("\n## Critical Issues (per HR)")
    for issue in report["hr_roast"].get("critical_issues", []):
        lines.append(f"- {issue}")

    lines.append("\n## Hiring Manager Notes")
    lines.append(str(report["hiring_manager"].get("team_fit_notes", "") or ""))
    lines.append(str(report["hiring_manager"].get("practical_skill_assessment", "") or ""))

    lines.append("\n## Humanized Resume (Final Improved Version)")
    lines.append(str(report["humanized_resume"].get("humanized_resume", "") or ""))

    lines.append("\n## Cover Letter")
    lines.append(str(report["cover_letter"] or ""))

    lines.append("\n## Interview Preparation")
    for category, questions in report["interview_prep"].items():
        lines.append(f"\n### {category.replace('_', ' ').title()}")
        if isinstance(questions, list):
            for q in questions:
                lines.append(f"- {q}")
        elif questions:
            lines.append(f"- {questions}")

    return "\n".join(lines)


# ── TWO-STAGE PIPELINE (for web app Two-Stage UI Loading) ────────────────────

def run_pipeline_stage1(
    resume_path: str,
    jd_text: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    STAGE 1 — Fast agents only (~15-25 seconds).
    Runs: Resume Reader + JD Analyzer + Audit + ATS + Gap + HR +
          Recruiter + Hiring Manager + Interview Coach (all parallel).
    Returns a partial report dict that app.py can show immediately.

    Does NOT run: Resume Rewrite, Humanizer, Cover Letter (those are slow).
    Final Decision is computed on partial scores so an overall score is
    visible before Stage 2 completes.
    """
    global agent_errors
    agent_errors = {}

    STAGE1_TOTAL = 3  # 3 progress steps for UI

    def report_progress(step: int, label: str):
        log(label)
        if progress_callback:
            try:
                progress_callback(step, STAGE1_TOTAL, label)
            except Exception:
                pass

    report_progress(0, "Stage1: Parsing resume & JD...")
    resume_text = extract_resume_text(resume_path)

    report_progress(1, "Stage1: Resume Reader + JD Analyzer (parallel)...")
    s1 = _run_parallel(
        {
            "Resume Reader": (resume_reader.run, resume_text),
            "JD Analyzer":   (jd_analyzer.run, jd_text),
        },
        max_workers=2,
    )
    resume_data = s1.get("Resume Reader") or {}
    jd_data     = s1.get("JD Analyzer") or {}

    report_progress(2, "Stage1: Audit + ATS + Gap + HR + Recruiter + Hiring Manager + Interview Coach (parallel)...")
    s2 = _run_parallel(
        {
            "Resume Audit":    (resume_audit.run, resume_text, resume_data),
            "ATS Review":      (ats_review.run, resume_text),
            "Gap Analysis":    (gap_analysis.run, resume_data, jd_data),
            "HR Roast":        (hr_roast.run, resume_data, jd_data),
            "Recruiter Review":(recruiter_review.run, resume_data, jd_data),
            "Hiring Manager":  (hiring_manager.run, resume_data, jd_data),
            "Interview Coach": (
                interview_coach.run, resume_data, jd_data,
                {
                    "_fallback": {
                        "hr_questions": [], "technical_questions": [],
                        "project_questions": [], "scenario_based_questions": [],
                    }
                }
            ),
        },
        max_workers=_STAGE2_WORKERS,
    )
    audit          = s2.get("Resume Audit") or {}
    ats            = s2.get("ATS Review") or {}
    gap            = s2.get("Gap Analysis") or {}
    hr             = s2.get("HR Roast") or {}
    recruiter      = s2.get("Recruiter Review") or {}
    hiring_mgr     = s2.get("Hiring Manager") or {}
    interview_prep = s2.get("Interview Coach") or {
        "hr_questions": [], "technical_questions": [],
        "project_questions": [], "scenario_based_questions": [],
    }

    # Partial final decision (Rewrite/Humanizer scores not available yet — excluded)
    partial_final = final_decision.run(
        audit, ats, gap, hr, recruiter, hiring_mgr,
        humanizer_data={},   # empty — Stage 2 not done yet
    )

    report_progress(3, "Stage1: Done!")

    return {
        "generated_at":       datetime.now().isoformat(),
        "resume_text":        resume_text,   # passed to Stage 2 so it re-parses nothing
        "resume_data":        resume_data,
        "jd_data":            jd_data,
        "audit":              audit,
        "ats_review":         ats,
        "gap_analysis":       gap,
        "hr_roast":           hr,
        "recruiter_review":   recruiter,
        "hiring_manager":     hiring_mgr,
        "interview_prep":     interview_prep,
        # Stage 2 placeholders — filled in later
        "rewritten_resume":   "",
        "humanized_resume":   {},
        "cover_letter":       "",
        "final_decision":     partial_final,
        "_stage":             1,             # sentinel: Stage 2 not complete yet
        "_agent_errors":      dict(agent_errors),
    }


def run_pipeline_stage2(
    stage1_report: dict,
    company_name: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    STAGE 2 — Slow chain (Resume Rewrite → Humanizer → Cover Letter).
    Takes the stage1_report dict returned by run_pipeline_stage1() and
    fills in the missing fields. Returns the complete report.

    Progress steps: 0-3 (Rewrite / Humanizer / Cover Letter / Final Decision).
    """
    global agent_errors
    # Merge existing stage1 errors so nothing is lost
    agent_errors = dict(stage1_report.get("_agent_errors", {}))

    STAGE2_TOTAL = 4

    def report_progress(step: int, label: str):
        log(label)
        if progress_callback:
            try:
                progress_callback(step, STAGE2_TOTAL, label)
            except Exception:
                pass

    resume_data  = stage1_report["resume_data"]
    jd_data      = stage1_report["jd_data"]
    resume_text  = stage1_report.get("resume_text", "")
    audit        = stage1_report["audit"]
    gap          = stage1_report["gap_analysis"]
    ats          = stage1_report["ats_review"]
    hr           = stage1_report["hr_roast"]
    recruiter    = stage1_report["recruiter_review"]
    hiring_mgr   = stage1_report["hiring_manager"]

    report_progress(0, "Stage2: Resume Rewrite...")
    rewrite_source   = resume_data if resume_data else {"raw_text": resume_text}
    rewritten_resume = safe_run(
        "Resume Rewrite", resume_rewrite.run, rewrite_source, audit, gap, fallback=""
    )

    report_progress(1, "Stage2: Humanizer...")
    humanizer_input = rewritten_resume if rewritten_resume and rewritten_resume.strip() else resume_text
    humanized = safe_run("Humanizer", humanizer.run, humanizer_input, fallback={})

    report_progress(2, "Stage2: Cover Letter...")
    humanized_raw = humanized.get("humanized_resume", "") if isinstance(humanized, dict) else ""
    humanized_text_for_cl = humanized_raw if isinstance(humanized_raw, str) else ""
    cover_letter_text = safe_run(
        "Cover Letter", cover_letter.run,
        resume_data, jd_data, company_name, humanized_text_for_cl,
        fallback=""
    )

    report_progress(3, "Stage2: Final Decision (complete)...")
    complete_final = final_decision.run(audit, ats, gap, hr, recruiter, hiring_mgr, humanized)

    # Build complete report by merging stage1 + stage2 results
    complete_report = {
        **stage1_report,
        "rewritten_resume": rewritten_resume,
        "humanized_resume": humanized,
        "cover_letter":     cover_letter_text,
        "final_decision":   complete_final,
        "_stage":           2,
        "_agent_errors":    dict(agent_errors),
    }

    # Build in-memory content for downloads
    json_content = json.dumps(complete_report, indent=2, ensure_ascii=False)
    md_content   = build_markdown_report(complete_report)
    complete_report["_report_content"] = {"json": json_content, "md": md_content}
    complete_report["_report_files"]   = {}

    log("Stage2 done!")
    return complete_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume AI Agent Pipeline (Parallel)")
    parser.add_argument("--resume",   required=True, help="Resume file path (.pdf or .docx)")
    parser.add_argument("--jd-file",  help="Job description text file path")
    parser.add_argument("--jd-text",  help="Job description text passed directly")
    parser.add_argument("--company",  default="", help="Company name (for cover letter)")
    args = parser.parse_args()

    run_pipeline(
        args.resume,
        jd_path=args.jd_file,
        jd_text=args.jd_text,
        company_name=args.company,
    )
