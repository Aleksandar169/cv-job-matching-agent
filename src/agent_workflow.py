from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, START, END
except ImportError as exc:
    raise ImportError(
        "LangGraph is not installed."
    ) from exc

from src.utils import (
    AGENT_WORKFLOW_OUTPUT_DIR,
    create_run_output_directory,
    save_json_file,
)

from src.pdf_extraction import process_cv_pdf
from src.cv_quality import process_cv_quality_analysis
from src.cv_extraction import process_structured_cv_extraction
from src.job_extraction import extract_structured_job
from src.job_storage import (
    get_jobs_collection,
    create_job_key,
    find_job_by_key,
    save_or_update_analyzed_job,
)
from src.matching import process_matching
from src.recommendations import process_recommendations
from src.final_report import process_final_report


class CVJobMatchingAgentState(TypedDict, total=False):
    cv_pdf_path: str
    job_inputs: List[Dict[str, Any]]

    user_wants_cv_recommendations: bool
    user_wants_job_matching: bool
    user_wants_job_recommendations: bool
    user_wants_detailed_report: bool
    selected_job_id: Optional[str]

    cv_quality_weights: Optional[Dict[str, float]]
    direct_matching_weights: Optional[Dict[str, float]]
    matching_mode_weights: Optional[Dict[str, float]]

    use_mongodb: bool
    model_name: str
    temperature: float

    run_id: str
    run_output_dir: str

    cv_processing_result: Dict[str, Any]
    cv_text: str
    cv_quality_result: Dict[str, Any]
    cv_quality_analysis: Dict[str, Any]
    structured_cv_result: Dict[str, Any]
    structured_cv: Dict[str, Any]

    cv_digest: Dict[str, Any]
    cv_recommendations_digest: Dict[str, Any]

    job_results: List[Dict[str, Any]]
    ranked_matches: List[Dict[str, Any]]
    matching_digest: Dict[str, Any]
    selected_job_result: Dict[str, Any]
    selected_job_recommendations_digest: Dict[str, Any]
    selected_job_final_report_summary: Dict[str, Any]

    workflow_log: List[str]
    errors: List[str]
    status: str
    started_at: str
    completed_at: str


def get_current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_workflow_log(state: Dict[str, Any], message: str) -> None:
    timestamp = get_current_timestamp()
    state.setdefault("workflow_log", []).append(f"[{timestamp}] {message}")


def truncate_text(text: Optional[str], max_length: int = 800) -> str:
    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."


def create_job_inputs_from_texts(job_texts: List[str]) -> List[Dict[str, Any]]:
    job_inputs = []

    for index, job_text in enumerate(job_texts, start=1):
        if job_text is None or not str(job_text).strip():
            continue

        job_inputs.append(
            {
                "job_id": f"job_{index:03d}",
                "job_text": job_text,
                "source": "streamlit_user_input",
            }
        )

    return job_inputs


def get_score_value(value: Any) -> float:
    if value is None:
        return -1

    try:
        return float(value)
    except (TypeError, ValueError):
        return -1


def create_markdown_list(
    items: Optional[List[Any]],
    empty_message: str = "- No items available.",
) -> str:
    if not items:
        return empty_message

    lines = []

    for item in items:
        if isinstance(item, dict):
            display_value = (
                item.get("job_requirement")
                or item.get("skill")
                or item.get("title")
                or item.get("action_title")
                or str(item)
            )
        else:
            display_value = str(item)

        lines.append(f"- {display_value}")

    return "\n".join(lines)


def build_cv_digest(
    cv_text: str,
    cv_quality_analysis: Dict[str, Any],
    structured_cv: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "candidate_name": structured_cv.get("candidate_name"),
        "email": structured_cv.get("email"),
        "phone": structured_cv.get("phone"),
        "location": structured_cv.get("location"),
        "linkedin_url": structured_cv.get("linkedin_url"),
        "github_url": structured_cv.get("github_url"),
        "portfolio_url": structured_cv.get("portfolio_url"),
        "profile_summary": structured_cv.get("profile_summary"),
        "total_years_of_experience": structured_cv.get("total_years_of_experience"),
        "final_cv_quality_score": cv_quality_analysis.get("final_cv_quality_score"),
        "cv_quality_category": cv_quality_analysis.get("cv_quality_category"),
        "overall_summary": cv_quality_analysis.get("overall_summary"),
        "top_strengths": cv_quality_analysis.get("strengths", [])[:5],
        "top_weaknesses": cv_quality_analysis.get("weaknesses", [])[:5],
        "missing_or_unclear_sections": cv_quality_analysis.get("missing_or_unclear_sections", [])[:5],
        "cv_quality_scores": cv_quality_analysis.get("scores", {}),
        "cv_quality_weights_used": cv_quality_analysis.get("cv_quality_weights_used", {}),
        "cv_text_preview": truncate_text(cv_text, max_length=800),
    }


def build_cv_recommendations_digest(
    cv_quality_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source": "cv_quality_analysis",
        "recommendations": cv_quality_analysis.get("cv_improvement_recommendations", []),
        "weaknesses_used_as_evidence": cv_quality_analysis.get("weaknesses", []),
        "missing_or_unclear_sections": cv_quality_analysis.get("missing_or_unclear_sections", []),
    }


def extract_or_load_structured_job_for_workflow(
    job_text: str,
    use_mongodb: bool = True,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    if job_text is None or not str(job_text).strip():
        raise ValueError("Job text is empty. Cannot process job posting.")

    job_key = create_job_key(job_text)

    if not use_mongodb:
        structured_job = extract_structured_job(
            job_text=job_text,
            job_key=job_key,
            source=source,
            source_file=source_file,
            source_row_index=source_row_index,
            model_name=model_name,
            temperature=temperature,
        )

        return {
            "structured_job": structured_job,
            "job_key": job_key,
            "loaded_from_database": False,
            "storage_summary": None,
        }

    jobs_collection = get_jobs_collection()

    existing_job = find_job_by_key(
        job_key=job_key,
        jobs_collection=jobs_collection,
    )

    if existing_job is not None:
        structured_job = existing_job.get("structured_job", {})

        storage_result = save_or_update_analyzed_job(
            job_text=job_text,
            structured_job=structured_job,
            jobs_collection=jobs_collection,
            source=source,
            source_file=source_file,
            source_row_index=source_row_index,
            update_structured_job_on_duplicate=False,
        )

        return {
            "structured_job": structured_job,
            "job_key": job_key,
            "loaded_from_database": True,
            "storage_summary": {
                "inserted": storage_result.get("inserted"),
                "updated_existing": storage_result.get("updated_existing"),
                "submission_count": storage_result.get("submission_count"),
            },
        }

    structured_job = extract_structured_job(
        job_text=job_text,
        job_key=job_key,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
        model_name=model_name,
        temperature=temperature,
    )

    storage_result = save_or_update_analyzed_job(
        job_text=job_text,
        structured_job=structured_job,
        jobs_collection=jobs_collection,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
        update_structured_job_on_duplicate=False,
    )

    return {
        "structured_job": structured_job,
        "job_key": job_key,
        "loaded_from_database": False,
        "storage_summary": {
            "inserted": storage_result.get("inserted"),
            "updated_existing": storage_result.get("updated_existing"),
            "submission_count": storage_result.get("submission_count"),
        },
    }


def build_matching_digest(
    ranked_matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not ranked_matches:
        return {
            "number_of_jobs_analyzed": 0,
            "best_match": None,
            "ranked_jobs": [],
        }

    ranked_jobs = []

    for rank, job_result in enumerate(ranked_matches, start=1):
        job_information = job_result.get("job_information", {})
        matching_result = job_result.get("matching_result", {})
        matched_items = matching_result.get("matched_items", {})
        missing_items = matching_result.get("missing_items", {})
        final_result = matching_result.get("final_result", {})

        ranked_jobs.append(
            {
                "rank": rank,
                "job_id": job_result.get("job_id"),
                "job_title": job_information.get("job_title"),
                "company_name": job_information.get("company_name"),
                "job_category": job_information.get("job_category"),
                "location": job_information.get("location"),
                "work_mode": job_information.get("work_mode"),
                "employment_type": job_information.get("employment_type"),
                "final_hybrid_score": final_result.get("final_hybrid_score"),
                "direct_matching_score": final_result.get("direct_matching_score"),
                "semantic_score": final_result.get("semantic_score"),
                "match_category": final_result.get("match_category"),
                "top_matched_required_skills": matched_items.get("matched_required_skills", [])[:5],
                "top_missing_required_skills": missing_items.get("missing_required_skills", [])[:5],
            }
        )

    return {
        "number_of_jobs_analyzed": len(ranked_matches),
        "best_match": ranked_jobs[0],
        "ranked_jobs": ranked_jobs,
    }


def build_selected_job_recommendations_digest(
    job_result: Dict[str, Any],
) -> Dict[str, Any]:
    recommendation_output = job_result.get("recommendation_output")

    if not recommendation_output:
        return {
            "job_id": job_result.get("job_id"),
            "available": False,
            "message": "Recommendations are not available for this selected CV-job pair.",
        }

    recommendations = recommendation_output.get("recommendations", recommendation_output)

    return {
        "job_id": job_result.get("job_id"),
        "available": True,
        "overall_recommendation_summary": recommendations.get("overall_recommendation_summary"),
        "priority_actions": recommendations.get("priority_actions", []),
        "cv_improvement_recommendations_count": len(
            recommendations.get("cv_improvement_recommendations", [])
        ),
        "missing_required_skills_recommendations_count": len(
            recommendations.get("missing_required_skills_recommendations", [])
        ),
        "technical_development_recommendations_count": len(
            recommendations.get("technical_development_recommendations", [])
        ),
        "project_recommendations_count": len(
            recommendations.get("project_recommendations", [])
        ),
        "soft_skills_recommendations_count": len(
            recommendations.get("soft_skills_recommendations", [])
        ),
        "methodology_notes": recommendations.get("methodology_notes", []),
    }


def build_selected_final_report_summary(
    job_result: Dict[str, Any],
) -> Dict[str, Any]:
    final_report = job_result.get("final_report")
    final_report_markdown = job_result.get("final_report_markdown")

    if not final_report and not final_report_markdown:
        return {
            "job_id": job_result.get("job_id"),
            "available": False,
            "message": "Final report is not available for this selected CV-job pair.",
        }

    return {
        "job_id": job_result.get("job_id"),
        "available": True,
        "final_report_json_available": final_report is not None,
        "final_report_markdown_available": final_report_markdown is not None,
        "markdown_preview": truncate_text(final_report_markdown, max_length=1500)
        if final_report_markdown
        else None,
        "final_report": final_report,
    }


def get_job_information_from_structured_job(
    structured_job: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "job_title": structured_job.get("job_title"),
        "company_name": structured_job.get("company_name"),
        "job_category": structured_job.get("job_category"),
        "location": structured_job.get("location"),
        "work_mode": structured_job.get("work_mode"),
        "employment_type": structured_job.get("employment_type"),
    }


def initialize_agent_state_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    state["started_at"] = get_current_timestamp()
    state["status"] = "running"
    state["workflow_log"] = []
    state["errors"] = []

    run_output_dir = create_run_output_directory(state.get("run_id"))
    state["run_output_dir"] = str(run_output_dir)
    state["run_id"] = run_output_dir.name

    append_workflow_log(state, "Agent workflow initialized.")
    return state


def process_cv_pdf_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    run_output_dir = Path(state["run_output_dir"])
    output_path = run_output_dir / "cv_extraction" / "cv_text.txt"

    cv_processing_result = process_cv_pdf(
        pdf_path=state["cv_pdf_path"],
        output_path=output_path,
    )

    state["cv_processing_result"] = cv_processing_result
    state["cv_text"] = cv_processing_result["cv_text"]

    append_workflow_log(state, "Processed CV PDF and extracted CV text.")
    return state


def analyze_cv_quality_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    run_output_dir = Path(state["run_output_dir"])

    cv_quality_result = process_cv_quality_analysis(
        cv_text=state["cv_text"],
        cv_quality_weights=state.get("cv_quality_weights"),
        model_name=state.get("model_name", "gpt-4o-mini"),
        temperature=state.get("temperature", 0),
        json_output_path=run_output_dir / "cv_quality" / "cv_quality_analysis.json",
        markdown_output_path=run_output_dir / "cv_quality" / "cv_quality_report.md",
    )

    state["cv_quality_result"] = cv_quality_result
    state["cv_quality_analysis"] = cv_quality_result["cv_quality_analysis"]

    append_workflow_log(state, "Completed CV quality analysis.")
    return state


def extract_structured_cv_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    run_output_dir = Path(state["run_output_dir"])

    structured_cv_result = process_structured_cv_extraction(
        cv_text=state["cv_text"],
        source_file=state.get("cv_pdf_path"),
        model_name=state.get("model_name", "gpt-4o-mini"),
        temperature=state.get("temperature", 0),
        output_path=run_output_dir / "cv_extraction" / "structured_cv.json",
    )

    state["structured_cv_result"] = structured_cv_result
    state["structured_cv"] = structured_cv_result["structured_cv"]

    append_workflow_log(state, "Extracted structured CV information.")
    return state


def create_cv_digest_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    state["cv_digest"] = build_cv_digest(
        cv_text=state["cv_text"],
        cv_quality_analysis=state["cv_quality_analysis"],
        structured_cv=state["structured_cv"],
    )

    append_workflow_log(state, "Created CV digest.")
    return state


def create_cv_recommendations_digest_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    state["cv_recommendations_digest"] = build_cv_recommendations_digest(
        cv_quality_analysis=state["cv_quality_analysis"],
    )

    append_workflow_log(state, "Created CV recommendations digest from CV quality analysis.")
    return state


def process_job_inputs_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    job_results = []
    job_inputs = state.get("job_inputs", [])

    run_output_dir = Path(state["run_output_dir"])

    if not job_inputs:
        state["job_results"] = []
        append_workflow_log(state, "No job inputs provided.")
        return state

    for index, job_input in enumerate(job_inputs, start=1):
        job_id = job_input.get("job_id") or f"job_{index:03d}"
        job_text = job_input.get("job_text")

        if job_text is None or not str(job_text).strip():
            state.setdefault("errors", []).append(f"{job_id}: empty job text.")
            continue

        job_output_dir = run_output_dir / "jobs" / job_id

        source = job_input.get("source", "streamlit_user_input")
        source_file = job_input.get("source_file")
        source_row_index = job_input.get("source_row_index")

        job_extraction_result = extract_or_load_structured_job_for_workflow(
            job_text=job_text,
            use_mongodb=state.get("use_mongodb", True),
            source=source,
            source_file=source_file,
            source_row_index=source_row_index,
            model_name=state.get("model_name", "gpt-4o-mini"),
            temperature=state.get("temperature", 0),
        )

        structured_job = job_extraction_result["structured_job"]

        structured_job_output_path = job_output_dir / "job_extraction" / "structured_job.json"

        save_json_file(
            data=structured_job,
            file_path=structured_job_output_path,
        )

        matching_result = None
        matching_markdown_report = None
        recommendation_output = None
        recommendations_markdown_report = None
        final_report = None
        final_report_markdown = None

        needs_matching = (
            state.get("user_wants_job_matching")
            or state.get("user_wants_job_recommendations")
            or state.get("user_wants_detailed_report")
        )

        if needs_matching:
            matching_output = process_matching(
                structured_cv=state["structured_cv"],
                structured_job=structured_job,
                direct_matching_weights=state.get("direct_matching_weights"),
                matching_mode_weights=state.get("matching_mode_weights"),
                model_name=state.get("model_name", "gpt-4o-mini"),
                temperature=state.get("temperature", 0),
                cv_source=state.get("cv_pdf_path"),
                job_source=job_id,
                json_output_path=job_output_dir / "matching" / "matching_result.json",
                markdown_output_path=job_output_dir / "matching" / "matching_report.md",
            )

            matching_result = matching_output["matching_result"]
            matching_markdown_report = matching_output["markdown_report"]

        needs_recommendations = (
            state.get("user_wants_job_recommendations")
            or state.get("user_wants_detailed_report")
        )

        if needs_recommendations and matching_result is not None:
            recommendation_result = process_recommendations(
                structured_cv=state["structured_cv"],
                structured_job=structured_job,
                cv_quality_analysis=state["cv_quality_analysis"],
                matching_result=matching_result,
                model_name=state.get("model_name", "gpt-4o-mini"),
                temperature=state.get("temperature", 0),
                cv_source=state.get("cv_pdf_path"),
                job_source=job_id,
                cv_quality_source=str(run_output_dir / "cv_quality" / "cv_quality_analysis.json"),
                matching_source=str(job_output_dir / "matching" / "matching_result.json"),
                json_output_path=job_output_dir / "recommendations" / "recommendations.json",
                markdown_output_path=job_output_dir / "recommendations" / "recommendations_report.md",
            )

            recommendation_output = recommendation_result["recommendation_output"]
            recommendations_markdown_report = recommendation_result["markdown_report"]

        if state.get("user_wants_detailed_report") and matching_result is not None:
            if recommendation_output is None:
                recommendation_output = {
                    "recommendations": {
                        "overall_recommendation_summary": None,
                        "cv_improvement_recommendations": [],
                        "missing_required_skills_recommendations": [],
                        "technical_development_recommendations": [],
                        "project_recommendations": [],
                        "soft_skills_recommendations": [],
                        "priority_actions": [],
                        "methodology_notes": [
                            "Recommendations were not generated before final report creation."
                        ],
                    }
                }

            final_report_result = process_final_report(
                cv_quality_analysis=state["cv_quality_analysis"],
                structured_cv=state["structured_cv"],
                structured_job=structured_job,
                matching_result=matching_result,
                recommendations_output=recommendation_output,
                source_files={
                    "cv_pdf": state.get("cv_pdf_path"),
                    "cv_text": state.get("cv_processing_result", {}).get("cv_text_path"),
                    "cv_quality": str(run_output_dir / "cv_quality" / "cv_quality_analysis.json"),
                    "structured_cv": str(run_output_dir / "cv_extraction" / "structured_cv.json"),
                    "structured_job": str(structured_job_output_path),
                    "matching_result": str(job_output_dir / "matching" / "matching_result.json"),
                    "recommendations": str(job_output_dir / "recommendations" / "recommendations.json"),
                },
                json_output_path=job_output_dir / "final_report" / "final_report.json",
                markdown_output_path=job_output_dir / "final_report" / "final_report.md",
            )

            final_report = final_report_result["final_report"]
            final_report_markdown = final_report_result["markdown_report"]

        job_information = get_job_information_from_structured_job(structured_job)

        final_result = {}

        if matching_result:
            final_result = matching_result.get("final_result", {})

        job_results.append(
            {
                "job_id": job_id,
                "job_key": job_extraction_result.get("job_key"),
                "loaded_from_database": job_extraction_result.get("loaded_from_database"),
                "storage_summary": job_extraction_result.get("storage_summary"),
                "job_information": job_information,
                "structured_job": structured_job,
                "structured_job_output_path": str(structured_job_output_path),
                "matching_result": matching_result,
                "matching_markdown_report": matching_markdown_report,
                "recommendation_output": recommendation_output,
                "recommendations_markdown_report": recommendations_markdown_report,
                "final_report": final_report,
                "final_report_markdown": final_report_markdown,
                "final_hybrid_score": final_result.get("final_hybrid_score"),
                "direct_matching_score": final_result.get("direct_matching_score"),
                "semantic_score": final_result.get("semantic_score"),
                "match_category": final_result.get("match_category"),
            }
        )

        append_workflow_log(state, f"Processed job input: {job_id}.")

    state["job_results"] = job_results

    append_workflow_log(state, f"Processed {len(job_results)} job posting(s).")
    return state


def rank_job_matches_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    job_results = state.get("job_results", [])

    ranked_matches = sorted(
        job_results,
        key=lambda item: get_score_value(item.get("final_hybrid_score")),
        reverse=True,
    )

    state["ranked_matches"] = ranked_matches

    append_workflow_log(state, "Ranked job postings by final hybrid matching score.")
    return state


def create_matching_digest_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    state["matching_digest"] = build_matching_digest(
        ranked_matches=state.get("ranked_matches", []),
    )

    append_workflow_log(state, "Created matching digest.")
    return state


def select_job_for_detail_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    ranked_matches = state.get("ranked_matches", [])

    if not ranked_matches:
        state["selected_job_result"] = {}
        append_workflow_log(state, "No ranked jobs available for selection.")
        return state

    requested_job_id = state.get("selected_job_id")

    if requested_job_id:
        selected_job = next(
            (
                job_result
                for job_result in ranked_matches
                if job_result.get("job_id") == requested_job_id
            ),
            None,
        )

        if selected_job is None:
            selected_job = ranked_matches[0]
            state["selected_job_id"] = selected_job.get("job_id")

            append_workflow_log(
                state,
                f"Requested selected_job_id was not found. Selected best ranked job: {selected_job.get('job_id')}.",
            )
        else:
            append_workflow_log(state, f"Selected requested job for details: {requested_job_id}.")
    else:
        selected_job = ranked_matches[0]
        state["selected_job_id"] = selected_job.get("job_id")

        append_workflow_log(
            state,
            f"Automatically selected best ranked job: {selected_job.get('job_id')}.",
        )

    state["selected_job_result"] = selected_job
    return state


def create_selected_job_recommendations_digest_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    selected_job_result = state.get("selected_job_result", {})

    state["selected_job_recommendations_digest"] = build_selected_job_recommendations_digest(
        job_result=selected_job_result,
    )

    append_workflow_log(state, "Prepared recommendation digest for selected job.")
    return state


def create_selected_final_report_summary_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    selected_job_result = state.get("selected_job_result", {})

    state["selected_job_final_report_summary"] = build_selected_final_report_summary(
        job_result=selected_job_result,
    )

    append_workflow_log(state, "Prepared final report summary for selected job.")
    return state


def save_agent_outputs_node(
    state: CVJobMatchingAgentState,
) -> CVJobMatchingAgentState:
    state["completed_at"] = get_current_timestamp()
    state["status"] = "completed"

    append_workflow_log(state, "Agent workflow completed.")

    run_output_dir = Path(state["run_output_dir"])
    agent_output_dir = run_output_dir / "agent_workflow"

    compact_job_results = []

    for job_result in state.get("job_results", []):
        compact_job_results.append(
            {
                "job_id": job_result.get("job_id"),
                "job_key": job_result.get("job_key"),
                "loaded_from_database": job_result.get("loaded_from_database"),
                "storage_summary": job_result.get("storage_summary"),
                "job_information": job_result.get("job_information"),
                "structured_job_output_path": job_result.get("structured_job_output_path"),
                "final_hybrid_score": job_result.get("final_hybrid_score"),
                "direct_matching_score": job_result.get("direct_matching_score"),
                "semantic_score": job_result.get("semantic_score"),
                "match_category": job_result.get("match_category"),
            }
        )

    output = {
        "metadata": {
            "created_at": get_current_timestamp(),
            "workflow_type": "langgraph_production_cv_job_matching_agent",
            "status": state.get("status"),
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "run_id": state.get("run_id"),
            "run_output_dir": state.get("run_output_dir"),
        },
        "user_choices": {
            "user_wants_cv_recommendations": state.get("user_wants_cv_recommendations"),
            "user_wants_job_matching": state.get("user_wants_job_matching"),
            "user_wants_job_recommendations": state.get("user_wants_job_recommendations"),
            "user_wants_detailed_report": state.get("user_wants_detailed_report"),
            "selected_job_id": state.get("selected_job_id"),
        },
        "cv_digest": state.get("cv_digest"),
        "cv_recommendations_digest": state.get("cv_recommendations_digest"),
        "matching_digest": state.get("matching_digest"),
        "selected_job_recommendations_digest": state.get("selected_job_recommendations_digest"),
        "selected_job_final_report_summary": state.get("selected_job_final_report_summary"),
        "job_results": compact_job_results,
        "workflow_log": state.get("workflow_log", []),
        "errors": state.get("errors", []),
    }

    save_json_file(
        data=output,
        file_path=agent_output_dir / "agent_workflow_output.json",
    )

    if state.get("cv_digest"):
        save_json_file(
            data=state["cv_digest"],
            file_path=agent_output_dir / "cv_digest.json",
        )

    if state.get("cv_recommendations_digest"):
        save_json_file(
            data=state["cv_recommendations_digest"],
            file_path=agent_output_dir / "cv_recommendations_digest.json",
        )

    if state.get("matching_digest"):
        save_json_file(
            data=state["matching_digest"],
            file_path=agent_output_dir / "matching_digest.json",
        )

    if state.get("ranked_matches"):
        ranked_matches_output = {
            "ranked_matches": compact_job_results,
        }

        save_json_file(
            data=ranked_matches_output,
            file_path=agent_output_dir / "ranked_matches.json",
        )

    state["agent_output"] = output
    state["agent_output_path"] = str(agent_output_dir / "agent_workflow_output.json")

    return state


def route_after_cv_digest(state: CVJobMatchingAgentState) -> str:
    if state.get("user_wants_cv_recommendations"):
        return "create_cv_recommendations_digest"

    if (
        state.get("user_wants_job_matching")
        or state.get("user_wants_job_recommendations")
        or state.get("user_wants_detailed_report")
    ):
        return "process_job_inputs"

    return "save_agent_outputs"


def route_after_cv_recommendations(state: CVJobMatchingAgentState) -> str:
    if (
        state.get("user_wants_job_matching")
        or state.get("user_wants_job_recommendations")
        or state.get("user_wants_detailed_report")
    ):
        return "process_job_inputs"

    return "save_agent_outputs"


def route_after_matching_digest(state: CVJobMatchingAgentState) -> str:
    if state.get("user_wants_job_recommendations"):
        return "create_selected_job_recommendations_digest"

    if state.get("user_wants_detailed_report"):
        return "create_selected_final_report_summary"

    return "save_agent_outputs"


def route_after_selected_job_recommendations(state: CVJobMatchingAgentState) -> str:
    if state.get("user_wants_detailed_report"):
        return "create_selected_final_report_summary"

    return "save_agent_outputs"


def build_agent_workflow():
    workflow = StateGraph(CVJobMatchingAgentState)

    workflow.add_node("initialize_agent_state", initialize_agent_state_node)
    workflow.add_node("process_cv_pdf", process_cv_pdf_node)
    workflow.add_node("analyze_cv_quality", analyze_cv_quality_node)
    workflow.add_node("extract_structured_cv", extract_structured_cv_node)
    workflow.add_node("create_cv_digest", create_cv_digest_node)
    workflow.add_node("create_cv_recommendations_digest", create_cv_recommendations_digest_node)
    workflow.add_node("process_job_inputs", process_job_inputs_node)
    workflow.add_node("rank_job_matches", rank_job_matches_node)
    workflow.add_node("create_matching_digest", create_matching_digest_node)
    workflow.add_node("select_job_for_detail", select_job_for_detail_node)
    workflow.add_node("create_selected_job_recommendations_digest", create_selected_job_recommendations_digest_node)
    workflow.add_node("create_selected_final_report_summary", create_selected_final_report_summary_node)
    workflow.add_node("save_agent_outputs", save_agent_outputs_node)

    workflow.add_edge(START, "initialize_agent_state")
    workflow.add_edge("initialize_agent_state", "process_cv_pdf")
    workflow.add_edge("process_cv_pdf", "analyze_cv_quality")
    workflow.add_edge("analyze_cv_quality", "extract_structured_cv")
    workflow.add_edge("extract_structured_cv", "create_cv_digest")

    workflow.add_conditional_edges(
        "create_cv_digest",
        route_after_cv_digest,
        {
            "create_cv_recommendations_digest": "create_cv_recommendations_digest",
            "process_job_inputs": "process_job_inputs",
            "save_agent_outputs": "save_agent_outputs",
        },
    )

    workflow.add_conditional_edges(
        "create_cv_recommendations_digest",
        route_after_cv_recommendations,
        {
            "process_job_inputs": "process_job_inputs",
            "save_agent_outputs": "save_agent_outputs",
        },
    )

    workflow.add_edge("process_job_inputs", "rank_job_matches")
    workflow.add_edge("rank_job_matches", "create_matching_digest")
    workflow.add_edge("create_matching_digest", "select_job_for_detail")

    workflow.add_conditional_edges(
        "select_job_for_detail",
        route_after_matching_digest,
        {
            "create_selected_job_recommendations_digest": "create_selected_job_recommendations_digest",
            "create_selected_final_report_summary": "create_selected_final_report_summary",
            "save_agent_outputs": "save_agent_outputs",
        },
    )

    workflow.add_conditional_edges(
        "create_selected_job_recommendations_digest",
        route_after_selected_job_recommendations,
        {
            "create_selected_final_report_summary": "create_selected_final_report_summary",
            "save_agent_outputs": "save_agent_outputs",
        },
    )

    workflow.add_edge("create_selected_final_report_summary", "save_agent_outputs")
    workflow.add_edge("save_agent_outputs", END)

    return workflow.compile()


def run_agent_workflow(
    cv_pdf_path: str,
    job_texts: Optional[List[str]] = None,
    job_inputs: Optional[List[Dict[str, Any]]] = None,
    user_wants_cv_recommendations: bool = True,
    user_wants_job_matching: bool = True,
    user_wants_job_recommendations: bool = True,
    user_wants_detailed_report: bool = True,
    selected_job_id: Optional[str] = None,
    cv_quality_weights: Optional[Dict[str, float]] = None,
    direct_matching_weights: Optional[Dict[str, float]] = None,
    matching_mode_weights: Optional[Dict[str, float]] = None,
    use_mongodb: bool = True,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:

    if cv_pdf_path is None or not str(cv_pdf_path).strip():
        raise ValueError("cv_pdf_path is required.")

    if job_inputs is None:
        job_inputs = create_job_inputs_from_texts(job_texts or [])

    initial_state: CVJobMatchingAgentState = {
        "cv_pdf_path": str(cv_pdf_path),
        "job_inputs": job_inputs,
        "user_wants_cv_recommendations": user_wants_cv_recommendations,
        "user_wants_job_matching": user_wants_job_matching,
        "user_wants_job_recommendations": user_wants_job_recommendations,
        "user_wants_detailed_report": user_wants_detailed_report,
        "selected_job_id": selected_job_id,
        "cv_quality_weights": cv_quality_weights,
        "direct_matching_weights": direct_matching_weights,
        "matching_mode_weights": matching_mode_weights,
        "use_mongodb": use_mongodb,
        "model_name": model_name,
        "temperature": temperature,
    }

    if run_id is not None:
        initial_state["run_id"] = run_id

    agent_workflow = build_agent_workflow()

    result = agent_workflow.invoke(initial_state)

    return result