from datetime import datetime
from typing import Dict, Any, List, Optional

from src.utils import (
    CV_QUALITY_OUTPUT_DIR,
    CV_EXTRACTION_OUTPUT_DIR,
    JOB_EXTRACTION_OUTPUT_DIR,
    MATCHING_OUTPUT_DIR,
    RECOMMENDATIONS_OUTPUT_DIR,
    FINAL_REPORT_OUTPUT_DIR,
    load_json_file,
    save_json_file,
    save_text_file,
)


def get_nested_value(
    data: Dict[str, Any],
    keys: List[str],
    default=None,
):

    current_value = data

    for key in keys:
        if not isinstance(current_value, dict):
            return default

        current_value = current_value.get(key)

        if current_value is None:
            return default

    return current_value


def format_score(score: Any) -> str:

    if isinstance(score, (int, float)):
        return f"{round(score, 2)}/100"

    return "Not available"


def create_markdown_list(
    items: Optional[List[Any]],
    empty_message: str = "- No items available.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for item in items:
        if item is None:
            continue

        if isinstance(item, dict):
            display_value = (
                item.get("title")
                or item.get("skill")
                or item.get("action_title")
                or item.get("job_requirement")
                or item.get("cv_evidence")
                or item.get("reason")
                or str(item)
            )
        else:
            display_value = str(item)

        if display_value and str(display_value).strip():
            lines.append(f"- {display_value}")

    if not lines:
        return empty_message

    return "\n".join(lines)


def create_action_list(
    actions: Optional[List[Dict[str, Any]]],
    empty_message: str = "- No recommended actions available.",
) -> str:

    if not actions:
        return empty_message

    lines = []

    for action in actions:
        if not isinstance(action, dict):
            lines.append(f"- {action}")
            continue

        action_title = action.get("action_title", "Action")
        action_description = action.get("action_description", "Not provided.")
        expected_impact = action.get("expected_impact", "Not provided.")
        priority = action.get("priority", "Not provided.")

        lines.append(f"- **{action_title}**")
        lines.append(f"  - Priority: {priority}")
        lines.append(f"  - Description: {action_description}")
        lines.append(f"  - Expected impact: {expected_impact}")

    return "\n".join(lines)


def create_recommendation_section(
    items: Optional[List[Dict[str, Any]]],
    empty_message: str = "- No recommendations available.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            lines.append(f"{index}. {item}")
            continue

        title = item.get("title", f"Recommendation {index}")
        reason = item.get("reason", "Not provided.")
        evidence = item.get("evidence", "Not provided.")
        recommended_actions = item.get("recommended_actions", [])

        lines.append(f"### {index}. {title}")
        lines.append("")
        lines.append(f"**Reason:** {reason}")
        lines.append("")
        lines.append(f"**Evidence:** {evidence}")
        lines.append("")
        lines.append("**Recommended actions:**")
        lines.append("")
        lines.append(create_action_list(recommended_actions))
        lines.append("")

    return "\n".join(lines)


def create_skill_recommendation_section(
    items: Optional[List[Dict[str, Any]]],
    empty_message: str = "- No skill recommendations available.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            lines.append(f"{index}. {item}")
            continue

        skill = item.get("skill", f"Skill {index}")
        current_status = item.get("current_status", "Not provided.")
        evidence = item.get("evidence", "Not provided.")
        recommended_actions = item.get("recommended_actions", [])

        lines.append(f"### {index}. {skill}")
        lines.append("")
        lines.append(f"**Current status:** {current_status}")
        lines.append("")
        lines.append(f"**Evidence:** {evidence}")
        lines.append("")
        lines.append("**Recommended actions:**")
        lines.append("")
        lines.append(create_action_list(recommended_actions))
        lines.append("")

    return "\n".join(lines)


def create_priority_actions_section(
    items: Optional[List[Dict[str, Any]]],
    empty_message: str = "- No priority actions available.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, action in enumerate(items, start=1):
        if not isinstance(action, dict):
            lines.append(f"{index}. {action}")
            continue

        action_title = action.get("action_title", f"Action {index}")
        action_description = action.get("action_description", "Not provided.")
        expected_impact = action.get("expected_impact", "Not provided.")
        priority = action.get("priority", "Not provided.")

        lines.append(f"{index}. **{action_title}**")
        lines.append(f"   - Priority: {priority}")
        lines.append(f"   - Description: {action_description}")
        lines.append(f"   - Expected impact: {expected_impact}")
        lines.append("")

    return "\n".join(lines)


def extract_candidate_information(
    structured_cv: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "candidate_name": structured_cv.get("candidate_name"),
        "email": structured_cv.get("email"),
        "phone": structured_cv.get("phone"),
        "location": structured_cv.get("location"),
        "linkedin": structured_cv.get("linkedin_url") or structured_cv.get("linkedin"),
        "github": structured_cv.get("github_url") or structured_cv.get("github"),
        "portfolio": structured_cv.get("portfolio_url") or structured_cv.get("portfolio"),
        "total_years_of_experience": structured_cv.get("total_years_of_experience"),
    }


def extract_job_information(
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


def create_cv_quality_summary(
    cv_quality_analysis: Dict[str, Any],
) -> Dict[str, Any]:

    scores = cv_quality_analysis.get("scores", {})

    return {
        "final_cv_quality_score": cv_quality_analysis.get("final_cv_quality_score"),
        "cv_quality_category": cv_quality_analysis.get("cv_quality_category"),
        "cv_quality_weighting_note": cv_quality_analysis.get("cv_quality_weighting_note"),
        "overall_summary": cv_quality_analysis.get("overall_summary"),
        "scores": scores,
        "strengths": cv_quality_analysis.get("strengths", []),
        "weaknesses": cv_quality_analysis.get("weaknesses", []),
        "missing_or_unclear_sections": cv_quality_analysis.get("missing_or_unclear_sections", []),
        "cv_improvement_recommendations": cv_quality_analysis.get("cv_improvement_recommendations", []),
        "evidence_notes": cv_quality_analysis.get("evidence_notes", []),
    }


def create_matching_summary(
    matching_result: Dict[str, Any],
) -> Dict[str, Any]:

    final_result = matching_result.get("final_result", {})
    score_breakdown = matching_result.get("score_breakdown", {})

    direct_matching_score = (
        final_result.get("direct_matching_score")
        or final_result.get("syntactic_matching_score")
        or final_result.get("rule_based_score")
    )

    return {
        "final_hybrid_score": final_result.get("final_hybrid_score"),
        "direct_matching_score": direct_matching_score,
        "semantic_score": final_result.get("semantic_score"),
        "match_category": final_result.get("match_category"),
        "score_breakdown": score_breakdown,
    }


def create_skills_and_requirements_summary(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    matching_result: Dict[str, Any],
) -> Dict[str, Any]:

    matched_items = matching_result.get("matched_items", {})
    missing_items = matching_result.get("missing_items", {})

    return {
        "cv_skills": {
            "technical_skills": structured_cv.get("technical_skills", []),
            "programming_languages": structured_cv.get("programming_languages", []),
            "frameworks_and_libraries": structured_cv.get("frameworks_and_libraries", []),
            "databases": structured_cv.get("databases", []),
            "cloud_and_devops_tools": structured_cv.get("cloud_and_devops_tools", []),
            "data_and_ai_tools": structured_cv.get("data_and_ai_tools", []),
            "other_tools": structured_cv.get("other_tools", []),
            "soft_skills": structured_cv.get("soft_skills", []),
            "languages": structured_cv.get("languages", []),
        },
        "job_requirements": {
            "required_skills": structured_job.get("required_skills", []),
            "nice_to_have_skills": structured_job.get("nice_to_have_skills", []),
            "programming_languages": structured_job.get("programming_languages", []),
            "frameworks_and_libraries": structured_job.get("frameworks_and_libraries", []),
            "databases": structured_job.get("databases", []),
            "cloud_and_devops_tools": structured_job.get("cloud_and_devops_tools", []),
            "data_and_ai_tools": structured_job.get("data_and_ai_tools", []),
            "testing_tools": structured_job.get("testing_tools", []),
            "other_tools": structured_job.get("other_tools", []),
            "certifications": structured_job.get("certifications", []),
            "language_requirements": structured_job.get("language_requirements", []),
            "soft_skills": structured_job.get("soft_skills", []),
        },
        "matched_items": matched_items,
        "missing_items": missing_items,
    }


def create_semantic_analysis_summary(
    matching_result: Dict[str, Any],
) -> Dict[str, Any]:

    semantic_analysis = matching_result.get("semantic_analysis", {})

    return {
        "role_fit_summary": semantic_analysis.get("role_fit_summary"),
        "responsibilities_alignment_score": semantic_analysis.get("responsibilities_alignment_score"),
        "soft_skills_evidence_score": semantic_analysis.get("soft_skills_evidence_score"),
        "contextual_experience_alignment_score": semantic_analysis.get("contextual_experience_alignment_score"),
        "semantic_skill_evidence_score": semantic_analysis.get("semantic_skill_evidence_score"),
        "responsibilities_evidenced": semantic_analysis.get("responsibilities_evidenced", []),
        "responsibilities_not_evidenced": semantic_analysis.get("responsibilities_not_evidenced", []),
        "soft_skills_evidenced": semantic_analysis.get("soft_skills_evidenced", []),
        "soft_skills_not_clearly_evidenced": semantic_analysis.get("soft_skills_not_clearly_evidenced", []),
        "contextual_experience_evidence": semantic_analysis.get("contextual_experience_evidence", []),
        "contextual_experience_gaps": semantic_analysis.get("contextual_experience_gaps", []),
        "semantic_skill_evidence": semantic_analysis.get("semantic_skill_evidence", []),
        "possible_direct_matching_false_negatives": semantic_analysis.get("possible_direct_matching_false_negatives", []),
        "evidence_notes": semantic_analysis.get("evidence_notes", []),
    }


def create_recommendation_summary(
    recommendations_output: Dict[str, Any],
) -> Dict[str, Any]:

    recommendations = recommendations_output.get("recommendations", recommendations_output)

    return {
        "overall_recommendation_summary": recommendations.get("overall_recommendation_summary"),
        "cv_improvement_recommendations": recommendations.get("cv_improvement_recommendations", []),
        "missing_required_skills_recommendations": recommendations.get("missing_required_skills_recommendations", []),
        "technical_development_recommendations": recommendations.get("technical_development_recommendations", []),
        "project_recommendations": recommendations.get("project_recommendations", []),
        "soft_skills_recommendations": recommendations.get("soft_skills_recommendations", []),
        "priority_actions": recommendations.get("priority_actions", []),
        "methodology_notes": recommendations.get("methodology_notes", []),
    }


def create_weighting_summary(
    cv_quality_analysis: Dict[str, Any],
    matching_result: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "cv_quality_weights_used": cv_quality_analysis.get("cv_quality_weights_used", {}),
        "cv_quality_weighting_note": cv_quality_analysis.get("cv_quality_weighting_note"),
        "matching_weights_used": matching_result.get("weights_used", {}),
    }


def create_final_report(
    cv_quality_analysis: Dict[str, Any],
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    matching_result: Dict[str, Any],
    recommendations_output: Dict[str, Any],
    source_files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:

    candidate_information = extract_candidate_information(structured_cv)
    job_information = extract_job_information(structured_job)

    cv_quality_summary = create_cv_quality_summary(cv_quality_analysis)

    matching_summary = create_matching_summary(matching_result)

    skills_and_requirements_summary = create_skills_and_requirements_summary(
        structured_cv=structured_cv,
        structured_job=structured_job,
        matching_result=matching_result,
    )

    semantic_analysis_summary = create_semantic_analysis_summary(
        matching_result=matching_result,
    )

    recommendation_summary = create_recommendation_summary(
        recommendations_output=recommendations_output,
    )

    weighting_summary = create_weighting_summary(
        cv_quality_analysis=cv_quality_analysis,
        matching_result=matching_result,
    )

    final_report = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "report_type": "final_cv_job_matching_report",
            "source_files": source_files or {},
            "methodological_note": (
                "This final report aggregates outputs generated by previous processing steps. "
                "It does not recalculate CV quality, structured extraction, matching or recommendations."
            ),
        },
        "candidate_information": candidate_information,
        "job_information": job_information,
        "cv_quality_summary": cv_quality_summary,
        "matching_summary": matching_summary,
        "skills_and_requirements_summary": skills_and_requirements_summary,
        "semantic_analysis_summary": semantic_analysis_summary,
        "recommendation_summary": recommendation_summary,
        "weighting_summary": weighting_summary,
    }

    return final_report


def create_final_markdown_report(
    final_report: Dict[str, Any],
) -> str:

    candidate_information = final_report.get("candidate_information", {})
    job_information = final_report.get("job_information", {})
    cv_quality_summary = final_report.get("cv_quality_summary", {})
    cv_quality_scores = cv_quality_summary.get("scores", {})
    matching_summary = final_report.get("matching_summary", {})
    score_breakdown = matching_summary.get("score_breakdown", {})
    skills_and_requirements_summary = final_report.get("skills_and_requirements_summary", {})
    semantic_analysis_summary = final_report.get("semantic_analysis_summary", {})
    recommendation_summary = final_report.get("recommendation_summary", {})
    weighting_summary = final_report.get("weighting_summary", {})

    cv_skills = skills_and_requirements_summary.get("cv_skills", {})
    job_requirements = skills_and_requirements_summary.get("job_requirements", {})
    matched_items = skills_and_requirements_summary.get("matched_items", {})
    missing_items = skills_and_requirements_summary.get("missing_items", {})

    final_markdown_report = f"""
# Final CV-Job Matching Report

## 1. Candidate Information

- Candidate name: {candidate_information.get("candidate_name")}
- Email: {candidate_information.get("email")}
- Phone: {candidate_information.get("phone")}
- Location: {candidate_information.get("location")}
- LinkedIn: {candidate_information.get("linkedin")}
- GitHub: {candidate_information.get("github")}
- Portfolio: {candidate_information.get("portfolio")}
- Total years of experience: {candidate_information.get("total_years_of_experience")}

## 2. Job Information

- Job title: {job_information.get("job_title")}
- Company: {job_information.get("company_name")}
- Job category: {job_information.get("job_category")}
- Location: {job_information.get("location")}
- Work mode: {job_information.get("work_mode")}
- Employment type: {job_information.get("employment_type")}

## 3. CV Quality Summary

- Final CV quality score: {format_score(cv_quality_summary.get("final_cv_quality_score"))}
- CV quality category: {cv_quality_summary.get("cv_quality_category")}

### Overall CV Quality Summary

{cv_quality_summary.get("overall_summary")}

### CV Quality Score Breakdown

- Structure and readability: {format_score(cv_quality_scores.get("structure_and_readability_score"))}
- Completeness: {format_score(cv_quality_scores.get("completeness_score"))}
- Technical skills clarity: {format_score(cv_quality_scores.get("technical_skills_clarity_score"))}
- Experience description: {format_score(cv_quality_scores.get("experience_description_score"))}
- Projects description: {format_score(cv_quality_scores.get("projects_description_score"))}
- Measurable results: {format_score(cv_quality_scores.get("measurable_results_score"))}
- IT relevance: {format_score(cv_quality_scores.get("it_relevance_score"))}

### CV Strengths

{create_markdown_list(cv_quality_summary.get("strengths", []), empty_message="- No CV strengths listed.")}

### CV Weaknesses

{create_markdown_list(cv_quality_summary.get("weaknesses", []), empty_message="- No CV weaknesses listed.")}

### Missing or Unclear CV Sections

{create_markdown_list(cv_quality_summary.get("missing_or_unclear_sections", []), empty_message="- No missing or unclear CV sections listed.")}

## 4. Matching Summary

- Final hybrid score: {format_score(matching_summary.get("final_hybrid_score"))}
- Match category: {matching_summary.get("match_category")}
- Direct matching score: {format_score(matching_summary.get("direct_matching_score"))}
- LLM semantic score: {format_score(matching_summary.get("semantic_score"))}

### Matching Score Breakdown

- Required skills score: {format_score(score_breakdown.get("required_skills_score"))}
- Technology score: {format_score(score_breakdown.get("technology_score"))}
- Experience score: {format_score(score_breakdown.get("experience_score"))}
- Education score: {format_score(score_breakdown.get("education_score"))}
- Nice-to-have score: {format_score(score_breakdown.get("nice_to_have_score"))}
- Certification score: {format_score(score_breakdown.get("certification_score"))}
- Language score: {format_score(score_breakdown.get("language_score"))}
- Responsibilities alignment score: {format_score(score_breakdown.get("responsibilities_alignment_score"))}
- Soft skills evidence score: {format_score(score_breakdown.get("soft_skills_evidence_score"))}
- Contextual experience alignment score: {format_score(score_breakdown.get("contextual_experience_alignment_score"))}
- Semantic skill evidence score: {format_score(score_breakdown.get("semantic_skill_evidence_score"))}

## 5. Skills and Requirements Summary

### CV Technical Skills

{create_markdown_list(cv_skills.get("technical_skills", []), empty_message="- No CV technical skills listed.")}

### CV Programming Languages

{create_markdown_list(cv_skills.get("programming_languages", []), empty_message="- No CV programming languages listed.")}

### CV Frameworks and Libraries

{create_markdown_list(cv_skills.get("frameworks_and_libraries", []), empty_message="- No CV frameworks or libraries listed.")}

### CV Databases

{create_markdown_list(cv_skills.get("databases", []), empty_message="- No CV databases listed.")}

### CV Cloud and DevOps Tools

{create_markdown_list(cv_skills.get("cloud_and_devops_tools", []), empty_message="- No CV cloud or DevOps tools listed.")}

### Job Required Skills

{create_markdown_list(job_requirements.get("required_skills", []), empty_message="- No job required skills listed.")}

### Job Nice-to-Have Skills

{create_markdown_list(job_requirements.get("nice_to_have_skills", []), empty_message="- No job nice-to-have skills listed.")}

### Job Technology Requirements

**Programming languages**

{create_markdown_list(job_requirements.get("programming_languages", []), empty_message="- No programming languages listed.")}

**Frameworks and libraries**

{create_markdown_list(job_requirements.get("frameworks_and_libraries", []), empty_message="- No frameworks or libraries listed.")}

**Databases**

{create_markdown_list(job_requirements.get("databases", []), empty_message="- No databases listed.")}

**Cloud and DevOps tools**

{create_markdown_list(job_requirements.get("cloud_and_devops_tools", []), empty_message="- No cloud or DevOps tools listed.")}

## 6. Matched and Missing Items

### Matched Required Skills

{create_markdown_list(matched_items.get("matched_required_skills", []), empty_message="- No matched required skills listed.")}

### Missing Required Skills

{create_markdown_list(missing_items.get("missing_required_skills", []), empty_message="- No missing required skills listed.")}

### Matched Technology Skills

{create_markdown_list(matched_items.get("matched_technology_skills", []), empty_message="- No matched technology skills listed.")}

### Missing Technology Skills

{create_markdown_list(missing_items.get("missing_technology_skills", []), empty_message="- No missing technology skills listed.")}

### Missing Certifications

{create_markdown_list(missing_items.get("missing_certifications", []), empty_message="- No missing certifications listed.")}

### Missing Language Requirements

{create_markdown_list(missing_items.get("missing_languages", []), empty_message="- No missing language requirements listed.")}

## 7. Semantic Analysis Summary

### Role Fit Summary

{semantic_analysis_summary.get("role_fit_summary")}

### Semantic Scores

- Responsibilities alignment score: {format_score(semantic_analysis_summary.get("responsibilities_alignment_score"))}
- Soft skills evidence score: {format_score(semantic_analysis_summary.get("soft_skills_evidence_score"))}
- Contextual experience alignment score: {format_score(semantic_analysis_summary.get("contextual_experience_alignment_score"))}
- Semantic skill evidence score: {format_score(semantic_analysis_summary.get("semantic_skill_evidence_score"))}

### Responsibilities Evidenced

{create_markdown_list(semantic_analysis_summary.get("responsibilities_evidenced", []), empty_message="- No evidenced responsibilities listed.")}

### Responsibilities Not Clearly Evidenced

{create_markdown_list(semantic_analysis_summary.get("responsibilities_not_evidenced", []), empty_message="- No responsibility gaps listed.")}

### Soft Skills Evidenced

{create_markdown_list(semantic_analysis_summary.get("soft_skills_evidenced", []), empty_message="- No evidenced soft skills listed.")}

### Soft Skills Not Clearly Evidenced

{create_markdown_list(semantic_analysis_summary.get("soft_skills_not_clearly_evidenced", []), empty_message="- No soft skill evidence gaps listed.")}

### Contextual Experience Evidence

{create_markdown_list(semantic_analysis_summary.get("contextual_experience_evidence", []), empty_message="- No contextual experience evidence listed.")}

### Contextual Experience Gaps

{create_markdown_list(semantic_analysis_summary.get("contextual_experience_gaps", []), empty_message="- No contextual experience gaps listed.")}

### Semantic Skill Evidence

{create_markdown_list(semantic_analysis_summary.get("semantic_skill_evidence", []), empty_message="- No semantic skill evidence listed.")}

### Possible Direct Matching False Negatives

{create_markdown_list(semantic_analysis_summary.get("possible_direct_matching_false_negatives", []), empty_message="- No possible direct matching false negatives listed.")}

### Semantic Evidence Notes

{create_markdown_list(semantic_analysis_summary.get("evidence_notes", []), empty_message="- No semantic evidence notes listed.")}

## 8. Recommendation Summary

### Overall Recommendation Summary

{recommendation_summary.get("overall_recommendation_summary")}

## 9. CV Improvement Recommendations

{create_recommendation_section(recommendation_summary.get("cv_improvement_recommendations", []), empty_message="- No CV improvement recommendations.")}

## 10. Missing Required Skills Recommendations

{create_skill_recommendation_section(recommendation_summary.get("missing_required_skills_recommendations", []), empty_message="- No missing required skills recommendations.")}

## 11. Technical Development Recommendations

{create_skill_recommendation_section(recommendation_summary.get("technical_development_recommendations", []), empty_message="- No technical development recommendations.")}

## 12. Project Recommendations

{create_recommendation_section(recommendation_summary.get("project_recommendations", []), empty_message="- No project recommendations.")}

## 13. Soft Skills Recommendations

{create_recommendation_section(recommendation_summary.get("soft_skills_recommendations", []), empty_message="- No soft skills recommendations.")}

## 14. Priority Actions

{create_priority_actions_section(recommendation_summary.get("priority_actions", []), empty_message="- No priority actions.")}

## 15. Methodology Notes and Limitations

{create_markdown_list(recommendation_summary.get("methodology_notes", []), empty_message="- No additional methodology notes.")}

## 16. Weighting Summary

### CV Quality Weights

{create_markdown_list([
    f"{key}: {value}"
    for key, value in weighting_summary.get("cv_quality_weights_used", {}).items()
], empty_message="- No CV quality weights available.")}

### CV Quality Weighting Note

{weighting_summary.get("cv_quality_weighting_note")}

### Matching Weights

{create_markdown_list([
    f"{key}: {value}"
    for key, value in weighting_summary.get("matching_weights_used", {}).items()
], empty_message="- No matching weights available.")}

## 17. Final Conclusion

This report combines CV quality analysis, structured CV extraction, structured job posting extraction, hybrid CV-job matching and recommendation generation.

The final result should be interpreted as an evidence-based support tool for CV improvement and job matching, not as a final hiring decision.

The system does not assume skills or experience that are not evidenced in the CV.
"""

    return final_markdown_report.strip()


def save_final_report_outputs(
    final_report: Dict[str, Any],
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    if json_output_path is None:
        json_output_path = FINAL_REPORT_OUTPUT_DIR / "final_report.json"

    if markdown_output_path is None:
        markdown_output_path = FINAL_REPORT_OUTPUT_DIR / "final_report.md"

    markdown_report = create_final_markdown_report(final_report)

    saved_json_path = save_json_file(
        data=final_report,
        file_path=json_output_path,
    )

    saved_markdown_path = save_text_file(
        text=markdown_report,
        file_path=markdown_output_path,
    )

    return {
        "json_output_path": str(saved_json_path),
        "markdown_output_path": str(saved_markdown_path),
        "markdown_report": markdown_report,
    }


def process_final_report(
    cv_quality_analysis: Dict[str, Any],
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    matching_result: Dict[str, Any],
    recommendations_output: Dict[str, Any],
    source_files: Optional[Dict[str, str]] = None,
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    final_report = create_final_report(
        cv_quality_analysis=cv_quality_analysis,
        structured_cv=structured_cv,
        structured_job=structured_job,
        matching_result=matching_result,
        recommendations_output=recommendations_output,
        source_files=source_files,
    )

    saved_outputs = save_final_report_outputs(
        final_report=final_report,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )

    return {
        "final_report": final_report,
        "json_output_path": saved_outputs["json_output_path"],
        "markdown_output_path": saved_outputs["markdown_output_path"],
        "markdown_report": saved_outputs["markdown_report"],
    }


def process_final_report_from_files(
    cv_quality_path=None,
    structured_cv_path=None,
    structured_job_path=None,
    matching_result_path=None,
    recommendations_path=None,
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    if cv_quality_path is None:
        cv_quality_path = CV_QUALITY_OUTPUT_DIR / "cv_quality_analysis.json"

    if structured_cv_path is None:
        structured_cv_path = CV_EXTRACTION_OUTPUT_DIR / "structured_cv.json"

    if structured_job_path is None:
        structured_job_path = JOB_EXTRACTION_OUTPUT_DIR / "structured_job.json"

    if matching_result_path is None:
        matching_result_path = MATCHING_OUTPUT_DIR / "matching_result.json"

    if recommendations_path is None:
        recommendations_path = RECOMMENDATIONS_OUTPUT_DIR / "recommendations.json"

    cv_quality_analysis = load_json_file(cv_quality_path)
    structured_cv = load_json_file(structured_cv_path)
    structured_job = load_json_file(structured_job_path)
    matching_result = load_json_file(matching_result_path)
    recommendations_output = load_json_file(recommendations_path)

    source_files = {
        "cv_quality_analysis": str(cv_quality_path),
        "structured_cv": str(structured_cv_path),
        "structured_job": str(structured_job_path),
        "matching_result": str(matching_result_path),
        "recommendations": str(recommendations_path),
    }

    return process_final_report(
        cv_quality_analysis=cv_quality_analysis,
        structured_cv=structured_cv,
        structured_job=structured_job,
        matching_result=matching_result,
        recommendations_output=recommendations_output,
        source_files=source_files,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )