import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils import (
    PROJECT_ROOT,
    RECOMMENDATIONS_OUTPUT_DIR,
    save_json_file,
    save_text_file,
)


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class PriorityAction(BaseModel):
    action_title: str = Field(
        description=(
            "Short title of the concrete action the candidate should take."
        )
    )

    action_description: str = Field(
        description=(
            "Concrete next action written in practical language. "
            "Do not suggest adding false information to the CV. "
            "If the action depends on information that is not evidenced in the CV, "
            "clearly use conditional wording."
        )
    )

    expected_impact: str = Field(
        description=(
            "Expected impact of this action on CV quality, job matching, "
            "or candidate profile improvement."
        )
    )

    priority: str = Field(
        description=(
            "Priority level of the action. "
            "Use one of the following values: High, Medium, Low."
        )
    )


class RecommendationItem(BaseModel):
    title: str = Field(
        description=(
            "Short title of the recommendation. "
            "The title should clearly describe what should be improved or developed."
        )
    )

    reason: str = Field(
        description=(
            "Explanation of why this recommendation is important for the selected job posting "
            "or CV quality. The reason must be based on the provided CV, job posting, "
            "matching result or CV quality analysis."
        )
    )

    evidence: str = Field(
        description=(
            "Evidence from the input data that supports this recommendation. "
            "This may refer to a missing skill, weak CV quality area, missing responsibility evidence, "
            "or job requirement. If the evidence is that something is not present in the CV, "
            "clearly state that it is not evidenced."
        )
    )

    recommended_actions: List[PriorityAction] = Field(
        default_factory=list,
        description=(
            "Concrete prioritized actions related to this recommendation. "
            "Each action should explain what the candidate should do, why it matters, and its priority."
        )
    )


class SkillDevelopmentRecommendation(BaseModel):
    skill: str = Field(
        description=(
            "Name of the missing or weakly evidenced skill, tool, technology or knowledge area."
        )
    )

    current_status: str = Field(
        description=(
            "Current status based on the CV and matching result. "
            "Use phrases such as 'not evidenced in CV', 'partially evidenced', or 'unclear from CV'. "
            "Do not claim that the candidate has the skill unless it is evidenced."
        )
    )

    evidence: str = Field(
        description=(
            "Evidence explaining why this skill is recommended. "
            "This should refer to the job requirement, matching result, or lack of evidence in the CV."
        )
    )

    recommended_actions: List[PriorityAction] = Field(
        default_factory=list,
        description=(
            "Concrete prioritized actions for developing this skill and/or representing it better in the CV. "
            "If the candidate already has this skill but it is not clearly shown, use conditional wording such as: "
            "'If this is true, add it clearly to the CV with project or experience evidence.' "
            "Never recommend falsely adding a skill."
        )
    )


class RecommendationOutput(BaseModel):
    overall_recommendation_summary: str = Field(
        description=(
            "Concise summary of the most important recommendations for improving the CV "
            "and candidate profile in relation to the selected job posting."
        )
    )

    cv_improvement_recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description=(
            "Recommendations focused on improving the CV document itself. "
            "Examples include improving structure, adding clearer project descriptions, "
            "adding measurable results, clarifying experience, or better grouping technical skills."
        )
    )

    missing_required_skills_recommendations: List[SkillDevelopmentRecommendation] = Field(
        default_factory=list,
        description=(
            "Recommendations for required job skills that are missing or not clearly evidenced in the CV. "
            "This should be based primarily on missing required skills from the matching result."
        )
    )

    technical_development_recommendations: List[SkillDevelopmentRecommendation] = Field(
        default_factory=list,
        description=(
            "Recommendations for technical skills, tools, frameworks, databases, cloud tools "
            "or other technologies that the candidate should develop to improve job fit."
        )
    )

    project_recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description=(
            "Recommendations for project work that could help the candidate demonstrate missing "
            "or weakly evidenced skills. Do not invent projects that the candidate already completed. "
            "Suggest possible future projects or portfolio improvements."
        )
    )

    soft_skills_recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description=(
            "Recommendations for better evidencing soft skills requested in the job posting. "
            "Do not suggest writing generic claims such as 'excellent communication skills'. "
            "Suggest showing soft skills through experience, responsibilities, teamwork, mentoring, "
            "documentation or project examples."
        )
    )

    priority_actions: List[PriorityAction] = Field(
        default_factory=list,
        description=(
            "Short prioritized list of the most important next actions selected from the recommendations above. "
            "These actions should be practical and based on the strongest gaps found in the analysis."
        )
    )

    methodology_notes: List[str] = Field(
        default_factory=list,
        description=(
            "General notes about the limitations of the recommendation process. "
            "Do not use this field for evidence related to individual recommendations, because evidence should be stated "
            "inside each specific recommendation."
        )
    )


recommendation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant specialized in CV improvement and career development recommendations for IT job matching.

Your task is to generate evidence-based recommendations using:
- structured CV data,
- structured job posting data,
- CV quality analysis,
- CV-job matching result.

Important rules:
- Use only the information provided in the input data.
- Do not invent candidate skills, experience, projects, education, certificates or achievements.
- Do not suggest adding false information to the CV.
- If a skill or experience is not evidenced in the CV, clearly say that it is not evidenced.
- If the candidate may have a skill but it is not visible in the CV, use wording such as:
  "If this is true, add it clearly to the CV with evidence."
- For missing required skills, recommend learning, practicing, building a project, or adding evidence only if true.
- Recommendations should be concrete, practical and useful.
- Prioritize required job skills and important CV quality weaknesses.
- Do not calculate a new match score.
- Do not override the previous matching result.
- Return the result using the required structured schema.

Recommendation structure:
- For each recommendation, provide a reason and specific supporting evidence.
- Evidence must be stated inside each specific recommendation.
- Do not place evidence for individual recommendations in a global notes field.
- Use recommended_actions with the PriorityAction schema for concrete actions, priorities and expected impact.
- Use methodology_notes only for general limitations of the recommendation process.

Recommendation style:
- Be clear and professional.
- Avoid generic advice.
- Explain why each recommendation matters.
- Connect recommendations to job requirements and CV evidence.
"""
        ),
        (
            "human",
            """
Generate recommendations using the following data.

Structured CV:

{structured_cv}

Structured job posting:

{structured_job}

CV quality analysis:

{cv_quality_analysis}

Matching result:

{matching_result}
"""
        )
    ]
)


def create_recommendation_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    structured_recommendation_llm = llm.with_structured_output(RecommendationOutput)

    return recommendation_prompt | structured_recommendation_llm


def model_to_dict(model_result: Any) -> Dict[str, Any]:

    if hasattr(model_result, "model_dump"):
        return model_result.model_dump()

    return model_result.dict()


def validate_recommendation_inputs(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    cv_quality_analysis: Dict[str, Any],
    matching_result: Dict[str, Any],
):

    if not isinstance(structured_cv, dict):
        raise ValueError("structured_cv must be a dictionary.")

    if not isinstance(structured_job, dict):
        raise ValueError("structured_job must be a dictionary.")

    if not isinstance(cv_quality_analysis, dict):
        raise ValueError("cv_quality_analysis must be a dictionary.")

    if not isinstance(matching_result, dict):
        raise ValueError("matching_result must be a dictionary.")


def generate_recommendations(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    cv_quality_analysis: Dict[str, Any],
    matching_result: Dict[str, Any],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    validate_recommendation_inputs(
        structured_cv=structured_cv,
        structured_job=structured_job,
        cv_quality_analysis=cv_quality_analysis,
        matching_result=matching_result,
    )

    recommendation_chain = create_recommendation_chain(
        model_name=model_name,
        temperature=temperature,
    )

    recommendation_result = recommendation_chain.invoke(
        {
            "structured_cv": json.dumps(structured_cv, indent=2, ensure_ascii=False),
            "structured_job": json.dumps(structured_job, indent=2, ensure_ascii=False),
            "cv_quality_analysis": json.dumps(cv_quality_analysis, indent=2, ensure_ascii=False),
            "matching_result": json.dumps(matching_result, indent=2, ensure_ascii=False),
        }
    )

    recommendation_dict = model_to_dict(recommendation_result)

    return recommendation_dict


def create_recommendation_output(
    recommendation_dict: Dict[str, Any],
    structured_job: Dict[str, Any],
    matching_result: Dict[str, Any],
    model_name: str = "gpt-4o-mini",
    cv_source: Optional[str] = None,
    job_source: Optional[str] = None,
    cv_quality_source: Optional[str] = None,
    matching_source: Optional[str] = None,
) -> Dict[str, Any]:

    final_result = matching_result.get("final_result", {})

    direct_matching_score = (
        final_result.get("direct_matching_score")
        or final_result.get("syntactic_matching_score")
        or final_result.get("rule_based_score")
    )

    recommendation_output = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "recommendation_type": "llm_based_evidence_recommendations",
            "llm_model": model_name,
            "cv_source": cv_source,
            "job_source": job_source,
            "cv_quality_source": cv_quality_source,
            "matching_source": matching_source,
        },
        "job_information": {
            "job_title": structured_job.get("job_title"),
            "company_name": structured_job.get("company_name"),
            "job_category": structured_job.get("job_category"),
            "location": structured_job.get("location"),
            "work_mode": structured_job.get("work_mode"),
            "employment_type": structured_job.get("employment_type"),
        },
        "matching_summary": {
            "final_hybrid_score": final_result.get("final_hybrid_score"),
            "direct_matching_score": direct_matching_score,
            "semantic_score": final_result.get("semantic_score"),
            "match_category": final_result.get("match_category"),
        },
        "recommendations": recommendation_dict,
    }

    return recommendation_output


def create_action_list(
    actions: List[Dict[str, Any]],
    empty_message: str = "- No recommended actions generated.",
) -> str:

    if not actions:
        return empty_message

    lines = []

    for action in actions:
        action_title = action.get("action_title", "Action")
        action_description = action.get("action_description", "")
        expected_impact = action.get("expected_impact", "")
        priority = action.get("priority", "")

        lines.append(f"- **{action_title}**")
        lines.append(f"  - Priority: {priority}")
        lines.append(f"  - Description: {action_description}")
        lines.append(f"  - Expected impact: {expected_impact}")

    return "\n".join(lines)


def create_recommendation_section(
    items: List[Dict[str, Any]],
    empty_message: str = "- No recommendations generated.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, item in enumerate(items, start=1):
        title = item.get("title", f"Recommendation {index}")
        reason = item.get("reason", "")
        evidence = item.get("evidence", "")
        recommended_actions = item.get("recommended_actions", [])

        lines.append(f"### {index}. {title}")
        lines.append("")
        lines.append(f"- Reason: {reason}")
        lines.append(f"- Evidence: {evidence}")
        lines.append("")
        lines.append("Recommended actions:")
        lines.append("")
        lines.append(create_action_list(recommended_actions))
        lines.append("")

    return "\n".join(lines)


def create_skill_recommendation_section(
    items: List[Dict[str, Any]],
    empty_message: str = "- No skill development recommendations generated.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, item in enumerate(items, start=1):
        skill = item.get("skill", f"Skill {index}")
        current_status = item.get("current_status", "")
        evidence = item.get("evidence", "")
        recommended_actions = item.get("recommended_actions", [])

        lines.append(f"### {index}. {skill}")
        lines.append("")
        lines.append(f"- Current status: {current_status}")
        lines.append(f"- Evidence: {evidence}")
        lines.append("")
        lines.append("Recommended actions:")
        lines.append("")
        lines.append(create_action_list(recommended_actions))
        lines.append("")

    return "\n".join(lines)


def create_priority_actions_section(
    items: List[Dict[str, Any]],
    empty_message: str = "- No priority actions generated.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for index, item in enumerate(items, start=1):
        action_title = item.get("action_title", f"Action {index}")
        action_description = item.get("action_description", "")
        expected_impact = item.get("expected_impact", "")
        priority = item.get("priority", "")

        lines.append(f"{index}. **{action_title}**")
        lines.append(f"   - Priority: {priority}")
        lines.append(f"   - Description: {action_description}")
        lines.append(f"   - Expected impact: {expected_impact}")

    return "\n".join(lines)


def create_markdown_list(
    items: List[Any],
    empty_message: str = "- No items found.",
) -> str:

    if not items:
        return empty_message

    return "\n".join([f"- {item}" for item in items])


def create_recommendations_markdown_report(
    recommendation_output: Dict[str, Any],
) -> str:

    job_information = recommendation_output.get("job_information", {})
    matching_summary = recommendation_output.get("matching_summary", {})
    recommendations = recommendation_output.get("recommendations", {})

    report = f"""
# CV Improvement and Professional Development Recommendations

## Job Information

- Job title: {job_information.get("job_title")}
- Company: {job_information.get("company_name")}
- Job category: {job_information.get("job_category")}
- Location: {job_information.get("location")}
- Work mode: {job_information.get("work_mode")}
- Employment type: {job_information.get("employment_type")}

## Matching Summary

- Final hybrid score: {matching_summary.get("final_hybrid_score")}/100
- Direct matching score: {matching_summary.get("direct_matching_score")}/100
- LLM semantic score: {matching_summary.get("semantic_score")}/100
- Match category: {matching_summary.get("match_category")}

## Overall Recommendation Summary

{recommendations.get("overall_recommendation_summary")}

## CV Improvement Recommendations

{create_recommendation_section(recommendations.get("cv_improvement_recommendations", []))}

## Missing Required Skills Recommendations

{create_skill_recommendation_section(recommendations.get("missing_required_skills_recommendations", []))}

## Technical Development Recommendations

{create_skill_recommendation_section(recommendations.get("technical_development_recommendations", []))}

## Project Recommendations

{create_recommendation_section(recommendations.get("project_recommendations", []))}

## Soft Skills Recommendations

{create_recommendation_section(recommendations.get("soft_skills_recommendations", []))}

## Priority Actions

{create_priority_actions_section(recommendations.get("priority_actions", []))}

## Methodology Notes

{create_markdown_list(
    recommendations.get("methodology_notes", []),
    empty_message="- No additional methodology notes."
)}
"""

    return report.strip()


def save_recommendation_outputs(
    recommendation_output: Dict[str, Any],
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    if json_output_path is None:
        json_output_path = RECOMMENDATIONS_OUTPUT_DIR / "recommendations.json"

    if markdown_output_path is None:
        markdown_output_path = RECOMMENDATIONS_OUTPUT_DIR / "recommendations_report.md"

    markdown_report = create_recommendations_markdown_report(
        recommendation_output=recommendation_output,
    )

    saved_json_path = save_json_file(
        data=recommendation_output,
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


def process_recommendations(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    cv_quality_analysis: Dict[str, Any],
    matching_result: Dict[str, Any],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    cv_source: Optional[str] = None,
    job_source: Optional[str] = None,
    cv_quality_source: Optional[str] = None,
    matching_source: Optional[str] = None,
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    recommendation_dict = generate_recommendations(
        structured_cv=structured_cv,
        structured_job=structured_job,
        cv_quality_analysis=cv_quality_analysis,
        matching_result=matching_result,
        model_name=model_name,
        temperature=temperature,
    )

    recommendation_output = create_recommendation_output(
        recommendation_dict=recommendation_dict,
        structured_job=structured_job,
        matching_result=matching_result,
        model_name=model_name,
        cv_source=cv_source,
        job_source=job_source,
        cv_quality_source=cv_quality_source,
        matching_source=matching_source,
    )

    saved_outputs = save_recommendation_outputs(
        recommendation_output=recommendation_output,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )

    return {
        "recommendation_output": recommendation_output,
        "recommendations": recommendation_output["recommendations"],
        "json_output_path": saved_outputs["json_output_path"],
        "markdown_output_path": saved_outputs["markdown_output_path"],
        "markdown_report": saved_outputs["markdown_report"],
    }