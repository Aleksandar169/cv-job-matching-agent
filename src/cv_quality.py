from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils import (
    PROJECT_ROOT,
    CV_QUALITY_OUTPUT_DIR,
    save_json_file,
    save_text_file,
)


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


DEFAULT_CV_QUALITY_WEIGHTS = {
    "structure_and_readability_score": 0.15,
    "completeness_score": 0.15,
    "technical_skills_clarity_score": 0.20,
    "experience_description_score": 0.20,
    "projects_description_score": 0.15,
    "measurable_results_score": 0.10,
    "it_relevance_score": 0.05,
}


CV_QUALITY_WEIGHTING_NOTE = (
    "CV quality weights are default configurable values. "
    "They are not treated as universally fixed values and can be adjusted by the user "
    "depending on the evaluation context."
)


class CVQualityScores(BaseModel):
    structure_and_readability_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for CV structure, readability and section organization. "
            "0-20: very poorly structured, hard to read, no clear sections. "
            "21-40: weak structure, inconsistent formatting, important sections difficult to identify. "
            "41-60: acceptable structure but with noticeable readability or organization issues. "
            "61-80: well structured and mostly easy to read, with minor formatting or organization issues. "
            "81-100: very clear, professional, logically organized and easy to scan."
        ),
    )

    completeness_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for completeness of basic CV information. "
            "Evaluate whether the CV includes relevant contact information, education, work experience, "
            "skills, projects, certifications or other important sections. "
            "0-20: most essential information is missing. "
            "21-40: several important sections are missing or unclear. "
            "41-60: basic information is present but incomplete. "
            "61-80: most important information is included. "
            "81-100: CV is complete and contains all key information expected for an IT candidate."
        ),
    )

    technical_skills_clarity_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for clarity and organization of technical skills. "
            "Evaluate whether programming languages, frameworks, databases, tools and platforms are clearly listed "
            "and grouped in a readable way. "
            "0-20: technical skills are missing or almost impossible to identify. "
            "21-40: skills are mentioned vaguely or mixed with unrelated text. "
            "41-60: skills are present but poorly organized or too generic. "
            "61-80: skills are clear and mostly well organized. "
            "81-100: skills are clearly structured, specific and highly readable."
        ),
    )

    experience_description_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for clarity and detail of work experience descriptions. "
            "Evaluate whether job roles, responsibilities, technologies used and contributions are clearly described. "
            "0-20: work experience is missing or not described. "
            "21-40: experience is listed but with very little detail. "
            "41-60: responsibilities are partially described but remain generic. "
            "61-80: experience is clearly described with relevant responsibilities and technologies. "
            "81-100: experience is detailed, specific and clearly shows candidate contribution."
        ),
    )

    projects_description_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for clarity and detail of project descriptions. "
            "Evaluate whether projects include goal, role, technologies, implementation details and outcome. "
            "0-20: projects are missing or not described. "
            "21-40: projects are listed with minimal explanation. "
            "41-60: projects are understandable but lack technical or outcome details. "
            "61-80: projects are clearly described with technologies and candidate role. "
            "81-100: projects are detailed, technically clear and include concrete outcomes or impact."
        ),
    )

    measurable_results_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for presence of measurable achievements and concrete results. "
            "Look for numbers, percentages, performance improvements, reduced costs, increased efficiency, "
            "number of users, project scale or other measurable outcomes. "
            "0-20: no measurable results are present. "
            "21-40: very few vague achievements are mentioned. "
            "41-60: some results are present but mostly not quantified. "
            "61-80: several concrete results or achievements are included. "
            "81-100: CV strongly uses quantified achievements and measurable impact."
        ),
    )

    it_relevance_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Score from 0 to 100 for relevance of the CV to IT roles. "
            "Evaluate how strongly the CV matches IT-related positions based on education, experience, projects, "
            "technical skills, tools and technologies. "
            "0-20: CV is not relevant to IT roles. "
            "21-40: weak IT relevance with only limited technical content. "
            "41-60: partially relevant to IT but missing important technical evidence. "
            "61-80: clearly relevant to IT roles. "
            "81-100: highly relevant to IT roles with strong technical background, projects and skills."
        ),
    )


class CVQualityAnalysis(BaseModel):
    overall_summary: str = Field(
        description="Short summary of the overall CV quality."
    )

    scores: CVQualityScores = Field(
        description="Individual CV quality scores."
    )

    strengths: List[str] = Field(
        description="Strong aspects of the CV that are clearly visible from the provided text."
    )

    weaknesses: List[str] = Field(
        description="Weak aspects of the CV based only on the provided text."
    )

    missing_or_unclear_sections: List[str] = Field(
        description="Sections or information that are missing, unclear or not detailed enough."
    )

    cv_improvement_recommendations: List[str] = Field(
        description="Practical and honest recommendations for improving CV clarity, structure and completeness."
    )

    evidence_notes: List[str] = Field(
        description="Short notes explaining which parts of the CV text support the analysis."
    )


cv_quality_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant specialized in CV quality analysis for IT job candidates.

Your task is to analyze the quality of the provided CV independently from any specific job posting.

Important rules:
- Analyze only the information present in the CV text.
- Do not invent skills, experience, education, certifications, projects or achievements.
- Do not assume that the candidate has a skill if it is not visible in the CV.
- If something is unclear or only partially presented, explicitly say that it is unclear or partially presented.
- Focus on clarity, structure, completeness, technical skills presentation, experience descriptions, project descriptions and IT relevance.
- Recommendations should improve the CV honestly, without suggesting false information.
- The analysis should be useful for an IT candidate.

Scoring rules:
- Assign all scores as integers from 0 to 100.
- Strictly follow the scoring rubrics defined in the CVQualityScores schema.
- Use the full 0-100 scale consistently across different CVs.
- Do not give high scores when the relevant information is missing, vague, generic or unsupported by the CV text.
- A score above 80 should be given only when that criterion is clearly strong and supported by specific evidence from the CV.
- A score between 60 and 80 means that the criterion is mostly satisfactory, but there are still minor issues or missing details.
- A score between 40 and 60 means that the criterion is partially satisfied, but important information is incomplete, unclear or too generic.
- A score below 40 means that the criterion is weak, mostly missing or poorly presented.
- Every score should be consistent with the strengths, weaknesses, missing_or_unclear_sections and evidence_notes fields.
- Evidence notes should briefly explain which parts of the CV support the assigned scores.

Return the analysis using the required structured schema.
""",
        ),
        (
            "human",
            """
Analyze the following CV text:

{cv_text}
""",
        ),
    ]
)


def create_cv_quality_chain(model_name: str = "gpt-4o-mini", temperature: float = 0):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    structured_llm = llm.with_structured_output(CVQualityAnalysis)

    return cv_quality_prompt | structured_llm


def clamp_score(value: Any) -> int:

    try:
        value = int(value)
    except (ValueError, TypeError):
        value = 0

    return max(0, min(100, value))


def normalize_cv_quality_scores(cv_quality_dict: Dict[str, Any]) -> Dict[str, Any]:

    scores = cv_quality_dict.get("scores", {})

    for score_name, score_value in scores.items():
        scores[score_name] = clamp_score(score_value)

    cv_quality_dict["scores"] = scores

    return cv_quality_dict


def calculate_weighted_score(scores: Dict[str, Optional[float]],weights: Dict[str, float],) -> Optional[float]:

    valid_scores = {
        score_name: score_value
        for score_name, score_value in scores.items()
        if score_value is not None and score_name in weights
    }

    valid_weights = {
        score_name: weights[score_name]
        for score_name in valid_scores
    }

    total_weight = sum(valid_weights.values())

    if total_weight == 0:
        return None

    weighted_score = 0

    for score_name, score_value in valid_scores.items():
        normalized_weight = valid_weights[score_name] / total_weight
        weighted_score += score_value * normalized_weight

    return round(weighted_score, 2)


def get_cv_quality_category(score: Optional[float]) -> str:

    if score is None:
        return "Unknown CV Quality"

    if score < 50:
        return "Weak CV"

    if score < 70:
        return "Basic CV"

    if score < 85:
        return "Good CV"

    return "Strong CV"


def calculate_score_breakdown(scores: Dict[str, float],weights: Dict[str, float],) -> List[Dict[str, Any]]:

    total_weight = sum(
        weights.get(score_name, 0)
        for score_name in scores
        if scores.get(score_name) is not None
    )

    score_breakdown_rows = []

    for score_name, score in scores.items():
        raw_weight = weights.get(score_name, 0)

        if total_weight > 0 and score is not None:
            normalized_weight = raw_weight / total_weight
            weighted_score = round(score * normalized_weight, 2)
        else:
            normalized_weight = 0
            weighted_score = None

        score_breakdown_rows.append(
            {
                "criterion": score_name,
                "score": score,
                "weight": round(normalized_weight, 4),
                "weight_percent": round(normalized_weight * 100, 2),
                "weighted_score": weighted_score,
            }
        )

    return score_breakdown_rows


def format_bullet_list(items: List[str]) -> str:

    if not items:
        return "- No items identified."

    return "\n".join([f"- {item}" for item in items])


def format_score_breakdown_table(score_breakdown_rows: List[Dict[str, Any]]) -> str:

    lines = []
    lines.append("| Criterion | Score | Weight | Weighted score |")
    lines.append("|---|---:|---:|---:|")

    for row in score_breakdown_rows:
        criterion = (
            row["criterion"]
            .replace("_", " ")
            .replace(" score", "")
            .title()
        )

        score = row["score"]
        weight_percent = row["weight_percent"]
        weighted_score = row["weighted_score"]

        lines.append(
            f"| {criterion} | {score} | {weight_percent}% | {weighted_score} |"
        )

    return "\n".join(lines)


def create_cv_quality_markdown_report(cv_quality_dict: Dict[str, Any],score_breakdown_rows: List[Dict[str, Any]],) -> str:

    cv_quality_markdown_report = f"""
# CV Quality Analysis Report

## Final CV Quality Score

**Score:** {cv_quality_dict.get("final_cv_quality_score")}/100  
**Category:** {cv_quality_dict.get("cv_quality_category")}

## Weighting Note

{cv_quality_dict.get("cv_quality_weighting_note")}

## Overall Summary

{cv_quality_dict.get("overall_summary")}

## Score Breakdown

{format_score_breakdown_table(score_breakdown_rows)}

## Strengths

{format_bullet_list(cv_quality_dict.get("strengths", []))}

## Weaknesses

{format_bullet_list(cv_quality_dict.get("weaknesses", []))}

## Missing or Unclear Sections

{format_bullet_list(cv_quality_dict.get("missing_or_unclear_sections", []))}

## CV Improvement Recommendations

{format_bullet_list(cv_quality_dict.get("cv_improvement_recommendations", []))}

## Evidence Notes

{format_bullet_list(cv_quality_dict.get("evidence_notes", []))}
"""

    return cv_quality_markdown_report.strip()


def analyze_cv_quality(cv_text: str,cv_quality_weights: Optional[Dict[str, float]] = None,model_name: str = "gpt-4o-mini",temperature: float = 0,) -> Dict[str, Any]:

    if cv_text is None or not str(cv_text).strip():
        raise ValueError("CV text is empty. Cannot perform CV quality analysis.")

    if cv_quality_weights is None:
        cv_quality_weights = DEFAULT_CV_QUALITY_WEIGHTS

    cv_quality_chain = create_cv_quality_chain(
        model_name=model_name,
        temperature=temperature,
    )

    cv_quality_result = cv_quality_chain.invoke(
        {
            "cv_text": cv_text,
        }
    )

    if hasattr(cv_quality_result, "model_dump"):
        cv_quality_dict = cv_quality_result.model_dump()
    else:
        cv_quality_dict = cv_quality_result.dict()

    cv_quality_dict = normalize_cv_quality_scores(cv_quality_dict)

    final_cv_quality_score = calculate_weighted_score(
        scores=cv_quality_dict["scores"],
        weights=cv_quality_weights,
    )

    cv_quality_category = get_cv_quality_category(final_cv_quality_score)

    score_breakdown_rows = calculate_score_breakdown(
        scores=cv_quality_dict["scores"],
        weights=cv_quality_weights,
    )

    cv_quality_dict["final_cv_quality_score"] = final_cv_quality_score
    cv_quality_dict["cv_quality_category"] = cv_quality_category
    cv_quality_dict["cv_quality_weights_used"] = cv_quality_weights
    cv_quality_dict["cv_quality_weighting_note"] = CV_QUALITY_WEIGHTING_NOTE
    cv_quality_dict["score_breakdown"] = score_breakdown_rows

    return cv_quality_dict


def save_cv_quality_outputs(cv_quality_dict: Dict[str, Any],json_output_path=None,markdown_output_path=None,):

    if json_output_path is None:
        json_output_path = CV_QUALITY_OUTPUT_DIR / "cv_quality_analysis.json"

    if markdown_output_path is None:
        markdown_output_path = CV_QUALITY_OUTPUT_DIR / "cv_quality_report.md"

    score_breakdown_rows = cv_quality_dict.get("score_breakdown", [])

    cv_quality_markdown_report = create_cv_quality_markdown_report(
        cv_quality_dict=cv_quality_dict,
        score_breakdown_rows=score_breakdown_rows,
    )

    saved_json_path = save_json_file(
        data=cv_quality_dict,
        file_path=json_output_path,
    )

    saved_markdown_path = save_text_file(
        text=cv_quality_markdown_report,
        file_path=markdown_output_path,
    )

    return {
        "json_output_path": str(saved_json_path),
        "markdown_output_path": str(saved_markdown_path),
        "markdown_report": cv_quality_markdown_report,
    }


def process_cv_quality_analysis(cv_text: str,cv_quality_weights: Optional[Dict[str, float]] = None,model_name: str = "gpt-4o-mini",temperature: float = 0,json_output_path=None,markdown_output_path=None,) -> Dict[str, Any]:

    cv_quality_dict = analyze_cv_quality(
        cv_text=cv_text,
        cv_quality_weights=cv_quality_weights,
        model_name=model_name,
        temperature=temperature,
    )

    saved_outputs = save_cv_quality_outputs(
        cv_quality_dict=cv_quality_dict,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )

    return {
        "cv_quality_analysis": cv_quality_dict,
        "json_output_path": saved_outputs["json_output_path"],
        "markdown_output_path": saved_outputs["markdown_output_path"],
        "markdown_report": saved_outputs["markdown_report"],
    }