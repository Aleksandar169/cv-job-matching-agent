import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils import (
    PROJECT_ROOT,
    MATCHING_OUTPUT_DIR,
    save_json_file,
    save_text_file,
)


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


DEFAULT_DIRECT_MATCHING_WEIGHTS = {
    "required_skills_score": 0.35,
    "technology_score": 0.25,
    "experience_score": 0.20,
    "education_score": 0.08,
    "nice_to_have_score": 0.04,
    "certification_score": 0.02,
    "language_score": 0.01,
}


INTERNAL_SEMANTIC_MATCHING_WEIGHTS = {
    "responsibilities_alignment_score": 0.40,
    "soft_skills_evidence_score": 0.20,
    "contextual_experience_alignment_score": 0.25,
    "semantic_skill_evidence_score": 0.15,
}


DEFAULT_MATCHING_MODE_WEIGHTS = {
    "direct_matching_score": 0.70,
    "semantic_score": 0.30,
}


MATCHING_WEIGHTING_NOTE = (
    "Direct matching weights and direct-vs-semantic matching weights are configurable values. "
    "Semantic matching weights are internal default values used to compose the semantic score."
)


CV_SKILL_FIELDS = [
    "technical_skills",
    "programming_languages",
    "frameworks_and_libraries",
    "databases",
    "cloud_and_devops_tools",
    "data_and_ai_tools",
    "testing_tools",
    "other_tools",
    "soft_skills",
]


CV_TEXT_EVIDENCE_FIELDS = [
    "profile_summary",
    "technical_skills",
    "programming_languages",
    "frameworks_and_libraries",
    "databases",
    "cloud_and_devops_tools",
    "data_and_ai_tools",
    "testing_tools",
    "other_tools",
    "soft_skills",
    "education",
    "work_experience",
    "projects",
    "certifications",
    "languages",
]


JOB_TECHNOLOGY_FIELDS = [
    "programming_languages",
    "frameworks_and_libraries",
    "databases",
    "cloud_and_devops_tools",
    "data_and_ai_tools",
    "testing_tools",
    "other_tools",
]


class SemanticMatchingAnalysis(BaseModel):
    role_fit_summary: str = Field(
        description=(
            "Concise semantic assessment of how well the candidate fits the job role. "
            "The summary must be evidence-based and must not assume skills, experience or achievements "
            "that are not present in the CV."
        )
    )

    responsibilities_evidenced: List[str] = Field(
        default_factory=list,
        description=(
            "List of job responsibilities that are clearly or reasonably evidenced in the CV. "
            "Responsibilities may be evidenced through work experience, project descriptions, achievements, "
            "technical tasks, collaboration, documentation, implementation work, or similar role-related activities. "
            "Do not include responsibilities that are not supported by the CV."
        ),
    )

    responsibilities_not_evidenced: List[str] = Field(
        default_factory=list,
        description=(
            "List of job responsibilities from the job posting that are not clearly evidenced in the CV. "
            "Use conservative wording. If a responsibility may be partially evidenced but not clear enough, "
            "mark it as not clearly evidenced rather than fully missing."
        ),
    )

    soft_skills_evidenced: List[str] = Field(
        default_factory=list,
        description=(
            "List of requested or implied soft skills that are evidenced in the CV. "
            "Soft skills do not need to appear as exact phrases. They may be evidenced through teamwork, "
            "communication, mentoring, documentation, stakeholder collaboration, problem solving, ownership, "
            "leadership, project work or similar experience."
        ),
    )

    soft_skills_not_clearly_evidenced: List[str] = Field(
        default_factory=list,
        description=(
            "List of requested or implied soft skills that are not clearly evidenced in the CV. "
            "Do not claim that a soft skill is missing only because the exact phrase is not used. "
            "If there is no clear supporting experience or project evidence, mark it as not clearly evidenced."
        ),
    )

    contextual_experience_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "List of CV experience, projects, responsibilities or activities that are contextually relevant "
            "for the job role, even if the exact wording from the job posting is not used. "
            "This field should capture broader role fit, such as similar tasks, similar technical environment, "
            "similar project type, similar domain context, or similar responsibility level."
        ),
    )

    contextual_experience_gaps: List[str] = Field(
        default_factory=list,
        description=(
            "List of important contextual experience areas from the job posting that are not clearly evidenced "
            "in the CV. This may include missing domain context, missing type of project experience, missing role "
            "responsibility, missing collaboration context, or missing evidence of working in a similar environment."
        ),
    )

    semantic_skill_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "List of skills or technologies that may be evidenced semantically through synonyms, abbreviations, "
            "alternative names, related tools or contextual evidence in projects and experience. "
            "This field is used to capture skills that direct/syntactic matching may miss."
        ),
    )

    possible_direct_matching_false_negatives: List[str] = Field(
        default_factory=list,
        description=(
            "List of items that direct matching marked as missing, but semantic analysis suggests may actually "
            "be evidenced in the CV through synonyms, abbreviations, alternative technology names or context. "
            "Only include an item here if there is clear or reasonably strong evidence from the CV."
        ),
    )

    responsibilities_alignment_score: int = Field(
        description=(
            "Score from 0 to 100 that evaluates how well the CV evidences the responsibilities required "
            "by the job posting. "
            "0-20: responsibilities are mostly not evidenced. "
            "21-40: only a small part of the responsibilities is evidenced. "
            "41-60: some responsibilities are evidenced, but important gaps remain. "
            "61-80: most key responsibilities are evidenced. "
            "81-100: responsibilities are strongly evidenced and closely aligned with the job posting."
        )
    )

    soft_skills_evidence_score: int = Field(
        description=(
            "Score from 0 to 100 that evaluates how well requested or implied soft skills are evidenced "
            "in the CV. "
            "0-20: soft skills are mostly not evidenced. "
            "21-40: limited soft skill evidence is present. "
            "41-60: some soft skills are evidenced, but several remain unclear. "
            "61-80: most relevant soft skills are evidenced through experience or projects. "
            "81-100: soft skills are strongly and clearly evidenced through concrete CV content."
        )
    )

    contextual_experience_alignment_score: int = Field(
        description=(
            "Score from 0 to 100 that evaluates whether the candidate's previous experience, projects and "
            "responsibilities are contextually aligned with the job role. "
            "This score is broader than exact skill matching and should consider similar tasks, similar projects, "
            "similar technical environment, similar role expectations and relevant domain context. "
            "0-20: very weak contextual alignment. "
            "21-40: limited contextual alignment. "
            "41-60: partial contextual alignment with noticeable gaps. "
            "61-80: good contextual alignment with most relevant role expectations. "
            "81-100: strong contextual alignment with clear evidence of similar work or project context."
        )
    )

    semantic_skill_evidence_score: int = Field(
        description=(
            "Score from 0 to 100 that evaluates whether skills marked as missing by direct matching may actually "
            "be evidenced semantically in the CV. This includes synonyms, abbreviations, alternative technology "
            "names, related tools or project context. "
            "0-20: almost no semantic evidence for missing skills. "
            "21-40: limited semantic evidence for a small number of missing skills. "
            "41-60: some missing skills may be partially evidenced semantically. "
            "61-80: several missing skills appear to be evidenced through context or alternative wording. "
            "81-100: strong semantic evidence suggests that many direct matching missing items are false negatives."
        )
    )

    evidence_notes: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence-based notes explaining the semantic matching decisions. "
            "Use this field to clarify why certain responsibilities, soft skills, contextual experience areas "
            "or possible direct matching false negatives were identified. "
            "Do not use this field to invent unsupported candidate abilities."
        ),
    )


semantic_matching_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant specialized in semantic CV-job matching.

Your task is to compare a structured CV with a structured IT job posting.

General rules:
- Use only the provided structured CV, structured job posting and direct matching context.
- Do not invent skills, experience, projects, certifications, education or achievements.
- Do not assume that the candidate has a skill if it is not evidenced in the CV.
- Be conservative and evidence-based.
- Return the result using the required structured schema.

Direct matching context:
- Direct matching results are provided as initial results from syntactic/string-based matching.
- These results should not be treated as final decisions.
- Independently check whether some items marked as missing may actually be evidenced through synonyms, abbreviations, alternative technology names or contextual evidence.
- If an item marked as missing appears to be evidenced in the CV, include it in possible_direct_matching_false_negatives and explain it in evidence_notes.
- Do not mark an item as evidenced unless there is clear or reasonably strong CV evidence.

Responsibilities analysis:
- First identify job responsibilities that are evidenced in the CV.
- Then identify job responsibilities that are not clearly evidenced in the CV.
- Then assign responsibilities_alignment_score.

Soft skills analysis:
- First identify requested or implied soft skills.
- Soft skills may be evidenced through teamwork, communication, mentoring, documentation, stakeholder work, problem solving, ownership or project activity.
- Then identify evidenced and not clearly evidenced soft skills.
- Then assign soft_skills_evidence_score.

Contextual experience analysis:
- Analyze whether the candidate's previous experience, projects and responsibilities are contextually relevant for the job.
- This is broader than exact skill matching.
- Consider whether the CV shows similar tasks, similar project context, similar technical environment or similar role expectations.
- Identify contextual_experience_evidence and contextual_experience_gaps.
- Then assign contextual_experience_alignment_score.

Semantic skill evidence analysis:
- Review missing required skills and missing technology skills from direct matching.
- Check whether any of these skills may be evidenced through synonyms, abbreviations, alternative names or project context.
- Identify semantic_skill_evidence.
- Identify possible_direct_matching_false_negatives.
- Then assign semantic_skill_evidence_score.

Scoring:
- responsibilities_alignment_score must be based on responsibility evidence.
- soft_skills_evidence_score must be based on soft skills evidence.
- contextual_experience_alignment_score must be based on contextual experience and project relevance.
- semantic_skill_evidence_score must be based on semantic evidence of skills, synonyms, abbreviations and possible false negatives from direct matching.
- Do not assign high scores without clear evidence from the CV.
""",
        ),
        (
            "human",
            """
Structured CV:

{structured_cv}

Structured job posting:

{structured_job}

Initial direct matching missing required skills:

{missing_required_skills}

Initial direct matching missing technology skills:

{missing_technology_skills}

Perform semantic matching analysis in this order:

1. Identify responsibilities evidenced in the CV.
2. Identify responsibilities not clearly evidenced in the CV.
3. Assign responsibilities_alignment_score.
4. Identify soft skills evidenced in the CV.
5. Identify soft skills not clearly evidenced in the CV.
6. Assign soft_skills_evidence_score.
7. Identify contextual experience evidence.
8. Identify contextual experience gaps.
9. Assign contextual_experience_alignment_score.
10. Review missing required and technology skills for synonyms, abbreviations and contextual evidence.
11. Identify semantic skill evidence.
12. Identify possible direct matching false negatives.
13. Assign semantic_skill_evidence_score.
14. Provide the final role fit summary and evidence notes.
""",
        ),
    ]
)


def normalize_text(text: Any) -> str:

    if text is None:
        return ""

    text = str(text).lower().strip()

    replacements = {
        "asp.net": "aspnet",
        "angular.js": "angular",
        "node.js": "nodejs",
        "vue.js": "vue",
        "react.js": "react",
        "c#": "csharp",
        "c++": "cplusplus",
        ".net": "dotnet",
        "html 5": "html5",
        "j son": "json",
        "json": "json",
        "js": "javascript",
        "ts": "typescript",
        "postgres": "postgresql",
        "postgre sql": "postgresql",
        "gcp": "google cloud platform",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_normalized_phrase(text: str, phrase: str) -> bool:

    if not text or not phrase:
        return False

    phrase_tokens = phrase.split()

    if len(phrase_tokens) == 1:
        pattern = r"\b" + re.escape(phrase) + r"\b"
    else:
        pattern = r"\b" + r"\s+".join([re.escape(token) for token in phrase_tokens]) + r"\b"

    return re.search(pattern, text) is not None


def ensure_list(value: Any) -> List[Any]:

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def extract_text_from_item(item: Any) -> str:

    if item is None:
        return ""

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        values = []

        for key, value in item.items():
            if value is None:
                continue

            if key in ["is_required", "metadata"]:
                continue

            if isinstance(value, bool):
                continue

            values.append(str(value))

        return " ".join(values)

    return str(item)


def clean_list_items(items: Any) -> List[str]:

    cleaned_items = []

    for item in ensure_list(items):
        item_text = extract_text_from_item(item).strip()

        if item_text == "":
            continue

        if item_text.lower() in ["none", "null", "nan", "unknown", "n/a"]:
            continue

        cleaned_items.append(item_text)

    return cleaned_items


def unique_list(items: List[str]) -> List[str]:

    seen = set()
    unique_items = []

    for item in items:
        normalized_item = normalize_text(item)

        if normalized_item == "":
            continue

        if normalized_item not in seen:
            seen.add(normalized_item)
            unique_items.append(item)

    return unique_items


def build_cv_skill_list(structured_cv: Dict[str, Any]) -> List[str]:

    cv_all_skills = []

    for field in CV_SKILL_FIELDS:
        cv_all_skills.extend(clean_list_items(structured_cv.get(field, [])))

    return unique_list(cv_all_skills)


def build_searchable_cv_text(structured_cv: Dict[str, Any]) -> str:

    cv_evidence_parts = []

    for field in CV_TEXT_EVIDENCE_FIELDS:
        values = clean_list_items(structured_cv.get(field, []))
        cv_evidence_parts.extend(values)

    searchable_cv_text = normalize_text(" ".join(cv_evidence_parts))

    return searchable_cv_text


def extract_job_requirement_lists(
    structured_job: Dict[str, Any],
) -> Dict[str, List[str]]:

    job_required_skills = clean_list_items(structured_job.get("required_skills", []))
    job_nice_to_have_skills = clean_list_items(structured_job.get("nice_to_have_skills", []))

    job_technology_skills = []

    for field in JOB_TECHNOLOGY_FIELDS:
        job_technology_skills.extend(clean_list_items(structured_job.get(field, [])))

    return {
        "job_required_skills": unique_list(job_required_skills),
        "job_nice_to_have_skills": unique_list(job_nice_to_have_skills),
        "job_technology_skills": unique_list(job_technology_skills),
    }


def match_items(
    cv_items: List[str],
    job_items: List[str],
    searchable_cv_text: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:

    cv_normalized_map = {}

    for cv_item in cv_items:
        normalized_cv_item = normalize_text(cv_item)

        if normalized_cv_item:
            cv_normalized_map[normalized_cv_item] = cv_item

    matched_items = []
    missing_items = []

    for job_item in job_items:
        normalized_job_item = normalize_text(job_item)

        if normalized_job_item == "":
            continue

        if normalized_job_item in cv_normalized_map:
            matched_items.append(
                {
                    "job_requirement": job_item,
                    "cv_evidence": cv_normalized_map[normalized_job_item],
                    "match_type": "exact_skill_match",
                }
            )

        elif contains_normalized_phrase(searchable_cv_text, normalized_job_item):
            matched_items.append(
                {
                    "job_requirement": job_item,
                    "cv_evidence": "Found in structured CV evidence text",
                    "match_type": "text_evidence_match",
                }
            )

        else:
            missing_items.append(job_item)

    return matched_items, missing_items


def calculate_match_percentage(
    matched_items: List[Any],
    total_items: int,
    empty_result: Optional[float] = None,
) -> Optional[float]:

    if total_items == 0:
        return empty_result

    return round((len(matched_items) / total_items) * 100, 2)


def extract_number_from_text(text: Any) -> Optional[int]:

    if text is None:
        return None

    match = re.search(r"\d+", str(text))

    if match:
        return int(match.group())

    return None


def get_job_minimum_years(structured_job: Dict[str, Any]) -> Optional[int]:

    experience_requirements = structured_job.get("experience_requirements", [])

    if not isinstance(experience_requirements, list):
        return None

    years = []

    for requirement in experience_requirements:
        if not isinstance(requirement, dict):
            continue

        minimum_years = extract_number_from_text(requirement.get("minimum_years"))

        if minimum_years is not None:
            years.append(minimum_years)

    if len(years) == 0:
        return None

    return max(years)


def get_cv_years_of_experience(structured_cv: Dict[str, Any]) -> Optional[int]:

    possible_fields = [
        "years_of_experience",
        "total_years_of_experience",
        "professional_experience_years",
    ]

    for field in possible_fields:
        years = extract_number_from_text(structured_cv.get(field))

        if years is not None:
            return years

    return None


def calculate_experience_score(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
) -> Dict[str, Any]:

    job_minimum_years = get_job_minimum_years(structured_job)
    cv_years_of_experience = get_cv_years_of_experience(structured_cv)

    if job_minimum_years is None:
        experience_score = None
        experience_note = (
            "The job posting does not clearly specify minimum years of experience, "
            "so experience is not included in the matching score."
        )

    elif cv_years_of_experience is None:
        experience_score = 0
        experience_note = (
            "The job posting specifies experience requirements, "
            "but years of experience are not clearly evidenced in the CV."
        )

    elif cv_years_of_experience >= job_minimum_years:
        experience_score = 100
        experience_note = (
            "The CV meets or exceeds the minimum years of experience requirement."
        )

    else:
        experience_score = round((cv_years_of_experience / job_minimum_years) * 100, 2)
        experience_note = (
            "The CV shows fewer years of experience than required by the job posting."
        )

    return {
        "job_minimum_years": job_minimum_years,
        "cv_years_of_experience": cv_years_of_experience,
        "experience_score": experience_score,
        "experience_note": experience_note,
    }


def calculate_education_score(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
) -> Dict[str, Any]:

    job_education_requirements = clean_list_items(structured_job.get("education_requirements", []))
    cv_education = clean_list_items(structured_cv.get("education", []))

    if len(job_education_requirements) == 0:
        education_score = None
        education_note = (
            "The job posting does not clearly specify education requirements, "
            "so education is not included in the matching score."
        )

    elif len(cv_education) == 0:
        education_score = 0
        education_note = (
            "The job posting specifies education requirements, "
            "but education is not clearly evidenced in the CV."
        )

    else:
        education_score = None
        education_note = (
            "Education is present in the CV, but direct matching cannot reliably determine "
            "whether it is equivalent to the job education requirement. "
            "This component should be reviewed semantically or manually."
        )

    return {
        "job_education_requirements": job_education_requirements,
        "cv_education": cv_education,
        "education_score": education_score,
        "education_note": education_note,
    }


def calculate_weighted_score(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> Optional[float]:

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


def calculate_direct_matching(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    direct_matching_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:

    if direct_matching_weights is None:
        direct_matching_weights = DEFAULT_DIRECT_MATCHING_WEIGHTS

    cv_all_skills = build_cv_skill_list(structured_cv)
    searchable_cv_text = build_searchable_cv_text(structured_cv)

    job_requirement_lists = extract_job_requirement_lists(structured_job)

    job_required_skills = job_requirement_lists["job_required_skills"]
    job_nice_to_have_skills = job_requirement_lists["job_nice_to_have_skills"]
    job_technology_skills = job_requirement_lists["job_technology_skills"]

    matched_required_skills, missing_required_skills = match_items(
        cv_items=cv_all_skills,
        job_items=job_required_skills,
        searchable_cv_text=searchable_cv_text,
    )

    matched_nice_to_have_skills, missing_nice_to_have_skills = match_items(
        cv_items=cv_all_skills,
        job_items=job_nice_to_have_skills,
        searchable_cv_text=searchable_cv_text,
    )

    matched_technology_skills, missing_technology_skills = match_items(
        cv_items=cv_all_skills,
        job_items=job_technology_skills,
        searchable_cv_text=searchable_cv_text,
    )

    required_skills_score = calculate_match_percentage(
        matched_items=matched_required_skills,
        total_items=len(job_required_skills),
    )

    nice_to_have_score = calculate_match_percentage(
        matched_items=matched_nice_to_have_skills,
        total_items=len(job_nice_to_have_skills),
    )

    technology_score = calculate_match_percentage(
        matched_items=matched_technology_skills,
        total_items=len(job_technology_skills),
    )

    experience_analysis = calculate_experience_score(
        structured_cv=structured_cv,
        structured_job=structured_job,
    )

    education_analysis = calculate_education_score(
        structured_cv=structured_cv,
        structured_job=structured_job,
    )

    cv_certifications = clean_list_items(structured_cv.get("certifications", []))
    job_certifications = clean_list_items(structured_job.get("certifications", []))

    cv_languages = clean_list_items(structured_cv.get("languages", []))
    job_language_requirements = clean_list_items(structured_job.get("language_requirements", []))

    matched_certifications, missing_certifications = match_items(
        cv_items=cv_certifications,
        job_items=job_certifications,
        searchable_cv_text=searchable_cv_text,
    )

    matched_languages, missing_languages = match_items(
        cv_items=cv_languages,
        job_items=job_language_requirements,
        searchable_cv_text=searchable_cv_text,
    )

    certification_score = calculate_match_percentage(
        matched_items=matched_certifications,
        total_items=len(job_certifications),
    )

    language_score = calculate_match_percentage(
        matched_items=matched_languages,
        total_items=len(job_language_requirements),
    )

    direct_matching_component_scores = {
        "required_skills_score": required_skills_score,
        "technology_score": technology_score,
        "experience_score": experience_analysis["experience_score"],
        "education_score": education_analysis["education_score"],
        "nice_to_have_score": nice_to_have_score,
        "certification_score": certification_score,
        "language_score": language_score,
    }

    direct_matching_score = calculate_weighted_score(
        scores=direct_matching_component_scores,
        weights=direct_matching_weights,
    )

    return {
        "direct_matching_score": direct_matching_score,
        "direct_matching_component_scores": direct_matching_component_scores,
        "direct_matching_weights": direct_matching_weights,
        "matched_items": {
            "matched_required_skills": matched_required_skills,
            "matched_nice_to_have_skills": matched_nice_to_have_skills,
            "matched_technology_skills": matched_technology_skills,
            "matched_certifications": matched_certifications,
            "matched_languages": matched_languages,
        },
        "missing_items": {
            "missing_required_skills": missing_required_skills,
            "missing_nice_to_have_skills": missing_nice_to_have_skills,
            "missing_technology_skills": missing_technology_skills,
            "missing_certifications": missing_certifications,
            "missing_languages": missing_languages,
        },
        "experience_analysis": experience_analysis,
        "education_analysis": education_analysis,
        "cv_all_skills": cv_all_skills,
        "searchable_cv_text": searchable_cv_text,
        "job_requirement_lists": job_requirement_lists,
    }


def create_semantic_matching_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    structured_semantic_llm = llm.with_structured_output(SemanticMatchingAnalysis)

    return semantic_matching_prompt | structured_semantic_llm


def model_to_dict(model_result: Any) -> Dict[str, Any]:

    if hasattr(model_result, "model_dump"):
        return model_result.model_dump()

    return model_result.dict()


def run_semantic_matching(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    missing_required_skills: List[str],
    missing_technology_skills: List[str],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    semantic_matching_chain = create_semantic_matching_chain(
        model_name=model_name,
        temperature=temperature,
    )

    semantic_matching_result = semantic_matching_chain.invoke(
        {
            "structured_cv": json.dumps(structured_cv, indent=2, ensure_ascii=False),
            "structured_job": json.dumps(structured_job, indent=2, ensure_ascii=False),
            "missing_required_skills": json.dumps(missing_required_skills, indent=2, ensure_ascii=False),
            "missing_technology_skills": json.dumps(missing_technology_skills, indent=2, ensure_ascii=False),
        }
    )

    semantic_matching_dict = model_to_dict(semantic_matching_result)

    return semantic_matching_dict


def clamp_score(value: Any) -> int:

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0

    return max(0, min(100, value))


def normalize_semantic_scores(
    semantic_matching_dict: Dict[str, Any],
) -> Dict[str, Any]:

    semantic_score_fields = [
        "responsibilities_alignment_score",
        "soft_skills_evidence_score",
        "contextual_experience_alignment_score",
        "semantic_skill_evidence_score",
    ]

    for field in semantic_score_fields:
        semantic_matching_dict[field] = clamp_score(semantic_matching_dict.get(field))

    return semantic_matching_dict


def calculate_semantic_score(
    semantic_matching_dict: Dict[str, Any],
    semantic_matching_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:

    if semantic_matching_weights is None:
        semantic_matching_weights = INTERNAL_SEMANTIC_MATCHING_WEIGHTS

    semantic_component_scores = {
        "responsibilities_alignment_score": semantic_matching_dict.get("responsibilities_alignment_score"),
        "soft_skills_evidence_score": semantic_matching_dict.get("soft_skills_evidence_score"),
        "contextual_experience_alignment_score": semantic_matching_dict.get("contextual_experience_alignment_score"),
        "semantic_skill_evidence_score": semantic_matching_dict.get("semantic_skill_evidence_score"),
    }

    semantic_score = calculate_weighted_score(
        scores=semantic_component_scores,
        weights=semantic_matching_weights,
    )

    return {
        "semantic_score": semantic_score,
        "semantic_component_scores": semantic_component_scores,
        "semantic_matching_weights": semantic_matching_weights,
    }


def define_match_category(score: Optional[float]) -> str:

    if score is None:
        return "Unknown match"

    if score >= 85:
        return "Strong match"

    if score >= 70:
        return "Good match"

    if score >= 50:
        return "Partial match"

    return "Weak match"


def create_matching_result(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    direct_matching_result: Dict[str, Any],
    semantic_matching_dict: Dict[str, Any],
    semantic_score_result: Dict[str, Any],
    final_hybrid_score: Optional[float],
    matching_mode_weights: Dict[str, float],
    cv_source: Optional[str] = None,
    job_source: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:

    match_category = define_match_category(final_hybrid_score)

    direct_component_scores = direct_matching_result["direct_matching_component_scores"]
    semantic_component_scores = semantic_score_result["semantic_component_scores"]

    matching_result = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "matching_type": "hybrid_direct_and_llm_semantic_matching",
            "cv_source": cv_source,
            "job_source": job_source,
            "llm_model": model_name,
        },
        "job_information": {
            "job_title": structured_job.get("job_title"),
            "company_name": structured_job.get("company_name"),
            "job_category": structured_job.get("job_category"),
            "location": structured_job.get("location"),
            "work_mode": structured_job.get("work_mode"),
            "employment_type": structured_job.get("employment_type"),
        },
        "final_result": {
            "final_hybrid_score": final_hybrid_score,
            "direct_matching_score": direct_matching_result.get("direct_matching_score"),
            "semantic_score": semantic_score_result.get("semantic_score"),
            "match_category": match_category,
        },
        "weights_used": {
            "direct_matching_weights": direct_matching_result.get("direct_matching_weights"),
            "internal_semantic_matching_weights": semantic_score_result.get("semantic_matching_weights"),
            "matching_mode_weights": matching_mode_weights,
            "weighting_note": MATCHING_WEIGHTING_NOTE,
        },
        "score_breakdown": {
            "required_skills_score": direct_component_scores.get("required_skills_score"),
            "technology_score": direct_component_scores.get("technology_score"),
            "experience_score": direct_component_scores.get("experience_score"),
            "education_score": direct_component_scores.get("education_score"),
            "nice_to_have_score": direct_component_scores.get("nice_to_have_score"),
            "certification_score": direct_component_scores.get("certification_score"),
            "language_score": direct_component_scores.get("language_score"),
            "direct_matching_score": direct_matching_result.get("direct_matching_score"),
            "responsibilities_alignment_score": semantic_component_scores.get("responsibilities_alignment_score"),
            "soft_skills_evidence_score": semantic_component_scores.get("soft_skills_evidence_score"),
            "contextual_experience_alignment_score": semantic_component_scores.get("contextual_experience_alignment_score"),
            "semantic_skill_evidence_score": semantic_component_scores.get("semantic_skill_evidence_score"),
            "semantic_score": semantic_score_result.get("semantic_score"),
            "final_hybrid_score": final_hybrid_score,
        },
        "matched_items": direct_matching_result.get("matched_items", {}),
        "missing_items": direct_matching_result.get("missing_items", {}),
        "experience_analysis": direct_matching_result.get("experience_analysis", {}),
        "education_analysis": direct_matching_result.get("education_analysis", {}),
        "semantic_analysis": semantic_matching_dict,
    }

    return matching_result


def calculate_complete_matching_result(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    direct_matching_weights: Optional[Dict[str, float]] = None,
    matching_mode_weights: Optional[Dict[str, float]] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    cv_source: Optional[str] = None,
    job_source: Optional[str] = None,
) -> Dict[str, Any]:

    if direct_matching_weights is None:
        direct_matching_weights = DEFAULT_DIRECT_MATCHING_WEIGHTS

    if matching_mode_weights is None:
        matching_mode_weights = DEFAULT_MATCHING_MODE_WEIGHTS

    direct_matching_result = calculate_direct_matching(
        structured_cv=structured_cv,
        structured_job=structured_job,
        direct_matching_weights=direct_matching_weights,
    )

    missing_required_skills = direct_matching_result["missing_items"].get("missing_required_skills", [])
    missing_technology_skills = direct_matching_result["missing_items"].get("missing_technology_skills", [])

    semantic_matching_dict = run_semantic_matching(
        structured_cv=structured_cv,
        structured_job=structured_job,
        missing_required_skills=missing_required_skills,
        missing_technology_skills=missing_technology_skills,
        model_name=model_name,
        temperature=temperature,
    )

    semantic_matching_dict = normalize_semantic_scores(semantic_matching_dict)

    semantic_score_result = calculate_semantic_score(
        semantic_matching_dict=semantic_matching_dict,
        semantic_matching_weights=INTERNAL_SEMANTIC_MATCHING_WEIGHTS,
    )

    matching_mode_scores = {
        "direct_matching_score": direct_matching_result.get("direct_matching_score"),
        "semantic_score": semantic_score_result.get("semantic_score"),
    }

    final_hybrid_score = calculate_weighted_score(
        scores=matching_mode_scores,
        weights=matching_mode_weights,
    )

    matching_result = create_matching_result(
        structured_cv=structured_cv,
        structured_job=structured_job,
        direct_matching_result=direct_matching_result,
        semantic_matching_dict=semantic_matching_dict,
        semantic_score_result=semantic_score_result,
        final_hybrid_score=final_hybrid_score,
        matching_mode_weights=matching_mode_weights,
        cv_source=cv_source,
        job_source=job_source,
        model_name=model_name,
    )

    return matching_result


def create_markdown_list(
    items: List[Any],
    key: Optional[str] = None,
    empty_message: str = "- No items found.",
) -> str:

    if not items:
        return empty_message

    lines = []

    for item in items:
        if isinstance(item, dict) and key is not None:
            value = item.get(key)
        else:
            value = item

        if value is not None and str(value).strip() != "":
            lines.append(f"- {value}")

    if not lines:
        return empty_message

    return "\n".join(lines)


def format_score_for_report(score: Optional[float]) -> str:

    if score is None:
        return "Not included in score"

    return f"{score}/100"


def format_weights_for_report(weights: Dict[str, float]) -> str:

    lines = []

    for weight_name, weight_value in weights.items():
        formatted_name = weight_name.replace("_", " ").title()
        lines.append(f"- {formatted_name}: {weight_value}")

    return "\n".join(lines)


def create_matching_markdown_report(matching_result: Dict[str, Any]) -> str:

    job_information = matching_result.get("job_information", {})
    final_result = matching_result.get("final_result", {})
    weights_used = matching_result.get("weights_used", {})
    score_breakdown = matching_result.get("score_breakdown", {})
    matched_items = matching_result.get("matched_items", {})
    missing_items = matching_result.get("missing_items", {})
    experience_analysis = matching_result.get("experience_analysis", {})
    education_analysis = matching_result.get("education_analysis", {})
    semantic_analysis = matching_result.get("semantic_analysis", {})

    report = f"""
# CV-Job Matching Report

## Job Information

- Job title: {job_information.get("job_title")}
- Company: {job_information.get("company_name")}
- Job category: {job_information.get("job_category")}
- Location: {job_information.get("location")}
- Work mode: {job_information.get("work_mode")}
- Employment type: {job_information.get("employment_type")}

## Final Hybrid Matching Result

- Final hybrid match score: {format_score_for_report(final_result.get("final_hybrid_score"))}
- Match category: {final_result.get("match_category")}
- Direct matching score: {format_score_for_report(final_result.get("direct_matching_score"))}
- LLM semantic score: {format_score_for_report(final_result.get("semantic_score"))}

## Weighting Note

{weights_used.get("weighting_note")}

## Direct Matching Weights Used

{format_weights_for_report(weights_used.get("direct_matching_weights", {}))}

## Internal Semantic Matching Weights Used

{format_weights_for_report(weights_used.get("internal_semantic_matching_weights", {}))}

## Matching Mode Weights Used

{format_weights_for_report(weights_used.get("matching_mode_weights", {}))}

## Direct Matching Score Breakdown

- Required skills score: {format_score_for_report(score_breakdown.get("required_skills_score"))}
- Technology score: {format_score_for_report(score_breakdown.get("technology_score"))}
- Experience score: {format_score_for_report(score_breakdown.get("experience_score"))}
- Education score: {format_score_for_report(score_breakdown.get("education_score"))}
- Nice-to-have score: {format_score_for_report(score_breakdown.get("nice_to_have_score"))}
- Certification score: {format_score_for_report(score_breakdown.get("certification_score"))}
- Language score: {format_score_for_report(score_breakdown.get("language_score"))}

## Matched Required Skills

{create_markdown_list(matched_items.get("matched_required_skills", []), key="job_requirement", empty_message="- No required skills clearly matched.")}

## Missing Required Skills

{create_markdown_list(missing_items.get("missing_required_skills", []), empty_message="- No required skills are missing.")}

## Matched Nice-to-Have Skills

{create_markdown_list(matched_items.get("matched_nice_to_have_skills", []), key="job_requirement", empty_message="- No nice-to-have skills clearly matched.")}

## Missing Nice-to-Have Skills

{create_markdown_list(missing_items.get("missing_nice_to_have_skills", []), empty_message="- No nice-to-have skills are missing.")}

## Matched Technology Skills

{create_markdown_list(matched_items.get("matched_technology_skills", []), key="job_requirement", empty_message="- No technology skills clearly matched.")}

## Missing Technology Skills

{create_markdown_list(missing_items.get("missing_technology_skills", []), empty_message="- No technology skills are missing.")}

## Experience Analysis

- Job minimum years: {experience_analysis.get("job_minimum_years")}
- CV years of experience: {experience_analysis.get("cv_years_of_experience")}
- Experience score: {format_score_for_report(experience_analysis.get("experience_score"))}
- Note: {experience_analysis.get("experience_note")}

## Education Analysis

- Education score: {format_score_for_report(education_analysis.get("education_score"))}
- Note: {education_analysis.get("education_note")}

## Certifications

### Matched Certifications

{create_markdown_list(matched_items.get("matched_certifications", []), key="job_requirement", empty_message="- No certifications clearly matched.")}

### Missing Certifications

{create_markdown_list(missing_items.get("missing_certifications", []), empty_message="- No certifications are missing.")}

## Language Requirements

### Matched Languages

{create_markdown_list(matched_items.get("matched_languages", []), key="job_requirement", empty_message="- No language requirements clearly matched.")}

### Missing Languages

{create_markdown_list(missing_items.get("missing_languages", []), empty_message="- No language requirements are missing.")}

## LLM Semantic Analysis

### Role Fit Summary

{semantic_analysis.get("role_fit_summary")}

### Responsibilities Alignment Score

{format_score_for_report(semantic_analysis.get("responsibilities_alignment_score"))}

### Soft Skills Evidence Score

{format_score_for_report(semantic_analysis.get("soft_skills_evidence_score"))}

### Contextual Experience Alignment Score

{format_score_for_report(semantic_analysis.get("contextual_experience_alignment_score"))}

### Semantic Skill Evidence Score

{format_score_for_report(semantic_analysis.get("semantic_skill_evidence_score"))}

### Responsibilities Evidenced in CV

{create_markdown_list(semantic_analysis.get("responsibilities_evidenced", []), empty_message="- No responsibilities clearly evidenced.")}

### Responsibilities Not Clearly Evidenced

{create_markdown_list(semantic_analysis.get("responsibilities_not_evidenced", []), empty_message="- No responsibilities listed as unclear.")}

### Soft Skills Evidenced in CV

{create_markdown_list(semantic_analysis.get("soft_skills_evidenced", []), empty_message="- No soft skills clearly evidenced.")}

### Soft Skills Not Clearly Evidenced

{create_markdown_list(semantic_analysis.get("soft_skills_not_clearly_evidenced", []), empty_message="- No soft skills listed as unclear.")}

### Contextual Experience Evidence

{create_markdown_list(semantic_analysis.get("contextual_experience_evidence", []), empty_message="- No contextual experience evidence listed.")}

### Contextual Experience Gaps

{create_markdown_list(semantic_analysis.get("contextual_experience_gaps", []), empty_message="- No contextual experience gaps listed.")}

### Semantic Skill Evidence

{create_markdown_list(semantic_analysis.get("semantic_skill_evidence", []), empty_message="- No semantic skill evidence listed.")}

### Possible Direct Matching False Negatives

{create_markdown_list(semantic_analysis.get("possible_direct_matching_false_negatives", []), empty_message="- No possible direct matching false negatives identified.")}

### Evidence Notes

{create_markdown_list(semantic_analysis.get("evidence_notes", []), empty_message="- No semantic evidence notes provided.")}
"""

    return report.strip()


def save_matching_outputs(
    matching_result: Dict[str, Any],
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    if json_output_path is None:
        json_output_path = MATCHING_OUTPUT_DIR / "matching_result.json"

    if markdown_output_path is None:
        markdown_output_path = MATCHING_OUTPUT_DIR / "matching_report.md"

    markdown_report = create_matching_markdown_report(matching_result)

    saved_json_path = save_json_file(
        data=matching_result,
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


def process_matching(
    structured_cv: Dict[str, Any],
    structured_job: Dict[str, Any],
    direct_matching_weights: Optional[Dict[str, float]] = None,
    matching_mode_weights: Optional[Dict[str, float]] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    cv_source: Optional[str] = None,
    job_source: Optional[str] = None,
    json_output_path=None,
    markdown_output_path=None,
) -> Dict[str, Any]:

    matching_result = calculate_complete_matching_result(
        structured_cv=structured_cv,
        structured_job=structured_job,
        direct_matching_weights=direct_matching_weights,
        matching_mode_weights=matching_mode_weights,
        model_name=model_name,
        temperature=temperature,
        cv_source=cv_source,
        job_source=job_source,
    )

    saved_outputs = save_matching_outputs(
        matching_result=matching_result,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )

    return {
        "matching_result": matching_result,
        "json_output_path": saved_outputs["json_output_path"],
        "markdown_output_path": saved_outputs["markdown_output_path"],
        "markdown_report": saved_outputs["markdown_report"],
    }