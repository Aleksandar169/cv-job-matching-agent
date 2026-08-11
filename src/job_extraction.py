import os
import re
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo import MongoClient

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils import (
    PROJECT_ROOT,
    JOB_EXTRACTION_OUTPUT_DIR,
    save_json_file,
)


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class ExperienceRequirement(BaseModel):
    minimum_years: Optional[str] = Field(
        default=None,
        description="Minimum years of experience required for this specific requirement, if explicitly mentioned."
    )

    seniority_level: Optional[str] = Field(
        default=None,
        description="Seniority level related to this requirement, such as Junior, Mid, Medior, Senior, Lead, if visible or clearly inferable."
    )

    related_area_or_skill: Optional[str] = Field(
        default=None,
        description="Technology, role, domain or skill area related to this experience requirement, if mentioned."
    )

    experience_description: str = Field(
        description="One specific experience requirement explicitly mentioned in the job posting."
    )

    is_required: Optional[bool] = Field(
        default=None,
        description="True if this experience is required, False if it is preferred or nice-to-have, None if unclear."
    )


class EducationRequirement(BaseModel):
    degree_required: Optional[str] = Field(
        default=None,
        description="Required or preferred degree or education level, if explicitly mentioned."
    )

    field_of_study: Optional[str] = Field(
        default=None,
        description="Required or preferred field of study, if mentioned."
    )

    education_description: str = Field(
        description="One specific education requirement explicitly mentioned in the job posting."
    )

    is_required: Optional[bool] = Field(
        default=None,
        description="True if this education requirement is required, False if it is preferred or optional, None if unclear."
    )


class StructuredJobPosting(BaseModel):
    job_title: Optional[str] = Field(
        default=None,
        description="Job title extracted from the posting."
    )

    company_name: Optional[str] = Field(
        default=None,
        description="Company name, if visible in the job posting."
    )

    location: Optional[str] = Field(
        default=None,
        description="Job location, if visible."
    )

    work_mode: Optional[str] = Field(
        default=None,
        description="Work mode such as remote, hybrid, on-site or unknown."
    )

    employment_type: Optional[str] = Field(
        default=None,
        description="Employment type such as full-time, part-time, contract, internship or unknown."
    )

    job_category: Optional[str] = Field(
        default=None,
        description="Broad IT job category, such as Frontend Development, Backend Development, Data Analytics, DevOps, QA, Cybersecurity, etc."
    )

    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills that are clearly required in the job posting."
    )

    nice_to_have_skills: List[str] = Field(
        default_factory=list,
        description="Skills that are listed as preferred, optional or nice-to-have."
    )

    programming_languages: List[str] = Field(
        default_factory=list,
        description="Programming languages mentioned in the job posting."
    )

    frameworks_and_libraries: List[str] = Field(
        default_factory=list,
        description="Frameworks and libraries mentioned in the job posting."
    )

    databases: List[str] = Field(
        default_factory=list,
        description="Databases mentioned in the job posting."
    )

    cloud_and_devops_tools: List[str] = Field(
        default_factory=list,
        description="Cloud platforms, DevOps tools and infrastructure tools mentioned in the job posting."
    )

    data_and_ai_tools: List[str] = Field(
        default_factory=list,
        description="Data analysis, BI, machine learning or AI tools mentioned in the job posting."
    )

    testing_tools: List[str] = Field(
        default_factory=list,
        description="Testing or QA tools mentioned in the job posting."
    )

    other_tools: List[str] = Field(
        default_factory=list,
        description="Other software tools mentioned in the job posting."
    )

    responsibilities: List[str] = Field(
        default_factory=list,
        description="Main responsibilities and tasks described in the job posting."
    )

    experience_requirements: List[ExperienceRequirement] = Field(
        default_factory=list,
        description=(
            "List of experience and seniority requirements explicitly mentioned in the job posting. "
            "Each item should represent one separate requirement, such as total years of experience, "
            "experience with a specific technology, previous role experience, domain experience or seniority level."
        )
    )

    education_requirements: List[EducationRequirement] = Field(
        default_factory=list,
        description=(
            "List of education requirements explicitly mentioned in the job posting. "
            "Each item should represent one separate education-related requirement, such as degree level, "
            "field of study, formal education, or equivalent practical experience."
        )
    )

    certifications: List[str] = Field(
        default_factory=list,
        description="Required or preferred certifications mentioned in the posting."
    )

    language_requirements: List[str] = Field(
        default_factory=list,
        description="Human language requirements mentioned in the job posting."
    )

    soft_skills: List[str] = Field(
        default_factory=list,
        description="Soft skills mentioned in the job posting."
    )

    unclear_or_missing_information: List[str] = Field(
        default_factory=list,
        description=(
            "Important job posting information that is missing, unclear or not explicitly stated, "
            "based only on the information needed for CV-job matching. "
            "Check whether the posting clearly states: job title, company name, location, work mode, "
            "employment type, seniority or experience level, required technical skills, nice-to-have skills, "
            "main responsibilities, education requirements, certifications and language requirements. "
            "Do not list irrelevant missing information."
        )
    )


job_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant specialized in structured extraction from IT job postings.

Your task is to extract structured information from the provided IT job advertisement text.

General rules:
- Extract only information that is explicitly present in the job posting.
- Do not invent skills, tools, experience, education, certifications, responsibilities or company information.
- If information is not clearly mentioned, return null or an empty list.
- Do not return the string "None", "N/A" or "Unknown" for missing values.
- Avoid duplicate values inside the same list.
- Keep extracted values concise and useful for later CV-job matching.

Required and nice-to-have skills:
- Separate required skills from nice-to-have skills whenever possible.
- Classify a skill as required if the posting uses words such as "required", "must have", "need", "minimum", "strong experience", "proficiency", or if the skill is listed under requirements.
- Classify a skill as nice-to-have if the posting uses words such as "preferred", "nice to have", "plus", "advantage", "bonus", or "familiarity with".
- If it is unclear whether a skill is required or optional, classify it as required only if the text strongly suggests that it is necessary for the role.

Field interpretation:
- employment_type means the contract or engagement type, such as Full-time, Part-time, Contract, Freelance, Internship or Temporary.
- work_mode means the working arrangement, such as Remote, Hybrid, On-site, or a clearly stated arrangement such as "On-site 75% of the time".
- Do not confuse employment_type with work_mode.
- job_category should be a broad IT category, such as Software Development, Frontend Development, Backend Development, Full Stack Development, Data Analytics, Data Science / AI, DevOps / Cloud, QA / Testing, Cybersecurity, or IT Support / Administration.

Technology extraction:
- Focus on IT-related information: programming languages, frameworks, libraries, databases, cloud platforms, DevOps tools, testing tools, data tools, AI tools and software tools.
- Normalize common technology names when possible:
  - JS -> JavaScript
  - TS -> TypeScript
  - GCP -> Google Cloud Platform
  - Postgres -> PostgreSQL
  - HTML 5 -> HTML5
  - jSON -> JSON
- Do not place the same technology in many unrelated categories. For example, React should be a framework/library, Python should be a programming language, and PostgreSQL should be a database.

Experience and education:
- Extract experience requirements only if the posting mentions years of experience, seniority, previous role experience, or experience in a specific area.
- If seniority level is not clearly stated, return null.
- Extract education requirements only if the posting mentions degree, formal education, technical certification, or equivalent practical experience.
- If education is not mentioned, leave education_requirements as an empty list.

Unclear or missing information:
- Add an item to unclear_or_missing_information only if the information is truly missing or unclear and useful for later CV-job matching.
- Before adding a missing-information note, check whether the same information was already extracted in another field.
- Do not say that employment type is missing if employment_type has a value.
- Do not say that work mode is missing if work_mode has a value.
- Do not say that location is missing if location has a value.
- Do not say that certifications are missing if certifications has at least one item.
- Do not say that language requirements are missing if language_requirements has at least one item.
- Do not add generic missing notes that contradict extracted fields.

Task boundaries:
- Do not compare the job posting with a CV.
- Do not generate candidate recommendations.
- Do not evaluate candidate fit.

Return the result using the required structured schema.
"""
        ),
        (
            "human",
            """
Extract structured information from the following IT job posting:

{job_text}
"""
        )
    ]
)


def normalize_text_for_hash(text: Optional[str]) -> str:

    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def create_job_key(job_text: str) -> str:

    normalized_text = normalize_text_for_hash(job_text)

    text_hash = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()

    return f"text_hash_{text_hash}"


def create_job_extraction_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    structured_llm = llm.with_structured_output(StructuredJobPosting)

    return job_extraction_prompt | structured_llm


def model_to_dict(model_result: Any) -> Dict[str, Any]:

    if hasattr(model_result, "model_dump"):
        return model_result.model_dump()

    return model_result.dict()


def has_meaningful_value(value: Any) -> bool:

    if value is None:
        return False

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, str):
        return value.strip().lower() not in ["", "none", "null", "nan", "unknown", "n/a"]

    return True


def normalize_list_field(data: Dict[str, Any], field_name: str) -> Dict[str, Any]:

    value = data.get(field_name)

    if value is None:
        data[field_name] = []

    elif isinstance(value, list):
        data[field_name] = value

    else:
        data[field_name] = [value]

    return data


def normalize_structured_job(structured_job_dict: Dict[str, Any]) -> Dict[str, Any]:

    list_fields = [
        "required_skills",
        "nice_to_have_skills",
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "cloud_and_devops_tools",
        "data_and_ai_tools",
        "testing_tools",
        "other_tools",
        "responsibilities",
        "experience_requirements",
        "education_requirements",
        "certifications",
        "language_requirements",
        "soft_skills",
        "unclear_or_missing_information",
    ]

    for field_name in list_fields:
        structured_job_dict = normalize_list_field(
            data=structured_job_dict,
            field_name=field_name,
        )

    return structured_job_dict


def clean_unclear_or_missing_information(
    structured_job_dict: Dict[str, Any],
) -> Dict[str, Any]:

    unclear_items = structured_job_dict.get("unclear_or_missing_information", [])

    if not isinstance(unclear_items, list):
        unclear_items = []

    cleaned_items = []

    for item in unclear_items:
        item_text = str(item)
        item_lower = item_text.lower()

        if "employment type" in item_lower and has_meaningful_value(structured_job_dict.get("employment_type")):
            continue

        if "work mode" in item_lower and has_meaningful_value(structured_job_dict.get("work_mode")):
            continue

        if "location" in item_lower and has_meaningful_value(structured_job_dict.get("location")):
            continue

        if "certification" in item_lower and has_meaningful_value(structured_job_dict.get("certifications")):
            continue

        if "language requirement" in item_lower and has_meaningful_value(structured_job_dict.get("language_requirements")):
            continue

        if "required skill" in item_lower and has_meaningful_value(structured_job_dict.get("required_skills")):
            continue

        cleaned_items.append(item_text)

    structured_job_dict["unclear_or_missing_information"] = cleaned_items

    return structured_job_dict


def add_structured_job_metadata(
    structured_job_dict: Dict[str, Any],
    job_key: Optional[str] = None,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:

    structured_job_dict["metadata"] = {
        "job_key": job_key,
        "model": model_name,
        "extraction_type": "structured_job_extraction",
        "source": source,
        "source_file": source_file,
        "source_row_index": source_row_index,
        "notes": "Only information visible in the job posting text should be included."
    }

    return structured_job_dict


def extract_structured_job(
    job_text: str,
    job_key: Optional[str] = None,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    if job_text is None or not str(job_text).strip():
        raise ValueError("Job posting text is empty. Cannot extract structured job information.")

    if job_key is None:
        job_key = create_job_key(job_text)

    job_extraction_chain = create_job_extraction_chain(
        model_name=model_name,
        temperature=temperature,
    )

    structured_job_result = job_extraction_chain.invoke(
        {
            "job_text": job_text,
        }
    )

    structured_job_dict = model_to_dict(structured_job_result)

    structured_job_dict = normalize_structured_job(structured_job_dict)

    structured_job_dict = clean_unclear_or_missing_information(structured_job_dict)

    structured_job_dict = add_structured_job_metadata(
        structured_job_dict=structured_job_dict,
        job_key=job_key,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
        model_name=model_name,
    )

    return structured_job_dict


def get_mongodb_jobs_collection(
    mongodb_uri: Optional[str] = None,
    database_name: str = "cv_job_matching_agent",
    collection_name: str = "analyzed_jobs",
):

    if mongodb_uri is None:
        mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        raise ValueError(
            "MONGODB_URI is not defined. "
            "Set it in .env or pass mongodb_uri manually."
        )

    client = MongoClient(mongodb_uri)

    db = client[database_name]
    jobs_collection = db[collection_name]

    jobs_collection.create_index("job_key", unique=True)

    return jobs_collection


def extract_or_load_structured_job(
    job_text: str,
    jobs_collection=None,
    use_mongodb: bool = True,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    if job_text is None or not str(job_text).strip():
        raise ValueError("Job posting text is empty. Cannot extract or load structured job.")

    job_key = create_job_key(job_text)

    if use_mongodb and jobs_collection is None:
        jobs_collection = get_mongodb_jobs_collection()

    if use_mongodb and jobs_collection is not None:
        existing_job = jobs_collection.find_one(
            {
                "job_key": job_key,
            }
        )

        current_time = datetime.now().isoformat(timespec="seconds")

        if existing_job is not None:
            jobs_collection.update_one(
                {
                    "job_key": job_key,
                },
                {
                    "$inc": {
                        "submission_count": 1,
                    },
                    "$set": {
                        "last_seen_at": current_time,
                    },
                },
            )

            structured_job_dict = existing_job["structured_job"]

            structured_job_dict = normalize_structured_job(structured_job_dict)
            structured_job_dict = clean_unclear_or_missing_information(structured_job_dict)

            structured_job_dict["loaded_from_database"] = True

            return structured_job_dict

    structured_job_dict = extract_structured_job(
        job_text=job_text,
        job_key=job_key,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
        model_name=model_name,
        temperature=temperature,
    )

    structured_job_dict["loaded_from_database"] = False

    if use_mongodb and jobs_collection is not None:
        current_time = datetime.now().isoformat(timespec="seconds")

        job_document = {
            "job_key": job_key,
            "created_at": current_time,
            "first_seen_at": current_time,
            "last_seen_at": current_time,
            "submission_count": 1,
            "source": source,
            "source_file": source_file,
            "source_row_index": source_row_index,
            "original_job_text": job_text,
            "structured_job": structured_job_dict,
        }

        jobs_collection.insert_one(job_document)

    return structured_job_dict


def create_job_profile_summary(
    structured_job_dict: Dict[str, Any],
) -> Dict[str, Any]:

    job_profile = {
        "job_title": structured_job_dict.get("job_title"),
        "company_name": structured_job_dict.get("company_name"),
        "location": structured_job_dict.get("location"),
        "work_mode": structured_job_dict.get("work_mode"),
        "employment_type": structured_job_dict.get("employment_type"),
        "job_category": structured_job_dict.get("job_category"),
    }

    return job_profile


def create_job_skills_summary(
    structured_job_dict: Dict[str, Any],
) -> Dict[str, List[str]]:

    skills_summary = {
        "required_skills": structured_job_dict.get("required_skills", []),
        "nice_to_have_skills": structured_job_dict.get("nice_to_have_skills", []),
        "programming_languages": structured_job_dict.get("programming_languages", []),
        "frameworks_and_libraries": structured_job_dict.get("frameworks_and_libraries", []),
        "databases": structured_job_dict.get("databases", []),
        "cloud_and_devops_tools": structured_job_dict.get("cloud_and_devops_tools", []),
        "data_and_ai_tools": structured_job_dict.get("data_and_ai_tools", []),
        "testing_tools": structured_job_dict.get("testing_tools", []),
        "other_tools": structured_job_dict.get("other_tools", []),
        "soft_skills": structured_job_dict.get("soft_skills", []),
        "certifications": structured_job_dict.get("certifications", []),
        "language_requirements": structured_job_dict.get("language_requirements", []),
    }

    return skills_summary


def save_structured_job_output(
    structured_job_dict: Dict[str, Any],
    output_path=None,
):

    if output_path is None:
        output_path = JOB_EXTRACTION_OUTPUT_DIR / "structured_job.json"

    saved_output_path = save_json_file(
        data=structured_job_dict,
        file_path=output_path,
    )

    return saved_output_path


def process_job_extraction(
    job_text: str,
    jobs_collection=None,
    use_mongodb: bool = True,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    output_path=None,
) -> Dict[str, Any]:

    structured_job_dict = extract_or_load_structured_job(
        job_text=job_text,
        jobs_collection=jobs_collection,
        use_mongodb=use_mongodb,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
        model_name=model_name,
        temperature=temperature,
    )

    job_profile = create_job_profile_summary(
        structured_job_dict=structured_job_dict,
    )

    skills_summary = create_job_skills_summary(
        structured_job_dict=structured_job_dict,
    )

    saved_output_path = save_structured_job_output(
        structured_job_dict=structured_job_dict,
        output_path=output_path,
    )

    return {
        "structured_job": structured_job_dict,
        "job_profile": job_profile,
        "skills_summary": skills_summary,
        "structured_job_output_path": str(saved_output_path),
    }