from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils import (
    PROJECT_ROOT,
    CV_EXTRACTION_OUTPUT_DIR,
    save_json_file,
)


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class EducationItem(BaseModel):
    institution: Optional[str] = Field(
        default=None,
        description="Name of the school, university or educational institution."
    )

    degree: Optional[str] = Field(
        default=None,
        description="Degree or qualification, if visible in the CV."
    )

    field_of_study: Optional[str] = Field(
        default=None,
        description="Field of study, if visible in the CV."
    )

    start_year: Optional[str] = Field(
        default=None,
        description="Start year or date, if visible."
    )

    end_year: Optional[str] = Field(
        default=None,
        description="End year or date, if visible."
    )


class ExperienceItem(BaseModel):
    company: Optional[str] = Field(
        default=None,
        description="Company or organization name, if visible."
    )

    position: Optional[str] = Field(
        default=None,
        description="Job title or role."
    )

    start_date: Optional[str] = Field(
        default=None,
        description="Start date, if visible."
    )

    end_date: Optional[str] = Field(
        default=None,
        description="End date, if visible. Use Present if the CV clearly says the role is current."
    )

    responsibilities: List[str] = Field(
        default_factory=list,
        description="Main responsibilities and tasks explicitly mentioned in the CV."
    )

    technologies_used: List[str] = Field(
        default_factory=list,
        description="Technologies, tools or programming languages explicitly connected to this experience."
    )


class ProjectItem(BaseModel):
    project_name: Optional[str] = Field(
        default=None,
        description="Project name, if visible."
    )

    description: Optional[str] = Field(
        default=None,
        description="Short description of the project."
    )

    technologies_used: List[str] = Field(
        default_factory=list,
        description="Technologies, tools or programming languages explicitly mentioned for this project."
    )

    project_result: Optional[str] = Field(
        default=None,
        description="Result or outcome of the project, if visible."
    )


class CertificationItem(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="Certification name."
    )

    issuer: Optional[str] = Field(
        default=None,
        description="Certification issuer or organization, if visible."
    )

    year: Optional[str] = Field(
        default=None,
        description="Year or date of certification, if visible."
    )


class StructuredCV(BaseModel):
    candidate_name: Optional[str] = Field(
        default=None,
        description="Candidate full name, if visible in the CV."
    )

    email: Optional[str] = Field(
        default=None,
        description="Candidate email address, if visible."
    )

    phone: Optional[str] = Field(
        default=None,
        description="Candidate phone number, if visible."
    )

    location: Optional[str] = Field(
        default=None,
        description="Candidate location, if visible."
    )

    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL, if visible."
    )

    github_url: Optional[str] = Field(
        default=None,
        description="GitHub profile URL, if visible."
    )

    portfolio_url: Optional[str] = Field(
        default=None,
        description="Portfolio or personal website URL, if visible."
    )

    profile_summary: Optional[str] = Field(
        default=None,
        description="Short professional summary based only on the CV text."
    )

    total_years_of_experience: Optional[str] = Field(
        default=None,
        description="Total years of experience only if clearly visible or directly inferable from dates."
    )

    technical_skills: List[str] = Field(
        default_factory=list,
        description="Technical skills explicitly mentioned in the CV."
    )

    programming_languages: List[str] = Field(
        default_factory=list,
        description="Programming languages explicitly mentioned in the CV."
    )

    frameworks_and_libraries: List[str] = Field(
        default_factory=list,
        description="Frameworks and libraries explicitly mentioned in the CV."
    )

    databases: List[str] = Field(
        default_factory=list,
        description="Databases explicitly mentioned in the CV."
    )

    cloud_and_devops_tools: List[str] = Field(
        default_factory=list,
        description="Cloud platforms, DevOps tools and infrastructure tools explicitly mentioned in the CV."
    )

    data_and_ai_tools: List[str] = Field(
        default_factory=list,
        description="Data analysis, machine learning, AI or BI tools explicitly mentioned in the CV."
    )

    other_tools: List[str] = Field(
        default_factory=list,
        description="Other software tools explicitly mentioned in the CV."
    )

    soft_skills: List[str] = Field(
        default_factory=list,
        description="Soft skills explicitly mentioned in the CV."
    )

    languages: List[str] = Field(
        default_factory=list,
        description="Human languages explicitly mentioned in the CV."
    )

    education: List[EducationItem] = Field(
        default_factory=list,
        description="Education entries extracted from the CV."
    )

    work_experience: List[ExperienceItem] = Field(
        default_factory=list,
        description="Work experience entries extracted from the CV."
    )

    projects: List[ProjectItem] = Field(
        default_factory=list,
        description="Project entries extracted from the CV."
    )

    certifications: List[CertificationItem] = Field(
        default_factory=list,
        description="Certifications explicitly mentioned in the CV."
    )

    unclear_or_missing_information: List[str] = Field(
        default_factory=list,
        description="Important information that is missing or unclear in the CV."
    )


cv_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant specialized in structured CV extraction for IT candidates.

Your task is to extract structured information from the provided CV text.

Important rules:
- Extract only information that is explicitly present in the CV text.
- Do not invent skills, tools, programming languages, certifications, education, projects or work experience.
- Do not assume that the candidate has a skill if it is not visible in the CV.
- If information is missing or unclear, leave the field empty or add a note to unclear_or_missing_information.
- Keep extracted skill names concise and normalized when possible.
- Focus on IT-related information such as programming languages, frameworks, databases, cloud tools, DevOps tools, data tools, AI tools and software tools.
- Do not evaluate the CV quality in this step.
- Do not compare the CV with a job posting in this step.

Return the result using the required structured schema.
"""
        ),
        (
            "human",
            """
Extract structured information from the following CV text:

{cv_text}
"""
        )
    ]
)


def create_cv_extraction_chain(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
):

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    structured_llm = llm.with_structured_output(StructuredCV)

    return cv_extraction_prompt | structured_llm


def model_to_dict(model_result: Any) -> Dict[str, Any]:

    if hasattr(model_result, "model_dump"):
        return model_result.model_dump()

    return model_result.dict()


def normalize_list_field(data: Dict[str, Any], field_name: str) -> Dict[str, Any]:

    value = data.get(field_name)

    if value is None:
        data[field_name] = []

    elif isinstance(value, list):
        data[field_name] = value

    else:
        data[field_name] = [value]

    return data


def normalize_structured_cv(structured_cv_dict: Dict[str, Any]) -> Dict[str, Any]:

    list_fields = [
        "technical_skills",
        "programming_languages",
        "frameworks_and_libraries",
        "databases",
        "cloud_and_devops_tools",
        "data_and_ai_tools",
        "other_tools",
        "soft_skills",
        "languages",
        "education",
        "work_experience",
        "projects",
        "certifications",
        "unclear_or_missing_information",
    ]

    for field_name in list_fields:
        structured_cv_dict = normalize_list_field(
            data=structured_cv_dict,
            field_name=field_name,
        )

    return structured_cv_dict


def add_structured_cv_metadata(
    structured_cv_dict: Dict[str, Any],
    source_file: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:

    structured_cv_dict["metadata"] = {
        "source_file": source_file,
        "model": model_name,
        "extraction_type": "structured_cv_extraction",
        "notes": "Only information visible in the CV text should be included."
    }

    return structured_cv_dict


def extract_structured_cv(
    cv_text: str,
    source_file: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
) -> Dict[str, Any]:

    if cv_text is None or not str(cv_text).strip():
        raise ValueError("CV text is empty. Cannot extract structured CV information.")

    cv_extraction_chain = create_cv_extraction_chain(
        model_name=model_name,
        temperature=temperature,
    )

    structured_cv_result = cv_extraction_chain.invoke(
        {
            "cv_text": cv_text,
        }
    )

    structured_cv_dict = model_to_dict(structured_cv_result)

    structured_cv_dict = normalize_structured_cv(structured_cv_dict)

    structured_cv_dict = add_structured_cv_metadata(
        structured_cv_dict=structured_cv_dict,
        source_file=source_file,
        model_name=model_name,
    )

    return structured_cv_dict


def create_candidate_profile_summary(
    structured_cv_dict: Dict[str, Any],
) -> Dict[str, Any]:

    candidate_profile = {
        "candidate_name": structured_cv_dict.get("candidate_name"),
        "email": structured_cv_dict.get("email"),
        "phone": structured_cv_dict.get("phone"),
        "location": structured_cv_dict.get("location"),
        "linkedin_url": structured_cv_dict.get("linkedin_url"),
        "github_url": structured_cv_dict.get("github_url"),
        "portfolio_url": structured_cv_dict.get("portfolio_url"),
        "total_years_of_experience": structured_cv_dict.get("total_years_of_experience"),
    }

    return candidate_profile


def create_skills_summary(
    structured_cv_dict: Dict[str, Any],
) -> Dict[str, List[str]]:

    skills_summary = {
        "technical_skills": structured_cv_dict.get("technical_skills", []),
        "programming_languages": structured_cv_dict.get("programming_languages", []),
        "frameworks_and_libraries": structured_cv_dict.get("frameworks_and_libraries", []),
        "databases": structured_cv_dict.get("databases", []),
        "cloud_and_devops_tools": structured_cv_dict.get("cloud_and_devops_tools", []),
        "data_and_ai_tools": structured_cv_dict.get("data_and_ai_tools", []),
        "other_tools": structured_cv_dict.get("other_tools", []),
        "soft_skills": structured_cv_dict.get("soft_skills", []),
        "languages": structured_cv_dict.get("languages", []),
    }

    return skills_summary


def save_structured_cv_output(
    structured_cv_dict: Dict[str, Any],
    output_path=None,
):

    if output_path is None:
        output_path = CV_EXTRACTION_OUTPUT_DIR / "structured_cv.json"

    saved_output_path = save_json_file(
        data=structured_cv_dict,
        file_path=output_path,
    )

    return saved_output_path


def process_structured_cv_extraction(
    cv_text: str,
    source_file: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0,
    output_path=None,
) -> Dict[str, Any]:

    structured_cv_dict = extract_structured_cv(
        cv_text=cv_text,
        source_file=source_file,
        model_name=model_name,
        temperature=temperature,
    )

    candidate_profile = create_candidate_profile_summary(
        structured_cv_dict=structured_cv_dict,
    )

    skills_summary = create_skills_summary(
        structured_cv_dict=structured_cv_dict,
    )

    saved_output_path = save_structured_cv_output(
        structured_cv_dict=structured_cv_dict,
        output_path=output_path,
    )

    return {
        "structured_cv": structured_cv_dict,
        "candidate_profile": candidate_profile,
        "skills_summary": skills_summary,
        "structured_cv_output_path": str(saved_output_path),
    }