from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from src.utils import (
    MARKET_STATISTICS_OUTPUT_DIR,
    save_json_file,
)

from src.job_storage import (
    get_jobs_collection,
    get_all_analyzed_jobs,
    get_analyzed_jobs_by_date_range,
)


def get_structured_job(document: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(document, dict):
        return {}

    if isinstance(document.get("structured_job"), dict):
        return document["structured_job"]

    return document


def get_submission_weight(
    document: Dict[str, Any],
    weighted: bool = True,
) -> int:

    if not weighted:
        return 1

    try:
        return int(document.get("submission_count", 1))
    except (TypeError, ValueError):
        return 1


def normalize_value(value: Any) -> Optional[str]:

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in ["none", "null", "nan", "unknown", "n/a"]:
        return None

    return value


def normalize_list_values(values: Any) -> List[str]:

    if values is None:
        return []

    if isinstance(values, list):
        raw_values = values
    else:
        raw_values = [values]

    cleaned_values = []

    for value in raw_values:
        normalized_value = normalize_value(value)

        if normalized_value:
            cleaned_values.append(normalized_value)

    return cleaned_values


def count_single_field_values(
    job_documents: List[Dict[str, Any]],
    field_name: str,
    weighted: bool = True,
) -> Counter:

    counter = Counter()

    for document in job_documents:
        structured_job = get_structured_job(document)
        value = normalize_value(structured_job.get(field_name))

        if value is None:
            continue

        weight = get_submission_weight(document, weighted=weighted)
        counter[value] += weight

    return counter


def count_list_field_values(
    job_documents: List[Dict[str, Any]],
    field_name: str,
    weighted: bool = True,
) -> Counter:

    counter = Counter()

    for document in job_documents:
        structured_job = get_structured_job(document)
        values = normalize_list_values(structured_job.get(field_name, []))

        if not values:
            continue

        weight = get_submission_weight(document, weighted=weighted)

        for value in values:
            counter[value] += weight

    return counter


def counter_to_records(
    counter: Counter,
    top_n: Optional[int] = 20,
    value_name: str = "item",
    count_name: str = "count",
) -> List[Dict[str, Any]]:

    if top_n is None:
        most_common_items = counter.most_common()
    else:
        most_common_items = counter.most_common(top_n)

    records = []

    for value, count in most_common_items:
        records.append(
            {
                value_name: value,
                count_name: count,
            }
        )

    return records


def counter_to_dataframe(
    counter: Counter,
    top_n: Optional[int] = 20,
    value_name: str = "item",
    count_name: str = "count",
) -> pd.DataFrame:

    records = counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name=value_name,
        count_name=count_name,
    )

    return pd.DataFrame(records)


def parse_datetime_safe(value: Any) -> Optional[datetime]:

    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    value = str(value).strip()

    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return None


def get_month_key(value: Any) -> Optional[str]:

    parsed_date = parse_datetime_safe(value)

    if parsed_date is None:
        return None

    return parsed_date.strftime("%Y-%m")


def calculate_monthly_unique_job_counts(
    job_documents: List[Dict[str, Any]],
    date_field: str = "first_seen_at",
) -> List[Dict[str, Any]]:

    month_counter = Counter()

    for document in job_documents:
        month_key = get_month_key(document.get(date_field))

        if month_key is None:
            continue

        month_counter[month_key] += 1

    records = [
        {
            "month": month,
            "unique_jobs_count": count,
        }
        for month, count in sorted(month_counter.items())
    ]

    return records


def calculate_monthly_submission_counts(
    job_documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    month_counter = Counter()

    for document in job_documents:
        submission_events = document.get("submission_events")

        if isinstance(submission_events, list) and len(submission_events) > 0:
            for event in submission_events:
                if not isinstance(event, dict):
                    continue

                month_key = get_month_key(event.get("submitted_at"))

                if month_key is not None:
                    month_counter[month_key] += 1

        else:
            month_key = get_month_key(document.get("first_seen_at"))

            if month_key is None:
                continue

            weight = get_submission_weight(document, weighted=True)
            month_counter[month_key] += weight

    records = [
        {
            "month": month,
            "submission_count": count,
        }
        for month, count in sorted(month_counter.items())
    ]

    return records


def calculate_top_required_skills(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="required_skills",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="skill",
        count_name="count",
    )


def calculate_top_nice_to_have_skills(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="nice_to_have_skills",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="skill",
        count_name="count",
    )


def calculate_top_programming_languages(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="programming_languages",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="programming_language",
        count_name="count",
    )


def calculate_top_frameworks_and_libraries(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="frameworks_and_libraries",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="framework_or_library",
        count_name="count",
    )


def calculate_top_databases(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="databases",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="database",
        count_name="count",
    )


def calculate_top_cloud_and_devops_tools(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="cloud_and_devops_tools",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="cloud_or_devops_tool",
        count_name="count",
    )


def calculate_top_data_and_ai_tools(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_list_field_values(
        job_documents=job_documents,
        field_name="data_and_ai_tools",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="data_or_ai_tool",
        count_name="count",
    )


def calculate_jobs_by_category(
    job_documents: List[Dict[str, Any]],
    top_n: int = 20,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_single_field_values(
        job_documents=job_documents,
        field_name="job_category",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=top_n,
        value_name="job_category",
        count_name="count",
    )


def calculate_jobs_by_work_mode(
    job_documents: List[Dict[str, Any]],
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_single_field_values(
        job_documents=job_documents,
        field_name="work_mode",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=None,
        value_name="work_mode",
        count_name="count",
    )


def calculate_jobs_by_employment_type(
    job_documents: List[Dict[str, Any]],
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    counter = count_single_field_values(
        job_documents=job_documents,
        field_name="employment_type",
        weighted=weighted,
    )

    return counter_to_records(
        counter=counter,
        top_n=None,
        value_name="employment_type",
        count_name="count",
    )


def calculate_average_required_skills_count(
    job_documents: List[Dict[str, Any]],
) -> float:

    skill_counts = []

    for document in job_documents:
        structured_job = get_structured_job(document)
        required_skills = normalize_list_values(
            structured_job.get("required_skills", [])
        )

        skill_counts.append(len(required_skills))

    if not skill_counts:
        return 0.0

    return round(sum(skill_counts) / len(skill_counts), 2)


def get_demand_level(count: int) -> str:

    if count >= 20:
        return "High"

    if count >= 10:
        return "Medium"

    return "Low"


def calculate_top_it_roles(
    job_documents: List[Dict[str, Any]],
    top_n: int = 10,
    weighted: bool = True,
) -> List[Dict[str, Any]]:

    role_counter = Counter()
    role_skills = defaultdict(Counter)

    for document in job_documents:
        structured_job = get_structured_job(document)
        job_title = normalize_value(structured_job.get("job_title"))

        if job_title is None:
            continue

        weight = get_submission_weight(document, weighted=weighted)

        role_counter[job_title] += weight

        required_skills = normalize_list_values(
            structured_job.get("required_skills", [])
        )

        for skill in required_skills:
            role_skills[job_title][skill] += weight

    records = []

    for role, count in role_counter.most_common(top_n):
        top_skills = [
            skill
            for skill, _ in role_skills[role].most_common(5)
        ]

        records.append(
            {
                "role": role,
                "count": count,
                "demand_level": get_demand_level(count),
                "top_required_skills": top_skills,
            }
        )

    return records


def calculate_market_metrics(
    job_documents: List[Dict[str, Any]],
) -> Dict[str, Any]:

    unique_jobs_count = len(job_documents)

    total_submissions = sum(
        get_submission_weight(document, weighted=True)
        for document in job_documents
    )

    required_skills = calculate_top_required_skills(
        job_documents=job_documents,
        top_n=1,
        weighted=True,
    )

    jobs_by_category = calculate_jobs_by_category(
        job_documents=job_documents,
        top_n=1,
        weighted=True,
    )

    most_requested_skill = None
    top_job_category = None

    if required_skills:
        most_requested_skill = required_skills[0]["skill"]

    if jobs_by_category:
        top_job_category = jobs_by_category[0]["job_category"]

    average_required_skills = calculate_average_required_skills_count(
        job_documents=job_documents,
    )

    return {
        "unique_jobs_count": unique_jobs_count,
        "total_submissions": total_submissions,
        "most_requested_skill": most_requested_skill,
        "top_job_category": top_job_category,
        "average_required_skills_count": average_required_skills,
    }


def create_market_statistics_dashboard(
    job_documents: List[Dict[str, Any]],
    top_n: int = 10,
    weighted: bool = True,
) -> Dict[str, Any]:

    dashboard = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "statistics_type": "job_market_statistics_from_analyzed_jobs",
            "weighted_by_submission_count": weighted,
            "source": "MongoDB analyzed_jobs or provided job documents",
        },
        "metrics": calculate_market_metrics(job_documents),
        "monthly_unique_jobs": calculate_monthly_unique_job_counts(job_documents),
        "monthly_submissions": calculate_monthly_submission_counts(job_documents),
        "jobs_by_category": calculate_jobs_by_category(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "jobs_by_work_mode": calculate_jobs_by_work_mode(
            job_documents=job_documents,
            weighted=weighted,
        ),
        "jobs_by_employment_type": calculate_jobs_by_employment_type(
            job_documents=job_documents,
            weighted=weighted,
        ),
        "most_requested_required_skills": calculate_top_required_skills(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_nice_to_have_skills": calculate_top_nice_to_have_skills(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_programming_languages": calculate_top_programming_languages(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_frameworks_and_libraries": calculate_top_frameworks_and_libraries(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_databases": calculate_top_databases(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_cloud_and_devops_tools": calculate_top_cloud_and_devops_tools(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "most_requested_data_and_ai_tools": calculate_top_data_and_ai_tools(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
        "top_it_roles": calculate_top_it_roles(
            job_documents=job_documents,
            top_n=top_n,
            weighted=weighted,
        ),
    }

    return dashboard


def load_market_statistics_from_mongodb(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 10,
    weighted: bool = True,
    jobs_collection=None,
) -> Dict[str, Any]:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    if start_date or end_date:
        job_documents = get_analyzed_jobs_by_date_range(
            start_date=start_date,
            end_date=end_date,
            date_field="first_seen_at",
            jobs_collection=jobs_collection,
            include_original_text=False,
        )
    else:
        job_documents = get_all_analyzed_jobs(
            jobs_collection=jobs_collection,
            include_original_text=False,
        )

    dashboard = create_market_statistics_dashboard(
        job_documents=job_documents,
        top_n=top_n,
        weighted=weighted,
    )

    dashboard["metadata"]["start_date"] = start_date
    dashboard["metadata"]["end_date"] = end_date

    return dashboard


def save_market_statistics_dashboard(
    dashboard: Dict[str, Any],
    output_path=None,
):

    if output_path is None:
        output_path = MARKET_STATISTICS_OUTPUT_DIR / "market_statistics_dashboard.json"

    saved_output_path = save_json_file(
        data=dashboard,
        file_path=output_path,
    )

    return saved_output_path


def process_market_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 10,
    weighted: bool = True,
    jobs_collection=None,
    output_path=None,
) -> Dict[str, Any]:

    dashboard = load_market_statistics_from_mongodb(
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        weighted=weighted,
        jobs_collection=jobs_collection,
    )

    saved_output_path = save_market_statistics_dashboard(
        dashboard=dashboard,
        output_path=output_path,
    )

    return {
        "market_statistics_dashboard": dashboard,
        "market_statistics_output_path": str(saved_output_path),
    }


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:

    return pd.DataFrame(records)


def dashboard_section_to_dataframe(
    dashboard: Dict[str, Any],
    section_name: str,
) -> pd.DataFrame:

    records = dashboard.get(section_name, [])

    if not isinstance(records, list):
        records = []

    return records_to_dataframe(records)