import os
import re
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from pymongo import MongoClient

from src.utils import PROJECT_ROOT


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


DEFAULT_DATABASE_NAME = "cv_job_matching_agent"
DEFAULT_JOBS_COLLECTION_NAME = "analyzed_jobs"


def get_current_timestamp() -> str:

    return datetime.now().isoformat(timespec="seconds")


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


def get_mongodb_client(mongodb_uri: Optional[str] = None) -> MongoClient:

    if mongodb_uri is None:
        mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        raise ValueError(
            "MONGODB_URI is not defined. "
        )

    return MongoClient(mongodb_uri)


def get_jobs_collection(
    mongodb_uri: Optional[str] = None,
    database_name: str = DEFAULT_DATABASE_NAME,
    collection_name: str = DEFAULT_JOBS_COLLECTION_NAME,
):
    client = get_mongodb_client(mongodb_uri)

    db = client[database_name]
    jobs_collection = db[collection_name]

    create_jobs_collection_indexes(jobs_collection)

    return jobs_collection


def create_jobs_collection_indexes(jobs_collection):

    jobs_collection.create_index("job_key", unique=True)
    jobs_collection.create_index("first_seen_at")
    jobs_collection.create_index("last_seen_at")
    jobs_collection.create_index("submission_count")
    jobs_collection.create_index("structured_job.job_category")
    jobs_collection.create_index("structured_job.job_title")
    jobs_collection.create_index("structured_job.work_mode")

    return jobs_collection


def prepare_submission_event(
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
) -> Dict[str, Any]:

    return {
        "submitted_at": get_current_timestamp(),
        "source": source,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }


def prepare_job_document(
    job_text: str,
    structured_job: Dict[str, Any],
    job_key: Optional[str] = None,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
) -> Dict[str, Any]:

    if job_key is None:
        job_key = create_job_key(job_text)

    current_time = get_current_timestamp()

    submission_event = {
        "submitted_at": current_time,
        "source": source,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }

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
        "structured_job": structured_job,
        "submission_events": [submission_event],
    }

    return job_document


def find_job_by_key(
    job_key: str,
    jobs_collection=None,
) -> Optional[Dict[str, Any]]:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    return jobs_collection.find_one({"job_key": job_key})


def save_or_update_analyzed_job(
    job_text: str,
    structured_job: Dict[str, Any],
    jobs_collection=None,
    source: str = "streamlit_user_input",
    source_file: Optional[str] = None,
    source_row_index: Optional[int] = None,
    update_structured_job_on_duplicate: bool = False,
) -> Dict[str, Any]:

    if job_text is None or not str(job_text).strip():
        raise ValueError("Job text is empty. Cannot save analyzed job.")

    if structured_job is None or not isinstance(structured_job, dict):
        raise ValueError("structured_job must be a dictionary.")

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    job_key = create_job_key(job_text)

    existing_job = jobs_collection.find_one({"job_key": job_key})

    current_time = get_current_timestamp()

    submission_event = {
        "submitted_at": current_time,
        "source": source,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }

    if existing_job is not None:
        set_values = {
            "last_seen_at": current_time,
        }

        if update_structured_job_on_duplicate:
            set_values["structured_job"] = structured_job

        jobs_collection.update_one(
            {"job_key": job_key},
            {
                "$inc": {
                    "submission_count": 1,
                },
                "$set": set_values,
                "$push": {
                    "submission_events": submission_event,
                },
            },
        )

        updated_job = jobs_collection.find_one({"job_key": job_key})

        return {
            "job_key": job_key,
            "inserted": False,
            "updated_existing": True,
            "submission_count": updated_job.get("submission_count"),
            "job_document": updated_job,
        }

    job_document = prepare_job_document(
        job_text=job_text,
        structured_job=structured_job,
        job_key=job_key,
        source=source,
        source_file=source_file,
        source_row_index=source_row_index,
    )

    jobs_collection.insert_one(job_document)

    inserted_job = jobs_collection.find_one({"job_key": job_key})

    return {
        "job_key": job_key,
        "inserted": True,
        "updated_existing": False,
        "submission_count": 1,
        "job_document": inserted_job,
    }


def get_all_analyzed_jobs(
    jobs_collection=None,
    include_original_text: bool = False,
) -> List[Dict[str, Any]]:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    projection = None

    if not include_original_text:
        projection = {
            "original_job_text": 0,
        }

    jobs = list(jobs_collection.find({}, projection))

    return jobs


def get_analyzed_jobs_by_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_field: str = "first_seen_at",
    jobs_collection=None,
    include_original_text: bool = False,
) -> List[Dict[str, Any]]:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    query = {}

    if start_date or end_date:
        query[date_field] = {}

        if start_date:
            query[date_field]["$gte"] = start_date

        if end_date:
            query[date_field]["$lte"] = end_date

    projection = None

    if not include_original_text:
        projection = {
            "original_job_text": 0,
        }

    jobs = list(jobs_collection.find(query, projection))

    return jobs


def get_analyzed_jobs_count(jobs_collection=None) -> int:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    return jobs_collection.count_documents({})


def get_total_submission_count(jobs_collection=None) -> int:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_submissions": {
                    "$sum": "$submission_count"
                },
            }
        }
    ]

    result = list(jobs_collection.aggregate(pipeline))

    if not result:
        return 0

    return int(result[0].get("total_submissions", 0))


def get_structured_jobs_from_documents(
    job_documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    structured_jobs = []

    for document in job_documents:
        structured_job = document.get("structured_job")

        if isinstance(structured_job, dict):
            structured_jobs.append(structured_job)

    return structured_jobs


def delete_job_by_key(
    job_key: str,
    jobs_collection=None,
) -> int:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    result = jobs_collection.delete_one({"job_key": job_key})

    return result.deleted_count


def clear_analyzed_jobs_collection(
    jobs_collection=None,
) -> int:

    if jobs_collection is None:
        jobs_collection = get_jobs_collection()

    result = jobs_collection.delete_many({})

    return result.deleted_count