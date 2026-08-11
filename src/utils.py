from pathlib import Path
from datetime import datetime
import json
import re
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

USER_UPLOADS_DIR = OUTPUTS_DIR / "user_uploads"
CV_UPLOADS_DIR = USER_UPLOADS_DIR / "cv"
JOB_UPLOADS_DIR = USER_UPLOADS_DIR / "jobs"

MARKET_STATISTICS_OUTPUT_DIR = OUTPUTS_DIR / "market_statistics"
CV_EXTRACTION_OUTPUT_DIR = OUTPUTS_DIR / "cv_extraction"
CV_TEXT_OUTPUT_DIR = CV_EXTRACTION_OUTPUT_DIR
CV_QUALITY_OUTPUT_DIR = OUTPUTS_DIR / "cv_quality"
JOB_EXTRACTION_OUTPUT_DIR = OUTPUTS_DIR / "job_extraction"
MATCHING_OUTPUT_DIR = OUTPUTS_DIR / "matching"
RECOMMENDATIONS_OUTPUT_DIR = OUTPUTS_DIR / "recommendations"
FINAL_REPORT_OUTPUT_DIR = OUTPUTS_DIR / "final_report"
AGENT_WORKFLOW_OUTPUT_DIR = OUTPUTS_DIR / "agent_workflow"


def ensure_directory(directory_path):

    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path
    
def sanitize_filename(filename):

    filename = str(filename).strip()
    filename = filename.replace(" ", "_")
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)

    if not filename:
        filename = "uploaded_file"

    return filename


def create_unique_filename(original_filename, prefix=None):

    original_filename = sanitize_filename(original_filename)
    file_suffix = Path(original_filename).suffix
    file_stem = Path(original_filename).stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]

    if prefix:
        prefix = sanitize_filename(prefix)
        new_filename = f"{prefix}_{timestamp}_{short_id}_{file_stem}{file_suffix}"
    else:
        new_filename = f"{timestamp}_{short_id}_{file_stem}{file_suffix}"

    return new_filename


def save_uploaded_file(uploaded_file, target_directory, prefix=None):

    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    target_directory = ensure_directory(target_directory)

    unique_filename = create_unique_filename(
        original_filename=uploaded_file.name,
        prefix=prefix
    )

    saved_file_path = target_directory / unique_filename

    with open(saved_file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return saved_file_path


def save_uploaded_cv_pdf(uploaded_cv_file):

    if uploaded_cv_file is None:
        raise ValueError("No CV PDF file was uploaded.")

    file_suffix = Path(uploaded_cv_file.name).suffix.lower()

    if file_suffix != ".pdf":
        raise ValueError("Uploaded CV file must be a PDF.")

    return save_uploaded_file(
        uploaded_file=uploaded_cv_file,
        target_directory=CV_UPLOADS_DIR,
        prefix="cv"
    )


def list_pdf_files(directory_path):

    directory_path = Path(directory_path)

    if not directory_path.exists():
        return []

    return sorted(directory_path.glob("*.pdf"))


def get_latest_cv_pdf(cv_directory=CV_UPLOADS_DIR):

    cv_files = list_pdf_files(cv_directory)

    if not cv_files:
        raise FileNotFoundError(f"No CV PDF files found in: {cv_directory}")

    latest_cv_file = max(cv_files, key=lambda file_path: file_path.stat().st_mtime)

    return latest_cv_file


def get_cv_pdf_path(file_name=None, cv_directory=CV_UPLOADS_DIR):

    cv_directory = Path(cv_directory)

    if file_name is None:
        return get_latest_cv_pdf(cv_directory)

    file_name = sanitize_filename(file_name)
    cv_pdf_path = cv_directory / file_name

    if not cv_pdf_path.exists():
        raise FileNotFoundError(f"CV PDF file not found: {cv_pdf_path}")

    if cv_pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Selected CV file must be a PDF.")

    return cv_pdf_path


def load_json_file(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(data, file_path):

    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return file_path


def load_text_file(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    return file_path.read_text(encoding="utf-8")


def save_text_file(text, file_path):

    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    file_path.write_text(text, encoding="utf-8")

    return file_path


def create_run_id():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]

    return f"run_{timestamp}_{short_id}"


def create_run_output_directory(run_id=None):
    if run_id is None:
        run_id = create_run_id()

    run_output_directory = OUTPUTS_DIR / "runs" / run_id
    ensure_directory(run_output_directory)

    return run_output_directory