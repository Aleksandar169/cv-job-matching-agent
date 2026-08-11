import re
from pathlib import Path

import pdfplumber

from src.utils import CV_TEXT_OUTPUT_DIR, save_text_file


def extract_text_with_pdfplumber(pdf_path):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"CV PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    extracted_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text is None:
                page_text = ""

            extracted_pages.append(page_text)

    full_text = "\n\n".join(extracted_pages)

    return full_text


def clean_extracted_cv_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\x00", " ")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text


def extract_text_from_pdf(pdf_path):

    raw_text = extract_text_with_pdfplumber(pdf_path)
    cleaned_text = clean_extracted_cv_text(raw_text)

    if not cleaned_text:
        raise ValueError(
            "No searchable text was extracted from the CV PDF. "
            "The file may be scanned or image-based."
        )

    return cleaned_text


def save_extracted_cv_text(cv_text, output_path=None):

    if output_path is None:
        output_path = CV_TEXT_OUTPUT_DIR / "cv_text.txt"

    return save_text_file(cv_text, output_path)


def process_cv_pdf(pdf_path, output_path=None):

    pdf_path = Path(pdf_path)

    cv_text = extract_text_from_pdf(pdf_path)

    saved_text_path = save_extracted_cv_text(
        cv_text=cv_text,
        output_path=output_path
    )

    result = {
        "cv_pdf_path": str(pdf_path),
        "cv_text_path": str(saved_text_path),
        "cv_text": cv_text
    }

    return result