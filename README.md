# CV-Job Matching AI Agent

CV-Job Matching AI Agent is a thesis project focused on AI-assisted CV analysis and IT job matching.

The goal of the project is to help candidates analyze the quality of their CV, extract structured CV information, compare the CV with IT job postings, identify missing skills, and generate practical improvement recommendations.

The application is implemented as a practical Streamlit app with a Python backend. The analytical logic was first developed and tested in Jupyter notebooks and then refactored into reusable Python modules inside the `src/` directory.

---

## Main Features

The project supports the following functionality:

- PDF CV text extraction
- CV quality analysis
- Structured CV data extraction
- Structured IT job posting extraction
- CV-job matching and scoring
- Direct and semantic matching evaluation
- Skill gap analysis
- CV improvement recommendations
- Job-specific recommendations
- Final report generation
- Market statistics based on analyzed job postings
- MongoDB storage for structured job posting data
- Streamlit user interface

---

## Application Modes

The Streamlit application supports three main analysis modes.

### 1. CV Review only

The user uploads only a CV PDF.

The system analyzes:

- CV quality score
- CV category
- strengths
- weaknesses
- missing or unclear sections
- CV improvement suggestions
- extracted CV data

This mode does not require a job posting.

### 2. CV + Job Match

The user uploads a CV PDF and pastes one IT job posting.

The system analyzes:

- CV quality
- structured CV data
- structured job posting data
- direct matching score
- semantic matching score
- final hybrid matching score
- matched skills
- missing skills
- recommendations
- final report on demand

### 3. Compare with multiple jobs

The user uploads a CV PDF and pastes multiple IT job postings.

Multiple job postings should be separated with:

```text
---JOB---
```

The system analyzes each job posting and ranks jobs by matching score.

---

## Project Structure

```text
cv-job-matching-agent/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── notebooks/
│   ├── 01_resume_eda_and_cleaning.ipynb
│   ├── 02_it_jobs_eda_and_cleaning.ipynb
│   ├── 03_pdf_cv_extraction.ipynb
│   ├── 04_cv_quality_analysis.ipynb
│   ├── 05_cv_structuring.ipynb
│   ├── 06_job_structured_extraction.ipynb
│   ├── 07_job_storage_and_market_statistics.ipynb
│   ├── 08_matching_and_scoring.ipynb
│   ├── 09_recommendations.ipynb
│   ├── 10_final_report_generation.ipynb
│   └── 11_langgraph_workflow.ipynb
│
├── outputs/
│   ├── agent_workflow/
│   ├── cv_extraction/
│   ├── cv_quality/
│   ├── final_report/
│   ├── job_extraction/
│   ├── market_statistics/
│   ├── matching/
│   ├── recommendations/
│   └── user_uploads/
│
├── src/
│   ├── __init__.py
│   ├── agent_workflow.py
│   ├── cv_extraction.py
│   ├── cv_quality.py
│   ├── final_report.py
│   ├── job_extraction.py
│   ├── job_storage.py
│   ├── market_statistics.py
│   ├── matching.py
│   ├── pdf_extraction.py
│   ├── recommendations.py
│   └── utils.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies Used

The project uses:

- Python
- Streamlit
- Jupyter Notebook
- OpenAI API
- LangChain
- LangGraph
- Pydantic
- MongoDB
- PyMongo
- pandas
- pdfplumber
- pypdf

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd cv-job-matching-agent
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Create the `.env` file

The repository contains an `.env.example` file.

Create a local `.env` file from it:

```bash
copy .env.example .env
```

Then open `.env` and add your own API keys and connection strings.


---

## Running the Streamlit App

Make sure the virtual environment is activated.

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

If the command does not work, use:

```bash
python -m streamlit run app/streamlit_app.py
```

The application will open in the browser.

---


## Backend Workflow

The main backend workflow is implemented in:

```text
src/agent_workflow.py
```

The Streamlit app calls:

```python
run_agent_workflow()
```

This function coordinates the full workflow:

1. Extract text from the uploaded CV PDF
2. Analyze CV quality
3. Extract structured CV data
4. Extract structured job data if a job posting is provided
5. Store job data in MongoDB if applicable
6. Calculate direct and semantic matching scores
7. Generate recommendations
8. Generate final report on demand
9. Return results to the Streamlit interface

---


## Important Notes

The system does not invent candidate skills, work experience, certifications, education, or project experience.

If information is not clearly present in the CV or job posting, the system should mark it as missing, unclear, or not evidenced.

Examples:

- If a job posting does not explicitly specify required years of experience, the system should not invent a number.
- If the CV does not show formal work experience, the system should clearly state that formal work experience is not evidenced.
- Academic projects and coursework can be treated as relevant evidence for entry-level roles when appropriate.

---

## Thesis Context

This project was developed as a thesis project to demonstrate how large language models, structured extraction, rule-based scoring, semantic analysis, and agentic workflow orchestration can be combined into a practical AI assistant for CV analysis and IT job matching.


---
