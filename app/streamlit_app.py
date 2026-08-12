import sys
from pathlib import Path
import json
import textwrap

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils import save_uploaded_cv_pdf
from src.agent_workflow import run_agent_workflow
from src.market_statistics import process_market_statistics


st.set_page_config(
    page_title="CV-Job Matching AI Agent",
    page_icon="🧠",
    layout="wide",
)


DEFAULT_CV_QUALITY_WEIGHTS = {
    "structure_and_readability_score": 0.15,
    "completeness_score": 0.15,
    "technical_skills_clarity_score": 0.20,
    "experience_description_score": 0.20,
    "projects_description_score": 0.15,
    "measurable_results_score": 0.10,
    "it_relevance_score": 0.05,
}


DEFAULT_DIRECT_MATCHING_WEIGHTS = {
    "required_skills_score": 0.35,
    "technology_score": 0.25,
    "experience_score": 0.25,
    "education_score": 0.08,
    "nice_to_have_score": 0.04,
    "certification_score": 0.02,
    "language_score": 0.01,
}


DEFAULT_MATCHING_MODE_WEIGHTS = {
    "direct_matching_score": 0.70,
    "semantic_score": 0.30,
}


ANALYSIS_MODE_CV_ONLY = "CV Review only"
ANALYSIS_MODE_ONE_JOB = "CV + Job Match"
ANALYSIS_MODE_MULTIPLE_JOBS = "Compare with multiple jobs"


def initialize_session_state():
    defaults = {
        "agent_result": None,
        "last_error": None,
        "market_statistics": None,
        "cv_quality_weights": DEFAULT_CV_QUALITY_WEIGHTS.copy(),
        "direct_matching_weights": DEFAULT_DIRECT_MATCHING_WEIGHTS.copy(),
        "matching_mode_weights": DEFAULT_MATCHING_MODE_WEIGHTS.copy(),
        "current_analysis_mode": ANALYSIS_MODE_CV_ONLY,
        "display_job_id": None,
        "show_cv_details": False,
        "show_recommendations": False,
        "show_matching_details": False,
        "show_final_report": False,
        "show_cv_quality_report": False,
        "show_extracted_cv_data": False,
        "last_saved_cv_path": None,
        "last_job_texts": [],
        "last_use_mongodb": False,
        "_download_button_counter": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_result_visibility():
    st.session_state.show_cv_details = False
    st.session_state.show_recommendations = False
    st.session_state.show_matching_details = False
    st.session_state.show_final_report = False
    st.session_state.show_cv_quality_report = False
    st.session_state.show_extracted_cv_data = False


def split_job_texts(raw_text):
    if raw_text is None:
        return []

    raw_text = raw_text.strip()

    if not raw_text:
        return []

    if "---JOB---" in raw_text:
        parts = raw_text.split("---JOB---")
        return [part.strip() for part in parts if part.strip()]

    return [raw_text]


def make_json_serializable(data):
    return json.loads(json.dumps(data, default=str, ensure_ascii=False))


def shorten_text(text, max_length=140):
    if text is None:
        return "N/A"

    text = str(text).strip()

    if not text:
        return "N/A"

    return textwrap.shorten(text, width=max_length, placeholder="...")


def format_score(score):
    if score is None:
        return "N/A"

    try:
        return f"{round(float(score), 2)}/100"
    except (TypeError, ValueError):
        return str(score)


def create_unique_download_key(prefix):
    st.session_state["_download_button_counter"] += 1
    return f"{prefix}_{st.session_state['_download_button_counter']}"


def download_markdown_button(markdown_text, file_name, label):
    st.download_button(
        label=label,
        data=markdown_text,
        file_name=file_name,
        mime="text/markdown",
        key=create_unique_download_key("download_markdown"),
    )


def download_json_button(data, file_name, label):
    st.download_button(
        label=label,
        data=json.dumps(data, indent=4, ensure_ascii=False),
        file_name=file_name,
        mime="application/json",
        key=create_unique_download_key("download_json"),
    )


def section_break(title):
    st.divider()
    st.markdown(f"## 🟢 {title}")


def show_bullet_list(items, empty_message="No data available."):
    if not items:
        st.caption(empty_message)
        return

    for item in items:
        if isinstance(item, dict):
            value = (
                item.get("title")
                or item.get("skill")
                or item.get("action_title")
                or item.get("job_requirement")
                or item.get("reason")
                or item.get("action_description")
                or str(item)
            )
        else:
            value = str(item)

        st.markdown(f"- {value}")


def show_dict_as_table(data):
    if not data:
        st.info("No data available.")
        return

    rows = []

    for key, value in data.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, default=str)

        rows.append(
            {
                "Field": key,
                "Value": value,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def records_to_dataframe(records):
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def show_recommendation_actions(actions):
    if not actions:
        st.caption("No recommended actions available.")
        return

    for action in actions:
        if not isinstance(action, dict):
            st.markdown(f"- {action}")
            continue

        action_title = action.get("action_title", "Action")
        priority = action.get("priority", "N/A")
        action_description = action.get("action_description", "")
        expected_impact = action.get("expected_impact", "")

        with st.expander(f"{action_title} — {priority}", expanded=False):
            if action_description:
                st.markdown(f"**Description:** {action_description}")

            if expected_impact:
                st.markdown(f"**Expected impact:** {expected_impact}")


def show_recommendation_items(items):
    if not items:
        st.caption("No recommendations available.")
        return

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            st.markdown(f"{index}. {item}")
            continue

        title = item.get("title", f"Recommendation {index}")
        reason = item.get("reason", "Not provided.")
        evidence = item.get("evidence", "Not provided.")
        actions = item.get("recommended_actions", [])

        with st.expander(f"{index}. {title}", expanded=False):
            st.markdown(f"**Reason:** {reason}")
            st.markdown(f"**Evidence:** {evidence}")

            if actions:
                st.markdown("**Recommended actions:**")
                show_recommendation_actions(actions)


def show_skill_recommendation_items(items):
    if not items:
        st.caption("No skill recommendations available.")
        return

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            st.markdown(f"{index}. {item}")
            continue

        skill = item.get("skill", f"Skill {index}")
        current_status = item.get("current_status", "Not provided.")
        evidence = item.get("evidence", "Not provided.")
        actions = item.get("recommended_actions", [])

        with st.expander(f"{index}. {skill}", expanded=False):
            st.markdown(f"**Current status:** {current_status}")
            st.markdown(f"**Evidence:** {evidence}")

            if actions:
                st.markdown("**Recommended actions:**")
                show_recommendation_actions(actions)


def render_sidebar():
    st.sidebar.title("CV-Job Matching AI")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Analysis Workspace",
            "Market Statistics",
            "Settings",
        ],
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "AI-assisted CV analysis and job matching for IT candidates."
    )

    return page


def render_feature_overview():
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📄 CV Review")
            st.write("Get a quality score and practical feedback to improve your CV.")

    with col2:
        with st.container(border=True):
            st.markdown("### 🎯 Job Fit")
            st.write("See how well your CV matches a selected IT job posting.")

    with col3:
        with st.container(border=True):
            st.markdown("### ⭐ Improvement Priorities")
            st.write("Focus on the changes that can have the biggest impact.")


def format_job_label(job_result):
    job_id = job_result.get("job_id", "job")
    job_information = job_result.get("job_information", {})

    job_title = job_information.get("job_title") or "Unknown title"
    company_name = job_information.get("company_name") or "Unknown company"
    score = job_result.get("final_hybrid_score")

    if score is None:
        score_text = "N/A"
    else:
        score_text = f"{round(float(score), 2)}/100"

    return f"{job_id} | {job_title} | {company_name} | {score_text}"


def get_selected_job_result(job_results):
    if not job_results:
        return {}

    job_ids = [job.get("job_id") for job in job_results]

    if st.session_state.display_job_id not in job_ids:
        st.session_state.display_job_id = job_ids[0]

    labels = [format_job_label(job) for job in job_results]
    default_index = job_ids.index(st.session_state.display_job_id)

    selected_label = st.selectbox(
        "Select job for detailed view",
        labels,
        index=default_index,
        key="selected_job_for_detailed_view",
    )

    selected_index = labels.index(selected_label)
    selected_job = job_results[selected_index]

    st.session_state.display_job_id = selected_job.get("job_id")

    return selected_job


def extract_top_priority(agent_result, selected_job_result):
    recommendation_output = selected_job_result.get("recommendation_output", {})

    if recommendation_output:
        recommendations = recommendation_output.get("recommendations", recommendation_output)
        priority_actions = recommendations.get("priority_actions", [])

        if priority_actions:
            first_action = priority_actions[0]

            if isinstance(first_action, dict):
                return (
                    first_action.get("action_title")
                    or first_action.get("action_description")
                    or "Review job-specific recommendations"
                )

            return str(first_action)

    cv_recommendations_digest = agent_result.get("cv_recommendations_digest", {})
    cv_recommendations = cv_recommendations_digest.get("recommendations", [])

    if cv_recommendations:
        return str(cv_recommendations[0])

    cv_digest = agent_result.get("cv_digest", {})
    weaknesses = cv_digest.get("top_weaknesses", [])

    if weaknesses:
        return str(weaknesses[0])

    return "Review detailed CV analysis"


def get_job_fit_score(selected_job_result):
    matching_result = selected_job_result.get("matching_result", {})
    final_result = matching_result.get("final_result", {})

    return final_result.get("final_hybrid_score")


def get_job_fit_category(selected_job_result):
    matching_result = selected_job_result.get("matching_result", {})
    final_result = matching_result.get("final_result", {})

    return final_result.get("match_category")


def get_recommended_next_steps(agent_result, selected_job_result):
    recommendation_output = selected_job_result.get("recommendation_output", {})

    if recommendation_output:
        recommendations = recommendation_output.get("recommendations", recommendation_output)
        priority_actions = recommendations.get("priority_actions", [])

        if priority_actions:
            return priority_actions

    cv_recommendations_digest = agent_result.get("cv_recommendations_digest", {})
    cv_recommendations = cv_recommendations_digest.get("recommendations", [])

    if cv_recommendations:
        return cv_recommendations

    cv_quality_analysis = agent_result.get("cv_quality_analysis", {})
    return cv_quality_analysis.get("cv_improvement_recommendations", [])


def render_agent_page():
    st.title("Improve Your CV for IT Job Applications")
    st.caption(
        "Powered by the CV-Job Matching AI Agent. Upload your CV, optionally add a job posting, "
        "and get practical improvement suggestions."
    )

    render_feature_overview()

    st.markdown("## Analysis Setup")

    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_cv = st.file_uploader(
                "1. Upload your CV PDF",
                type=["pdf"],
            )

            analysis_mode = st.radio(
                "2. Analysis type",
                [
                    ANALYSIS_MODE_CV_ONLY,
                    ANALYSIS_MODE_ONE_JOB,
                    ANALYSIS_MODE_MULTIPLE_JOBS,
                ],
                index=0,
            )

        with col2:
            job_mode_enabled = analysis_mode != ANALYSIS_MODE_CV_ONLY
            job_text = ""

            if job_mode_enabled:
                if analysis_mode == ANALYSIS_MODE_ONE_JOB:
                    job_label = "3. Paste job posting"
                    job_placeholder = "Paste one IT job posting here."
                else:
                    job_label = "3. Paste multiple job postings"
                    job_placeholder = (
                        "Paste multiple IT job postings here.\n\n"
                        "Separate job ads with:\n---JOB---"
                    )

                job_text = st.text_area(
                    job_label,
                    height=220,
                    placeholder=job_placeholder,
                )
            else:
                st.info(
                    "Job posting is optional for CV Review only. "
                    "You can add a job posting later to calculate job fit."
                )

        if analysis_mode == ANALYSIS_MODE_CV_ONLY:
            button_label = "Analyze My CV"
        elif analysis_mode == ANALYSIS_MODE_ONE_JOB:
            button_label = "Analyze CV and Job"
        else:
            button_label = "Compare Jobs"

        analyze_button = st.button(
            button_label,
            type="primary",
            use_container_width=True,
        )

    if analyze_button:
        st.session_state.last_error = None
        st.session_state.agent_result = None
        st.session_state.display_job_id = None
        reset_result_visibility()

        if uploaded_cv is None:
            st.warning("Please upload a CV PDF first.")
            return

        job_texts = []

        if job_mode_enabled:
            job_texts = split_job_texts(job_text)

            if not job_texts:
                st.warning("Please paste at least one job posting.")
                return

            if analysis_mode == ANALYSIS_MODE_ONE_JOB:
                job_texts = [job_texts[0]]

        use_mongodb = job_mode_enabled

        try:
            with st.spinner("Saving CV PDF..."):
                saved_cv_path = save_uploaded_cv_pdf(uploaded_cv)

            with st.spinner("Running analysis... This may take a few minutes."):
                agent_result = run_agent_workflow(
                    cv_pdf_path=str(saved_cv_path),
                    job_texts=job_texts,
                    user_wants_cv_recommendations=True,
                    user_wants_job_matching=job_mode_enabled,
                    user_wants_job_recommendations=job_mode_enabled,
                    user_wants_detailed_report=False,
                    use_mongodb=use_mongodb,
                    cv_quality_weights=st.session_state.cv_quality_weights,
                    direct_matching_weights=st.session_state.direct_matching_weights,
                    matching_mode_weights=st.session_state.matching_mode_weights,
                    model_name="gpt-4o-mini",
                    temperature=0,
                )

            st.session_state.agent_result = agent_result
            st.session_state.current_analysis_mode = analysis_mode
            st.session_state.last_saved_cv_path = str(saved_cv_path)
            st.session_state.last_job_texts = job_texts
            st.session_state.last_use_mongodb = use_mongodb

        except Exception as error:
            st.session_state.last_error = str(error)
            st.error("Analysis failed.")
            st.exception(error)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    render_agent_results()


def render_agent_results():
    agent_result = st.session_state.agent_result

    if agent_result is None:
        return

    st.markdown("## Your Results")

    job_results = agent_result.get("job_results", [])
    selected_job_result = {}

    if job_results:
        selected_job_result = get_selected_job_result(job_results)

    render_quick_summary(agent_result, selected_job_result)
    render_next_actions(selected_job_result)
    render_requested_sections(agent_result, selected_job_result)


def render_quick_summary(agent_result, selected_job_result):
    cv_digest = agent_result.get("cv_digest", {})

    st.success("Your CV analysis is ready.")
    st.caption("Here is a short summary of the most important findings and recommended next steps.")

    cv_quality_score = cv_digest.get("final_cv_quality_score")
    cv_category = cv_digest.get("cv_quality_category")

    job_fit_score = get_job_fit_score(selected_job_result)
    job_fit_category = get_job_fit_category(selected_job_result)

    top_priority = extract_top_priority(agent_result, selected_job_result)

    if selected_job_result:
        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            with st.container(border=True):
                st.metric(
                    "CV Quality Score",
                    format_score(cv_quality_score),
                )
                st.caption(cv_category or "")

        with col2:
            with st.container(border=True):
                st.metric(
                    "Job Fit Score",
                    format_score(job_fit_score),
                )
                st.caption(job_fit_category or "")

        with col3:
            with st.container(border=True):
                st.markdown("#### Top Priority")
                st.write(shorten_text(top_priority, 135))

                if len(str(top_priority)) > 135:
                    with st.expander("Show full priority"):
                        st.write(top_priority)
    else:
        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            with st.container(border=True):
                st.metric(
                    "CV Quality Score",
                    format_score(cv_quality_score),
                )
                st.caption(cv_category or "")

        with col2:
            with st.container(border=True):
                st.metric(
                    "CV Category",
                    cv_category or "N/A",
                )

        with col3:
            with st.container(border=True):
                st.markdown("#### Top Priority")
                st.write(shorten_text(top_priority, 135))

                if len(str(top_priority)) > 135:
                    with st.expander("Show full priority"):
                        st.write(top_priority)

    st.markdown("### Summary")

    strengths = cv_digest.get("top_strengths", [])[:4]
    weaknesses = cv_digest.get("top_weaknesses", [])[:4]
    next_steps = get_recommended_next_steps(agent_result, selected_job_result)[:4]

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("#### Main strengths")
            show_bullet_list(
                strengths,
                empty_message="No strengths listed.",
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### Improvement priorities")
            show_bullet_list(
                weaknesses,
                empty_message="No improvement priorities listed.",
            )

    with col3:
        with st.container(border=True):
            st.markdown("#### Recommended next steps")
            show_bullet_list(
                next_steps,
                empty_message="No recommended next steps available.",
            )

    if selected_job_result:
        render_compact_job_fit_preview(selected_job_result)


def render_compact_job_fit_preview(selected_job_result):
    matching_result = selected_job_result.get("matching_result", {})
    final_result = matching_result.get("final_result", {})
    missing_items = matching_result.get("missing_items", {})

    st.markdown("### Job Fit Preview")

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                "Job Fit Score",
                format_score(final_result.get("final_hybrid_score")),
            )
            st.caption(final_result.get("match_category") or "")

        with col2:
            st.markdown("#### Missing skills / requirements")

            missing_required = missing_items.get("missing_required_skills", [])
            missing_technology = missing_items.get("missing_technology_skills", [])

            combined_missing = (missing_required + missing_technology)[:8]

            show_bullet_list(
                combined_missing,
                empty_message="No major missing skills listed.",
            )


def render_next_actions(selected_job_result):
    st.markdown("## Next Actions")

    has_job_result = bool(selected_job_result)

    if has_job_result:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("Detailed CV analysis", use_container_width=True):
                st.session_state.show_cv_details = True

        with col2:
            if st.button("Recommendations", use_container_width=True):
                st.session_state.show_recommendations = True

        with col3:
            if st.button("Matching details", use_container_width=True):
                st.session_state.show_matching_details = True

        with col4:
            if st.button("Generate full report", use_container_width=True):
                generate_full_report_for_current_analysis()

        with col5:
            if st.button("Extracted CV data", use_container_width=True):
                st.session_state.show_extracted_cv_data = True
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Detailed CV analysis", use_container_width=True):
                st.session_state.show_cv_details = True

        with col2:
            if st.button("Recommendations", use_container_width=True):
                st.session_state.show_recommendations = True

        with col3:
            if st.button("CV quality report", use_container_width=True):
                st.session_state.show_cv_quality_report = True

        with col4:
            if st.button("Extracted CV data", use_container_width=True):
                st.session_state.show_extracted_cv_data = True


def generate_full_report_for_current_analysis():
    if not st.session_state.last_saved_cv_path:
        st.warning("No previous CV analysis was found. Please run the analysis first.")
        return

    if not st.session_state.last_job_texts:
        st.info(
            "A full CV-job report requires a job posting. "
            "For CV Review only, use the CV quality report instead."
        )
        st.session_state.show_cv_quality_report = True
        return

    try:
        with st.spinner("Generating full report... This may take a few minutes."):
            agent_result = run_agent_workflow(
                cv_pdf_path=st.session_state.last_saved_cv_path,
                job_texts=st.session_state.last_job_texts,
                user_wants_cv_recommendations=True,
                user_wants_job_matching=True,
                user_wants_job_recommendations=True,
                user_wants_detailed_report=True,
                selected_job_id=st.session_state.display_job_id,
                use_mongodb=st.session_state.last_use_mongodb,
                cv_quality_weights=st.session_state.cv_quality_weights,
                direct_matching_weights=st.session_state.direct_matching_weights,
                matching_mode_weights=st.session_state.matching_mode_weights,
                model_name="gpt-4o-mini",
                temperature=0,
            )

        st.session_state.agent_result = agent_result
        st.session_state.show_final_report = True

    except Exception as error:
        st.error("Full report generation failed.")
        st.exception(error)


def render_requested_sections(agent_result, selected_job_result):
    if st.session_state.show_cv_details:
        render_cv_analysis_section(agent_result)

    if st.session_state.show_recommendations:
        render_recommendations_section(agent_result, selected_job_result)

    if st.session_state.show_matching_details:
        render_matching_section(selected_job_result)

    if st.session_state.show_final_report:
        render_final_report_section(agent_result, selected_job_result)

    if st.session_state.show_cv_quality_report:
        render_cv_quality_report_section(agent_result)

    if st.session_state.show_extracted_cv_data:
        render_extracted_cv_data_section(agent_result)


def render_cv_analysis_section(agent_result):
    cv_quality_analysis = agent_result.get("cv_quality_analysis", {})

    section_break("Detailed CV Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Final CV Quality Score",
            format_score(cv_quality_analysis.get("final_cv_quality_score")),
        )

    with col2:
        st.metric(
            "CV Quality Category",
            cv_quality_analysis.get("cv_quality_category") or "N/A",
        )

    st.markdown("### Overall summary")
    st.write(cv_quality_analysis.get("overall_summary"))

    st.markdown("### Score breakdown")
    scores = cv_quality_analysis.get("scores", {})

    if scores:
        show_dict_as_table(scores)
    else:
        st.info("No CV score breakdown available.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Strengths")
        show_bullet_list(
            cv_quality_analysis.get("strengths", []),
            empty_message="No strengths listed.",
        )

    with col2:
        st.markdown("### Weaknesses")
        show_bullet_list(
            cv_quality_analysis.get("weaknesses", []),
            empty_message="No weaknesses listed.",
        )

    st.markdown("### Missing or unclear sections")
    show_bullet_list(
        cv_quality_analysis.get("missing_or_unclear_sections", []),
        empty_message="No missing or unclear sections listed.",
    )


def render_cv_quality_report_section(agent_result):
    section_break("CV Quality Report")

    cv_quality_result = agent_result.get("cv_quality_result", {})
    cv_markdown_report = cv_quality_result.get("markdown_report")

    if not cv_markdown_report:
        st.info("CV quality report is not available.")
        return

    with st.expander("Preview CV Quality Report", expanded=True):
        st.markdown(cv_markdown_report)

    download_markdown_button(
        markdown_text=cv_markdown_report,
        file_name="cv_quality_report.md",
        label="Download CV Quality Report",
    )


def render_recommendations_section(agent_result, selected_job_result):
    section_break("Recommendations")

    cv_recommendations_digest = agent_result.get("cv_recommendations_digest", {})

    st.markdown("### CV Improvement Suggestions")

    if cv_recommendations_digest:
        show_bullet_list(
            cv_recommendations_digest.get("recommendations", []),
            empty_message="No CV recommendations available.",
        )
    else:
        cv_quality_analysis = agent_result.get("cv_quality_analysis", {})
        show_bullet_list(
            cv_quality_analysis.get("cv_improvement_recommendations", []),
            empty_message="No CV recommendations available.",
        )

    if not selected_job_result:
        return

    recommendation_output = selected_job_result.get("recommendation_output")

    if not recommendation_output:
        st.info("Job-specific recommendations were not generated.")
        return

    recommendations = recommendation_output.get("recommendations", recommendation_output)

    st.markdown("### Job-Specific Recommendations")

    st.markdown("#### Overall recommendation summary")
    st.write(recommendations.get("overall_recommendation_summary"))

    st.markdown("#### Priority actions")
    show_recommendation_actions(
        recommendations.get("priority_actions", []),
    )

    with st.expander("CV improvement recommendations", expanded=False):
        show_recommendation_items(
            recommendations.get("cv_improvement_recommendations", []),
        )

    with st.expander("Missing required skills recommendations", expanded=False):
        show_skill_recommendation_items(
            recommendations.get("missing_required_skills_recommendations", []),
        )

    with st.expander("Technical development recommendations", expanded=False):
        show_skill_recommendation_items(
            recommendations.get("technical_development_recommendations", []),
        )

    with st.expander("Project recommendations", expanded=False):
        show_recommendation_items(
            recommendations.get("project_recommendations", []),
        )

    with st.expander("Soft skills recommendations", expanded=False):
        show_recommendation_items(
            recommendations.get("soft_skills_recommendations", []),
        )

    recommendations_markdown_report = selected_job_result.get("recommendations_markdown_report")

    if recommendations_markdown_report:
        download_markdown_button(
            markdown_text=recommendations_markdown_report,
            file_name="recommendations_report.md",
            label="Download Recommendations Report",
        )


def render_matching_section(selected_job_result):
    section_break("Matching Details")

    if not selected_job_result:
        st.info("No matching details are available because no job posting was analyzed.")
        return

    matching_result = selected_job_result.get("matching_result")

    if not matching_result:
        st.info("Matching result is not available for the selected job.")
        return

    final_result = matching_result.get("final_result", {})
    score_breakdown = matching_result.get("score_breakdown", {})
    matched_items = matching_result.get("matched_items", {})
    missing_items = matching_result.get("missing_items", {})
    semantic_analysis = matching_result.get("semantic_analysis", {})
    job_information = matching_result.get("job_information", {})

    st.markdown("### Selected Job")
    show_dict_as_table(job_information)

    st.markdown("### Matching Scores")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Hybrid Score",
            format_score(final_result.get("final_hybrid_score")),
        )

    with col2:
        st.metric(
            "Direct Matching Score",
            format_score(final_result.get("direct_matching_score")),
        )

    with col3:
        st.metric(
            "Semantic Score",
            format_score(final_result.get("semantic_score")),
        )

    with col4:
        st.metric(
            "Match Category",
            final_result.get("match_category") or "N/A",
        )

    st.markdown("### Score breakdown")
    show_dict_as_table(score_breakdown)

    st.markdown("### Matched and Missing Requirements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matched required skills")
        show_bullet_list(
            matched_items.get("matched_required_skills", []),
            empty_message="No matched required skills.",
        )

        st.markdown("#### Matched technology skills")
        show_bullet_list(
            matched_items.get("matched_technology_skills", []),
            empty_message="No matched technology skills.",
        )

    with col2:
        st.markdown("#### Missing required skills")
        show_bullet_list(
            missing_items.get("missing_required_skills", []),
            empty_message="No missing required skills.",
        )

        st.markdown("#### Missing technology skills")
        show_bullet_list(
            missing_items.get("missing_technology_skills", []),
            empty_message="No missing technology skills.",
        )

    st.markdown("### Semantic Matching Analysis")

    st.markdown("#### Role fit summary")
    st.write(semantic_analysis.get("role_fit_summary"))

    with st.expander("Responsibilities and soft skills evidence", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Responsibilities evidenced")
            show_bullet_list(
                semantic_analysis.get("responsibilities_evidenced", []),
                empty_message="No evidenced responsibilities listed.",
            )

            st.markdown("#### Soft skills evidenced")
            show_bullet_list(
                semantic_analysis.get("soft_skills_evidenced", []),
                empty_message="No evidenced soft skills listed.",
            )

        with col2:
            st.markdown("#### Responsibilities not clearly evidenced")
            show_bullet_list(
                semantic_analysis.get("responsibilities_not_evidenced", []),
                empty_message="No responsibility gaps listed.",
            )

            st.markdown("#### Soft skills not clearly evidenced")
            show_bullet_list(
                semantic_analysis.get("soft_skills_not_clearly_evidenced", []),
                empty_message="No soft skill gaps listed.",
            )

    matching_markdown_report = selected_job_result.get("matching_markdown_report")

    if matching_markdown_report:
        download_markdown_button(
            markdown_text=matching_markdown_report,
            file_name="matching_report.md",
            label="Download Matching Report",
        )


def render_final_report_section(agent_result, selected_job_result):
    section_break("Final Report")

    if not selected_job_result:
        st.info(
            "A full CV-job report requires a job posting. "
            "For CV Review only, use the CV quality report instead."
        )
        return

    final_report_markdown = selected_job_result.get("final_report_markdown")

    if not final_report_markdown:
        st.info("Full report was not generated yet. Click 'Generate full report' first.")
        return

    with st.expander("Preview Final Report", expanded=True):
        st.markdown(final_report_markdown)

    download_markdown_button(
        markdown_text=final_report_markdown,
        file_name="final_report.md",
        label="Download Final Report",
    )


def clean_table_records(records):
    if not records:
        return pd.DataFrame()

    cleaned_records = []

    for record in records:
        if isinstance(record, dict):
            cleaned_records.append(record)
        else:
            cleaned_records.append({"Value": record})

    return pd.DataFrame(cleaned_records)


def render_extracted_cv_data_section(agent_result):
    section_break("Extracted CV Data")

    structured_cv = agent_result.get("structured_cv", {})

    if not structured_cv:
        st.info("Extracted CV data is not available.")
        return

    st.markdown("### Candidate Information")

    candidate_data = {
        "Name": structured_cv.get("candidate_name"),
        "Email": structured_cv.get("email"),
        "Phone": structured_cv.get("phone"),
        "Location": structured_cv.get("location"),
        "LinkedIn": structured_cv.get("linkedin_url"),
        "GitHub": structured_cv.get("github_url"),
        "Portfolio": structured_cv.get("portfolio_url"),
        "Total years of experience": structured_cv.get("total_years_of_experience"),
        "Profile summary": structured_cv.get("profile_summary"),
    }

    show_dict_as_table(candidate_data)

    st.markdown("### Skills")

    skill_fields = {
        "Technical skills": structured_cv.get("technical_skills", []),
        "Programming languages": structured_cv.get("programming_languages", []),
        "Frameworks and libraries": structured_cv.get("frameworks_and_libraries", []),
        "Databases": structured_cv.get("databases", []),
        "Cloud and DevOps tools": structured_cv.get("cloud_and_devops_tools", []),
        "Data and AI tools": structured_cv.get("data_and_ai_tools", []),
        "Other tools": structured_cv.get("other_tools", []),
        "Soft skills": structured_cv.get("soft_skills", []),
        "Languages": structured_cv.get("languages", []),
    }

    skill_rows = []

    for category, values in skill_fields.items():
        if not values:
            continue

        for value in values:
            skill_rows.append(
                {
                    "Category": category,
                    "Extracted value": value,
                }
            )

    if skill_rows:
        st.dataframe(
            pd.DataFrame(skill_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No skills were extracted.")

    with st.expander("Education", expanded=False):
        df = clean_table_records(structured_cv.get("education", []))

        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No education entries extracted.")

    with st.expander("Work Experience", expanded=False):
        df = clean_table_records(structured_cv.get("work_experience", []))

        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No work experience entries extracted.")

    with st.expander("Projects", expanded=False):
        df = clean_table_records(structured_cv.get("projects", []))

        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No project entries extracted.")

    with st.expander("Certifications", expanded=False):
        df = clean_table_records(structured_cv.get("certifications", []))

        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No certifications extracted.")

    with st.expander("Missing or unclear extracted information", expanded=False):
        show_bullet_list(
            structured_cv.get("unclear_or_missing_information", []),
            empty_message="No unclear extracted information listed.",
        )


def render_market_statistics_page():
    st.title("Market Statistics")
    st.caption("Statistics are calculated from analyzed job postings stored in MongoDB.")

    col1, col2, col3 = st.columns(3)

    with col1:
        top_n = st.number_input(
            "Top N",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

    with col2:
        weighted = st.checkbox(
            "Weight by submission count",
            value=True,
        )

    with col3:
        load_button = st.button("Load Statistics", type="primary")

    start_date = st.text_input(
        "Start date optional",
        placeholder="YYYY-MM-DD",
    )

    end_date = st.text_input(
        "End date optional",
        placeholder="YYYY-MM-DD",
    )

    if load_button:
        try:
            with st.spinner("Loading market statistics from MongoDB..."):
                result = process_market_statistics(
                    start_date=start_date or None,
                    end_date=end_date or None,
                    top_n=int(top_n),
                    weighted=weighted,
                )

            st.session_state.market_statistics = result["market_statistics_dashboard"]

        except Exception as error:
            st.error("Could not load market statistics.")
            st.exception(error)

    dashboard = st.session_state.market_statistics

    if dashboard is None:
        st.info("Click 'Load Statistics' to show dashboard data.")
        return

    metrics = dashboard.get("metrics", {})

    st.markdown("## Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Unique Jobs", metrics.get("unique_jobs_count"))

    with c2:
        st.metric("Total Submissions", metrics.get("total_submissions"))

    with c3:
        st.metric("Top Skill", metrics.get("most_requested_skill"))

    with c4:
        st.metric("Top Category", metrics.get("top_job_category"))

    st.markdown("## Monthly Trends")

    monthly_submissions = records_to_dataframe(
        dashboard.get("monthly_submissions", [])
    )

    if not monthly_submissions.empty:
        st.bar_chart(
            data=monthly_submissions,
            x="month",
            y="submission_count",
            use_container_width=True,
        )
    else:
        st.info("No monthly submission data available.")

    st.markdown("## Jobs by Category")

    jobs_by_category = records_to_dataframe(
        dashboard.get("jobs_by_category", [])
    )

    if not jobs_by_category.empty:
        st.bar_chart(
            data=jobs_by_category,
            x="job_category",
            y="count",
            use_container_width=True,
        )

        st.dataframe(
            jobs_by_category,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No job category data available.")

    st.markdown("## Most Requested Required Skills")

    required_skills = records_to_dataframe(
        dashboard.get("most_requested_required_skills", [])
    )

    if not required_skills.empty:
        st.bar_chart(
            data=required_skills,
            x="skill",
            y="count",
            use_container_width=True,
        )

        st.dataframe(
            required_skills,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No required skills data available.")

    st.markdown("## Technology Statistics")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Programming Languages",
            "Frameworks",
            "Databases",
            "Cloud / DevOps",
        ]
    )

    with tab1:
        df = records_to_dataframe(
            dashboard.get("most_requested_programming_languages", [])
        )

        if not df.empty:
            st.bar_chart(df, x="programming_language", y="count")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No programming language data available.")

    with tab2:
        df = records_to_dataframe(
            dashboard.get("most_requested_frameworks_and_libraries", [])
        )

        if not df.empty:
            st.bar_chart(df, x="framework_or_library", y="count")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No framework data available.")

    with tab3:
        df = records_to_dataframe(
            dashboard.get("most_requested_databases", [])
        )

        if not df.empty:
            st.bar_chart(df, x="database", y="count")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No database data available.")

    with tab4:
        df = records_to_dataframe(
            dashboard.get("most_requested_cloud_and_devops_tools", [])
        )

        if not df.empty:
            st.bar_chart(df, x="cloud_or_devops_tool", y="count")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No cloud or DevOps data available.")


def calculate_weights_sum(weights):
    if not weights:
        return 0.0

    total = 0.0

    for value in weights.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue

    return round(total, 4)


def show_weight_group_status(group_name, weights):
    total_weight = calculate_weights_sum(weights)
    total_percent = round(total_weight * 100, 2)

    if abs(total_weight - 1.0) <= 0.001:
        st.success(f"{group_name} total: {total_weight} / {total_percent}%")
        return True

    if total_weight < 1.0:
        st.warning(
            f"{group_name} total: {total_weight} / {total_percent}%. "
            "Weights are below 100%."
        )
        return False

    st.error(
        f"{group_name} total: {total_weight} / {total_percent}%. "
        "Weights exceed 100%."
    )
    return False


def render_weight_group_inputs(title, weights, key_prefix):
    st.markdown(f"## {title}")

    updated_weights = {}

    for key, value in weights.items():
        updated_weights[key] = st.number_input(
            key,
            value=float(value),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=f"{key_prefix}_{key}",
        )

    group_is_valid = show_weight_group_status(
        title,
        updated_weights,
    )

    return updated_weights, group_is_valid


def render_settings_page():
    st.title("Settings")
    st.caption("Default scoring weights used by the application.")

    st.info(
        "Each weight group should sum to 1.00, which represents 100%. "
        "The backend can normalize weights automatically, but the UI shows a warning "
        "when a group is below or above 100%."
    )

    cv_quality_weights, cv_quality_valid = render_weight_group_inputs(
        title="CV Quality Weights",
        weights=st.session_state.cv_quality_weights,
        key_prefix="cv_quality",
    )

    st.session_state.cv_quality_weights = cv_quality_weights

    direct_matching_weights, direct_matching_valid = render_weight_group_inputs(
        title="Direct Matching Weights",
        weights=st.session_state.direct_matching_weights,
        key_prefix="direct_matching",
    )

    st.session_state.direct_matching_weights = direct_matching_weights

    matching_mode_weights, matching_mode_valid = render_weight_group_inputs(
        title="Final Matching Mode Weights",
        weights=st.session_state.matching_mode_weights,
        key_prefix="matching_mode",
    )

    st.session_state.matching_mode_weights = matching_mode_weights

    st.divider()

    if cv_quality_valid and direct_matching_valid and matching_mode_valid:
        st.success("All weight groups are valid.")
    else:
        st.warning(
            "One or more weight groups do not sum to 1.00. "
            "You can still run the analysis, but it is recommended to adjust them."
        )

    if st.button("Reset Default Weights"):
        st.session_state.cv_quality_weights = DEFAULT_CV_QUALITY_WEIGHTS.copy()
        st.session_state.direct_matching_weights = DEFAULT_DIRECT_MATCHING_WEIGHTS.copy()
        st.session_state.matching_mode_weights = DEFAULT_MATCHING_MODE_WEIGHTS.copy()
        st.rerun()


def main():
    initialize_session_state()

    page = render_sidebar()

    if page == "Analysis Workspace":
        render_agent_page()

    elif page == "Market Statistics":
        render_market_statistics_page()

    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()