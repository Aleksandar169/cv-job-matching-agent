import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def load_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #071A2F 0%, #061525 100%);
            color: #F8FAFC;
        }

        section[data-testid="stSidebar"] {
            background-color: #061525;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .main-title {
            font-size: 36px;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 6px;
            line-height: 1.15;
        }

        .subtitle {
            font-size: 16px;
            color: #CBD5E1;
            margin-bottom: 24px;
            line-height: 1.5;
        }

        .section-title {
            font-size: 23px;
            font-weight: 750;
            color: #F8FAFC;
            margin-top: 30px;
            margin-bottom: 14px;
        }

        .feature-card {
            background-color: #0D2742;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 18px;
            min-height: 128px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.15);
        }

        .feature-icon {
            color: #22C55E;
            font-size: 28px;
            margin-bottom: 8px;
        }

        .feature-title {
            color: #F8FAFC;
            font-size: 18px;
            font-weight: 750;
            margin-bottom: 6px;
        }

        .feature-text {
            color: #CBD5E1;
            font-size: 14px;
            line-height: 1.45;
        }

        .summary-banner {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.16), rgba(13, 39, 66, 0.88));
            border: 1px solid rgba(34, 197, 94, 0.55);
            border-radius: 16px;
            padding: 18px;
            margin-top: 14px;
            margin-bottom: 18px;
        }

        .summary-banner-title {
            color: #22C55E;
            font-weight: 800;
            font-size: 20px;
            margin-bottom: 4px;
        }

        .summary-banner-text {
            color: #CBD5E1;
            font-size: 14px;
        }

        .metric-card {
            background-color: #0D2742;
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 12px;
            min-height: 118px;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.16);
        }

        .metric-label {
            color: #CBD5E1;
            font-size: 14px;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #22C55E;
            font-size: 28px;
            font-weight: 800;
            word-break: break-word;
        }

        .metric-note {
            color: #94A3B8;
            font-size: 13px;
            margin-top: 5px;
        }

        .info-card {
            background-color: #0D2742;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .info-card-title {
            color: #F8FAFC;
            font-size: 17px;
            font-weight: 750;
            margin-bottom: 8px;
        }

        .info-card-text {
            color: #CBD5E1;
            font-size: 14px;
            line-height: 1.5;
        }

        div.stButton > button {
            background-color: #22C55E;
            color: #061525;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        div.stButton > button:hover {
            background-color: #16A34A;
            color: white;
        }

        textarea {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<div class='main-title'>{title}</div>",
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f"<div class='subtitle'>{subtitle}</div>",
            unsafe_allow_html=True,
        )


def section_title(title: str):
    st.markdown(
        f"<div class='section-title'>{title}</div>",
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, text: str):
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_banner(title: str, text: str):
    st.markdown(
        f"""
        <div class="summary-banner">
            <div class="summary-banner-title">{title}</div>
            <div class="summary-banner-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, note: str = ""):
    if value is None:
        value = "N/A"

    note_html = ""

    if note:
        note_html = f"<div class='metric-note'>{note}</div>"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dict_as_table(data: Dict[str, Any]):
    if not data:
        st.info("No data available.")
        return

    rows = []

    for key, value in data.items():
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


def records_to_dataframe(records: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def safe_get(data: Dict[str, Any], keys: List[str], default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def show_bullet_list(items: Optional[List[Any]], empty_message: str = "No items available."):
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


def calculate_weights_sum(weights: Dict[str, float]) -> float:
    if not weights:
        return 0.0

    total = 0.0

    for value in weights.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue

    return round(total, 4)


def show_weight_group_status(group_name: str, weights: Dict[str, float]) -> bool:
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


def show_recommendation_actions(actions: Optional[List[Dict[str, Any]]]):
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


def show_recommendation_items(items: Optional[List[Dict[str, Any]]]):
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


def show_skill_recommendation_items(items: Optional[List[Dict[str, Any]]]):
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


def create_unique_download_key(prefix: str) -> str:
    if "_download_button_counter" not in st.session_state:
        st.session_state["_download_button_counter"] = 0

    st.session_state["_download_button_counter"] += 1

    return f"{prefix}_{st.session_state['_download_button_counter']}"


def download_json_button(data: Any, file_name: str, label: str, key: str = None):
    if key is None:
        key = create_unique_download_key("download_json")

    st.download_button(
        label=label,
        data=json.dumps(data, indent=4, ensure_ascii=False),
        file_name=file_name,
        mime="application/json",
        key=key,
    )


def download_markdown_button(markdown_text: str, file_name: str, label: str, key: str = None):
    if key is None:
        key = create_unique_download_key("download_markdown")

    st.download_button(
        label=label,
        data=markdown_text,
        file_name=file_name,
        mime="text/markdown",
        key=key,
    )