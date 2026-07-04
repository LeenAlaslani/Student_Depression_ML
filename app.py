from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 1. APP SETTINGS
# =========================================================
st.set_page_config(
    page_title="Behind the Numbers | Student Depression",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "student_lifestyle.csv"
SCALER_FILE = BASE_DIR / "scaler.joblib"
FEATURES_FILE = BASE_DIR / "feature_columns.joblib"
RESULTS_FILE = BASE_DIR / "model_results.csv"
MATRICES_FILE = BASE_DIR / "confusion_matrices.joblib"

MODEL_FILES = {
    "Logistic Regression": BASE_DIR / "logistic_model.joblib",
    "Decision Tree": BASE_DIR / "decision_tree_model.joblib",
    "Random Forest": BASE_DIR / "random_forest_model.joblib",
}

COLORS = {
    "ink": "#101828",
    "muted": "#667085",
    "surface": "#FFFFFF",
    "surface_soft": "#F7F8FC",
    "line": "#E7EAF2",
    "purple": "#6D5DFB",
    "purple_dark": "#3D35A5",
    "purple_soft": "#EFEDFF",
    "blue": "#3578F6",
    "teal": "#19A989",
    "orange": "#F4A261",
    "rose": "#EB5F76",
    "navy": "#161B3B",
}

MODEL_COLORS = {
    "Logistic Regression": COLORS["purple"],
    "Decision Tree": COLORS["teal"],
    "Random Forest": COLORS["orange"],
}

CLASS_COLORS = {
    "No Depression": COLORS["teal"],
    "Depression": COLORS["rose"],
}


# =========================================================
# 2. GLOBAL DESIGN
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --ink: #101828;
            --muted: #667085;
            --surface: #ffffff;
            --surface-soft: #f7f8fc;
            --line: #e7eaf2;
            --purple: #6d5dfb;
            --purple-dark: #3d35a5;
            --purple-soft: #efedff;
            --blue: #3578f6;
            --teal: #19a989;
            --orange: #f4a261;
            --rose: #eb5f76;
            --navy: #161b3b;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 4%, rgba(109, 93, 251, 0.08), transparent 24%),
                linear-gradient(180deg, #fbfbfe 0%, #f7f8fc 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        h1 {
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.02;
        }

        h2 {
            margin-top: 1.8rem;
        }

        p, label, .stCaption {
            color: var(--muted);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(109, 93, 251, .20), transparent 28%),
                linear-gradient(180deg, #171b3c 0%, #101329 100%);
            border-right: 0;
        }

        section[data-testid="stSidebar"] * {
            color: rgba(255,255,255,.92);
        }

        section[data-testid="stSidebar"] .stCaption {
            color: rgba(255,255,255,.58) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: rgba(255,255,255,.72);
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,.12);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: transparent;
            border: 1px solid transparent;
            padding: .65rem .72rem;
            border-radius: 12px;
            transition: .18s ease;
            margin-bottom: .22rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,.08);
            border-color: rgba(255,255,255,.10);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(109,93,251,.95), rgba(79,70,190,.95));
            box-shadow: 0 8px 24px rgba(0,0,0,.20);
        }

        /* Buttons */
        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 14px;
            border: 0;
            font-weight: 700;
            min-height: 46px;
            background: linear-gradient(135deg, var(--purple), var(--purple-dark));
            color: white;
            box-shadow: 0 10px 24px rgba(109, 93, 251, .22);
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border: 0;
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(109, 93, 251, .30);
        }

        .stButton > button p,
        .stFormSubmitButton > button p {
            color: white !important;
            font-weight: 800 !important;
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput input {
            border-radius: 12px !important;
            border-color: var(--line) !important;
            background: white !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255,255,255,.75);
            box-shadow: 0 10px 30px rgba(16,24,40,.04);
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
            background: white;
            box-shadow: 0 10px 30px rgba(16,24,40,.04);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: .4rem;
            background: #eef0f7;
            padding: .35rem;
            border-radius: 14px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: white;
            box-shadow: 0 4px 14px rgba(16,24,40,.08);
        }

        /* Plotly container */
        div[data-testid="stPlotlyChart"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: .4rem;
            box-shadow: 0 14px 34px rgba(16,24,40,.05);
        }

        /* Custom components */
        .brand-wrap {
            padding: .65rem .2rem 1.1rem .2rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            font-size: 1.25rem;
            background: linear-gradient(135deg, #7c6cff, #4f46be);
            box-shadow: 0 12px 28px rgba(109,93,251,.36);
            margin-bottom: .8rem;
        }

        .brand-title {
            color: white;
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            color: rgba(255,255,255,.56);
            font-size: .78rem;
            margin-top: .2rem;
        }

        .side-note {
            margin-top: 1rem;
            padding: .9rem;
            border-radius: 14px;
            background: rgba(255,255,255,.06);
            border: 1px solid rgba(255,255,255,.09);
            font-size: .77rem;
            line-height: 1.5;
            color: rgba(255,255,255,.66);
        }

        .hero {
            position: relative;
            overflow: hidden;
            min-height: 300px;
            padding: clamp(2rem, 5vw, 4.3rem);
            border-radius: 30px;
            background:
                radial-gradient(circle at 85% 18%, rgba(161,151,255,.55), transparent 21%),
                radial-gradient(circle at 72% 70%, rgba(53,120,246,.25), transparent 27%),
                linear-gradient(135deg, #151a3b 0%, #2a2e68 54%, #594aa0 100%);
            box-shadow: 0 28px 70px rgba(31, 35, 79, .22);
            color: white;
            margin-bottom: 1.35rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 310px;
            height: 310px;
            right: -60px;
            bottom: -100px;
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 50%;
            box-shadow:
                0 0 0 38px rgba(255,255,255,.035),
                0 0 0 78px rgba(255,255,255,.025);
        }

        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 820px;
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .45rem .7rem;
            border-radius: 999px;
            background: rgba(255,255,255,.10);
            border: 1px solid rgba(255,255,255,.15);
            color: #d8d4ff;
            font-size: .73rem;
            font-weight: 800;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: 1.1rem;
        }

        .hero h1 {
            color: white;
            margin: 0 0 .9rem 0;
            font-size: clamp(2.6rem, 6vw, 5.2rem);
            max-width: 780px;
        }

        .hero p {
            color: rgba(255,255,255,.76);
            max-width: 760px;
            font-size: 1.05rem;
            line-height: 1.8;
            margin: 0;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.35rem;
        }

        .hero-tag {
            padding: .48rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.13);
            color: rgba(255,255,255,.82);
            font-size: .8rem;
        }

        .section-kicker {
            display: inline-block;
            margin-top: .45rem;
            margin-bottom: .25rem;
            color: var(--purple);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .10em;
            text-transform: uppercase;
        }

        .section-title {
            margin: 0 0 .35rem 0;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
            line-height: 1.1;
            color: var(--ink);
        }

        .section-copy {
            max-width: 850px;
            color: var(--muted);
            line-height: 1.7;
            margin-bottom: 1.2rem;
        }

        .metric-card {
            height: 100%;
            min-height: 126px;
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            background: rgba(255,255,255,.88);
            border: 1px solid var(--line);
            box-shadow: 0 12px 34px rgba(16,24,40,.055);
            position: relative;
            overflow: hidden;
        }

        .metric-card::after {
            content: "";
            position: absolute;
            width: 78px;
            height: 78px;
            right: -30px;
            top: -30px;
            border-radius: 50%;
            background: var(--accent-soft, #efedff);
        }

        .metric-label {
            color: var(--muted);
            font-size: .82rem;
            font-weight: 650;
            margin-bottom: .55rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: clamp(1.6rem, 3vw, 2.35rem);
            font-weight: 820;
            letter-spacing: -0.04em;
            line-height: 1;
        }

        .metric-note {
            margin-top: .55rem;
            color: var(--muted);
            font-size: .75rem;
        }

        .story-card {
            height: 100%;
            min-height: 205px;
            padding: 1.35rem;
            border-radius: 22px;
            background: white;
            border: 1px solid var(--line);
            box-shadow: 0 14px 36px rgba(16,24,40,.05);
            transition: .2s ease;
        }

        .story-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 44px rgba(16,24,40,.08);
        }

        .story-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            background: var(--purple-soft);
            color: var(--purple);
            font-size: 1.15rem;
            margin-bottom: 1rem;
        }

        .story-card h3 {
            font-size: 1.05rem;
            margin: 0 0 .6rem 0;
        }

        .story-card p {
            font-size: .88rem;
            line-height: 1.65;
            margin: 0;
        }

        .timeline {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: .75rem;
            margin-top: .7rem;
        }

        .timeline-item {
            min-height: 118px;
            padding: 1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid var(--line);
            box-shadow: 0 10px 28px rgba(16,24,40,.04);
        }

        .timeline-number {
            width: 26px;
            height: 26px;
            display: grid;
            place-items: center;
            border-radius: 9px;
            background: var(--purple-soft);
            color: var(--purple);
            font-size: .72rem;
            font-weight: 800;
            margin-bottom: .72rem;
        }

        .timeline-title {
            color: var(--ink);
            font-size: .85rem;
            font-weight: 750;
            margin-bottom: .2rem;
        }

        .timeline-status {
            color: var(--teal);
            font-size: .72rem;
            font-weight: 700;
        }

        .insight-card {
            padding: 1.15rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(109,93,251,.10), rgba(53,120,246,.06));
            border: 1px solid rgba(109,93,251,.18);
            color: var(--ink);
            line-height: 1.7;
        }

        .warning-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: #fff8e8;
            border: 1px solid #f8d991;
            color: #744b00;
            font-size: .84rem;
            line-height: 1.6;
        }

        .clean-card {
            height: 100%;
            min-height: 170px;
            padding: 1.25rem;
            border-radius: 20px;
            background: white;
            border: 1px solid var(--line);
            box-shadow: 0 12px 34px rgba(16,24,40,.05);
        }

        .clean-card h3 {
            margin: 0 0 .55rem 0;
            font-size: 1rem;
        }

        .clean-card p {
            margin: 0;
            font-size: .86rem;
            line-height: 1.65;
        }

        .model-card {
            height: 100%;
            min-height: 220px;
            padding: 1.3rem;
            border-radius: 22px;
            background: white;
            border: 1px solid var(--line);
            box-shadow: 0 14px 38px rgba(16,24,40,.055);
            border-top: 5px solid var(--model-color, var(--purple));
        }

        .model-name {
            font-size: 1rem;
            color: var(--ink);
            font-weight: 800;
            margin-bottom: .8rem;
        }

        .model-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .5rem;
            margin-bottom: .9rem;
        }

        .model-metric {
            padding: .65rem .55rem;
            border-radius: 12px;
            background: var(--surface-soft);
            text-align: center;
        }

        .model-metric b {
            display: block;
            color: var(--ink);
            font-size: 1rem;
        }

        .model-metric span {
            color: var(--muted);
            font-size: .68rem;
        }

        .model-copy {
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.58;
        }

        .prediction-card {
            height: 100%;
            min-height: 190px;
            padding: 1.3rem;
            border-radius: 22px;
            background: var(--prediction-bg, white);
            border: 1px solid var(--prediction-line, var(--line));
            box-shadow: 0 14px 36px rgba(16,24,40,.05);
        }

        .prediction-model {
            color: var(--muted);
            font-size: .76rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .07em;
        }

        .prediction-state {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 820;
            margin-top: .7rem;
        }

        .prediction-probability {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 850;
            margin-top: .4rem;
            letter-spacing: -.04em;
        }

        .prediction-note {
            color: var(--muted);
            font-size: .76rem;
            margin-top: .45rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .38rem .6rem;
            border-radius: 999px;
            font-size: .73rem;
            font-weight: 750;
            background: var(--purple-soft);
            color: var(--purple);
        }

        .footer-note {
            margin-top: 2.6rem;
            text-align: center;
            color: #98a2b3;
            font-size: .75rem;
        }

        @media (max-width: 980px) {
            .timeline {
                grid-template-columns: repeat(2, 1fr);
            }

            .hero {
                min-height: auto;
            }
        }

        @media (max-width: 640px) {
            .timeline {
                grid-template-columns: 1fr;
            }

            .hero {
                border-radius: 22px;
                padding: 1.6rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_block(content: str) -> None:
    """Render custom HTML without Markdown exposing indented tags as raw code."""
    cleaned = dedent(content).strip()
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())

    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


def section_header(kicker: str, title: str, copy: str = "") -> None:
    html_block(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """
    )


def metric_card(
    label: str,
    value: str,
    note: str = "",
    accent_soft: str = "#EFEDFF",
) -> None:
    html_block(
        f"""
        <div class="metric-card" style="--accent-soft:{accent_soft};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """
    )


def style_figure(
    figure: go.Figure,
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    figure.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, Arial, sans-serif",
            color=COLORS["ink"],
        ),
        title=dict(
            font=dict(size=18, color=COLORS["ink"]),
            x=0.03,
            xanchor="left",
        ),
        margin=dict(l=40, r=24, t=70, b=45),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, Arial, sans-serif",
        ),
    )

    figure.update_xaxes(
        showgrid=False,
        linecolor=COLORS["line"],
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
    )
    figure.update_yaxes(
        gridcolor="#EFF1F6",
        zeroline=False,
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
    )
    return figure


# =========================================================
# 3. CHECK AND LOAD FILES
# =========================================================
def check_files() -> None:
    required_files = [
        DATA_FILE,
        SCALER_FILE,
        FEATURES_FILE,
        RESULTS_FILE,
        MATRICES_FILE,
        *MODEL_FILES.values(),
    ]

    missing_files = [file.name for file in required_files if not file.exists()]

    if missing_files:
        st.error(
            "The app cannot start because these files are missing: "
            + ", ".join(missing_files)
        )
        st.stop()


check_files()


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    model_results = pd.read_csv(RESULTS_FILE)
    rename_map: dict[str, str] = {}

    for column in model_results.columns:
        simple_name = column.strip().lower().replace("_", " ")

        if simple_name == "model":
            rename_map[column] = "Model"
        elif simple_name == "accuracy":
            rename_map[column] = "Accuracy"
        elif simple_name == "precision":
            rename_map[column] = "Precision"
        elif simple_name == "recall":
            rename_map[column] = "Recall"
        elif simple_name in {"f1", "f1 score", "f1score"}:
            rename_map[column] = "F1 Score"

    model_results = model_results.rename(columns=rename_map)
    needed_columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

    missing_columns = [
        column for column in needed_columns if column not in model_results.columns
    ]

    if missing_columns:
        st.error(
            "model_results.csv is missing these columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    return model_results[needed_columns]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    loaded_models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_FILES.items()
    }

    loaded_scaler = joblib.load(SCALER_FILE)
    loaded_feature_columns = list(joblib.load(FEATURES_FILE))
    loaded_confusion_matrices = joblib.load(MATRICES_FILE)

    return (
        loaded_models,
        loaded_scaler,
        loaded_feature_columns,
        loaded_confusion_matrices,
    )


dataset = load_dataset()
results = load_results()

models, scaler, feature_columns, confusion_matrices = load_artifacts()


# =========================================================
# 4. HELPERS
# =========================================================
def depression_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "Depression" if bool(value) else "No Depression"

    if isinstance(value, (int, float, np.integer, np.floating)):
        return "Depression" if int(value) == 1 else "No Depression"

    text = str(value).strip().lower()
    positive_values = {"1", "true", "yes", "depression", "depressed"}

    return "Depression" if text in positive_values else "No Depression"


def add_target_label(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_dataframe = dataframe.copy()
    new_dataframe["Depression_Label"] = new_dataframe["Depression"].map(
        depression_label
    )
    return new_dataframe


def prepare_model_input(values: dict[str, Any]) -> pd.DataFrame:
    row = {column: 0 for column in feature_columns}

    numerical_values = {
        "Age": values["Age"],
        "CGPA": values["CGPA"],
        "Sleep_Duration": values["Sleep_Duration"],
        "Study_Hours": values["Study_Hours"],
        "Social_Media_Hours": values["Social_Media_Hours"],
        "Physical_Activity": values["Physical_Activity"],
        "Stress_Level": values["Stress_Level"],
        "Low_Sleep": int(values["Sleep_Duration"] < 6),
        "High_Stress": int(values["Stress_Level"] >= 7),
    }

    for column, value in numerical_values.items():
        if column in row:
            row[column] = value

    selected_gender = str(values["Gender"]).strip().lower()
    selected_department = str(values["Department"]).strip().lower()

    for column in feature_columns:
        lower_column = column.lower()

        if lower_column.startswith("gender_"):
            category = column.split("_", 1)[1].strip().lower()
            row[column] = int(category == selected_gender)

        elif lower_column.startswith("department_"):
            category = column.split("_", 1)[1].strip().lower()
            row[column] = int(category == selected_department)

    return pd.DataFrame([row], columns=feature_columns)


def predict_with_model(
    model_name: str,
    prepared_input: pd.DataFrame,
) -> dict[str, Any]:
    selected_model = models[model_name]

    if model_name == "Logistic Regression":
        model_input = scaler.transform(prepared_input)
    else:
        model_input = prepared_input

    prediction = int(selected_model.predict(model_input)[0])
    probability = None

    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(model_input)[0]
        classes = list(selected_model.classes_)

        positive_class_index = None

        if 1 in classes:
            positive_class_index = classes.index(1)
        elif True in classes:
            positive_class_index = classes.index(True)
        else:
            for index, class_value in enumerate(classes):
                if depression_label(class_value) == "Depression":
                    positive_class_index = index
                    break

        if positive_class_index is not None:
            probability = float(probabilities[positive_class_index])

    return {
        "model": model_name,
        "prediction": prediction,
        "probability": probability,
    }


def get_confusion_matrix(model_name: str) -> np.ndarray | None:
    if not isinstance(confusion_matrices, dict):
        return None

    model_aliases = {
        "Logistic Regression": ["logistic regression", "logistic", "log"],
        "Decision Tree": ["decision tree", "tree"],
        "Random Forest": ["random forest", "forest"],
    }

    for key, matrix in confusion_matrices.items():
        key_text = str(key).strip().lower()

        if key_text == model_name.lower():
            return np.asarray(matrix)

        if any(alias in key_text for alias in model_aliases[model_name]):
            return np.asarray(matrix)

    return None


def get_result_row(model_name: str) -> pd.Series | None:
    matching_row = results[
        results["Model"].astype(str).str.strip().str.lower()
        == model_name.strip().lower()
    ]

    if matching_row.empty:
        return None

    return matching_row.iloc[0]


labeled_dataset = add_target_label(dataset)
overall_depression_rate = (
    labeled_dataset["Depression_Label"].eq("Depression").mean()
)


# =========================================================
# 5. SIDEBAR
# =========================================================
with st.sidebar:
    html_block(
        """
        <div class="brand-wrap">
            <div class="brand-mark">✦</div>
            <div class="brand-title">Behind the Numbers</div>
            <div class="brand-subtitle">Student Depression ML Explorer</div>
            <div style="margin-top:.55rem;display:inline-block;padding:.22rem .5rem;border-radius:999px;background:rgba(109,93,251,.28);font-size:.68rem;font-weight:800;letter-spacing:.05em;">UI VERSION 3.0</div>
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Student Story & EDA",
            "Data & Workflow",
            "Model Arena",
            "Prediction Studio",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    html_block(
        """
        <div class="side-note">
            Educational machine learning project.<br><br>
            The predictions are model outputs, not a medical diagnosis.
        </div>
        """
    )


# =========================================================
# 6. OVERVIEW
# =========================================================
if page == "Overview":
    html_block(
        """
        <section class="hero">
            <div class="hero-content">
                <div class="hero-eyebrow">✦ Student Depression Machine Learning Project</div>
                <h1>Behind the Numbers</h1>
                <p>
                    Explore how sleep, stress, study habits, social media use,
                    physical activity, and academic performance connect to
                    machine-learning predictions of student depression.
                </p>
                <div class="hero-tags">
                    <span class="hero-tag">100K student records</span>
                    <span class="hero-tag">Interactive EDA</span>
                    <span class="hero-tag">3 classification models</span>
                    <span class="hero-tag">What-if simulator</span>
                </div>
            </div>
        </section>
        """
    )

    metric_columns = st.columns(4)

    with metric_columns[0]:
        metric_card(
            "Students",
            f"{len(dataset):,}",
            "Records available for exploration",
            "#EFEDFF",
        )

    with metric_columns[1]:
        metric_card(
            "Original features",
            f"{max(dataset.shape[1] - 1, 0)}",
            "Before engineered features",
            "#EAF3FF",
        )

    with metric_columns[2]:
        metric_card(
            "Depression cases",
            f"{overall_depression_rate:.1%}",
            "Share of the full dataset",
            "#FFF0F3",
        )

    with metric_columns[3]:
        metric_card(
            "Models compared",
            f"{len(models)}",
            "Different learning approaches",
            "#EAFBF6",
        )

    st.write("")

    section_header(
        "Project story",
        "What the dashboard is trying to show",
        "The project is not only about a final prediction. It explains the data, the imbalance, and why the models can disagree.",
    )

    story_columns = st.columns(3)

    story_content = [
        (
            "01",
            "The hidden imbalance",
            "Most records belong to the No Depression class. A model can look accurate while still missing many real depression cases.",
        ),
        (
            "02",
            "Daily student routines",
            "Sleep, stress, study hours, social media, physical activity, and CGPA form the lifestyle profile used by the models.",
        ),
        (
            "03",
            "Models think differently",
            "Logistic Regression, Decision Tree, and Random Forest make different trade-offs between accuracy, recall, and F1 score.",
        ),
    ]

    for column, (icon, title, body) in zip(story_columns, story_content):
        with column:
            html_block(
                f"""
                <div class="story-card">
                    <div class="story-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{body}</p>
                </div>
                """
            )

    st.write("")

    section_header(
        "Workflow",
        "From raw records to an interactive prediction",
        "The application keeps the analysis steps visible instead of hiding everything behind one prediction button.",
    )

    workflow_items = [
        "Load & inspect",
        "Clean data",
        "Engineer features",
        "Encode & scale",
        "Train & evaluate",
        "Compare models",
        "Build prediction",
        "Test scenarios",
        "Explain limits",
        "Deploy app",
    ]

    timeline_html = ['<div class="timeline">']

    for index, item in enumerate(workflow_items, start=1):
        timeline_html.append(
            f"""
            <div class="timeline-item">
                <div class="timeline-number">{index:02d}</div>
                <div class="timeline-title">{item}</div>
                <div class="timeline-status">Completed</div>
            </div>
            """
        )

    timeline_html.append("</div>")
    html_block("".join(timeline_html))

    st.write("")

    html_block(
        """
        <div class="warning-card">
            <b>Important:</b> This is an educational machine-learning project.
            It must not be used as a medical diagnosis, screening tool, or clinical assessment.
        </div>
        """
    )


# =========================================================
# 7. STUDENT STORY & EDA
# =========================================================
elif page == "Student Story & EDA":
    section_header(
        "Interactive analysis",
        "Student Story & EDA",
        "Filter the dataset and watch every KPI and chart update. The filters are inside the page so the sidebar stays clean.",
    )

    genders = sorted(dataset["Gender"].dropna().astype(str).unique())
    departments = sorted(dataset["Department"].dropna().astype(str).unique())

    with st.expander("Open analysis filters", expanded=True):
        filter_row_1 = st.columns([1, 1.25, 1])
        filter_row_2 = st.columns([1, 1, 1])

        with filter_row_1[0]:
            selected_genders = st.multiselect(
                "Gender",
                genders,
                default=list(genders),
            )

        with filter_row_1[1]:
            selected_departments = st.multiselect(
                "Department",
                departments,
                default=list(departments),
            )

        with filter_row_1[2]:
            selected_classes = st.multiselect(
                "Depression class",
                ["No Depression", "Depression"],
                default=["No Depression", "Depression"],
            )

        with filter_row_2[0]:
            age_range = st.slider(
                "Age range",
                int(dataset["Age"].min()),
                int(dataset["Age"].max()),
                (
                    int(dataset["Age"].min()),
                    int(dataset["Age"].max()),
                ),
            )

        with filter_row_2[1]:
            stress_range = st.slider(
                "Stress level",
                int(dataset["Stress_Level"].min()),
                int(dataset["Stress_Level"].max()),
                (
                    int(dataset["Stress_Level"].min()),
                    int(dataset["Stress_Level"].max()),
                ),
            )

        with filter_row_2[2]:
            sleep_range = st.slider(
                "Sleep duration",
                float(dataset["Sleep_Duration"].min()),
                float(dataset["Sleep_Duration"].max()),
                (
                    float(dataset["Sleep_Duration"].min()),
                    float(dataset["Sleep_Duration"].max()),
                ),
                step=0.1,
            )

    filtered = labeled_dataset[
        labeled_dataset["Gender"].astype(str).isin(selected_genders)
        & labeled_dataset["Department"].astype(str).isin(selected_departments)
        & labeled_dataset["Age"].between(age_range[0], age_range[1])
        & labeled_dataset["Stress_Level"].between(
            stress_range[0], stress_range[1]
        )
        & labeled_dataset["Sleep_Duration"].between(
            sleep_range[0], sleep_range[1]
        )
        & labeled_dataset["Depression_Label"].isin(selected_classes)
    ].copy()

    if filtered.empty:
        st.warning("No students match the selected filters.")
        st.stop()

    selected_depression_rate = (
        filtered["Depression_Label"].eq("Depression").mean()
    )
    rate_difference = selected_depression_rate - overall_depression_rate

    metric_columns = st.columns(5)

    with metric_columns[0]:
        metric_card(
            "Selected students",
            f"{len(filtered):,}",
            "Records after filtering",
            "#EFEDFF",
        )

    with metric_columns[1]:
        metric_card(
            "Depression rate",
            f"{selected_depression_rate:.1%}",
            f"{rate_difference:+.1%} versus overall",
            "#FFF0F3",
        )

    with metric_columns[2]:
        metric_card(
            "Average sleep",
            f"{filtered['Sleep_Duration'].mean():.1f} h",
            "For the selected group",
            "#EAF3FF",
        )

    with metric_columns[3]:
        metric_card(
            "Average stress",
            f"{filtered['Stress_Level'].mean():.1f}",
            "For the selected group",
            "#FFF6E8",
        )

    with metric_columns[4]:
        metric_card(
            "Average CGPA",
            f"{filtered['CGPA'].mean():.2f}",
            "For the selected group",
            "#EAFBF6",
        )

    st.write("")

    if rate_difference > 0.02:
        message = (
            f"The selected group's depression rate is "
            f"{abs(rate_difference):.1%} higher than the full dataset."
        )
    elif rate_difference < -0.02:
        message = (
            f"The selected group's depression rate is "
            f"{abs(rate_difference):.1%} lower than the full dataset."
        )
    else:
        message = (
            "The selected group's depression rate is close to the overall dataset."
        )

    html_block(
        f"""
        <div class="insight-card">
            <span class="pill">Live insight</span><br><br>
            {message}
        </div>
        """
    )

    st.write("")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        class_counts = (
            filtered["Depression_Label"]
            .value_counts()
            .reindex(["No Depression", "Depression"], fill_value=0)
            .reset_index()
        )
        class_counts.columns = ["Class", "Count"]

        figure = px.pie(
            class_counts,
            names="Class",
            values="Count",
            hole=0.68,
            color="Class",
            color_discrete_map=CLASS_COLORS,
            title="Class balance in the selected group",
        )
        figure.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(line=dict(color="white", width=3)),
        )
        style_figure(figure, height=430)
        st.plotly_chart(figure, use_container_width=True)

    with chart_right:
        department_rates = (
            filtered.assign(
                Depression_Binary=filtered["Depression_Label"]
                .eq("Depression")
                .astype(int)
            )
            .groupby("Department", as_index=False)["Depression_Binary"]
            .mean()
            .sort_values("Depression_Binary", ascending=True)
        )

        figure = px.bar(
            department_rates,
            x="Depression_Binary",
            y="Department",
            orientation="h",
            text_auto=".1%",
            color="Depression_Binary",
            color_continuous_scale=["#EAFBF6", COLORS["purple"]],
            title="Depression rate by department",
        )
        figure.update_xaxes(title="Depression rate", tickformat=".0%")
        figure.update_layout(coloraxis_showscale=False)
        style_figure(figure, height=430, show_legend=False)
        st.plotly_chart(figure, use_container_width=True)

    st.write("")

    section_header(
        "Lifestyle patterns",
        "How daily routines differ",
        "Select a feature to compare its distribution between the two target classes.",
    )

    feature_options = {
        "Sleep Duration": "Sleep_Duration",
        "Study Hours": "Study_Hours",
        "Social Media Hours": "Social_Media_Hours",
        "Physical Activity": "Physical_Activity",
        "CGPA": "CGPA",
        "Stress Level": "Stress_Level",
        "Age": "Age",
    }

    selected_feature_label = st.selectbox(
        "Choose a feature",
        list(feature_options.keys()),
    )
    selected_feature = feature_options[selected_feature_label]

    figure = px.histogram(
        filtered,
        x=selected_feature,
        color="Depression_Label",
        barmode="overlay",
        opacity=0.72,
        nbins=28,
        color_discrete_map=CLASS_COLORS,
        title=f"{selected_feature_label} distribution by depression class",
    )
    style_figure(figure, height=430)
    st.plotly_chart(figure, use_container_width=True)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        figure = px.box(
            filtered,
            x="Depression_Label",
            y="Sleep_Duration",
            color="Depression_Label",
            color_discrete_map=CLASS_COLORS,
            points=False,
            title="Sleep duration by depression class",
        )
        style_figure(figure, height=420, show_legend=False)
        st.plotly_chart(figure, use_container_width=True)

    with chart_right:
        scatter_data = filtered.sample(
            n=min(3500, len(filtered)),
            random_state=42,
        )

        figure = px.scatter(
            scatter_data,
            x="Sleep_Duration",
            y="Stress_Level",
            color="Depression_Label",
            hover_data=["Age", "CGPA", "Department"],
            opacity=0.60,
            color_discrete_map=CLASS_COLORS,
            title="Sleep, stress, and depression",
        )
        figure.update_traces(marker=dict(size=7))
        style_figure(figure, height=420)
        st.plotly_chart(figure, use_container_width=True)

    st.write("")

    section_header(
        "Relationships",
        "No single feature explains everything",
        "The heatmap gives a quick numerical view of the relationships between numeric variables.",
    )

    correlation_data = dataset.drop(columns=["Student_ID"], errors="ignore").copy()
    correlation_data["Depression"] = correlation_data["Depression"].map(
        lambda value: 1
        if depression_label(value) == "Depression"
        else 0
    )

    correlation = correlation_data.corr(numeric_only=True)

    figure = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation heatmap",
    )
    figure.update_traces(
        hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>"
    )
    style_figure(figure, height=560, show_legend=False)
    st.plotly_chart(figure, use_container_width=True)

    depression_correlations = (
        correlation["Depression"]
        .drop("Depression")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

    strongest_feature = depression_correlations.index[0]
    strongest_value = depression_correlations.iloc[0]

    html_block(
        f"""
        <div class="insight-card">
            <span class="pill">Main EDA insight</span><br><br>
            <b>{strongest_feature}</b> has the strongest numeric correlation
            with Depression at <b>{strongest_value:.2f}</b>.
            This still does not mean that one feature alone explains the target,
            and correlation does not prove cause and effect.
        </div>
        """
    )


# =========================================================
# 8. DATA & WORKFLOW
# =========================================================
elif page == "Data & Workflow":
    section_header(
        "Transparency",
        "Data, Cleaning & Workflow",
        "A clear view of the dataset and the preprocessing steps used before model training.",
    )

    metric_columns = st.columns(4)

    with metric_columns[0]:
        metric_card(
            "Rows",
            f"{dataset.shape[0]:,}",
            "Total observations",
            "#EFEDFF",
        )

    with metric_columns[1]:
        metric_card(
            "Columns",
            f"{dataset.shape[1]}",
            "Including the target",
            "#EAF3FF",
        )

    with metric_columns[2]:
        metric_card(
            "Missing values",
            f"{int(dataset.isna().sum().sum()):,}",
            "Across the full dataset",
            "#FFF6E8",
        )

    with metric_columns[3]:
        metric_card(
            "Duplicated rows",
            f"{int(dataset.duplicated().sum()):,}",
            "Exact duplicated records",
            "#EAFBF6",
        )

    st.write("")

    tabs = st.tabs(["Dataset sample", "Column guide", "Cleaning decisions"])

    with tabs[0]:
        st.dataframe(
            dataset.head(15),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        column_guide = pd.DataFrame(
            {
                "Column": dataset.columns,
                "Data type": dataset.dtypes.astype(str).values,
                "Role": [
                    "Identifier"
                    if column == "Student_ID"
                    else "Target"
                    if column == "Depression"
                    else "Feature"
                    for column in dataset.columns
                ],
            }
        )

        st.dataframe(
            column_guide,
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        cleaning_columns = st.columns(3)

        cleaning_items = [
            (
                "Missing values",
                "No missing values were found, so no imputation was required.",
            ),
            (
                "Duplicated rows",
                "No exact duplicated rows were found, so no rows were removed for duplication.",
            ),
            (
                "Outliers",
                "High or low values were kept because they can still represent valid student behavior.",
            ),
        ]

        for column, (title, body) in zip(cleaning_columns, cleaning_items):
            with column:
                html_block(
                    f"""
                    <div class="clean-card">
                        <div class="story-icon">✓</div>
                        <h3>{title}</h3>
                        <p>{body}</p>
                    </div>
                    """
                )

    st.write("")

    section_header(
        "Preprocessing",
        "The model-ready pipeline",
        "Each step has a specific role. The pipeline does not modify the original CSV file.",
    )

    pipeline = pd.DataFrame(
        {
            "Step": [
                "Remove Student_ID",
                "Create Low_Sleep",
                "Create High_Stress",
                "Encode Gender and Department",
                "Split train and test data",
                "Scale Logistic Regression data",
                "Apply PCA for comparison",
                "Train three classifiers",
            ],
            "Why it was used": [
                "The ID identifies rows but does not describe student behavior.",
                "Marks sleep duration below 6 hours.",
                "Marks stress level 7 or higher.",
                "Changes text categories into numeric model inputs.",
                "Keeps unseen records for final evaluation.",
                "Handles the different numerical feature ranges.",
                "Tests whether fewer components keep enough information.",
                "Compares linear and tree-based learning approaches.",
            ],
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True,
    )

    st.write("")

    section_header(
        "PCA experiment",
        "Reduction was tested, not blindly accepted",
        "PCA kept most of the variance, but the final Logistic Regression performance became slightly lower.",
    )

    pca_columns = st.columns(3)

    with pca_columns[0]:
        metric_card(
            "Features before PCA",
            "16",
            "Encoded and engineered inputs",
            "#EFEDFF",
        )

    with pca_columns[1]:
        metric_card(
            "Components after PCA",
            "12",
            "Reduced dimensions",
            "#EAF3FF",
        )

    with pca_columns[2]:
        metric_card(
            "Variance kept",
            "95.6%",
            "Information retained",
            "#EAFBF6",
        )

    st.write("")

    html_block(
        """
        <div class="insight-card">
            <span class="pill">Decision</span><br><br>
            PCA was not used in the final prediction workflow because the simpler
            reduction did not improve the final Logistic Regression result.
        </div>
        """
    )


# =========================================================
# 9. MODEL ARENA
# =========================================================
elif page == "Model Arena":
    section_header(
        "Model comparison",
        "Model Arena",
        "Accuracy alone can be misleading with imbalanced classes. Compare accuracy, precision, recall, and F1 score together.",
    )

    display_results = results.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for column in metric_columns:
        display_results[column] = pd.to_numeric(
            display_results[column],
            errors="coerce",
        )

    best_accuracy = display_results.loc[
        display_results["Accuracy"].idxmax(), "Model"
    ]
    best_recall = display_results.loc[
        display_results["Recall"].idxmax(), "Model"
    ]
    best_f1 = display_results.loc[
        display_results["F1 Score"].idxmax(), "Model"
    ]

    highlight_columns = st.columns(3)

    with highlight_columns[0]:
        metric_card(
            "Highest accuracy",
            str(best_accuracy),
            "Best overall correct rate",
            "#EAF3FF",
        )

    with highlight_columns[1]:
        metric_card(
            "Highest recall",
            str(best_recall),
            "Finds the largest share of positive cases",
            "#FFF0F3",
        )

    with highlight_columns[2]:
        metric_card(
            "Highest F1 score",
            str(best_f1),
            "Best precision–recall balance",
            "#EAFBF6",
        )

    st.write("")

    long_results = display_results.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    figure = px.bar(
        long_results,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".1%",
        color_discrete_sequence=[
            COLORS["purple"],
            COLORS["teal"],
            COLORS["orange"],
            COLORS["rose"],
        ],
        title="Model performance comparison",
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    figure.update_traces(textposition="outside", cliponaxis=False)
    style_figure(figure, height=500)
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("See the exact metric table"):
        st.dataframe(
            display_results.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.write("")

    section_header(
        "Behavior",
        "How each model behaves",
        "The models learn patterns differently, so the most accurate model is not automatically the most useful one.",
    )

    explanations = {
        "Logistic Regression": (
            "It has lower accuracy, but it detects the largest share of real "
            "depression cases. In this project, it gives the strongest recall and F1 balance."
        ),
        "Decision Tree": (
            "It reaches higher accuracy than Logistic Regression, but it misses "
            "more real depression cases and can be sensitive to the exact training split."
        ),
        "Random Forest": (
            "It reaches very high accuracy, but its recall is extremely low. "
            "That suggests it often defaults to the majority No Depression class."
        ),
    }

    model_columns = st.columns(3)

    for column, model_name in zip(model_columns, MODEL_FILES):
        result_row = get_result_row(model_name)

        if result_row is None:
            continue

        with column:
            html_block(
                f"""
                <div class="model-card" style="--model-color:{MODEL_COLORS[model_name]};">
                    <div class="model-name">{model_name}</div>
                    <div class="model-metrics">
                        <div class="model-metric">
                            <b>{result_row['Accuracy']:.1%}</b>
                            <span>Accuracy</span>
                        </div>
                        <div class="model-metric">
                            <b>{result_row['Recall']:.1%}</b>
                            <span>Recall</span>
                        </div>
                        <div class="model-metric">
                            <b>{result_row['F1 Score']:.1%}</b>
                            <span>F1 score</span>
                        </div>
                    </div>
                    <div class="model-copy">{explanations[model_name]}</div>
                </div>
                """
            )

    st.write("")

    section_header(
        "Error analysis",
        "Confusion matrix explorer",
        "Choose a model and inspect the exact types of correct and incorrect predictions.",
    )

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    matrix = get_confusion_matrix(selected_model)

    if matrix is None:
        st.warning("No confusion matrix was found for this model.")
    elif np.asarray(matrix).shape != (2, 2):
        st.warning(
            f"The saved confusion matrix has shape {np.asarray(matrix).shape}, "
            "but this page expects a 2×2 binary matrix."
        )
    else:
        matrix = np.asarray(matrix)

        figure = px.imshow(
            matrix,
            text_auto="d",
            x=["No Depression", "Depression"],
            y=["No Depression", "Depression"],
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count",
            },
            color_continuous_scale=[
                [0, "#F3F1FF"],
                [1, COLORS["purple_dark"]],
            ],
            title=f"{selected_model} confusion matrix",
        )
        figure.update_traces(
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
        )
        style_figure(figure, height=520, show_legend=False)
        st.plotly_chart(figure, use_container_width=True)

        true_negative, false_positive, false_negative, true_positive = (
            matrix.ravel()
        )

        value_columns = st.columns(4)

        values = [
            ("True negatives", f"{true_negative:,}", "#EAFBF6"),
            ("False positives", f"{false_positive:,}", "#FFF6E8"),
            ("False negatives", f"{false_negative:,}", "#FFF0F3"),
            ("True positives", f"{true_positive:,}", "#EFEDFF"),
        ]

        for column, (label, value, accent) in zip(value_columns, values):
            with column:
                metric_card(label, value, "Count in the test set", accent)

        st.write("")

        html_block(
            """
            <div class="warning-card">
                <b>Why false negatives matter:</b> they are real Depression cases
                predicted as No Depression. This is why recall must be checked instead
                of trusting accuracy alone.
            </div>
            """
        )


# =========================================================
# 10. PREDICTION STUDIO
# =========================================================
elif page == "Prediction Studio":
    section_header(
        "Interactive prediction",
        "Prediction Studio",
        "Choose one trained model, build a student profile, and view that model's prediction.",
    )

    html_block(
        """
        <div class="warning-card">
            This page shows model output only. It is not a diagnosis and must not be
            used for medical or clinical decisions.
        </div>
        """
    )

    st.write("")

    gender_values = sorted(dataset["Gender"].dropna().astype(str).unique())
    department_values = sorted(
        dataset["Department"].dropna().astype(str).unique()
    )

    model_names = list(models.keys())

    # Use the highest F1 model as the initial selection when the results file allows it.
    default_model_index = 0
    try:
        result_names = results["Model"].astype(str).str.strip().str.lower()
        best_f1_row = pd.to_numeric(results["F1 Score"], errors="coerce").idxmax()
        best_f1_name = str(results.loc[best_f1_row, "Model"]).strip().lower()
        for index, model_name in enumerate(model_names):
            if model_name.lower() == best_f1_name:
                default_model_index = index
                break
    except Exception:
        default_model_index = 0

    with st.form("prediction_form"):
        st.markdown("### Choose the prediction model")

        selected_prediction_model = st.selectbox(
            "Model",
            model_names,
            index=default_model_index,
            help=(
                "The selected model alone will produce the prediction below. "
                "Use Model Arena to compare test accuracy, recall, F1 score, and confusion matrices."
            ),
        )

        st.caption(
            "A model is not automatically correct for one new student. Its reliability is judged using its test-set metrics."
        )

        st.markdown("### Build a student profile")

        input_columns = st.columns(3)

        with input_columns[0]:
            age = st.slider(
                "Age",
                int(dataset["Age"].min()),
                int(dataset["Age"].max()),
                int(dataset["Age"].median()),
            )

            gender = st.selectbox(
                "Gender",
                gender_values,
            )

            department = st.selectbox(
                "Department",
                department_values,
            )

        with input_columns[1]:
            cgpa = st.number_input(
                "CGPA",
                min_value=float(dataset["CGPA"].min()),
                max_value=float(dataset["CGPA"].max()),
                value=float(dataset["CGPA"].median()),
                step=0.1,
            )

            sleep_duration = st.slider(
                "Sleep duration",
                float(dataset["Sleep_Duration"].min()),
                float(dataset["Sleep_Duration"].max()),
                float(dataset["Sleep_Duration"].median()),
                step=0.1,
            )

            study_hours = st.slider(
                "Study hours",
                float(dataset["Study_Hours"].min()),
                float(dataset["Study_Hours"].max()),
                float(dataset["Study_Hours"].median()),
                step=0.1,
            )

        with input_columns[2]:
            social_media_hours = st.slider(
                "Social media hours",
                float(dataset["Social_Media_Hours"].min()),
                float(dataset["Social_Media_Hours"].max()),
                float(dataset["Social_Media_Hours"].median()),
                step=0.1,
            )

            physical_activity = st.slider(
                "Physical activity",
                int(dataset["Physical_Activity"].min()),
                int(dataset["Physical_Activity"].max()),
                int(dataset["Physical_Activity"].median()),
            )

            stress_level = st.slider(
                "Stress level",
                int(dataset["Stress_Level"].min()),
                int(dataset["Stress_Level"].max()),
                int(dataset["Stress_Level"].median()),
            )

        submitted = st.form_submit_button(
            "Run prediction",
            use_container_width=True,
        )

    if submitted:
        st.session_state["student_profile"] = {
            "Age": age,
            "Gender": gender,
            "Department": department,
            "CGPA": cgpa,
            "Sleep_Duration": sleep_duration,
            "Study_Hours": study_hours,
            "Social_Media_Hours": social_media_hours,
            "Physical_Activity": physical_activity,
            "Stress_Level": stress_level,
        }
        st.session_state["selected_prediction_model"] = selected_prediction_model

    if "student_profile" not in st.session_state:
        html_block(
            """
            <div class="insight-card">
                <span class="pill">Ready</span><br><br>
                Choose a model, complete the profile, and click <b>Run prediction</b>.
            </div>
            """
        )
    else:
        student_profile = st.session_state["student_profile"]
        chosen_model = st.session_state.get(
            "selected_prediction_model",
            model_names[default_model_index],
        )
        prepared_input = prepare_model_input(student_profile)
        selected_result = predict_with_model(chosen_model, prepared_input)

        st.write("")

        section_header(
            "Result",
            "Selected model prediction",
            f"This output was produced only by {chosen_model}.",
        )

        predicted_text = (
            "Higher predicted risk"
            if selected_result["prediction"] == 1
            else "Lower predicted risk"
        )
        probability_text = (
            f"{selected_result['probability']:.1%}"
            if selected_result["probability"] is not None
            else "N/A"
        )

        if selected_result["prediction"] == 1:
            prediction_background = "#FFF4F6"
            prediction_line = "#F3C3CC"
        else:
            prediction_background = "#F1FBF8"
            prediction_line = "#BFE9DC"

        result_columns = st.columns([1.15, 1.85])

        with result_columns[0]:
            html_block(
                f"""
                <div class="prediction-card"
                     style="--prediction-bg:{prediction_background};--prediction-line:{prediction_line};min-height:245px;">
                    <div class="prediction-model">{chosen_model}</div>
                    <div class="prediction-state">{predicted_text}</div>
                    <div class="prediction-probability">{probability_text}</div>
                    <div class="prediction-note">Predicted Depression probability</div>
                </div>
                """
            )

        with result_columns[1]:
            matching_metrics = results[
                results["Model"].astype(str).str.strip().str.lower()
                == chosen_model.strip().lower()
            ]

            if not matching_metrics.empty:
                metric_row = matching_metrics.iloc[0]
                model_metric_columns = st.columns(3)

                metric_values = [
                    (
                        "Test accuracy",
                        f"{float(metric_row['Accuracy']):.1%}",
                        "All correct test predictions",
                        "#EFEDFF",
                    ),
                    (
                        "Test recall",
                        f"{float(metric_row['Recall']):.1%}",
                        "Depression cases detected",
                        "#EAF3FF",
                    ),
                    (
                        "Test F1 score",
                        f"{float(metric_row['F1 Score']):.1%}",
                        "Balance of precision and recall",
                        "#EAFBF6",
                    ),
                ]

                for column, (label, value, note, accent) in zip(
                    model_metric_columns,
                    metric_values,
                ):
                    with column:
                        metric_card(label, value, note, accent)

                html_block(
                    """
                    <div class="insight-card">
                        <span class="pill">How to judge it</span><br><br>
                        The test metrics tell you how this model performed on unseen labeled data.
                        They do not prove that its prediction for one individual student is correct.
                    </div>
                    """
                )
            else:
                html_block(
                    """
                    <div class="insight-card">
                        The selected model produced a prediction, but its evaluation row was not found in model_results.csv.
                    </div>
                    """
                )

        st.write("")

        with st.expander("Compare the same profile with the other models"):
            comparison_results = [
                predict_with_model(model_name, prepared_input)
                for model_name in model_names
            ]

            comparison_rows = []
            for result in comparison_results:
                comparison_rows.append(
                    {
                        "Model": result["model"],
                        "Prediction": (
                            "Higher predicted risk"
                            if result["prediction"] == 1
                            else "Lower predicted risk"
                        ),
                        "Depression probability": result["probability"],
                    }
                )

            comparison_table = pd.DataFrame(comparison_rows)
            st.dataframe(
                comparison_table.style.format(
                    {"Depression probability": "{:.1%}"},
                    na_rep="N/A",
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.write("")

        section_header(
            "Scenario testing",
            "What-if simulator",
            f"Change sleep and stress, then see how {chosen_model} changes its output.",
        )

        scenario_columns = st.columns(2)

        with scenario_columns[0]:
            scenario_sleep = st.slider(
                "Scenario sleep duration",
                float(dataset["Sleep_Duration"].min()),
                float(dataset["Sleep_Duration"].max()),
                float(student_profile["Sleep_Duration"]),
                step=0.1,
                key="scenario_sleep",
            )

        with scenario_columns[1]:
            scenario_stress = st.slider(
                "Scenario stress level",
                int(dataset["Stress_Level"].min()),
                int(dataset["Stress_Level"].max()),
                int(student_profile["Stress_Level"]),
                key="scenario_stress",
            )

        scenario_profile = student_profile.copy()
        scenario_profile["Sleep_Duration"] = scenario_sleep
        scenario_profile["Stress_Level"] = scenario_stress

        original_result = predict_with_model(
            chosen_model,
            prepare_model_input(student_profile),
        )
        scenario_result = predict_with_model(
            chosen_model,
            prepare_model_input(scenario_profile),
        )

        if (
            original_result["probability"] is not None
            and scenario_result["probability"] is not None
        ):
            probability_change = (
                scenario_result["probability"] - original_result["probability"]
            )

            scenario_metrics = st.columns(3)

            with scenario_metrics[0]:
                metric_card(
                    "Original probability",
                    f"{original_result['probability']:.1%}",
                    f"Sleep {student_profile['Sleep_Duration']:.1f} h · Stress {student_profile['Stress_Level']}",
                    "#EFEDFF",
                )

            with scenario_metrics[1]:
                metric_card(
                    "Scenario probability",
                    f"{scenario_result['probability']:.1%}",
                    f"Sleep {scenario_sleep:.1f} h · Stress {scenario_stress}",
                    "#EAF3FF",
                )

            with scenario_metrics[2]:
                accent = "#FFF0F3" if probability_change > 0 else "#EAFBF6"
                metric_card(
                    "Model output change",
                    f"{probability_change:+.1%}",
                    "Scenario minus original",
                    accent,
                )

        st.write("")

        html_block(
            f"""
            <div class="warning-card">
                The scenario uses <b>{chosen_model}</b>. It shows how that model's output changes
                after changing the inputs; it does not prove cause and effect.
            </div>
            """
        )

        with st.expander("See the exact encoded model input"):
            st.dataframe(
                prepared_input,
                use_container_width=True,
                hide_index=True,
            )


html_block(
    """
    <div class="footer-note">
        Behind the Numbers · Educational Machine Learning Dashboard
    </div>
    """
)
