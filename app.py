from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE SETTINGS
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

MODEL_COLORS = {
    "Logistic Regression": "#6C63FF",
    "Decision Tree": "#24A89A",
    "Random Forest": "#F4A261",
}

CLASS_COLORS = {
    "No Depression": "#24A89A",
    "Depression": "#EF6A78",
}


# =========================================================
# 2. CUSTOM DESIGN
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2.4rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at 88% 12%, rgba(160,150,255,.30), transparent 30%),
                linear-gradient(135deg, #171B3A 0%, #2B3169 60%, #4A4386 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 20px 50px rgba(23, 27, 58, .18);
        }

        .hero h1 {
            margin: 0 0 .7rem 0;
            font-size: 3.2rem;
            line-height: 1;
            letter-spacing: -0.04em;
        }

        .hero p {
            max-width: 850px;
            margin: 0;
            color: rgba(255,255,255,.84);
            font-size: 1.08rem;
        }

        .eyebrow {
            color: #C6C2FF;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .17em;
            text-transform: uppercase;
            margin-bottom: .8rem;
        }

        .card {
            min-height: 150px;
            padding: 1.15rem 1.2rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #E7E9F2;
            box-shadow: 0 10px 28px rgba(30, 35, 75, .06);
        }

        .card strong {
            display: block;
            margin-bottom: .45rem;
            color: #252A53;
        }

        .soft-card {
            padding: 1.1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(108,99,255,.07), white);
            border: 1px solid rgba(108,99,255,.18);
        }

        .risk-high {
            padding: 1.1rem;
            border-radius: 18px;
            background: #FFF0F2;
            border: 1px solid #FFC8D0;
            min-height: 145px;
        }

        .risk-low {
            padding: 1.1rem;
            border-radius: 18px;
            background: #ECFBF7;
            border: 1px solid #B7EEE0;
            min-height: 145px;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #E7E9F2;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(30, 35, 75, .05);
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid #ECEEF6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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

    missing_files = [
        file.name
        for file in required_files
        if not file.exists()
    ]

    if missing_files:
        st.error(
            "Missing files: " + ", ".join(missing_files)
        )
        st.stop()


check_files()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_results() -> pd.DataFrame:
    model_results = pd.read_csv(RESULTS_FILE)

    rename_map = {}

    for column in model_results.columns:
        simple_name = (
            column.strip()
            .lower()
            .replace("_", " ")
        )

        if simple_name == "model":
            rename_map[column] = "Model"

        elif simple_name == "accuracy":
            rename_map[column] = "Accuracy"

        elif simple_name == "precision":
            rename_map[column] = "Precision"

        elif simple_name == "recall":
            rename_map[column] = "Recall"

        elif simple_name in {
            "f1",
            "f1 score",
            "f1score",
        }:
            rename_map[column] = "F1 Score"

    model_results = model_results.rename(
        columns=rename_map
    )

    needed_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    return model_results[needed_columns]


@st.cache_resource
def load_artifacts():
    loaded_models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_FILES.items()
    }

    loaded_scaler = joblib.load(SCALER_FILE)

    loaded_feature_columns = list(
        joblib.load(FEATURES_FILE)
    )

    loaded_confusion_matrices = joblib.load(
        MATRICES_FILE
    )

    return (
        loaded_models,
        loaded_scaler,
        loaded_feature_columns,
        loaded_confusion_matrices,
    )


dataset = load_dataset()
results = load_results()

(
    models,
    scaler,
    feature_columns,
    confusion_matrices,
) = load_artifacts()


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def depression_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return (
            "Depression"
            if bool(value)
            else "No Depression"
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return (
            "Depression"
            if int(value) == 1
            else "No Depression"
        )

    text = str(value).strip().lower()

    positive_values = {
        "1",
        "true",
        "yes",
        "depression",
        "depressed",
    }

    return (
        "Depression"
        if text in positive_values
        else "No Depression"
    )


def add_target_label(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    new_dataframe = dataframe.copy()

    new_dataframe["Depression_Label"] = (
        new_dataframe["Depression"]
        .map(depression_label)
    )

    return new_dataframe


def prepare_model_input(
    values: dict[str, Any],
) -> pd.DataFrame:
    row = {
        column: 0
        for column in feature_columns
    }

    numerical_values = {
        "Age": values["Age"],
        "CGPA": values["CGPA"],
        "Sleep_Duration": values[
            "Sleep_Duration"
        ],
        "Study_Hours": values["Study_Hours"],
        "Social_Media_Hours": values[
            "Social_Media_Hours"
        ],
        "Physical_Activity": values[
            "Physical_Activity"
        ],
        "Stress_Level": values["Stress_Level"],

        # Feature Engineering
        "Low_Sleep": int(
            values["Sleep_Duration"] < 6
        ),

        "High_Stress": int(
            values["Stress_Level"] >= 7
        ),
    }

    for column, value in numerical_values.items():
        if column in row:
            row[column] = value

    selected_gender = (
        str(values["Gender"])
        .strip()
        .lower()
    )

    selected_department = (
        str(values["Department"])
        .strip()
        .lower()
    )

    for column in feature_columns:
        lower_column = column.lower()

        if lower_column.startswith("gender_"):
            category = (
                column.split("_", 1)[1]
                .strip()
                .lower()
            )

            row[column] = int(
                category == selected_gender
            )

        elif lower_column.startswith(
            "department_"
        ):
            category = (
                column.split("_", 1)[1]
                .strip()
                .lower()
            )

            row[column] = int(
                category == selected_department
            )

    return pd.DataFrame(
        [row],
        columns=feature_columns,
    )


def predict_with_model(
    model_name: str,
    prepared_input: pd.DataFrame,
) -> dict[str, Any]:
    selected_model = models[model_name]

    # Logistic Regression needs scaled data.
    if model_name == "Logistic Regression":
        model_input = scaler.transform(
            prepared_input
        )

    # Trees use the original numerical values.
    else:
        model_input = prepared_input

    prediction = int(
        selected_model.predict(
            model_input
        )[0]
    )

    probability = None

    if hasattr(
        selected_model,
        "predict_proba",
    ):
        probabilities = (
            selected_model.predict_proba(
                model_input
            )[0]
        )

        classes = list(
            selected_model.classes_
        )

        if 1 in classes:
            probability = float(
                probabilities[
                    classes.index(1)
                ]
            )

        elif True in classes:
            probability = float(
                probabilities[
                    classes.index(True)
                ]
            )

    return {
        "model": model_name,
        "prediction": prediction,
        "probability": probability,
    }


def get_confusion_matrix(
    model_name: str,
) -> np.ndarray | None:
    if not isinstance(
        confusion_matrices,
        dict,
    ):
        return None

    model_aliases = {
        "Logistic Regression": [
            "logistic",
            "log",
        ],

        "Decision Tree": [
            "decision tree",
            "tree",
        ],

        "Random Forest": [
            "random forest",
            "forest",
        ],
    }

    for key, matrix in confusion_matrices.items():
        key_text = str(key).strip().lower()

        if key_text == model_name.lower():
            return np.asarray(matrix)

        if any(
            alias in key_text
            for alias in model_aliases[
                model_name
            ]
        ):
            return np.asarray(matrix)

    return None


labeled_dataset = add_target_label(dataset)

overall_depression_rate = (
    labeled_dataset["Depression_Label"]
    .eq("Depression")
    .mean()
)


# =========================================================
# 5. SIDEBAR
# =========================================================
st.sidebar.markdown(
    "## Behind the Numbers"
)

st.sidebar.caption(
    "Student Depression ML Explorer"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Student Story & EDA",
        "Data & Workflow",
        "Model Arena",
        "Prediction Studio",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Educational machine learning project. "
    "Predictions are not a medical diagnosis."
)


# =========================================================
# 6. HOME PAGE
# =========================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                Student Depression Machine Learning Project
            </div>

            <h1>Behind the Numbers</h1>

            <p>
                Explore how sleep, stress, study habits,
                social media use, physical activity, and
                academic performance connect to the
                model's prediction of student depression.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Students",
        f"{len(dataset):,}",
    )

    column2.metric(
        "Original columns",
        dataset.shape[1],
    )

    column3.metric(
        "Depression cases",
        f"{overall_depression_rate:.1%}",
    )

    column4.metric(
        "Models compared",
        len(models),
    )

    st.subheader(
        "The story of this project"
    )

    story1, story2, story3 = st.columns(3)

    with story1:
        st.markdown(
            """
            <div class="card">
                <strong>
                    1. The hidden imbalance
                </strong>

                Most students belong to the
                No Depression class. A model can
                look accurate while still missing
                many real depression cases.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with story2:
        st.markdown(
            """
            <div class="card">
                <strong>
                    2. Daily student routines
                </strong>

                Sleep, study hours, social media,
                physical activity, stress, and CGPA
                form the student's lifestyle profile.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with story3:
        st.markdown(
            """
            <div class="card">
                <strong>
                    3. Models do not think the same
                </strong>

                Logistic Regression, Decision Tree,
                and Random Forest make different
                trade-offs between accuracy and recall.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("The full workflow")

    workflow = pd.DataFrame(
        {
            "Step": [
                "Load data",
                "Explore",
                "Clean",
                "Feature engineering",
                "Encode",
                "Scale",
                "PCA",
                "Train 3 models",
                "Evaluate",
                "Predict",
            ],

            "Status": [
                "Completed"
            ] * 10,
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "This application is an educational "
        "project and must not be used as a "
        "medical diagnosis or clinical assessment."
    )


# =========================================================
# 7. STUDENT STORY & EDA
# =========================================================
elif page == "Student Story & EDA":
    st.title(
        "Student Story & Interactive EDA"
    )

    st.caption(
        "Use the filters to explore student "
        "groups. The KPIs and charts update "
        "automatically."
    )

    with st.sidebar:
        st.divider()
        st.markdown("### EDA filters")

        genders = sorted(
            dataset["Gender"]
            .dropna()
            .astype(str)
            .unique()
        )

        departments = sorted(
            dataset["Department"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_genders = st.multiselect(
            "Gender",
            genders,
            default=list(genders),
        )

        selected_departments = (
            st.multiselect(
                "Department",
                departments,
                default=list(departments),
            )
        )

        age_range = st.slider(
            "Age range",

            int(dataset["Age"].min()),

            int(dataset["Age"].max()),

            (
                int(dataset["Age"].min()),
                int(dataset["Age"].max()),
            ),
        )

        stress_range = st.slider(
            "Stress level",

            int(
                dataset["Stress_Level"].min()
            ),

            int(
                dataset["Stress_Level"].max()
            ),

            (
                int(
                    dataset[
                        "Stress_Level"
                    ].min()
                ),

                int(
                    dataset[
                        "Stress_Level"
                    ].max()
                ),
            ),
        )

        sleep_range = st.slider(
            "Sleep duration",

            float(
                dataset[
                    "Sleep_Duration"
                ].min()
            ),

            float(
                dataset[
                    "Sleep_Duration"
                ].max()
            ),

            (
                float(
                    dataset[
                        "Sleep_Duration"
                    ].min()
                ),

                float(
                    dataset[
                        "Sleep_Duration"
                    ].max()
                ),
            ),

            step=0.1,
        )

        selected_classes = st.multiselect(
            "Depression class",

            [
                "No Depression",
                "Depression",
            ],

            default=[
                "No Depression",
                "Depression",
            ],
        )

    filtered = labeled_dataset[
        labeled_dataset["Gender"]
        .astype(str)
        .isin(selected_genders)

        & labeled_dataset["Department"]
        .astype(str)
        .isin(selected_departments)

        & labeled_dataset["Age"].between(
            age_range[0],
            age_range[1],
        )

        & labeled_dataset[
            "Stress_Level"
        ].between(
            stress_range[0],
            stress_range[1],
        )

        & labeled_dataset[
            "Sleep_Duration"
        ].between(
            sleep_range[0],
            sleep_range[1],
        )

        & labeled_dataset[
            "Depression_Label"
        ].isin(selected_classes)
    ].copy()

    if filtered.empty:
        st.warning(
            "No students match these filters."
        )
        st.stop()

    selected_depression_rate = (
        filtered["Depression_Label"]
        .eq("Depression")
        .mean()
    )

    rate_difference = (
        selected_depression_rate
        - overall_depression_rate
    )

    metric1, metric2, metric3, metric4, metric5 = (
        st.columns(5)
    )

    metric1.metric(
        "Selected students",
        f"{len(filtered):,}",
    )

    metric2.metric(
        "Depression rate",
        f"{selected_depression_rate:.1%}",
        delta=(
            f"{rate_difference:+.1%} "
            "vs overall"
        ),
    )

    metric3.metric(
        "Average sleep",
        (
            f"{filtered['Sleep_Duration'].mean():.1f} h"
        ),
    )

    metric4.metric(
        "Average stress",
        f"{filtered['Stress_Level'].mean():.1f}",
    )

    metric5.metric(
        "Average CGPA",
        f"{filtered['CGPA'].mean():.2f}",
    )

    if rate_difference > 0.02:
        st.info(
            "The selected group has a depression "
            f"rate {abs(rate_difference):.1%} "
            "higher than the full dataset."
        )

    elif rate_difference < -0.02:
        st.success(
            "The selected group has a depression "
            f"rate {abs(rate_difference):.1%} "
            "lower than the full dataset."
        )

    else:
        st.info(
            "The selected group's depression "
            "rate is close to the overall dataset."
        )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        class_counts = (
            filtered["Depression_Label"]
            .value_counts()
            .reindex(
                [
                    "No Depression",
                    "Depression",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        class_counts.columns = [
            "Class",
            "Count",
        ]

        figure = px.pie(
            class_counts,
            names="Class",
            values="Count",
            hole=0.62,
            color="Class",
            color_discrete_map=CLASS_COLORS,
            title=(
                "Chapter 1 — The class imbalance"
            ),
        )

        figure.update_layout(
            template="plotly_white",
            legend_title_text="",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    with right_chart:
        department_rates = (
            filtered.assign(
                Depression_Binary=(
                    filtered[
                        "Depression_Label"
                    ]
                    .eq("Depression")
                    .astype(int)
                )
            )

            .groupby(
                "Department",
                as_index=False,
            )["Depression_Binary"]

            .mean()

            .sort_values(
                "Depression_Binary",
                ascending=False,
            )
        )

        figure = px.bar(
            department_rates,
            x="Department",
            y="Depression_Binary",
            text_auto=".1%",
            color="Depression_Binary",

            color_continuous_scale=[
                "#D8F5EE",
                "#6C63FF",
            ],

            title=(
                "Depression rate by department"
            ),
        )

        figure.update_yaxes(
            title="Depression rate",
            tickformat=".0%",
        )

        figure.update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    st.subheader(
        "Chapter 2 — Daily routines"
    )

    feature_options = {
        "Sleep Duration": "Sleep_Duration",
        "Study Hours": "Study_Hours",
        "Social Media Hours": (
            "Social_Media_Hours"
        ),
        "Physical Activity": (
            "Physical_Activity"
        ),
        "CGPA": "CGPA",
        "Stress Level": "Stress_Level",
        "Age": "Age",
    }

    selected_feature_label = st.selectbox(
        "Choose a feature",
        list(feature_options.keys()),
    )

    selected_feature = feature_options[
        selected_feature_label
    ]

    figure = px.histogram(
        filtered,
        x=selected_feature,
        color="Depression_Label",
        barmode="overlay",
        opacity=0.72,
        nbins=25,
        color_discrete_map=CLASS_COLORS,

        title=(
            f"{selected_feature_label} "
            "distribution by depression class"
        ),
    )

    figure.update_layout(
        template="plotly_white",
        legend_title_text="",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    sleep_chart, scatter_chart = (
        st.columns(2)
    )

    with sleep_chart:
        figure = px.box(
            filtered,
            x="Depression_Label",
            y="Sleep_Duration",
            color="Depression_Label",
            color_discrete_map=CLASS_COLORS,
            points=False,

            title=(
                "Sleep duration by "
                "depression class"
            ),
        )

        figure.update_layout(
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    with scatter_chart:
        scatter_data = filtered.sample(
            n=min(5000, len(filtered)),
            random_state=42,
        )

        figure = px.scatter(
            scatter_data,
            x="Sleep_Duration",
            y="Stress_Level",
            color="Depression_Label",

            hover_data=[
                "Age",
                "CGPA",
                "Department",
            ],

            opacity=0.65,

            color_discrete_map=CLASS_COLORS,

            title=(
                "Sleep, stress, and depression"
            ),
        )

        figure.update_layout(
            template="plotly_white",
            legend_title_text="",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    st.subheader(
        "Chapter 3 — No one feature "
        "explains everything"
    )

    correlation_data = dataset.drop(
        columns=["Student_ID"],
        errors="ignore",
    ).copy()

    correlation_data["Depression"] = (
        correlation_data["Depression"]
        .map(
            lambda value: (
                1
                if depression_label(value)
                == "Depression"
                else 0
            )
        )
    )

    correlation = correlation_data.corr(
        numeric_only=True
    )

    figure = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation heatmap",
    )

    figure.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    depression_correlations = (
        correlation["Depression"]
        .drop("Depression")
        .sort_values(
            key=lambda values: (
                values.abs()
            ),
            ascending=False,
        )
    )

    strongest_feature = (
        depression_correlations.index[0]
    )

    strongest_value = (
        depression_correlations.iloc[0]
    )

    st.markdown(
        f"""
        <div class="soft-card">
            <strong>Main EDA insight</strong><br>

            <b>{strongest_feature}</b> has the
            strongest numerical correlation with
            Depression at <b>{strongest_value:.2f}</b>,
            but the relationship is still weak.

            This means the target cannot be
            explained using only one feature.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. DATA & WORKFLOW
# =========================================================
elif page == "Data & Workflow":
    st.title("Data, Cleaning & Workflow")

    st.caption(
        "A transparent view of how the raw "
        "data became model-ready."
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Rows",
        f"{dataset.shape[0]:,}",
    )

    column2.metric(
        "Columns",
        dataset.shape[1],
    )

    column3.metric(
        "Missing values",
        int(dataset.isna().sum().sum()),
    )

    column4.metric(
        "Duplicated rows",
        int(dataset.duplicated().sum()),
    )

    st.subheader("Dataset sample")

    st.dataframe(
        dataset.head(12),
        use_container_width=True,
    )

    st.subheader("Column guide")

    column_guide = pd.DataFrame(
        {
            "Column": dataset.columns,

            "Data type": (
                dataset.dtypes
                .astype(str)
                .values
            ),

            "Role": [
                (
                    "Identifier"
                    if column == "Student_ID"

                    else "Target"
                    if column == "Depression"

                    else "Feature"
                )

                for column in dataset.columns
            ],
        }
    )

    st.dataframe(
        column_guide,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Cleaning decisions")

    decision1, decision2, decision3 = (
        st.columns(3)
    )

    with decision1:
        st.markdown(
            """
            <div class="card">
                <strong>Missing values</strong>

                No missing values were found,
                so no imputation was needed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with decision2:
        st.markdown(
            """
            <div class="card">
                <strong>Duplicated rows</strong>

                No duplicated rows were found,
                so no rows were removed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with decision3:
        st.markdown(
            """
            <div class="card">
                <strong>Outliers</strong>

                High values were kept because
                they can still represent real
                student behavior.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader(
        "Preprocessing pipeline"
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
                "Apply PCA",
                "Train three models",
            ],

            "Why": [
                "The ID only identifies rows.",
                "Marks sleep below 6 hours.",
                "Marks stress level 7 or higher.",
                "Changes text categories into numbers.",
                "Keeps unseen data for evaluation.",
                "Handles different numerical ranges.",
                "Checks whether fewer components help.",
                "Compares different classifiers.",
            ],
        }
    )

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("PCA summary")

    pca1, pca2, pca3 = st.columns(3)

    pca1.metric(
        "Features before PCA",
        16,
    )

    pca2.metric(
        "Components after PCA",
        12,
    )

    pca3.metric(
        "Variance kept",
        "95.6%",
    )

    st.info(
        "PCA reduced the number of features, "
        "but the Logistic Regression results "
        "became slightly lower. PCA was not used "
        "in the final prediction model."
    )


# =========================================================
# 9. MODEL ARENA
# =========================================================
elif page == "Model Arena":
    st.title("Model Arena")

    st.caption(
        "Compare the three classifiers. "
        "High accuracy is not automatically "
        "the best result with imbalanced data."
    )

    display_results = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    for column in metric_columns:
        display_results[column] = (
            pd.to_numeric(
                display_results[column],
                errors="coerce",
            )
        )

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
            "#6C63FF",
            "#24A89A",
            "#F4A261",
            "#EF6A78",
        ],

        title="Model performance comparison",
    )

    figure.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    figure.update_layout(
        template="plotly_white",
        legend_title_text="",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    best_accuracy = display_results.loc[
        display_results["Accuracy"].idxmax(),
        "Model",
    ]

    best_recall = display_results.loc[
        display_results["Recall"].idxmax(),
        "Model",
    ]

    best_f1 = display_results.loc[
        display_results["F1 Score"].idxmax(),
        "Model",
    ]

    best1, best2, best3 = st.columns(3)

    best1.metric(
        "Highest accuracy",
        best_accuracy,
    )

    best2.metric(
        "Highest recall",
        best_recall,
    )

    best3.metric(
        "Highest F1 score",
        best_f1,
    )

    st.subheader(
        "How each model behaves"
    )

    explanations = {
        "Logistic Regression": (
            "It has lower accuracy, but it "
            "detects the largest share of real "
            "depression cases. It has the best "
            "recall and F1 score."
        ),

        "Decision Tree": (
            "Its accuracy is higher, but it "
            "misses many real depression cases."
        ),

        "Random Forest": (
            "Its accuracy is very high, but "
            "its recall is extremely low. "
            "It mostly predicts No Depression."
        ),
    }

    for model_name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ]:
        matching_row = display_results[
            display_results["Model"]
            .astype(str)
            .str.lower()
            == model_name.lower()
        ]

        if matching_row.empty:
            continue

        result_row = matching_row.iloc[0]

        st.markdown(
            f"""
            <div class="card">
                <strong>{model_name}</strong>

                Accuracy:
                <b>{result_row['Accuracy']:.1%}</b>

                &nbsp;&nbsp;

                Recall:
                <b>{result_row['Recall']:.1%}</b>

                &nbsp;&nbsp;

                F1:
                <b>{result_row['F1 Score']:.1%}</b>

                <br><br>

                {explanations[model_name]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

    st.subheader(
        "Confusion matrix explorer"
    )

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    matrix = get_confusion_matrix(
        selected_model
    )

    if matrix is None:
        st.warning(
            "No confusion matrix was found "
            "for this model."
        )

    else:
        figure = px.imshow(
            matrix,
            text_auto="d",

            x=[
                "No Depression",
                "Depression",
            ],

            y=[
                "No Depression",
                "Depression",
            ],

            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count",
            },

            color_continuous_scale="Blues",

            title=(
                f"{selected_model} "
                "confusion matrix"
            ),
        )

        figure.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        true_negative, false_positive, false_negative, true_positive = (
            np.asarray(matrix).ravel()
        )

        value1, value2, value3, value4 = (
            st.columns(4)
        )

        value1.metric(
            "True negatives",
            f"{true_negative:,}",
        )

        value2.metric(
            "False positives",
            f"{false_positive:,}",
        )

        value3.metric(
            "False negatives",
            f"{false_negative:,}",
        )

        value4.metric(
            "True positives",
            f"{true_positive:,}",
        )

        st.info(
            "False negatives are important "
            "because they are real depression "
            "cases predicted as No Depression."
        )


# =========================================================
# 10. PREDICTION STUDIO
# =========================================================
elif page == "Prediction Studio":
    st.title("Prediction Studio")

    st.caption(
        "Create a student profile, compare "
        "all three models, and test a "
        "what-if scenario."
    )

    st.warning(
        "This is not a medical diagnosis. "
        "It only shows what the trained models "
        "predict from the supplied values."
    )

    gender_values = sorted(
        dataset["Gender"]
        .dropna()
        .astype(str)
        .unique()
    )

    department_values = sorted(
        dataset["Department"]
        .dropna()
        .astype(str)
        .unique()
    )

    with st.form("prediction_form"):
        st.subheader("Student profile")

        input1, input2, input3 = (
            st.columns(3)
        )

        with input1:
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

        with input2:
            cgpa = st.number_input(
                "CGPA",

                min_value=float(
                    dataset["CGPA"].min()
                ),

                max_value=float(
                    dataset["CGPA"].max()
                ),

                value=float(
                    dataset["CGPA"].median()
                ),

                step=0.1,
            )

            sleep_duration = st.slider(
                "Sleep duration",

                float(
                    dataset[
                        "Sleep_Duration"
                    ].min()
                ),

                float(
                    dataset[
                        "Sleep_Duration"
                    ].max()
                ),

                float(
                    dataset[
                        "Sleep_Duration"
                    ].median()
                ),

                step=0.1,
            )

            study_hours = st.slider(
                "Study hours",

                float(
                    dataset[
                        "Study_Hours"
                    ].min()
                ),

                float(
                    dataset[
                        "Study_Hours"
                    ].max()
                ),

                float(
                    dataset[
                        "Study_Hours"
                    ].median()
                ),

                step=0.1,
            )

        with input3:
            social_media_hours = st.slider(
                "Social media hours",

                float(
                    dataset[
                        "Social_Media_Hours"
                    ].min()
                ),

                float(
                    dataset[
                        "Social_Media_Hours"
                    ].max()
                ),

                float(
                    dataset[
                        "Social_Media_Hours"
                    ].median()
                ),

                step=0.1,
            )

            physical_activity = st.slider(
                "Physical activity",

                int(
                    dataset[
                        "Physical_Activity"
                    ].min()
                ),

                int(
                    dataset[
                        "Physical_Activity"
                    ].max()
                ),

                int(
                    dataset[
                        "Physical_Activity"
                    ].median()
                ),
            )

            stress_level = st.slider(
                "Stress level",

                int(
                    dataset[
                        "Stress_Level"
                    ].min()
                ),

                int(
                    dataset[
                        "Stress_Level"
                    ].max()
                ),

                int(
                    dataset[
                        "Stress_Level"
                    ].median()
                ),
            )

        submitted = st.form_submit_button(
            "Run all three models",
            use_container_width=True,
        )

    if submitted:
        st.session_state[
            "student_profile"
        ] = {
            "Age": age,
            "Gender": gender,
            "Department": department,
            "CGPA": cgpa,
            "Sleep_Duration": sleep_duration,
            "Study_Hours": study_hours,
            "Social_Media_Hours": (
                social_media_hours
            ),
            "Physical_Activity": (
                physical_activity
            ),
            "Stress_Level": stress_level,
        }

    if "student_profile" in st.session_state:
        student_profile = st.session_state[
            "student_profile"
        ]

        prepared_input = prepare_model_input(
            student_profile
        )

        predictions = [
            predict_with_model(
                model_name,
                prepared_input,
            )

            for model_name in models
        ]

        st.subheader(
            "Three-model comparison"
        )

        prediction_columns = st.columns(
            len(predictions)
        )

        for display_column, result in zip(
            prediction_columns,
            predictions,
        ):
            with display_column:
                predicted_text = (
                    "Higher predicted risk"

                    if result["prediction"] == 1

                    else "Lower predicted risk"
                )

                probability_text = (
                    f"{result['probability']:.1%}"

                    if result[
                        "probability"
                    ] is not None

                    else "Not available"
                )

                card_class = (
                    "risk-high"

                    if result["prediction"] == 1

                    else "risk-low"
                )

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <strong>
                            {result['model']}
                        </strong>

                        <h4>
                            {predicted_text}
                        </h4>

                        Depression probability:
                        <b>{probability_text}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        probability_rows = [
            {
                "Model": result["model"],
                "Probability": (
                    result["probability"]
                ),
            }

            for result in predictions

            if result["probability"]
            is not None
        ]

        if probability_rows:
            probability_data = pd.DataFrame(
                probability_rows
            )

            figure = px.bar(
                probability_data,
                x="Model",
                y="Probability",
                color="Model",
                text_auto=".1%",
                color_discrete_map=MODEL_COLORS,

                title=(
                    "Predicted depression "
                    "probability by model"
                ),
            )

            figure.update_yaxes(
                tickformat=".0%",
                range=[0, 1],
            )

            figure.update_layout(
                template="plotly_white",
                showlegend=False,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        st.info(
            "The models can disagree because "
            "they learn patterns in different ways."
        )

        st.subheader("What-if simulator")

        what_if_model = st.selectbox(
            "Choose a model for the scenario",
            list(models.keys()),
        )

        scenario1, scenario2 = st.columns(2)

        with scenario1:
            scenario_sleep = st.slider(
                "Scenario sleep duration",

                float(
                    dataset[
                        "Sleep_Duration"
                    ].min()
                ),

                float(
                    dataset[
                        "Sleep_Duration"
                    ].max()
                ),

                float(
                    student_profile[
                        "Sleep_Duration"
                    ]
                ),

                step=0.1,
                key="scenario_sleep",
            )

        with scenario2:
            scenario_stress = st.slider(
                "Scenario stress level",

                int(
                    dataset[
                        "Stress_Level"
                    ].min()
                ),

                int(
                    dataset[
                        "Stress_Level"
                    ].max()
                ),

                int(
                    student_profile[
                        "Stress_Level"
                    ]
                ),

                key="scenario_stress",
            )

        scenario_profile = (
            student_profile.copy()
        )

        scenario_profile[
            "Sleep_Duration"
        ] = scenario_sleep

        scenario_profile[
            "Stress_Level"
        ] = scenario_stress

        original_result = predict_with_model(
            what_if_model,

            prepare_model_input(
                student_profile
            ),
        )

        scenario_result = predict_with_model(
            what_if_model,

            prepare_model_input(
                scenario_profile
            ),
        )

        if (
            original_result["probability"]
            is not None

            and scenario_result[
                "probability"
            ] is not None
        ):
            probability_change = (
                scenario_result[
                    "probability"
                ]

                - original_result[
                    "probability"
                ]
            )

            result1, result2, result3 = (
                st.columns(3)
            )

            result1.metric(
                "Original probability",

                f"{original_result['probability']:.1%}",
            )

            result2.metric(
                "Scenario probability",

                f"{scenario_result['probability']:.1%}",
            )

            result3.metric(
                "Model output change",

                f"{probability_change:+.1%}",
            )

        st.caption(
            "The what-if result only shows "
            "how the model output changes. "
            "It does not prove cause and effect."
        )

        with st.expander(
            "See the exact encoded model input"
        ):
            st.dataframe(
                prepared_input,
                use_container_width=True,
            )
