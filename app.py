import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Student Depression Analysis",
    page_icon="🎓",
    layout="wide",
)

DATA_FILE = "student_lifestyle.csv"
MODEL_FILE = "logistic_model.joblib"
SCALER_FILE = "scaler.joblib"
FEATURES_FILE = "feature_columns.joblib"
RESULTS_FILE = "model_results.csv"
MATRICES_FILE = "confusion_matrices.joblib"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_FILE)


@st.cache_resource
def load_saved_files():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_columns = joblib.load(FEATURES_FILE)
    confusion_matrices = joblib.load(MATRICES_FILE)
    return model, scaler, feature_columns, confusion_matrices


dataset = load_data()
results = load_results()
model, scaler, feature_columns, confusion_matrices = load_saved_files()

st.title("🎓 Student Depression Analysis")
st.write(
    "This app shows the full machine learning workflow for predicting student "
    "depression, including the data, EDA, preprocessing, model results, and a live prediction."
)

page = st.sidebar.radio(
    "Choose a section",
    [
        "Data Overview",
        "EDA",
        "Cleaning & Preprocessing",
        "Model Results",
        "Interactive Prediction",
    ],
)

if page == "Data Overview":
    st.header("1. Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", dataset.shape[0])
    col2.metric("Columns", dataset.shape[1])
    col3.metric("Target", "Depression")

    st.subheader("Problem")
    st.write(
        "The goal is to use student lifestyle information such as sleep, study hours, "
        "social media use, physical activity, stress, and CGPA to predict the Depression target."
    )

    st.subheader("Sample of the dataset")
    st.dataframe(dataset.head(10), use_container_width=True)

    st.subheader("Column types")
    types_df = pd.DataFrame(
        {
            "Column": dataset.columns,
            "Type": dataset.dtypes.astype(str).values,
        }
    )
    st.dataframe(types_df, use_container_width=True)


elif page == "EDA":
    st.header("2. Exploratory Data Analysis")

    st.subheader("Depression target distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=dataset, x="Depression", ax=ax)
    ax.set_title("Depression Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The target is imbalanced. Most values are no depression, while the depression class is much smaller."
    )

    st.subheader("Numerical feature distribution")
    numerical_options = [
        col
        for col in [
            "Age",
            "CGPA",
            "Sleep_Duration",
            "Study_Hours",
            "Social_Media_Hours",
            "Physical_Activity",
            "Stress_Level",
        ]
        if col in dataset.columns
    ]

    selected_feature = st.selectbox("Choose a numerical feature", numerical_options)

    fig, ax = plt.subplots()
    if dataset[selected_feature].nunique() <= 15:
        sns.countplot(data=dataset, x=selected_feature, ax=ax)
    else:
        sns.histplot(data=dataset, x=selected_feature, bins=20, ax=ax)
    ax.set_title(f"{selected_feature} Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Gender and Department")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=dataset, x="Gender", ax=ax)
        ax.set_title("Gender Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=dataset, x="Department", ax=ax)
        ax.set_title("Department Distribution")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation heatmap")
    corr_data = dataset.drop(columns=["Student_ID"], errors="ignore").copy()

    if "Depression" in corr_data.columns and corr_data["Depression"].dtype == bool:
        corr_data["Depression"] = corr_data["Depression"].astype(int)

    corr = corr_data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Main EDA insights")
    st.write(
        """
        - The Depression target is not balanced because the no-depression class is much bigger.
        - Male and Female counts are almost equal.
        - The departments have almost the same number of students.
        - Most students sleep around 6 to 8 hours.
        - Most students have stress levels around 3 to 5.
        - CGPA has the strongest numerical correlation with Depression, but the relation is still weak.
        """
    )


elif page == "Cleaning & Preprocessing":
    st.header("3. Cleaning & Preprocessing Summary")

    missing_count = int(dataset.isnull().sum().sum())
    duplicate_count = int(dataset.duplicated().sum())

    col1, col2 = st.columns(2)
    col1.metric("Missing values", missing_count)
    col2.metric("Duplicated rows", duplicate_count)

    st.subheader("Cleaning")
    st.write(
        "No missing values or duplicated rows were removed when their counts were zero. "
        "Possible high outliers were kept because the values can represent real student behavior."
    )

    st.subheader("Feature engineering")
    st.code(
        'Low_Sleep = 1 when Sleep_Duration < 6\n'
        'High_Stress = 1 when Stress_Level >= 7',
        language="text",
    )

    st.subheader("Encoding")
    st.write(
        "Gender and Department were changed from text categories into dummy columns containing 0 and 1."
    )

    st.subheader("Scaling")
    st.write(
        "StandardScaler was fitted only on the training data. Logistic Regression and PCA used scaled data. "
        "Decision Tree and Random Forest did not need scaling."
    )

    st.subheader("PCA")
    st.write(
        "PCA reduced the features from 16 to 12 components and kept around 95.6% of the variance. "
        "The model results became slightly lower, so PCA did not improve the final model."
    )


elif page == "Model Results":
    st.header("4. Model Results")

    st.subheader("Model comparison")
    st.dataframe(results, use_container_width=True)

    st.write(
        "Random Forest had the highest accuracy, but its recall was very low. "
        "Logistic Regression had the highest recall and F1 score, so it was selected as the final model."
    )

    st.subheader("Confusion matrix")
    model_name = st.selectbox(
        "Choose a model",
        list(confusion_matrices.keys()),
    )

    cm = confusion_matrices[model_name]

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Depression", "Depression"],
        yticklabels=["No Depression", "Depression"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)


elif page == "Interactive Prediction":
    st.header("5. Interactive Prediction")
    st.warning(
        "This is a student machine learning project and not a medical diagnosis tool."
    )

    gender_values = sorted(dataset["Gender"].dropna().astype(str).unique().tolist())
    department_values = sorted(
        dataset["Department"].dropna().astype(str).unique().tolist()
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 24, 21)
            gender = st.selectbox("Gender", gender_values)
            department = st.selectbox("Department", department_values)
            cgpa = st.number_input(
                "CGPA",
                min_value=float(dataset["CGPA"].min()),
                max_value=float(dataset["CGPA"].max()),
                value=float(dataset["CGPA"].median()),
                step=0.1,
            )
            sleep_duration = st.slider(
                "Sleep Duration",
                float(dataset["Sleep_Duration"].min()),
                float(dataset["Sleep_Duration"].max()),
                float(dataset["Sleep_Duration"].median()),
                step=0.1,
            )

        with col2:
            study_hours = st.slider(
                "Study Hours",
                float(dataset["Study_Hours"].min()),
                float(dataset["Study_Hours"].max()),
                float(dataset["Study_Hours"].median()),
                step=0.1,
            )
            social_media_hours = st.slider(
                "Social Media Hours",
                float(dataset["Social_Media_Hours"].min()),
                float(dataset["Social_Media_Hours"].max()),
                float(dataset["Social_Media_Hours"].median()),
                step=0.1,
            )
            physical_activity = st.slider(
                "Physical Activity",
                int(dataset["Physical_Activity"].min()),
                int(dataset["Physical_Activity"].max()),
                int(dataset["Physical_Activity"].median()),
            )
            stress_level = st.slider(
                "Stress Level",
                int(dataset["Stress_Level"].min()),
                int(dataset["Stress_Level"].max()),
                int(dataset["Stress_Level"].median()),
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {column: 0 for column in feature_columns}

        base_values = {
            "Age": age,
            "CGPA": cgpa,
            "Sleep_Duration": sleep_duration,
            "Study_Hours": study_hours,
            "Social_Media_Hours": social_media_hours,
            "Physical_Activity": physical_activity,
            "Stress_Level": stress_level,
            "Low_Sleep": int(sleep_duration < 6),
            "High_Stress": int(stress_level >= 7),
        }

        for name, value in base_values.items():
            if name in row:
                row[name] = value

        # Match dummy columns without assuming capital or lower-case spelling.
        for column in feature_columns:
            lower_column = column.lower()

            if lower_column.startswith("gender_"):
                category = column.split("_", 1)[1]
                row[column] = int(category.lower() == str(gender).lower())

            if lower_column.startswith("department_"):
                category = column.split("_", 1)[1]
                row[column] = int(category.lower() == str(department).lower())

        input_data = pd.DataFrame([row], columns=feature_columns)
        input_scaled = scaler.transform(input_data)

        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])

        if prediction == 1:
            st.error("Prediction: Depression")
        else:
            st.success("Prediction: No Depression")

        st.metric("Predicted depression probability", f"{probability:.1%}")

        with st.expander("See the prepared model input"):
            st.dataframe(input_data, use_container_width=True)
