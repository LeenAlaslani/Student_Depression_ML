# Student Depression Streamlit App

This repository contains a machine learning project for student depression classification.

## Files

- `app.py`: Streamlit application
- `student_lifestyle.csv`: dataset
- `logistic_model.joblib`: final trained model
- `scaler.joblib`: fitted StandardScaler
- `feature_columns.joblib`: model feature names and order
- `model_results.csv`: model evaluation results
- `confusion_matrices.joblib`: saved confusion matrices
- `requirements.txt`: Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application includes:
1. Data overview
2. EDA visualizations
3. Cleaning and preprocessing summary
4. Model results and confusion matrices
5. Interactive prediction
