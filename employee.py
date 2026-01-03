import streamlit as st
import pandas as pd
import numpy as np
import joblib
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL & FEATURES
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("employee_retention_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, feature_columns = load_model()

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("📊 Employee Retention Prediction Dashboard")
st.markdown(
    """
    This application predicts **employee attrition risk** using a trained  
    **LightGBM machine learning model**.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR - INPUT FEATURES
# --------------------------------------------------
st.sidebar.header("🧾 Employee Information")

experience = st.sidebar.slider(
    "Years of Experience", 0, 25, 5
)

training_hours = st.sidebar.slider(
    "Training Hours", 0, 350, 40
)

city_dev_index = st.sidebar.slider(
    "City Development Index", 0.0, 1.0, 0.75
)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["Primary School", "High School", "Graduate", "Masters", "Phd"]
)

education_map = {
    "Primary School": 0,
    "High School": 1,
    "Graduate": 2,
    "Masters": 3,
    "Phd": 4
}

company_size = st.sidebar.selectbox(
    "Company Size",
    ["Very Small", "Small", "Medium", "Large"]
)

company_size_map = {
    "Very Small": 0,
    "Small": 1,
    "Medium": 2,
    "Large": 3
}

last_new_job = st.sidebar.selectbox(
    "Years Since Last Job Change",
    ["never", "1", "2", "3", "4", ">4"]
)

last_job_map = {
    "never": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    ">4": 5
}

threshold = st.sidebar.slider(
    "Prediction Threshold",
    0.1, 0.9, 0.5, step=0.05
)

# --------------------------------------------------
# CREATE INPUT DATA
# --------------------------------------------------
input_data = {
    "experience": experience,
    "training_hours": training_hours,
    "city_development_index": city_dev_index,
    "education_level": education_map[education_level],
    "company_size_ord": company_size_map[company_size],
    "last_new_job": last_job_map[last_new_job]
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --------------------------------------------------
# MAIN PANEL - PREDICTION
# --------------------------------------------------
st.subheader("🔮 Prediction Result")

if st.button("Predict Employee Attrition"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = "Likely to Leave" if prob >= threshold else "Likely to Stay"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Attrition Probability",
            value=f"{prob:.2f}"
        )

    with col2:
        st.metric(
            label="Decision Threshold",
            value=f"{threshold:.2f}"
        )

    with col3:
        if prediction == "Likely to Leave":
            st.error("🚨 Likely to Leave")
        else:
            st.success("✅ Likely to Stay")

    st.progress(int(prob * 100))

    st.markdown("### 📌 Prediction Insight")
    if prediction == "Likely to Leave":
        st.write(
            "The model indicates a **high risk of attrition**. "
            "HR teams should consider proactive retention actions."
        )
    else:
        st.write(
            "The employee is predicted to be **stable** with low attrition risk."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Model: LightGBM | Metric Focus: Recall | Built for Employee Retention Analytics"
)
