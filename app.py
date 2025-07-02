import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and feature columns
model = joblib.load("employee_salary_model.pkl")
columns = joblib.load("model_features.pkl")

# App Configuration
st.set_page_config(page_title="IBM Salary Prediction Tool", layout="centered", page_icon="💼")

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        text-align: center;
        color: #1F4E79;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .salary-box {
        background-color: #f3f8fc;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ccc;
        text-align: center;
        font-size: 24px;
        color: #0077b6;
        font-weight: bold;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        color: #888;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>💼 IBM Employee Salary Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Developed by Lakshmi Priya R | IBM AI & Data Science Internship Project</div>", unsafe_allow_html=True)

# Input Layout
with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Info")
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
        job_role = st.selectbox("Job Role", [
            "Analyst", "Engineer", "Manager", "Data Scientist", "Software Developer",
            "System Admin", "Business Analyst", "Team Lead", "Intern", "HR", "QA Tester"
        ])

    with col2:
        st.subheader("💼 Experience")
        prior_exp = st.slider("Prior Experience (Years)", 0, 20, 2)
        current_exp = st.slider("Current Company Experience (Years)", 0, 20, 3)
        total_exp = prior_exp + current_exp
        st.info(f"🧠 Total Experience: {total_exp} years")

    submitted = st.form_submit_button("🔍 Predict Salary")

if submitted:
    # Build the feature vector
    features = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    features.at[0, 'Age'] = age
    features.at[0, 'Prior_Exp'] = prior_exp
    features.at[0, 'Current_Exp'] = current_exp
    features.at[0, 'Total_Exp'] = total_exp

    if f"Education_{education}" in columns:
        features.at[0, f"Education_{education}"] = 1
    if f"Job_Role_{job_role}" in columns:
        features.at[0, f"Job_Role_{job_role}"] = 1

    # Prediction
    salary = model.predict(features)[0]

    # Output
    st.markdown("---")
    st.markdown("<div class='salary-box'>💰 Estimated Salary: ₹ {:,}</div>".format(round(salary)), unsafe_allow_html=True)

    with st.expander("📋 Employee Summary"):
        st.write(f"**Age:** {age}")
        st.write(f"**Education:** {education}")
        st.write(f"**Job Role:** {job_role}")
        st.write(f"**Prior Experience:** {prior_exp} years")
        st.write(f"**Current Experience:** {current_exp} years")
        st.write(f"**Total Experience:** {total_exp} years")

    st.markdown("---")
    st.caption("📊 Model powered by Random Forest Regressor | Built as part of the IBM AI & Data Science Internship")

# IBM Footer
st.markdown("<div class='footer'>IBM SkillsBuild Program | Project Code: IBM2025DS23 | Confidential & Educational Use Only</div>", unsafe_allow_html=True)




