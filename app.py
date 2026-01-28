import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD ARTIFACTS ----------------
pipeline = joblib.load("attrition_pipeline.pkl")      # sklearn Pipeline
model = joblib.load("best_xgb_model.pkl")                   # raw XGBoost model (for SHAP)
feature_columns = joblib.load("feature_columns.pkl")  # list of all encoded features
default_values = joblib.load("default_values.pkl")    # dict of defaults

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Employee Attrition Risk Analyzer", layout="wide")
st.title("Employee Attrition Risk Analyzer")
st.caption("Designed and built by Bhargavan PV")

st.write(
    "Predict employee attrition risk using real HR signals — and understand *what factors are driving the risk* before it becomes a resignation."
)

# ---------------- BASE INPUT DF (CRITICAL) ----------------
input_df = pd.DataFrame([default_values]).reindex(
    columns=feature_columns, fill_value=0
)

# =========================================================

#  NUMERIC INPUTS

# =========================================================
st.header("Employee Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    distance = st.number_input("Distance From Home", 1, 30, 10)
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 3000)

with col2:
    total_working_years = st.number_input("Total Working Years", 0, 40, 8)
    years_at_company = st.number_input("Years At Company", 0, 40, 2)
    years_in_role = st.number_input("Years In Current Role", 0, 20, 2)

with col3:
    years_since_promo = st.number_input("Years Since Last Promotion", 0, 15, 1)
    num_companies = st.number_input("Num Companies Worked", 0, 10, 2)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

# Assign numeric values
input_df.loc[0, "Age"] = age
input_df.loc[0, "DistanceFromHome"] = distance
input_df.loc[0, "MonthlyIncome"] = monthly_income
input_df.loc[0, "TotalWorkingYears"] = total_working_years
input_df.loc[0, "YearsAtCompany"] = years_at_company
input_df.loc[0, "YearsInCurrentRole"] = years_in_role
input_df.loc[0, "YearsSinceLastPromotion"] = years_since_promo
input_df.loc[0, "NumCompaniesWorked"] = num_companies
input_df.loc[0, "JobLevel"] = job_level

# =========================================================

# BINARY INPUTS

# =========================================================
st.header("Work Conditions")

col1, col2 = st.columns(2)

with col1:
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

input_df.loc[0, "OverTime"] = 1 if overtime == "Yes" else 0
input_df.loc[0, "Gender"] = 1 if gender == "Male" else 0
input_df.loc[0, "StockOptionLevel"] = stock_option

# =========================================================

# 😊 SATISFACTION & ENGAGEMENT

# ========================================================

# ---------------- ASSUME NEUTRAL SATISFACTION ----------------
input_df.loc[0, "JobSatisfaction"] = 3
input_df.loc[0, "EnvironmentSatisfaction"] = 3
input_df.loc[0, "RelationshipSatisfaction"] = 3
input_df.loc[0, "WorkLifeBalance"] = 3

# =========================================================

# 🧩 CATEGORICAL → ONE-HOT

# =========================================================

st.header("Job & Personal Details")

# ---- Job Role ----
job_roles = [
    "Sales Executive",
    "Research Scientist",
    "Laboratory Technician",
    "Manufacturing Director",
    "Healthcare Representative",
    "Manager",
    "Sales Representative",
    "Research Director",
    "Human Resources"
]
job_role = st.selectbox("Job Role", job_roles)

for role in job_roles:
    col = f"JobRole_{role}"
    if col in input_df.columns:
        input_df.loc[0, col] = 1 if role == job_role else 0

# ---- Department ----
departments = ["Sales", "Research & Development", "Human Resources"]
department = st.selectbox("Department", departments)

for dept in departments:
    col = f"Department_{dept}"
    if col in input_df.columns:
        input_df.loc[0, col] = 1 if dept == department else 0

# ---- Marital Status ----
marital_statuses = ["Single", "Married", "Divorced"]
marital_status = st.selectbox("Marital Status", marital_statuses)

for status in marital_statuses:
    col = f"MaritalStatus_{status}"
    if col in input_df.columns:
        input_df.loc[0, col] = 1 if status == marital_status else 0

# ---- Business Travel ----
travel_options = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
business_travel = st.selectbox("Business Travel", travel_options)

for travel in travel_options:
    col = f"BusinessTravel_{travel}"
    if col in input_df.columns:
        input_df.loc[0, col] = 1 if travel == business_travel else 0

# =========================================================
# THRESHOLD
# =========================================================
threshold = st.slider(
    "Attrition Sensitivity Threshold",
    min_value=0.1,
    max_value=0.5,
    value=0.35,
    step=0.05
)

# =========================================================
# PREDICTION + EXPLANATION
# =========================================================
if st.button("Predict Attrition Risk"):
    X_infer = input_df.values

    probability = pipeline.predict_proba(X_infer)[0][1]

    if probability < 0.20:
        risk_level = "Low Risk"
        color = "green"
    elif probability < 0.35:
        risk_level = "Medium Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    st.subheader("📈 Prediction Result")
    st.markdown(f"### Attrition Probability: **{probability:.2f}**")
    st.markdown(f"### Risk Level: **:{color}[{risk_level}]**")

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("Why this prediction? (SHAP Explanation)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_infer)

    fig, ax = plt.subplots()
    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_infer[0],
            feature_names=feature_columns
        ),
        show=False
    )
    st.pyplot(fig)

    st.markdown("""
    **How to read this chart:**
    - Features pushing risk **up** increase attrition likelihood  
    - Features pushing risk **down** reduce attrition likelihood  
    - The model evaluates **combined effects**, not single rules
    """)

    # ---------------- PROJECT EXPLANATION ----------------
    st.subheader("About this Project")
    st.write("""
    This system predicts employee attrition risk using a machine learning model trained on historical HR data.
    
    Attrition is a **rare and multi-factorial event**, so predictions are expressed as **risk levels**
    rather than absolute yes/no outcomes.
    
    The objective is **early identification of potential risk**, enabling proactive HR interventions.
    """)

    # ---------------- PERSONAL LEARNINGS ----------------
    st.subheader("Key Learnings")
    st.write("""
    - Attrition prediction requires combining compensation, engagement, and career signals  
    - Imbalanced datasets produce **compressed but meaningful probabilities**  
    - Explainability (SHAP) is critical for trust in HR analytics  
    - Deployment demands strict feature alignment between training and inference  
    """)
