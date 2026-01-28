# Employee Attrition Risk Prediction System

This project is an end-to-end **machine learning system** designed to estimate **employee attrition risk** and explain **why** a particular prediction was made.

The system combines predictive modeling, probability-based risk interpretation, and SHAP-based explainability, and is deployed as an interactive **Streamlit web application**.

---

## Problem Statement

Employee attrition is costly and difficult to predict because it depends on multiple interacting factors such as compensation, job role, satisfaction, work-life balance, and career progression.

The goal of this project is **not to predict attrition with certainty**, but to:

* Identify **relative attrition risk**
* Provide **early warning signals**
* Explain **key factors influencing risk** for each employee

---

## Approach \& Methodology

### 1\.  Data Preprocessing

* Removed non-informative and constant columns
* Applied categorical encoding (binary + one-hot encoding)
* Scaled numerical features using `StandardScaler`
* Prepared a fully numeric dataset suitable for ML models

### 2\. Model Training \& Selection

Trained and evaluated multiple classification models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost

Models were compared using:

* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC–AUC and Precision–Recall curves

Because attrition is a **class-imbalanced problem**, **Recall and F1-score** were prioritized over raw accuracy.

**XGBoost** was selected as the final model due to:

* Strong recall for attrition cases
* Robust performance after hyperparameter tuning
* Good balance between false positives and false negatives

---

## Probability-Based Risk Interpretation

Attrition is a **rare event**, so predicted probabilities are naturally **compressed** (e.g., 0.25–0.40).

Instead of using a fixed yes/no decision, predictions are presented as **risk levels**:

* **Low Risk** → probability < 0.20
* **Medium Risk** → 0.20 ≤ probability < 0.35
* **High Risk** → probability ≥ 0.35

This reflects **real-world HR decision-making**, where risk ranking is more valuable than absolute certainty.

---

## Explainability with SHAP

To ensure transparency and trust:

* **SHAP (SHapley Additive exPlanations)** is used
* For each prediction, the app shows:

  * Features increasing attrition risk
  * Features decreasing attrition risk

* This allows HR users to understand **why** a prediction was made, not just the result

---

## Web Application (Streamlit)

The deployed Streamlit app allows users to:

* Enter detailed employee information
* View attrition probability and risk level
* Inspect SHAP-based explanations for individual predictions
* Understand the overall project logic directly within the UI

The application reconstructs the **full encoded feature space** used during training, ensuring consistency between training and inference.

---

## Project Structure

```text
Employee-Attrition-Risk-Prediction/
│
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── attrition_pipeline.pkl       # Scaler + trained ML pipeline
├── best_xgb_model.pkl           # XGBoost model (used for SHAP)
├── feature_columns.pkl          # Encoded feature list (44 features)
├── default_values.pkl           # Default feature values
│
└── WA_Fn-UseC_-HR-Employee-Attrition.csv

```
---
##Development Notebook

The complete model development process — including data preprocessing, feature engineering, model training, evaluation, and explainability — is documented in the notebook below:

**[View Full Project Notebook](https://github.com/pvb-king/Employee-Attrition-Risk-Prediction/blob/6d655e0213c062c65f7686b2d7c269a7d04176ba/Employee_Attrition_Prediction_BhargavanPV.ipynb)**

This notebook captures the full learning, experimentation, and decision-making journey behind the final deployed system.
---

## How to Run the Project

### 1\. Install Dependencies

```bash
pip install -r requirements.txt

### 2. Run the Streamlit app

The application will open in your browser

### Key Learning

- Attrition prediction is \*\*Multi-Factorial\*\*, not rule-based
- Imbalanced datasets require probability-aware interpretation
- High explainability is critical for HR analytics adoptation
- Deployment requires strict feature alignment between training and interference
- SHAP greatly improves trust and interpretability of ML system



