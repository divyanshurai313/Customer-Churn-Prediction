import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
import os

import os

current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, "model.pkl")
scaler_path = os.path.join(current_folder, "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn.")

st.sidebar.header("Customer Information")

# -----------------------------
# Numeric Inputs
# -----------------------------
senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)

# Feature Engineering
if tenure == 0:
    avg_spend = monthly
else:
    avg_spend = total / tenure

# -----------------------------
# Customer Details
# -----------------------------
gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes"]
)

tech = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# -----------------------------
# Create Feature Vector
# -----------------------------
data = {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "avg_monthly_spend": avg_spend,

    "gender_Male": 1 if gender == "Male" else 0,

    "Partner_Yes": 1 if partner == "Yes" else 0,

    "Dependents_Yes": 1 if dependents == "Yes" else 0,

    "PhoneService_Yes": 1 if phone == "Yes" else 0,

    "MultipleLines_Yes": 1 if multiple == "Yes" else 0,

    "InternetService_Fiber optic":
        1 if internet == "Fiber optic" else 0,

    "InternetService_No":
        1 if internet == "No" else 0,

    "OnlineSecurity_Yes":
        1 if security == "Yes" else 0,

    "OnlineBackup_Yes":
        1 if backup == "Yes" else 0,

    "DeviceProtection_Yes":
        1 if device == "Yes" else 0,

    "TechSupport_Yes":
        1 if tech == "Yes" else 0,

    "StreamingTV_Yes":
        1 if tv == "Yes" else 0,

    "StreamingMovies_Yes":
        1 if movies == "Yes" else 0,

    "Contract_One year":
        1 if contract == "One year" else 0,

    "Contract_Two year":
        1 if contract == "Two year" else 0,

    "PaperlessBilling_Yes":
        1 if paperless == "Yes" else 0,

    "PaymentMethod_Credit card (automatic)":
        1 if payment == "Credit card (automatic)" else 0,

    "PaymentMethod_Electronic check":
        1 if payment == "Electronic check" else 0,

    "PaymentMethod_Mailed check":
        1 if payment == "Mailed check" else 0,

    "tenure_group_13-24":
        1 if 13 <= tenure <= 24 else 0,

    "tenure_group_25-48":
        1 if 25 <= tenure <= 48 else 0,

    "tenure_group_49+":
        1 if tenure >= 49 else 0
}

feature_order = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'avg_monthly_spend',
    'gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'PhoneService_Yes',
    'MultipleLines_Yes',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
    'tenure_group_13-24',
    'tenure_group_25-48',
    'tenure_group_49+'
]

input_df = pd.DataFrame([data])
input_df = input_df[feature_order]

scaled_input = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    probability = model.predict_proba(scaled_input)[0]

    no_churn_probability = probability[0] * 100
    churn_probability = probability[1] * 100

    st.subheader("Prediction Result")

    if churn_probability >= 50:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to CHURN")

    st.write(f"**Churn Probability:** {churn_probability:.2f}%")
    st.write(f"**No Churn Probability:** {no_churn_probability:.2f}%")

    st.markdown("---")

    st.subheader("Input Summary")
    st.dataframe(input_df)

    # -----------------------------
# Project Visualizations
# -----------------------------
st.markdown("---")
st.header("📊 Project Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "Images/Churn_Count.png",
        caption="Customer Churn Distribution",
        use_container_width=True
    )

    st.image(
        "Images/contract_vs_churn.png",
        caption="Contract Type vs Churn",
        use_container_width=True
    )

    st.image(
        "Images/payment_method_vs_churn.png",
        caption="Payment Method vs Churn",
        use_container_width=True
    )

    st.image(
        "Images/confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

with col2:
    st.image(
        "Images/tenure_distribution.png",
        caption="Tenure Distribution",
        use_container_width=True
    )

    st.image(
        "Images/internet_service_vs_churn.png",
        caption="Internet Service vs Churn",
        use_container_width=True
    )

    st.image(
        "Images/correlation_heatmap.png",
        caption="Correlation Heatmap",
        use_container_width=True
    )
