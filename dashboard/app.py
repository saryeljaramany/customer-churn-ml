import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Customer Churn Risk Dashboard", page_icon="📉", layout="wide")

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_URL).rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "30"))


@st.cache_data(ttl=60)
def fetch_api_meta(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/meta", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def build_payload(customer_data: dict) -> dict:
    return {"customers": [customer_data]}


def format_risk(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High Risk", "🔴"
    if probability >= 0.4:
        return "Moderate Risk", "🟠"
    return "Low Risk", "🟢"


st.title("Customer Churn Risk Dashboard")
st.caption("Estimate the chance of a customer leaving and identify who may need proactive support.")

st.info(
    "This dashboard sends customer information to the prediction service "
    "and returns a churn risk score."
)

with st.sidebar:
    st.header("Service Settings")
    api_url = st.text_input("Prediction Service URL", value=API_BASE_URL)
    st.caption("Use your API URL in deployment, or keep the local URL for development.")

try:
    meta = fetch_api_meta(api_url)
    health_response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    health_response.raise_for_status()
    health = health_response.json()
except requests.RequestException:
    st.error(
        "We could not connect to the prediction service. "
        "Please verify the service URL and try again."
    )
    st.stop()

if not health.get("model_ready", False):
    st.warning(
        "The prediction service is online, but the model is not ready yet. "
        "Please check deployment artifacts and try again shortly."
    )

fields = meta["fields"]

st.subheader("Customer Details")
with st.form("prediction_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        customer_id = st.text_input("Customer ID", value="CUST-1001")
        gender = st.selectbox("Gender", options=fields["gender"])
        senior_citizen = st.selectbox("Senior Citizen", options=fields["SeniorCitizen"])
        partner = st.selectbox("Partner", options=fields["Partner"])
        dependents = st.selectbox("Dependents", options=fields["Dependents"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        phone_service = st.selectbox("Phone Service", options=fields["PhoneService"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", options=fields["MultipleLines"])
        internet_service = st.selectbox("Internet Service", options=fields["InternetService"])
        online_security = st.selectbox("Online Security", options=fields["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", options=fields["OnlineBackup"])
        device_protection = st.selectbox("Device Protection", options=fields["DeviceProtection"])
        tech_support = st.selectbox("Tech Support", options=fields["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", options=fields["StreamingTV"])

    with col3:
        streaming_movies = st.selectbox("Streaming Movies", options=fields["StreamingMovies"])
        contract = st.selectbox("Contract Type", options=fields["Contract"])
        paperless_billing = st.selectbox("Paperless Billing", options=fields["PaperlessBilling"])
        payment_method = st.selectbox("Payment Method", options=fields["PaymentMethod"])
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=0.01
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=20000.0, value=840.0, step=0.01
        )

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    customer_payload = {
        "customerID": customer_id.strip(),
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    if not customer_payload["customerID"]:
        st.error("Please enter a customer ID before submitting.")
        st.stop()

    try:
        response = requests.post(
            f"{api_url}/predict",
            json=build_payload(customer_payload),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            error_message = "The prediction request could not be completed."
            try:
                detail = response.json().get("detail")
                if isinstance(detail, str) and detail:
                    error_message = detail
            except ValueError:
                pass
            st.error(error_message)
            st.stop()

        result = response.json()["results"][0]
    except requests.RequestException:
        st.error("The prediction service did not respond. Please try again in a moment.")
        st.stop()

    churn_probability = float(result["churn_probability"])
    confidence = float(result["confidence"])
    risk_label, risk_icon = format_risk(churn_probability)

    st.subheader("Prediction Result")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Churn Probability", f"{churn_probability:.1%}")
    kpi2.metric("Confidence", f"{confidence:.1%}")
    kpi3.metric("Risk Category", f"{risk_icon} {risk_label}")

    st.write("Customer Summary")
    st.dataframe(pd.DataFrame([result]), hide_index=True, use_container_width=True)

