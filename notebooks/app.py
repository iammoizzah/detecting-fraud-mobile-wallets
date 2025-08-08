# cd ~/Desktop/detecting-fraud-mobile-wallets/notebooks
# streamlit run app.py

import streamlit as st
import numpy as np
import joblib
import time
import pandas as pd
from datetime import time as dtime  # For default time

# Set page configuration
st.set_page_config(page_title="Mobile Wallet Fraud Detection", page_icon="📱", layout="wide")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("rf_model_smote.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'rf_model_smote.pkl' and 'scaler.pkl' are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Sidebar for model info and settings
with st.sidebar:
    st.header("About the Model")
    st.markdown("""
        This app uses a Random Forest model trained with SMOTE to detect fraudulent mobile wallet transactions.
        Use scenario buttons or adjust inputs to test predictions.
    """)
    st.markdown("**Developed by**: Moizzah Rehman")
    st.header("Model Performance (Test Set)")
    st.markdown("""
        **Test Set Metrics**:
        - **Accuracy**: 99.96%
        - **Precision (Fraud)**: 89%
        - **Recall (Fraud)**: 85%
        - **F1-Score (Fraud)**: 87%
        *Based on historical test data. Actual performance may vary.*
    """)

# Main title and description
st.title("📱 Mobile Wallet Fraud Detection")
st.markdown("Enter transaction details or use scenario buttons to check if the transaction is fraudulent. All fields are required.")

# Tabs for prediction and performance
tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    # Input form
    with st.container():
        st.subheader("Transaction Details")
        col1, col2 = st.columns(2)

        with col1:
            # UPDATED: Take time in HH:MM AM/PM format instead of seconds
            time_obj = st.time_input(
                "Transaction Time (e.g., 6:30 PM)",
                value=dtime(18, 30),  # Default 6:30 PM
                help="Choose transaction time. It will be converted to seconds from midnight."
            )
            # Convert to seconds
            time_input = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

            amount_input = st.number_input(
                "Transaction Amount (PKR)",
                min_value=0.0,
                step=0.01,
                value=88.35,  # Legitimate mean
                help="Enter the transaction amount. Fraud mean: 123 PKR. Legitimate mean: 88.35 PKR."
            )

        with col2:
            operator_input = st.selectbox(
                "Operator",
                ["EasyPaisa", "JazzCash", "SadaPay"],
                index=0,
                help="JazzCash (50%) and SadaPay (43%) are common in fraud. Legitimate: ~25% each."
            )
            region_input = st.selectbox(
                "Region",
                ["Balochistan", "KPK", "Punjab", "Sindh"],
                index=0,
                help="Sindh (46%) and Punjab (45%) are common in fraud. Legitimate: ~25% each."
            )
            txn_type_input = st.selectbox(
                "Transaction Type",
                ["Bank Transfer", "Bill Payment", "Mobile Load", "Send Money"],
                index=0,
                help="Send Money (48%) is common in fraud. Legitimate: ~25% each."
            )

        # V1–V28 toggle and inputs
        st.subheader("Additional Features (V1–V28)")
        scenario_toggle = st.selectbox(
            "Select V1–V28 Scenario",
            ["Legitimate Means", "Fraudulent Means"],
            index=0,
            help="Choose legitimate or fraudulent means for V1–V28."
        )
        v_fraud_means = [-4.57, 3.67, -7.04, 4.64, -3.09, -1.45, -5.59, 1.39, -2.68, -5.72, 3.80, -6.21, -0.06, -7.18, 0.43, -4.69, -7.54, -2.84, 1.10, 0.75, 0.74, 0.01, -0.04, -0.06, -0.04, 0.09, 0.17, 0.08]
        v_fraud_std = [6.15, 3.70, 6.45, 2.58, 4.92, 1.60, 6.36, 4.85, 3.11, 5.28, 3.09, 5.59, 2.22, 5.24, 1.37, 4.13, 7.03, 3.28, 2.08, 2.22, 2.42, 1.68, 2.00, 1.25, 1.18, 1.05, 1.23, 0.83]
        v_legit_means = [0.01, -0.01, 0.01, -0.01, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00, -0.01, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        v_inputs = []
        for i, (v_fraud, v_legit, v_std) in enumerate(zip(v_fraud_means, v_legit_means, v_fraud_std), 1):
            default_value = v_legit if scenario_toggle == "Legitimate Means" else v_fraud
            v_input = st.number_input(
                f"V{i}",
                min_value=-30.0,
                max_value=30.0,
                value=default_value,
                step=0.1,
                help=f"Fraud mean: {v_fraud:.2f}, std: {v_std:.2f}. Legitimate mean: {v_legit:.2f}."
            )
            v_inputs.append(v_input)

    # Predict button
    if st.button("🔍 Predict", use_container_width=True):
        if model is None or scaler is None:
            st.warning("Cannot make predictions. Model or scaler not loaded.")
        else:
            # Input validation
            if time_input < 0 or amount_input < 0:
                st.error("Time and Amount must be non-negative.")
            else:
                with st.spinner("Analyzing transaction..."):
                    time.sleep(1)
                    # One-hot encoding (match training)
                    operator_encoded = [
                        1 if operator_input == "JazzCash" else 0,
                        1 if operator_input == "SadaPay" else 0,
                    ]
                    region_encoded = [
                        1 if region_input == "KPK" else 0,
                        1 if region_input == "Punjab" else 0,
                        1 if region_input == "Sindh" else 0,
                    ]
                    txn_type_encoded = [
                        1 if txn_type_input == "Bill Payment" else 0,
                        1 if txn_type_input == "Mobile Load" else 0,
                        1 if txn_type_input == "Send Money" else 0,
                    ]

                    # Construct feature vector (38 features in order)
                    feature_vector = (
                        [time_input] + v_inputs + [amount_input] +
                        operator_encoded + region_encoded + txn_type_encoded
                    )
                    feature_array = np.array(feature_vector).reshape(1, -1)

                    # Scale and predict
                    try:
                        scaled_input = scaler.transform(feature_array)
                        prediction = model.predict(scaled_input)

                        # Display results
                        st.subheader("Prediction Result")
                        if prediction[0] == 1:
                            st.error("⚠️ Fraudulent Transaction Detected!")
                        else:
                            st.success("✅ Legitimate Transaction")

                        # Feedback form
                        st.subheader("Feedback")
                        feedback = st.radio("Was this prediction correct?", ("Yes", "No", "Not Sure"))
                        if st.button("Submit Feedback"):
                            with open("feedback.csv", "a") as f:
                                f.write(f"{prediction[0]},{feedback}\n")
                            st.success("Thank you for your feedback!")

                        # Display dynamic accuracy from feedback
                        try:
                            feedback_df = pd.read_csv("feedback.csv", names=["prediction", "feedback"])
                            correct = len(feedback_df[feedback_df["feedback"] == "Yes"])
                            total = len(feedback_df[feedback_df["feedback"].isin(["Yes", "No"])])
                            if total > 0:
                                accuracy = correct / total
                                st.markdown(f"**User Feedback Accuracy**: {accuracy:.2%} (based on {total} responses)")
                            else:
                                st.markdown("No feedback data available yet.")
                        except FileNotFoundError:
                            st.markdown("No feedback data available yet.")

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.header("Model Performance")
    st.markdown("""
        **Test Set Metrics**:
        - **Accuracy**: 99.96%
        - **Precision (Fraud)**: 89%
        - **Recall (Fraud)**: 85%
        - **F1-Score (Fraud)**: 87%
        *Based on historical test data. Actual performance may vary.*
    """)
