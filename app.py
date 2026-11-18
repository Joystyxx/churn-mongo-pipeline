import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from src.features import add_engineered_features

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Churn Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    # Load the saved preprocessor and Keras model into memory once
    prep = joblib.load("artifacts/preprocessor.joblib")
    model = tf.keras.models.load_model("artifacts/churn_nn.keras")
    return prep, model

def main():
    st.title("📞 Telco Customer Churn Predictor")
    st.markdown("""
    This app uses a trained Keras Neural Network to predict the likelihood of a customer churning based on their account details.
    *Adjust the sliders and dropdowns on the left to see how the probability changes.*
    """)

    # Load models
    try:
        prep, model = load_artifacts()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 8000.0, 1000.0)

    with col2:
        st.subheader("Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # Prediction logic
    st.divider()
    if st.button("Predict Churn Probability", type="primary", use_container_width=True):
        # Build the raw dictionary
        raw_data = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup, 
            "DeviceProtection": device_protection, "TechSupport": tech_support, 
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # Convert to DataFrame & Engineer Features
        df = pd.DataFrame([raw_data])
        df = add_engineered_features(df)

        # Transform and Predict
        X_trans = prep.transform(df)
        prob = float(model.predict(X_trans, verbose=0)[0][0])

        # Display results beautifully
        if prob >= 0.5:
            st.error(f"### ⚠️ High Risk of Churn!\n**Probability:** {prob:.2%}\n\n*Action:* Offer this customer a retention discount or upgrade.")
        else:
            st.success(f"### ✅ Likely to Stay\n**Probability:** {prob:.2%}\n\n*Action:* Customer is stable. Standard engagement is fine.")

if __name__ == "__main__":
    main()