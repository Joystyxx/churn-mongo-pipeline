from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf
from src.features import add_engineered_features

app = FastAPI(title="Telco Churn Prediction API")

# Load artifacts globally so they stay in memory (efficient)
try:
    prep = joblib.load("artifacts/preprocessor.joblib")
    model = tf.keras.models.load_model("artifacts/churn_nn.keras")
except Exception as e:
    print(f"Error loading models: {e}")

# Define the exact input schema using Pydantic (Data Validation)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert Pydantic model to dictionary
    raw_dict = data.model_dump()
    
    # Convert to DataFrame for the pipeline
    df = pd.DataFrame([raw_dict])
    
    # Apply feature engineering
    df = add_engineered_features(df)
    
    # Transform and predict
    try:
        X_trans = prep.transform(df)
        prob = float(model.predict(X_trans, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "churn_probability": round(prob, 4),
        "prediction": "WILL CHURN" if prob >= 0.5 else "WILL NOT CHURN"
    }

@app.get("/")
def health_check():
    return {"status": "API is running"}