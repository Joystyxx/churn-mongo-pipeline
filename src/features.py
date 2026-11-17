"""Phase 2: Scikit-Learn ColumnTransformer and feature engineering."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# The raw columns that are strictly numerical
NUMERIC_FEATURES = [
    "tenure", 
    "MonthlyCharges", 
    "TotalCharges", 
    "SeniorCitizen",
    # Our engineered features will be appended here dynamically
]

# The raw columns that are categorical
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Columns used to calculate TotalServices
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the two engineered features requested in the brief."""
    df = df.copy()
    # TotalServices: count of 'Yes' across all service columns
    df["TotalServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    # ChargeRatio: MonthlyCharges divided by tenure (+1 to avoid div/0)
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df

def build_preprocessor() -> ColumnTransformer:
    """Builds the Scikit-Learn ColumnTransformer."""
    # We add our new engineered features to the numerical list
    num_cols = NUMERIC_FEATURES + ["TotalServices", "ChargeRatio"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop" # Drop customerID and Churn
    )
    
    return preprocessor