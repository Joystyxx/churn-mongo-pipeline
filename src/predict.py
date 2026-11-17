"""Phase 5: Inference CLI for a single customer."""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import tensorflow as tf
from src.features import add_engineered_features


def main():
    parser = argparse.ArgumentParser(description="Predict churn probability")
    parser.add_argument("--data", required=True, type=str, help="JSON string of customer data")
    args = parser.parse_args()

    try:
        raw_dict = json.loads(args.data)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string.")
        sys.exit(1)

    df = pd.DataFrame([raw_dict])
    df = add_engineered_features(df)

    try:
        prep = joblib.load("artifacts/preprocessor.joblib")
        model = tf.keras.models.load_model("artifacts/churn_nn.keras")
    except FileNotFoundError:
        print("Error: Model artifacts not found. Run the training scripts first.")
        sys.exit(1)

    try:
        X_trans = prep.transform(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Ensure your JSON contains all required raw fields.")
        sys.exit(1)

    prob = float(model.predict(X_trans, verbose=0)[0][0])

    print("\n--- Churn Prediction ---")
    print(f"Customer ID: {raw_dict.get('customerID', 'Unknown')}")
    print(f"Churn Probability: {prob:.2%}")
    print(f"Prediction: {'WILL CHURN' if prob >= 0.5 else 'WILL NOT CHURN'}")


if __name__ == "__main__":
    main()