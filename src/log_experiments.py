"""Phase 3/4: Log evaluation metrics to MongoDB experiments collection."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import tensorflow as tf

from src.config import get_db
from src.features import add_engineered_features, build_preprocessor

def get_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

def main():
    db = get_db()
    exp_coll = db["experiments"]
    exp_coll.drop()  # Clean slate for the demo
    
    print("Loading data...")
    df = pd.DataFrame(list(db["raw_customers"].find({}, {"_id": 0})))
    df = add_engineered_features(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    prep = build_preprocessor()
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    print("Evaluating XGBoost...")
    xgb = XGBClassifier(max_depth=3, n_estimators=100, eval_metric="logloss", random_state=42)
    xgb.fit(X_train_t, y_train)
    xgb_pred = xgb.predict(X_test_t)
    xgb_prob = xgb.predict_proba(X_test_t)[:, 1]
    xgb_metrics = get_metrics(y_test, xgb_pred, xgb_prob)

    print("Evaluating Keras NN...")
    nn = tf.keras.models.load_model("artifacts/churn_nn.keras")
    nn_prob = nn.predict(X_test_t, verbose=0).ravel()
    nn_pred = (nn_prob >= 0.5).astype(int)
    nn_metrics = get_metrics(y_test, nn_pred, nn_prob)

    # Build the documents
    xgb_doc = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "model_type": "XGBoost",
        "hyperparameters": {"max_depth": 3, "n_estimators": 100},
        "metrics": xgb_metrics,
        "notes": "Tuned via GridSearchCV. Baseline conservative model."
    }
    
    nn_doc = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "model_type": "TensorFlow_Keras",
        "hyperparameters": {"layers": [32, 16, 1], "dropout": 0.2, "epochs": 20, "class_weight": True},
        "metrics": nn_metrics,
        "notes": "Class weights used to boost minority class recall."
    }
    
    exp_coll.insert_many([xgb_doc, nn_doc])
    print("\nSuccessfully inserted 2 experiment logs into MongoDB!")
    
    # Verify
    docs = list(exp_coll.find({}, {"_id": 0}))
    for d in docs:
        print(f"\n{d['model_type']} -> ROC-AUC: {d['metrics']['roc_auc']:.4f}, F1: {d['metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()