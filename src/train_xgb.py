import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


"""Phase 3: Train XGBoost baseline with GridSearchCV."""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.config import get_db
from src.features import add_engineered_features, build_preprocessor

def main():
    print("Loading data from MongoDB...")
    db = get_db()
    df = pd.DataFrame(list(db.raw_customers.find({}, {"_id": 0})))
    
    print("Engineering features...")
    df = add_engineered_features(df)
    
    # Encode target: 'Yes' -> 1, 'No' -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    
    # 80/20 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build the full pipeline (preprocessor + model)
    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
    ])
    
    # Hyperparameter grid as requested in the brief
    param_grid = {
        "clf__max_depth": [3, 5],
        "clf__n_estimators": [100, 200],
    }
    
    print("Starting GridSearchCV (this will take a minute)...")
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    
    # Evaluate
    best_model = grid.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n=== XGBoost Results ===")
    print(f"Best Params: {grid.best_params_}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()