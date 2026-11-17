"""Phase 3: Train TensorFlow/Keras Neural Network."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import get_db
from src.features import add_engineered_features, build_preprocessor


def build_model(input_dim: int) -> keras.Model:
    """Exact architecture from the brief, via Functional API."""
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Dense(16, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="churn_mlp")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    print("Loading data from MongoDB...")
    db = get_db()
    df = pd.DataFrame(list(db.raw_customers.find({}, {"_id": 0})))

    print("Engineering features...")
    df = add_engineered_features(df)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Fitting preprocessor...")
    prep = build_preprocessor()
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    input_dim = X_train_t.shape[1]
    print(f"Input dimension: {input_dim}")

    # Class weights (justified in chat: NN-side boost to handle imbalance)
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"Class weights: {class_weight}")

    model = build_model(input_dim)
    model.summary()

    print("Training for 20 epochs...")
    history = model.fit(
        X_train_t, y_train,
        validation_split=0.15,
        epochs=20,
        batch_size=32,
        class_weight=class_weight,
        verbose=2,
    )

    y_prob = model.predict(X_test_t, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Keras NN Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model for Phase 5
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/churn_nn.keras")
    print("\nSaved model to artifacts/churn_nn.keras")


if __name__ == "__main__":
    main()