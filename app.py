import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Wine Type Classifier",
    page_icon="🍷",
    layout="centered"
)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

DATA_FILE = "wine_quality_merged (1)(1).csv"


@st.cache_resource
def train_model():
    """Train and cache the model when the app first starts."""
    df = pd.read_csv(DATA_FILE)

    required = FEATURES + ["type"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")

    df = df[required].dropna().copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df = df[df["type"].isin(["red", "white"])].copy()
    df["type"] = df["type"].map({"red": 0, "white": 1})

    X = df[FEATURES]
    y = df["type"]

    # Hold out test data for an evaluation estimate.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    candidates = []
    for penalty in ["l1", "l2"]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42
                ))
            ])
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            candidates.append({
                "penalty": penalty,
                "C": C,
                "validation_accuracy": accuracy_score(y_val, val_pred)
            })

    best = max(candidates, key=lambda item: item["validation_accuracy"])

    # Evaluate selected settings on the untouched test split.
    evaluation_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty=best["penalty"],
            C=best["C"],
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])
    evaluation_model.fit(X_train, y_train)
    test_pred = evaluation_model.predict(X_test)
    test_prob = evaluation_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Test accuracy": accuracy_score(y_test, test_pred),
        "Test precision (White)": precision_score(y_test, test_pred, zero_division=0),
        "Test recall (White)": recall_score(y_test, test_pred, zero_division=0),
        "Test F1 (White)": f1_score(y_test, test_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, test_prob),
        "Selected regularization": best["penalty"].upper(),
        "Selected C": best["C"],
        "Validation accuracy": best["validation_accuracy"],
        "Rows used": len(df),
    }

    # For the deployed predictor, refit selected settings on all available labelled rows.
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty=best["penalty"],
            C=best["C"],
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])
    final_model.fit(X, y)

    return final_model, metrics


st.title("🍷 Wine Type Classification")
st.write(
    "Enter the wine's physicochemical measurements to predict whether it is "
    "**Red Wine** or **White Wine**."
)
st.caption("Model: Regularized Logistic Regression with L1/L2 selection.")

try:
    model, metrics = train_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Make sure the CSV dataset is in the same "
        "folder as app.py and that its filename matches the README instructions."
    )
    st.exception(exc)
    st.stop()

with st.form("wine_input_form"):
    st.subheader("Enter the 11 wine measurements")

    col1, col2 = st.columns(2)
    with col1:
        fixed_acidity = st.number_input("Fixed acidity", min_value=0.0, value=7.0, step=0.1, format="%.3f")
        volatile_acidity = st.number_input("Volatile acidity", min_value=0.0, value=0.27, step=0.01, format="%.3f")
        citric_acid = st.number_input("Citric acid", min_value=0.0, value=0.36, step=0.01, format="%.3f")
        residual_sugar = st.number_input("Residual sugar", min_value=0.0, value=2.0, step=0.1, format="%.3f")
        chlorides = st.number_input("Chlorides", min_value=0.0, value=0.045, step=0.001, format="%.4f")
        free_sulfur_dioxide = st.number_input("Free sulfur dioxide", min_value=0.0, value=30.0, step=1.0, format="%.3f")
    with col2:
        total_sulfur_dioxide = st.number_input("Total sulfur dioxide", min_value=0.0, value=100.0, step=1.0, format="%.3f")
        density = st.number_input("Density", min_value=0.0, value=0.995, step=0.0001, format="%.5f")
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=3.2, step=0.01, format="%.3f")
        sulphates = st.number_input("Sulphates", min_value=0.0, value=0.5, step=0.01, format="%.3f")
        alcohol = st.number_input("Alcohol (%)", min_value=0.0, value=10.0, step=0.1, format="%.3f")

    submitted = st.form_submit_button("Predict Wine Type", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }])

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    wine_type = "White Wine" if prediction == 1 else "Red Wine"

    st.divider()
    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"🍾 The model predicts: **{wine_type}**")
    else:
        st.success(f"🍷 The model predicts: **{wine_type}**")

    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric("Red wine probability", f"{probabilities[0] * 100:.2f}%")
    prob_col2.metric("White wine probability", f"{probabilities[1] * 100:.2f}%")
    st.progress(float(probabilities[1]))
    st.caption("The probability is the model's estimate, not a guarantee.")

with st.expander("About this project"):
    st.write(
        "This application uses 11 physicochemical features. The target labels are "
        "encoded as red = 0 and white = 1. L1 and L2 regularization are compared "
        "using a validation split. The app reports test-set metrics below."
    )
    st.write("The `quality` column is not used as a prediction input.")

with st.expander("Model evaluation details"):
    st.write("These test metrics are calculated on a held-out test split before the deployed model is refit on all labelled data.")
    st.json(metrics)
