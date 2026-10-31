import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

st.title("Breast Cancer Classification Model Evaluation")
st.write("Upload the test dataset and select a machine learning model.")

# Load preprocessing information
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "SVM": "model/svm.pkl"
}

selected_model = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(data.head())

    if "diagnosis" not in data.columns:
        st.error("The uploaded CSV must contain the diagnosis target column.")

    else:
        X_test = data[feature_names]
        y_test = data["diagnosis"]

        model = joblib.load(models[selected_model])

        # Scale data for models trained using StandardScaler
        if selected_model in [
            "Logistic Regression",
            "KNN",
            "SVM"
        ]:
            X_input = scaler.transform(X_test)
        else:
            X_input = X_test

        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        mcc = matthews_corrcoef(y_test, predictions)

        st.subheader("Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["Benign", "Malignant"],
            ax=ax
        )

        st.pyplot(fig)