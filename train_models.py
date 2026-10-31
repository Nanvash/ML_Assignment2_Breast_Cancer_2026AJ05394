import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print("Libraries imported successfully!")

data = pd.read_csv("wdbc.data", header=None)

print("Dataset shape:", data.shape)
print(data.head())

columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

data.columns = columns

print(data.head())
print("\nMissing values:", data.isnull().sum().sum())
print("\nDiagnosis counts:")
print(data["diagnosis"].value_counts())

# Convert diagnosis to numeric
data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})

# Features and target
X = data.drop(columns=["id", "diagnosis"])
y = data["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Save test data with target column for Streamlit evaluation
test_data = X_test.copy()
test_data["diagnosis"] = y_test.values

test_data.to_csv("test_data.csv", index=False)

print("test_data.csv created successfully!")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
log_accuracy = accuracy_score(y_test, log_pred)
log_auc = roc_auc_score(y_test, log_prob)
log_precision = precision_score(y_test, log_pred)
log_recall = recall_score(y_test, log_pred)
log_f1 = f1_score(y_test, log_pred)
log_mcc = matthews_corrcoef(y_test, log_pred)

print("\nLogistic Regression Results")
print("Accuracy:", log_accuracy)
print("AUC:", log_auc)
print("Precision:", log_precision)
print("Recall:", log_recall)
print("F1 Score:", log_f1)
print("MCC:", log_mcc)

from sklearn.tree import DecisionTreeClassifier

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_prob)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_mcc = matthews_corrcoef(y_test, dt_pred)

print("\nDecision Tree Results")
print("Accuracy:", dt_accuracy)
print("AUC:", dt_auc)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
print("MCC:", dt_mcc)


from sklearn.neighbors import KNeighborsClassifier

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)
knn_prob = knn_model.predict_proba(X_test_scaled)[:, 1]

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_auc = roc_auc_score(y_test, knn_prob)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_mcc = matthews_corrcoef(y_test, knn_pred)

print("\nKNN Results")
print("Accuracy:", knn_accuracy)
print("AUC:", knn_auc)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print("MCC:", knn_mcc)


from sklearn.naive_bayes import GaussianNB

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
nb_prob = nb_model.predict_proba(X_test)[:, 1]

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_auc = roc_auc_score(y_test, nb_prob)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_mcc = matthews_corrcoef(y_test, nb_pred)

print("\nNaive Bayes Results")
print("Accuracy:", nb_accuracy)
print("AUC:", nb_auc)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("MCC:", nb_mcc)


from sklearn.svm import SVC

# Support Vector Machine
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_prob)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_mcc = matthews_corrcoef(y_test, svm_pred)

print("\nSVM Results")
print("Accuracy:", svm_accuracy)
print("AUC:", svm_auc)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)
print("MCC:", svm_mcc)


import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(log_model, "model/logistic_regression.pkl")
joblib.dump(dt_model, "model/decision_tree.pkl")
joblib.dump(knn_model, "model/knn.pkl")
joblib.dump(nb_model, "model/naive_bayes.pkl")
joblib.dump(svm_model, "model/svm.pkl")

# Save scaler for models that require scaled data
joblib.dump(scaler, "model/scaler.pkl")

print("\nAll models and scaler saved successfully!")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ],
    "Accuracy": [
        log_accuracy, dt_accuracy, knn_accuracy, nb_accuracy, svm_accuracy
    ],
    "AUC": [
        log_auc, dt_auc, knn_auc, nb_auc, svm_auc
    ],
    "Precision": [
        log_precision, dt_precision, knn_precision, nb_precision, svm_precision
    ],
    "Recall": [
        log_recall, dt_recall, knn_recall, nb_recall, svm_recall
    ],
    "F1": [
        log_f1, dt_f1, knn_f1, nb_f1, svm_f1
    ],
    "MCC": [
        log_mcc, dt_mcc, knn_mcc, nb_mcc, svm_mcc
    ]
})

print("\nModel Comparison")
print(results.round(4))

results.to_csv("model_comparison.csv", index=False)

print("\nmodel_comparison.csv created successfully!")

joblib.dump(list(X.columns), "model/feature_names.pkl")
print("Feature names saved successfully!")