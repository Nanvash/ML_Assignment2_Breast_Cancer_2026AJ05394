# Breast Cancer Classification using Machine Learning

## Problem Statement

This project compares five machine learning classification algorithms for predicting whether a breast tumor is benign or malignant.

## Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)

Instances: 569  
Features: 30  
Target: Diagnosis  
Classes:
- 0 = Benign
- 1 = Malignant

## Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## Model Observations

Logistic Regression: Strong overall performance with the highest AUC.

Decision Tree: Good performance but lower accuracy and MCC than other models.

KNN: Strong performance after feature scaling.

Naive Bayes: Perfect precision but comparatively lower recall.

SVM: Best overall model with the highest accuracy, F1 score and MCC.

## Overall Winner

Support Vector Machine (SVM) is the overall best-performing model for this dataset.