# Telco Customer Churn Prediction – Machine Learning Pipeline

End-to-end machine learning project focused on predicting customer churn using a public Telco Customer Churn dataset from Kaggle.

The project demonstrates a complete ML workflow, including data preprocessing, feature transformation, model development, evaluation, and comparison of multiple classification algorithms.

---

## 🔷 Project Overview

Customer churn prediction is a common business problem in subscription-based industries, where identifying customers at risk of leaving can help companies improve retention strategies.

The goal of this project is to build a classification model that predicts whether a customer is likely to churn based on demographic information, account details, and subscribed services.

The project combines:

- Python for data processing and machine learning
- scikit-learn pipelines for reproducible preprocessing and modeling
- XGBoost for gradient boosting classification
- Model evaluation using classification metrics and business-focused analysis

---

## 🎯 Business Problem

Customer retention is often more cost-effective than acquiring new customers.

The objective of this project is to identify customers with a high probability of churn, allowing businesses to target retention campaigns more effectively.

Since missing a potential churn customer can result in lost revenue, recall was considered an important evaluation metric.

---

## 🗄️ Dataset

**Source:** Kaggle – Telco Customer Churn dataset

The dataset contains customer demographic, account, and service information.

### Target Variable

- `Churn`
  - Yes → customer left the company
  - No → customer stayed

### Features include:

- Customer demographics
- Contract information
- Internet and phone services
- Monthly charges
- Total charges
- Customer tenure

---

## ⚙️ Data Processing Pipeline

The project follows a modular ML pipeline:

### Data Loading
- Automated dataset loading using `kagglehub`

### Data Cleaning
- Converted `TotalCharges` into numeric format
- Removed missing values
- Encoded target variable (`Yes` → 1, `No` → 0)

### Feature Processing
- Numerical feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Combined preprocessing and modeling using scikit-learn `Pipeline`

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

### Logistic Regression
Used as a baseline model with class balancing to improve detection of minority churn cases.

### Random Forest
Tree-based ensemble model used to capture nonlinear customer behavior patterns.

### XGBoost
Gradient boosting model optimized for tabular classification problems.

---

## ⚖️ Handling Class Imbalance

The dataset contains fewer churn cases than retained customers.

To improve churn detection:

- Logistic Regression uses `class_weight="balanced"`
- Random Forest uses `class_weight="balanced"`

This helps the models focus more on identifying customers likely to leave.

---

## 📊 Model Evaluation

Models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Final Results

<img width="508" height="106" alt="wyniki" src="https://github.com/user-attachments/assets/81896b1c-7ac1-49db-b913-d9f2b6bb14f0" />


---

## 📈 Key Insights

- Logistic Regression achieved the highest recall, identifying approximately 80% of actual churn customers.
- XGBoost achieved the highest ROC-AUC score, showing strong overall classification performance.
- Tree-based models did not outperform Logistic Regression on this dataset, highlighting that more complex models do not always provide better results.
- The model evaluation demonstrates the trade-off between identifying more churn customers (recall) and reducing false positives (precision).
