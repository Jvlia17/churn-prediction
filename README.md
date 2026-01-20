# Telco Customer Churn Prediction

This project presents an end-to-end machine learning pipeline for predicting customer churn using a public Telco dataset from Kaggle.

The goal is to demonstrate practical data analysis and machine learning skills, including data preprocessing, feature engineering, and model comparison.

---

## Dataset
- **Source:** Kaggle – Telco Customer Churn (public dataset)
- **Target variable:** Churn (Yes / No)

---

## Project Structure
- `data_loader.py` – downloading and loading the dataset  
- `preprocessing.py` – data cleaning and target encoding  
- `features.py` – feature preprocessing (scaling and encoding)  
- `models.py` – machine learning models  
- `train.py` – model training and evaluation pipeline  

---

## Techniques Used
- Data cleaning and preprocessing
- Feature encoding and scaling
- Train / test split with stratification
- Model evaluation using classification metrics

---

## Machine Learning Models
- Random Forest
- XGBoost

---

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Git

---

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
