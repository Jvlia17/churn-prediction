from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_telco_data
from preprocessing import clean_data
from features import build_preprocessor
from models import random_forest_model, xgboost_model


def main():

    df = load_telco_data()
    df = clean_data(df)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    # Random Forest
    rf = random_forest_model(preprocessor)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("Random Forest results:")
    print(classification_report(y_test, rf_preds))

    # XGBoost
    xgb = xgboost_model(preprocessor)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    print("XGBoost results:")
    print(classification_report(y_test, xgb_preds))

if __name__ == "__main__":
    main()
