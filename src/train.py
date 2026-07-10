from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import pandas as pd

from data_loader import load_telco_data
from preprocessing import clean_data
from features import build_preprocessor
from models import (
    logistic_regression_model,
    random_forest_model,
    xgboost_model
)


def evaluate_model(model, name, X_test, y_test):
    """
    Evaluate classification model and return metrics.
    """

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        preds,
        output_dict=True
    )

    cm = confusion_matrix(
        y_test,
        preds
    )

    print(f"\n{name} results:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(cm)

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probs), 3)
    )

    results = {
        "Model": name,
        "Precision": round(report["1"]["precision"], 3),
        "Recall": round(report["1"]["recall"], 3),
        "F1-score": round(report["1"]["f1-score"], 3),
        "ROC-AUC": round(
            roc_auc_score(y_test, probs),
            3
        )
    }

    return results



def main():

    # Load and clean data
    df = load_telco_data()
    df = clean_data(df)

    # Features and target
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    preprocessor = build_preprocessor(X)

    results = []


    # Logistic Regression
    lr = logistic_regression_model(preprocessor)
    lr.fit(X_train, y_train)

    results.append(
        evaluate_model(
            lr,
            "Logistic Regression",
            X_test,
            y_test
        )
    )


    # Random Forest
    rf = random_forest_model(preprocessor)
    rf.fit(X_train, y_train)

    results.append(
        evaluate_model(
            rf,
            "Random Forest",
            X_test,
            y_test
        )
    )


    # XGBoost
    xgb = xgboost_model(preprocessor)
    xgb.fit(X_train, y_train)

    results.append(
        evaluate_model(
            xgb,
            "XGBoost",
            X_test,
            y_test
        )
    )


    # Model comparison table
    results_df = pd.DataFrame(results)

    print("\n============================")
    print("Model Comparison:")
    print("============================")

    print(results_df)


if __name__ == "__main__":
    main()