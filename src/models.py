from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def logistic_regression_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )
def random_forest_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )

def xgboost_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )),
        ]
    )
