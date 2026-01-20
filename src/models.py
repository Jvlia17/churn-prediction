from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def random_forest_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )),
        ]
    )

def xgboost_model(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                eval_metric="logloss",
                random_state=42
            )),
        ]
    )
