import kagglehub
import pandas as pd
import os

def load_telco_data():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    return df
