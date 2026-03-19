import numpy as np
import pandas as pd

from zsxgboost import ZeroShotXGBClassifier
from lazyqsar.agnostic import LazyBinaryClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

bench_group = "tdc"
bench_task = "carcinogens_lagunin"
descriptor = "morgan"

data_path = f"data/{bench_group}/classification/{bench_task}"

print("Loading data...")
X = np.load(f"{data_path}/morgan.npy")
try:
    y = pd.read_csv(f"{data_path}/data.csv")["value"].values
except:
    y = pd.read_csv(f"{data_path}/data.csv")["activity"].values
print("Data loaded.")

print("Training Model...")
model = ZeroShotXGBClassifier(verbose=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def select_model(model_name):
    if model_name == "xgboost":
        return ZeroShotXGBClassifier(verbose=True)
    elif model_name == "random_forest":
        return RandomForestClassifier(verbose=True)
    elif model_name == "lazyqsar":
        return LazyBinaryClassifier()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_evaluate(model_name):
    model = select_model(model_name)
    start_time = time.time()
    model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_hat)
    print(f"{model_name} AUROC: {auroc:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    return auroc

aurocs = []
for model_name in ["random_forest", "xgboost"]:
    aurocs.append(train_and_evaluate(model_name))

for model_name, auroc in zip(["random_forest", "xgboost"], aurocs):
    print(f"{model_name} AUROC: {auroc:.4f}")