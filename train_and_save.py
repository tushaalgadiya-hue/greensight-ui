"""
train_and_save.py  —  GreenSight: train once, save the model.
=============================================================
Run this ONCE. It trains the RISE predictor (CatBoost) on the six selected
features and saves everything the Streamlit app needs into  models/.
After this, the app loads the saved model instead of retraining.

RUN:
    python train_and_save.py
"""
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

RS = 42
DATA = "online_shoppers_intention.csv"
FEATURES = ["TrafficType", "Administrative_Duration", "Month",
            "ProductRelated", "BounceRates", "PageValues"]
os.makedirs("models", exist_ok=True)

# ---- load ----
df = pd.read_csv(DATA)
y = df["Revenue"].astype(int)

# ---- encode Month exactly like LabelEncoder (sorted unique -> 0..k-1) ----
month_names = sorted(df["Month"].astype(str).unique())
month_map = {name: i for i, name in enumerate(month_names)}
df["Month"] = df["Month"].astype(str).map(month_map)

# ---- keep the six features ----
X = df[FEATURES].astype(float)

# ---- remove the most unusual 1% of sessions ----
keep = IsolationForest(contamination=0.01, random_state=RS).fit_predict(X) == 1
X, y = X[keep], y[keep]

# ---- split, then balance the TRAINING data only ----
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
Xtr, ytr = SMOTE(random_state=RS).fit_resample(Xtr, ytr)

# ---- train RISE (CatBoost) ----
model = CatBoostClassifier(learning_rate=0.09, depth=8, iterations=300,
                           verbose=0, random_state=RS)
model.fit(Xtr, ytr)

# ---- quick quality check on the untouched test set ----
proba = model.predict_proba(Xte)[:, 1]
metrics = {"accuracy": round(accuracy_score(yte, (proba >= 0.5).astype(int)), 3),
           "roc_auc": round(roc_auc_score(yte, proba), 3)}

# ---- save everything the app needs ----
model.save_model("models/rise_catboost.cbm")
joblib.dump(month_map, "models/month_map.joblib")
joblib.dump(FEATURES, "models/features.joblib")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Saved model to models/  |  test metrics:", metrics)
print("Month options:", list(month_map.keys()))
