"""
train_and_save.py  —  GreenSight: train once, save the model + real examples.
=============================================================================
Run this ONCE. It trains RISE (CatBoost) on the six features and saves:
  - the trained model
  - a few REAL example sessions taken from your dataset (for the UI dropdown)
After this, the app loads these instead of retraining.

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

df = pd.read_csv(DATA)
y = df["Revenue"].astype(int)

month_names = sorted(df["Month"].astype(str).unique())
month_map = {name: i for i, name in enumerate(month_names)}
inv_month = {i: name for name, i in month_map.items()}
df["Month"] = df["Month"].astype(str).map(month_map)

X = df[FEATURES].astype(float)
keep = IsolationForest(contamination=0.01, random_state=RS).fit_predict(X) == 1
X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
Xtr, ytr = SMOTE(random_state=RS).fit_resample(Xtr, ytr)

model = CatBoostClassifier(learning_rate=0.09, depth=8, iterations=300,
                           verbose=0, random_state=RS)
model.fit(Xtr, ytr)

proba = model.predict_proba(Xte)[:, 1]
metrics = {"accuracy": round(accuracy_score(yte, (proba >= 0.5).astype(int)), 3),
           "roc_auc": round(roc_auc_score(yte, proba), 3)}

# ---- pick REAL example sessions from the data, spanning the three bands ----
Xr = X.copy()
Xr["_p"] = model.predict_proba(X)[:, 1]

def row_to_example(r, label):
    ex = {}
    for f in FEATURES:
        if f == "Month":
            ex[f] = inv_month[int(round(r[f]))]
        elif f in ("TrafficType", "ProductRelated"):
            ex[f] = int(round(r[f]))
        else:
            ex[f] = round(float(r[f]), 3)
    ex["_label"] = label
    return ex

examples = []
hi = Xr[Xr["_p"] >= 0.75].sort_values("_p", ascending=False).head(2)
md = Xr[(Xr["_p"] >= 0.45) & (Xr["_p"] < 0.65)].head(2)
lo = Xr[(Xr["_p"] < 0.35) & (Xr["PageValues"] > 0)].head(2)
if len(lo) < 2:
    lo = pd.concat([lo, Xr[Xr["_p"] < 0.35].head(2)]).head(2)

for grp, name in [(hi, "Likely buyer"), (md, "Possible buyer"), (lo, "Unlikely buyer")]:
    for k, (_, r) in enumerate(grp.iterrows(), 1):
        examples.append(row_to_example(r, f"{name} {k}  (real session)"))

model.save_model("models/rise_catboost.cbm")
joblib.dump(month_map, "models/month_map.joblib")
joblib.dump(FEATURES, "models/features.joblib")
json.dump(metrics, open("models/metrics.json", "w"))
json.dump(examples, open("models/examples.json", "w"))

print("Saved model + examples to models/  |  test metrics:", metrics)
print("Example sessions saved:", len(examples))
for e in examples:
    print("  ", e["_label"], {k: e[k] for k in FEATURES})
