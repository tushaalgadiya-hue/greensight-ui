# GreenSight — Streamlit UI (how to run)

This UI demonstrates the GreenSight pipeline live:
**RISE** (predict) → **SHINE** (explain with SHAP) → **SAGE** (recommend with an LLM),
plus the SDG 12 carbon saving. You enter six features, click **Run**, and the
three modules run in the backend and show the result in your browser.

---

## Files
- `train_and_save.py` — trains the model **once** and saves it to `models/`.
- `app.py` — the Streamlit UI (loads the saved model, no retraining).
- `requirements.txt` — Python packages.
- `online_shoppers_intention.csv` — the dataset (put your copy here).

---

## One-time setup

1. Put `online_shoppers_intention.csv` in this folder (same folder as the scripts).

2. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install **Ollama** (this is what powers SAGE's recommendation):
   - Download from https://ollama.com and install it.
   - Pull a small model:
     ```
     ollama pull llama3.2
     ```
   - Ollama runs in the background automatically after install.
   - (If you skip this, the app still works — SAGE falls back to a clear
     rule-based recommendation, so a demo never breaks.)

---

## Step 1 — train once and save the model
```
python train_and_save.py
```
This creates a `models/` folder with the trained CatBoost model and the
helper files the app needs. You only run this **once** (or again if you
change the data or settings). After this, the app never retrains.

## Step 2 — launch the UI
```
streamlit run app.py
```
A browser tab opens automatically at **http://localhost:8501**.
Enter the six features (or pick an example from the dropdown), click
**Run GreenSight**, and you'll see:
- **RISE** — the purchase-intention probability and its band (low/medium/high)
- **SHINE** — a SHAP chart showing which features pushed the score up or down
- **SAGE** — a recommended action for the retailer and the carbon saving

To stop the app, press `Ctrl + C` in the terminal.

---

## Notes
- **Change the LLM model:** edit `OLLAMA_MODEL = "llama3.2"` at the top of
  `app.py` to any model you have pulled.
- **Model fidelity:** the UI trains CatBoost on the six selected features with
  outlier removal and SMOTE, exactly as in the paper. It does not standardise
  the six inputs, because CatBoost's tree splits are scale-invariant (so the
  model is the same) and it keeps the SHAP explanation in real, readable units
  for the demo. On the full dataset the model reaches the same performance
  reported in the paper (ROC AUC ≈ 0.917).
