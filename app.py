"""
GreenSight — Streamlit UI
=========================
Enter the six features (or load an example), click Run, and watch
RISE -> SHINE -> SAGE work, plus the SDG 12 carbon saving.

RUN LOCALLY (after train_and_save.py has been run once):
    python -m streamlit run app.py

SAGE (the recommendation) uses an LLM. It picks the first that is available:
  1) a hosted LLM (Groq) if a GROQ_API_KEY is set  -> works on the website
  2) a local Ollama model if one is running         -> works on your laptop
  3) a clear rule-based recommendation otherwise     -> always works
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier
import shap

# ----------------------------------------------------------------- config
OLLAMA_MODEL = "llama3.2"                 # used only when running on your laptop
GROQ_MODEL   = "llama-3.1-8b-instant"     # used on the website (free, fast)
CARBON_PER_RETURN = 2.4                   # kg CO2 per prevented return
BAND_N = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
COLORS = {"dark": "#0E3B2E", "green": "#2E8B6B", "blue": "#1C7293",
          "amber": "#E0A93B", "pos": "#2E8B6B", "neg": "#C0564B"}

st.set_page_config(page_title="GreenSight", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1200px;}}
.gs-hero {{background: linear-gradient(135deg,{COLORS['dark']} 0%,{COLORS['green']} 100%);
  padding: 22px 28px; border-radius: 16px; color: white; margin-bottom: 6px;}}
.gs-hero h1 {{margin:0; font-size: 2.0rem; font-weight: 800; letter-spacing:.2px;}}
.gs-hero p {{margin:.25rem 0 0 0; opacity:.9; font-size:1.0rem;}}
.gs-step {{display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700;
  font-size:.85rem; color:white; margin-right:8px;}}
.gs-tag {{font-size:.78rem; font-weight:700; letter-spacing:.3px; text-transform:uppercase; opacity:.7;}}
.gs-band {{display:inline-block; padding:8px 18px; border-radius:12px; font-size:1.5rem;
  font-weight:800; color:white;}}
.gs-reco {{background:#FBF7EC; border:1px solid #EAD9AD; border-radius:12px; padding:16px 18px;
  font-size:1.02rem; line-height:1.5; color:#3a3326;}}
.stButton>button {{background:{COLORS['green']}; color:white; font-weight:700; border:0;
  border-radius:12px; padding:.6rem 1.2rem; font-size:1.05rem;}}
.stButton>button:hover {{background:{COLORS['dark']};}}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- load once
@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier(); model.load_model("models/rise_catboost.cbm")
    month_map = joblib.load("models/month_map.joblib")
    features = joblib.load("models/features.joblib")
    explainer = shap.TreeExplainer(model)
    try:
        metrics = json.load(open("models/metrics.json"))
    except Exception:
        metrics = {}
    return model, month_map, features, explainer, metrics

try:
    model, MONTH_MAP, FEATURES, EXPLAINER, METRICS = load_artifacts()
except Exception as e:
    st.error("Could not load the model. Make sure the 'models' folder is present "
             "(run: python train_and_save.py). Details: %s" % e)
    st.stop()

# ----------------------------------------------------------------- helpers
def rise_predict(row_values):
    X = pd.DataFrame([row_values], columns=FEATURES).astype(float)
    prob = float(model.predict_proba(X)[0, 1])
    band = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")
    return X, prob, band

def shine_explain(X):
    sv = np.array(EXPLAINER.shap_values(X))[0]
    base = float(EXPLAINER.expected_value)
    contribs = sorted(zip(FEATURES, [float(v) for v in sv]),
                      key=lambda t: abs(t[1]), reverse=True)
    return base, contribs

def _get_groq_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")

def _build_prompt(prob, band, contribs):
    top = ", ".join(f for f, _ in contribs[:2])
    return ("You are SAGE, a shop's assistant. "
            f"A visitor's chance of buying is {prob*100:.0f}% ({band}). "
            f"Main reasons: {top}. "
            "Tell the shop ONE action to take, in ONE or TWO very short sentences. "
            "Use simple everyday words a non-expert understands. Be direct and practical. "
            "No jargon, no long explanation, no lists. "
            "Do not suggest a discount if the visitor is already likely to buy.")

def sage_recommend(prob, band, contribs):
    prompt = _build_prompt(prob, band, contribs)
    # 1) hosted LLM (works on the website)
    key = _get_groq_key()
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model=GROQ_MODEL, temperature=0.3, max_tokens=70,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content.strip(), "LLM (Groq)"
        except Exception:
            pass
    # 2) local Ollama (works on your laptop)
    try:
        import ollama
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip(), "LLM (Ollama, local)"
    except Exception:
        pass
    # 3) always-available rule-based recommendation
    fallback = {
        "HIGH": "This shopper is very likely to buy. Don't give a discount. Show related products and offer free shipping to grow the order.",
        "MEDIUM": "This shopper might buy. Give a small nudge like free shipping and clear delivery info. No discount needed.",
        "LOW": "This shopper probably won't buy. Keep it simple and cheap: clear product info and an easy checkout. Don't spend on discounts.",
    }
    return fallback[band], "rule-based (no LLM connected)"

def shap_figure(base, contribs):
    feats = [f for f, _ in contribs][::-1]
    vals = [v for _, v in contribs][::-1]
    bar_colors = [COLORS["pos"] if v >= 0 else COLORS["neg"] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=bar_colors,
                           text=[f"{v:+.2f}" for v in vals], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=10, r=30, t=10, b=10),
                      xaxis_title="push on the score  (green = towards purchase)",
                      plot_bgcolor="white", font=dict(size=13))
    fig.add_vline(x=0, line_width=1, line_color="#999")
    return fig

# ----------------------------------------------------------------- header
st.markdown(f"""
<div class="gs-hero">
  <h1>🌱 GreenSight</h1>
  <p>Explainable, sustainability-aware prediction of online purchase intention</p>
  <div style="margin-top:12px">
    <span class="gs-step" style="background:{COLORS['green']}">1 · RISE — predict</span>
    <span class="gs-step" style="background:{COLORS['blue']}">2 · SHINE — explain</span>
    <span class="gs-step" style="background:{COLORS['amber']};color:#3a3326">3 · SAGE — recommend</span>
  </div>
</div>
""", unsafe_allow_html=True)

if METRICS:
    st.caption(f"Model: CatBoost  ·  test accuracy {METRICS.get('accuracy','–')}  ·  "
               f"test ROC AUC {METRICS.get('roc_auc','–')}")

# ----------------------------------------------------------------- inputs
st.subheader("Enter the six session features")

presets = {
    "— type my own —": None,
    "High-intent example": dict(TrafficType=2, Administrative_Duration=180.0, Month="Nov",
                                ProductRelated=90, BounceRates=0.0, PageValues=95.0),
    "Medium-intent example": dict(TrafficType=3, Administrative_Duration=40.0, Month="May",
                                  ProductRelated=18, BounceRates=0.02, PageValues=6.0),
    "Low-intent example": dict(TrafficType=1, Administrative_Duration=0.0, Month="Feb",
                               ProductRelated=2, BounceRates=0.20, PageValues=0.0),
}
choice = st.selectbox("Load an example, or choose “type my own” to enter values",
                      list(presets.keys()))
p = presets[choice] or dict(TrafficType=2, Administrative_Duration=40.0, Month="May",
                            ProductRelated=20, BounceRates=0.02, PageValues=10.0)
months = list(MONTH_MAP.keys())

c1, c2, c3 = st.columns(3)
with c1:
    page_values = st.number_input("PageValues", 0.0, 400.0, float(p["PageValues"]), 0.5,
                                  help="Value of the pages visited — the strongest signal.")
    product_related = st.number_input("ProductRelated", 0, 800, int(p["ProductRelated"]), 1,
                                       help="Number of product pages viewed.")
with c2:
    bounce = st.number_input("BounceRates", 0.0, 1.0, float(p["BounceRates"]), 0.001, format="%.3f",
                             help="Share of visits that leave immediately.")
    admin_dur = st.number_input("Administrative_Duration", 0.0, 4000.0,
                                float(p["Administrative_Duration"]), 1.0,
                                help="Seconds spent on account/admin pages.")
with c3:
    month = st.selectbox("Month", months,
                         index=months.index(p["Month"]) if p["Month"] in months else 0)
    traffic = st.number_input("TrafficType", 1, 20, int(p["TrafficType"]), 1,
                              help="How the visitor reached the site (coded 1–20).")

run = st.button("▶  Run GreenSight")

# ----------------------------------------------------------------- run
if run:
    row = {"TrafficType": traffic, "Administrative_Duration": admin_dur,
           "Month": MONTH_MAP[month], "ProductRelated": product_related,
           "BounceRates": bounce, "PageValues": page_values}
    row_values = [row[f] for f in FEATURES]

    X, prob, band = rise_predict(row_values)
    base, contribs = shine_explain(X)
    reco, source = sage_recommend(prob, band, contribs)
    carbon = BAND_N[band] * CARBON_PER_RETURN
    band_color = {"HIGH": COLORS["green"], "MEDIUM": COLORS["amber"], "LOW": COLORS["neg"]}[band]

    st.markdown("### Result")
    col1, col2, col3 = st.columns([1, 1.3, 1.3])

    with col1:
        with st.container(border=True):
            st.markdown(f'<span class="gs-tag" style="color:{COLORS["green"]}">1 · RISE — prediction</span>',
                        unsafe_allow_html=True)
            st.markdown('<h3>Purchase intention</h3>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:2.6rem;font-weight:800;color:{COLORS["dark"]}">{prob*100:.1f}%</div>',
                        unsafe_allow_html=True)
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown(f'<div style="margin-top:10px"><span class="gs-band" style="background:{band_color}">{band}</span></div>',
                        unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown(f'<span class="gs-tag" style="color:{COLORS["blue"]}">2 · SHINE — explanation</span>',
                        unsafe_allow_html=True)
            st.markdown('<h3>Why this score</h3>', unsafe_allow_html=True)
            st.plotly_chart(shap_figure(base, contribs), use_container_width=True)
            top_f, top_v = contribs[0]
            st.caption(f"Base value {base:+.2f}. Largest driver: **{top_f}** ({top_v:+.2f}).")

    with col3:
        with st.container(border=True):
            st.markdown(f'<span class="gs-tag" style="color:{COLORS["amber"]}">3 · SAGE — recommendation</span>',
                        unsafe_allow_html=True)
            st.markdown('<h3>Recommended action</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="gs-reco">{reco}</div>', unsafe_allow_html=True)
            st.metric("Estimated carbon saving (SDG 12)", f"{carbon:.1f} kg CO₂")
            st.caption(f"SAGE engine: {source}")

    st.success("RISE → SHINE → SAGE completed. Each module fed the next.")
