"""
GreenSight — Streamlit UI
=========================
Pick a real session from the dataset (or type your own), click Run, and
RISE -> SHINE -> SAGE run in the backend, with the SDG 12 carbon saving.

RUN LOCALLY (after train_and_save.py has been run once):
    python -m streamlit run app.py

SAGE (the recommendation) uses an LLM, picking the first available:
  1) hosted LLM (Groq) if GROQ_API_KEY is set   -> works on the website
  2) local Ollama if it is running              -> works on your laptop
  3) a clear rule-based recommendation          -> always works
"""
import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostClassifier
import shap

# ----------------------------------------------------------------- config
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
               "llama3-70b-8192", "llama3-8b-8192"]
OLLAMA_MODEL = "llama3.2"
CARBON_PER_RETURN = 2.4
BAND_N = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
INK = "#1C2420"; GREEN = "#1F6F54"; BLUE = "#2E6E8E"; AMBER = "#C98A2B"
POS = "#2F8F6B"; NEG = "#C0564B"; MUTE = "#6B766F"; LINE = "#E3E7E2"

st.set_page_config(page_title="GreenSight", page_icon="◆", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.1rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.gs-head {{ display:flex; align-items:center; gap:16px; padding-bottom:14px;
  border-bottom:1px solid {LINE}; margin-bottom:6px; }}
.gs-mark {{ width:46px; height:46px; border-radius:11px; background:{GREEN};
  color:white; font-weight:700; font-size:19px; display:flex; align-items:center;
  justify-content:center; letter-spacing:.5px; flex:0 0 auto; }}
.gs-name {{ font-family:'Fraunces', serif; font-weight:600; font-size:30px;
  color:{INK}; line-height:1; }}
.gs-sub {{ color:{MUTE}; font-size:13.5px; margin-top:4px; }}
.gs-flow {{ color:{MUTE}; font-size:12.5px; letter-spacing:.4px; margin:10px 0 2px 0; }}
.gs-flow b {{ color:{INK}; font-weight:600; }}

.gs-mod {{ font-size:11px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; }}
.gs-h {{ font-family:'Fraunces',serif; font-weight:600; font-size:20px; color:{INK};
  margin:2px 0 10px 0; }}
.gs-prob {{ font-family:'Fraunces',serif; font-size:46px; font-weight:600; color:{INK};
  line-height:1; }}
.gs-band {{ display:inline-block; padding:7px 16px; border-radius:9px; font-size:16px;
  font-weight:700; color:white; letter-spacing:.5px; }}
.gs-reco {{ font-size:15.5px; line-height:1.55; color:{INK}; }}
.stButton>button {{ background:{GREEN}; color:white; font-weight:600; border:0;
  border-radius:10px; padding:.55rem 1.4rem; font-size:1.0rem; }}
.stButton>button:hover {{ background:#17563f; color:white; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- load once
@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier(); model.load_model("models/rise_catboost.cbm")
    month_map = joblib.load("models/month_map.joblib")
    features = joblib.load("models/features.joblib")
    explainer = shap.TreeExplainer(model)
    metrics = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else {}
    examples = json.load(open("models/examples.json")) if os.path.exists("models/examples.json") else []
    return model, month_map, features, explainer, metrics, examples

try:
    model, MONTH_MAP, FEATURES, EXPLAINER, METRICS, EXAMPLES = load_artifacts()
except Exception as e:
    st.error("Could not load the model. Run  python train_and_save.py  first. Details: %s" % e)
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
    contribs = sorted(zip(FEATURES, [float(v) for v in sv]), key=lambda t: abs(t[1]), reverse=True)
    return base, contribs

def _groq_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")

def _prompt(prob, band, contribs):
    top = ", ".join(f for f, _ in contribs[:2])
    return ("You are SAGE, a shop's assistant. "
            f"A visitor's chance of buying is {prob*100:.0f}% ({band}). Main reasons: {top}. "
            "Tell the shop ONE action to take, in ONE or TWO very short, simple sentences a "
            "non-expert understands. Be direct. No jargon, no lists, no long explanation. "
            "Do not suggest a discount if the visitor is already likely to buy.")

def sage_recommend(prob, band, contribs):
    prompt = _prompt(prob, band, contribs)
    key = _groq_key()
    if key:
        model_pref = None
        try:
            model_pref = st.secrets.get("GROQ_MODEL")
        except Exception:
            pass
        models = ([model_pref] if model_pref else []) + GROQ_MODELS
        last_err = ""
        for mdl in models:
            if not mdl:
                continue
            try:
                from groq import Groq
                client = Groq(api_key=key)
                resp = client.chat.completions.create(
                    model=mdl, temperature=0.3, max_tokens=70,
                    messages=[{"role": "user", "content": prompt}])
                return resp.choices[0].message.content.strip(), f"LLM · Groq ({mdl})"
            except Exception as e:
                last_err = str(e)[:120]
                continue
        st.warning("Groq was reached but every model call failed. Last error: " + last_err)
    try:
        import ollama
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"].strip(), "LLM · Ollama (local)"
    except Exception:
        pass
    fb = {
        "HIGH": "This shopper is very likely to buy. Don't give a discount. Show related products and offer free shipping to grow the order.",
        "MEDIUM": "This shopper might buy. Give a small nudge like free shipping and clear delivery info. No discount needed.",
        "LOW": "This shopper probably won't buy. Keep it simple and cheap: clear product info and an easy checkout. Don't spend on discounts.",
    }
    return fb[band], "rule-based (no LLM connected)"

def shap_figure(contribs):
    feats = [f for f, _ in contribs][::-1]
    vals = [v for _, v in contribs][::-1]
    cols = [POS if v >= 0 else NEG for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=cols,
                           text=[f"{v:+.2f}" for v in vals], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(height=290, margin=dict(l=8, r=34, t=6, b=6),
                      xaxis_title="effect on the score  ·  green pushes towards a purchase",
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Inter", size=12, color=INK))
    fig.update_xaxes(zeroline=True, zerolinecolor="#C9D0CB", gridcolor="#EEF1EE")
    fig.update_yaxes(gridcolor="white")
    return fig

# ----------------------------------------------------------------- header
st.markdown(f"""
<div class="gs-head">
  <div class="gs-mark">GS</div>
  <div>
    <div class="gs-name">GreenSight</div>
    <div class="gs-sub">Predicting online purchase intention, explaining it, and recommending a sustainable action</div>
  </div>
</div>
<div class="gs-flow"><b>RISE</b> predicts &nbsp;→&nbsp; <b>SHINE</b> explains &nbsp;→&nbsp; <b>SAGE</b> recommends</div>
""", unsafe_allow_html=True)

if METRICS:
    st.caption(f"Model: CatBoost · test accuracy {METRICS.get('accuracy','–')} · "
               f"test ROC AUC {METRICS.get('roc_auc','–')}")

# ----------------------------------------------------------------- inputs
st.write("")
st.markdown("##### Choose a session")

labels = ["— type my own —"] + [e["_label"] for e in EXAMPLES]
sel = st.selectbox("Pick a real session from the dataset, then adjust the values if you like",
                   labels, label_visibility="visible")

if sel != "— type my own —":
    p = next(e for e in EXAMPLES if e["_label"] == sel)
else:
    p = (EXAMPLES[2] if len(EXAMPLES) > 2 else
         dict(TrafficType=2, Administrative_Duration=40.0, Month=list(MONTH_MAP)[0],
              ProductRelated=20, BounceRates=0.02, PageValues=10.0))

months = list(MONTH_MAP.keys())
st.markdown("##### The six features")
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
                                help="Seconds spent on account / admin pages.")
with c3:
    month = st.selectbox("Month", months,
                         index=months.index(p["Month"]) if p["Month"] in months else 0)
    traffic = st.number_input("TrafficType", 1, 20, int(p["TrafficType"]), 1,
                              help="How the visitor reached the site (coded 1–20).")

st.write("")
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
    band_color = {"HIGH": GREEN, "MEDIUM": AMBER, "LOW": NEG}[band]

    st.write("")
    st.markdown("#### Result")
    col1, col2, col3 = st.columns([1, 1.25, 1.25])

    with col1:
        with st.container(border=True):
            st.markdown(f'<div class="gs-mod" style="color:{GREEN}">RISE · prediction</div>', unsafe_allow_html=True)
            st.markdown('<div class="gs-h">Purchase intention</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gs-prob">{prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown(f'<div style="margin-top:12px"><span class="gs-band" style="background:{band_color}">{band}</span></div>',
                        unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown(f'<div class="gs-mod" style="color:{BLUE}">SHINE · explanation</div>', unsafe_allow_html=True)
            st.markdown('<div class="gs-h">Why this score</div>', unsafe_allow_html=True)
            st.plotly_chart(shap_figure(contribs), use_container_width=True)
            tf, tv = contribs[0]
            st.caption(f"Starting value {base:+.2f}. Biggest factor: **{tf}** ({tv:+.2f}).")

    with col3:
        with st.container(border=True):
            st.markdown(f'<div class="gs-mod" style="color:{AMBER}">SAGE · recommendation</div>', unsafe_allow_html=True)
            st.markdown('<div class="gs-h">Recommended action</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gs-reco">{reco}</div>', unsafe_allow_html=True)
            st.write("")
            st.metric("Estimated carbon saving (SDG 12)", f"{carbon:.1f} kg CO₂")
            st.caption(f"SAGE engine: {source}")
