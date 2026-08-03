"""
Streamlit app: paste a URL, get a phishing risk score plus the
top features driving that score.

Run locally with:
    streamlit run app.py
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Single-source the feature logic: the canonical implementation lives in
# src/features.py and is shared verbatim between training and serving, so
# there is no chance of train/serve feature drift.
sys.path.insert(0, str(BASE_DIR / "src"))
from features import extract_features  # noqa: E402

st.set_page_config(page_title="Phishing URL Risk Checker", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_models():
    with open(MODELS_DIR / "xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODELS_DIR / "random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    return xgb, rf, feature_cols


xgb_model, rf_model, feature_cols = load_models()

MODEL_OPTIONS = {"XGBoost (recommended)": xgb_model, "Random Forest": rf_model}

st.title("🛡️ Phishing URL Risk Checker")
st.caption(
    "Paste a URL to get a phishing risk score from a model trained on lexical & "
    "structural URL features — no live browsing or content fetching involved."
)

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model = MODEL_OPTIONS[model_choice]
    st.markdown("---")
    st.markdown(
        "**About this tool**\n\n"
        "Trained on ~188K URLs (legitimate + phishing) using 18 lexical/structural "
        "features — HTTPS usage, subdomain count, hyphen count, character entropy, "
        "suspicious keywords, and more. No page content or network requests are used, "
        "so it works instantly and offline.\n\n"
        "**Known limitation:** URL-only lexical classifiers like this one can still "
        "misjudge unusual-but-legitimate URLs (e.g. certain deep links or URL "
        "fragments), since they have no way to check what's actually on the page. "
        "This is a documented limitation of the entire approach, not just this model — "
        "treat the score as one signal, not a verdict.\n\n"
        "This is a portfolio/learning project, not a production security tool. "
        "Always verify suspicious links through official channels."
    )

url_input = st.text_input("Enter a URL to check", placeholder="https://example.com/login")
check_clicked = st.button("Check URL", type="primary")

if check_clicked and url_input.strip():
    url = url_input.strip()
    if "://" not in url:
        # Most sites redirect http -> https today, and browsers default to
        # https when you type a bare domain. Without this, has_https would
        # be wrongly scored as 0 for any URL typed without a scheme,
        # regardless of whether the real site actually uses HTTPS.
        url = "https://" + url
        st.caption(f"Interpreting input as: `{url}`")

    try:
        feats = extract_features(url)
        X = pd.DataFrame([feats])[feature_cols]
        proba = model.predict_proba(X)[0]
        risk_score = proba[1]  # probability of class 1 = phishing
        prediction = model.predict(X)[0]
    except Exception:  # noqa: BLE001 - never surface a raw traceback to the user
        st.error(
            "Couldn't analyze that input — it doesn't look like a URL this tool "
            "can parse. Try something like `https://example.com/login`."
        )
        st.stop()

    st.markdown("---")

    # --- Risk score display ---
    col1, col2 = st.columns([1, 2])
    with col1:
        if risk_score >= 0.7:
            st.error(f"### 🚨 High Risk\n**{risk_score*100:.1f}%**")
        elif risk_score >= 0.35:
            st.warning(f"### ⚠️ Medium Risk\n**{risk_score*100:.1f}%**")
        else:
            st.success(f"### ✅ Low Risk\n**{risk_score*100:.1f}%**")

    with col2:
        st.progress(min(int(risk_score * 100), 100))
        verdict = "Likely Phishing" if prediction == 1 else "Likely Legitimate"
        st.write(f"**Model verdict:** {verdict}")
        st.write(f"**Confidence:** legit {proba[0]*100:.1f}% · phishing {proba[1]*100:.1f}%")

    # --- Feature breakdown ---
    st.markdown("### Why this score?")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importances.sort_values(ascending=False).head(8).index.tolist()

        rows = []
        for f in top_features:
            rows.append({"Feature": f, "Value for this URL": feats[f]})
        st.table(pd.DataFrame(rows).set_index("Feature"))

    with st.expander("See all extracted features"):
        st.json(feats)

    st.markdown("---")
    st.caption(
        "⚠️ No automated tool is perfect. Never enter credentials on a site you're "
        "unsure about, and verify unexpected links directly with the sender through "
        "a separate, trusted channel."
    )

elif check_clicked:
    st.info("Enter a URL first.")

st.markdown("---")
with st.expander("Try some examples"):
    examples = [
        "https://www.google.com",
        "http://paypal-secure-login.verify-account.tk/webscr?cmd=login",
        "https://github.com/anthropics",
        "http://192.168.1.1/login/verify-account.php",
    ]
    for ex in examples:
        st.code(ex)
