"""
CodeAlpha - Task 4: Heart Disease Prediction - Streamlit Demo
Interactive, LinkedIn-ready web application.
"""
import streamlit as st
import pandas as pd
import joblib

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- Hero banner ---------- */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(30, 58, 138, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.12em;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
        position: relative;
        z-index: 1;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
        max-width: 720px;
        position: relative;
        z-index: 1;
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---------- Result cards ---------- */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    .result-card.high {
        border-left: 6px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    .result-card.low {
        border-left: 6px solid #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    .result-card.neutral {
        background: #f8fafc;
        border-style: dashed;
        text-align: center;
    }

    /* ---------- Gauge ---------- */
    .gauge-wrapper {
        display: flex;
        justify-content: center;
        margin: 1rem 0 1.5rem 0;
    }
    .gauge {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
    }
    .gauge-inner {
        width: 138px;
        height: 138px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563eb, #1e40af) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(37, 99, 235, 0.5) !important;
    }

    /* ---------- Inputs ---------- */
    .stSelectbox label, .stSlider label, .stSelectSlider label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 2.5rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        line-height: 1.7;
    }
    .footer strong {
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ============ LOAD ARTIFACTS ============
@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Run the notebook's save cell to generate "
        "`model.pkl`, `scaler.pkl`, and `features.pkl`."
    )
    st.stop()

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🫀 Heart Disease AI")
    st.caption("Clinical decision support demo")
    st.divider()
    st.markdown("### 📊 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", "85.9%")
        st.metric("F1-Score", "0.879")
    with c2:
        st.metric("ROC-AUC", "0.919")
        st.metric("Precision", "0.839")
    st.divider()
    st.markdown("### 🧠 Algorithm")
    st.caption("Support Vector Machine (RBF kernel), probability-calibrated")
    st.divider()
    st.markdown("### 📁 Dataset")
    st.caption("""
**UCI Heart Disease**  
920 patients · 4 hospitals  
Cleveland · Hungary · Switzerland · Long Beach
    """)
    st.divider()
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")

# ============ HERO ============
st.markdown("""
<div class="hero">
    <div class="hero-badge">MACHINE LEARNING · CODEALPHA INTERNSHIP</div>
    <h1>🫀 Heart Disease Risk Predictor</h1>
    <p>An AI-powered clinical decision support tool trained on 920 patient records from 4 international hospitals.</p>
</div>
""", unsafe_allow_html=True)

# ============ LAYOUT ============
left, right = st.columns([1.6, 1], gap="large")

with left:
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 20, 90, 50, 1)
        with c2:
            sex = st.selectbox("Sex", ["Male", "Female"])

        st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 120, 5)
            chol = st.slider("Cholesterol (mg/dL)", 100, 600, 200, 10)
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", [False, True])
        with c2:
            thalch = st.slider("Max heart rate achieved", 60, 220, 150, 5)
            oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
            exang = st.selectbox("Exercise-induced angina", [False, True])

        st.markdown('<div class="section-title">🧪 Test Results</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            cp = st.selectbox(
                "Chest pain type",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
            )
            restecg = st.selectbox(
                "Resting ECG",
                ["normal", "st-t abnormality", "lv hypertrophy"]
            )
        with c2:
            slope = st.selectbox(
                "Slope of peak exercise ST segment",
                ["upsloping", "flat", "downsloping"]
            )
            thal = st.selectbox(
                "Thalassemia",
                ["normal", "fixed defect", "reversable defect"]
            )

        ca = st.select_slider(
            "Number of major vessels (0-3)",
            options=[0.0, 1.0, 2.0, 3.0],
            value=0.0
        )

        st.markdown("")
        submitted = st.form_submit_button(
            "🔍 Predict Heart Disease Risk",
            type="primary",
            use_container_width=True
        )

with right:
    st.markdown('<div class="section-title">📋 Risk Assessment</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown("""
        <div class="result-card neutral">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🩻</div>
            <div style="color: #64748b; font-weight: 500; line-height: 1.6;">
                Fill in the patient data on the left and click
                <strong style="color:#1e293b;">Predict</strong> to see the risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build input row
        raw = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": str(fbs),
            "restecg": restecg,
            "thalch": thalch,
            "exang": str(exang),
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": float(ca),
            "thal": thal,
        }])

        # Encode + align with training features
        raw = pd.get_dummies(raw)
        raw = raw.reindex(columns=feature_names, fill_value=0)
        raw_scaled = scaler.transform(raw)

        proba = float(model.predict_proba(raw_scaled)[0, 1])
        prediction = int(model.predict(raw_scaled)[0])

        # Color per risk level
        if proba < 0.33:
            color = "#10b981"
        elif proba < 0.66:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        # Circular gauge
        st.markdown(f"""
        <div class="gauge-wrapper">
            <div class="gauge" style="background: conic-gradient({color} {proba*100}%, #e2e8f0 {proba*100}% 100%);">
                <div class="gauge-inner" style="color: {color};">{proba:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Result card
        if prediction == 1:
            st.markdown("""
            <div class="result-card high">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #dc2626; margin-bottom: 0.5rem;">
                    High Risk Detected
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates a significant probability of heart disease.
                    <strong>Cardiology consultation is recommended.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card low">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">✅</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #059669; margin-bottom: 0.5rem;">
                    Low Risk
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates a low probability of heart disease.
                    <strong>Routine follow-up is advised.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Top features
        st.markdown("""
        <div class="result-card" style="background: #f8fafc;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                🔬 Top Predictive Features
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; color: #334155; font-size: 0.88rem; line-height: 1.9;">
                <li>Cholesterol level</li>
                <li>Age</li>
                <li>Max heart rate achieved</li>
                <li>ST depression (oldpeak)</li>
                <li>Exercise-induced angina</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚠️ This is a machine learning estimate, not a medical diagnosis.")

# ============ FOOTER ============
st.markdown("""
<div class="footer">
    <strong>CodeAlpha Machine Learning Internship</strong> · Built by <strong>Sawadogo Noé</strong><br>
    UCI Heart Disease dataset · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)