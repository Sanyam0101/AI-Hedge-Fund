import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------
# 🎨 PAGE CONFIGURATION
# ---------------------
st.set_page_config(
    page_title="AI Hedge Fund Dashboard",
    layout="wide",
    page_icon="💹"
)

# Custom dark theme
st.markdown("""
    <style>
    body { background-color: #0e1117; color: #fafafa; }
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6, p { color: #fafafa !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------
# 🧠 HEADER
# ---------------------
st.title("💹 AI Hedge Fund Predictor")
st.markdown("### Predict potential returns using AI-powered models")

# ---------------------
# 📦 LOAD TRAINED MODEL
# ---------------------
@st.cache_resource
def load_model():
    with open("hedge_fund_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------------
# 📊 USER INPUT SECTION
# ---------------------
st.markdown("#### Enter Market Indicators")
col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("📊 Feature 1 (e.g., Market Volatility Index)", value=3.0)
with col2:
    feature2 = st.number_input("📈 Feature 2 (e.g., Interest Rate)", value=6.0)

# ---------------------
# 🔮 PREDICTION & VISUALS
# ---------------------
if st.button("🔮 Predict Return"):
    try:
        # Predict return
        features = np.array([[feature1, feature2]])
        result = model.predict(features)[0]

        st.success(f"📈 Predicted Return: **{result:.2f}%**")

        # Visualization - input features
        df = pd.DataFrame({
            "Feature": ["Feature 1", "Feature 2"],
            "Value": [feature1, feature2]
        })
        fig = px.bar(df, x="Feature", y="Value", color="Feature",
                     title="Input Features Overview")
        st.plotly_chart(fig, use_container_width=True)

        # Visualization - historical performance (sample)
        st.markdown("##### Historical Performance (Sample Data)")
        data = {
            "Date": pd.date_range(start="2024-01-01", periods=6, freq="M"),
            "Return": [1.1, 1.4, 1.6, 1.8, 2.0, result]
        }
        hist_df = pd.DataFrame(data)
        fig2 = px.line(hist_df, x="Date", y="Return", markers=True,
                       title="Simulated Hedge Fund Performance")
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# ---------------------
# 📘 FOOTER
# ---------------------
st.markdown("""
---
💡 *This dashboard is a prototype AI hedge fund predictor built with Streamlit and scikit-learn.*
""")