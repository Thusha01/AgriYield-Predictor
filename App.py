import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="PaddyPredict",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .centered-title {
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .centered-subtitle {
        text-align: center;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f766e;
    }

    /* Labels white */
    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    /* FIX: Input fields readable */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {
        color: black !important;
        background-color: white !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }

    /* Button styling */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #f97316;
        color: white;
        font-weight: 600;
        border-radius: 999px;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: #fb923c;
        border-color: #fed7aa;
    }

    .how-it-works p {
        font-size: 0.95rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        padding: 1.8rem 2rem;
        border-radius: 1.25rem;
        color: white;
        box-shadow: 0 14px 30px rgba(15, 118, 110, 0.35);
        margin-bottom: 1.5rem;
    }

    .prediction-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.9;
    }

    .prediction-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }

    .prediction-unit {
        font-size: 0.95rem;
        opacity: 0.9;
    }

    h3 {
        margin-top: 1.2rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("paddy_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

# -----------------------------
# SIDEBAR (INPUTS)
# -----------------------------
st.sidebar.title("🌾 PaddyPredict Inputs")

st.sidebar.markdown("Provide field details to forecast **paddy production**.")

district = st.sidebar.selectbox(
    "Select District",
    [
        "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya",
        "Galle", "Matara", "Hambantota", "Jaffna", "Mannar", "Vavuniya",
        "Mullaitivu", "Kilinochchi", "Batticaloa", "Ampara", "Trincomalee",
        "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa",
        "Badulla", "Monaragala", "Ratnapura", "Kegalle"
    ]
)

season = st.sidebar.selectbox("Season", ["Maha", "Yala"])

year = st.sidebar.number_input("Year", 2000, 2035, 2025)

cultivated_extent = st.sidebar.number_input(
    "Cultivated Extent (ha)",
    value=50000
)

avg_yield = st.sidebar.number_input(
    "Average Yield (kg/ha)",
    value=4000
)

prev_production = st.sidebar.number_input(
    "Previous Production (MT)",
    value=200000
)

predict_btn = st.sidebar.button("🔮 Predict Production")

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown(
    "<h1 class='centered-title'>🌾 PaddyPredict</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 class='centered-subtitle'>AI-based paddy production forecasting for Sri Lankan districts</h4>",
    unsafe_allow_html=True
)

# -----------------------------
# TOP LAYOUT (IMAGE + INFO)
# -----------------------------
top_left, top_right = st.columns([2, 1.4])

with top_left:
    st.image(
        "https://images.unsplash.com/photo-1560493676-04071c5f467b",
        use_container_width=True
    )

with top_right:
    with st.container():
        st.info("""
    **How it works:**
    - Enter agricultural data
    - Click Predict
    - View prediction + explanation
    """)

# -----------------------------
# PREPROCESS INPUT
# -----------------------------
season_encoded = 1 if season == "Maha" else 0
year_index = year - 2006

input_data = pd.DataFrame([{
    "Year_Index": year_index,
    "Cultivated_Extent": cultivated_extent,
    "Avg_Yield": avg_yield,
    "Season_Encoded": season_encoded,
    "Prev_Production": prev_production
}])

# -----------------------------
# PREDICTION + EXPLANATIONS
# -----------------------------
if predict_btn:
    prediction = model.predict(input_data)[0]
    prediction_int = int(prediction)

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Predicted Paddy Production</div>
            <div class="prediction-value">{prediction_int:,}</div>
            <div class="prediction-unit">Metric Tons (MT)</div>
            <div style="margin-top:0.7rem;font-size:0.9rem;opacity:0.9;">
                For {district}, {season} season, {year}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    expl_col1, expl_col2 = st.columns([1.4, 1])

    with expl_col1:
        st.subheader("🔍 Model Explanation (SHAP)")
        shap_values = explainer.shap_values(input_data)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=input_data.iloc[0],
                feature_names=input_data.columns
            ),
            show=False
        )
        st.pyplot(fig, use_container_width=True)

    with expl_col2:
        st.subheader("📊 Feature Importance")
        importances = model.feature_importances_
        features = input_data.columns

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(features, importances, color="#0f766e")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Features")
        ax2.set_title("Global Feature Importance", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

# -----------------------------
# VISUALIZATION SECTION
# -----------------------------
st.markdown("---")
st.subheader("📈 Historical Production Trends")

trend_col1, trend_col2 = st.columns([2, 1])

with trend_col1:
    try:
        df = pd.read_csv("Final_Data.csv")

        maha = df[df['Season'] == 'Maha']
        yala = df[df['Season'] == 'Yala']

        maha_year = maha.groupby('Year')['Production'].sum()
        yala_year = yala.groupby('Year')['Production'].sum()

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(maha_year.index, maha_year.values, label='Maha', color="#0f766e", linewidth=2)
        ax3.plot(yala_year.index, yala_year.values, label='Yala', color="#f97316", linewidth=2)

        ax3.set_xlabel("Year")
        ax3.set_ylabel("Total Production (MT)")
        ax3.set_title("Seasonal Production Over Time", fontsize=11)
        ax3.legend()
        ax3.grid(alpha=0.2)

        st.pyplot(fig3, use_container_width=True)

    except Exception:
        st.warning("Dataset not found for visualization")

with trend_col2:
    st.markdown(
        """
        - View trends for **Maha** and **Yala** seasons.  
        - Helps compare predicted production with historical values.  
        - Useful for planning storage, logistics, and market decisions.
        """
    )