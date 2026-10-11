from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "my_src"
PRODUCTION_DIR = PROJECT_ROOT / "models" / "v2_30d" / "production"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from db_connector import DatabaseConnector
from feature_engineering.snapshot_builder import SnapshotBuilder
from feature_engineering.feature_engineer import FeatureEngineer
from prediction.churn_predictor import ChurnPredictor


# Project configuration
DATABASE_NAME = "cleaned_ravenstack_dw"
OBSERVATION_END_DATE = "2024-12-31"
PREDICTION_HORIZON_DAYS = 30
DEFAULT_PREDICTION_DATE = "2024-12-01"

MODEL_PATH = PRODUCTION_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = PRODUCTION_DIR / "preprocessor.joblib"
FEATURE_COLUMNS_PATH = PRODUCTION_DIR / "feature_columns.joblib"


# Dashboard styling
st.markdown(
    """
<style>
.stApp {
    background: #f7f9fc;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a33, #0b2545, #102a43);
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

[data-testid="stSidebar"] label {
    color: #dbeafe !important;
    font-weight: 600;
}

[data-testid="stSidebar"] input {
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111827 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.12);
}

.hero {
    padding: 1.1rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #ffffff, #e8f2ff);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.hero-title {
    color: #0f172a;
    font-size: 1.75rem;
    font-weight: 800;
    margin-bottom: .3rem;
}

.hero-subtitle {
    color: #64748b;
    font-size: .9rem;
    line-height: 1.45;
}


/* RavenStack premium header */
.hero {
    position: relative;
    overflow: hidden;
    padding: 1.65rem 1.85rem;
    border-radius: 22px;
    border: 1px solid #dce7f6;
    background: linear-gradient(110deg, #ffffff 0%, #f4f8ff 53%, #dbeaff 100%);
    box-shadow: 0 12px 32px rgba(38, 91, 159, .09);
    margin-bottom: 1.1rem;
}
.hero:before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    border-radius: 50%;
    background: rgba(106, 163, 250, .13);
    right: -110px;
    top: -180px;
    pointer-events: none;
}
.hero-top, .hero-body { position: relative; z-index: 1; }
.hero-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .7rem;
    flex-wrap: wrap;
    margin-bottom: 1.15rem;
}
.hero-brand {
    font-size: .78rem;
    font-weight: 850;
    letter-spacing: .075em;
    color: #2462cf;
}
.hero-brand span { color: #8194ad; font-weight: 650; }
.hero-status {
    font-size: .72rem;
    color: #386080;
    background: rgba(255,255,255,.85);
    border: 1px solid #dbe7f8;
    padding: .38rem .75rem;
    border-radius: 99px;
}
.hero-body {
    display: grid;
    grid-template-columns: minmax(0,1.3fr) minmax(340px, .9fr);
    align-items: center;
    gap: 1.8rem;
}
.hero-title {
    font-size: clamp(1.75rem, 2.45vw, 2.5rem);
    font-weight: 850;
    line-height: 1.12;
    letter-spacing: -.035em;
    color: #11223e;
    margin: 0 0 .75rem 0;
}
.hero-subtitle {
    color: #5d718f;
    font-size: .94rem;
    line-height: 1.65;
    max-width: 650px;
}
.hero-chips {
    display: flex;
    gap: .55rem;
    flex-wrap: wrap;
    margin-top: 1.15rem;
}
.hero-chip {
    background: rgba(255,255,255,.9);
    border: 1px solid #dfe8f5;
    border-radius: 10px;
    color: #26466d;
    font-size: .75rem;
    font-weight: 700;
    padding: .55rem .7rem;
    box-shadow: 0 4px 10px rgba(36, 86, 154, .04);
}
.hero-insights {
    display: grid;
    grid-template-columns: 1.1fr .9fr;
    gap: .7rem;
}
.hero-panel {
    background: rgba(255,255,255,.95);
    border: 1px solid #e0e9f6;
    border-radius: 16px;
    padding: .95rem;
    box-shadow: 0 8px 24px rgba(34,82,151,.07);
    min-width: 0;
}
.hero-panel-label {
    color: #637998;
    font-size: .73rem;
    font-weight: 750;
    margin-bottom: .4rem;
}
.hero-panel-value {
    color: #102a4c;
    font-size: clamp(1.5rem, 2vw, 2.1rem);
    font-weight: 850;
    letter-spacing: -.035em;
    margin-top: .8rem;
}
.hero-panel-note {
    color: #8a9ab0;
    font-size: .69rem;
    margin-top: .4rem;
}
.hero-sparkline { display: block; width: 100%; height: 82px; }
@media (max-width: 1050px) {
    .hero-body { grid-template-columns: 1fr; }
    .hero-insights { max-width: 530px; }
}
@media (max-width: 600px) {
    .hero { padding: 1.1rem; }
    .hero-insights { grid-template-columns: 1fr 1fr; }
    .hero-panel { padding: .7rem; }
    .hero-title { font-size: 1.65rem; }
}

.section-label {
    color: #2563eb;
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .2rem;
}

.section-title {
    color: #0f172a;
    font-size: 1.4rem;
    font-weight: 780;
    margin-bottom: .25rem;
}

.section-description {
    color: #64748b;
    font-size: .9rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.kpi-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 15px;
    padding: 1.1rem;
    min-height: 135px;
    box-shadow: 0 5px 18px rgba(15,23,42,.04);
}

.kpi-icon {
    font-size: 1.25rem;
    margin-bottom: .5rem;
}

.kpi-label {
    color: #64748b;
    font-size: .78rem;
    font-weight: 650;
}

.kpi-value {
    color: #0f172a;
    font-size: 1.7rem;
    font-weight: 800;
}

.kpi-note {
    color: #94a3b8;
    font-size: .72rem;
    margin-top: .35rem;
}

.empty-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1rem;
}

.empty-title {
    color: #0f172a;
    font-size: 1.3rem;
    font-weight: 750;
    margin-top: .5rem;
}

.empty-description {
    color: #64748b;
    margin-top: .5rem;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}

.stDownloadButton > button {
    border-radius: 10px;
    min-height: 44px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5eaf2;
    border-radius: 12px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 1rem;
}

button[data-baseweb="tab"] {
    font-weight: 650;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 1.4rem;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# Database connection
@st.cache_resource
def create_database_connection():
    connector = DatabaseConnector(database=DATABASE_NAME)
    return connector.get_engine()


# Load prediction pipeline
@st.cache_resource
def create_predictor():
    engine = create_database_connection()

    snapshot_builder = SnapshotBuilder(
        engine=engine,
        observation_end_date=OBSERVATION_END_DATE,
        prediction_horizon_days=PREDICTION_HORIZON_DAYS,
    )

    feature_engineer = FeatureEngineer(engine=engine)

    return ChurnPredictor.from_files(
        engine=engine,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        feature_columns_path=FEATURE_COLUMNS_PATH,
        snapshot_builder=snapshot_builder,
        feature_engineer=feature_engineer,
    )


try:
    predictor = create_predictor()
except Exception as error:
    st.error("The prediction pipeline could not be loaded.")
    st.exception(error)
    st.stop()


# Sidebar
st.sidebar.markdown(
    """
<div style="padding:.5rem 0 1rem 0;">
<div style="font-size:1.45rem;font-weight:800;color:white;">
📊 RavenStack
</div>
<div style="color:#93c5fd;font-size:.82rem;">
Customer Intelligence
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Prediction Settings")

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=pd.Timestamp(DEFAULT_PREDICTION_DATE).date(),
    min_value=pd.Timestamp("2023-01-01").date(),
    max_value=pd.Timestamp(DEFAULT_PREDICTION_DATE).date(),
)

top_k = st.sidebar.selectbox(
    "Weekly Retention Capacity",
    options=[25, 50, 75, 100],
    index=0,
)

threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.10,
    step=0.01,
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**🗓 Prediction Horizon**  
30 days

**🔄 Scoring Frequency**  
Weekly

**🧠 Production Model**  
XGBoost

**☁️ Data Source**  
Aiven Cloud MySQL
"""
)


# Reserve the header position so it can refresh after prediction completes
header_placeholder = st.empty()


# Prediction controls
left, right = st.columns([4, 1])

with left:
    st.markdown(
        """
<div class="section-label">Prediction Engine</div>
<div class="section-title">Weekly Customer Churn Risk</div>
<div class="section-description">
Generate point-in-time customer risk scores and prioritize retention efforts.
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.write("")

    run_prediction = st.button(
        "⚡ Generate Risk Scores",
        type="primary",
        use_container_width=True,
    )


# Generate predictions
if run_prediction:
    try:
        with st.spinner("Generating customer churn-risk scores..."):
            predictions = predictor.predict(
                prediction_date=str(prediction_date),
                threshold=threshold,
            )

        if predictions is None:
            raise ValueError("Prediction pipeline returned no results.")

        if predictions.empty:
            st.warning("No eligible customers found.")
            st.session_state.pop("predictions", None)
        else:
            st.session_state["predictions"] = (
                predictions.sort_values(
                    "churn_probability",
                    ascending=False,
                ).reset_index(drop=True)
            )

            st.session_state["scored_date"] = str(prediction_date)

    except Exception as error:
        st.session_state.pop("predictions", None)
        st.error("Prediction failed.")
        st.exception(error)


# Update the header using the latest completed prediction
header_risk = "—"
header_date = "Generate scores to view risk"
if "predictions" in st.session_state:
    header_data = st.session_state["predictions"]
    if header_data is not None and not header_data.empty:
        header_risk = f"{header_data['churn_probability'].max():.1%}"
        header_date = f"Scored: {st.session_state.get('scored_date', '—')}"

header_placeholder.markdown(
    f"""
<div class="hero">
<div class="hero-top">
<div class="hero-brand">◈ RAVENSTACK <span>/ CUSTOMER INTELLIGENCE</span></div>
<div class="hero-status">● ML-Powered &nbsp;·&nbsp; {header_date}</div>
</div>
<div class="hero-body">
<div>
<div class="hero-title">Predict churn.<br>Prioritize retention.</div>
<div class="hero-subtitle">
Turn customer behavior into actionable retention intelligence
with weekly risk rankings and 30-day churn forecasts.
</div>
<div class="hero-chips">
<div class="hero-chip">📅 30-day forecast</div>
<div class="hero-chip">📊 Weekly scoring</div>
<div class="hero-chip">🎯 Customer prioritization</div>
</div>
</div>
<div class="hero-insights">
<div class="hero-panel">
<div class="hero-panel-label">RISK INTELLIGENCE</div>
<svg class="hero-sparkline" viewBox="0 0 230 90" xmlns="http://www.w3.org/2000/svg" aria-label="Decorative risk trend">
<line x1="0" y1="20" x2="230" y2="20" stroke="#e5edf8" stroke-dasharray="3 4"/>
<line x1="0" y1="48" x2="230" y2="48" stroke="#e5edf8" stroke-dasharray="3 4"/>
<line x1="0" y1="76" x2="230" y2="76" stroke="#e5edf8" stroke-dasharray="3 4"/>
<path d="M0 74 C18 70 28 45 48 56 S75 71 98 44 S130 58 149 36 S176 45 195 26 S216 30 230 11 L230 90 L0 90 Z" fill="#3b82f6" fill-opacity=".10"/>
<path d="M0 74 C18 70 28 45 48 56 S75 71 98 44 S130 58 149 36 S176 45 195 26 S216 30 230 11" fill="none" stroke="#4387f4" stroke-width="3" stroke-linecap="round"/>
<circle cx="230" cy="11" r="4" fill="#4387f4"/>
</svg>
<div class="hero-panel-note">Illustrative signal · not model output</div>
</div>
<div class="hero-panel">
<div class="hero-panel-label">HIGHEST SCORED RISK</div>
<div class="hero-panel-value">{header_risk}</div>
<div class="hero-panel-note">Capacity: {top_k} customers/week</div>
</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# Dashboard navigation
overview_tab, rankings_tab, lookup_tab, export_tab, model_tab = st.tabs(
    [
        "📊 Overview",
        "🏆 Risk Rankings",
        "🔎 Customer Lookup",
        "📥 Export Results",
        "🧠 Model & Methodology",
    ]
)


# Prepare prediction results
has_predictions = "predictions" in st.session_state

if has_predictions:
    predictions = st.session_state["predictions"].copy()
    scored_date = st.session_state["scored_date"]

    predictions["churn_prediction"] = (
        predictions["churn_probability"] >= threshold
    ).astype(int)

    predictions["threshold_used"] = threshold

    top_customers = predictions.head(top_k).copy()

    total_customers = len(predictions)
    selected_customers = len(top_customers)

    average_risk = float(
        predictions["churn_probability"].mean()
    )

    highest_risk = float(
        predictions["churn_probability"].max()
    )

    above_threshold = int(
        (predictions["churn_probability"] >= threshold).sum()
    )

    if str(prediction_date) != scored_date:
        st.info(
            f"Displaying results for {scored_date}. "
            "Generate new scores to update the prediction date."
        )


# Overview tab
with overview_tab:
    if has_predictions:
        st.markdown(
            """
<div class="section-label">Executive Overview</div>
<div class="section-title">Retention Intelligence</div>
<div class="section-description">
Customer risk portfolio and weekly retention workload.
</div>
""",
            unsafe_allow_html=True,
        )

        kpis = [
            (
                "👥",
                "Active Customers",
                f"{total_customers:,}",
                "Customers scored",
            ),
            (
                "🎯",
                "Retention Capacity",
                f"{selected_customers:,}",
                "Customers prioritized",
            ),
            (
                "🛡️",
                "Average Risk Score",
                f"{average_risk:.1%}",
                "Portfolio average",
            ),
            (
                "⚠️",
                "Highest Risk Score",
                f"{highest_risk:.1%}",
                "Highest-ranked customer",
            ),
            (
                "🔔",
                "Above Threshold",
                f"{above_threshold:,}",
                f"Threshold ≥ {threshold:.0%}",
            ),
        ]

        columns = st.columns(5)

        for column, (icon, label, value, note) in zip(columns, kpis):
            with column:
                st.markdown(
                    f"""
<div class="kpi-card">
<div class="kpi-icon">{icon}</div>
<div class="kpi-label">{label}</div>
<div class="kpi-value">{value}</div>
<div class="kpi-note">{note}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.write("")

        st.markdown("### 📊 Customer Risk Distribution")

        bins = [
            0, 0.05, 0.10, 0.15, 0.20,
            0.30, 0.40, 0.50, 0.75, 1.0,
        ]

        distribution = (
            pd.cut(
                predictions["churn_probability"],
                bins=bins,
                include_lowest=True,
            )
            .value_counts()
            .sort_index()
        )

        distribution_df = pd.DataFrame(
            {
                "Risk Band": distribution.index.astype(str),
                "Customers": distribution.values,
            }
        )

        st.bar_chart(
            distribution_df,
            x="Risk Band",
            y="Customers",
            height=350,
        )

    else:
        st.markdown(
            """
<div class="empty-card">
<div style="font-size:2.5rem;">📈</div>
<div class="empty-title">Ready to Generate Customer Intelligence</div>
<div class="empty-description">
Choose your prediction settings in the sidebar,
then click Generate Risk Scores.
</div>
</div>
""",
            unsafe_allow_html=True,
        )


# Risk rankings tab
with rankings_tab:
    if has_predictions:
        st.markdown(
            f"### 🏆 Top {selected_customers} Customers by Churn Risk"
        )

        st.caption(
            "Highest-risk customers recommended for retention review."
        )

        display_df = top_customers[
            [
                "risk_rank",
                "account_id",
                "churn_probability",
                "prediction_date",
            ]
        ].copy()

        display_df["risk_level"] = pd.cut(
            display_df["churn_probability"],
            bins=[-np.inf, 0.05, 0.10, np.inf],
            labels=["Low", "Medium", "High"],
        )

        display_df["risk_percent"] = (
            display_df["churn_probability"] * 100
        )

        display_df = display_df.rename(
            columns={
                "risk_rank": "Rank",
                "account_id": "Customer ID",
                "prediction_date": "Prediction Date",
                "risk_level": "Risk Level",
                "risk_percent": "Risk Score",
            }
        )

        display_df = display_df.drop(
            columns=["churn_probability"]
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=550,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank",
                    format="#%d",
                ),
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Risk Level": st.column_config.TextColumn(
                    "Risk Level",
                ),
            },
        )

        st.info(
            f"Showing {selected_customers} customers based on "
            "your weekly retention capacity."
        )

    else:
        st.info("Generate predictions to view customer rankings.")


# Customer lookup tab
with lookup_tab:
    if has_predictions:
        st.markdown("### 🔎 Customer Risk Lookup")

        st.caption(
            "Select a customer to inspect their risk score and ranking."
        )

        customer_options = (
            predictions["account_id"].astype(str).tolist()
        )

        selected_customer = st.selectbox(
            "Select Customer",
            customer_options,
        )

        customer_result = predictions.loc[
            predictions["account_id"].astype(str)
            == selected_customer
        ].iloc[0]

        customer_risk = float(
            customer_result["churn_probability"]
        )

        customer_rank = int(
            customer_result["risk_rank"]
        )

        lookup1, lookup2 = st.columns(2)

        lookup1.metric(
            "Risk Rank",
            f"#{customer_rank}",
        )

        lookup2.metric(
            "Risk Score",
            f"{customer_risk:.1%}",
        )

        if customer_risk >= threshold:
            st.warning(
                "⚠️ This customer is above the selected risk threshold."
            )
        else:
            st.success(
                "✓ This customer is below the selected risk threshold."
            )

        if customer_rank <= top_k:
            st.info(
                "This customer is included in the current "
                "weekly retention queue."
            )
        else:
            st.info(
                "This customer is outside the current "
                "weekly retention capacity."
            )

    else:
        st.info("Generate predictions to look up customers.")


# Export tab
with export_tab:
    st.markdown("### 📥 Download Prediction Results")

    if has_predictions:
        st.caption(
            "Export the complete scored customer population "
            "for retention analysis."
        )

        csv = predictions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download All Predictions",
            data=csv,
            file_name=f"churn_predictions_{scored_date}.csv",
            mime="text/csv",
            use_container_width=False,
        )

        st.write("")

        st.metric(
            "Available Customer Records",
            f"{total_customers:,}",
        )

    else:
        st.info("Generate predictions before downloading results.")


# Model performance tab
with model_tab:
    st.markdown("### 🧠 Model Performance & Methodology")

    st.caption(
        "Held-out test results and modelling decisions."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric("Test PR-AUC", "0.1474")
    m2.metric("Test ROC-AUC", "0.5729")
    m3.metric("PR-AUC Lift", "1.22×")

    st.markdown("### Operational Performance")

    st.markdown(
        """
Using a **Top-100 weekly retention strategy** on the held-out test period:

- **30.6%** of churn-positive customer-week observations were captured.
- Mean precision was approximately **15.7%**.
- Mean lift was approximately **1.31×**.
- Approximately **23.6%** of eligible customers were targeted.
"""
    )

    st.markdown("### Prediction Architecture")

    st.code(
        "Aiven MySQL → Customer Snapshot → "
        "Feature Engineering → Preprocessing → "
        "XGBoost → Risk Scores → Customer Rankings",
        language=None,
    )

    st.markdown("### Modelling Strategy")

    st.markdown(
        """
- Weekly customer scoring
- 30-day churn prediction horizon
- Point-in-time feature engineering
- Chronological train/validation/test splitting
- XGBoost production model
- Capacity-based customer prioritization
"""
    )

    st.markdown("### Model Limitations")

    st.info(
        "The model has modest predictive discrimination. "
        "Its risk scores are used primarily for ranking and "
        "prioritizing customers, rather than as perfectly "
        "calibrated churn probabilities."
    )

    st.caption(
        "This application uses historical customer data "
        "for portfolio demonstration."
    )


# Footer
st.markdown(
    """
<div style="
text-align:center;
color:#94a3b8;
font-size:.78rem;
padding:1.5rem 0 1rem 0;
">
Customer Retention & Revenue Intelligence Platform
&nbsp;·&nbsp;
Machine Learning Decision Support
&nbsp;·&nbsp;
XGBoost
</div>
""",
    unsafe_allow_html=True,
)
