
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

# Project imports
from db_connector import DatabaseConnector
from feature_engineering.snapshot_builder import SnapshotBuilder
from feature_engineering.feature_engineer import FeatureEngineer
from prediction.churn_predictor import ChurnPredictor

# Configuration
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
    background:
        radial-gradient(circle at 90% 0%, rgba(59,130,246,.10), transparent 28%),
        #f7f9fc;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: -.03em;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071a33 0%, #0b2545 55%, #102a43 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

[data-testid="stSidebar"] label {
    color: #dbeafe !important;
    font-weight: 600;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.12);
}

[data-testid="stSidebar"] input {
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111827 !important;
}

.hero {
    padding: 2.3rem 2.5rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at 88% 20%, rgba(59,130,246,.22), transparent 30%),
        linear-gradient(135deg, #ffffff 0%, #f5f9ff 55%, #e8f2ff 100%);
    border: 1px solid #e2e8f0;
    box-shadow: 0 14px 40px rgba(15,23,42,.07);
    margin-bottom: 1.8rem;
}

.badge {
    display: inline-block;
    padding: .4rem .85rem;
    border-radius: 999px;
    background: #e8f1ff;
    color: #2563eb;
    font-size: .82rem;
    font-weight: 750;
    margin-bottom: .9rem;
}

.hero-title {
    color: #0f172a;
    font-size: 2.55rem;
    font-weight: 850;
    line-height: 1.08;
    max-width: 950px;
    margin-bottom: .75rem;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1.04rem;
    line-height: 1.7;
    max-width: 850px;
}

.hero-pills {
    margin-top: 1.3rem;
    display: flex;
    flex-wrap: wrap;
    gap: .65rem;
}

.hero-pill {
    background: rgba(255,255,255,.88);
    border: 1px solid #e2e8f0;
    padding: .55rem .85rem;
    border-radius: 11px;
    color: #334155;
    font-weight: 650;
    font-size: .84rem;
    box-shadow: 0 3px 10px rgba(15,23,42,.04);
}

.section-label {
    color: #2563eb;
    font-size: .76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-bottom: .25rem;
}

.section-title {
    color: #0f172a;
    font-size: 1.5rem;
    font-weight: 780;
    margin-bottom: .25rem;
}

.section-description {
    color: #64748b;
    font-size: .92rem;
    line-height: 1.6;
    margin-bottom: 1.15rem;
}

.kpi-card {
    background: rgba(255,255,255,.96);
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 1.25rem 1.3rem;
    min-height: 155px;
    box-shadow: 0 7px 24px rgba(15,23,42,.055);
}

.kpi-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eff6ff;
    font-size: 1.2rem;
    margin-bottom: .8rem;
}

.kpi-label {
    color: #64748b;
    font-size: .80rem;
    font-weight: 650;
    margin-bottom: .25rem;
}

.kpi-value {
    color: #0f172a;
    font-size: 1.8rem;
    font-weight: 820;
    line-height: 1.1;
}

.kpi-note {
    color: #94a3b8;
    font-size: .73rem;
    margin-top: .45rem;
}

.empty-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 20px;
    padding: 3.2rem;
    text-align: center;
    box-shadow: 0 8px 25px rgba(15,23,42,.04);
    margin-top: 1rem;
}

.empty-icon {
    font-size: 3rem;
}

.empty-title {
    color: #0f172a;
    font-size: 1.35rem;
    font-weight: 780;
    margin-top: .7rem;
}

.empty-description {
    color: #64748b;
    max-width: 600px;
    margin: .6rem auto 0 auto;
    line-height: 1.6;
}

.stButton > button {
    border-radius: 11px;
    border: 0;
    font-weight: 700;
    min-height: 46px;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    box-shadow: 0 5px 16px rgba(37,99,235,.18);
}

.stDownloadButton > button {
    border-radius: 11px;
    min-height: 44px;
    font-weight: 650;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    padding: 1rem;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 1.9rem;
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


# Load saved model and prediction pipeline
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
<div style="padding:.5rem 0 1.2rem 0;">
    <div style="font-size:1.45rem;font-weight:800;color:white;">
        📊 RavenStack
    </div>
    <div style="color:#93c5fd;font-size:.82rem;margin-top:.2rem;">
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

**📊 Database Name**  
cleaned_ravenstack_dw
"""
)


# Hero section

# Hero section
st.markdown(
    """
<div class="hero">
<div class="badge">✦ AI-Powered Customer Intelligence</div>
<div class="hero-title">Customer Retention & Revenue Intelligence Platform</div>
<div class="hero-subtitle">
Identify customers showing elevated churn risk,
prioritize retention efforts and transform customer
behavior into actionable retention intelligence.
</div>
<div class="hero-pills">
<div class="hero-pill">🔮 Churn Prediction</div>
<div class="hero-pill">🏆 Customer Prioritization</div>
<div class="hero-pill">💰 Revenue Protection</div>
<div class="hero-pill">⚡ XGBoost Engine</div>
<div class="hero-pill">📅 30-Day Forecast</div>
<div class="hero-pill">📊 Weekly Risk Scoring</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)



# Prediction controls
left, right = st.columns([4, 1])

with left:
    st.markdown(
        """
<div class="section-label">Prediction Engine</div>
<div class="section-title">Weekly Customer Churn Risk</div>
<div class="section-description">
    Generate point-in-time customer risk scores and identify
    the customers your retention team should prioritize.
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


# Generate and retain predictions
if run_prediction:
    try:
        with st.spinner(
            "Building customer snapshots, engineering features "
            "and generating risk scores..."
        ):
            predictions = predictor.predict(
                prediction_date=str(prediction_date),
                threshold=threshold,
            )

        if predictions.empty:
            st.warning("No eligible customers found for this prediction date.")
            st.session_state.pop("predictions", None)
        else:
            st.session_state["predictions"] = (
                predictions.sort_values(
                    "churn_probability", ascending=False
                ).reset_index(drop=True)
            )
            st.session_state["scored_date"] = str(prediction_date)

    except Exception as error:
        st.session_state.pop("predictions", None)
        st.error("Prediction failed.")
        st.exception(error)


# Display predictions
if "predictions" in st.session_state:
    predictions = st.session_state["predictions"].copy()
    scored_date = st.session_state["scored_date"]

    if str(prediction_date) != scored_date:
        st.info(
            f"Displaying results for {scored_date}. "
            "Click Generate Risk Scores to score the newly selected date."
        )

    # Apply current business settings
    predictions["churn_prediction"] = (
        predictions["churn_probability"] >= threshold
    ).astype(int)
    predictions["threshold_used"] = threshold

    top_customers = predictions.head(top_k).copy()

    total_customers = len(predictions)
    selected_customers = len(top_customers)
    average_risk = float(predictions["churn_probability"].mean())
    highest_risk = float(predictions["churn_probability"].max())

    above_threshold = int(
        (predictions["churn_probability"] >= threshold).sum()
    )

    st.success(
        f"✓ Prediction results available — "
        f"{total_customers:,} active customers scored for {scored_date}."
    )

    # Executive overview
    st.markdown(
        """
<div class="section-label">Executive Overview</div>
<div class="section-title">Retention Intelligence</div>
<div class="section-description">
    A snapshot of the customer risk portfolio and retention workload.
</div>
""",
        unsafe_allow_html=True,
    )

    kpis = [
        ("👥", "Active Customers", f"{total_customers:,}",
         "Customers scored this week"),
        ("🎯", "Retention Capacity", f"{selected_customers:,}",
         "Highest-risk customers selected"),
        ("🛡️", "Average Risk Score", f"{average_risk:.1%}",
         "Across active customers"),
        ("⚠️", "Highest Risk Score", f"{highest_risk:.1%}",
         "Highest-ranked customer"),
        ("🔔", "Above Threshold", f"{above_threshold:,}",
         f"Threshold ≥ {threshold:.0%}"),
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

    table_col, analytics_col = st.columns([1.7, 1])

    # Customer ranking table
    with table_col:
        st.markdown(
            f"""
<div class="section-label">Retention Queue</div>
<div class="section-title">
    🏆 Top {selected_customers} Customers by Churn Risk
</div>
<div class="section-description">
    Highest-risk customers recommended for retention review.
    Rankings are based on the model's churn-risk score.
</div>
""",
            unsafe_allow_html=True,
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

        display_df = display_df.drop(columns=["churn_probability"])

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank", format="#%d"
                ),
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Risk Level": st.column_config.TextColumn(
                    "Risk Level"
                ),
            },
        )

    # Portfolio analytics
    with analytics_col:
        st.markdown(
            """
<div class="section-label">Portfolio Analytics</div>
<div class="section-title">📊 Customer Risk Distribution</div>
<div class="section-description">
    Distribution of predicted churn-risk scores
    across the active customer population.
</div>
""",
            unsafe_allow_html=True,
        )

        bins = [0, .05, .10, .15, .20, .30, .40, .50, .75, 1.0]

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
            height=270,
        )

        st.write("")

        # Customer lookup
        st.markdown(
            """
<div class="section-title">🔎 Customer Risk Lookup</div>
<div class="section-description">
    Select a customer to inspect their risk score and portfolio rank.
</div>
""",
            unsafe_allow_html=True,
        )

        customer_options = predictions["account_id"].astype(str).tolist()

        selected_customer = st.selectbox(
            "Customer",
            customer_options,
            label_visibility="collapsed",
        )

        customer_result = predictions.loc[
            predictions["account_id"].astype(str) == selected_customer
        ].iloc[0]

        customer_risk = float(customer_result["churn_probability"])
        customer_rank = int(customer_result["risk_rank"])

        lookup1, lookup2 = st.columns(2)

        lookup1.metric("Risk Rank", f"#{customer_rank}")
        lookup2.metric("Risk Score", f"{customer_risk:.1%}")

        if customer_risk >= threshold:
            st.warning(
                "⚠️ This customer is above the selected risk threshold."
            )
        else:
            st.success(
                "✓ This customer is below the selected risk threshold."
            )

    # Export predictions
    st.write("")

    export_col1, export_col2 = st.columns([3, 1])

    with export_col1:
        st.markdown(
            """
<div class="section-label">Export</div>
<div class="section-title">Download Prediction Results</div>
<div class="section-description">
    Export the complete scored customer population
    for retention analysis or operational follow-up.
</div>
""",
            unsafe_allow_html=True,
        )

    with export_col2:
        st.write("")

        csv = predictions.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Results",
            data=csv,
            file_name=f"churn_predictions_{scored_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # Initial dashboard state
    st.markdown(
        """
<div class="empty-card">
    <div class="empty-icon">📈</div>
    <div class="empty-title">
        Ready to Generate Customer Intelligence
    </div>
    <div class="empty-description">
        Configure the prediction date, retention capacity
        and risk threshold in the sidebar, then generate
        this week's customer churn-risk rankings.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# Model performance and methodology
st.write("")

with st.expander("🧠 Model Performance & Methodology"):
    st.markdown("### Held-Out Test Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric("Test PR-AUC", "0.1474")
    m2.metric("Test ROC-AUC", "0.5729")
    m3.metric("PR-AUC Lift", "1.22×")

    st.markdown(
        """
### Operational Performance

Using a **Top-100 weekly retention strategy** on the held-out test period:

- **30.6%** of churn-positive customer-week observations were captured.
- Mean precision was approximately **15.7%**.
- Mean lift was approximately **1.31×**.
- Approximately **23.6%** of eligible customers were targeted.

### Prediction Architecture

`Aiven MySQL → Customer Snapshot → Point-in-Time Feature Engineering → Preprocessing → XGBoost → Risk Score → Customer Ranking`

### Modelling Strategy

The system uses:

- **Weekly customer scoring**
- **30-day churn prediction horizon**
- Point-in-time feature engineering
- Chronological train/validation/test splitting
- XGBoost as the selected production model
- Capacity-based customer prioritization

### Important Interpretation

The model is designed primarily as a **customer prioritization and
decision-support system**.

Risk scores are useful for ranking customers from higher to lower risk.
They should not currently be interpreted as perfectly calibrated
real-world churn probabilities.

This application uses historical data for portfolio demonstration.
"""
    )


# Footer
st.markdown(
    """
<div style="
    text-align:center;
    color:#94a3b8;
    font-size:.78rem;
    padding:2.5rem 0 1rem 0;
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
