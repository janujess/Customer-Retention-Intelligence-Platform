from pathlib import Path
import html
import sys

import numpy as np
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon=str(Path(__file__).parent / "favicon.png"),
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
# Brand mark: three ascending bars, the riskiest in coral (matches favicon.png)
BRAND_MARK = """<svg width="34" height="34" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
<rect width="512" height="512" rx="112" fill="#0b2545" stroke="#2a4a75" stroke-width="16"/>
<rect x="118" y="254" width="76" height="150" rx="18" fill="#5b7aa6"/>
<rect x="218" y="174" width="76" height="230" rx="18" fill="#c9d6ea"/>
<rect x="318" y="84" width="76" height="320" rx="18" fill="#ff6b4a"/>
</svg>"""

st.markdown(
    """
<style>
.stApp {
    background: #f7f9fc;
}

.block-container {
    padding-top: 3.2rem;
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
    -webkit-text-fill-color: #111827 !important;
    color-scheme: light;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stDateInputField"] *,
[data-testid="stSidebar"] input::-webkit-datetime-edit,
[data-testid="stSidebar"] input::-webkit-datetime-edit-year-field,
[data-testid="stSidebar"] input::-webkit-datetime-edit-month-field,
[data-testid="stSidebar"] input::-webkit-datetime-edit-day-field,
[data-testid="stSidebar"] input::-webkit-datetime-edit-text {
    color: #111827 !important;
}

.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: #ff5a36 !important;
    border-color: #ff5a36 !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #e5482a !important;
    border-color: #e5482a !important;
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
    border: none;
    background: radial-gradient(circle at 85% 0%, #1d4f91 0%, transparent 45%),
                linear-gradient(120deg, #0b2545 0%, #102a43 60%, #13315c 100%);
    box-shadow: 0 18px 40px rgba(11, 37, 69, .25);
    margin-bottom: 1.1rem;
}
.hero:before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    border-radius: 50%;
    background: rgba(255, 107, 74, .12);
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
    color: #ff8a6b;
}
.hero-brand span { color: #9fb3d1; font-weight: 650; }
.hero-status {
    font-size: .72rem;
    color: #dbeafe;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
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
    color: #ffffff;
    margin: 0 0 .75rem 0;
}
.hero-subtitle {
    color: #b6c6dd;
    font-size: .94rem;
    line-height: 1.65;
    max-width: 650px;
}
.radar {
    position: relative;
    display: inline-block;
    width: 62px;
    height: 62px;
    margin-right: 1rem;
    vertical-align: middle;
    border-radius: 50%;
    overflow: hidden;
    border: 1.5px solid rgba(255,107,74,.6);
    box-shadow: 0 0 22px rgba(255,107,74,.25), inset 0 0 14px rgba(255,107,74,.15);
    background:
        linear-gradient(rgba(255,107,74,.25), rgba(255,107,74,.25)) center / 1px 100% no-repeat,
        linear-gradient(rgba(255,107,74,.25), rgba(255,107,74,.25)) center / 100% 1px no-repeat,
        repeating-radial-gradient(circle, transparent 0 9px, rgba(255,107,74,.28) 9px 10px),
        rgba(255,107,74,.05);
}
.radar::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    margin: -2.5px 0 0 -2.5px;
    border-radius: 50%;
    background: #ff6b4a;
}
.radar-sweep {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(from 0deg,
        transparent 0deg 260deg,
        rgba(255,107,74,.08) 290deg,
        rgba(255,107,74,.55) 357deg,
        #ffb49f 360deg);
    animation: radar-sweep 2.4s linear infinite;
}
.radar-blip {
    position: absolute;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ffd2c4;
    box-shadow: 0 0 8px 2px rgba(255,107,74,.9);
    opacity: 0;
    animation: radar-blip 2.4s linear infinite both;
}
/* each blip lights up as the sweep line passes its angle */
.radar-blip.b1 { top: 26%; left: 64%; animation-delay: .26s; }
.radar-blip.b2 { top: 66%; left: 28%; animation-delay: 1.52s; }
@keyframes radar-sweep {
    to { transform: rotate(360deg); }
}
@keyframes radar-blip {
    0% { opacity: 1; transform: scale(1.4); }
    15% { opacity: .9; transform: scale(1); }
    70% { opacity: .1; }
    100% { opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .radar-sweep, .radar-blip { animation: none; }
    .radar-blip { opacity: .8; }
}
.radar-blip.b3 { top: 40%; left: 20%; animation-delay: 1.98s; }
.radar.radar-lg {
    width: 96px;
    height: 96px;
    margin-right: 0;
    flex: none;
}
.scan-card {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin: .6rem 0 1.2rem 0;
    padding: 1.3rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #0b2545 0%, #102a43 60%, #13315c 100%);
    box-shadow: 0 14px 32px rgba(11, 37, 69, .22);
}
.scan-title {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -.01em;
}
.scan-note {
    color: #9fb3d1;
    font-size: .85rem;
    margin-top: .25rem;
}
.message-preview {
    white-space: pre-wrap;
    background: #f7f9fc;
    border: 1px solid #e5eaf2;
    border-left: 3px solid #ff6b4a;
    border-radius: 10px;
    padding: .9rem 1rem;
    font-size: .88rem;
    line-height: 1.55;
    color: #1e293b;
    margin-bottom: .8rem;
}
.st-key-contact_overview button,
.st-key-contact_rankings button {
    background: #16a34a;
    border: 1px solid #16a34a;
    color: #ffffff;
    font-weight: 700;
    border-radius: 10px;
    min-height: 44px;
    animation: contact-pulse 1.8s ease-out infinite;
}
.st-key-contact_overview button:hover,
.st-key-contact_rankings button:hover {
    background: #15803d;
    border-color: #15803d;
    color: #ffffff;
}
.st-key-contact_overview button p,
.st-key-contact_rankings button p {
    color: #ffffff;
    font-weight: 700;
}
@keyframes contact-pulse {
    0% { box-shadow: 0 0 0 0 rgba(22, 163, 74, .55); }
    70% { box-shadow: 0 0 0 12px rgba(22, 163, 74, 0); }
    100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
}
@media (prefers-reduced-motion: reduce) {
    .st-key-contact_overview button,
    .st-key-contact_rankings button { animation: none; }
}
.hero-banner {
    display: inline-flex;
    align-items: center;
    gap: .6rem;
    margin-top: .4rem;
    padding: .6rem 1rem;
    border-radius: 10px;
    border-left: 3px solid #ff6b4a;
    background: rgba(255,255,255,.07);
    color: #dbe6f5;
    font-size: .85rem;
    font-weight: 600;
}
.hero-banner-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff6b4a;
    box-shadow: 0 0 0 4px rgba(255,107,74,.2);
    flex: none;
}
.hero-chips {
    display: flex;
    gap: .55rem;
    flex-wrap: wrap;
    margin-top: 1.15rem;
}
.hero-chip {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 10px;
    color: #e2ecf9;
    font-size: .75rem;
    font-weight: 700;
    padding: .55rem .7rem;
}
.hero-insights {
    display: grid;
    grid-template-columns: 1.1fr .9fr;
    gap: .7rem;
}
.hero-panel {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 16px;
    padding: .95rem;
    min-width: 0;
}
.hero-panel-label {
    color: #9fb3d1;
    font-size: .73rem;
    font-weight: 750;
    margin-bottom: .4rem;
}
.hero-panel-value {
    color: #ffffff;
    font-size: clamp(1.5rem, 2vw, 2.1rem);
    font-weight: 850;
    letter-spacing: -.035em;
    margin-top: .8rem;
}
.hero-panel-note {
    color: #8ea3c0;
    font-size: .69rem;
    margin-top: .4rem;
}
.hero-panel-value.hot { color: #ff6b4a; }
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
    min-height: 120px;
    box-shadow: 0 5px 18px rgba(15,23,42,.04);
    border-top: 3px solid #0b2545;
}

.kpi-card.hot { border-top-color: #ff6b4a; }
.kpi-card.hot .kpi-value { color: #e5482a; }

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
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -.03em;
    margin-top: .2rem;
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
<div style="display:flex;align-items:center;gap:.7rem;padding:.5rem 0 1rem 0;">
{BRAND_MARK}
<div>
<div style="font-size:1.35rem;font-weight:800;color:white;line-height:1.1;">RavenStack</div>
<div style="color:#93c5fd;font-size:.8rem;">Customer Intelligence</div>
</div>
</div>
""".replace("{BRAND_MARK}", BRAND_MARK),
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Prediction settings")

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
**Prediction horizon**  
30 days

**Scoring frequency**  
Weekly

**Production model**  
XGBoost

**Data source**  
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
        "Generate risk scores",
        type="primary",
        use_container_width=True,
    )


# Generate predictions
if run_prediction:
    try:
        # Radar loading card shown while the model scores customers
        scan_placeholder = st.empty()
        scan_placeholder.markdown(
            """
<div class="scan-card">
<span class="radar radar-lg"><span class="radar-sweep"></span><span class="radar-blip b1"></span><span class="radar-blip b2"></span><span class="radar-blip b3"></span></span>
<div>
<div class="scan-title">Scanning customers…</div>
<div class="scan-note">Building snapshots, engineering features and scoring with XGBoost</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        try:
            predictions = predictor.predict(
                prediction_date=str(prediction_date),
                threshold=threshold,
            )
        finally:
            scan_placeholder.empty()

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
header_date = "Not scored yet"
header_queue = "—"
header_queue_note = f"Capacity: {top_k} customers/week"
if "predictions" in st.session_state:
    header_data = st.session_state["predictions"]
    if header_data is not None and not header_data.empty:
        header_risk = f"{header_data['churn_probability'].max():.1%}"
        header_date = f"Scored {st.session_state.get('scored_date', '—')}"
        queue_avg = header_data["churn_probability"].head(top_k).mean()
        overall_avg = header_data["churn_probability"].mean()
        header_queue = f"{queue_avg / overall_avg:.1f}×"
        header_queue_note = f"Predicted risk, top {top_k} vs all customers"

header_placeholder.markdown(
    f"""
<div class="hero">
<div class="hero-top">
<div class="hero-brand">◈ RAVENSTACK <span>/ CUSTOMER INTELLIGENCE</span></div>
<div class="hero-status">{header_date}</div>
</div>
<div class="hero-body">
<div>
<div class="hero-title"><span class="radar"><span class="radar-sweep"></span><span class="radar-blip b1"></span><span class="radar-blip b2"></span></span>Churn Radar</div>
<div class="hero-banner">
<span class="hero-banner-dot"></span>
Churn risk scored weekly &nbsp;·&nbsp; 30-day horizon &nbsp;·&nbsp; ranked to fit your team's capacity
</div>
</div>
<div class="hero-insights">
<div class="hero-panel">
<div class="hero-panel-label">QUEUE VS AVERAGE RISK</div>
<div class="hero-panel-value">{header_queue}</div>
<div class="hero-panel-note">{header_queue_note}</div>
</div>
<div class="hero-panel">
<div class="hero-panel-label">HIGHEST RISK</div>
<div class="hero-panel-value hot">{header_risk}</div>
<div class="hero-panel-note">Top-ranked customer</div>
</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# Outreach (demo): contact the weekly queue by call, email or message
EMAIL_TEMPLATE = (
    "Hi there,\n\n"
    "We noticed you haven't been getting as much out of your account lately "
    "and wanted to check in. Is there anything we can help with?\n\n"
    "Reply to this email and we'll get back to you within a day.\n\n"
    "The RavenStack team"
)
MESSAGE_TEMPLATE = (
    "Hi from RavenStack! Just checking in on your account. "
    "Anything we can help with? Reply here and we'll get back to you."
)


def build_queue_groups(queue_size):
    """Recipient groups within the weekly queue, keyed by label -> (first rank, last rank)."""
    groups = {f"Whole queue (top {queue_size})": (1, queue_size)}

    for label, (first, last) in [
        ("Top 5", (1, 5)),
        ("Next 5 (ranks 6-10)", (6, 10)),
        ("Next 10 (ranks 11-20)", (11, 20)),
    ]:
        if first <= queue_size:
            groups[label] = (first, min(last, queue_size))

    if queue_size > 20:
        groups[f"The rest (ranks 21-{queue_size})"] = (21, queue_size)

    if queue_size >= 4:
        quartiles = np.array_split(np.arange(1, queue_size + 1), 4)
        for number, ranks in enumerate(quartiles, start=1):
            note = " - highest risk" if number == 1 else ""
            groups[f"Quartile {number} (ranks {ranks[0]}-{ranks[-1]}){note}"] = (
                int(ranks[0]),
                int(ranks[-1]),
            )

    return groups


def personalise(text, customer):
    return (
        text.replace("{customer_id}", str(customer["account_id"]))
        .replace("{rank}", f"#{int(customer['risk_rank'])}")
    )


def outreach_composer(queue):
    queue_size = len(queue)

    st.caption(
        "Demo mode: customer contact details aren't connected yet, "
        "so nothing is actually sent. Each send is logged below."
    )

    channel = st.radio(
        "How do you want to reach them?",
        ["Call", "Email", "Message"],
        horizontal=True,
        key="outreach_channel",
    )

    groups = build_queue_groups(queue_size)
    group_label = st.selectbox(
        "Who should get this?",
        list(groups) + ["Custom range"],
        key="outreach_group",
    )

    if group_label == "Custom range" and queue_size > 1:
        first, last = st.slider(
            "Ranks",
            min_value=1,
            max_value=queue_size,
            value=(1, min(5, queue_size)),
            key="outreach_range",
        )
    elif group_label == "Custom range":
        first, last = 1, 1
    else:
        first, last = groups[group_label]

    recipients = queue.iloc[first - 1:last]
    count = len(recipients)

    st.markdown(
        f"**{count} customer{'s' if count != 1 else ''}** "
        f"· ranks {first}-{last} "
        f"· average risk {recipients['churn_probability'].mean():.1%}"
    )

    recipient_table = pd.DataFrame(
        {
            "Rank": recipients["risk_rank"].astype(int),
            "Customer ID": recipients["account_id"].astype(str),
            "Risk Score": (recipients["churn_probability"] * 100).round(1),
            "Contact": "pending",
        }
    )

    with st.expander("See recipients"):
        st.dataframe(recipient_table, hide_index=True, use_container_width=True)

    log = st.session_state.setdefault("outreach_log", [])

    if channel == "Call":
        st.markdown("Download a call sheet for the team, then log the calls as assigned.")

        call_sheet = recipient_table.assign(Outcome="", Notes="")
        st.download_button(
            f"Download call sheet ({count} customers)",
            data=call_sheet.to_csv(index=False).encode("utf-8"),
            file_name=f"call_sheet_ranks_{first}_{last}.csv",
            mime="text/csv",
            key="outreach_call_sheet",
        )

        if st.button(f"Log {count} calls as assigned", type="primary", key="outreach_log_calls"):
            log.append(
                {
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Channel": "Call",
                    "Group": group_label,
                    "Customers": count,
                    "Message": "Call sheet assigned",
                }
            )
            st.success(f"{count} calls logged as assigned.")

    else:
        subject = ""
        if channel == "Email":
            subject = st.text_input(
                "Subject",
                value="Checking in on your account",
                key="outreach_subject",
            )

        body = st.text_area(
            "Message",
            value=EMAIL_TEMPLATE if channel == "Email" else MESSAGE_TEMPLATE,
            height=200 if channel == "Email" else 110,
            key=f"outreach_body_{channel}",
        )

        hint = "Personalise with {customer_id} and {rank}."
        if channel == "Message":
            parts = max(1, -(-len(body) // 160))
            hint = f"{len(body)} characters ({parts} SMS part{'s' if parts > 1 else ''}). " + hint
        st.caption(hint)

        if count:
            preview = personalise(body, recipients.iloc[0])
            st.markdown("**Preview for the first recipient**")
            preview_text = (
                f"Subject: {personalise(subject, recipients.iloc[0])}\n\n" if subject else ""
            ) + preview
            st.markdown(
                f'<div class="message-preview">{html.escape(preview_text)}</div>',
                unsafe_allow_html=True,
            )

        if st.button(
            f"Send {channel.lower()} to {count} customer{'s' if count != 1 else ''}",
            type="primary",
            key="outreach_send",
        ):
            if not body.strip():
                st.warning("Write a message before sending.")
            else:
                log.append(
                    {
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "Channel": channel,
                        "Group": group_label,
                        "Customers": count,
                        "Message": (subject or body)[:60],
                    }
                )
                st.success(
                    f"{count} {channel.lower()}{'s' if count != 1 else ''} queued "
                    "(demo - not actually sent)."
                )

    if log:
        st.markdown("**Outreach log (this session)**")
        st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

if _dialog is not None:

    @_dialog("Contact customers", width="large")
    def open_outreach(queue):
        outreach_composer(queue)

else:
    open_outreach = None


def contact_button(queue, key):
    if st.button(
        f"📞 📧  Contact top {len(queue)}",
        key=key,
        use_container_width=True,
    ):
        if open_outreach is not None:
            open_outreach(queue)
        else:
            with st.expander("Contact customers", expanded=True):
                outreach_composer(queue)


# Dashboard navigation
overview_tab, rankings_tab, lookup_tab, export_tab, model_tab = st.tabs(
    [
        "Overview",
        "Risk rankings",
        "Customer lookup",
        "Export",
        "Model & methodology",
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
        # Section title and the Contact button share one row
        title_column, contact_column = st.columns([4, 1], vertical_alignment="center")

        with title_column:
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

        with contact_column:
            contact_button(top_customers, key="contact_overview")

        kpis = [
            (
                "",
                "Active Customers",
                f"{total_customers:,}",
                "Customers scored",
            ),
            (
                "",
                "Retention Capacity",
                f"{selected_customers:,}",
                "Customers prioritized",
            ),
            (
                "",
                "Average Risk Score",
                f"{average_risk:.1%}",
                "Portfolio average",
            ),
            (
                "",
                "Highest Risk Score",
                f"{highest_risk:.1%}",
                "Highest-ranked customer",
            ),
            (
                "",
                "Above Threshold",
                f"{above_threshold:,}",
                f"Threshold ≥ {threshold:.0%}",
            ),
        ]

        columns = st.columns(5)

        for index, (column, (icon, label, value, note)) in enumerate(zip(columns, kpis)):
            card_class = "kpi-card hot" if index in (3, 4) else "kpi-card"
            with column:
                st.markdown(
                    f"""
<div class="{card_class}">
<div class="kpi-label">{label}</div>
<div class="kpi-value">{value}</div>
<div class="kpi-note">{note}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.write("")

        st.markdown("### How risk is spread")

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
            color="#0b2545",
        )

    else:
        st.markdown(
            """
<div class="empty-card">
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
        title_column, contact_column = st.columns([4, 1], vertical_alignment="bottom")

        with title_column:
            st.markdown(
                f"### Top {selected_customers} customers by churn risk"
            )

            st.caption(
                "Highest-risk customers recommended for retention review."
            )

        with contact_column:
            contact_button(top_customers, key="contact_rankings")

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
        st.markdown("### Customer lookup")

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
                "This customer is above the selected risk threshold."
            )
        else:
            st.success(
                "This customer is below the selected risk threshold."
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
    st.markdown("### Download predictions")

    if has_predictions:
        st.caption(
            "Export the complete scored customer population "
            "for retention analysis."
        )

        csv = predictions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download all predictions (CSV)",
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
    st.markdown("### Model performance & methodology")

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