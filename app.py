import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Intelligence Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS (BLOOMBERG-STYLE DARK UI)
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    padding: 15px;
    border-radius: 10px;
}

.css-1d391kg {
    background-color: #0E1117;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

# risk_scores = pd.read_csv("woe_xgb_lab1.csv")

# risk_scores = risk_scores.rename(columns={
#     "pred_prob": "DefaultProbability",
#     "true_label": "ActualDefault"
# })

# def assign_risk_tier(p):
#     if p >= 0.8:
#         return "Critical Risk"
#     elif p >= 0.6:
#         return "High Risk"
#     elif p >= 0.4:
#         return "Moderate Risk"
#     else:
#         return "Low Risk"

# risk_scores["RiskTier"] = risk_scores["DefaultProbability"].apply(assign_risk_tier)

DATASETS = {
    "2 Level CatBoost + OOF": "cb_meta_oof_lab2.csv",
    "CatBoost + OOF": "cb_oof_lab2.csv",
    "LightGBM + Focal Loss + OOF": "lgbm_fl_oof_lab2.csv",
    "2 Level LightGBM + OOF": "lgbm_meta_oof_lab2.csv",
    "LightGBM Baseline": "lgbm_oof_lab2.csv",
    "XGBoost + Focal Loss + OOF": "xgb_fl_oof_lab2.csv",
    "XGBoost Baseline": "xgb_oof_lab2.csv",
    "Random Forest Baseline": "rf_oof_lab2.csv",
    "Logistic Regression Baseline": "lr_oof_lab2.csv",

    # "Risk Backup": "risk_scores_backup.csv"
}

def build_risk_tier(df):
    def assign(p):
        if p >= 0.8:
            return "Critical Risk"
        elif p >= 0.6:
            return "High Risk"
        elif p >= 0.4:
            return "Moderate Risk"
        else:
            return "Low Risk"
    df["RiskTier"] = df["DefaultProbability"].apply(assign)
    return df

data_dict = {}
for name, path in DATASETS.items():
    df = pd.read_csv(path)
    df = df.rename(columns={
        "pred_prob": "DefaultProbability",
        "true_label": "ActualDefault"
    })
    df = build_risk_tier(df)
    data_dict[name] = df


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Risk Intelligence Engine")

st.sidebar.header("📊 模型选择")
selected_model = st.sidebar.selectbox(
    "选择风险数据集",
    list(data_dict.keys())
)

risk_scores = data_dict[selected_model]

# st.sidebar.markdown("""
# ### System Overview

# - Risk Segmentation
# - Threshold Optimization
# - Portfolio Analytics
# - Institutional Risk Scoring
# """)

st.sidebar.markdown("---")

selected_risk = st.sidebar.selectbox(
    "Filter Risk Tier",
    [
        "All",
        "Low Risk",
        "Moderate Risk",
        "High Risk",
        "Critical Risk"
    ]
)


# ---------------------------------------------------
# HIGH RISK FILTER
# ---------------------------------------------------

high_risk_accounts = risk_scores[
    risk_scores['RiskTier'].isin(
        ['High Risk', 'Critical Risk']
    )
]

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("📈 Credit Risk Intelligence Dashboard")

st.caption(f"当前模型：{selected_model}")

st.markdown("""
Institutional-grade AI-powered borrower delinquency detection and portfolio surveillance system.
""")


# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------

if selected_risk != "All":

    filtered_data = risk_scores[
        risk_scores['RiskTier'] == selected_risk
    ]

else:

    filtered_data = risk_scores.copy()

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Borrowers",
    f"{len(filtered_data):,}"
)

col2.metric(
    "Average Default Probability",
    round(
        filtered_data['DefaultProbability'].mean(),
        3
    )
)

col3.metric(
    "High Risk Accounts",
    len(
        filtered_data[
            filtered_data['RiskTier']
            == 'High Risk'
        ]
    )
)

col4.metric(
    "Critical Accounts",
    len(
        filtered_data[
            filtered_data['RiskTier']
            == 'Critical Risk'
        ]
    )
)

# ---------------------------------------------------
# ALERT BOX
# ---------------------------------------------------

st.error(
    f"⚠️ Portfolio monitoring detected {len(high_risk_accounts)} elevated-risk borrower accounts."
)

# ---------------------------------------------------
# CHARTS SECTION
# ---------------------------------------------------

left_col, right_col = st.columns(2)

# ---------------------------------------------------
# PORTFOLIO RISK DISTRIBUTION
# ---------------------------------------------------

with left_col:

    risk_counts = (
        filtered_data['RiskTier']
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ['RiskTier', 'Count']

    fig = px.bar(
        risk_counts,
        x='RiskTier',
        y='Count',
        color='RiskTier',
        title='Portfolio Risk Distribution'
    )

    fig.update_layout(
        template='plotly_dark',
        height=450
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

# ---------------------------------------------------
# DEFAULT PROBABILITY DISTRIBUTION
# ---------------------------------------------------

with right_col:

    fig2 = px.histogram(
        filtered_data,
        x='DefaultProbability',
        nbins=40,
        title='Default Probability Distribution'
    )

    fig2.update_layout(
        template='plotly_dark',
        height=450
    )

    st.plotly_chart(
        fig2,
        width='stretch'
    )

# ---------------------------------------------------
# RISK TIER PIE CHART
# ---------------------------------------------------

left_col2, right_col2 = st.columns(2)

with left_col2:

    pie = px.pie(
        filtered_data,
        names='RiskTier',
        title='Risk Tier Composition'
    )

    pie.update_layout(
        template='plotly_dark',
        height=450
    )

    st.plotly_chart(
        pie,
        width='stretch'
    )

# ---------------------------------------------------
# TOP DEFAULT PROBABILITIES
# ---------------------------------------------------

with right_col2:

    top_risk = filtered_data.sort_values(
        by='DefaultProbability',
        ascending=False
    ).head(10)

    fig3 = px.bar(
        top_risk,
        x=top_risk.index.astype(str),
        y='DefaultProbability',
        title='Top Borrower Risk Scores'
    )

    fig3.update_layout(
        template='plotly_dark',
        height=450,
        xaxis_title='Borrower Index'
    )

    st.plotly_chart(
        fig3,
        width='stretch'
    )

# ---------------------------------------------------
# HIGH RISK TABLE
# ---------------------------------------------------

st.subheader("🚨 High Risk Borrower Surveillance")

display_table = high_risk_accounts[
    [
        'ActualDefault',
        'DefaultProbability',
        'RiskTier'
    ]
].sort_values(
    by='DefaultProbability',
    ascending=False
)

st.dataframe(
    display_table.head(25),
    width='stretch'
)

# ---------------------------------------------------
# SYSTEM FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption("""
Credit Risk Intelligence System • Explainable AI • Institutional Portfolio Analytics
""")