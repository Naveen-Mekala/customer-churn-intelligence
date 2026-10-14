from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import ensure_dataset
from src.score_customers import main as generate_scores


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"

SCORES_PATH = REPORTS_DIR / "customer_churn_scores.csv"
MODEL_PATH = REPORTS_DIR / "model_comparison.csv"

RAW_DATA_PATH = (
    ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
)

FIGURES_DIR = REPORTS_DIR / "figures"

@st.cache_data(show_spinner="Preparing customer churn data...")
def prepare_dashboard_files():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset only if it is missing.
    ensure_dataset()

    # Generate customer risk scores only if they are missing.
    if not SCORES_PATH.exists():
        generate_scores()

    # The model comparison report is required by the dashboard.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "reports/model_comparison.csv is missing. "
            "Run python src/train.py before launching the dashboard."
        )

    return True



# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Intelligence")
st.caption(
    "Customer analytics, churn prediction, and retention prioritization"
)


# =========================================================
# 3. LOAD DATA
# =========================================================

@st.cache_data
def load_scores():
    data = pd.read_csv(SCORES_PATH)

    required = [
        "customerID",
        "actual_churn",
        "churn_probability",
        "risk_category",
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    data["churn_probability"] = pd.to_numeric(
        data["churn_probability"], errors="coerce"
    )

    data = data.dropna(subset=["churn_probability"]).copy()

    data["churn_probability_pct"] = (
        data["churn_probability"] * 100
    )

    return data


@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_PATH)


@st.cache_data
def load_raw_data():
    data = pd.read_csv(RAW_DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    return data


try:
    prepare_dashboard_files()

    df = load_scores()
    model_results = load_model_results()
    raw_df = load_raw_data()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.info(
        "Check the dataset download, model scoring, and "
        "reports/model_comparison.csv setup."
    )
    st.stop()


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Executive Overview",
        "Churn Insights",
        "Customer Risk Explorer",
        "Retention Prioritization",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Customer Churn Intelligence")
st.sidebar.caption("Machine Learning Portfolio Project")


# =========================================================
# 5. EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    total_customers = len(df)

    historical_churn_rate = (
        df["actual_churn"].eq("Yes").mean() * 100
    )

    high_risk_count = df["risk_category"].eq("High").sum()

    average_predicted_risk = (
        df["churn_probability"].mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test Customers", f"{total_customers:,}")
    col2.metric(
        "Historical Churn Rate",
        f"{historical_churn_rate:.1f}%",
    )
    col3.metric("High-Risk Customers", f"{high_risk_count:,}")
    col4.metric(
        "Average Predicted Risk",
        f"{average_predicted_risk:.1f}%",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Risk Distribution")

        risk_counts = (
            df["risk_category"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        st.bar_chart(risk_counts)

    with right:
        st.subheader("Historical Churn Distribution")

        # IMPORTANT: use the same held-out customers as the
        # risk metrics, not all 7,043 original dataset rows.
        churn_counts = (
            df["actual_churn"]
            .value_counts()
            .reindex(["No", "Yes"], fill_value=0)
        )

        st.bar_chart(churn_counts)

    st.subheader("Key Business Observations")

    contract_churn = (
        raw_df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda values: values.eq("Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    if not contract_churn.empty:
        st.write(
            f"**Highest-churn contract group in the full dataset:** "
            f"{contract_churn.index[0]} "
            f"({contract_churn.iloc[0]:.1f}% historical churn)"
        )

    st.write(
        f"**High-risk test customers:** {high_risk_count:,} "
        f"out of {total_customers:,}."
    )

    st.info(
        "Customer metrics and churn distribution above use the held-out "
        "test subset. Contract observations use the full historical dataset. "
        "Risk predictions are estimates, not guarantees of future churn."
    )


# =========================================================
# 6. CHURN INSIGHTS
# =========================================================

elif page == "Churn Insights":

    st.header("Churn Insights")

    st.write(
        "Explore historical churn patterns across contracts, "
        "tenure, and monthly charges."
    )

    contract_chart = FIGURES_DIR / "churn_by_contract.png"
    distribution_chart = FIGURES_DIR / "churn_distribution.png"
    charges_chart = FIGURES_DIR / "monthly_charges_by_churn.png"

    if contract_chart.exists():
        st.subheader("Churn by Contract Type")
        st.image(str(contract_chart), use_container_width=True)
    else:
        st.warning("Contract chart not found. Run src/visualize.py.")

    if distribution_chart.exists():
        st.subheader("Churn Distribution")
        st.image(str(distribution_chart), use_container_width=True)
    else:
        st.warning("Churn distribution chart not found.")

    if charges_chart.exists():
        st.subheader("Monthly Charges by Churn Outcome")
        st.image(str(charges_chart), use_container_width=True)
    else:
        st.warning("Monthly charges chart not found.")

    st.divider()

    st.subheader("Churn Rate by Contract")

    contract_summary = (
        raw_df.groupby("Contract", observed=True)
        .agg(
            total_customers=("Churn", "size"),
            churned_customers=(
                "Churn",
                lambda values: values.eq("Yes").sum(),
            ),
        )
        .reset_index()
    )

    contract_summary["churn_rate_pct"] = (
        contract_summary["churned_customers"]
        / contract_summary["total_customers"]
        * 100
    )

    st.dataframe(
        contract_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Churn Rate by Tenure")

    tenure_df = raw_df.copy()

    tenure_df["tenure_group"] = pd.cut(
        tenure_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months",
        ],
    )

    tenure_summary = (
        tenure_df.groupby("tenure_group", observed=False)
        .agg(
            total_customers=("Churn", "size"),
            churn_rate_pct=(
                "Churn",
                lambda values: values.eq("Yes").mean() * 100,
            ),
        )
        .reset_index()
    )

    st.bar_chart(
        tenure_summary.set_index("tenure_group")["churn_rate_pct"]
    )

    st.dataframe(
        tenure_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 7. CUSTOMER RISK EXPLORER
# =========================================================

elif page == "Customer Risk Explorer":

    st.header("Customer Risk Explorer")

    st.write(
        "Filter held-out customers by predicted churn probability "
        "and risk category."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_risks = st.multiselect(
            "Risk categories",
            options=["High", "Medium", "Low"],
            default=["High"],
        )

    with col2:
        minimum_probability = st.slider(
            "Minimum churn probability (%)",
            min_value=0,
            max_value=100,
            value=50,
        )

    col3, col4 = st.columns(2)

    with col3:
        maximum_rows = st.number_input(
            "Maximum rows to display",
            min_value=10,
            max_value=max(10, len(df)),
            value=min(50, max(10, len(df))),
            step=10,
        )

    with col4:
        outcome_filter = st.selectbox(
            "Historical churn outcome",
            ["All", "Yes", "No"],
        )

    filtered_df = df[
        df["risk_category"].isin(selected_risks)
        & (df["churn_probability_pct"] >= minimum_probability)
    ].copy()

    if outcome_filter != "All":
        filtered_df = filtered_df[
            filtered_df["actual_churn"] == outcome_filter
        ]

    filtered_df = filtered_df.sort_values(
        "churn_probability",
        ascending=False,
    )

    st.metric(
        "Customers matching filters",
        f"{len(filtered_df):,}",
    )

    display_df = filtered_df.head(int(maximum_rows))

    st.dataframe(
        display_df[
            [
                "customerID",
                "churn_probability_pct",
                "risk_category",
                "actual_churn",
            ]
        ].rename(
            columns={
                "customerID": "Customer ID",
                "churn_probability_pct": "Churn Probability (%)",
                "risk_category": "Risk Category",
                "actual_churn": "Historical Churn",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Customers CSV",
        data=csv_data,
        file_name="filtered_churn_customers.csv",
        mime="text/csv",
    )

    st.caption(
        "Historical outcomes are included for retrospective evaluation. "
        "They would not be known when scoring genuinely new customers."
    )


# =========================================================
# 8. RETENTION PRIORITIZATION
# =========================================================

elif page == "Retention Prioritization":

    st.header("Retention Prioritization")

    st.write(
        "Simulate a retention team with a limited customer-contact budget."
    )

    max_budget = max(50, min(2000, len(df)))
    default_budget = min(500, len(df))

    contact_budget = st.slider(
        "Number of customers the team can contact",
        min_value=min(50, max_budget),
        max_value=max_budget,
        value=max(min(50, max_budget), default_budget),
        step=50 if max_budget >= 100 else 1,
    )

    priority_df = (
        df.sort_values("churn_probability", ascending=False)
        .head(contact_budget)
        .copy()
    )

    priority_df.insert(
        0,
        "contact_priority",
        range(1, len(priority_df) + 1),
    )

    historical_churn_rate = (
        priority_df["actual_churn"].eq("Yes").mean() * 100
    )

    average_risk = (
        priority_df["churn_probability"].mean() * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers Selected", f"{len(priority_df):,}")
    col2.metric(
        "Historical Churn Rate",
        f"{historical_churn_rate:.1f}%",
    )
    col3.metric("Average Predicted Risk", f"{average_risk:.1f}%")

    st.subheader("Priority Contact List")

    st.dataframe(
        priority_df[
            [
                "contact_priority",
                "customerID",
                "churn_probability_pct",
                "risk_category",
                "actual_churn",
            ]
        ].rename(
            columns={
                "contact_priority": "Priority",
                "customerID": "Customer ID",
                "churn_probability_pct": "Churn Probability (%)",
                "risk_category": "Risk Category",
                "actual_churn": "Historical Churn",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = priority_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Retention Priority List",
        data=csv_data,
        file_name="retention_priority_list.csv",
        mime="text/csv",
    )

    st.warning(
        "This is a prioritization demonstration using held-out customers. "
        "Historical churn does not establish that a retention intervention "
        "would prevent churn."
    )


# =========================================================
# 9. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Compare Logistic Regression and Random Forest using "
        "the saved model evaluation results."
    )

    required_metrics = [
        "model",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
    ]

    missing_metrics = [
        col for col in required_metrics if col not in model_results.columns
    ]

    if missing_metrics:
        st.error(
            "The model comparison file is missing: "
            + ", ".join(missing_metrics)
        )
        st.stop()

    st.dataframe(
        model_results[required_metrics].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Metric Comparison")

    metric = st.selectbox(
        "Choose a metric",
        [
            "test_precision",
            "test_recall",
            "test_f1",
            "test_pr_auc",
            "test_roc_auc",
        ],
        format_func=lambda value: (
            value.replace("test_", "").replace("_", " ").title()
        ),
    )

    st.bar_chart(
        model_results.set_index("model")[metric]
    )

    if "selected_threshold" in model_results.columns:
        st.subheader("Selected Classification Thresholds")

        st.dataframe(
            model_results[["model", "selected_threshold"]].round(3),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Recall measures the proportion of actual churners detected. "
        "Precision measures the proportion of flagged customers who "
        "actually churned. PR-AUC summarizes precision-recall performance "
        "across thresholds."
    )

    st.caption(
        "These metrics come from reports/model_comparison.csv. "
        "They may differ from the newer scoring evaluation in "
        "reports/scoring_evaluation.txt."
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn Intelligence | Machine Learning Portfolio Project"
)

st.caption(
    "Predictions are estimates based on historical data, "
    "not guarantees of future customer behavior."
)