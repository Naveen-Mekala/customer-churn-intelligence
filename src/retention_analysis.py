from pathlib import Path

import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

SCORES_PATH = (
    ROOT_DIR
    / "reports"
    / "customer_churn_scores.csv"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "reports"
    / "retention_priority_500.csv"
)

CONTACT_BUDGET = 500


# =========================================================
# 2. LOAD HELD-OUT CUSTOMER SCORES
# =========================================================

df = pd.read_csv(SCORES_PATH)

required_columns = [
    "customerID",
    "actual_churn",
    "churn_probability",
    "risk_category",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("The scoring file contains no customers.")


# =========================================================
# 3. RANK CUSTOMERS BY PREDICTED RISK
# =========================================================

df = df.sort_values(
    "churn_probability",
    ascending=False,
).reset_index(drop=True)

budget = min(CONTACT_BUDGET, len(df))

priority_customers = df.head(budget).copy()

priority_customers.insert(
    0,
    "contact_priority",
    range(1, len(priority_customers) + 1),
)


# =========================================================
# 4. SAVE PRIORITY LIST
# =========================================================

priority_customers.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =========================================================
# 5. SUMMARIZE RESULTS
# =========================================================

historical_churn_rate = (
    priority_customers["actual_churn"]
    .eq("Yes")
    .mean()
)

average_predicted_risk = (
    priority_customers["churn_probability"].mean()
)

print("RETENTION PRIORITIZATION REPORT")
print("=" * 40)

print(f"Available test customers: {len(df):,}")
print(f"Contact budget: {CONTACT_BUDGET:,}")
print(f"Customers selected: {len(priority_customers):,}")

print(
    f"\nAverage predicted risk: "
    f"{average_predicted_risk:.2%}"
)

print(
    f"Historical churn rate in selected group: "
    f"{historical_churn_rate:.2%}"
)

print("\nRisk categories:")
print(
    priority_customers["risk_category"]
    .value_counts()
)

print("\nHistorical outcomes:")
print(
    priority_customers["actual_churn"]
    .value_counts()
)

print("\nTop 10 priority customers:")
print(
    priority_customers.head(10).to_string(index=False)
)

print(f"\nSaved to: {OUTPUT_PATH}")

print(
    "\nInterpretation: this is a ranking of held-out customers "
    "for analytical demonstration. It does not measure the causal "
    "effect of retention campaigns."
)