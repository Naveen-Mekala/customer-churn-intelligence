
from pathlib import Path

import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_customer_churn.csv"
REPORT_PATH = ROOT / "reports" / "eda_summary.txt"


def main():
    df = pd.read_csv(DATA_PATH)

    # Convert numeric-looking columns to numeric values.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    print("=" * 60)
    print("CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Unique customer IDs: {df['customerID'].nunique()}")

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print(f"\nOverall churn rate: {df['Churn'].eq('Yes').mean():.2%}")

    print("\nChurn rate by contract:")
    print(
        df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda group: (group == "Yes").mean())
        .sort_values(ascending=False)
        .round(3)
    )

    print("\nChurn rate by tenure group:")
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
    )
    print(
        df.groupby("TenureGroup", observed=True)["Churn"]
        .apply(lambda group: (group == "Yes").mean())
        .round(3)
    )

    print("\nAverage monthly charges by churn status:")
    print(df.groupby("Churn", observed=True)["MonthlyCharges"].mean().round(2))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        report.write("CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS\n\n")
        report.write(f"Dataset shape: {df.shape}\n")
        report.write(f"Duplicate rows: {df.duplicated().sum()}\n")
        report.write(f"Overall churn rate: {df['Churn'].eq('Yes').mean():.2%}\n")
        report.write("\nChurn rate by contract:\n")
        report.write(
            df.groupby("Contract")["Churn"]
            .apply(lambda group: (group == "Yes").mean())
            .to_string()
        )
        report.write("\n\nChurn rate by tenure group:\n")
        report.write(
            df.groupby("TenureGroup", observed=True)["Churn"]
            .apply(lambda group: (group == "Yes").mean())
            .to_string()
        )

    print(f"\nSummary saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
