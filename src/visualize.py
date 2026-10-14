
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUTPUT_DIR = ROOT / "reports" / "figures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid", context="talk")

    # 1. Overall churn distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Churn rate by contract
    contract_rates = (
        df.groupby("Contract")["Churn"]
        .apply(lambda s: s.eq("Yes").mean() * 100)
        .sort_values(ascending=False)
        .reset_index(name="ChurnRate")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=contract_rates,
        x="Contract",
        y="ChurnRate",
        ax=ax,
    )
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    # 3. Monthly charges by churn status
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_charges_by_churn.png", dpi=160)
    plt.close(fig)

    print(f"Charts saved to: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
