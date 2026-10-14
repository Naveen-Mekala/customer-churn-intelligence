
from pathlib import Path

import kagglehub
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "telco_customer_churn.csv"

DATASET_ID = "blastchar/telco-customer-churn"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(
        kagglehub.dataset_download(DATASET_ID)
    )

    csv_files = list(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {dataset_path}"
        )

    source_csv = next(
        (
            path
            for path in csv_files
            if "telco" in path.name.lower()
            and "churn" in path.name.lower()
        ),
        csv_files[0],
    )

    # Validate the downloaded file before copying it.
    df = pd.read_csv(source_csv)

    required_columns = {
        "customerID",
        "Churn",
        "TotalCharges",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset downloaded and validated: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
