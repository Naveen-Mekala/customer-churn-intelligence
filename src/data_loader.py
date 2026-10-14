
from pathlib import Path

import kagglehub
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"


def ensure_dataset() -> Path:
    """Return the local dataset path, downloading it if necessary."""
    if DATA_PATH.exists():
        return DATA_PATH

    dataset_dir = Path(
        kagglehub.dataset_download(
            "blastchar/telco-customer-churn"
        )
    )

    candidates = list(dataset_dir.rglob("*.csv"))
    source = next(
        (
            path for path in candidates
            if "telco" in path.name.lower()
            and "churn" in path.name.lower()
        ),
        candidates[0] if candidates else None,
    )

    if source is None:
        raise FileNotFoundError(
            "KaggleHub downloaded no CSV file for the Telco Churn dataset."
        )

    data = pd.read_csv(source)

    required = {"customerID", "Churn", "TotalCharges"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)

    return DATA_PATH
