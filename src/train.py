from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
REPORTS_DIR = ROOT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load and prepare the dataset
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# 2. Split into training, validation, and test sets
# 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=2026,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=2026,
)

print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")


# 3. Build preprocessing pipelines
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# 4. Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=2026,
        n_jobs=-1,
    ),
}


# 5. Choose a threshold using validation data only
thresholds = [
    round(value, 2)
    for value in [0.20 + i * 0.05 for i in range(13)]
]

results = []
threshold_details = []
final_reports = []

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Fit using training data only
    pipeline.fit(X_train, y_train)

    # Select threshold using validation data
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_val_f1 = -1.0

    for threshold in thresholds:
        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0,
        )

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_threshold = threshold

    # Evaluate the selected threshold on the test set
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    pr_auc = average_precision_score(
        y_test, test_probabilities
    )
    roc_auc = roc_auc_score(
        y_test, test_probabilities
    )

    results.append(
        {
            "model": model_name,
            "selected_threshold": best_threshold,
            "validation_f1": best_val_f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_pr_auc": pr_auc,
            "test_roc_auc": roc_auc,
        }
    )

    threshold_details.append(
        f"\nModel: {model_name}\n"
        f"Selected threshold: {best_threshold:.2f}\n"
        f"Validation F1: {best_val_f1:.4f}\n"
        f"Test precision: {precision:.4f}\n"
        f"Test recall: {recall:.4f}\n"
        f"Test F1: {f1:.4f}\n"
        f"Test PR-AUC: {pr_auc:.4f}\n"
        f"Test ROC-AUC: {roc_auc:.4f}\n"
        f"Test confusion matrix:\n"
        f"{confusion_matrix(y_test, test_predictions)}\n"
        f"Test classification report:\n"
        f"{classification_report(y_test, test_predictions, zero_division=0)}\n"
    )

    final_reports.append(
        f"{model_name}\n"
        f"{classification_report(y_test, test_predictions, zero_division=0)}"
    )

    print(f"\n{model_name}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_val_f1:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))


# 6. Save evaluation results
comparison = pd.DataFrame(results)
comparison.to_csv(
    REPORTS_DIR / "model_comparison.csv",
    index=False,
)

with open(
    REPORTS_DIR / "model_evaluation.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("CUSTOMER CHURN MODEL EVALUATION\n")
    file.write("=" * 40 + "\n")
    file.write(
        "Data split: 60% training, 20% validation, 20% testing\n"
    )
    file.write(
        "Threshold selection: highest validation F1 "
        "among thresholds from 0.20 to 0.80\n"
    )
    file.write(
        "\nIMPORTANT: Test results are for evaluation, "
        "not threshold selection.\n"
    )
    file.write("\n".join(threshold_details))

print("\nModel comparison:")
print(comparison.to_string(index=False))

print("\nSaved reports:")
print(REPORTS_DIR / "model_comparison.csv")
print(REPORTS_DIR / "model_evaluation.txt")
