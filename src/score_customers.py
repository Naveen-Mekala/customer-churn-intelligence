from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # =========================================================
    # 1. PROJECT PATHS
    # =========================================================

    ROOT_DIR = Path(__file__).resolve().parents[1]

    DATA_PATH = (
        ROOT_DIR
        / "data"
        / "raw"
        / "telco_customer_churn.csv"
    )

    REPORTS_DIR = ROOT_DIR / "reports"

    OUTPUT_PATH = (
        REPORTS_DIR
        / "customer_churn_scores.csv"
    )

    REPORT_PATH = (
        REPORTS_DIR
        / "scoring_evaluation.txt"
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================
    # 2. LOAD AND PREPARE DATA
    # =========================================================

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df["Churn"] = df["Churn"].map(
        {"No": 0, "Yes": 1}
    )

    customer_ids = df["customerID"].copy()

    X = df.drop(
        columns=["Churn", "customerID"]
    )

    y = df["Churn"]


    # =========================================================
    # 3. SPLIT DATA BEFORE TRAINING
    # =========================================================

    X_train, X_test, y_train, y_test, ids_train, ids_test = (
        train_test_split(
            X,
            y,
            customer_ids,
            test_size=0.20,
            stratify=y,
            random_state=2026,
        )
    )

    print(f"Training customers: {len(X_train)}")
    print(f"Test customers: {len(X_test)}")


    # =========================================================
    # 4. IDENTIFY FEATURE TYPES
    # =========================================================

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()


    # =========================================================
    # 5. BUILD PREPROCESSING PIPELINES
    # =========================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


    # =========================================================
    # 6. TRAIN MODEL
    # =========================================================

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)


    # =========================================================
    # 7. PREDICT TEST CUSTOMERS ONLY
    # =========================================================

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # 0.50 is used here as a baseline threshold.
    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # =========================================================
    # 8. EVALUATE HELD-OUT TEST PREDICTIONS
    # =========================================================

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    conf_matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0,
    )


    # =========================================================
    # 9. CREATE OUT-OF-SAMPLE SCORES
    # =========================================================

    scores = pd.DataFrame(
        {
            "customerID": ids_test.to_numpy(),
            "actual_churn": y_test.map(
                {0: "No", 1: "Yes"}
            ).to_numpy(),
            "churn_probability": probabilities,
            "risk_category": pd.cut(
                probabilities,
                bins=[-0.001, 0.30, 0.60, 1.00],
                labels=["Low", "Medium", "High"],
            ).astype(str),
        }
    )

    scores["churn_probability"] = (
        scores["churn_probability"].round(4)
    )

    scores = scores.sort_values(
        "churn_probability",
        ascending=False,
    )

    scores.to_csv(
        OUTPUT_PATH,
        index=False,
    )


    # =========================================================
    # 10. SAVE EVALUATION REPORT
    # =========================================================

    report_text = f"""
    CUSTOMER CHURN SCORING EVALUATION
    =================================

    Dataset: IBM Telco Customer Churn
    Training customers: {len(X_train)}
    Test customers: {len(X_test)}
    Decision threshold: {threshold:.2f}

    Precision: {precision:.4f}
    Recall: {recall:.4f}
    PR-AUC: {pr_auc:.4f}
    ROC-AUC: {roc_auc:.4f}

    Confusion matrix:
    {conf_matrix}

    Classification report:
    {report}

    Important limitations:
    - These predictions are for a held-out test split.
    - The 0.50 threshold is a baseline, not a business-optimized threshold.
    - Risk categories are illustrative probability bands.
    - Historical churn prediction does not prove that an intervention
      will prevent a customer from leaving.
    """

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)


    # =========================================================
    # 11. PRINT SUMMARY
    # =========================================================

    print("\nOut-of-sample scoring completed.")

    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(conf_matrix)

    print("\nRisk category counts:")
    print(scores["risk_category"].value_counts())

    print("\nTop 10 highest-risk test customers:")
    print(scores.head(10).to_string(index=False))

    print(f"\nScores saved to: {OUTPUT_PATH}")
    print(f"Evaluation report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()
