# -*- coding: utf-8 -*-
"""Compact home-equity loan default model using logistic regression.

Place ``hmeq (1).xlsx`` beside this script and run:
    python Home_Equity_Default_Model.py

Optional:
    python Home_Equity_Default_Model.py --data path/to/hmeq.xlsx --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET = "BAD"
DEFAULT_DATA_FILE = "hmeq (1).xlsx"

# Five economically distinct risk factors: prior delinquency, derogatory
# history, leverage, recent credit inquiries, and age of credit history.
# They are fixed before model validation rather than selected on the test folds.
FIVE_VARIABLES = ["DELINQ", "DEROG", "DEBTINC", "NINQ", "CLAGE"]
THREE_VARIABLES = ["DELINQ", "DEROG", "DEBTINC"]


def parse_args() -> argparse.Namespace:
    """Read optional input and output paths."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / DEFAULT_DATA_FILE,
        help="Path to the HMEQ Excel or CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Folder for result tables and figures.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and validate the required fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Place '{DEFAULT_DATA_FILE}' beside the script or use --data."
        )

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        raise ValueError("The data file must be CSV or Excel.")

    required = set(FIVE_VARIABLES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    return data


def build_model() -> Pipeline:
    """Create a leakage-safe preprocessing and estimation pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
        ]
    )


def calculate_vif(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate VIF after median imputation."""
    imputed = SimpleImputer(strategy="median").fit_transform(data[features])
    matrix = pd.DataFrame(imputed, columns=features)
    rows = []
    for name in features:
        other_features = [feature for feature in features if feature != name]
        r_squared = LinearRegression().fit(matrix[other_features], matrix[name]).score(
            matrix[other_features], matrix[name]
        )
        vif = np.inf if np.isclose(r_squared, 1.0) else 1.0 / (1.0 - r_squared)
        rows.append({"variable": name, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def evaluate_model(
    data: pd.DataFrame,
    features: list[str],
    model_name: str,
    cv: StratifiedKFold,
) -> dict:
    """Produce out-of-fold predictions and fit a final explanatory model."""
    X = data[features]
    y = data[TARGET]
    model = build_model()

    fold_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    oof_pd = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_class = (oof_pd >= 0.50).astype(int)

    # The final full-sample fit is used only to interpret standardized drivers.
    model.fit(X, y)
    coefficients = model.named_steps["logistic"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "variable": features,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_1sd": np.exp(coefficients),
        }
    ).sort_values("standardized_coefficient", key=abs, ascending=False)

    tn, fp, fn, tp = confusion_matrix(y, oof_class, labels=[0, 1]).ravel()
    metrics = {
        "model": model_name,
        "n_features": len(features),
        "mean_fold_auc": fold_auc.mean(),
        "fold_auc_std": fold_auc.std(),
        "oof_auc": roc_auc_score(y, oof_pd),
        "oof_brier_score": brier_score_loss(y, oof_pd),
        "sensitivity_at_0.50": tp / (tp + fn),
        "specificity_at_0.50": tn / (tn + fp),
    }

    return {
        "probabilities": oof_pd,
        "classes": oof_class,
        "coefficients": coefficient_table,
        "metrics": metrics,
    }


def make_risk_deciles(y: pd.Series, predicted_pd: np.ndarray) -> pd.DataFrame:
    """Compare predicted and observed default rates by risk rank."""
    calibration = pd.DataFrame({"actual_bad": y.to_numpy(), "predicted_pd": predicted_pd})
    calibration["risk_decile"] = pd.qcut(
        calibration["predicted_pd"], q=10, labels=False, duplicates="drop"
    ) + 1
    return (
        calibration.groupby("risk_decile", observed=True)
        .agg(
            accounts=("actual_bad", "size"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("actual_bad", "mean"),
        )
        .reset_index()
    )


def save_outputs(
    data: pd.DataFrame,
    five_result: dict,
    three_result: dict,
    output_dir: Path,
) -> None:
    """Save the central result tables and four publication-ready figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = data[TARGET]

    comparison = pd.DataFrame([three_result["metrics"], five_result["metrics"]])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    five_result["coefficients"].to_csv(output_dir / "coefficient_interpretation.csv", index=False)
    calculate_vif(data, FIVE_VARIABLES).to_csv(output_dir / "vif_table.csv", index=False)

    predictions = pd.DataFrame(
        {
            "actual_bad": y,
            "oof_predicted_pd": five_result["probabilities"],
            "predicted_class_at_0.50": five_result["classes"],
        }
    )
    if "CustomerID" in data.columns:
        predictions.insert(0, "CustomerID", data["CustomerID"].to_numpy())
    predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)

    report = classification_report(
        y, five_result["classes"], output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report_at_0.50.csv")

    deciles = make_risk_deciles(y, five_result["probabilities"])
    deciles.to_csv(output_dir / "risk_decile_calibration.csv", index=False)

    plt.figure(figsize=(7, 5))
    for result, label in [
        (three_result, "Three-variable benchmark"),
        (five_result, "Five-variable model"),
    ]:
        fpr, tpr, _ = roc_curve(y, result["probabilities"])
        auc = result["metrics"]["oof_auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Out-of-Fold ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_model_comparison.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        five_result["classes"],
        display_labels=["Non-default", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Out-of-Fold Classification at 0.50")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 5))
    plt.plot(deciles["risk_decile"], deciles["mean_predicted_pd"], marker="o", label="Predicted PD")
    plt.plot(
        deciles["risk_decile"],
        deciles["observed_default_rate"],
        marker="o",
        label="Observed default rate",
    )
    plt.xlabel("Risk decile (1 = lowest risk)")
    plt.ylabel("Default rate")
    plt.title("Calibration by Risk Decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "risk_decile_calibration.png", dpi=200)
    plt.close()

    plot_data = five_result["coefficients"].sort_values("standardized_coefficient")
    colors = np.where(plot_data["standardized_coefficient"] >= 0, "#B22222", "#2E5EAA")
    plt.figure(figsize=(7, 4.5))
    plt.barh(plot_data["variable"], plot_data["standardized_coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized logistic coefficient")
    plt.title("Direction and Strength of Default-Risk Drivers")
    plt.tight_layout()
    plt.savefig(output_dir / "standardized_coefficients.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    five_result = evaluate_model(data, FIVE_VARIABLES, "Five-variable model", cv)
    three_result = evaluate_model(data, THREE_VARIABLES, "Three-variable benchmark", cv)
    save_outputs(data, five_result, three_result, args.output_dir)

    comparison = pd.DataFrame([three_result["metrics"], five_result["metrics"]])
    print(f"Observations: {len(data):,}")
    print(f"Observed default rate: {data[TARGET].mean():.2%}")
    print("\nCross-validated model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFive-variable coefficient interpretation:")
    print(five_result["coefficients"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs saved to: {args.output_dir.resolve()}")
    print(
        "Note: the 0.50 cutoff is illustrative. A production cutoff should reflect "
        "approval strategy, false-negative costs, and calibration."
    )


if __name__ == "__main__":
    main()