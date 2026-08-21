# Home Equity Loan Default Prediction

An interpretable credit-risk modeling project that estimates the probability of default for home equity loan applicants using logistic regression. The project focuses on a small set of economically meaningful borrower risk factors and tests whether they improve discrimination relative to a simpler benchmark model.

## Project Objective

The goal is to answer three practical credit-risk questions:

1. Which borrower characteristics are most strongly associated with default risk?
2. Can a focused five-variable logistic model meaningfully separate higher-risk from lower-risk borrowers?
3. Does the additional complexity improve performance relative to a simpler three-variable model?

The project emphasizes interpretability, validation, and reproducibility rather than building the most complex predictive model possible.

## Dataset

The analysis uses the Home Equity Loan (`HMEQ`) dataset. The target variable is `BAD`, where `1` represents a bad loan outcome and `0` represents a non-default outcome.

The final model uses five credit-risk drivers:

| Variable | Credit-risk interpretation |
|---|---|
| `DELINQ` | Number of delinquent credit lines |
| `DEROG` | Number of major derogatory reports |
| `DEBTINC` | Debt-to-income ratio |
| `NINQ` | Number of recent credit inquiries |
| `CLAGE` | Age of the oldest credit line |

These variables cover repayment history, adverse credit events, leverage, recent credit demand, and length of credit history. A simpler benchmark uses only `DELINQ`, `DEROG`, and `DEBTINC`.

The dataset is not included in this repository. The script accepts either an Excel or CSV version through the `--data` argument.

## Modeling Approach

### 1. Data preparation

- Validate the target and required predictor columns.
- Impute missing numerical values using the median.
- Standardize predictors before estimating logistic regression.
- Keep imputation and scaling inside the modeling pipeline so they are fitted separately within each cross-validation fold.

### 2. Multicollinearity review

Pairwise relationships and variance inflation factors are used to check whether the selected predictors contain excessive overlapping information. In the original analysis, no selected variable had a VIF above 5, suggesting no severe multicollinearity concern.

### 3. Model estimation

Two logistic-regression models are compared:

- **Three-variable benchmark:** `DELINQ`, `DEROG`, and `DEBTINC`
- **Five-variable model:** adds `NINQ` and `CLAGE`

Standardized coefficients allow the relative strength of the five risk drivers to be compared. Odds ratios show how the estimated default odds change for a one-standard-deviation increase in each variable.

### 4. Model validation

Performance is evaluated using stratified five-fold cross-validation. The script generates out-of-fold probabilities, meaning each borrower's predicted probability is produced by a model that was not trained on that observation.

The main validation measures are:

- **ROC-AUC:** Measures the model's ability to rank bad loans above good loans.
- **Brier score:** Measures the accuracy of the predicted probabilities; lower is better.
- **Sensitivity and specificity:** Summarize classification performance at an illustrative 0.50 cutoff.
- **Risk-decile calibration:** Compares average predicted PD with the observed default rate across ranked risk groups.

## Key Results

The original model run produced the following cross-validated discrimination results:

| Model | Mean cross-validated AUC |
|---|---:|
| Three-variable benchmark | 0.736 |
| Five-variable model | 0.785 |
| Improvement | approximately 0.049 |

The five-variable model provided a meaningful improvement over the simpler benchmark while remaining easy to explain.

`DELINQ` was the strongest default-risk driver in the original run. It had the largest absolute standardized coefficient and an estimated odds ratio of approximately **2.19** for a one-standard-deviation increase. This is economically intuitive: a history of delinquent credit lines is a strong signal of future repayment risk.

These results should be reproduced from the included script before publication because exact values can vary with the dataset version and validation setup.

## Generated Outputs

Running the script creates an `outputs` folder containing:

- `model_comparison.csv` — validation metrics for both models
- `coefficient_interpretation.csv` — standardized coefficients and odds ratios
- `vif_table.csv` — multicollinearity diagnostics
- `out_of_fold_predictions.csv` — actual outcomes and cross-validated predicted PDs
- `classification_report_at_0.50.csv` — threshold-based classification metrics
- `risk_decile_calibration.csv` — predicted and observed default rates by risk group
- `roc_model_comparison.png` — ROC curves for both models
- `confusion_matrix.png` — out-of-fold confusion matrix at the 0.50 cutoff
- `risk_decile_calibration.png` — predicted versus observed risk by decile
- `standardized_coefficients.png` — direction and relative strength of risk drivers

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

Run the model with an Excel file:

```bash
python Home_Equity_Default_Model.py --data "path/to/hmeq.xlsx"
```

Or specify a separate output folder:

```bash
python Home_Equity_Default_Model.py \
  --data "path/to/hmeq.xlsx" \
  --output-dir "outputs"
```

## Repository Structure

```text
Home-Equity-Default-Model/
├── Home_Equity_Default_Model.py
├── README.md
└── outputs/
    ├── model_comparison.csv
    ├── coefficient_interpretation.csv
    ├── roc_model_comparison.png
    └── ...
```

## Limitations

- The model is evaluated on one historical dataset and has no independent out-of-time validation sample.
- The five variables are intentionally limited for interpretability and do not capture every factor used in real underwriting.
- The 0.50 classification cutoff is illustrative rather than economically optimized. A lending decision threshold should reflect risk appetite, approval strategy, expected loss, and the relative costs of false approvals and false declines.
- Before production use, the model would require additional calibration testing, stability analysis, fairness review, documentation, monitoring, and independent validation.

## Conclusion

This project shows how a transparent logistic-regression framework can convert a small set of borrower characteristics into an interpretable default-risk score. The expanded five-variable model improves risk ranking over the simpler benchmark, while the standardized coefficients, odds ratios, and risk-decile analysis keep the results understandable from a credit-risk perspective.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not intended for real lending or underwriting decisions.
