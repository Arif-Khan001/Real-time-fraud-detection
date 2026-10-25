# Fraud Detection Project

## Objective
Detect fraudulent financial transactions with machine learning and serve the model as a real-time scoring app, cutting payment fraud losses while keeping false alarms manageable.

## Dataset
[PaySim synthetic mobile-money transactions](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle `ealaxi/paysim1`): **6,362,620 transactions, ~0.13% fraud**.

| Column | Meaning |
|---|---|
| `step` | time step (1 step = 1 hour) |
| `type` | PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT |
| `amount` | transaction amount |
| `nameOrig` / `nameDest` | sender / receiver account |
| `oldbalanceOrg` / `newbalanceOrig` | sender balance before / after |
| `oldbalanceDest` / `newbalanceDest` | receiver balance before / after |
| `isFraud` | target (1 = fraud) |
| `isFlaggedFraud` | output of a simple built-in rule (not used as a feature) |

## Tools & Libraries
- Python 3.11
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn (Logistic Regression, Random Forest, pipelines), XGBoost
- Joblib (model serialisation)
- Streamlit (app deployment)

## Approach
1. **EDA**: class imbalance, fraud by type, amount and time-of-day analysis, data-quality checks, evaluation of the built-in `isFlaggedFraud` rule.
2. **Scope**: fraud occurs only in `TRANSFER` and `CASH_OUT`, so the model is trained on those two types (all fraud retained, about 56% of rows removed).
3. **Feature engineering**: `balanceDiffOrig`, `balanceDiffDest`, `errorBalanceOrig`, `errorBalanceDest`, `origEmptied`, `destUntouched`, `amountToBalance`.
4. **Split**: stratified train / validation / test; the test set is used once, for reporting.
5. **Model**: one Scikit-Learn `Pipeline` of `ColumnTransformer` (StandardScaler + OneHotEncoder) and `LogisticRegression(class_weight="balanced")`.
6. **Threshold**: tuned on validation data (F1), not on test.
7. **Comparison and checks**: majority-class and unweighted baselines, Random Forest and XGBoost, 5-fold cross-validation, an ablation of the engineered features, and error analysis.

## Model Performance
Evaluated on the held-out test set. Accuracy is deliberately not the headline metric (a model that never flags fraud is ~99.9% accurate on this data).

> After running the notebook, paste the table printed in **Section 23** here.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| _fill from notebook_ | | | | | |

The notebook also reports the share of fraud **value** stopped, alerts per true fraud, and a cumulative-gains table for a fixed review budget.

## Visualization
- Class distribution and fraud rate by transaction type
- Amount histograms and boxplots by class
- Fraud by hour of day and by day
- Engineered-feature distributions by class
- Correlation heatmap
- Confusion matrices, ROC and Precision-Recall curves, threshold curve, gains curve
- Logistic Regression coefficients and XGBoost feature importance

## Streamlit App
- Enter transaction type, amount and sender / receiver balances (or load an example)
- Engineered features are computed automatically
- Shows `st.error` for a fraud alert or `st.success` for a legitimate transaction, the fraud score, and the risk signals found
- Adjustable decision threshold in the sidebar

![App screenshot](screenshots/app.png)

## How to Run
```bash
pip install -r requirements.txt
# 1. download PaySim from Kaggle and put the CSV in this folder
# 2. run every cell of fraud_detection.ipynb  (writes the .pkl and .json)
streamlit run streamlit_app.py
```

## Files in Repository
- `fraud_detection.ipynb`: EDA, feature engineering, model training, evaluation, and model export
- `streamlit_app.py`: Streamlit interface for predictions
- `fraud_detection_pipeline.pkl`: trained pipeline (created by the notebook)
- `model_metadata.json`: input columns, tuned threshold and test metrics (created by the notebook)
- `requirements.txt`
- `screenshots/`: app screenshots

## Key Insights
- Only ~0.13% of transactions are fraudulent: a highly imbalanced problem where accuracy is misleading.
- Fraud appears only in `TRANSFER` and `CASH_OUT` transactions.
- Fraudulent transactions tend to empty the sender's account and to be larger than typical transactions.
- The built-in `isFlaggedFraud` rule catches almost none of the fraud, so a learned model adds real value.
- Class weighting and threshold tuning matter more than the choice of algorithm on imbalanced data.

## Limitations
PaySim is simulated data with a simple fraud pattern; real fraud is adversarial and drifts over time. The split is random rather than temporal, and the app scores one manually entered transaction rather than a live stream.
