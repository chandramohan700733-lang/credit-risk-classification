# Credit Risk Classification (South German Credit data)

A small machine learning project that predicts whether a credit applicant is a **bad credit risk**,
using a public dataset of 1,000 credits, scikit-learn and a careful evaluation.

**Tools:** Python, pandas, scikit-learn, matplotlib

This is a practice project on a public consumer credit dataset. It is not a lending model for any
company, and it has nothing to do with agricultural lending.

## Data
- Source: UCI Machine Learning Repository, "South German Credit (UPDATE)", CC BY 4.0
  (https://archive.ics.uci.edu/dataset/573/south%2Bgerman%2Bcredit)
- 1,000 credits, 20 predictors (status of the checking account, duration, credit history, purpose,
  amount, savings, age and so on) and the outcome (good or bad credit)
- No missing values and no duplicates (checked in `01_prepare_data.py`)
- 30% of the credits are "bad". UCI states that bad credits were heavily oversampled, so **this is
  not a real default rate**. For this reason I use ROC-AUC as the main metric, because it does not
  depend on the share of bad credits.
- The data is from the 1970s, so it says little about credit today.
- I chose this corrected version because UCI notes that the older, widely used "German Credit" file
  has coding errors.

## What I did
1. **Prepared the data** (`01_prepare_data.py`): English column names, validity checks for every code,
   and the target `bad_credit` (1 = bad), because the event we want to find is the bad credit.
2. **Split first:** 80% train and 20% test (stratified). The test set is used only once, at the end.
3. **Compared models with 5-fold cross-validation on the training data only** (`02_train_evaluate.py`):
   a baseline that predicts the base rate, logistic regression, random forest and gradient boosting.
   Categorical codes were one-hot encoded and the loan amount was log-transformed.
4. **Chose the best model and the decision threshold using training data only.** The threshold comes
   from out-of-fold predictions and an assumed cost: approving a bad credit costs 5 times more than
   refusing a good one. The 5:1 ratio is an assumption for this exercise, not a bank's figure, and it is
   a single setting at the top of the script.
5. **Evaluated once on the test set** and checked which features matter (permutation importance).
6. **Checked the sensitive columns** (`personal_status_sex`, `age`, `foreign_worker`).

## Results
These numbers come from my run with `random_state=42`. Your numbers can differ slightly with other
library versions.

| Model (5-fold CV on the training set) | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline (base rate) | 0.500 | 0.300 |
| Logistic regression | 0.778 | 0.616 |
| Random forest | **0.796** | 0.630 |
| Gradient boosting | 0.791 | 0.604 |

- **Test set (200 credits), random forest:** ROC-AUC **0.758** (95% bootstrap interval about 0.68 to 0.83),
  PR-AUC 0.612 (a model with no skill gets about 0.30). The interval is wide because the test set is small.
- **At the default threshold of 0.50:** it finds 53% of the bad credits (32 of 60) with 53% precision.
- **At the threshold chosen with the 5:1 cost (0.33):** it finds 85% of the bad credits (51 of 60), but only
  43% of the credits it flags are truly bad (68 good credits are refused). Under the assumed costs this
  lowers the total cost from 168 to 113 on the test set. A different cost ratio would give a different threshold.
- **Most important feature:** the status of the checking account (`status`) by far. Shuffling it lowers the test
  AUC by about 0.12, far more than any other feature. Duration, credit history, age, savings and amount follow
  with much smaller effects.
- **Sensitive columns:** removing `personal_status_sex`, `age` and `foreign_worker` changed the cross-validated
  AUC only from 0.796 to 0.790. Removing a column does not make a model fair, because other columns can act as
  proxies, so this is only a first check.

Figures: `results/roc_pr_curves.png`, `results/confusion_matrix.png`, `results/permutation_importance.png`.

## Limitations
- Small dataset (1,000 rows) from the 1970s, with oversampled bad credits. The results show how to build and
  evaluate a model carefully, not how a real lender's portfolio behaves.
- The test set has only 200 rows, so the test AUC is uncertain (see the interval).
- The cost ratio is an assumption.
- I did not do heavy hyperparameter tuning, to avoid overfitting a small dataset.
- A real credit model would also need fairness, legal and monitoring checks.

## How to run
```
pip install -r requirements.txt
python 01_prepare_data.py
python 02_train_evaluate.py
```
