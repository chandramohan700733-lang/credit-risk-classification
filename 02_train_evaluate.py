"""
Step 2: Train and compare credit risk models, then evaluate the best one once on a held-out test set.

Rules followed to avoid leakage and overfitting:
  * the train/test split happens first and the test set is used only once, at the end
  * model choice and the decision threshold use training data only (cross-validation)
  * the main metric is ROC-AUC (it does not depend on the oversampled bad rate)
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("results", exist_ok=True)

# Illustrative cost assumption: approving a bad credit costs 5 times more than refusing a good one.
# This is an assumption for the exercise, not a bank's real number. Change it to test other ratios.
COST_FN, COST_FP = 5, 1

df = pd.read_csv("data/clean/credit_clean.csv")
y = df["bad_credit"]
X = df.drop(columns=["bad_credit", "credit_risk"])

NUMERIC = ["duration", "amount", "age"]
CATEGORICAL = [c for c in X.columns if c not in NUMERIC]

def make_preprocessor(numeric, categorical):
    log_amount = FunctionTransformer(np.log1p)
    return ColumnTransformer([
        ("amount", Pipeline([("log", log_amount), ("scale", StandardScaler())]), [c for c in numeric if c == "amount"]),
        ("num", StandardScaler(), [c for c in numeric if c != "amount"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

def models(numeric, categorical):
    prep = lambda: make_preprocessor(numeric, categorical)
    return {
        "Baseline (always predicts the base rate)": Pipeline([("m", DummyClassifier(strategy="prior"))]),
        "Logistic regression": Pipeline([("p", prep()), ("m", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))]),
        "Random forest": Pipeline([("p", prep()), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("p", prep()), ("m", GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED))]),
    }

# 1) Split first (stratified), keep the test set untouched
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print("Train:", X_train.shape, "bad rate", round(y_train.mean(), 3), "| Test:", X_test.shape, "bad rate", round(y_test.mean(), 3))

# 2) Compare models with 5-fold cross-validation on the training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in models(NUMERIC, CATEGORICAL).items():
    s = cross_validate(model, X_train, y_train, cv=cv, scoring={"auc": "roc_auc", "pr": "average_precision"})
    rows.append({"model": name, "cv_roc_auc_mean": s["test_auc"].mean(), "cv_roc_auc_std": s["test_auc"].std(),
                 "cv_pr_auc_mean": s["test_pr"].mean()})
cv_table = pd.DataFrame(rows).round(3)
print("\nCross-validation on the training set (positive class = bad credit):\n", cv_table.to_string(index=False))
cv_table.to_csv("results/cv_comparison.csv", index=False)

best_name = cv_table[cv_table.model != cv_table.model.iloc[0]].sort_values("cv_roc_auc_mean", ascending=False).iloc[0].model
print("\nBest model by CV ROC-AUC:", best_name)

# 3) Choose the decision threshold on training data only (out-of-fold predictions)
best = models(NUMERIC, CATEGORICAL)[best_name]
oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
grid = np.arange(0.05, 0.96, 0.01)
costs = []
for t in grid:
    tn, fp, fn, tp = confusion_matrix(y_train, oof >= t).ravel()
    costs.append(COST_FN * fn + COST_FP * fp)
threshold = float(grid[int(np.argmin(costs))])
print(f"Threshold chosen on training data with cost ratio {COST_FN}:{COST_FP} -> {threshold:.2f}")

# 4) Final fit and ONE evaluation on the test set
best.fit(X_train, y_train)
proba = best.predict_proba(X_test)[:, 1]
auc, pr = roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
dummy_pr = float(y_test.mean())
# The test set is small (200 rows), so show how uncertain the AUC is (bootstrap 95% interval)
rng = np.random.default_rng(SEED)
boot = []
yt = y_test.to_numpy()
for _ in range(1000):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].min() != yt[idx].max():
        boot.append(roc_auc_score(yt[idx], proba[idx]))
auc_lo, auc_hi = np.percentile(boot, [2.5, 97.5])

def report(th):
    tn, fp, fn, tp = confusion_matrix(y_test, proba >= th).ravel()
    return {"threshold": round(th, 2), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "recall_bad": round(float(tp / (tp + fn)), 3), "precision_bad": round(float(tp / (tp + fp)), 3) if tp + fp else None,
            "cost": int(COST_FN * fn + COST_FP * fp)}
r05, rcost = report(0.5), report(threshold)
print(f"\nTEST  ROC-AUC {auc:.3f} (95% bootstrap interval {auc_lo:.2f} to {auc_hi:.2f}) | PR-AUC {pr:.3f} (a model with no skill would score about {dummy_pr:.2f})")
print("At threshold 0.50:", r05)
print(f"At chosen threshold {threshold:.2f}:", rcost)

# 5) Which features matter? (permutation importance on the test set, original columns)
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED)
imp_df = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std}).sort_values("importance", ascending=False)
print("\nTop 8 features by permutation importance (drop in test AUC):\n", imp_df.head(8).round(3).to_string(index=False))
imp_df.round(4).to_csv("results/permutation_importance.csv", index=False)

# 6) Sensitive attributes check: does the model need them?
SENSITIVE = ["personal_status_sex", "age", "foreign_worker"]
num2 = [c for c in NUMERIC if c not in SENSITIVE]
cat2 = [c for c in CATEGORICAL if c not in SENSITIVE]
m2 = models(num2, cat2)[best_name]
s_with = cross_validate(models(NUMERIC, CATEGORICAL)[best_name], X_train, y_train, cv=cv, scoring="roc_auc")["test_score"]
s_without = cross_validate(m2, X_train.drop(columns=SENSITIVE), y_train, cv=cv, scoring="roc_auc")["test_score"]
print(f"\nCV ROC-AUC with sensitive columns {s_with.mean():.3f}, without {s_without.mean():.3f}")

# 7) Figures
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fpr, tpr, _ = roc_curve(y_test, proba)
ax[0].plot(fpr, tpr, label=f"{best_name} (AUC {auc:.2f})"); ax[0].plot([0, 1], [0, 1], "--", color="grey", label="No skill")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate (bad credits found)", title="ROC curve, test set"); ax[0].legend(loc="lower right")
p, r, _ = precision_recall_curve(y_test, proba)
ax[1].plot(r, p, label=f"PR-AUC {pr:.2f}"); ax[1].axhline(dummy_pr, ls="--", color="grey", label=f"No skill ({dummy_pr:.2f})")
ax[1].set(xlabel="Recall (bad credits found)", ylabel="Precision", title="Precision-recall curve, test set"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/roc_pr_curves.png", dpi=150); plt.close()

top = imp_df.head(10).iloc[::-1]
plt.figure(figsize=(7, 4)); plt.barh(top.feature, top.importance, xerr=top["std"], color="#1F5FBF")
plt.xlabel("Drop in test ROC-AUC when the feature is shuffled"); plt.title("Permutation importance (top 10)")
plt.tight_layout(); plt.savefig("results/permutation_importance.png", dpi=150); plt.close()

cm = confusion_matrix(y_test, proba >= threshold)
plt.figure(figsize=(4.2, 3.8)); plt.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): plt.text(j, i, v, ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Predicted good", "Predicted bad"]); plt.yticks([0, 1], ["Actually good", "Actually bad"])
plt.title(f"Confusion matrix, threshold {threshold:.2f}"); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

json.dump({"best_model": best_name, "test_roc_auc": round(auc, 3), "test_roc_auc_95ci": [round(float(auc_lo), 3), round(float(auc_hi), 3)], "test_pr_auc": round(pr, 3), "no_skill_pr_auc": round(dummy_pr, 3),
           "cost_ratio_FN_to_FP": f"{COST_FN}:{COST_FP}", "threshold_050": r05, "threshold_cost_based": rcost,
           "cv_auc_with_sensitive": round(s_with.mean(), 3), "cv_auc_without_sensitive": round(s_without.mean(), 3)},
          open("results/metrics.json", "w"), indent=2)
print("\nSaved: results/metrics.json, cv_comparison.csv, permutation_importance.csv and 3 figures")
