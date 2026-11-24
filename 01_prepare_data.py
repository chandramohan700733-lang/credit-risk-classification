"""
Step 1: Load, check and prepare the South German Credit data.

Source : UCI Machine Learning Repository, "South German Credit (UPDATE)", CC BY 4.0
Input  : data/raw/SouthGermanCredit.asc  (space separated, German column names)
Output : data/clean/credit_clean.csv     (English column names, same numeric codes)
"""
import os
import pandas as pd

RAW = "data/raw/SouthGermanCredit.asc"
os.makedirs("data/clean", exist_ok=True)

df = pd.read_csv(RAW, sep=r"\s+")

# 1) English column names (from the code table that comes with the data)
names = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration",
    "rate": "installment_rate", "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing", "bishkred": "number_credits",
    "beruf": "job", "pers": "people_liable", "telef": "telephone", "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}
df = df.rename(columns=names)

# 2) Checks
print("Shape:", df.shape)
assert df.shape == (1000, 21), "unexpected shape"
assert df.isna().sum().sum() == 0, "missing values found"
assert not df.duplicated().any(), "duplicate rows found"

allowed = {
    "status": range(1, 5), "credit_history": range(0, 5), "purpose": range(0, 11),
    "savings": range(1, 6), "employment_duration": range(1, 6), "installment_rate": range(1, 5),
    "personal_status_sex": range(1, 5), "other_debtors": range(1, 4), "present_residence": range(1, 5),
    "property": range(1, 5), "other_installment_plans": range(1, 4), "housing": range(1, 4),
    "number_credits": range(1, 5), "job": range(1, 5), "people_liable": range(1, 3),
    "telephone": range(1, 3), "foreign_worker": range(1, 3), "credit_risk": range(0, 2),
}
for col, ok in allowed.items():
    bad_values = set(df[col].unique()) - set(ok)
    assert not bad_values, f"{col}: unexpected codes {bad_values}"
assert df.duration.between(1, 120).all() and (df.amount > 0).all() and (df.age >= 18).all()
print("All checks passed: no missing values, no duplicates, all codes valid.")

# 3) Target. In credit risk work the event we want to find is the BAD credit.
df["bad_credit"] = 1 - df["credit_risk"]
print("\nTarget balance (bad_credit):", df.bad_credit.value_counts().to_dict(),
      "-> {:.0%} bad".format(df.bad_credit.mean()))
print("Note: UCI says bad credits were heavily oversampled, so this is NOT a real default rate.")
print("Purpose codes present:", sorted(df.purpose.unique()), "(code 7 never occurs in the data)")

df.to_csv("data/clean/credit_clean.csv", index=False)
print("\nSaved: data/clean/credit_clean.csv")
