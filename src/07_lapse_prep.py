from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from lapse_features import build_cohort, split, NUMERIC, NOMINAL

df = build_cohort()
print("cohort:", len(df), "(ref 1156)")
print("lapsed:", df["lapsed"].sum(), "=", round(df["lapsed"].mean() * 100, 2), "% (ref 443 = 38.32%)")

X_train, X_test, y_train, y_test = split(df)
print("train/test rows:", len(X_train), len(X_test), "(ref 924 / 232)")

# baseline: always predict "not lapsed"
base = [0] * len(y_test)
print("baseline accuracy:", round(accuracy_score(y_test, base) * 100, 1), "% (ref 61.6)")
print("baseline lapsed-class F1:", f1_score(y_test, base, zero_division=0), "(ref 0)")

# no missing values in any feature column
assert df[NUMERIC + NOMINAL].isna().sum().sum() == 0
print("missing values in features: 0")

# preprocessing, fitted on TRAIN rows only, then applied to both
prep = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                          ("cat", OneHotEncoder(drop="first"), NOMINAL)])
prep.fit(X_train)
Xtr, Xte = prep.transform(X_train), prep.transform(X_test)
print("encoded columns:", Xtr.shape[1])

learned = prep.named_transformers_["num"].mean_[NUMERIC.index("frequency")]
print("frequency mean learned from train:", round(learned, 3),
      "| full cohort:", round(df["frequency"].mean(), 3))