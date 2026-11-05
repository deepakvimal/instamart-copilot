import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from lapse_features import build_cohort, split, make_prep

os.makedirs("outputs", exist_ok=True)
SELECTED = "rf"        # "rf" or "lr": set this to YOUR choice after you write your justification

df = build_cohort()
X_train, X_test, y_train, y_test = split(df)

lr_search = GridSearchCV(
    Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))]),
    {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_search = GridSearchCV(
    Pipeline([("prep", make_prep()),
              ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1").fit(X_train, y_train)

# ---------- 8a: interpretation ----------
names = rf_search.best_estimator_.named_steps["prep"].get_feature_names_out()
importance = pd.Series(rf_search.best_estimator_.named_steps["model"].feature_importances_,
                       index=names).sort_values(ascending=False)
print("Top 5 random-forest features:\n", importance.head(5).round(3))

coef = pd.Series(lr_search.best_estimator_.named_steps["model"].coef_[0], index=names)
print("\nLogistic regression coefficients (numeric features):\n",
      coef[[n for n in names if n.startswith("num__")]].round(3))

# ---------- 8b: save, reload, compare ----------
model = rf_search.best_estimator_ if SELECTED == "rf" else lr_search.best_estimator_
joblib.dump(model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(model.predict(X_test), reloaded.predict(X_test))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

# ---------- 8c: retention list for Premium customers in the test set ----------
seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
test = df.loc[X_test.index, ["customer_id", "lapsed"]].copy()
test["lapse_probability"] = reloaded.predict_proba(X_test)[:, 1]
test = test.merge(seg, on="customer_id", how="left")
test["is_premium"] = test["is_premium"].fillna(False).astype(bool)

premium = test[test["is_premium"]]
top = premium.sort_values("lapse_probability", ascending=False).head(10)
out = top.rename(columns={"lapsed": "actually_lapsed"})[
    ["customer_id", "lapse_probability", "actually_lapsed"]].copy()
out["lapse_probability"] = out["lapse_probability"].round(4)
out.to_csv("outputs/retention_list.csv", index=False)

print("\nPremium customers in test set:", len(premium))
print("retention list rows:", len(out), "| truly lapsed:", int(out["actually_lapsed"].sum()))
print(out.to_string(index=False))