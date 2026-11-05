import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lapse_features import build_cohort, split, make_prep

df = build_cohort()
X_train, X_test, y_train, y_test = split(df)

def evaluate(name, model):
    pred = model.predict(X_test)
    m = {"precision": precision_score(y_test, pred, zero_division=0),
         "recall": recall_score(y_test, pred, zero_division=0),
         "f1": f1_score(y_test, pred, zero_division=0)}
    print(f"\n{name}  (confusion matrix: rows = actual 0/1, columns = predicted 0/1)")
    print(confusion_matrix(y_test, pred))
    print({k: round(v, 3) for k, v in m.items()})
    return m

# ---------- Task 6b: logistic regression ----------
lr_search = GridSearchCV(
    Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=1000))]),
    {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_search.fit(X_train, y_train)
print("LR best params:", lr_search.best_params_, "| best CV F1:", round(lr_search.best_score_, 3))

# ---------- Task 6c: random forest (45 fits x 5 folds) ----------
rf_search = GridSearchCV(
    Pipeline([("prep", make_prep()),
              ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1")
rf_search.fit(X_train, y_train)
print("RF best params:", rf_search.best_params_, "| best CV F1:", round(rf_search.best_score_, 3))

# ---------- Task 6d: first and only look at the test set ----------
lr_metrics = evaluate("Logistic regression (tuned)", lr_search.best_estimator_)
rf_metrics = evaluate("Random forest (tuned)", rf_search.best_estimator_)

# ---------- Task 7: oversample the TRAINING rows only ----------
n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
extra_idx = y_train[y_train == 1].sample(n=n_extra, replace=True, random_state=42).index
all_idx = list(y_train.index) + list(extra_idx)
X_bal, y_bal = X_train.loc[all_idx], y_train.loc[all_idx]
print("\nbalanced training counts:", y_bal.value_counts().to_dict(), "(ref 570 each)")

lr_bal = clone(lr_search.best_estimator_).fit(X_bal, y_bal)
bal_metrics = evaluate("Logistic regression (balanced training data)", lr_bal)

print("\nChange after balancing:",
      {k: round(bal_metrics[k] - lr_metrics[k], 3) for k in lr_metrics})