import pandas as pd

# ---------- Part 1: general audit (shape, types, missing values) ----------
# Normal load: pandas turns blanks into NaN, which is what we want for counting missing values.
tables = {
    "customers": pd.read_csv("data/raw/customers.csv"),
    "products": pd.read_csv("data/raw/products.csv"),
    "orders": pd.read_csv("data/raw/orders.csv"),
}

for name, df in tables.items():
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    print("dtypes:\n", df.dtypes)
    summary = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(summary)

# ---------- Part 2: the specific audit figures ----------
# Load as plain text so pandas can't silently "fix" anything.
orders = pd.read_csv("data/raw/orders.csv", dtype=str, keep_default_na=False)
customers = pd.read_csv("data/raw/customers.csv", dtype=str, keep_default_na=False)

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
ages = pd.to_numeric(customers["age"], errors="coerce")
gender = customers["gender"]

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": orders.duplicated().sum(),
    "pending rows": (orders["status"] == "pending").sum(),
    "distinct order_id": orders["order_id"].nunique(),
    "non-numeric amounts": amount.isna().sum(),
    "negative delivery_minutes": (delivery < 0).sum(),
    "blank delivery_minutes": (orders["delivery_minutes"] == "").sum(),
    "blank ages": (customers["age"] == "").sum(),
    "ages outside 10-100": ((ages < 10) | (ages > 100)).sum(),
    "distinct raw city spellings": customers["city"].nunique(),
    "distinct non-blank gender spellings": gender[gender != ""].nunique(),
    "blank genders": (gender == "").sum(),
}
print("\n===== audit figures =====")
for k, v in audit.items():
    print(f"{k}: {v}")