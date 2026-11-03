import pandas as pd
import os, sqlite3

# Load as text so we control every conversion ourselves.
customers = pd.read_csv("data/raw/customers.csv", dtype=str, keep_default_na=False)
raw_city = customers["city"].copy()

# (a) City: strip spaces, then Title Case -> exactly 6 cities remain
customers["city"] = customers["city"].str.strip().str.title()

# (b) Gender: normalise to male / female / unknown
gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
customers["gender"] = (customers["gender"].str.strip().str.lower()
                       .map(gender_map).fillna("unknown"))

# (c) Age: impossible values -> missing, then fill with the median age of the customer's city
age = pd.to_numeric(customers["age"], errors="coerce")   # blanks -> NaN
age = age.where((age >= 10) & (age <= 100))              # outside 10-100 -> NaN
n_missing_before_fill = age.isna().sum()
customers["age"] = age
city_median = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median)

# (d) signup_date -> datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------- checks (these numbers also go into your README) ----------
print("city values changed:", (raw_city != customers["city"]).sum())
print("ages missing before fill (61 blank + 6 impossible):", n_missing_before_fill)
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages:", customers["age"].isna().sum())
print("age min/max:", customers["age"].min(), customers["age"].max())

# ======================= ORDERS =======================
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")
n0 = len(orders)

# Rule 1: order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: remove exact duplicate rows
orders = orders.drop_duplicates()
n_exact = n0 - len(orders)

# Rule 3: one record per order_id -> keep the one with the latest order_ts
n1 = len(orders)
orders = (orders.sort_values("order_ts")
                .drop_duplicates(subset="order_id", keep="last")
                .sort_values("order_id").reset_index(drop=True))
n_partial = n1 - len(orders)

# Rule 4: order_amount -> numeric; fill missing from product price
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
miss = orders["order_amount"].isna()
n_amount_filled = miss.sum()
orders.loc[miss, "order_amount"] = [
    round(p * q * (1 - d / 100), 2)
    for p, q, d in zip(orders.loc[miss, "price_inr"],
                       orders.loc[miss, "quantity"],
                       orders.loc[miss, "discount_pct"])]

# Rule 5: negative delivery -> missing; fill COMPLETED orders with the category median
neg = orders["delivery_minutes"] < 0
n_negative = neg.sum()
orders.loc[neg, "delivery_minutes"] = float("nan")

completed = orders["status"] == "completed"
n_deliv_missing = (orders["delivery_minutes"].isna() & completed).sum()
cat_median = (orders["delivery_minutes"].where(completed)       # medians use completed only
              .groupby(orders["category"]).transform("median"))
orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(cat_median[completed]))

# ---------- checks (numbers for your README) ----------
print("\nrows removed, exact duplicates:", n_exact)
print("rows removed, older pending records:", n_partial)
print("amounts filled from price:", n_amount_filled)
print("negative delivery times set to missing:", n_negative)
print("completed delivery times filled with category median:", n_deliv_missing)
print("rows:", len(orders))
print(orders["status"].value_counts())
print("missing order_amount:", orders["order_amount"].isna().sum())
print("missing delivery, completed:", (orders["delivery_minutes"].isna() & completed).sum())
print("missing delivery, cancelled:", (orders["delivery_minutes"].isna() & (orders["status"] == "cancelled")).sum())
print("completed revenue:", round(orders.loc[completed, "order_amount"].sum(), 2))

# ======================= SAVE =======================

os.makedirs("data/clean", exist_ok=True)

order_cols = ["order_id", "customer_id", "product_id", "quantity", "order_ts",
              "status", "discount_pct", "order_amount", "delivery_minutes"]
orders_out = orders[order_cols].copy()                  # drop the helper columns (category, price_inr)
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")  # text, as the brief requires

customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")

customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

con = sqlite3.connect("data/instamart.db")
try:
    customers_out.to_sql("customers", con, if_exists="replace", index=False)
    products.to_sql("products", con, if_exists="replace", index=False)
    orders_out.to_sql("orders", con, if_exists="replace", index=False)
finally:
    con.close()
print("saved clean CSVs and data/instamart.db")