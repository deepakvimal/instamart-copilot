import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount",
           "cancel_rate", "avg_delivery", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]


def build_cohort():
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("data/clean/products.csv")

    # FEATURES: only orders strictly before the cutoff
    before = orders[orders["order_ts"] < CUTOFF]
    done = before[before["status"] == "completed"]

    feats = (done.groupby("customer_id")
                 .agg(last_order=("order_ts", "max"),
                      frequency=("order_id", "nunique"),
                      monetary=("order_amount", "sum"),
                      avg_discount=("discount_pct", "mean"),
                      avg_delivery=("delivery_minutes", "mean"))
                 .reset_index())
    feats["recency"] = (CUTOFF - feats["last_order"]).dt.days

    # cancellation rate = cancelled / all orders with a final status, before the cutoff
    final = before[before["status"].isin(["completed", "cancelled"])]
    cancel = ((final["status"] == "cancelled")
              .groupby(final["customer_id"]).mean().rename("cancel_rate").reset_index())
    feats = feats.merge(cancel, on="customer_id", how="left")

    # most frequent product category (ties broken alphabetically so results are reproducible)
    cat = done.merge(products[["product_id", "category"]], on="product_id")
    counts = cat.groupby(["customer_id", "category"]).size().reset_index(name="n")
    fav = (counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
                 .drop_duplicates("customer_id")[["customer_id", "category"]]
                 .rename(columns={"category": "fav_category"}))
    feats = feats.merge(fav, on="customer_id")

    # customer attributes
    feats = feats.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                        on="customer_id")
    feats["tenure_days"] = (CUTOFF - feats["signup_date"]).dt.days

    # LABEL: the only place September is used
    sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                  & (orders["status"] == "completed")]
    feats["lapsed"] = (~feats["customer_id"].isin(sept["customer_id"])).astype(int)

    return feats[["customer_id"] + NUMERIC + NOMINAL + ["lapsed"]].reset_index(drop=True)


def split(df):
    X, y = df[NUMERIC + NOMINAL], df["lapsed"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="first"), NOMINAL)])