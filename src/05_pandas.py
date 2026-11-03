import pandas as pd

cust = pd.read_csv("data/clean/customers.csv")
prod = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

done = orders[orders["status"] == "completed"]
df = done.merge(cust[["customer_id", "city"]], on="customer_id") \
         .merge(prod[["product_id", "category"]], on="product_id")

# ---------- (a) pivot_table cross-check of Q2 ----------
pivot = pd.pivot_table(df, values="order_amount", index="city",
                       columns="category", aggfunc="sum", margins=True)
grand = round(pivot.loc["All", "All"], 2)
print(pivot.round(2))
print("pivot grand total:", grand, "| SQL total: 2998668.80 | match:", abs(grand - 2998668.80) < 0.01)

# groupby with named aggregation, per city
by_city = df.groupby("city")["order_amount"].agg(total="sum", mean="mean", count="count").round(2)
print(by_city)

# ---------- (b) distribution of completed order_amount ----------
x = df["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("mean", round(x.mean(), 2), "| median", x.median(), "| skew", round(x.skew(), 2))
print("Q1", q1, "| Q3", q3, "| IQR", iqr)
print("fences:", round(low, 2), round(high, 2))
print("outliers below:", (x < low).sum(), "| above:", (x > high).sum())

# correlation matrix over completed orders
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print(corr.round(3))
best = None
for a in cols:
    for b in cols:
        if a < b:                       # visit each pair once, skip a column paired with itself
            r = corr.loc[a, b]
            if best is None or abs(r) > abs(best[2]):
                best = (a, b, r)
print("largest |correlation|:", best[0], "vs", best[1], round(best[2], 3))