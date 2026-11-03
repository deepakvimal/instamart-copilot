import sqlite3
import pandas as pd

cust = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

# customers
assert len(cust) == 1200, "customer count"
assert cust["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197,
    "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
assert cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
assert cust["age"].isna().sum() == 0, "missing ages"
assert cust["age"].between(18, 60).all(), "age range"

# orders
assert len(orders) == 12309, "order rows"
assert orders["order_id"].is_unique, "one row per order_id"
assert len(completed) == 11110 and len(cancelled) == 1199, "status counts"
assert (orders["status"] == "pending").sum() == 0, "no pending rows"
assert orders["order_amount"].isna().sum() == 0, "missing amounts"
assert completed["delivery_minutes"].isna().sum() == 0, "completed delivery missing"
assert cancelled["delivery_minutes"].isna().sum() == 1199, "cancelled delivery missing"
assert abs(completed["order_amount"].sum() - 2998668.80) < 0.01, "completed revenue"

# database
con = sqlite3.connect("data/instamart.db")
for table, n in [("customers", 1200), ("products", 40), ("orders", 12309)]:
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    assert count == n, f"{table} rows in database"
con.close()

print("ALL CHECKPOINTS PASSED")