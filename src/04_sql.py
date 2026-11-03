import re
import sqlite3
import pandas as pd

sql = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)\b.*$", sql, flags=re.M)   # [header, 'Q1', body, 'Q2', body, ...]

con = sqlite3.connect("data/instamart.db")
res = {name: pd.read_sql_query(body, con) for name, body in zip(parts[1::2], parts[2::2])}
con.close()

print("Q1 customers:", len(res["Q1"]), "(ref 30)")
q2 = res["Q2"]
print("Q2 rows:", len(q2), "(ref 30) | total:", round(q2["revenue"].sum(), 2), "(ref 2998668.80)")
print("Q2 largest cell:", q2.iloc[0].tolist(), "(ref Bengaluru / Electronics 233909.45)")
print("Q3 products:", len(res["Q3"]), "(ref 11)")
print("Q4 customers:", len(res["Q4"]), "(ref 25)")
print("Q5 customers:", len(res["Q5"]), "| average:", res["Q5"]["avg_total"].iloc[0], "(ref 550, 2552.06)")
q6 = res["Q6"]
print("Q6 top per city:\n", q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", (q6["rnk"] != q6["row_num"]).sum(), "(ref 8)")
q7 = res["Q7"]
print("Q7 days:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1], "(ref 183, 2998668.80)")