## Instamart Customer Intelligence & Support Copilot
One internal tool in four connected parts: clean the order data (A), segment customers and predict lapse (B), triage support tickets safely (C), answer policy questions and decide refunds with an agent (D). All data is synthetic, all money is in INR, and every model call uses the offline mock_llm.py (no API key, no network).
Before submitting: write each TODO in your own words, then delete this note.

## Overview
Part	        What it does	                Main outputs
  A	    Audit, clean, store, query	      data/clean/*.csv, data/instamart.db, sql/queries.sql
  B	    K-means segments, lapse model	    outputs/customer_segments.csv, lapse_model.joblib, retention_list.csv
  C	    Redact, classify, route tickets	  outputs/triage_results.csv, triage_budget_demo.csv
  D	    Policy RAG and refund agent	      outputs/agent_runs.jsonl, refund_ledger.jsonl, state/

Code is in src/ (one numbered script per step, plus lapse_features.py, rag.py, agent.py).
  
## Reproduce
Run from the repo root. Python 3.11 or newer (built on 3.14).

1.	git clone https://github.com/deepakvimal/instamart-copilot.git
2.	cd instamart-copilot
3.	python -m venv .venv
4.	Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
5.	python -m pip install -r requirements.txt
   Your project has a requirements.txt in the file list, so this installs everything the project needs, including pandas. If that fails for any reason, install pandas numpy.
6.  python -m pip install pandas numpy
7.	python generate_data.py (last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164)
8.	python src/01_audit.py
9.	python src/02_clean.py
10.	python src/03_validate.py (prints ALL CHECKPOINTS PASSED)
11.	python src/04_sql.py
12.	python src/05_pandas.py
13.	python src/06_segments.py
14.	python src/07_lapse_prep.py
15.	python src/08_lapse_models.py
16.	python src/09_lapse_outputs.py
17.	python src/10_triage.py
18.	python src/11_rag_eval.py
19.	python src/12_agent_runs.py

Seeds are fixed (42 everywhere), so results repeat.

## Part A
the Reproduce steps, the audit table, the cleaning rules with their counts, the SQL results, and the pandas findings. 
I’ll help you assemble these once all the parts are done, so you only write the README once.

Setup: branch main, virtual environment, requirements.txt from pip freeze, .gitignore with .env, .venv/, __pycache__/.
Audit (raw data): customers 1,200 x 5, products 40 x 4, orders 13,084 x 9.

Figure	Value	Figure	Value
Order rows	13,084	Blank ages	61
Exact duplicates	119	Ages outside 10 to 100	6
Pending rows	656	Raw city spellings	24
Distinct order_id	12,309	Raw gender spellings	10
Non-numeric amounts	120	Blank genders	30
Negative delivery_minutes	106	Blank delivery_minutes	2,541

Missing values in a normal load: age 61 (5.08%), gender 30 (2.50%), order_amount 53 (0.41%, only the literal N/A), delivery_minutes 2,541 (19.42%). Products has none.

## Cleaning rules and what each changed

Table	Rule	Count
customers	strip and Title Case city (24 spellings to 6)	476 values
customers	gender to male / female / unknown	30 unknown
customers	impossible ages set missing, all filled with city median	67 filled
orders	remove exact duplicates	119 rows
orders	keep latest order_ts per order_id (drops older pending)	656 rows
orders	refill missing amounts from price x quantity x (1 - discount)	112 values
orders	negative delivery set to missing	105 values
orders	completed orders: fill delivery with category median	774 values
orders	cancelled orders: delivery left missing (never delivered, so filling would invent data)	1,199 left

Result: 12,309 orders (11,110 completed, 1,199 cancelled, no pending), completed revenue INR 2,998,668.80. Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. Ages all 18 to 60. Clean tables are saved as CSV and in data/instamart.db; 03_validate.py checks 16 conditions.

SQL (sql/queries.sql)
Query	Result
Q1 WHERE + HAVING	30 customers
Q2 three-table join	30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45
Q3 scalar subquery	11 products
Q4 LEFT JOIN	25 customers with no completed order
Q5 CTEs	550 customers above the average INR 2,552.06
Q6 window ranks	8 rows where RANK differs from ROW_NUMBER. Top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10
Q7 running total	183 days, last cumulative INR 2,998,668.80

TODO (one sentence): why ties make RANK() differ from ROW_NUMBER().

pandas cross-check: the pivot table (margins=True) grand total is INR 2,998,668.80, equal to the SQL total.

City	Sum (INR)	Mean (INR)	Count
Bengaluru	523,643.40	276.47	1,894
Chennai	463,629.15	272.72	1,700
Delhi	502,829.80	243.62	2,064
Hyderabad	528,589.90	284.95	1,855
Mumbai	479,058.50	297.00	1,613
Pune	500,918.05	252.48	1,984

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 / 645.30, outliers 0 below and 897 above. Strongest correlation: order_amount with discount_pct, -0.095 (order_amount with quantity +0.087; all others near 0).

TODO (at most 200 words): answer the four questions (what is the data about, what do distributions look like, outliers or anomalies, relationships). Say that a correlation this small does not establish causation.

Part B
RFM (snapshot 2025-10-01, completed orders): 1,175 customers; recency median 19 (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

K-means (StandardScaler, n_init=10, random_state=42). Chosen K = 4.

K	2	3	4	5	6	7	8
Inertia	1820.8	1268.2	743.1	617.4	503.2	408.5	366.5
Silhouette	0.465	0.445	0.526	0.480	0.482	0.416	0.412

TODO (2 to 3 sentences): why K = 4 (elbow, silhouette peak, usable by the business). 
TODO (2 to 3 sentences): why K=8 always has lower inertia than K=2, yet that cannot pick K.

Segments (cluster numbers are only labels with no ordering; Premium = highest mean monetary).

Segment (id)	Size	Recency	Frequency	Monetary (INR)	Premium	Action
Occasional Shoppers (0)	436	28.86	3.77	1,057.62	no	TODO
Premium Big-Ticket Buyers (1)	138	24.12	6.77	7,129.70	yes	TODO
Frequent Loyalists (2)	419	8.77	19.40	3,410.02	no	TODO
Dormant At-Risk (3)	182	109.04	2.20	685.99	no	TODO

PCA explained variance: 0.634 and 0.218 (total 0.852).

Lapse problem (cutoff 2025-09-01): cohort 1,156; lapsed 443 (38.32%); train/test 924 / 232; baseline "always not lapsed" gives accuracy 61.6% and lapsed-class F1 0. Features come only from orders before the cutoff plus the customer table. One-hot encoding (drop="first", 19 encoded columns) and scaling sit inside a Pipeline fitted on training rows; no missing values in any feature. Frequency mean: 8.115 (training rows) vs 8.029 (full cohort).

TODO (one sentence): why those two means differ.

Models (lapsed class is positive; confusion matrix rows are actual 0/1, columns predicted 0/1)
Model	Best parameters	CV F1	Test confusion	Precision	Recall	F1
Logistic regression, C in {0.01,0.1,1,10,100}	C = 100	0.651	[[111,32],[31,58]]	0.644	0.652	0.648
Random forest, depth {3,5,8} x leaf {5,10,20}	depth 8, leaf 10	0.681	[[113,30],[31,58]]	0.659	0.652	0.655
Logistic regression, balanced training data	C = 100	n/a	[[90,53],[13,76]]	0.589	0.854	0.697

Balancing: lapsed training rows duplicated (random_state=42) to 570 per class; recall +0.202, precision -0.055, F1 +0.049.

TODO (3 to 5 sentences): your chosen model and why (cost of missing a lapsing customer vs wrongly targeting a loyal one). Make SELECTED in src/09_lapse_outputs.py match. 
TODO (2 to 3 sentences): which metric moved most after balancing and what it means for retention.

Interpretation: top random-forest features: frequency 0.280, monetary 0.271, recency 0.115, avg delivery 0.061, cancellation rate 0.052. Logistic coefficients: recency +0.040, frequency -1.698, monetary +0.285, avg discount +0.004, cancellation rate -0.124, avg delivery +0.079, age +0.023, tenure -0.050.

TODO (one line each): business reading of three coefficient signs.

Saved model and retention list: the model is saved with joblib.dump, and the reloaded copy reproduces the test predictions exactly. The test set has 40 Premium customers; retention_list.csv holds the top 10 by lapse probability, and 1 of them truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes September.

Leakage: no encoder, scaler or resampler saw test rows; no data from 2025-09-01 onwards entered the features; the segment labels were not used as features.

## Part C
Pipeline order: empty/long guardrails, redaction, model call with retries, output validation, token budget, routing.

Check	Result
Empty tickets skipped / long tickets flagged (734 tokens each)	4 / 4
Redactions: emails / phones / order ids	55 / 91 / 132 = 278 (0 left; assertion on every prompt)
Transient errors, all recovered by retry	19 (waits 0.0453 to 0.0562 s)
Statuses: classified_json / classified_keyword / flagged_invalid_output	135 / 12 / 9

147 outputs were rule-validated; 17 tickets (4 empty, 4 long, 9 invalid) went to human review. Prompt sizes: zero-shot 29 tokens, few-shot 116.

Run	Classified	Accuracy	Total tokens	Tokens per ticket
Zero-shot	147	85.7% (126/147)	8,932	60.8
Few-shot	147	100% (147/147)	22,504	153.1

Zero-shot errors: 11 fraud tickets predicted billing, 10 technical tickets predicted billing (all other cells correct: billing 38, delivery 36, technical 29, fraud 23).

TODO (3 to 4 sentences): when the extra few-shot tokens (about 92 per ticket) are worth paying for.

Budget and routing: zero-shot with TOKEN_BUDGET = 6000 used 6,023 tokens and 51 tickets ended budget_exhausted (triage_budget_demo.csv). Final few-shot run (budget 100000), 164 rows in triage_results.csv: human_review 17, security_team (HIGH) 34, premium_desk (HIGH) 11, support_queue 102.

## Part D
RAG: 40-word chunks with 10-word overlap give 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). TF-IDF with cosine similarity; the query "refund policy late delivery" returns three refund_policy.txt chunks.

Setting	Doc in top 3 (of 12)	Fact in answer (of 12)	Fallback (of 2)
Overlap 10	12	12	2
Overlap 0	12	10	2

At overlap 0 two questions break: the refund delivery-time question and the payment-failure question.

TODO (2 sentences): what TF and IDF measure, plus one limitation of TF-IDF. TODO (2 to 3 sentences): why those two questions break at overlap 0 and how overlap fixes it.

Tools (TOOLS_SPEC, TOOL_MAP): get_order, search_policy, calculate_refund (100% if more than 45 minutes, else 0), issue_refund and escalate_to_human (both write to refund_ledger.jsonl).

Scenario	Status	Refund (INR)	Model calls	Tokens
S1	REFUND	128.25	5	1,112
S2	REFUND	135.00	5	1,112
S3	ESCALATE	1,999.00	5	1,099
S4	DENIED	0.00	4	777
S5	DENIED	0.00	3	466
S6	NOT_FOUND	0.00	2	158
S7	FALLBACK_MAX_ITERATIONS	0.00	6	3,258

Stop-safely demos: S1 with token_limit = 200 ends FALLBACK_TOKEN_LIMIT; S1 with time_limit_seconds = 0 ends FALLBACK_TIMEOUT (0 model calls).

Crash and resume (S1): crash after the second tool, then resume_agent. Tools executed in order: get_order, search_policy, calculate_refund, issue_refund (four, none repeated); final REFUND 128.25.

TODO (at most 250 words): ReAct vs plan-and-execute, using S1 5, S4 4, S5 3, S6 2 model calls; cover calls and tokens, surprises like NOT_FOUND or a cancelled order, and your choice.]’

## Limitations and Next Steps
•	Data is synthetic and the model is a rule-based mock, so accuracy figures (such as 100% few-shot) say little about real tickets.
•	The retention list is illustrative and not a leakage-free evaluation: the Premium flag is a post-hoc whole-window label that includes September.
•	Lapse model uses one cutoff and one random split; TF-IDF retrieval is lexical; the refund ledger has no idempotency check (the crash demo refunds S1 twice across runs).

TODO: two or three next steps in your own words (for example time-based split, idempotency key, embedding retrieval).


