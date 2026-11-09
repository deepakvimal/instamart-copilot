## Instamart Customer Intelligence & Support Copilot
One internal tool in four connected parts: clean the order data (A), segment customers and predict lapse (B), triage support tickets safely (C), answer policy questions and decide refunds with an agent (D). All data is synthetic, all money is in INR, and every model call uses the offline mock_llm.py (no API key, no network).

## Overview
Part	        What it does	                Main outputs
  A	    Audit, clean, store, query	      data/clean/*.csv, data/instamart.db, sql/queries.sql
  B	    K-means segments, lapse model	  outputs/customer_segments.csv, lapse_model.joblib,            retention_list.csv
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

When two customers in a city have exactly the same total spend, RANK() gives them the same rank and then skips a number, while ROW_NUMBER() still gives every row a unique number, so the two columns differ from the second tied row onward.

pandas cross-check: the pivot table (margins=True) grand total is INR 2,998,668.80, equal to the SQL total.

City	Sum (INR)	Mean (INR)	Count
Bengaluru	523,643.40	276.47	1,894
Chennai	463,629.15	272.72	1,700
Delhi	502,829.80	243.62	2,064
Hyderabad	528,589.90	284.95	1,855
Mumbai	479,058.50	297.00	1,613
Pune	500,918.05	252.48	1,984

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 / 645.30, outliers 0 below and 897 above. Strongest correlation: order_amount with discount_pct, -0.095 (order_amount with quantity +0.087; all others near 0).

This data is 11,110 completed Instamart orders across six cities, with amount, discount, quantity and delivery time, all in INR. Order amounts are strongly right-skewed: the mean (269.91) is about double the median (135.00), and skewness is 4.61, so most orders are small and a few large ones pull the mean up. There are 897 outliers above the upper fence of 645.30 and none below. They look like genuine high-priced purchases, mostly electronics, so I kept them. The data errors in the raw files (duplicates, negative delivery times, unparseable amounts) were fixed during cleaning. All relationships are weak. The largest correlation is order_amount with discount_pct at -0.095, and order_amount with quantity is +0.087. A correlation this small does not establish causation, so I cannot say that discounts cause lower order amounts.

## Part B
RFM (snapshot 2025-10-01, completed orders): 1,175 customers; recency median 19 (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

K-means (StandardScaler, n_init=10, random_state=42). Chosen K = 4.

K	2	3	4	5	6	7	8
Inertia	1820.8	1268.2	743.1	617.4	503.2	408.5	366.5
Silhouette	0.465	0.445	0.526	0.480	0.482	0.416	0.412

## Why K = 4:
Inertia drops about 41% from K=3 to K=4 but only about 17% from K=4 to K=5, so the elbow is at 4. The silhouette score also peaks at K=4 (0.526). Four segments are few enough for the business to run a separate campaign for each.

## K=8 vs K=2:
Every extra cluster puts a centre closer to some points, so inertia can only fall as K grows, and it would reach zero if each customer were its own cluster. A lower inertia at K=8 therefore shows only that there are more clusters, so K has to be chosen using the elbow, the silhouette score and business usefulness.

## Segment actions (table column):

Occasional Shoppers: send small nudges and starter discounts to build an ordering habit.
Premium Big-Ticket Buyers: give early access to electronics launches and free-delivery perks.
Frequent Loyalists: run a rewards programme and bundle offers to raise basket size.
Dormant At-Risk: run a time-limited win-back coupon.

Frequency means (one sentence):
The scaler learned its mean from the 924 training rows only, while the full-cohort mean also includes the 232 test rows, which have a slightly lower average frequency (about 7.7), so the two numbers differ.

Model choice:
I selected the random forest. It had the higher cross-validated F1 (0.681 vs 0.651) and a slightly higher test F1 (0.655 vs 0.648), and it raised fewer false alarms (30 vs 32). Both models catch the same 58 of 89 lapsing customers. Missing a lapsing customer loses someone who could have been retained, while wrongly targeting a loyal customer only wastes a discount, so recall matters. The retention team’s budget is limited, so precision still counts, and the forest balances the two best.

Balancing:
Recall moved most, from 0.652 to 0.854 (+0.202), while precision fell slightly (0.644 to 0.589). The retention team would miss far fewer lapsing customers (31 down to 13) but would contact more loyal customers by mistake (32 up to 53), so this suits a cheap campaign better than an expensive one.

Segments (cluster numbers are only labels with no ordering; Premium = highest mean monetary).

Segment (id)	Size	Recency	Frequency	Monetary (INR)	Premium	Action
Occasional Shoppers (0)	436	28.86	3.77	1,057.62	no	
Premium Big-Ticket Buyers (1)	138	24.12	6.77	7,129.70	yes	
Frequent Loyalists (2)	419	8.77	19.40	3,410.02	no
Dormant At-Risk (3)	182	109.04	2.20	685.99	no

PCA explained variance: 0.634 and 0.218 (total 0.852).

Lapse problem (cutoff 2025-09-01): cohort 1,156; lapsed 443 (38.32%); train/test 924 / 232; baseline "always not lapsed" gives accuracy 61.6% and lapsed-class F1 0. Features come only from orders before the cutoff plus the customer table. One-hot encoding (drop="first", 19 encoded columns) and scaling sit inside a Pipeline fitted on training rows; no missing values in any feature. Frequency mean: 8.115 (training rows) vs 8.029 (full cohort).

Models (lapsed class is positive; confusion matrix rows are actual 0/1, columns predicted 0/1)
Model	Best parameters	CV F1	Test confusion	Precision	Recall	F1
Logistic regression, C in {0.01,0.1,1,10,100}	C = 100	0.651	[[111,32],[31,58]]	0.644	0.652	0.648
Random forest, depth {3,5,8} x leaf {5,10,20}	depth 8, leaf 10	0.681	[[113,30],[31,58]]	0.659	0.652	0.655
Logistic regression, balanced training data	C = 100	n/a	[[90,53],[13,76]]	0.589	0.854	0.697

Balancing: lapsed training rows duplicated (random_state=42) to 570 per class; recall +0.202, precision -0.055, F1 +0.049.

Interpretation: top random-forest features: frequency 0.280, monetary 0.271, recency 0.115, avg delivery 0.061, cancellation rate 0.052. Logistic coefficients: recency +0.040, frequency -1.698, monetary +0.285, avg discount +0.004, cancellation rate -0.124, avg delivery +0.079, age +0.023, tenure -0.050.

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

Few-shot cost:
The few-shot prompt costs about 92 extra tokens per ticket (153.1 vs 60.8, so 22,504 tokens in total against 8,932), but accuracy rises from 85.7% to 100%, removing all 21 errors. Eleven of those were fraud tickets sent to billing, which would have missed the security team and HIGH priority, so for fraud the extra tokens are clearly worth paying. When the stakes are low or the token budget is tight, zero-shot may be enough.

Budget and routing: zero-shot with TOKEN_BUDGET = 6000 used 6,023 tokens and 51 tickets ended budget_exhausted (triage_budget_demo.csv). Final few-shot run (budget 100000), 164 rows in triage_results.csv: human_review 17, security_team (HIGH) 34, premium_desk (HIGH) 11, support_queue 102.

## Part D
RAG: 40-word chunks with 10-word overlap give 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). TF-IDF with cosine similarity; the query "refund policy late delivery" returns three refund_policy.txt chunks.

Setting	Doc in top 3 (of 12)	Fact in answer (of 12)	Fallback (of 2)
Overlap 10	12	12	2
Overlap 0	12	10	2

At overlap 0 two questions break: the refund delivery-time question and the payment-failure question.

TF and IDF:
TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) lowers the weight of words that appear in many chunks, so rare and informative words count more. A limitation is that TF-IDF matches exact words only, so a question such as “money back” can miss a policy that says “refund”.

Why overlap 0 breaks two questions:
Without overlap, a chunk boundary cut the answer sentence in two, so the retrieved chunk held only part of it, for example the refund answer lost “more than 45 minutes” and the payment answer lost “within 48 hours”. The 10-word overlap repeats the end of one chunk at the start of the next, so each sentence appears whole in at least one chunk.

ReAct vs plan-and-execute (about 200 words, limit 250):
My ReAct agent asks the model for one tool at a time. It used 5 model calls for a full refund (S1), 4 for an on-time order (S4), 3 for a cancelled order (S5) and 2 for a missing order (S6). Each call resends the growing message history, so tokens rise at every step (S1 used 1,112), but the agent stops as soon as it has an answer. A plan-and-execute design asks for the whole step list in one call, which could be fewer calls and tokens for a normal refund. But it plans before seeing any result, so for a missing or cancelled order the planned steps would still run or fail, and the agent would need to replan, costing extra calls. ReAct handles these surprises naturally: S6 stopped after 2 calls and S5 after 3, because each decision depends on the last observation. I would choose ReAct for this refund workflow. Decisions depend on the order’s status, delivery time and amount (above INR 1,000 goes to a human), money is involved, and checking each result before acting is safer. Each step is also checkpointed, so a crashed run can resume.

Limitations and next steps
Three next steps:

* Use a time-based train and test split (train on earlier months, test on later ones) to reflect real use.
* Add an idempotency key per order so a refund cannot be issued twice.
* Replace TF-IDF with embedding-based retrieval so differently worded questions still find the right policy.

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

ReAct vs plan-and-execute, using S1 5, S4 4, S5 3, S6 2 model calls; cover calls and tokens, surprises like NOT_FOUND or a cancelled order, and your choice.

## Limitations and Next Steps
•	Data is synthetic and the model is a rule-based mock, so accuracy figures (such as 100% few-shot) say little about real tickets.
•	The retention list is illustrative and not a leakage-free evaluation: the Premium flag is a post-hoc whole-window label that includes September.
•	Lapse model uses one cutoff and one random split; TF-IDF retrieval is lexical; the refund ledger has no idempotency check (the crash demo refunds S1 twice across runs).


