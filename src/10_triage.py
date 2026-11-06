import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm                                          # lives in the repo root
from mock_llm import est_tokens, TransientLLMError

random.seed(42)
os.makedirs("outputs", exist_ok=True)

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATS = ["billing", "delivery", "technical", "fraud"]
CLASSIFIED = ["classified_json", "classified_keyword"]
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output",
         "budget_exhausted", "failed_permanent"}
COLS = ["ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used"]

# ---------- redaction (order matters: email, phone, order id) ----------
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, n = pattern.subn(placeholder, text)
        counts[name] = n
    return text, counts

def assert_no_pii(messages):
    for m in messages:
        for name, pattern, _ in PATTERNS:
            assert not pattern.search(m["content"]), f"{name} found in a prompt"

# ---------- prompts ----------
ZERO_SHOT = ("You CLASSIFY customer support tickets. "
             "Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Money was taken twice for my order [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}
Ticket: My parcel is very late and the rider is not answering. -> {"category": "delivery", "reason": "late delivery"}
Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app problem"}
Ticket: Someone used my card without permission, I did not buy this. -> {"category": "fraud", "reason": "unauthorized transaction"}"""

# ---------- resilient call ----------
def call_with_retry(messages, ticket_id, state):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            state["transient"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            line = f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s"
            state["retry_log"].append((ticket_id, attempt, wait, line))
            time.sleep(wait)

# ---------- output validation ----------
def validate(content):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        found = set(re.findall(r"[a-z]+", (content or "").lower())) & set(CATS)
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    cat = obj.get("category") if isinstance(obj, dict) else None
    if isinstance(cat, str) and cat in CATS:
        return "classified_json", cat
    return "flagged_invalid_output", ""

# ---------- one full run ----------
def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()                                     # immediately before each full run
    state = {"transient": 0, "retry_log": []}
    total_tokens, rows = 0, []
    for t in tickets.itertuples(index=False):
        used, category = 0, ""
        if t.text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            clean, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean}]
            assert messages[1]["content"] == clean
            assert_no_pii(messages)
            try:
                resp = call_with_retry(messages, t.ticket_id, state)
            except TransientLLMError:
                status = "failed_permanent"
            else:
                used = resp.usage.total_tokens
                total_tokens += used
                status, category = validate(resp.choices[0].message.content)
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id,
                     "status": status, "category": category,
                     "true_category": t.true_category, "tokens_used": used})
    return pd.DataFrame(rows), state, total_tokens

# ---------- routing with Part B segments ----------
seg = pd.read_csv("outputs/customer_segments.csv")
seg["is_premium"] = seg["is_premium"].astype(str).str.lower().eq("true")
seg_name = seg.set_index("customer_id")["segment_name"]
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])

def add_routing(df):
    df = df.copy()
    df["segment_name"] = df["customer_id"].map(seg_name).fillna("")
    routes, priorities = [], []
    for status, cat, cid in zip(df["status"], df["category"], df["customer_id"]):
        if status in HUMAN:
            r, p = "human_review", "NORMAL"
        elif cat == "fraud":
            r, p = "security_team", "HIGH"
        elif cid in premium_ids:
            r, p = "premium_desk", "HIGH"
        else:
            r, p = "support_queue", "NORMAL"
        routes.append(r); priorities.append(p)
    df["route"], df["priority"] = routes, priorities
    return df

def summarize(name, df, state, total_tokens):
    ok = df[df["status"].isin(CLASSIFIED)]
    correct = int((ok["category"] == ok["true_category"]).sum())
    print(f"\n=== {name} ===")
    print(df["status"].value_counts().to_string())
    print(f"classified: {len(ok)} | accuracy: {correct}/{len(ok)} = {correct / len(ok) * 100:.1f}%")
    print(f"total tokens: {total_tokens} | per classified ticket: {total_tokens / len(ok):.1f}")
    print(f"transient errors: {state['transient']} | retries logged: {len(state['retry_log'])}")
    return ok

def check_retry_log(state, show):
    for tid, n, wait, line in state["retry_log"]:
        assert BASE_DELAY * 2 ** n * 0.9 <= wait <= BASE_DELAY * 2 ** n * 1.15, line
        if show:
            print(line)

# =====================================================================
tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)
print("tickets:", len(tickets))

# ---- Task 1: guardrail counts ----
tok = tickets["text"].map(est_tokens)
print("empty:", int((tickets["text"].str.strip() == "").sum()), "(ref 4) | long:",
      int((tok > MAX_INPUT_TOKENS).sum()), "(ref 4) | long ticket tokens:",
      sorted(set(tok[tok > MAX_INPUT_TOKENS])), "(ref 734)")

# ---- Task 2: redaction counts over all 164 tickets ----
totals = {name: 0 for name, _, _ in PATTERNS}
redacted = []
for text in tickets["text"]:
    clean, c = redact(text)
    redacted.append(clean)
    for k, v in c.items():
        totals[k] += v
print("substitutions:", totals, "| total:", sum(totals.values()), "(ref 55 / 91 / 132 / 278)")
left = sum(len(p.findall(x)) for x in redacted for _, p, _ in PATTERNS)
print("left after re-applying the patterns:", left, "(ref 0)")
assert left == 0

# ---- Task 3: prompt sizes ----
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
print("prompt tokens: zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

# ---- Tasks 4-6: zero-shot and few-shot full runs ----
zs, zs_state, zs_tok = run_pipeline(tickets, ZERO_SHOT)
print("\nRETRY LOG (zero-shot run):")
check_retry_log(zs_state, show=True)
zs_ok = summarize("ZERO-SHOT", zs, zs_state, zs_tok)
print("(ref statuses 135 / 12 / 9 | accuracy 85.7% | 147 classified | 19 transient errors)")
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zs_ok["true_category"], zs_ok["category"]))

fs, fs_state, fs_tok = run_pipeline(tickets, FEW_SHOT)
check_retry_log(fs_state, show=False)
summarize("FEW-SHOT", fs, fs_state, fs_tok)
print("(ref accuracy 100%)")

# ---- Task 7a: budget demo ----
BUDGET_DEMO = 6000
demo, demo_state, demo_tok = run_pipeline(tickets, ZERO_SHOT, token_budget=BUDGET_DEMO)
n_ex = int((demo["status"] == "budget_exhausted").sum())
print(f"\nBUDGET DEMO: budget {BUDGET_DEMO} | tokens used {demo_tok} | budget_exhausted: {n_ex}")
assert n_ex >= 1
add_routing(demo)[COLS].to_csv("outputs/triage_budget_demo.csv", index=False)

# ---- Task 7b: final results (few-shot, budget 100000) ----
final, final_state, final_tok = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
results = add_routing(final)
assert len(results) == 164
results[COLS].to_csv("outputs/triage_results.csv", index=False)
print("\nFINAL ROUTING:")
print(results.groupby(["route", "priority"]).size().to_string())
print("\nsaved outputs/triage_budget_demo.csv and outputs/triage_results.csv")