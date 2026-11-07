import collections
import pandas as pd
from rag import PolicyIndex, answer, retrieve

ev = pd.read_csv("data/raw/policy_eval.csv", dtype=str, keep_default_na=False)


def evaluate(index, label):
    n_ans = n_none = doc_hit = fact_hit = fallback = 0
    problems = []
    for row in ev.itertuples(index=False):
        text, sources = answer(row.question, index)
        if row.expected_doc == "NONE":
            n_none += 1
            if text.startswith("I could not find"):
                fallback += 1
            else:
                problems.append((row.question, "expected the fallback, got: " + text))
        else:
            n_ans += 1
            d, f = row.expected_doc in sources, row.expected_fact in text   # case-sensitive substring
            doc_hit += d
            fact_hit += f
            if not (d and f):
                problems.append((row.question, f"doc in top 3: {d} | fact in answer: {f} | answer: {text}"))
    print(f"\n=== {label} ===")
    print(f"expected doc in top 3: {doc_hit}/{n_ans} | expected fact in answer: {fact_hit}/{n_ans} "
          f"| fallback on unanswerable: {fallback}/{n_none}")
    for q, why in problems:
        print("  PROBLEM:", q, "->", why)


main = PolicyIndex(40, 10)
print("chunks:", len(main.chunks), dict(collections.Counter(c["source"] for c in main.chunks)),
      "(ref 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
print("\nquery 'refund policy late delivery':")
for h in retrieve("refund policy late delivery"):
    print("  ", h["source"], h["score"])
print("(ref: all three from refund_policy.txt)")

evaluate(main, "overlap 10")
evaluate(PolicyIndex(40, 0), "overlap 0")