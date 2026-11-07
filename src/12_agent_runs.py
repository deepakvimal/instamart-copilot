import glob
import json
import os

import pandas as pd
from agent import run_agent, resume_agent, SimulatedCrash

os.makedirs("outputs", exist_ok=True)
for p in ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl"] + glob.glob("state/*.json"):
    if os.path.exists(p):
        os.remove(p)                                      # start clean so reruns don't pile up lines


def log_run(res, note=""):
    line = {"scenario_id": res["scenario_id"], "status": res["status"],
            "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"], "steps": res["steps"],
            "run_id": res["run_id"], "note": note}
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


sc = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str, keep_default_na=False)

# ---------- Task 6: the seven scenarios ----------
rows = []
for r in sc.itertuples(index=False):
    res = run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
    log_run(res)
    expected_ok = (res["status"] == r.expected_outcome or
                   (r.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK")))
    amount_ok = abs(float(res["refund_inr"]) - float(r.expected_refund_inr)) < 0.01
    rows.append({"scenario": r.scenario_id, "status": res["status"], "refund_inr": res["refund_inr"],
                 "model_calls": res["llm_calls"], "tokens": res["tokens_used"],
                 "expected": r.expected_outcome, "expected_refund": r.expected_refund_inr,
                 "match": expected_ok and amount_ok})
print(pd.DataFrame(rows).to_string(index=False))
print("(ref: S1 REFUND 128.25 in 5 calls, S2 REFUND 135.00, S3 ESCALATE 1999.00, S4/S5 DENIED, "
      "S6 NOT_FOUND, S7 FALLBACK_MAX_ITERATIONS after 6 calls)")

s1 = sc[sc["scenario_id"] == "S1"].iloc[0]

# ---------- Task 7: stop-safely demonstrations ----------
tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_demo")
log_run(tok, "token_limit=200 demo")
print("\ntoken_limit=200 demo:", tok["status"], "| steps done:", len(tok["steps"]), "| tokens:", tok["tokens_used"])
print(tok["final_answer"])

tmo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
log_run(tmo, "time_limit_seconds=0 demo")
print("\ntime_limit_seconds=0 demo:", tmo["status"], "| llm_calls:", tmo["llm_calls"])

# ---------- Task 8: crash and resume ----------
try:
    run_agent(s1.message, "S1", run_id="S1_crash_demo", crash_after_tools=2)
except SimulatedCrash as exc:
    print("\nCRASH:", exc)
done = resume_agent("S1_crash_demo")
log_run(done, "crash after tool 2, then resume")
executed = [s["tool"] for s in done["steps"]]
print("tools executed across crash and resume:", executed)
print("final status:", done["status"], "| refund:", done["refund_inr"])
assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert done["status"] == "REFUND" and abs(done["refund_inr"] - 128.25) < 0.01
print("\nsaved outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and state/")