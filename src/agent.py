import datetime
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm
from rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = "state"
CURRENT = {"run_id": None, "scenario_id": None}          # lets ledger lines record which run wrote them


class SimulatedCrash(Exception):
    pass


# ---------------- tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                          "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    # more than 45 minutes -> 100% of the amount; 45 or less (or never delivered) -> 0
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _write_ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    line = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr,
            "run_id": CURRENT["run_id"], "scenario_id": CURRENT["scenario_id"]}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _write_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _write_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


def _obj(**props):
    return {"type": "object", "properties": props, "required": list(props)}


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": _obj(order_id={"type": "string"})},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching passages.",
     "parameters": _obj(query={"type": "string"})},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": _obj(order_amount={"type": "number"}, delivery_minutes={"type": "number"})},
    {"name": "issue_refund", "description": "Issue an automatic refund (up to INR 1000) and record it in the ledger.",
     "parameters": _obj(order_id={"type": "string"}, refund_inr={"type": "number"})},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": _obj(order_id={"type": "string"}, refund_inr={"type": "number"})},
]

SYSTEM_PROMPT = ("You are a refund AGENT for Instamart support. Decide each refund request by calling "
                 "tools one at a time: get_order, search_policy, calculate_refund, then issue_refund or "
                 "escalate_to_human. Finish with a final answer that starts with REFUND, ESCALATE, "
                 "DENIED or NOT_FOUND, followed by a colon.")


def _execute(name, args):
    fn = TOOL_MAP.get(name)                              # look the name up; never trust it blindly
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------- state, checkpoints, results ----------------
def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    state["step"] = len(state["steps"])
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)


def _refund_of(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human") and isinstance(s["observation"], dict):
            return s["observation"].get("refund_inr", 0)
    return 0


def _result(state):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": state["status"],
            "final_answer": state["final_answer"], "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": _refund_of(state["steps"])}


def _fallback(state, status, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    state["status"] = status
    state["final_answer"] = (f"{status}: stopped because {reason}. Handing this case to a human agent. "
                             f"Tools run so far: {done}. Tokens used: {state['tokens_used']}.")
    return _result(state)


# ---------------- the ReAct loop ----------------
def _loop(state, crash_after_tools=None):
    CURRENT["run_id"], CURRENT["scenario_id"] = state["run_id"], state["scenario_id"]
    lim = state["limits"]
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", "the time limit was reached")
        if state["llm_calls"] >= lim["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS",
                             "the maximum number of model calls was reached without a final answer")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > lim["token_limit"]:     # stop before running the requested tool
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", "the token limit was exceeded")

        msg = resp.choices[0].message
        if not msg.tool_calls:                            # no tool requested: this is the final answer
            final = msg.content or ""
            state["status"] = final.split(":")[0].strip()
            state["final_answer"] = final
            return _result(state)

        tc = msg.tool_calls[0]
        args = dict(tc.arguments)
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": args}]})
        observation = _execute(tc.name, args)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": tc.name, "arguments": args, "observation": observation})
        state["tools_executed"].append(tc.name)
        _save_checkpoint(state)                           # checkpoint AFTER every observation
        if crash_after_tools and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": []}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)                              # messages, tokens, steps, tools already run
    return _loop(state)                                   # continues; the time limit restarts on resume