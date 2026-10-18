
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "outputs"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

sys.path.insert(0, str(ROOT))
import mock_llm
from policy_rag import PolicyRetriever

retriever = PolicyRetriever(overlap=10)


class SimulatedCrash(RuntimeError):
    """Used only to demonstrate checkpoint and resume."""


# Five required tools: name, description, and parameter specification.
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant refund policy passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund eligibility and amount from order data.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible automatic refund in the simulated ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]


def get_order(order_id):
    """Read an order; never change the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            LIMIT 1
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            float(row["delivery_minutes"])
            if row["delivery_minutes"] is not None
            else None
        ),
    }


def search_policy(query):
    return {"results": retriever.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}

    # Full order amount for completed orders delivered after 45 minutes.
    # The model is responsible for checking order status first.
    return {"refund_inr": round(float(order_amount), 2)}


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def issue_refund(order_id, refund_inr):
    amount = round(float(refund_inr), 2)

    # Defense in depth: never automatically issue over INR 1000.
    if amount > 1000:
        return {
            "error": "refund exceeds automatic approval limit",
            "refund_inr": amount,
        }

    append_jsonl(LEDGER_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": amount,
    })
    return {"refund_inr": amount}


def escalate_to_human(order_id, refund_inr):
    amount = round(float(refund_inr), 2)
    append_jsonl(LEDGER_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": amount,
    })
    return {"refund_inr": amount}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


SYSTEM_PROMPT = """
You are an AGENT for a simulated refund support workflow.
Use tools in a ReAct sequence; inspect each observation before deciding the next step.
Look up the order, retrieve relevant policy, and check order status before calculating.
Only completed orders can be refunded.
If delivery took more than 45 minutes, the eligible refund is the full order amount.
If delivery took 45 minutes or less, the refund is zero.
A refund above INR 1000 must be escalated to a human and must never be issued automatically.
All simulated requests are treated as within the policy's seven-day window.
For missing orders, return a final answer beginning NOT_FOUND:.
For ineligible orders or zero refunds, return a final answer beginning DENIED:.
For an automatically issued refund, return REFUND:.
For a human escalation, return ESCALATE:.
Do not claim a refund was issued unless issue_refund returned successfully.
Use the tools supplied; do not invent order data.
"""


def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(state["run_id"]).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def finish(state, status, final_answer):
    result = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "tools_executed": state["tools_executed"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }
    append_jsonl(RUNS_PATH, result)
    return result


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    start = time.monotonic()

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
        "token_limit": token_limit,
        "max_iterations": max_iterations,
        "time_limit_seconds": time_limit_seconds,
    }
    return _run_loop(state, start, crash_after_tools)


def _run_loop(state, start, crash_after_tools=None):
    while state["llm_calls"] < state["max_iterations"]:
        elapsed = time.monotonic() - start
        if elapsed >= state["time_limit_seconds"]:
            return finish(
                state,
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Please hand this request to a human agent.",
            )

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        if state["tokens_used"] > state["token_limit"]:
            return finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Please hand this request to a human agent.",
            )

        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:
            text = message.content or "FALLBACK: No final answer was returned."
            status = text.split(":", 1)[0].strip()
            if status not in {
                "REFUND", "ESCALATE", "DENIED", "NOT_FOUND",
                "NEEDS_INFO", "FALLBACK",
            }:
                status = "FALLBACK"
            return finish(state, status, text)

        # Save the assistant's requested tool calls in the conversation.
        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in calls
            ],
        })

        for tc in calls:
            name = tc.name
            args = tc.arguments
            if isinstance(args, str):
                args = json.loads(args)

            state["step_number"] += 1
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**args)
                except Exception as exc:
                    observation = {"error": f"{type(exc).__name__}: {exc}"}

            state["tools_executed"].append(name)
            state["steps"].append({
                "step": state["step_number"],
                "tool": name,
                "arguments": args,
                "observation": observation,
            })

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(observation, ensure_ascii=False),
            })

            # Checkpoint only after the tool observation is safely recorded.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {crash_after_tools} tool executions"
                )

    return finish(
        state,
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Please hand this request to a human agent.",
    )


def resume_agent(run_id):
    path = checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run {run_id}")

    state = json.loads(path.read_text(encoding="utf-8"))
    return _run_loop(state, time.monotonic())


def load_scenarios():
    import pandas as pd
    path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    return pd.read_csv(path, keep_default_na=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios()

    print("RUNNING SEVEN REQUIRED SCENARIOS")
    for _, row in scenarios.iterrows():
        mock_llm.reset()
        result = run_agent(
            row["message"],
            row["scenario_id"],
            run_id=f"scenario_{row['scenario_id']}",
        )
        print(
            f"{row['scenario_id']} | {result['status']} | "
            f"calls={result['llm_calls']} | tokens={result['tokens_used']} | "
            f"tools={result['tools_executed']}"
        )

    # Required safety demonstrations.
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]
    mock_llm.reset()
    token_demo = run_agent(
        s1["message"], "S1_TOKEN_DEMO",
        token_limit=200, run_id="demo_token_limit",
    )
    print("\nTOKEN DEMO:", token_demo["status"])

    mock_llm.reset()
    timeout_demo = run_agent(
        s1["message"], "S1_TIMEOUT_DEMO",
        time_limit_seconds=0, run_id="demo_timeout",
    )
    print("TIMEOUT DEMO:", timeout_demo["status"])

    # Required crash-and-resume demonstration.
    mock_llm.reset()
    crash_id = "demo_crash_resume"
    try:
        run_agent(
            s1["message"], "S1_CRASH_RESUME",
            run_id=crash_id, crash_after_tools=2,
        )
    except SimulatedCrash as exc:
        print("\nEXPECTED:", exc)

    resumed = resume_agent(crash_id)
    print("RESUMED STATUS:", resumed["status"])
    print("RESUMED REFUND ANSWER:", resumed["final_answer"])
    print("TOOLS ACROSS CRASH + RESUME:", resumed["tools_executed"])

    print("\nLogs saved:")
    print(RUNS_PATH)
    print(LEDGER_PATH)
    print(STATE_DIR)


if __name__ == "__main__":
    main()
