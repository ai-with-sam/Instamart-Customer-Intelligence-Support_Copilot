
from pathlib import Path
import json
import random
import re
import time


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mock_llm


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

RNG = random.Random(42)

ZERO_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, using this form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include Markdown or any text outside the JSON object.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category.
Allowed categories: billing, delivery, technical, fraud.
Reply with JSON only, using this form:
{"category": "...", "reason": "..."}
The category must be one of the four allowed categories.
Do not include Markdown or any text outside the JSON object.

EXAMPLES:
Ticket: "I was charged twice for my purchase."
Reply: {"category": "billing", "reason": "Duplicate charge"}

Ticket: "My parcel arrived late."
Reply: {"category": "delivery", "reason": "Late delivery"}

Ticket: "Someone made an unauthorized payment on my account."
Reply: {"category": "fraud", "reason": "Unauthorized payment"}
""".strip()


# --------------------------------------------------
# 2. Redact personal information
# --------------------------------------------------
def redact_text(text):
    """Apply the required patterns in the required order."""
    text = text or ""

    text, emails = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phones = PHONE_PATTERN.subn("[PHONE]", text)
    text, orders = ORDER_PATTERN.subn("[ORDER_ID]", text)

    counts = {
        "emails": emails,
        "phones": phones,
        "order_ids": orders,
    }
    return text, counts


def prepare_tickets():
    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False,
    )

    required = {
        "ticket_id", "customer_id", "order_id",
        "text", "true_category",
    }
    missing = required - set(tickets.columns)
    if missing:
        raise ValueError(f"Missing ticket columns: {sorted(missing)}")

    redacted_texts = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}

    for text in tickets["text"]:
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)

        for key, count in counts.items():
            totals[key] += count

        # Verify no protected patterns remain.
        assert not EMAIL_PATTERN.search(redacted)
        assert not PHONE_PATTERN.search(redacted)
        assert not ORDER_PATTERN.search(redacted)

    tickets["redacted_text"] = redacted_texts

    print("PRIVACY AUDIT")
    print("-" * 50)
    print(f"Email substitutions: {totals['emails']}")
    print(f"Phone substitutions: {totals['phones']}")
    print(f"Order ID substitutions: {totals['order_ids']}")
    print(f"Total substitutions: {sum(totals.values())}")
    print("Remaining matches after redaction: 0")

    assert len(tickets) == 164
    assert totals == {"emails": 55, "phones": 91, "order_ids": 132}

    return tickets


# --------------------------------------------------
# 3. Retry model calls with exponential backoff + jitter
# --------------------------------------------------
def call_with_retry(ticket_id, messages):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * RNG.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry_number={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)

    raise RuntimeError("Retry loop ended unexpectedly")


# --------------------------------------------------
# 4. Validate model output
# --------------------------------------------------
def validate_output(content):
    content = content or ""

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        category = parsed.get("category")

        # Valid JSON with an invalid category must not be guessed.
        if category in ALLOWED_CATEGORIES:
            return category, "classified_json"

        return "", "flagged_invalid_output"

    # Keyword fallback is only for output that is not valid JSON.
    words = re.findall(r"[a-z]+", content.lower())
    found = ALLOWED_CATEGORIES.intersection(words)

    if len(found) == 1:
        return next(iter(found)), "classified_keyword"

    return "", "flagged_invalid_output"


# --------------------------------------------------
# 5. Run one complete triage pass
# --------------------------------------------------
def run_pipeline(tickets, system_prompt, token_budget=None):
    # The mock must be reset immediately before every independent run.
    mock_llm.reset()

    running_tokens = 0
    results = []

    for row in tickets.to_dict("records"):
        ticket_id = row["ticket_id"]
        raw_text = row["text"]
        safe_text = row["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": "",
            "category": "",
            "tokens_used": 0,
        }

        # Guardrail 1: empty text never reaches the model.
        if not str(raw_text).strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        # Guardrail 2: long text never reaches the model.
        if mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # Budget check occurs before every model call.
        if token_budget is not None and running_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        # The user message must be exactly the redacted ticket text.
        assert not EMAIL_PATTERN.search(safe_text)
        assert not PHONE_PATTERN.search(safe_text)
        assert not ORDER_PATTERN.search(safe_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": safe_text},
        ]

        try:
            response = call_with_retry(ticket_id, messages)
        except mock_llm.TransientLLMError:
            # Human review is safer than guessing after retry exhaustion.
            result["status"] = "flagged_invalid_output"
            results.append(result)
            continue

        message = response.choices[0].message
        result["tokens_used"] = response.usage.total_tokens
        running_tokens += response.usage.total_tokens

        category, status = validate_output(message.content)
        result["category"] = category
        result["status"] = status
        results.append(result)

    return pd.DataFrame(results), running_tokens


def report_run(name, results, total_tokens):
    print(f"\n{name}")
    print("=" * 50)

    print("Status counts:")
    print(results["status"].value_counts().sort_index().to_string())

    classified = results[
        results["status"].isin(["classified_json", "classified_keyword"])
    ]

    if len(classified):
        accuracy = accuracy_score(
            classified["true_category"],
            classified["category"],
        )
        print(f"Tickets classified: {len(classified)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Tokens per classified ticket: {total_tokens / len(classified):.2f}")

    return classified


# --------------------------------------------------
# 6. Route tickets using customer segments
# --------------------------------------------------
def create_routing_output(results):
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)

    required = {"customer_id", "segment_name", "is_premium"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Missing segment columns: {sorted(missing)}")

    # Customers missing from the segment file are not Premium.
    segment_map = {}
    for row in segments.to_dict("records"):
        premium_value = str(row["is_premium"]).strip().lower()
        segment_map[row["customer_id"]] = {
            "segment_name": row["segment_name"],
            "is_premium": premium_value in {"true", "1", "yes"},
        }

    output_rows = []

    for row in results.to_dict("records"):
        segment = segment_map.get(
            row["customer_id"],
            {"segment_name": "", "is_premium": False},
        )

        status = row["status"]
        category = row["category"]
        segment_name = segment["segment_name"]

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif segment["is_premium"]:
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        output_rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"],
        })

    output = pd.DataFrame(output_rows)
    output.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

    print("\nFINAL ROUTING COUNTS")
    print(output.groupby(["route", "priority"]).size().to_string())
    print(f"Saved: {OUTPUT_DIR / 'triage_results.csv'}")
    assert len(output) == 164


# --------------------------------------------------
# 7. Run experiments and create required outputs
# --------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = prepare_tickets()

    # Experiment A: zero-shot, no budget.
    zero_results, zero_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
    )
    zero_classified = report_run(
        "ZERO-SHOT RUN",
        zero_results,
        zero_tokens,
    )

    print("\nZERO-SHOT CONFUSION MATRIX")
    labels = ["billing", "delivery", "technical", "fraud"]
    matrix = confusion_matrix(
        zero_classified["true_category"],
        zero_classified["category"],
        labels=labels,
    )
    print(pd.DataFrame(
        matrix,
        index=[f"actual_{x}" for x in labels],
        columns=[f"predicted_{x}" for x in labels],
    ).to_string())

    # Experiment B: few-shot, no budget.
    few_results, few_tokens = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
    )
    report_run("FEW-SHOT RUN", few_results, few_tokens)

    # Experiment C: zero-shot token budget demonstration.
    budget_results, budget_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_PROMPT,
        token_budget=6000,
    )
    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.drop(columns=["true_category"]).to_csv(
        budget_path,
        index=False,
    )
    exhausted = int((budget_results["status"] == "budget_exhausted").sum())

    print("\nTOKEN BUDGET DEMO")
    print(f"Budget: 6000 tokens")
    print(f"Actual tokens used: {budget_tokens}")
    print(f"Tickets stopped by budget: {exhausted}")
    print(f"Saved: {budget_path}")

    # Experiment D: required final few-shot run.
    final_results, final_tokens = run_pipeline(
        tickets,
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )
    report_run("FINAL FEW-SHOT RUN", final_results, final_tokens)

    create_routing_output(final_results)

    # Accuracy comparisons use only tickets successfully classified.
    few_classified = final_results[
        final_results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    print("\nZERO-SHOT VS FEW-SHOT")
    print(
        "Zero-shot classified accuracy: "
        f"{accuracy_score(zero_classified['true_category'], zero_classified['category']):.2%}"
    )
    print(
        "Few-shot classified accuracy:  "
        f"{accuracy_score(few_classified['true_category'], few_classified['category']):.2%}"
    )
    print("Part C processing complete.")


if __name__ == "__main__":
    main()
