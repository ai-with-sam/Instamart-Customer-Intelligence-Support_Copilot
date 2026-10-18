# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone project builds a customer intelligence and support copilot using synthetic Instamart-style data and an offline mock language model. It combines data cleaning, SQL and pandas analysis, customer segmentation, lapse prediction, support-ticket triage, policy retrieval, and a tool-using refund agent.

The project is organized into four parts:

- **Part A — Data Foundation:** Data auditing, cleaning, SQLite database creation, SQL queries, pandas analysis, and exploratory analysis.
- **Part B — Customer Intelligence:** RFM feature engineering, K-means customer segmentation, and lapse prediction.
- **Part C — Support Ticket Triage:** Privacy redaction, ticket classification, retry logic, validation, routing, and token-budget controls.
- **Part D — Policy RAG and Refund Agent:** Policy retrieval, answer-grounding evaluation, refund decisions, safety limits, and crash/resume checkpoints.

The project uses Python, pandas, NumPy, scikit-learn, SQLite, and the supplied `mock_llm.py`. The supplied `generate_data.py` and `mock_llm.py` are used without modification. No API key, paid model, or real external model call is required.

## Reproduce

Run these commands from the project root in a fresh Windows PowerShell environment.

### 1. Create the environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Generate the synthetic data

```powershell
python generate_data.py
```

### 3. Audit, clean, and validate the data

```powershell
python scripts\audit_data.py
python scripts\clean_data.py
python scripts\validate_data.py
```

### 4. Run SQL and pandas analysis

```powershell
python scripts\run_sql_analysis.py
python scripts\pandas_analysis.py
```

### 5. Build RFM features and customer segments

```powershell
python scripts\build_rfm.py
python scripts\cluster_customers.py
python scripts\profile_segments.py
```

### 6. Build the lapse dataset and train the models

```powershell
python scripts\build_lapse_dataset.py
python scripts\train_lapse_models.py
```

### 7. Run ticket triage

```powershell
python scripts\triage_tickets.py
```

### 8. Evaluate policy retrieval and run the refund agent

```powershell
python scripts\policy_rag.py
python scripts\inspect_refund_orders.py
python scripts\refund_agent.py
```

The scripts create the cleaned data, SQLite database, model artifacts, evaluation outputs, agent logs, and checkpoints. The required generated artifacts are also committed in the repository so that the evaluated results can be inspected without rerunning the entire pipeline.

## Part A — Data Foundation

### Data generation, cleaning, and validation

The project uses synthetic customer, product, order, and support-ticket data. The initial generated dataset contained:

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Support tickets | 164 |

The cleaning pipeline produces three cleaned CSV files and a SQLite database.

| Cleaned dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

Of the cleaned orders, **11,110 were completed** and **1,199 were cancelled**. Total revenue from completed orders was **INR 2,998,668.80**.

The cleaned data is stored in `data/clean/`, and the SQLite database is `data/instamart.db`. Data auditing and validation scripts are included under `scripts/`.

### SQL analysis and pandas cross-check

All seven SQL queries were executed against `data/instamart.db` using `pandas.read_sql_query()`. All seven queries passed their reference checks.

| Query | Technique | Result |
|---|---|---|
| Q1 | `WHERE` and `HAVING` | 30 qualifying high-spend customers. |
| Q2 | Three-table `JOIN` | 30 city/category combinations; total completed revenue: INR 2,998,668.80. The largest combination is Bengaluru / Electronics at INR 233,909.45. |
| Q3 | Scalar subquery | 11 products are priced above the average catalogue price. |
| Q4 | `LEFT JOIN` | 25 customers have no completed orders. |
| Q5 | Common table expressions (CTEs) | 550 customers spend above the average completed spend of INR 2,552.06. |
| Q6 | Window functions | 1,175 customers ranked. Eight rows have different `RANK()` and `ROW_NUMBER()` values. |
| Q7 | Running-total window function | 183 daily revenue rows; final cumulative completed revenue: INR 2,998,668.80. |

### Q6: Top customer in each city

| City | Customer ID | Completed spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

`RANK()` assigns the same rank to tied values and leaves gaps after ties. `ROW_NUMBER()` assigns a unique sequential number to every row. Therefore, their results can differ when customers have equal spend.

### pandas cross-check

The city/category revenue pivot table was rebuilt using `pandas.pivot_table()` with `margins=True`. Its grand total matches the SQL completed-revenue total of INR 2,998,668.80.

| City | Revenue sum (INR) | Mean per completed order (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| **Total** | **2,998,668.80** | — | **11,110** |

### Completed-order amount distribution

Statistics were calculated for 11,110 completed orders.

- **Mean:** INR 269.91
- **Median:** INR 135.00
- **Skewness:** 4.6088
- **Minimum:** INR 12.00
- **First quartile (Q1):** INR 64.80
- **Third quartile (Q3):** INR 297.00
- **Maximum:** INR 3,998.00
- **Interquartile range (IQR):** INR 232.20
- **Lower 1.5 × IQR fence:** INR -283.50
- **Upper 1.5 × IQR fence:** INR 645.30
- **Outliers below the lower fence:** 0
- **Outliers above the upper fence:** 897

### Correlation analysis

The Pearson correlation matrix was calculated for `order_amount`, `discount_pct`, `quantity`, and `delivery_minutes` among completed orders.

- **Pair with the largest absolute correlation:** `order_amount` and `discount_pct`
- **Correlation coefficient:** -0.0951

This indicates a very weak negative linear relationship. Correlation does not establish causation.

### Exploratory analysis: four questions

1. **What is this data about?**

   The synthetic dataset contains customer, product, and order records. This analysis examines completed-order spending, revenue by city and category, and delivery times.

2. **What do the statistics show?**

   The mean order amount (INR 269.91) exceeds the median (INR 135.00). Positive skewness (4.6088) indicates a strongly right-skewed distribution.

3. **Are there outliers or anomalies?**

   There are 897 order amounts above the upper IQR fence of INR 645.30 and none below the lower fence. The data audit also identified duplicates, invalid amounts, and negative delivery times.

4. **What relationships exist?**

   Order amount and discount percentage have the strongest absolute correlation (-0.0951), a very weak negative relationship. Correlation does not prove causation.

## Part B — Customer Segments and Lapse Prediction

### RFM features and customer segmentation

The RFM dataset contains **1,175 customers**. Recency has a median of 19 days and ranges from 0 to 180 days. Mean purchase frequency is 9.46 orders, with a maximum of 33. Mean monetary value is INR 2,552.06, with a maximum of INR 14,766.55.

K-means was evaluated for \(k=2\) through \(k=8\) using inertia and silhouette score. **Four clusters** were selected; this configuration achieved the highest tested silhouette score of **0.526**.

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary value (INR) |
|---|---:|---:|---:|---:|
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal Active | 419 | 8.77 | 19.40 | 3,410.02 |
| Regular | 436 | 28.86 | 3.77 | 1,057.62 |
| At Risk | 182 | 109.04 | 2.20 | 685.99 |

The first two principal components explain **85.2%** of the variance (PC1: 63.4%; PC2: 21.8%). The segment names are descriptive labels assigned to the clusters.

### Lapse prediction

The prediction cutoff is 1 September 2025. The cohort includes customers with at least one completed order before the cutoff. A customer is labelled lapsed if they have no completed order during September 2025. Model features use pre-cutoff order history and customer information.

The cohort contains **1,156 customers**: 443 lapsed (38.32%) and 713 not lapsed. A stratified 80/20 train-test split with `random_state=42` produces 924 training rows and 232 test rows.

| Model | Test accuracy | Precision (lapsed) | Recall (lapsed) | F1 (lapsed) |
|---|---:|---:|---:|---:|
| Always predict not lapsed | 0.616 | 0.000 | 0.000 | 0.000 |
| Tuned Logistic Regression | 0.728 | 0.644 | 0.652 | 0.648 |
| Tuned Random Forest | 0.724 | 0.640 | 0.640 | 0.640 |
| Oversampled Logistic Regression | 0.716 | 0.589 | 0.854 | 0.697 |

The Random Forest achieved the best cross-validation F1 score (**0.687**) and was selected as the final model. Oversampling was applied only to the training data. It improved lapsed-customer recall but reduced precision. The most important Random Forest features were monetary value, purchase frequency, and recency.

The saved model was reloaded and its predictions matched the original predictions. `outputs/retention_list.csv` contains the top 10 Premium customers in the test set ranked by predicted lapse probability.

**Evaluation caveat:** The Premium designation comes from RFM segmentation calculated over the full available order window, including September. Therefore, the retention list is illustrative and should not be interpreted as a fully leakage-free evaluation of the Premium targeting strategy.

## Part C — Support Ticket Triage with Guardrails

The triage pipeline processes **164 tickets** using the supplied offline `mock_llm.py`. It classifies tickets into billing, delivery, technical, or fraud categories and routes them according to classification, customer segment, and review requirements.

### Privacy and input guardrails

- Redaction runs before sending ticket text to the mock model: email addresses become `[EMAIL]`, 10-digit Indian mobile numbers become `[PHONE]`, and order IDs become `[ORDER_ID]`.
- The run replaced 55 email references, 91 phone references, and 132 order-ID references: **278 substitutions** in total. Rechecking the redacted text found zero remaining matches for the three target patterns.
- Four empty tickets were marked `skipped_empty`; four tickets exceeding the 400-token threshold were marked `flagged_long`. Neither group was sent for classification.
- Transient model failures were retried up to three times after the initial attempt, using exponential backoff with jitter.
- Valid JSON with an allowed category is marked `classified_json`. A non-JSON response is accepted only if exactly one allowed category is identifiable; otherwise, it is flagged for human review.

### Classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Valid JSON classifications | 135 | 135 |
| Keyword-fallback classifications | 12 | 12 |
| Invalid outputs flagged | 9 | 9 |
| Long tickets flagged | 4 | 4 |
| Empty tickets skipped | 4 | 4 |
| Tickets classified | 147 | 147 |
| Accuracy on classified tickets | 85.71% | 100.00% |
| Total estimated tokens | 14,080 | 22,348 |
| Tokens per classified ticket | 95.78 | 152.03 |

Few-shot examples improved accuracy on classified tickets in this deterministic mock evaluation, from 85.71% to 100%, while using 8,268 additional tokens (about 58.7% more). These results should not be treated as evidence of equivalent performance from a real-world language model.

The zero-shot confusion matrix for classified tickets, with rows as actual categories and columns as predicted categories, is:

| Actual / Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

### Routing and token budget

| Route | Priority | Tickets |
|---|---|---:|
| Human review | NORMAL | 17 |
| Premium desk | HIGH | 11 |
| Security team | HIGH | 34 |
| Support queue | NORMAL | 102 |

The budget demonstration used a 6,000-token limit and consumed 6,050 estimated tokens before stopping, with 89 tickets not processed. This small overshoot is expected because the implementation checks the running total before starting each call; one call can take the total past the threshold. Invalid classifications, empty tickets, and long tickets remain review cases rather than receiving guessed categories.

## Part D — Policy RAG Assistant and Refund Agent

### Policy retrieval and grounding

The policy assistant indexes five local policy documents using 40-word chunks with a 10-word overlap. This produces **15 chunks**: delivery (3), escalation (3), payment (3), privacy (2), and refund (4). Retrieval uses TF-IDF vectors and cosine similarity. No external API or network model is used.

For the query `refund policy late delivery`, all top-three retrieved chunks came from `refund_policy.txt`.

| Evaluation | Chunks | Expected document in top 3 | Expected fact found | Unanswerable fallback |
|---|---:|---:|---:|---:|
| 10-word overlap | 15 | 12/12 | 12/12 | 2/2 |
| Zero overlap | 13 | 12/12 | 10/12 | 2/2 |

The zero-overlap experiment demonstrates that retrieval of the expected document does not guarantee that the answer contains the required fact. Two questions lost fact coverage: the full-refund condition of delivery taking more than 45 minutes, and the 48-hour timeline for a deducted payment that failed. The latter response also contained unsupported text from the mock model. Chunk overlap improved fact coverage in this evaluation, but retrieval and answer grounding still require explicit checks.

### Refund-agent scenario results

The ReAct-style refund agent uses local database and policy tools. Its default token limit is 6,000, and its maximum-iteration guard stops repeated tool-request loops.

| Scenario | Final status | Refund (INR) | Model calls | Estimated tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,817 |
| S2 | REFUND | 135.00 | 5 | 1,817 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,804 |
| S4 | DENIED | 0.00 | 4 | 1,341 |
| S5 | DENIED | 0.00 | 3 | 889 |
| S6 | NOT_FOUND | 0.00 | 2 | 440 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 4,104 |

S1 and S2 were refunded; S3 was escalated; S4 and S5 were denied; S6 returned `NOT_FOUND`; and the repeated-tool scenario S7 safely ended with `FALLBACK_MAX_ITERATIONS` under the default 6,000-token limit.

### Safety demonstrations

| Demonstration | Configuration | Result |
|---|---|---|
| Token limit | S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| Time limit | S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |

The token-limit run stopped before executing the tool requested by the call that exceeded the configured budget. The zero-second time limit stopped the run before its first model call. Both fallback paths return a handoff summary.

### Crash and resume

The simulated S1 crash occurred immediately after the second tool execution and after its checkpoint was saved. Resuming from that checkpoint completed the refund without repeating an earlier tool execution. The final status was **REFUND**, with **INR 128.25** issued.

The exact tool sequence across the interrupted and resumed run was:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

Checkpoint files are stored in `state/`. Run details and step logs are stored in `outputs/agent_runs.jsonl`; refund and escalation actions are recorded in `outputs/refund_ledger.jsonl`.

### ReAct versus plan-and-execute

The implemented ReAct agent makes decisions using successive model calls and tool observations. In the measured runs, S1 required 5 calls and 1,817 estimated tokens, S5 required 3 calls and 889 tokens, and S6 required 2 calls and 440 tokens. A plan-and-execute design could ask the model for a complete step list once, potentially reducing calls and repeated prompt context on straightforward cases. Its actual token savings would depend on the length of the plan, the tool observations, and whether replanning is needed.

The trade-off is adaptability. If an order is missing, cancelled, or otherwise differs from expectations, ReAct can incorporate the new observation before choosing the next action. A precomputed plan needs explicit validation and a replan path to avoid continuing with invalid assumptions. For refunds, I would choose a guarded ReAct approach because refund eligibility and order state must be checked before money is issued. The additional calls are justified by stepwise verification, provided that token, time, and iteration limits are enforced and refund actions are idempotent.

## Limitations and Next Steps

- **Synthetic data:** The data and offline mock LLM are for demonstration, not evidence of production performance.
- **Lapse prediction:** The cohort and label depend on the chosen cutoff and September outcome window. The test set is relatively small, and the model's precision and recall indicate that predictions should support—not replace—retention decisions.
- **Segment targeting:** The Premium retention list uses a segment label derived from the full order window, including September; the list is illustrative rather than a leakage-free targeting evaluation.
- **Ticket triage:** Redaction covers only the specified email, phone, and order-ID patterns. Other personal information and alternate formats may remain. Invalid, empty, and long tickets require human review. Few-shot accuracy reflects the deterministic mock model and this synthetic ticket set.
- **Policy RAG:** TF-IDF retrieval is lexical and may miss semantically relevant content. Overlap improved fact coverage in the evaluation, but answers still need evidence checks and an explicit fallback for unsupported questions.
- **Refund safety:** The agent uses simulated tools and a local database, not a payment gateway. The ledger records a repeated refund for ORD-100048 because S1 and the crash/resume demonstration are separate runs. A real system should enforce idempotency keys, durable transaction boundaries, authorization checks, and audit controls to prevent duplicate refunds.
- **Next steps:** Test on representative, permissioned data; validate model performance across time-based holdouts; broaden privacy detection; measure retrieval quality on more policy questions; and add idempotent, transaction-safe refund execution before any real payment integration.
