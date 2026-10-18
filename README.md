# Instamart Customer Intelligence & Support Copilot

## Overview

This project uses synthetic data to build a customer intelligence and support copilot.

## Reproduce

Run these commands from the project root in a fresh environment:

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the environment in Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the synthetic data:

   ```powershell
   python generate_data.py
   ```

## Part A — Data Foundation

### SQL Analysis and pandas Cross-check

#### SQL analysis

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

#### Q6: Top customer in each city

| City | Customer ID | Completed spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

`RANK()` assigns the same rank to tied values and leaves gaps after ties. `ROW_NUMBER()` assigns a unique sequential number to every row. Therefore, their results can differ when customers have equal spend.

#### pandas cross-check

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

#### Completed-order amount distribution

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

#### Correlation analysis

The Pearson correlation matrix was calculated for `order_amount`, `discount_pct`, `quantity`, and `delivery_minutes` among completed orders.

- **Pair with the largest absolute correlation:** `order_amount` and `discount_pct`
- **Correlation coefficient:** -0.0951

This indicates a very weak negative linear relationship. Correlation does not establish causation.

#### Exploratory analysis: four questions

1. **What is this data about?**

   The synthetic dataset contains customer, product, and order records. This analysis examines completed-order spending, revenue by city and category, and delivery times.

2. **What do the statistics show?**

   The mean order amount (INR 269.91) exceeds the median (INR 135.00). Positive skewness (4.6088) indicates a strongly right-skewed distribution.

3. **Are there outliers or anomalies?**

   There are 897 order amounts above the upper IQR fence of INR 645.30 and none below the lower fence. The data audit also identified duplicates, invalid amounts, and negative delivery times.

4. **What relationships exist?**

   Order amount and discount percentage have the strongest absolute correlation (-0.0951), a very weak negative relationship. Correlation does not prove causation.

## Part B — Customer Segments and Lapse Prediction

To be completed.

## Part C — Support Ticket Triage with Guardrails

To be completed.

## Part D — Policy RAG Assistant and Refund Agent

To be completed.

## Limitations and Next Steps

To be completed.