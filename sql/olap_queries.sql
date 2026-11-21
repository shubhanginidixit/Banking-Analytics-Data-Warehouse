-- 1. ROLL-UP: total deposit amount by year -> quarter -> month
SELECT dd.year,
    dd.quarter,
    dd.month_name,
    SUM(ft.amount) AS total_deposits
FROM fact_transactions ft
    JOIN dim_date dd ON ft.date_id = dd.date_id
WHERE ft.transaction_type = 'Deposit'
GROUP BY ROLLUP (dd.year, dd.quarter, dd.month_name)
ORDER BY dd.year,
    dd.quarter,
    dd.month_name;
-- 2. DRILL-DOWN: transaction volume from region -> city -> branch
SELECT db.region,
    db.city,
    db.branch_name,
    COUNT(*) AS txn_count,
    SUM(ft.amount) AS txn_total
FROM fact_transactions ft
    JOIN dim_branch db ON ft.branch_id = db.branch_id
GROUP BY db.region,
    db.city,
    db.branch_name
ORDER BY db.region,
    db.city,
    db.branch_name;
-- 3. SLICE: loan portfolio for a single loan type (Home loans only)
SELECT dlt.loan_type_name,
    db.branch_name,
    COUNT(*) AS loan_count,
    SUM(fl.amount) AS total_disbursed
FROM fact_loans fl
    JOIN dim_loan_type dlt ON fl.loan_type_id = dlt.loan_type_id
    JOIN dim_branch db ON fl.branch_id = db.branch_id
WHERE dlt.loan_type_name = 'Home'
GROUP BY dlt.loan_type_name,
    db.branch_name
ORDER BY total_disbursed DESC;
-- 4. DICE: transactions in Q1 for branches in the South region only
SELECT db.region,
    db.branch_name,
    dd.quarter,
    ft.transaction_type,
    SUM(ft.amount) AS total_amount
FROM fact_transactions ft
    JOIN dim_branch db ON ft.branch_id = db.branch_id
    JOIN dim_date dd ON ft.date_id = dd.date_id
WHERE dd.quarter = 1
    AND db.region = 'South'
GROUP BY db.region,
    db.branch_name,
    dd.quarter,
    ft.transaction_type
ORDER BY db.branch_name;
-- 5. CUBE: transaction totals across every combination of branch region and transaction type
SELECT db.region,
    ft.transaction_type,
    SUM(ft.amount) AS total_amount,
    COUNT(*) AS txn_count
FROM fact_transactions ft
    JOIN dim_branch db ON ft.branch_id = db.branch_id
GROUP BY CUBE (db.region, ft.transaction_type)
ORDER BY db.region NULLS LAST,
    ft.transaction_type NULLS LAST;
-- 6. Branch-wise deposits (net inflow), ranked
SELECT db.branch_name,
    db.city,
    SUM(
        CASE
            WHEN ft.transaction_type = 'Deposit' THEN ft.amount
            ELSE 0
        END
    ) AS total_deposits
FROM fact_transactions ft
    JOIN dim_branch db ON ft.branch_id = db.branch_id
GROUP BY db.branch_name,
    db.city
ORDER BY total_deposits DESC
LIMIT 10;
-- 7. Loan distribution by loan type
SELECT dlt.loan_type_name,
    COUNT(*) AS loan_count,
    SUM(fl.amount) AS total_amount,
    ROUND(AVG(fl.interest_rate), 2) AS avg_interest_rate
FROM fact_loans fl
    JOIN dim_loan_type dlt ON fl.loan_type_id = dlt.loan_type_id
GROUP BY dlt.loan_type_name
ORDER BY total_amount DESC;
-- 8. Customer segmentation: transaction behaviour by segment and age group
SELECT dc.customer_segment,
    dc.age_group,
    COUNT(DISTINCT dc.customer_id) AS customers,
    ROUND(AVG(ft.amount), 2) AS avg_txn_amount
FROM fact_transactions ft
    JOIN dim_account da ON ft.account_id = da.account_id
    JOIN dim_customer dc ON da.customer_id = dc.customer_id
GROUP BY dc.customer_segment,
    dc.age_group
ORDER BY dc.customer_segment,
    dc.age_group;
-- 9. Transaction trends: month-over-month total by transaction type
SELECT dd.year,
    dd.month,
    dd.month_name,
    ft.transaction_type,
    SUM(ft.amount) AS total_amount
FROM fact_transactions ft
    JOIN dim_date dd ON ft.date_id = dd.date_id
GROUP BY dd.year,
    dd.month,
    dd.month_name,
    ft.transaction_type
ORDER BY dd.year,
    dd.month,
    ft.transaction_type;
-- 10. Monthly revenue proxy: net of deposits minus withdrawals, by month
SELECT dd.year,
    dd.month_name,
    SUM(
        CASE
            WHEN ft.transaction_type = 'Deposit' THEN ft.amount
            ELSE 0
        END
    ) - SUM(
        CASE
            WHEN ft.transaction_type = 'Withdrawal' THEN ft.amount
            ELSE 0
        END
    ) AS net_flow
FROM fact_transactions ft
    JOIN dim_date dd ON ft.date_id = dd.date_id
GROUP BY dd.year,
    dd.month,
    dd.month_name
ORDER BY dd.year,
    dd.month;
-- 11. Loan status breakdown by branch (defaulted vs active vs closed)
SELECT db.branch_name,
    fl.status,
    COUNT(*) AS loan_count,
    SUM(fl.amount) AS total_amount
FROM fact_loans fl
    JOIN dim_branch db ON fl.branch_id = db.branch_id
GROUP BY db.branch_name,
    fl.status
ORDER BY db.branch_name,
    fl.status;
-- 12. Weekday vs weekend transaction activity (from Daily activity angle)
SELECT dd.is_weekend,
    ft.transaction_type,
    COUNT(*) AS txn_count,
    ROUND(AVG(ft.amount), 2) AS avg_amount
FROM fact_transactions ft
    JOIN dim_date dd ON ft.date_id = dd.date_id
GROUP BY dd.is_weekend,
    ft.transaction_type
ORDER BY dd.is_weekend,
    ft.transaction_type;