-- Local DuckDB feature mart.
-- One row per loan, using only transactions strictly before the loan date.
CREATE OR REPLACE TABLE analytics.loan_features AS
WITH loans AS (
    SELECT
        l.loan_id,
        l.account_id,
        l.customer_id AS client_id,
        l.branch_id AS district_id,
        l.date_id AS loan_date_id,
        strptime(CAST(l.date_id AS VARCHAR), '%Y%m%d')::DATE AS loan_date,
        l.amount,
        l.tenure_months AS duration,
        l.payments,
        l.status,
        l.status_code,
        CASE WHEN l.status_code IN ('B', 'D') THEN 1 ELSE 0 END AS target
    FROM mart.fact_loans l
),
pre AS (
    SELECT
        l.loan_id,
        t.transaction_id,
        t.source_row,
        t.date_id,
        t.transaction_type,
        t.amount,
        t.balance
    FROM loans l
    JOIN mart.fact_transactions t
      ON t.account_id = l.account_id
     AND t.date_id < l.loan_date_id
),
agg AS (
    SELECT
        loan_id,
        COUNT(*) AS n_txn,
        COUNT(*) FILTER (WHERE transaction_type = 'Deposit') AS n_deposit,
        COUNT(*) FILTER (WHERE transaction_type = 'Withdrawal') AS n_withdrawal,
        COALESCE(SUM(amount) FILTER (WHERE transaction_type = 'Deposit'), 0) AS sum_deposit,
        COALESCE(SUM(amount) FILTER (WHERE transaction_type = 'Withdrawal'), 0) AS sum_withdrawal,
        AVG(amount) AS avg_amount,
        STDDEV_POP(amount) AS std_amount,
        AVG(balance) AS avg_balance,
        MIN(balance) AS min_balance,
        MAX(balance) AS max_balance,
        MIN(date_id) AS first_txn_date_id
    FROM pre
    GROUP BY loan_id
),
last_balance AS (
    SELECT loan_id, balance AS balance_at_loan
    FROM pre
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY loan_id
        ORDER BY date_id DESC, source_row DESC
    ) = 1
),
first_date AS (
    SELECT
        loan_id,
        strptime(CAST(first_txn_date_id AS VARCHAR), '%Y%m%d')::DATE AS first_txn_date
    FROM agg
)
SELECT
    l.loan_id,
    l.account_id,
    l.loan_date_id,
    strftime(l.loan_date, '%y%m%d') AS date,
    l.amount,
    l.duration,
    l.payments,
    l.status,
    l.loan_date AS parsed_date,
    l.target,
    l.client_id,
    COALESCE(a.n_txn, 0) AS n_txn,
    COALESCE(a.n_deposit, 0) AS n_deposit,
    COALESCE(a.n_withdrawal, 0) AS n_withdrawal,
    COALESCE(a.sum_deposit, 0) AS sum_deposit,
    COALESCE(a.sum_withdrawal, 0) AS sum_withdrawal,
    a.avg_amount,
    a.std_amount,
    a.avg_balance,
    a.min_balance,
    a.max_balance,
    b.balance_at_loan,
    COALESCE(GREATEST(date_diff('day', f.first_txn_date, l.loan_date) / 30.44, 1 / 30.44), 0) AS months_active,
    COALESCE(a.n_txn / GREATEST(date_diff('day', f.first_txn_date, l.loan_date) / 30.44, 1 / 30.44), 0) AS txn_per_month,
    c.gender,
    c.age,
    c.home_branch_id AS district_id,
    c.gender_bin
FROM loans l
LEFT JOIN agg a ON a.loan_id = l.loan_id
LEFT JOIN last_balance b ON b.loan_id = l.loan_id
LEFT JOIN first_date f ON f.loan_id = l.loan_id
JOIN mart.dim_customer c ON c.customer_id = l.client_id
ORDER BY l.loan_id;
