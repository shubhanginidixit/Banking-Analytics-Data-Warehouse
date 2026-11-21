-- ============================================================
-- Banking Analytics Data Warehouse - Schema
-- Galaxy schema: two fact tables (transactions, loans) sharing
-- conformed dimensions (date, branch, customer)
-- ============================================================

DROP TABLE IF EXISTS fact_loans CASCADE;
DROP TABLE IF EXISTS fact_transactions CASCADE;
DROP TABLE IF EXISTS dim_account CASCADE;
DROP TABLE IF EXISTS dim_loan_type CASCADE;
DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_branch CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;

-- ---------------- DIMENSIONS ----------------

CREATE TABLE dim_date (
    date_id       INT PRIMARY KEY,        -- YYYYMMDD
    full_date     DATE NOT NULL,
    day           INT NOT NULL,
    month         INT NOT NULL,
    month_name    VARCHAR(10) NOT NULL,
    quarter       INT NOT NULL,
    year          INT NOT NULL,
    day_of_week   VARCHAR(10) NOT NULL,
    is_weekend    BOOLEAN NOT NULL
);

CREATE TABLE dim_branch (
    branch_id     INT PRIMARY KEY,
    branch_name   VARCHAR(100) NOT NULL,
    city          VARCHAR(50) NOT NULL,
    region        VARCHAR(50) NOT NULL,
    branch_type   VARCHAR(20) NOT NULL      -- Urban / Semi-Urban / Rural
);

CREATE TABLE dim_customer (
    customer_id       INT PRIMARY KEY,
    first_name        VARCHAR(50) NOT NULL,
    last_name         VARCHAR(50) NOT NULL,
    gender            VARCHAR(10) NOT NULL,
    age               INT NOT NULL,
    age_group         VARCHAR(20) NOT NULL,  -- 18-25 / 26-35 / 36-50 / 51-65 / 65+
    city              VARCHAR(50) NOT NULL,
    home_branch_id    INT NOT NULL REFERENCES dim_branch(branch_id),
    customer_segment  VARCHAR(20) NOT NULL,  -- Regular / Premium / VIP
    account_open_date DATE NOT NULL
);

CREATE TABLE dim_account (
    account_id     INT PRIMARY KEY,
    customer_id    INT NOT NULL REFERENCES dim_customer(customer_id),
    account_type   VARCHAR(20) NOT NULL,     -- Savings / Current / Fixed Deposit
    open_date      DATE NOT NULL,
    status         VARCHAR(10) NOT NULL      -- Active / Closed
);

CREATE TABLE dim_loan_type (
    loan_type_id    INT PRIMARY KEY,
    loan_type_name  VARCHAR(30) NOT NULL,    -- Home / Personal / Auto / Education
    typical_rate_pct NUMERIC(4,2) NOT NULL
);

-- ---------------- FACTS ----------------

CREATE TABLE fact_transactions (
    transaction_id    BIGINT PRIMARY KEY,
    account_id        INT NOT NULL REFERENCES dim_account(account_id),
    date_id           INT NOT NULL REFERENCES dim_date(date_id),
    branch_id         INT NOT NULL REFERENCES dim_branch(branch_id),
    transaction_type  VARCHAR(20) NOT NULL,  -- Deposit / Withdrawal / Transfer
    amount            NUMERIC(12,2) NOT NULL
);

CREATE TABLE fact_loans (
    loan_id         INT PRIMARY KEY,
    customer_id     INT NOT NULL REFERENCES dim_customer(customer_id),
    branch_id       INT NOT NULL REFERENCES dim_branch(branch_id),
    loan_type_id    INT NOT NULL REFERENCES dim_loan_type(loan_type_id),
    date_id         INT NOT NULL REFERENCES dim_date(date_id),  -- disbursement date
    amount          NUMERIC(12,2) NOT NULL,
    interest_rate   NUMERIC(4,2) NOT NULL,
    tenure_months   INT NOT NULL,
    status          VARCHAR(15) NOT NULL     -- Active / Closed / Defaulted
);

-- Helpful indexes for OLAP-style aggregation
CREATE INDEX idx_txn_date   ON fact_transactions(date_id);
CREATE INDEX idx_txn_branch ON fact_transactions(branch_id);
CREATE INDEX idx_loan_date  ON fact_loans(date_id);
CREATE INDEX idx_loan_branch ON fact_loans(branch_id);
