-- Banking data warehouse on Snowflake.
-- Raw extracts live in RAW; modelled tables and reporting views live in MART.
CREATE WAREHOUSE IF NOT EXISTS PBI_WH WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 600 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
CREATE DATABASE IF NOT EXISTS LOAN_DEFAULT_DW;
CREATE SCHEMA IF NOT EXISTS LOAN_DEFAULT_DW.RAW;
CREATE SCHEMA IF NOT EXISTS LOAN_DEFAULT_DW.MART;

CREATE ROLE IF NOT EXISTS PBI_READER;
GRANT USAGE ON WAREHOUSE PBI_WH TO ROLE PBI_READER;
GRANT USAGE ON DATABASE LOAN_DEFAULT_DW TO ROLE PBI_READER;
GRANT USAGE ON SCHEMA LOAN_DEFAULT_DW.RAW TO ROLE PBI_READER;
GRANT USAGE ON SCHEMA LOAN_DEFAULT_DW.MART TO ROLE PBI_READER;

USE SCHEMA LOAN_DEFAULT_DW.RAW;

CREATE OR REPLACE TABLE RAW_LOAN (
  loan_id NUMBER, account_id NUMBER, loan_date NUMBER,
  amount NUMBER, duration NUMBER, payments NUMBER, status VARCHAR);
CREATE OR REPLACE TABLE RAW_TRANS (
  trans_id NUMBER, account_id NUMBER, trans_date NUMBER, type VARCHAR,
  operation VARCHAR, amount NUMBER, balance NUMBER,
  k_symbol VARCHAR, bank VARCHAR, account VARCHAR);
CREATE OR REPLACE TABLE RAW_CLIENT (
  client_id NUMBER, birth_number VARCHAR, district_id NUMBER);
CREATE OR REPLACE TABLE RAW_ACCOUNT (
  account_id NUMBER, district_id NUMBER, frequency VARCHAR, open_date NUMBER);
CREATE OR REPLACE TABLE RAW_DISP (
  disp_id NUMBER, client_id NUMBER, account_id NUMBER, type VARCHAR);

USE SCHEMA LOAN_DEFAULT_DW.MART;

CREATE OR REPLACE TABLE MART_LOAN_FEATURES (
  loan_id NUMBER, account_id NUMBER, loan_date NUMBER, amount NUMBER,
  duration NUMBER, payments NUMBER, status VARCHAR, parsed_date DATE,
  target NUMBER, client_id NUMBER, n_txn NUMBER, n_deposit NUMBER,
  n_withdrawal NUMBER, sum_deposit FLOAT, sum_withdrawal FLOAT,
  avg_amount FLOAT, std_amount FLOAT, avg_balance FLOAT, min_balance FLOAT,
  max_balance FLOAT, balance_at_loan FLOAT, months_active FLOAT,
  txn_per_month FLOAT, gender VARCHAR, age NUMBER, district_id NUMBER,
  gender_bin NUMBER);
CREATE OR REPLACE TABLE MART_MODEL_SCORES (
  loan_id NUMBER, account_id NUMBER, prob_default FLOAT, target NUMBER);
CREATE OR REPLACE TABLE MART_LOAN_SEGMENTS (loan_id NUMBER, segment NUMBER);
CREATE OR REPLACE TABLE MART_LOAN_RISK (
  loan_id NUMBER, account_id NUMBER, amount NUMBER, duration NUMBER,
  payments NUMBER, status VARCHAR, parsed_date DATE, target NUMBER,
  client_id NUMBER, n_txn NUMBER, n_deposit NUMBER, n_withdrawal NUMBER,
  sum_deposit FLOAT, sum_withdrawal FLOAT, avg_amount FLOAT, std_amount FLOAT,
  avg_balance FLOAT, min_balance FLOAT, max_balance FLOAT, balance_at_loan FLOAT,
  months_active FLOAT, txn_per_month FLOAT, gender VARCHAR, age NUMBER,
  district_id NUMBER, gender_bin NUMBER,
  prob_default FLOAT, segment NUMBER);
CREATE OR REPLACE TABLE MART_CLUSTER_LABELS (
  loan_id NUMBER, kmeans NUMBER, kmedoids NUMBER, agnes NUMBER, dbscan NUMBER);
CREATE OR REPLACE TABLE MART_CLUSTERING_COMPARISON (
  method VARCHAR, parameter VARCHAR, n_clusters NUMBER, silhouette FLOAT,
  davies_bouldin FLOAT, calinski_harabasz FLOAT, noise_share FLOAT,
  coverage FLOAT, min_cluster_share FLOAT, stability_ari_mean FLOAT,
  stability_ari_std FLOAT, role VARCHAR, recommended BOOLEAN);

CREATE OR REPLACE VIEW LOAN_DEFAULT_DW.MART.V_LOAN_PERF AS
SELECT r.*, s.prob_default, g.segment,
       CASE WHEN r.amount >= 116900 AND r.balance_at_loan >= 38500 THEN 'high-amt high-bal'
            WHEN r.amount >= 116900 THEN 'high-amt low-bal'
            WHEN r.balance_at_loan >= 38500 THEN 'low-amt high-bal'
            ELSE 'low-amt low-bal' END AS risk_quadrant,
       CASE WHEN r.balance_at_loan < 20000 THEN 1 ELSE 0 END AS low_balance_flag
FROM LOAN_DEFAULT_DW.MART.MART_LOAN_FEATURES r
JOIN LOAN_DEFAULT_DW.MART.MART_MODEL_SCORES s USING (loan_id)
JOIN LOAN_DEFAULT_DW.MART.MART_LOAN_SEGMENTS g USING (loan_id);

CREATE OR REPLACE VIEW LOAN_DEFAULT_DW.MART.V_TRANS_MONTHLY AS
SELECT t.account_id,
       DATE_TRUNC('month', TO_DATE(t.trans_date, 'YYMMDD')) AS ym,
       t.type AS txn_type, COUNT(*) AS n, SUM(t.amount) AS total_amount
FROM LOAN_DEFAULT_DW.RAW.RAW_TRANS t
WHERE EXISTS (
    SELECT 1
    FROM LOAN_DEFAULT_DW.RAW.RAW_LOAN l
    WHERE l.account_id = t.account_id
      AND TO_DATE(t.trans_date, 'YYMMDD') < TO_DATE(l.loan_date, 'YYMMDD')
)
GROUP BY 1, 2, 3;

GRANT SELECT ON ALL TABLES IN SCHEMA LOAN_DEFAULT_DW.MART TO ROLE PBI_READER;
GRANT SELECT ON FUTURE TABLES IN SCHEMA LOAN_DEFAULT_DW.MART TO ROLE PBI_READER;
GRANT SELECT ON ALL VIEWS IN SCHEMA LOAN_DEFAULT_DW.MART TO ROLE PBI_READER;
GRANT SELECT ON FUTURE VIEWS IN SCHEMA LOAN_DEFAULT_DW.MART TO ROLE PBI_READER;

-- If the old single-schema DDL was already run, these stale copies can be removed:
-- DROP TABLE IF EXISTS LOAN_DEFAULT_DW.RAW.MART_LOAN_FEATURES;
-- DROP TABLE IF EXISTS LOAN_DEFAULT_DW.RAW.MART_MODEL_SCORES;
-- DROP TABLE IF EXISTS LOAN_DEFAULT_DW.RAW.MART_LOAN_SEGMENTS;
-- DROP TABLE IF EXISTS LOAN_DEFAULT_DW.RAW.MART_LOAN_RISK;

SELECT (SELECT COUNT(*) FROM LOAN_DEFAULT_DW.MART.MART_LOAN_FEATURES) AS loan_features,
       (SELECT COUNT(*) FROM LOAN_DEFAULT_DW.RAW.RAW_TRANS) AS transactions,
       (SELECT AVG(target) FROM LOAN_DEFAULT_DW.MART.MART_LOAN_FEATURES) AS default_rate;
