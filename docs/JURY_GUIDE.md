# What we did and how — Jury guide

This file explains, in plain words, what the project does and how we built it,
one section for each item in the end-semester jury PDF
(`END-SEMESTER  DWDM JURY GUIDELINES.pdf`). The full technical write-up is in
[`final/Final_Report.md`](final/Final_Report.md); this is the short, plain-language
version to help explain the work in the viva.

The project answers one business question: **which loan applications should a bank
send to manual review because they are likely to default?** It uses real Czech
bank data from PKDD'99 — 682 loans, 76 of them defaults, and over a million
transactions. The golden rule everywhere is that we only use information that
existed *before* the loan was given.

A quick note on the PDF: it shows 37 out of 50 marks in the visible tables, and
page 6 cuts off part-way through the metrics list, so 13 marks are not defined in
the copy we have. We built to every requirement that is actually visible.

---

## A. Data Warehouse Design (Dashboard 1)

**What we did.** We designed a Kimball-style *galaxy* schema: two fact tables
that share several dimension tables.

- Facts: `fact_loans` (one row per loan) and `fact_transactions` (one row per
  account transaction).
- Dimensions: date, branch, customer, account, and loan type.
- Each fact plus the shared dimensions forms a star; together they form a galaxy,
  so loans and transactions can be compared on the same date, account, and branch.

**How we did it.** `scripts/etl_pipeline.py` reads the raw CSV files and builds all
tables in DuckDB. The "Data model card" page in the app draws the schema with
Graphviz and lists row counts, join keys, and a column dictionary. Validation in
`scripts/build_local.py` checks the fact counts and that every fact row links to a
dimension.

---

## B. Snowflake Deployment (Dashboard 2)

**What we did.** We deployed the same warehouse into Snowflake and the dashboard
now reads live from it.

- A warehouse (`PBI_WH`), database (`LOAN_DEFAULT_DW`), and two schemas: `RAW`
  (the five source extracts) and `MART` (the modelled tables and views).
- All 11 objects loaded and checked: 5 RAW tables, 6 MART tables, plus the views
  `V_LOAN_PERF` and `V_TRANS_MONTHLY`.
- Row counts matched exactly: e.g. 682 loans, 1,056,320 transactions, and 682 rows
  in each modelled table.

**How we did it.** `sql/snowflake/ddl_snowflake.sql` creates the objects,
`scripts/upload_to_snowflake.py` loads the CSVs (it clears each table first, so
re-running it never doubles the data), and `sql/snowflake/verify_load.sql` prints
the counts and sample rows. `scripts/deploy_snowflake.py` runs all three in one go.
Login uses a Programmatic Access Token (PAT) plus a network policy, so the app reads
Snowflake without asking for a code every time. The "Snowflake" page in the app
shows the schema split, live counts, and the verification SQL.

---

## C. Supervised Learning Dataset (Dashboard 3)

**What we did.** We built a dataset for predicting loan default.

- 682 loans, 27 columns in the feature table.
- Target (the answer column): whether the loan defaulted — 76 defaults, an 11.1%
  base rate.
- 18 input features describing the loan, the applicant, and their pre-loan
  behaviour.

**How we did it.** `sql/local/03_feature_mart.sql` builds `analytics.loan_features`.
The key step is the leakage rule: for each loan we only use transactions with a date
strictly *before* the loan date. `scripts/build_local.py` also checks that zero
feature rows contain any post-loan transaction. The "Data & leakage" page shows the
row counts, per-loan used/excluded transaction bars, and the quality checks.

---

## D. Pattern Matching — Apriori and FP-Growth

**What we did.** We mined the data for rules like "accounts with a negative balance
tend to default", using both Apriori and FP-Growth.

- Both algorithms found the same 41 frequent itemsets.
- Two rules point at default: `negative_balance -> default`
  (confidence 1.00, lift 8.97) and `type_VYBER -> default` (confidence 0.13,
  lift 1.17).

**How we did it.** `tools/pattern_mining.py` turns each account's pre-loan
transactions into a basket of flags (deposit, withdrawal, high amount, negative
balance, default), then runs `apriori` and `fpgrowth` from mlxtend with a minimum
support of 0.03. Rules are reported with support, confidence, and lift. The
"Patterns" page shows the rules and itemsets.

---

## E. Comparing Classifiers and Validating Them

**What we did.** We trained one main classifier and compared it against several
others to see which is best.

- Six algorithms: Decision Tree, Gaussian Naive Bayes, Random Forest, SVM, KNN,
  and Gradient Boosting — each with and without SMOTE and with default or tuned
  settings, giving 24 setups.
- Validation: a 75/25 stratified hold-out, tuning with 5-fold cross-validation on
  the training set only, and a repeated 5-fold × 10 repeat cross-validation for a
  fair comparison.
- Result: tuned Gradient Boosting (F1 0.571 ± 0.120, ROC-AUC 0.864) and Random
  Forest are statistically tied; we treat them as a leading pair.

**How we did it.** `notebooks/04_modeling.ipynb` does the training and produces the
comparison tables (`artifacts/metrics/`) and ROC/PR/confusion-matrix figures. The
"Models" page shows the ranking, the mean ± standard deviation, and the tuning
results.

---

## F. Classification Algorithm Implementation

**What we did.** We picked the tuned Gradient Boosting model as the final
classifier and saved it so it can score a new application.

- Preprocessing (scaling, encoding) is part of the saved pipeline, so scoring uses
  exactly the same steps as training.
- We also trained two challengers: an MLP neural network and a rule-based classifier
  (CBA). Neither beat the main model.
- The "Simulator" page lets you enter one application and see its default score and
  the main drivers.

**How we did it.** `scripts/build_advanced_models.py` trains and saves the models
to `artifacts/models/`. `app/views/simulator.py` loads the saved pipeline and
scores a single application at review time. No model is retrained when the app
starts.

---

## G. Unsupervised Learning Dataset

**What we did.** For clustering (grouping) we used a *different* view of the same
682 loans — one with no answer column at all.

- 13 behavioural features built only from transactions before the loan: number of
  transactions, deposits and withdrawals (counts and totals), average and spread of
  amounts, average/minimum/maximum/balance-at-loan, months active, and transactions
  per month.
- We left out loan amount, duration, payments, age, and gender, because those
  describe the product or the person, not how the account behaved.

**How we did it.** `scripts/build_advanced_models.py` selects the 13 features and
scales them with `StandardScaler`. The default label is **never** used to fit or
choose the groups; we only join it back afterwards to describe the groups. A
Hopkins statistic of about 0.85 confirms the data has real group structure. The
"Segments" page shows a dataset card explaining all of this.

---

## H. Clustering Algorithm Comparison (minimum four algorithms)

**What we did.** We grouped the loans with four different algorithms and compared
them side by side:

- **K-Means** and **K-Medoids** (partitional), **AGNES Ward** (hierarchical), and
  **DBSCAN** (density-based, used to flag outliers).
- We recorded the settings, cluster sizes, and three internal scores: silhouette
  (higher is better), Davies-Bouldin (lower is better), and Calinski-Harabasz
  (higher is better), plus stability across repeated subsamples.
- Verdict: K-Means is the most suitable primary method — its groups are clear and
  stable, it covers every loan, and it can score new applications. K-Medoids is
  close but unstable; AGNES is less stable; DBSCAN is best used as an outlier flag
  (it marked 129 unusual accounts).

**How we did it.** `scripts/build_advanced_models.py` runs all four algorithms on
the same 13 scaled features, computes the metrics and stability, and writes
`clustering_comparison.csv`, `clustering_tuning.csv`, and `clustering_stability.csv`
in `artifacts/metrics/`. PCA maps and metric charts are saved under
`artifacts/figures/`. The "Segments" page shows the full table, charts, each
algorithm's strengths and limitations, and the verdict. We also tested GMM, but
left it out of the main table because its two selection scores disagreed; the
PDF only requires a minimum of four algorithms.

---

## I. Clustering Algorithm Implementation

**What we did.** We implemented and explained the chosen method in detail and gave
the groups a practical meaning.

- K-Means with k=2: a **low-activity** group (421 loans, 12.35% default) and a
  **high-activity** group (261 loans, 9.20% default).
- DBSCAN's noise group (129 loans, 23.26% default) is a useful list for manual
  review.
- Practical action: focus early engagement and checks on quiet accounts; keep
  standard monitoring for busy accounts; review the flagged outliers.

**How we did it.** `scripts/build_advanced_models.py` holds the implementation
(`CLUSTER_FEATURES`, `fit_kmedoids`, `select_partitional`, `select_dbscan`,
`run_clusters`). `notebooks/05_segmentation.ipynb` shows how k was chosen by
silhouette and the elbow method. Outputs are in `artifacts/segments/`
(`cluster_labels.csv`, `cluster_profiles.csv`, `loan_segments.csv`) and the
interpretation is on the "Segments" page.

---

## J. Performance Metrics

**What we did.** We reported the standard classification metrics and internal
clustering scores.

- Classification: accuracy, precision, recall, F1, and confusion matrix, plus
  average precision, ROC-AUC, and precision-recall curves. These are in
  `artifacts/metrics/` and on the "Models" page.
- Clustering: silhouette, Davies-Bouldin, Calinski-Harabasz, and stability (Adjusted
  Rand Index), on the "Segments" page.

**How we did it.** Classification metrics come from the saved out-of-fold scores
and `notebooks/04_modeling.ipynb`. Clustering metrics are computed in
`scripts/build_advanced_models.py` and saved to the `clustering_*.csv` files. Note
that clustering metrics do not use the default label — accuracy, precision, recall,
and F1 belong only to the supervised model. The PDF's visible metrics list covers
classification; we added clustering scores because they are needed to justify the
clustering comparison in H and I.

---

## Where the code lives (quick reference)

| Part | Main files |
|---|---|
| ETL and warehouse | `scripts/etl_pipeline.py`, `scripts/build_local.py` |
| Feature mart | `sql/local/03_feature_mart.sql` |
| Snowflake | `sql/snowflake/`, `scripts/upload_to_snowflake.py`, `scripts/deploy_snowflake.py`, `scripts/fix_snowflake_network_policy.py` |
| Patterns | `tools/pattern_mining.py` |
| Models and clustering | `notebooks/04_modeling.ipynb`, `scripts/build_advanced_models.py` |
| Clustering notebooks | `notebooks/05_segmentation.ipynb`, `notebooks/06_clustering_comparison.ipynb` |
| Dashboards | `app/streamlit_app.py`, `app/views/`, `app/components/common.py` |
| Tests | `tests/test_local_db.py` |
| Full report | `docs/final/Final_Report.md` |

To rebuild everything locally: `venv\Scripts\python.exe scripts\build_local.py`.
To check it: `venv\Scripts\python.exe -m pytest -q` (12 tests).
