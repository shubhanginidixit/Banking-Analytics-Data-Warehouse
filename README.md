# Banking Analytics Data Warehouse

Modern banking institutions generate massive amounts of transactional and demographic data daily. However, operational databases (OLTP) are optimized for fast inserts and updates, making them highly inefficient for complex analytical queries. 

The objective of this project is to **create a Banking Analytics Data Warehouse** by integrating customer, account, loan, and transaction information. The goal is to design a multidimensional dimensional model (Galaxy schema) that supports rapid Business Intelligence (BI) reporting, enabling management to perform advanced OLAP analysis on branch-wise deposits, loan distribution, customer segmentation, and temporal revenue trends.

---

## Dataset

This project uses the [PKDD'99 Financial Dataset](https://www.kaggle.com/datasets/siavashraz/1999-czech-financial-dataset) — real anonymized banking data from a Czech bank (1993–1998) containing 4,500 accounts, 5,369 clients, and ~1M transactions. Place the CSV files in the `data/` directory before running the ETL pipeline.

---

## Data Warehouse Architecture

The architecture follows a standard Kimball Bottom-Up methodology, extracting raw historical data (based on the real-world [PKDD'99 Financial Dataset](https://www.kaggle.com/datasets/siavashraz/1999-czech-financial-dataset)), transforming it via Python, and loading it into a PostgreSQL relational database structured for OLAP.

```mermaid
graph LR
    %% Premium styling with high-contrast font colors
    classDef source fill:#1e293b,stroke:#64748b,stroke-width:2px,color:#f8fafc
    classDef process fill:#0284c7,stroke:#bae6fd,stroke-width:2px,color:#ffffff
    classDef database fill:#3730a3,stroke:#c7d2fe,stroke-width:2px,color:#ffffff
    classDef bi fill:#d97706,stroke:#fde68a,stroke-width:2px,color:#ffffff

    A[Raw CSV Data]:::source -->|Python / Pandas| B(ETL Processing):::process
    B -->|SQLAlchemy| C[(PostgreSQL Data Warehouse)]:::database
    C -->|DirectQuery / Import| D[Power BI Dashboard]:::bi
```

---

## Dimensional Modeling (Galaxy Schema)

To accommodate both transaction metrics and loan metrics, a **Galaxy Schema** (a schema with multiple Fact tables sharing conformed dimensions) was implemented.

### Dimensions (Context)
- `dim_date`: Conformed dimension for time-series analysis (Year, Quarter, Month, Day, Weekend flags).
- `dim_branch`: Geographic and organizational hierarchy (Branch Name, City, Region).
- `dim_customer`: Demographic data (Age, Gender, Age Group, Customer Segment).
- `dim_account`: Account metadata (Type, Open Date, Status).
- `dim_loan_type`: Loan categorizations.

### Facts (Measurements)
- `fact_transactions`: Contains over 1 million rows tracking every deposit and withdrawal event.
- `fact_loans`: Tracks disbursed loans, amounts, interest rates, and status (Active/Closed/Defaulted).

```mermaid
classDiagram
    direction BT
    
    %% Fact Tables (Measurements)
    class fact_transactions {
        bigint transaction_id [PK]
        numeric amount
        varchar transaction_type
    }
    
    class fact_loans {
        int loan_id [PK]
        numeric amount
        numeric interest_rate
        varchar status
    }
    
    %% Dimension Tables (Context)
    class dim_date {
        int date_id [PK]
        date full_date
        int year
        int quarter
        varchar month_name
    }
    
    class dim_branch {
        int branch_id [PK]
        varchar branch_name
        varchar city
        varchar region
    }
    
    class dim_customer {
        int customer_id [PK]
        varchar first_name
        varchar last_name
        varchar age_group
        varchar customer_segment
    }
    
    class dim_account {
        int account_id [PK]
        varchar account_type
    }
    
    class dim_loan_type {
        int loan_type_id [PK]
        varchar loan_type_name
    }

    %% Relationships (Arranged to minimize crossing lines)
    dim_date "1" <-- "*" fact_transactions : date_id
    dim_account "1" <-- "*" fact_transactions : account_id
    dim_branch "1" <-- "*" fact_transactions : branch_id
    
    dim_loan_type "1" <-- "*" fact_loans : loan_type_id
    dim_customer "1" <-- "*" fact_loans : customer_id
    dim_branch "1" <-- "*" fact_loans : branch_id
    dim_date "1" <-- "*" fact_loans : date_id
    
    dim_customer "1" <-- "*" dim_account : customer_id

    %% Aesthetic Premium Styling
    %% Facts: Deep Violet background with vibrant Purple border
    style fact_transactions fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#ddd6fe
    style fact_loans fill:#2e1065,stroke:#8b5cf6,stroke-width:2px,color:#ddd6fe
    
    %% Dimensions: Deep Cyan background with vibrant Cyan border
    style dim_date fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#cffafe
    style dim_branch fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#cffafe
    style dim_customer fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#cffafe
    style dim_account fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#cffafe
    style dim_loan_type fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#cffafe
```

---

## Data Cube Design & OLAP Queries

A Data Cube was designed to perform multidimensional analysis across three core axes. By navigating this cube, we can instantly slice and dice transaction amounts.

![Data Cube Design](docs/data_cube_v2.svg)

All requirements for OLAP operations have been successfully fulfilled in `sql/olap_queries.sql`, which contains **12 distinct analytical queries**:
- **Roll-up:** Aggregates deposits dynamically from Month -> Quarter -> Year.
- **Drill-down:** Steps down from Region -> City -> Branch.
- **Slice & Dice:** Isolates specific quarters, regions, and loan types.
- **Data Cube:** Leverages the `GROUP BY CUBE` SQL function to generate multidimensional sub-totals and grand totals for transactions across all combinations of branch regions and transaction types.

The final deliverable is an interactive **Power BI dashboard**, successfully connected to the PostgreSQL instance to render visual analytics on the PKDD dataset.

---

## Setup & Execution Instructions

### Prerequisites
- **VS Code** + extensions: `Python`, `Docker`, and `SQLTools`
- **Docker Desktop** (runs Postgres in a container)
- **Python 3.10+**
- **Power BI Desktop** (for the final dashboard)

### 1. Start the Database
Spin up the PostgreSQL data warehouse using Docker:
```bash
docker compose up -d
```

### 2. Create the Schema
Connect to the database (`localhost:5432`, user: `dw_user`, pass: `dw_pass`) and run the schema file:
```bash
docker exec -i banking_dw_postgres psql -U dw_user -d banking_dw < sql/schema.sql
```

### 3. Run the ETL Pipeline
Set up your Python environment and load the real dataset into the data warehouse:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/etl_pipeline.py
```

### 4. Build the Dashboard
- Open Power BI Desktop → **Get Data** → **PostgreSQL database**.
- Import `fact_transactions`, `fact_loans`, and all `dim_*` tables.
- Build visuals for Deposits Trend, Branch Performance, Loan Portfolio, and Customer Segments.
