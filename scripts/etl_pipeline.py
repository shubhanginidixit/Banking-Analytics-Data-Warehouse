import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB = ROOT / os.getenv("LOCAL_DB_PATH", "data/banking_dw.duckdb")
AS_OF_YEAR = 1999


def get_duckdb():
    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(LOCAL_DB))
    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    return con

def extract_data(data_dir: str) -> dict:
    """Load raw CSV/ASC files into pandas DataFrames."""
    files = ["client", "account", "disp", "district", "loan", "trans"]
    df = {}
    
    for f in files:
        path = f"{data_dir}/{f}.asc" if os.path.exists(f"{data_dir}/{f}.asc") else f"{data_dir}/{f}.csv"
        df[f] = pd.read_csv(path, sep=";", low_memory=False)

    if "A1" in df["district"].columns and "district_id" not in df["district"].columns:
        df["district"] = df["district"].rename(columns={"A1": "district_id"})
        
    return df

def transform_dimensions(df: dict) -> dict:
    """Transform raw data into dimension tables."""
    # --- DIMENSIONS ---
    dim_branch = pd.DataFrame({
        "branch_id": df["district"]["district_id"],
        "branch_name": df["district"]["A2"] + " Branch",
        "city": df["district"]["A2"],
        "region": df["district"]["A3"],
        "branch_type": np.where(df["district"]["A4"] > 100000, "Urban", "Rural")
    })

    # Vectorized birth number parsing
    bn = df["client"]["birth_number"].astype(str).str.zfill(6)
    month = bn.str[2:4].astype(int)
    age = AS_OF_YEAR - (bn.str[:2].astype(int) + 1900)
    district_names = df["district"].set_index("district_id")["A2"]

    gender = np.where(month > 50, "Female", "Male")
    dim_customer = pd.DataFrame({
        "customer_id": df["client"]["client_id"],
        "first_name": "Client",
        "last_name": df["client"]["client_id"].astype(str),
        "gender": gender,
        "gender_bin": (gender == "Female").astype(int),
        "age": age,
        "age_group": pd.cut(age, bins=[0, 25, 35, 50, 65, 100], labels=["18-25", "26-35", "36-50", "51-65", "65+"]),
        "city": df["client"]["district_id"].map(district_names),
        "home_branch_id": df["client"]["district_id"],
        "customer_segment": "Unsegmented"
    })

    df_acc_full = pd.merge(df["account"], df["disp"][df["disp"]["type"] == "OWNER"], on="account_id")
    account_type = df_acc_full["frequency"].map({"POPLATEK MESICNE": "Savings", "POPLATEK TYDNE": "Current", "POPLATEK PO OBRATU": "Fixed Deposit"})
    if account_type.isna().any():
        unknown = sorted(df_acc_full.loc[account_type.isna(), "frequency"].unique())
        raise ValueError(f"Unknown account frequency codes: {unknown}")
    dim_account = pd.DataFrame({
        "account_id": df_acc_full["account_id"],
        "customer_id": df_acc_full["client_id"],
        "account_type": account_type,
        "open_date": pd.to_datetime(df_acc_full["date"].astype(str), format='%y%m%d'),
        "status": "Active"
    })
    
    dim_customer["account_open_date"] = dim_customer["customer_id"].map(dim_account.groupby("customer_id")["open_date"].min())

    # PKDD'99 has no loan product or interest rate. Keep one explicit unknown member.
    dim_loan_type = pd.DataFrame([(1, "Unspecified", np.nan)], columns=["loan_type_id", "loan_type_name", "typical_rate_pct"])

    all_dates = pd.concat([
        pd.to_datetime(df["trans"]["date"].astype(str), format='%y%m%d'), 
        pd.to_datetime(df["loan"]["date"].astype(str), format='%y%m%d')
    ])
    date_range = pd.date_range(all_dates.min(), all_dates.max())
    dim_date = pd.DataFrame({
        "date_id": date_range.strftime("%Y%m%d").astype(int),
        "full_date": date_range,
        "day": date_range.day,
        "month": date_range.month,
        "month_name": date_range.strftime("%B"),
        "quarter": date_range.quarter,
        "year": date_range.year,
        "day_of_week": date_range.strftime("%A"),
        "is_weekend": date_range.weekday >= 5
    })
    
    return {
        "dim_branch": dim_branch,
        "dim_customer": dim_customer,
        "dim_account": dim_account,
        "dim_loan_type": dim_loan_type,
        "dim_date": dim_date
    }

def transform_facts(df: dict, dims: dict) -> dict:
    """Transform raw data into fact tables using dimension references."""
    dim_account = dims["dim_account"]
    account_customer = dim_account.set_index("account_id")["customer_id"]
    account_branch = df["account"].set_index("account_id")["district_id"]

    # Process Loans
    df_loan_date = pd.to_datetime(df["loan"]["date"].astype(str), format='%y%m%d')
    fact_loans = pd.DataFrame({
        "loan_id": df["loan"]["loan_id"],
        "account_id": df["loan"]["account_id"],
        "customer_id": df["loan"]["account_id"].map(account_customer),
        "branch_id": df["loan"]["account_id"].map(account_branch),
        "loan_type_id": 1,
        "date_id": df_loan_date.dt.strftime("%Y%m%d").astype(int),
        "amount": df["loan"]["amount"],
        "interest_rate": np.nan,
        "tenure_months": df["loan"]["duration"],
        "payments": df["loan"]["payments"],
        "status_code": df["loan"]["status"],
        "status": df["loan"]["status"].map({"A": "Closed", "B": "Defaulted", "C": "Active", "D": "Defaulted"})
    })
    missing = fact_loans[["customer_id", "branch_id"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} loans have no owner or branch mapping")

    # Process Transactions
    df_trans_date = pd.to_datetime(df["trans"]["date"].astype(str), format='%y%m%d')
    transaction_type = df["trans"]["type"].map({"PRIJEM": "Deposit", "VYDAJ": "Withdrawal", "VYBER": "Withdrawal"})
    if transaction_type.isna().any():
        unknown = sorted(df["trans"].loc[transaction_type.isna(), "type"].unique())
        raise ValueError(f"Unknown transaction type codes: {unknown}")
    fact_transactions = pd.DataFrame({
        "transaction_id": df["trans"]["trans_id"],
        "source_row": np.arange(len(df["trans"])),
        "account_id": df["trans"]["account_id"],
        "date_id": df_trans_date.dt.strftime("%Y%m%d").astype(int),
        "branch_id": df["trans"]["account_id"].map(account_branch),
        "transaction_type": transaction_type,
        "operation": df["trans"]["operation"],
        "amount": df["trans"]["amount"],
        "balance": df["trans"]["balance"]
    })
    
    return {
        "fact_loans": fact_loans,
        "fact_transactions": fact_transactions
    }

def write_tables(con, schema: str, frames: dict) -> None:
    for name, data in frames.items():
        con.register(f"frame_{name}", data)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{name} AS SELECT * FROM frame_{name}")
        con.unregister(f"frame_{name}")


def load_duckdb(dims: dict, facts: dict, con):
    """Write dimensions and facts to local DuckDB under the mart schema."""
    write_tables(con, "mart", {**dims, **facts})
    con.execute("CHECKPOINT")


def load_data(dims: dict, facts: dict):
    """Save CSV snapshots, then load the local DuckDB warehouse."""
    clean_dir = ROOT / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    for name, data in {**dims, **facts}.items():
        data.to_csv(clean_dir / f"{name}.csv", index=False)

    con = get_duckdb()
    load_duckdb(dims, facts, con)
    con.close()
    print(f"Loaded into DuckDB: {LOCAL_DB}")
    print("Done. Data loaded successfully.")


def main():
    print("Extracting Data...")
    raw_data = extract_data(str(ROOT / "data"))

    print("Transforming Dimensions...")
    dims = transform_dimensions(raw_data)

    print("Transforming Facts...")
    facts = transform_facts(raw_data, dims)

    load_data(dims, facts)

if __name__ == "__main__":
    main()
