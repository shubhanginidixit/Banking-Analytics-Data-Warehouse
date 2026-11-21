import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.sql import text

load_dotenv()

def get_engine():
    conn_str = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(conn_str)

def extract_data(data_dir: str) -> dict:
    """Load raw CSV/ASC files into pandas DataFrames."""
    files = ["client", "account", "disp", "district", "loan", "trans"]
    df = {}
    
    for f in files:
        path = f"{data_dir}/{f}.asc" if os.path.exists(f"{data_dir}/{f}.asc") else f"{data_dir}/{f}.csv"
        try:
            df[f] = pd.read_csv(path, sep=";", low_memory=False)
        except Exception as e:
            print(f"Error loading {f} at {path}: {e}")
            raise e
            
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
    age = 1999 - (bn.str[:2].astype(int) + 1900)
    
    dim_customer = pd.DataFrame({
        "customer_id": df["client"]["client_id"],
        "first_name": "Client", 
        "last_name": df["client"]["client_id"].astype(str),
        "gender": np.where(month > 50, "Female", "Male"),
        "age": age,
        "age_group": pd.cut(age, bins=[0, 25, 35, 50, 65, 100], labels=["18-25", "26-35", "36-50", "51-65", "65+"]),
        "city": df["client"]["district_id"].map(dict(zip(df["district"]["district_id"], df["district"]["A2"]))),
        "home_branch_id": df["client"]["district_id"],
        "customer_segment": np.where(age > 50, "Premium", "Regular")
    })

    df_acc_full = pd.merge(df["account"], df["disp"][df["disp"]["type"] == "OWNER"], on="account_id")
    dim_account = pd.DataFrame({
        "account_id": df_acc_full["account_id"],
        "customer_id": df_acc_full["client_id"],
        "account_type": df_acc_full["frequency"].map({"POPLATEK MESICNE": "Savings", "POPLATEK TYDNE": "Current", "POPLATEK PO OBRATU": "Fixed Deposit"}).fillna("Savings"),
        "open_date": pd.to_datetime(df_acc_full["date"].astype(str), format='%y%m%d'),
        "status": "Active"
    })
    
    dim_customer["account_open_date"] = dim_customer["customer_id"].map(dim_account.groupby("customer_id")["open_date"].min()).fillna(pd.to_datetime("1993-01-01"))

    dim_loan_type = pd.DataFrame([(1, "Home", 8.5), (2, "Personal", 12.0), (3, "Auto", 9.5), (4, "Education", 7.5)], columns=["loan_type_id", "loan_type_name", "typical_rate_pct"])

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
    
    # Process Loans
    df_loan_date = pd.to_datetime(df["loan"]["date"].astype(str), format='%y%m%d')
    fact_loans = pd.DataFrame({
        "loan_id": df["loan"]["loan_id"],
        "customer_id": df["loan"]["account_id"].map(dict(zip(dim_account["account_id"], dim_account["customer_id"]))),
        "branch_id": df["loan"]["account_id"].map(dict(zip(df["account"]["account_id"], df["account"]["district_id"]))),
        "loan_type_id": 2, 
        "date_id": df_loan_date.dt.strftime("%Y%m%d").astype(int),
        "amount": df["loan"]["amount"],
        "interest_rate": 12.0,
        "tenure_months": df["loan"]["duration"],
        "status": df["loan"]["status"].map({"A": "Closed", "B": "Defaulted", "C": "Active", "D": "Defaulted"})
    }).dropna()

    # Process Transactions
    df_trans_date = pd.to_datetime(df["trans"]["date"].astype(str), format='%y%m%d')
    fact_transactions = pd.DataFrame({
        "transaction_id": df["trans"]["trans_id"],
        "account_id": df["trans"]["account_id"],
        "date_id": df_trans_date.dt.strftime("%Y%m%d").astype(int),
        "branch_id": df["trans"]["account_id"].map(dict(zip(df["account"]["account_id"], df["account"]["district_id"]))),
        "transaction_type": df["trans"]["type"].map({"PRIJEM": "Deposit", "VYDAJ": "Withdrawal", "VYBER": "Withdrawal"}).fillna("Transfer"),
        "amount": df["trans"]["amount"]
    })
    
    return {
        "fact_loans": fact_loans,
        "fact_transactions": fact_transactions
    }

def load_data(dims: dict, facts: dict, engine):
    """Load transformed data into the data warehouse."""
    os.makedirs("data/clean", exist_ok=True)
    
    print("Saving clean data to local files and loading into Postgres...")
    for name, data in dims.items():
        data.to_csv(f"data/clean/{name}.csv", index=False)
    for name, data in facts.items():
        data.to_csv(f"data/clean/{name}.csv", index=False)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fact_transactions, fact_loans, dim_account, dim_customer, dim_branch, dim_date, dim_loan_type CASCADE;"))
    
    # Load dimensions
    dims["dim_date"].to_sql("dim_date", engine, if_exists="append", index=False)
    dims["dim_branch"].to_sql("dim_branch", engine, if_exists="append", index=False)
    dims["dim_customer"].to_sql("dim_customer", engine, if_exists="append", index=False)
    dims["dim_account"].to_sql("dim_account", engine, if_exists="append", index=False)
    dims["dim_loan_type"].to_sql("dim_loan_type", engine, if_exists="append", index=False)
    
    # Load facts
    facts["fact_loans"].to_sql("fact_loans", engine, if_exists="append", index=False)
    facts["fact_transactions"].to_sql("fact_transactions", engine, if_exists="append", index=False, chunksize=50000)
    
    print("Done. Data loaded successfully.")

def main():
    data_dir = "data"
    print("Extracting Data...")
    raw_data = extract_data(data_dir)
    
    print("Transforming Dimensions...")
    dims = transform_dimensions(raw_data)
    
    print("Transforming Facts...")
    facts = transform_facts(raw_data, dims)
    
    engine = get_engine()
    load_data(dims, facts, engine)

if __name__ == "__main__":
    main()
