import pytest
import pandas as pd
import numpy as np

# Adjust path to import from scripts
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.etl_pipeline import transform_dimensions, transform_facts

@pytest.fixture
def mock_raw_data():
    """Provides a minimal valid raw data dictionary to test transformations without reading disk."""
    return {
        "district": pd.DataFrame({
            "district_id": [1, 2], 
            "A2": ["Prague", "RuralTown"], 
            "A3": ["Prague", "Bohemia"], 
            "A4": [1200000, 50000]  # Population > 100k -> Urban
        }),
        "client": pd.DataFrame({
            "client_id": [101, 102], 
            # 706225: Year 70 (1970), Month 62 (Female)
            # 800101: Year 80 (1980), Month 01 (Male)
            "birth_number": [706225, 800101], 
            "district_id": [1, 2]
        }),
        "account": pd.DataFrame({
            "account_id": [1001, 1002], 
            "district_id": [1, 2], 
            "frequency": ["POPLATEK MESICNE", "POPLATEK TYDNE"], 
            "date": [930101, 940505]
        }),
        "disp": pd.DataFrame({
            "disp_id": [1, 2], 
            "client_id": [101, 102], 
            "account_id": [1001, 1002], 
            "type": ["OWNER", "OWNER"]
        }),
        "trans": pd.DataFrame({
            "trans_id": [1, 2], 
            "account_id": [1001, 1002], 
            "date": [930102, 940506], 
            "type": ["PRIJEM", "VYDAJ"], 
            "amount": [1000.0, 500.0]
        }),
        "loan": pd.DataFrame({
            "loan_id": [1], 
            "account_id": [1001], 
            "date": [940101], 
            "amount": [50000.0], 
            "duration": [12], 
            "status": ["A"] # A -> Closed
        })
    }

def test_transform_dimensions_branch(mock_raw_data):
    """Test that branches are correctly classified as Urban or Rural based on population."""
    dims = transform_dimensions(mock_raw_data)
    dim_branch = dims["dim_branch"]
    
    assert len(dim_branch) == 2
    urban_branch = dim_branch[dim_branch["branch_id"] == 1].iloc[0]
    rural_branch = dim_branch[dim_branch["branch_id"] == 2].iloc[0]
    
    assert urban_branch["branch_type"] == "Urban"
    assert rural_branch["branch_type"] == "Rural"

def test_transform_dimensions_customer_parsing(mock_raw_data):
    """Test the birth number parsing logic for age and gender."""
    dims = transform_dimensions(mock_raw_data)
    dim_customer = dims["dim_customer"]
    
    assert len(dim_customer) == 2
    female_client = dim_customer[dim_customer["customer_id"] == 101].iloc[0]
    male_client = dim_customer[dim_customer["customer_id"] == 102].iloc[0]
    
    # Client 101: Born 1970 -> 1999 - 1970 = 29 years old, Month 62 -> Female
    assert female_client["gender"] == "Female"
    assert female_client["age"] == 29
    assert female_client["age_group"] == "26-35"
    assert female_client["customer_segment"] == "Regular"
    
    # Client 102: Born 1980 -> 1999 - 1980 = 19 years old, Month 01 -> Male
    assert male_client["gender"] == "Male"
    assert male_client["age"] == 19
    assert male_client["age_group"] == "18-25"

def test_transform_facts_loans(mock_raw_data):
    """Test that loans correctly resolve dimension keys and map status codes."""
    dims = transform_dimensions(mock_raw_data)
    facts = transform_facts(mock_raw_data, dims)
    
    fact_loans = facts["fact_loans"]
    assert len(fact_loans) == 1
    
    loan = fact_loans.iloc[0]
    # Status 'A' maps to 'Closed'
    assert loan["status"] == "Closed"
    assert loan["customer_id"] == 101
    assert loan["branch_id"] == 1
    assert loan["interest_rate"] == 12.0

def test_transform_facts_transactions(mock_raw_data):
    """Test transaction type mapping."""
    dims = transform_dimensions(mock_raw_data)
    facts = transform_facts(mock_raw_data, dims)
    
    fact_transactions = facts["fact_transactions"]
    assert len(fact_transactions) == 2
    
    prijem_trans = fact_transactions[fact_transactions["transaction_id"] == 1].iloc[0]
    vydaj_trans = fact_transactions[fact_transactions["transaction_id"] == 2].iloc[0]
    
    assert prijem_trans["transaction_type"] == "Deposit"
    assert vydaj_trans["transaction_type"] == "Withdrawal"

def test_full_dataset_reconciliation():
    """
    PROVES ETL ACCURACY:
    Reads the actual raw CSV files and passes them through the transformation layer,
    asserting that exactly 0 rows are lost in the process for the core fact and dimension tables.
    This serves as programmatic proof for the Data Reconciliation table.
    """
    from scripts.etl_pipeline import extract_data, transform_dimensions, transform_facts
    import os
    
    # Only run if data directory exists (to avoid breaking CI without data)
    if not os.path.exists("data/trans.csv") and not os.path.exists("data/trans.asc"):
        pytest.skip("Raw data not found locally. Skipping full reconciliation test.")
        
    # 1. Extract
    raw_data = extract_data("data")
    
    # 2. Transform
    dims = transform_dimensions(raw_data)
    facts = transform_facts(raw_data, dims)
    
    # 3. Assert (Prove Accuracy / No Data Loss)
    assert len(facts["fact_transactions"]) == len(raw_data["trans"]), "Transaction row count mismatch!"
    assert len(facts["fact_loans"]) == len(raw_data["loan"]), "Loan row count mismatch!"
    assert len(dims["dim_account"]) == len(raw_data["account"]), "Account row count mismatch!"
    assert len(dims["dim_customer"]) == len(raw_data["client"]), "Customer row count mismatch!"
    
    print("\n--- ETL ACCURACY PROVEN ---")
    print(f"Transactions: {len(raw_data['trans'])} raw -> {len(facts['fact_transactions'])} transformed (100% Match)")
    print(f"Loans: {len(raw_data['loan'])} raw -> {len(facts['fact_loans'])} transformed (100% Match)")

if __name__ == "__main__":
    import pytest
    import sys
    # When you click "Run" in your IDE, this tells pytest to execute this file and print the output (-s)
    sys.exit(pytest.main(["-s", "-v", __file__]))
