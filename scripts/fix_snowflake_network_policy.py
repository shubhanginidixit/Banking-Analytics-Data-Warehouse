"""Attach a network policy so the PAT can log in, then verify the token.

Run: venv\\Scripts\\python.exe scripts\\fix_snowflake_network_policy.py
It prompts for a fresh 6-digit TOTP code (30-second window).
"""
from __future__ import annotations

import getpass
import os
import re
import sys
import urllib.request
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

POLICY = "DWDM_DASHBOARD_NP"

code = getpass.getpass("Snowflake TOTP code (6 digits): ").strip()
if not code.isdigit() or len(code) != 6:
    raise SystemExit("Enter a 6-digit TOTP code")

try:
    ip = urllib.request.urlopen("https://api.ipify.org", timeout=10).read().decode().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        raise ValueError(ip)
    print(f"Public IP: {ip}")
except Exception:
    ip = input("Could not detect public IP. Enter it manually: ").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        raise SystemExit("Invalid IP address")

user = os.environ["SNOWFLAKE_USER"]
if not re.fullmatch(r"[A-Za-z0-9_]+", user):
    raise SystemExit("SNOWFLAKE_USER contains unexpected characters")

print("Connecting with password + TOTP...", flush=True)
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=user,
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PBI_WH"),
    database=os.environ.get("SNOWFLAKE_DATABASE", "LOAN_DEFAULT_DW"),
    schema=os.environ.get("SNOWFLAKE_SCHEMA", "RAW"),
    authenticator="username_password_mfa",
    password=os.environ["SNOWFLAKE_PASSWORD"],
    passcode=code,
)
cur = conn.cursor()
try:
    cur.execute(
        f"CREATE OR REPLACE NETWORK POLICY {POLICY} ALLOWED_IP_LIST = ('{ip}/32')"
    )
    print(f"Created network policy {POLICY} for {ip}/32", flush=True)
    cur.execute(f"ALTER USER {user} SET NETWORK_POLICY = {POLICY}")
    print(f"Attached {POLICY} to user {user}", flush=True)
except snowflake.connector.errors.ProgrammingError as exc:
    print(f"DDL failed (need SECURITYADMIN/ACCOUNTADMIN privileges?): {exc}")
    raise SystemExit(1)
finally:
    cur.close()
    conn.close()

print("Verifying PAT...", flush=True)
sys.path.insert(0, str(ROOT / "app"))
from data_access import snowflake_query

df = snowflake_query(
    "SELECT COUNT(*) AS row_count FROM LOAN_DEFAULT_DW.MART.MART_LOAN_FEATURES"
)
print(f"PAT works. Feature rows: {int(df.iloc[0, 0])}")
