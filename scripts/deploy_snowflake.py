"""One-shot Snowflake deploy: DDL + upload + verify, then print row counts.

Prompts for the 6-digit TOTP at runtime, so the code only has to be valid once.
Run from anywhere with:

    venv\\Scripts\\python.exe scripts\\deploy_snowflake.py
"""
from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import snowflake.connector
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def connect() -> snowflake.connector.SnowflakeConnection:
    code = getpass.getpass("Snowflake TOTP code (6 digits): ").strip()
    if not code.isdigit() or len(code) != 6:
        raise SystemExit("Enter a 6-digit TOTP code")

    opts = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "role": os.environ["SNOWFLAKE_ROLE"],
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "PBI_WH"),
        "database": os.environ.get("SNOWFLAKE_DATABASE", "LOAN_DEFAULT_DW"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "RAW"),
    }
    print("Connecting...", flush=True)
    conn = snowflake.connector.connect(
        **opts,
        authenticator="username_password_mfa",
        password=os.environ["SNOWFLAKE_PASSWORD"],
        passcode=code,
    )
    print("Connected.", flush=True)
    return conn


def run_sql_file(cur, path: Path, skip_show: bool = False) -> int:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    ]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    for statement in statements:
        if skip_show and statement.upper().startswith("SHOW"):
            continue
        cur.execute(statement)
        if cur.description:
            columns = [column[0] for column in cur.description]
            rows = cur.fetchall()
            if rows:
                print(columns, flush=True)
                for row in rows:
                    print(row, flush=True)
    return len(statements)


def main() -> None:
    conn = connect()
    real_close = conn.close
    conn.close = lambda: None  # keep the authenticated session for all three steps
    cur = conn.cursor()
    try:
        print("Running DDL...", flush=True)
        count = run_sql_file(cur, ROOT / "sql" / "snowflake" / "ddl_snowflake.sql")
        print(f"DDL: {count} statements OK", flush=True)

        import upload_to_snowflake as uploader

        uploader.connect_snowflake = lambda: conn
        uploader.main()
        print("Upload OK", flush=True)

        print("Verifying...", flush=True)
        run_sql_file(cur, ROOT / "sql" / "snowflake" / "verify_load.sql", skip_show=True)
        print("Verify OK", flush=True)
    finally:
        real_close()


if __name__ == "__main__":
    main()
