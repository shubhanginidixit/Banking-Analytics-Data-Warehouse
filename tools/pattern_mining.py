"""Mine pre-loan account behavior with Apriori and FP-Growth."""

from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "artifacts" / "rules"

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.10
MAX_LEN = 3
DEFAULT = "default"


def build_baskets() -> pd.DataFrame:
    loans = pd.read_csv(DATA_DIR / "loan.csv", sep=";")
    transactions = pd.read_csv(DATA_DIR / "trans.csv", sep=";", low_memory=False)

    loans["date"] = pd.to_datetime(loans["date"], format="%y%m%d")
    transactions["date"] = pd.to_datetime(transactions["date"], format="%y%m%d")

    first_loan = loans.groupby("account_id", as_index=False)["date"].min()
    transactions = transactions.merge(first_loan, on="account_id", suffixes=("", "_loan"))
    transactions = transactions.loc[transactions["date"] < transactions["date_loan"]]

    high_amount = transactions["amount"] > transactions["amount"].quantile(0.75)
    flags = transactions.assign(
        type_PRIJEM=transactions["type"].eq("PRIJEM"),
        type_VYDAJ=transactions["type"].eq("VYDAJ"),
        type_VYBER=transactions["type"].eq("VYBER"),
        high_amount=high_amount,
        negative_balance=transactions["balance"] < 0,
    )
    baskets = flags.groupby("account_id")[[
        "type_PRIJEM", "type_VYDAJ", "type_VYBER", "high_amount", "negative_balance"
    ]].any()
    default_accounts = loans.loc[
        loans["status"].isin(["B", "D"]), "account_id"
    ]
    baskets[DEFAULT] = baskets.index.isin(default_accounts)
    return baskets


def prune_redundant(rules: pd.DataFrame) -> pd.DataFrame:
    """Drop a rule when a more general rule is at least as strong."""
    if rules.empty:
        return rules.drop(columns=["_len"], errors="ignore")

    rules = rules.copy()
    rules["_len"] = rules["antecedents"].map(len)
    rules = rules.sort_values(
        ["_len", "confidence", "lift"], ascending=[True, False, False]
    )

    kept: list[tuple[set, float, float]] = []
    rows = []
    for _, row in rules.iterrows():
        antecedent = set(row["antecedents"])
        redundant = any(
            antecedent.issuperset(previous)
            and row["confidence"] <= previous_confidence + 1e-12
            and row["lift"] <= previous_lift + 1e-12
            for previous, previous_confidence, previous_lift in kept
        )
        if redundant:
            continue
        kept.append((antecedent, row["confidence"], row["lift"]))
        rows.append(row)

    return pd.DataFrame(rows).drop(columns=["_len"], errors="ignore")


def mine(
    method_name: str, miner, baskets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    itemsets = miner(
        baskets, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN
    )
    itemsets["method"] = method_name

    rules = association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    rules = rules[
        rules["consequents"].map(lambda items: set(items) == {DEFAULT})
        & rules["antecedents"].map(lambda items: DEFAULT not in items)
        & (rules["lift"] > 1.0)
    ].copy()
    rules = prune_redundant(rules)
    rules["method"] = method_name
    return itemsets, rules


def main() -> None:
    baskets = build_baskets()
    results = [mine(name, miner, baskets) for name, miner in {
        "apriori": apriori,
        "fpgrowth": fpgrowth,
    }.items()]

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.concat([sets for sets, _ in results], ignore_index=True).to_csv(
        OUTPUT_DIR / "pattern_itemsets.csv", index=False
    )
    all_rules = pd.concat([rules for _, rules in results], ignore_index=True)
    all_rules = all_rules.sort_values(["lift", "confidence", "support"], ascending=False)
    all_rules.to_csv(OUTPUT_DIR / "pattern_rules.csv", index=False)

    print(f"Accounts: {len(baskets)}, defaults: {int(baskets[DEFAULT].sum())}")
    print(f"Apriori itemsets: {len(results[0][0])}")
    print(f"FP-Growth itemsets: {len(results[1][0])}")
    print(f"Default-oriented rules: {len(results[0][1])} Apriori, {len(results[1][1])} FP-Growth")
    if not all_rules.empty:
        print(all_rules[["method", "antecedents", "support", "confidence", "lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
