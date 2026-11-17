"""
Phase 1: ingest raw Telco churn CSV into MongoDB (churn_db.raw_customers).

Data-handling decisions (documented for README):
- TotalCharges arrives as a string with ' ' blanks for tenure==0 customers.
  We coerce to numeric (NaN -> BSON null) so Phase-1 aggregation can filter
  missing values cleanly with a single $ne: None predicate.
- Churn stays as 'Yes'/'No' strings in the raw collection; label encoding
  happens in Phase 2 against the sklearn pipeline.
- JSON round-trip is the BSON-safety boundary: pandas NaNs become JSON nulls,
  numpy scalars become Python natives. Anything pymongo might reject for
  BSON gets normalized in one place instead of being patched per column.
"""
import json
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "Telco-Customer-Churn.csv"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "churn_db"
COLL_NAME = "raw_customers"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nulls = int(df["TotalCharges"].isna().sum())
    print(f"[load] rows={len(df)} cols={len(df.columns)} total_charges_nulls={nulls}")
    return df


def ingest(df: pd.DataFrame) -> None:
    records = json.loads(df.to_json(orient="records"))

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    coll = client[DB_NAME][COLL_NAME]

    coll.drop()
    result = coll.insert_many(records)
    print(f"[ingest] inserted={len(result.inserted_ids)} into {DB_NAME}.{COLL_NAME}")

    print(f"[verify] count={coll.count_documents({})}")
    sample = coll.find_one({}, {"_id": 0})
    print(f"[sample] {sample}")


if __name__ == "__main__":
    ingest(load(CSV))