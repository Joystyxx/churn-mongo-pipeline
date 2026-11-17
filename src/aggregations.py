"""
Phase 1: MongoDB aggregation pipeline.

Brief deliverable:
- Filter rows with missing TotalCharges.
- Calculate baseline churn rate.

Added beyond the brief (justified in README):
- $facet: one server-side pass that computes per-label counts,
  per-label average MonthlyCharges, AND overall totals. Demonstrates
  real aggregation composition rather than four round trips.
- Churn rate by Contract: business narrative for the README's
  problem statement (month-to-month churn is the expected pain point).
"""
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "churn_db"
COLL_NAME = "raw_customers"


def report():
    coll = MongoClient(MONGO_URI)[DB_NAME][COLL_NAME]

    total_raw = coll.count_documents({})
    filter_stage = {"$match": {"TotalCharges": {"$ne": None}}}

    # One pass: per-label stats + overall totals, on the filtered set.
    facet = list(coll.aggregate([
        filter_stage,
        {"$facet": {
            "by_label": [
                {"$group": {
                    "_id": "$Churn",
                    "n": {"$sum": 1},
                    "avg_monthly": {"$avg": "$MonthlyCharges"},
                }},
                {"$sort": {"_id": 1}},
            ],
            "totals": [
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "churners": {"$sum": {
                        "$cond": [{"$eq": ["$Churn", "Yes"]}, 1, 0]
                    }},
                }},
            ],
        }},
    ]))[0]

    totals = facet["totals"][0]
    filtered_total = totals["total"]
    churners = totals["churners"]
    rate = churners / filtered_total

    print(f"[filter] raw={total_raw}  dropped={total_raw - filtered_total}  kept={filtered_total}")
    print(f"[baseline] churners={churners}  churn_rate={rate:.4f}")
    print("[by_label]")
    for row in facet["by_label"]:
        print(f"  {row['_id']:>3}  n={row['n']:>5}  avg_monthly={row['avg_monthly']:.2f}")

    # Churn rate by Contract (extra).
    by_contract = list(coll.aggregate([
        filter_stage,
        {"$group": {
            "_id": "$Contract",
            "total": {"$sum": 1},
            "churners": {"$sum": {"$cond": [{"$eq": ["$Churn", "Yes"]}, 1, 0]}},
        }},
        {"$addFields": {
            "churn_rate": {"$round": [{"$divide": ["$churners", "$total"]}, 4]}
        }},
        {"$sort": {"churn_rate": -1}},
    ]))
    print("[by_contract]")
    for row in by_contract:
        print(f"  {row['_id']:<16}  total={row['total']:>5}  churn_rate={row['churn_rate']:.4f}")


if __name__ == "__main__":
    report()