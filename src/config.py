"""Centralized configuration and DB connection logic."""
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "churn_db"
RAW_COLL = "raw_customers"
EXP_COLL = "experiments"

def get_db():
    """Returns the churn_db database handle."""
    return MongoClient(MONGO_URI)[DB_NAME]