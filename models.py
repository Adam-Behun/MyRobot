from pymongo import MongoClient
from typing import Optional, List
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PatientRecord:
    def __init__(self, db_client):
        self.client = db_client
        self.db_name = os.getenv("MONGO_DB_NAME", "alfons")
        self.db = db_client[self.db_name]
        self.patients = self.db.patients
    
    def find_patient_by_name_and_dob(self, name: str, date_of_birth: str) -> Optional[dict]:
        """Find patient by name and date of birth"""
        return self.patients.find_one({
            "patient_name": name,
            "date_of_birth": date_of_birth
        })

# Initialize MongoDB connection
def get_db_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    return MongoClient(mongo_uri)