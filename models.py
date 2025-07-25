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

    def find_patient_by_member_id(self, member_id: str) -> Optional[dict]:
        """Find patient by insurance member ID"""
        return self.patients.find_one({"insurance_member_id": member_id})
    
    def find_patient_by_name_and_dob(self, name: str, date_of_birth: str) -> Optional[dict]:
        """Find patient by name and date of birth"""
        return self.patients.find_one({
            "patient_name": name,
            "date_of_birth": date_of_birth
        })
    
    def update_prior_auth_status(self, patient_id: str, status: str) -> bool:
        """Update the prior authorization status for a patient"""
        from bson import ObjectId
        try:
            result = self.patients.update_one(
                {"_id": ObjectId(patient_id)},
                {
                    "$set": {
                        "prior_auth_status": status,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            return result.modified_count > 0
        except:
            return False

# Initialize MongoDB connection
def get_db_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    return MongoClient(mongo_uri)