from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AsyncPatientRecord:
    """Asynchronous PatientRecord class with full schema support"""
    def __init__(self, db_client: AsyncIOMotorClient):
        self.client = db_client
        self.db_name = os.getenv("MONGO_DB_NAME", "alfons")
        self.db = db_client[self.db_name]
        self.patients = self.db.patients
    
    async def find_patient_by_id(self, patient_id: str) -> Optional[dict]:
        """Find patient by MongoDB ObjectId"""
        from bson import ObjectId
        try:
            return await self.patients.find_one({"_id": ObjectId(patient_id)})
        except:
            return None
    
    async def get_complete_patient_record(self, patient_id: str) -> Optional[dict]:
        """Get complete patient record with all information"""
        return await self.find_patient_by_id(patient_id)
    
    async def update_prior_auth_status(self, patient_id: str, status: str) -> bool:
        """Update the prior authorization status for a patient"""
        from bson import ObjectId
        try:
            result = await self.patients.update_one(
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
    
    async def find_patients_pending_auth(self) -> List[dict]:
        """Find all patients with pending authorization"""
        try:
            cursor = self.patients.find({"prior_auth_status": "Pending"})
            return await cursor.to_list(length=None)
        except:
            return []

def get_async_db_client():
    """Get asynchronous MongoDB client"""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    return AsyncIOMotorClient(mongo_uri)

# Global async client instance
_async_client = None

def get_async_patient_db() -> AsyncPatientRecord:
    """Get async patient database instance"""
    global _async_client
    if not _async_client:
        _async_client = get_async_db_client()
    return AsyncPatientRecord(_async_client)