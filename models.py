from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AsyncPatientRecord:
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
    
    async def find_patient_by_phone(self, phone_number: str) -> Optional[dict]:
        """Find patient by phone number"""
        return await self.patients.find_one({"patient_phone_number": phone_number})
    
    async def find_patient_by_member_id(self, member_id: str) -> Optional[dict]:
        """Find patient by insurance member ID"""
        return await self.patients.find_one({"insurance_member_id": member_id})
    
    async def find_patient_by_name_and_dob(self, name: str, date_of_birth: str = None) -> Optional[dict]:
        """Find patient by name and optionally date of birth"""
        query = {"patient_name": name}
        if date_of_birth:
            query["date_of_birth"] = date_of_birth
        return await self.patients.find_one(query)
    
    async def find_patient_by_name(self, name: str) -> Optional[dict]:
        """Find patient by name with case-insensitive search"""
        try:
            # Try exact match first
            patient = await self.patients.find_one({"patient_name": name})
            
            # If not found, try case-insensitive
            if not patient:
                patient = await self.patients.find_one({
                    "patient_name": {"$regex": f"^{name}$", "$options": "i"}
                })
            
            return patient
        except Exception:
            return None
    
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

# Initialize async MongoDB connection
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