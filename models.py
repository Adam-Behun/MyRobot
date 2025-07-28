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
    
    async def get_patient_demographics(self, patient_id: str) -> Optional[dict]:
        """Get patient demographic information"""
        patient = await self.find_patient_by_id(patient_id)
        if patient:
            return {
                "patient_name": patient.get("patient_name"),
                "date_of_birth": patient.get("date_of_birth"),
                "sex": patient.get("sex"),
                "phone": patient.get("patient_phone_number"),
                "address": {
                    "street": patient.get("address"),
                    "city": patient.get("city"),
                    "state": patient.get("state"),
                    "zip": patient.get("zip_code")
                }
            }
        return None
    
    async def get_patient_insurance(self, patient_id: str) -> Optional[dict]:
        """Get patient insurance information"""
        patient = await self.find_patient_by_id(patient_id)
        if patient:
            return {
                "company_name": patient.get("insurance_company_name"),
                "member_id": patient.get("insurance_member_id"),
                "phone_number": patient.get("insurance_phone_number"),
                "plan_type": patient.get("plan_type")
            }
        return None
    
    async def get_patient_medical_info(self, patient_id: str) -> Optional[dict]:
        """Get patient medical/procedure information"""
        patient = await self.find_patient_by_id(patient_id)
        if patient:
            return {
                "cpt_code": patient.get("cpt_code"),
                "icd10_code": patient.get("icd10_code"),
                "appointment_time": patient.get("appointment_time"),
                "prior_auth_status": patient.get("prior_auth_status")
            }
        return None
    
    async def get_provider_info(self, patient_id: str) -> Optional[dict]:
        """Get provider information for patient"""
        patient = await self.find_patient_by_id(patient_id)
        if patient:
            return {
                "provider_name": patient.get("provider_name"),
                "provider_npi": patient.get("provider_npi"),
                "provider_phone": patient.get("provider_phone_number"),
                "provider_specialty": patient.get("provider_specialty")
            }
        return None
    
    async def get_facility_info(self, patient_id: str) -> Optional[dict]:
        """Get facility information for patient"""
        patient = await self.find_patient_by_id(patient_id)
        if patient:
            return {
                "facility_name": patient.get("facility_name"),
                "facility_npi": patient.get("facility_npi"),
                "place_of_service_code": patient.get("place_of_service_code")
            }
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
    
    async def find_patients_by_facility(self, facility_name: str) -> List[dict]:
        """Find all patients for a specific facility"""
        try:
            cursor = self.patients.find({"facility_name": facility_name})
            return await cursor.to_list(length=None)
        except:
            return []
    
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