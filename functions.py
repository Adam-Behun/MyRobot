import time
import logging
from typing import Optional, Dict, Any
from models import get_async_patient_db, AsyncPatientRecord

logger = logging.getLogger(__name__)

# Initialize database connection
db_client = get_db_client()
patient_db = PatientRecord(db_client)

def search_patient_by_name(first_name: str, last_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a patient by first and last name.
    Returns patient data if found, None if not found.
    """
    start_time = time.time()
    
    try:
        # Combine first and last name for search
        full_name = f"{first_name} {last_name}"
        patient = patient_db.find_patient_by_name_and_dob(full_name, None)
        
        # If exact match not found, try case-insensitive search
        if not patient:
            patient = patient_db.patients.find_one({
                "patient_name": {"$regex": f"^{full_name}$", "$options": "i"}
            })
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Patient search latency: {latency:.2f}ms")
        
        if patient:
            logger.info(f"Found patient: {patient.get('patient_name')}")
            return patient
        else:
            logger.info(f"No patient found for: {full_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error searching for patient {first_name} {last_name}: {e}")
        return None

def update_prior_auth_status(patient_id: str, status: str) -> bool:
    """
    Update the prior authorization status for a patient.
    
    Args:
        patient_id: MongoDB ObjectId as string
        status: New authorization status (e.g., "Approved", "Denied", "Pending")
    
    Returns:
        True if update successful, False otherwise
    """
    start_time = time.time()
    
    try:
        success = patient_db.update_prior_auth_status(patient_id, status)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Prior auth update latency: {latency:.2f}ms")
        
        if success:
            logger.info(f"Updated prior auth status to '{status}' for patient ID: {patient_id}")
        else:
            logger.warning(f"Failed to update prior auth status for patient ID: {patient_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error updating prior auth status for patient {patient_id}: {e}")
        return False

# Function definitions for LLM function calling
PATIENT_FUNCTIONS = [
    {
        "name": "search_patient_by_name",
        "description": "Search for a patient by their first and last name",
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "Patient's first name"
                },
                "last_name": {
                    "type": "string", 
                    "description": "Patient's last name"
                }
            },
            "required": ["first_name", "last_name"]
        }
    },
    {
        "name": "update_prior_auth_status",
        "description": "Update the prior authorization status for a patient",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                },
                "status": {
                    "type": "string",
                    "description": "New authorization status",
                    "enum": ["Approved", "Denied", "Pending", "Under Review"]
                }
            },
            "required": ["patient_id", "status"]
        }
    }
]

# Function registry for easy access
FUNCTION_REGISTRY = {
    "search_patient_by_name": search_patient_by_name,
    "update_prior_auth_status": update_prior_auth_status
}