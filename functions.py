import time
import logging
from typing import Optional, Dict, Any
from models import get_async_patient_db, AsyncPatientRecord

logger = logging.getLogger(__name__)

# Initialize database connection
patient_db = get_async_patient_db()

async def update_prior_auth_status(patient_id: str, status: str) -> bool:
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
        logger.info(f"Attempting to update patient {patient_id} to status '{status}'")
        
        # Verify patient exists first
        patient = await patient_db.find_patient_by_id(patient_id)
        if not patient:
            logger.error(f"Patient not found: {patient_id}")
            return False
            
        success = await patient_db.update_prior_auth_status(patient_id, status)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Prior auth update latency: {latency:.2f}ms")
        
        if success:
            logger.info(f"Successfully updated prior auth status to '{status}' for patient ID: {patient_id}")
            
            # Verify the update
            updated_patient = await patient_db.find_patient_by_id(patient_id)
            logger.info(f"Verification - new status: {updated_patient.get('prior_auth_status', 'ERROR')}")
        else:
            logger.error(f"Failed to update prior auth status for patient ID: {patient_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Exception updating prior auth status for patient {patient_id}: {e}")
        return False

# Function definitions for LLM function calling
PATIENT_FUNCTIONS = [
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
    "update_prior_auth_status": update_prior_auth_status
}