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
        success = await patient_db.update_prior_auth_status(patient_id, status)
        
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