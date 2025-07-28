import time
import logging
from typing import Optional, Dict, Any
from models import get_async_patient_db, AsyncPatientRecord

logger = logging.getLogger(__name__)

# Initialize database connection
patient_db = get_async_patient_db()

async def get_facility_name(patient_id: str) -> Optional[str]:
    """
    Get the facility name for a patient.
    This is used when introducing MyRobot to the insurance company.
    """
    start_time = time.time()
    
    try:
        facility_info = await patient_db.get_facility_info(patient_id)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Facility lookup latency: {latency:.2f}ms")
        
        if facility_info:
            facility_name = facility_info.get("facility_name")
            logger.info(f"Found facility: {facility_name}")
            return facility_name
        else:
            logger.warning(f"No facility found for patient ID: {patient_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting facility name for patient {patient_id}: {e}")
        return None

async def get_patient_demographics(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get patient demographic information (name, DOB, address, etc.)
    """
    start_time = time.time()
    
    try:
        demographics = await patient_db.get_patient_demographics(patient_id)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Demographics lookup latency: {latency:.2f}ms")
        
        if demographics:
            logger.info(f"Found demographics for: {demographics.get('patient_name')}")
            return demographics
        else:
            logger.warning(f"No demographics found for patient ID: {patient_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting patient demographics for {patient_id}: {e}")
        return None

async def get_patient_insurance_info(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get patient insurance information (company, member ID, plan type, etc.)
    """
    start_time = time.time()
    
    try:
        insurance = await patient_db.get_patient_insurance(patient_id)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Insurance lookup latency: {latency:.2f}ms")
        
        if insurance:
            logger.info(f"Found insurance: {insurance.get('company_name')} - {insurance.get('member_id')}")
            return insurance
        else:
            logger.warning(f"No insurance found for patient ID: {patient_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting patient insurance for {patient_id}: {e}")
        return None

async def get_patient_medical_info(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get patient medical information (CPT codes, ICD10, appointment, auth status)
    """
    start_time = time.time()
    
    try:
        medical_info = await patient_db.get_patient_medical_info(patient_id)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Medical info lookup latency: {latency:.2f}ms")
        
        if medical_info:
            logger.info(f"Found medical info - CPT: {medical_info.get('cpt_code')}, Status: {medical_info.get('prior_auth_status')}")
            return medical_info
        else:
            logger.warning(f"No medical info found for patient ID: {patient_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting patient medical info for {patient_id}: {e}")
        return None

async def get_provider_info(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get provider information for the patient
    """
    start_time = time.time()
    
    try:
        provider = await patient_db.get_provider_info(patient_id)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Provider lookup latency: {latency:.2f}ms")
        
        if provider:
            logger.info(f"Found provider: {provider.get('provider_name')} - {provider.get('provider_specialty')}")
            return provider
        else:
            logger.warning(f"No provider found for patient ID: {patient_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting provider info for {patient_id}: {e}")
        return None

async def search_patient_by_name(first_name: str, last_name: str) -> Optional[Dict[str, Any]]:
    """
    Search for a patient by first and last name.
    Returns patient data if found, None if not found.
    """
    start_time = time.time()
    
    try:
        # Combine first and last name for search
        full_name = f"{first_name} {last_name}"
        patient = await patient_db.find_patient_by_name_and_dob(full_name, None)
        
        # If exact match not found, try case-insensitive search
        if not patient:
            patient = await patient_db.patients.find_one({
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
        "name": "get_facility_name",
        "description": "Get the facility name for a patient - used when introducing MyRobot",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_demographics",
        "description": "Get patient demographic information (name, DOB, address, phone)",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_insurance_info",
        "description": "Get patient insurance information (company, member ID, plan type)",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_patient_medical_info",
        "description": "Get patient medical information (CPT codes, ICD10, appointment time, auth status)",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "get_provider_info",
        "description": "Get provider information for the patient (name, NPI, specialty)",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient's MongoDB ObjectId"
                }
            },
            "required": ["patient_id"]
        }
    },
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
    "get_facility_name": get_facility_name,
    "get_patient_demographics": get_patient_demographics,
    "get_patient_insurance_info": get_patient_insurance_info,
    "get_patient_medical_info": get_patient_medical_info,
    "get_provider_info": get_provider_info,
    "search_patient_by_name": search_patient_by_name,
    "update_prior_auth_status": update_prior_auth_status
}