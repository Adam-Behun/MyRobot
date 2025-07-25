import pytest
from unittest.mock import patch, AsyncMock
from models import AsyncPatientRecord
from bson import ObjectId

@pytest.mark.asyncio
class TestAsyncPatientRecord:
    
    @patch('models.AsyncIOMotorClient')
    async def test_patient_record_initialization(self, mock_motor_client):
        """Test AsyncPatientRecord can be initialized"""
        mock_client = AsyncMock()
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.patients = mock_collection
        patient_db = AsyncPatientRecord(mock_client)
        assert patient_db.db is not None
        assert patient_db.patients is not None
    
    @patch('models.AsyncIOMotorClient')
    async def test_find_patient_by_id(self, mock_motor_client):
        """Test patient lookup functionality by ID"""
        mock_client = AsyncMock()
        mock_db = AsyncMock()
        mock_collection = AsyncMock()
        mock_client.__getitem__.return_value = mock_db
        mock_db.patients = mock_collection
        mock_collection.find_one.return_value = {
            "_id": ObjectId("668131f380921c8f2f7567d3"),
            "patient_name": "John Doe",
            "date_of_birth": "1980-01-01",
            "sex": "Male",
            "patient_phone_number": "123-456-7890",
            "address": "123 Main St",
            "city": "Anytown",
            "state": "CA",
            "zip_code": "90210",
            "insurance_company_name": "BlueCross",
            "insurance_member_id": "ABC123",
            "insurance_phone_number": "516-566-7132",
            "plan_type": "PPO",
            "cpt_code": "99214",
            "icd10_code": "I10",
            "appointment_time": "2025-08-01T10:00:00Z",
            "provider_name": "Dr. Smith",
            "provider_npi": "1234567890",
            "provider_phone_number": "555-1234",
            "provider_specialty": "Family Medicine",
            "facility_name": "City Hospital",
            "facility_npi": "0987654321",
            "place_of_service_code": "11",
            "prior_auth_status": "Pending",
            "created_at": "2025-07-23T00:00:00Z",
            "updated_at": "2025-07-23T00:00:00Z"
        }
        
        patient_db = AsyncPatientRecord(mock_client)
        result = await patient_db.find_patient_by_id("668131f380921c8f2f7567d3")
        
        assert result is not None
        assert result["patient_name"] == "John Doe"
        mock_collection.find_one.assert_called_once_with({"_id": ObjectId("668131f380921c8f2f7567d3")})