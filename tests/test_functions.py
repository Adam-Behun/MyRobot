import pytest
import os
import sys
from unittest.mock import patch, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import search_patient_by_name, update_prior_auth_status, PATIENT_FUNCTIONS, FUNCTION_REGISTRY

class TestPatientFunctions:
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_search_patient_by_name_found(self, mock_patient_db):
        """Test successful patient search by name"""
        # Mock patient data
        mock_patient = {
            "_id": "507f1f77bcf86cd799439011",
            "patient_name": "John Doe",
            "date_of_birth": "1980-01-01",
            "prior_auth_status": "Pending"
        }
        
        mock_patient_db.find_patient_by_name_and_dob = AsyncMock(return_value=mock_patient)
        
        result = await search_patient_by_name("John", "Doe")
        
        assert result is not None
        assert result["patient_name"] == "John Doe"
        assert result["_id"] == "507f1f77bcf86cd799439011"
        mock_patient_db.find_patient_by_name_and_dob.assert_called_once_with("John Doe", None)
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_search_patient_by_name_not_found(self, mock_patient_db):
        """Test patient search when patient not found"""
        mock_patient_db.find_patient_by_name_and_dob = AsyncMock(return_value=None)
        mock_patient_db.patients.find_one = AsyncMock(return_value=None)
        
        result = await search_patient_by_name("Jane", "Smith")
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_search_patient_by_name_case_insensitive(self, mock_patient_db):
        """Test case-insensitive patient search"""
        mock_patient = {
            "_id": "507f1f77bcf86cd799439011",
            "patient_name": "John Doe"
        }
        
        # First call returns None, second call (case-insensitive) returns patient
        mock_patient_db.find_patient_by_name_and_dob = AsyncMock(return_value=None)
        mock_patient_db.patients.find_one = AsyncMock(return_value=mock_patient)
        
        result = await search_patient_by_name("john", "doe")
        
        assert result is not None
        assert result["patient_name"] == "John Doe"
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_search_patient_by_name_exception(self, mock_patient_db):
        """Test patient search with database exception"""
        mock_patient_db.find_patient_by_name_and_dob = AsyncMock(side_effect=Exception("Database error"))
        
        result = await search_patient_by_name("John", "Doe")
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_update_prior_auth_status_success(self, mock_patient_db):
        """Test successful prior auth status update"""
        mock_patient_db.update_prior_auth_status = AsyncMock(return_value=True)
        
        result = await update_prior_auth_status("507f1f77bcf86cd799439011", "Approved")
        
        assert result is True
        mock_patient_db.update_prior_auth_status.assert_called_once_with(
            "507f1f77bcf86cd799439011", "Approved"
        )
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_update_prior_auth_status_failure(self, mock_patient_db):
        """Test prior auth status update failure"""
        mock_patient_db.update_prior_auth_status = AsyncMock(return_value=False)
        
        result = await update_prior_auth_status("507f1f77bcf86cd799439011", "Denied")
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_update_prior_auth_status_exception(self, mock_patient_db):
        """Test prior auth status update with exception"""
        mock_patient_db.update_prior_auth_status = AsyncMock(side_effect=Exception("Database error"))
        
        result = await update_prior_auth_status("507f1f77bcf86cd799439011", "Approved")
        
        assert result is False

class TestFunctionDefinitions:
    
    def test_patient_functions_schema(self):
        """Test function definitions for LLM integration"""
        assert len(PATIENT_FUNCTIONS) == 2
        
        # Test search function schema
        search_func = PATIENT_FUNCTIONS[0]
        assert search_func["name"] == "search_patient_by_name"
        assert "first_name" in search_func["parameters"]["properties"]
        assert "last_name" in search_func["parameters"]["properties"]
        assert search_func["parameters"]["required"] == ["first_name", "last_name"]
        
        # Test update function schema
        update_func = PATIENT_FUNCTIONS[1]
        assert update_func["name"] == "update_prior_auth_status"
        assert "patient_id" in update_func["parameters"]["properties"]
        assert "status" in update_func["parameters"]["properties"]
        assert update_func["parameters"]["required"] == ["patient_id", "status"]
    
    def test_function_registry(self):
        """Test function registry contains all functions"""
        assert "search_patient_by_name" in FUNCTION_REGISTRY
        assert "update_prior_auth_status" in FUNCTION_REGISTRY
        
        # Test functions are callable
        assert callable(FUNCTION_REGISTRY["search_patient_by_name"])
        assert callable(FUNCTION_REGISTRY["update_prior_auth_status"])
    
    def test_status_enum_values(self):
        """Test that status enum contains expected values"""
        update_func = next(f for f in PATIENT_FUNCTIONS if f["name"] == "update_prior_auth_status")
        status_enum = update_func["parameters"]["properties"]["status"]["enum"]
        
        expected_statuses = ["Approved", "Denied", "Pending", "Under Review"]
        for status in expected_statuses:
            assert status in status_enum

class TestIntegration:
    
    @pytest.mark.asyncio
    @patch('functions.patient_db')
    async def test_function_registry_integration(self, mock_patient_db):
        """Test that function registry works with actual function calls"""
        mock_patient_db.find_patient_by_name_and_dob = AsyncMock(return_value={"patient_name": "Test Patient"})
        mock_patient_db.patients.find_one = AsyncMock(return_value=None)
        
        # Test calling function through registry
        search_func = FUNCTION_REGISTRY["search_patient_by_name"]
        result = await search_func("John", "Doe")
        
        assert result is not None
        assert result["patient_name"] == "Test Patient"

if __name__ == "__main__":
    pytest.main([__name__])