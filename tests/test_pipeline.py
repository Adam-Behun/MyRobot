import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import HealthcareAIPipeline
from models import PatientRecord, get_db_client

class TestHealthcareAIPipeline:
    
    def test_pipeline_initialization(self):
        """Test pipeline components can be initialized"""
        pipeline = HealthcareAIPipeline()
        assert pipeline.transport is None
        assert pipeline.pipeline is None
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'DEEPGRAM_API_KEY': 'test-key', 
        'CARTESIA_API_KEY': 'test-key',
        'LIVEKIT_URL': 'ws://test:7880'
    })
    def test_create_pipeline(self):
        """Test pipeline creation with mocked environment"""
        pipeline = HealthcareAIPipeline()
        
        # Mock the external services
        with patch('pipeline.LiveKitTransport'), \
             patch('pipeline.DeepgramSTTService'), \
             patch('pipeline.OpenAILLMService'), \
             patch('pipeline.CartesiaTTSService'):
            
            result = pipeline.create_pipeline()
            assert result is not None
            assert pipeline.pipeline is not None

class TestPatientRecord:
    
    @patch('models.MongoClient')
    def test_patient_record_initialization(self, mock_mongo):
        """Test PatientRecord can be initialized"""
        mock_client = Mock()
        patient_db = PatientRecord(mock_client)
        assert patient_db.db is not None
        assert patient_db.patients is not None
    
    @patch('models.MongoClient')
    def test_find_patient(self, mock_mongo):
        """Test patient lookup functionality"""
        mock_client = Mock()
        mock_client.healthcare_ai.patients.find_one.return_value = {
            "patient_id": "12345",
            "name": "John Doe"
        }
        
        patient_db = PatientRecord(mock_client)
        result = patient_db.find_patient("12345")
        
        assert result is not None
        assert result["patient_id"] == "12345"
        assert result["name"] == "John Doe"

if __name__ == "__main__":
    pytest.main([__name__])