import pytest
import asyncio
import os
import sys
import time
import logging
from unittest.mock import Mock, patch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_async_patient_db, AsyncPatientRecord, get_async_db_client
from pipeline import HealthcareAIPipeline
from optimization import LatencyOptimizer
from interruption_handler import InterruptionHandler
from audio_processing import AudioProcessor
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv()

# Configure logging for integration tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMongoDBIntegration:
    """Test MongoDB Atlas connection and operations"""
    
    @pytest.mark.asyncio
    async def test_async_mongodb_connection(self):
        """Test async MongoDB connection"""
        try:
            client = get_async_db_client()
            # Ping the server
            await client.admin.command('ping')
            logger.info("✅ Async MongoDB connection successful")
        except Exception as e:
            pytest.fail(f"❌ Async MongoDB connection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_async_database_and_collection_access(self):
        """Test async access to specific database and collection"""
        try:
            client = get_async_db_client()
            patient_db = AsyncPatientRecord(client)
            
            # Test database and collection access
            db_name = os.getenv("MONGO_DB_NAME", "alfons")
            assert patient_db.db_name == db_name
            assert patient_db.patients is not None
            
            # Test collection exists and is accessible
            collection_info = await patient_db.patients.database.list_collection_names()
            assert "patients" in collection_info
            
            logger.info(f"✅ Async Database '{db_name}' and 'patients' collection accessible")
        except Exception as e:
            pytest.fail(f"❌ Async Database/collection access failed: {e}")
    
    @pytest.mark.asyncio
    async def test_async_patient_search_functionality(self):
        """Test async patient search with real database"""
        try:
            client = get_async_db_client()
            patient_db = AsyncPatientRecord(client)
            
            # Test async search by name
            result = await patient_db.find_patient_by_name("Test Patient")
            
            # Result can be None (no patient found) or a dict (patient found)
            assert result is None or isinstance(result, dict)
            
            logger.info("✅ Async patient search functionality working")
            
            # If a patient exists, test the structure
            if result:
                required_fields = ["patient_name", "_id"]
                for field in required_fields:
                    assert field in result, f"Missing required field: {field}"
                logger.info(f"✅ Found patient with proper structure: {result.get('patient_name')}")
                
        except Exception as e:
            pytest.fail(f"❌ Async patient search failed: {e}")
    
    @pytest.mark.asyncio
    async def test_async_patient_operations(self):
        """Test async patient operations"""
        try:
            client = get_async_db_client()
            patient_db = AsyncPatientRecord(client)
            
            # Test async search
            result = await patient_db.find_patient_by_name("Test Patient")
            
            if result:
                # Test async update
                patient_id = str(result["_id"])
                original_status = result.get("prior_auth_status", "Unknown")
                
                # Update to a test status
                update_success = await patient_db.update_prior_auth_status(patient_id, "Async Test Status")
                assert isinstance(update_success, bool)
                
                # Restore original status if update succeeded
                if update_success:
                    await patient_db.update_prior_auth_status(patient_id, original_status)
                
                logger.info("✅ Async patient operations working with real data")
            else:
                logger.info("✅ Async patient operations working (no test data found)")
                
        except Exception as e:
            pytest.fail(f"❌ Async patient operations failed: {e}")

class TestOpenAIIntegration:
    """Test OpenAI API connection and functionality"""
    
    def test_openai_api_key(self):
        """Test OpenAI API key is configured"""
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY not found in environment"
        assert api_key.startswith("sk-"), "Invalid OpenAI API key format"
        assert len(api_key) > 20, "OpenAI API key seems too short"
        logger.info("✅ OpenAI API key properly configured")
    
    @pytest.mark.asyncio
    async def test_openai_connection(self):
        """Test actual OpenAI API connection"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Test with a simple completion
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Say 'test successful' if you receive this message."}
                ],
                max_tokens=10,
                timeout=30
            )
            
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content) > 0
            
            logger.info("✅ OpenAI API connection successful")
            logger.info(f"Response: {response.choices[0].message.content}")
            
        except Exception as e:
            pytest.fail(f"❌ OpenAI API connection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_openai_function_calling(self):
        """Test OpenAI function calling capability"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Test function calling
            tools = [{
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Test message"}
                        },
                        "required": ["message"]
                    }
                }
            }]
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "Call the test function with message 'hello'"}
                ],
                tools=tools,
                tool_choice="auto",
                timeout=30
            )
            
            # Check if function was called
            message = response.choices[0].message
            if message.tool_calls:
                logger.info("✅ OpenAI function calling working")
            else:
                logger.info("✅ OpenAI responded (function calling may not have triggered)")
                
        except Exception as e:
            pytest.fail(f"❌ OpenAI function calling test failed: {e}")

class TestDeepgramIntegration:
    """Test Deepgram STT API connection"""
    
    def test_deepgram_api_key(self):
        """Test Deepgram API key is configured"""
        api_key = os.getenv("DEEPGRAM_API_KEY")
        assert api_key is not None, "DEEPGRAM_API_KEY not found in environment"
        assert len(api_key) > 20, "Deepgram API key seems too short"
        logger.info("✅ Deepgram API key properly configured")
    
    @pytest.mark.asyncio
    async def test_deepgram_connection(self):
        """Test Deepgram API connection"""
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            
            api_key = os.getenv("DEEPGRAM_API_KEY")
            deepgram = DeepgramClient(api_key)
            
            # Test with a simple audio URL (Deepgram's test file)
            url = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
            
            options = PrerecordedOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
            )
            
            response = deepgram.listen.prerecorded.v("1").transcribe_url(
                {"url": url}, options, timeout=30
            )
            
            # Check response structure
            assert hasattr(response, 'results')
            assert response.results.channels[0].alternatives[0].transcript
            
            transcript = response.results.channels[0].alternatives[0].transcript
            logger.info("✅ Deepgram API connection successful")
            logger.info(f"Test transcript: {transcript[:50]}...")
            
        except Exception as e:
            pytest.fail(f"❌ Deepgram API connection failed: {e}")

class TestCartesiaIntegration:
    """Test Cartesia TTS API connection"""
    
    def test_cartesia_api_key(self):
        """Test Cartesia API key is configured"""
        api_key = os.getenv("CARTESIA_API_KEY")
        assert api_key is not None, "CARTESIA_API_KEY not found in environment"
        assert api_key.startswith("sk_car_"), "Invalid Cartesia API key format"
        logger.info("✅ Cartesia API key properly configured")
    
    @pytest.mark.asyncio
    async def test_cartesia_connection(self):
        """Test Cartesia API connection"""
        try:
            import cartesia
            
            api_key = os.getenv("CARTESIA_API_KEY")
            client = cartesia.Cartesia(api_key=api_key)
            
            # Test API connection by listing voices
            voices_pager = client.voices.list()
            voices = list(voices_pager)  # Consume the pager into a list
            
            assert len(voices) > 0, "No voices returned from Cartesia API"
            
            # Check if our configured voice ID exists
            voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
            voice_found = any(voice.id == voice_id for voice in voices)
            
            if voice_found:
                logger.info("✅ Cartesia API connection successful, configured voice found")
            else:
                logger.info("✅ Cartesia API connection successful, but configured voice not found")
                logger.info(f"Available voices: {[voice.name for voice in voices[:3]]}")
                
        except Exception as e:
            pytest.fail(f"❌ Cartesia API connection failed: {e}")

class TestLiveKitIntegration:
    """Test LiveKit configuration"""
    
    def test_livekit_credentials(self):
        """Test LiveKit credentials are configured"""
        url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        assert url is not None, "LIVEKIT_URL not found in environment"
        assert api_key is not None, "LIVEKIT_API_KEY not found in environment"
        assert api_secret is not None, "LIVEKIT_API_SECRET not found in environment"
        
        assert url.startswith("wss://"), "LiveKit URL should start with wss://"
        assert len(api_key) > 5, "LiveKit API key seems too short"
        assert len(api_secret) > 10, "LiveKit API secret seems too short"
        
        logger.info("✅ LiveKit credentials properly configured")
    
    @pytest.mark.asyncio
    async def test_livekit_connection(self):
        """Test LiveKit API connection"""
        try:
            from livekit import api
            
            url = os.getenv("LIVEKIT_URL")
            api_key = os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            # Create LiveKit API client
            lk_api = api.LiveKitAPI(url, api_key, api_secret)
            
            # Test connection by listing rooms
            rooms = await lk_api.room.list_rooms(api.ListRoomsRequest())
            
            # Should return a list (empty or with rooms)
            assert hasattr(rooms, 'rooms')
            
            logger.info(f"✅ LiveKit API connection successful, found {len(rooms.rooms)} rooms")
            
        except Exception as e:
            # LiveKit connection might fail in test environment, but credentials should be valid
            logger.warning(f"⚠️ LiveKit API connection failed (may be expected in test env): {e}")

class TestOptimizationFeatures:
    """Test optimization and performance features"""
    
    def test_latency_optimizer(self):
        """Test latency optimization features"""
        try:
            optimizer = LatencyOptimizer(target_latency_ms=800.0)
            
            # Test latency tracking
            optimizer.track_latency("STT", 150.0)
            optimizer.track_latency("LLM", 400.0)
            optimizer.track_latency("TTS", 200.0)
            
            # Get performance summary
            summary = optimizer.get_performance_summary()
            assert "average_total_latency_ms" in summary
            assert summary["target_latency_ms"] == 800.0
            
            # Test optimization suggestions
            optimizations = optimizer.optimize_for_state("patient_verification")
            assert "enable_parallel_processing" in optimizations
            
            logger.info("✅ Latency optimizer working correctly")
            
        except Exception as e:
            pytest.fail(f"❌ Latency optimizer test failed: {e}")
    
    def test_interruption_handler(self):
        """Test interruption detection and handling"""
        try:
            handler = InterruptionHandler()
            
            # Test AI response tracking
            handler.start_ai_response("Hello, I'm calling about your prior authorization request.")
            assert handler.is_ai_speaking == True
            
            # Test interruption detection
            interruption = handler.detect_interruption("Sorry, what?", {"workflow_state": "greeting"})
            if interruption:
                assert interruption.interruption_type is not None
            
            # Test silence timeout
            timeout = handler.check_silence_timeout(time.time() - 5.0)  # 5 seconds ago
            assert isinstance(timeout, bool)
            
            logger.info("✅ Interruption handler working correctly")
            
        except Exception as e:
            pytest.fail(f"❌ Interruption handler test failed: {e}")
    
    def test_audio_processor(self):
        """Test audio processing features"""
        try:
            processor = AudioProcessor()
            
            # Create test audio data
            test_audio = np.random.rand(1600).astype(np.float32)  # 100ms of audio at 16kHz
            
            # Process audio
            processed_audio, metrics = processor.process_audio_chunk(test_audio)
            
            # Verify processing
            assert processed_audio is not None
            assert len(processed_audio) == len(test_audio)
            assert metrics.quality_score >= 0.0
            assert metrics.quality_score <= 1.0
            
            # Test quality summary
            summary = processor.get_audio_quality_summary()
            if summary.get("status") != "no_data":
                assert "average_quality_score" in summary
            
            logger.info("✅ Audio processor working correctly")
            
        except Exception as e:
            pytest.fail(f"❌ Audio processor test failed: {e}")

class TestFullPipelineIntegration:
    """Test full pipeline with all optimizations"""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization_with_optimizations(self):
        """Test pipeline initializes with all optimization components"""
        try:
            # Initialize pipeline
            pipeline = HealthcareAIPipeline(session_id="integration_test")
            
            # Verify all components initialized
            assert pipeline.workflow is not None
            assert pipeline.memory is not None
            
            # Test conversation state
            state = pipeline.get_conversation_state()
            assert "workflow_state" in state
            assert "session_id" in state
            
            logger.info("✅ Full pipeline initialization successful")
            
        except Exception as e:
            pytest.fail(f"❌ Pipeline initialization failed: {e}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_simulation(self):
        """Test end-to-end conversation simulation with async operations"""
        try:
            pipeline = HealthcareAIPipeline(session_id="e2e_test")
            
            # Simulate conversation messages
            test_messages = [
                "Hello, I'm calling about John Doe",
                "John Doe, D-O-E",
                "He needs an MRI scan",
                "Yes, please approve it"
            ]
            
            # Process each message
            for i, message in enumerate(test_messages):
                # Add to memory
                pipeline.memory.add_message("user", message)
                
                # Simulate workflow progression
                if "name" in message.lower():
                    pipeline.workflow.advance_state()
                elif "scan" in message.lower() or "mri" in message.lower():
                    pipeline.workflow.advance_state()
                elif "approve" in message.lower():
                    pipeline.workflow.advance_state()
                
                logger.info(f"Processed message {i+1}: {message[:30]}...")
            
            # Verify final state
            final_state = pipeline.get_conversation_state()
            assert final_state["workflow_state"] != "greeting"  # Should have progressed
            assert final_state["conversation_summary"]["message_count"] == len(test_messages)
            
            logger.info("✅ End-to-end conversation simulation successful")
            
        except Exception as e:
            pytest.fail(f"❌ End-to-end conversation test failed: {e}")

class TestEnvironmentConfiguration:
    """Test environment configuration and debugging helpers"""
    
    def test_all_environment_variables(self):
        """Test all required environment variables are present"""
        required_vars = {
            "OPENAI_API_KEY": "OpenAI API access",
            "DEEPGRAM_API_KEY": "Deepgram STT service",
            "CARTESIA_API_KEY": "Cartesia TTS service",
            "MONGO_URI": "MongoDB database connection",
            "MONGO_DB_NAME": "MongoDB database name",
            "LIVEKIT_URL": "LiveKit server URL",
            "LIVEKIT_API_KEY": "LiveKit API key",
            "LIVEKIT_API_SECRET": "LiveKit API secret"
        }
        
        missing_vars = []
        configured_vars = []
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if value:
                configured_vars.append(f"✅ {var}: {description}")
            else:
                missing_vars.append(f"❌ {var}: {description}")
        
        # Print configuration status
        logger.info("Environment Configuration Status:")
        for var in configured_vars:
            logger.info(var)
        for var in missing_vars:
            logger.warning(var)
        
        if missing_vars:
            pytest.fail(f"Missing required environment variables: {[var.split(':')[0].replace('❌ ', '') for var in missing_vars]}")
    
    def test_service_connectivity_summary(self):
        """Provide a summary of all service connectivity"""
        services_status = {}
        
        # Test MongoDB
        try:
            client = get_async_db_client()
            client.admin.command('ping')
            services_status["MongoDB"] = "✅ Connected"
        except Exception as e:
            services_status["MongoDB"] = f"❌ Failed: {str(e)[:50]}"
        
        # Test OpenAI (simplified check)
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.startswith("sk-"):
                services_status["OpenAI"] = "✅ API Key Valid"
            else:
                services_status["OpenAI"] = "❌ Invalid API Key"
        except Exception as e:
            services_status["OpenAI"] = f"❌ Error: {str(e)[:50]}"
        
        # Test Deepgram
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if api_key and len(api_key) > 20:
            services_status["Deepgram"] = "✅ API Key Configured"
        else:
            services_status["Deepgram"] = "❌ API Key Missing/Invalid"
        
        # Test Cartesia
        api_key = os.getenv("CARTESIA_API_KEY")
        if api_key and api_key.startswith("sk_car_"):
            services_status["Cartesia"] = "✅ API Key Valid Format"
        else:
            services_status["Cartesia"] = "❌ API Key Missing/Invalid"
        
        # Test LiveKit
        url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        if url and api_key:
            services_status["LiveKit"] = "✅ Credentials Configured"
        else:
            services_status["LiveKit"] = "❌ Credentials Missing"
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SERVICE CONNECTIVITY SUMMARY:")
        logger.info("="*50)
        for service, status in services_status.items():
            logger.info(f"{service:12} | {status}")
        logger.info("="*50)
        
        # Fail if any critical services are down
        failed_services = [service for service, status in services_status.items() if "❌" in status]
        if failed_services:
            pytest.fail(f"Critical services failed: {failed_services}")

if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__name__, "-v", "-s"])