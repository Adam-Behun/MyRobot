import pytest
import asyncio
import time
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization import LatencyOptimizer, LatencyMetrics
from pipeline import HealthcareAIPipeline

class TestLatencyOptimizer:
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes with correct target"""
        optimizer = LatencyOptimizer(target_latency_ms=800.0)
        
        assert optimizer.target_latency_ms == 800.0
        assert len(optimizer.metrics_history) == 0
        assert isinstance(optimizer.response_cache, dict)
        assert isinstance(optimizer.function_cache, dict)
    
    def test_latency_tracking(self):
        """Test latency tracking functionality"""
        optimizer = LatencyOptimizer()
        
        # Track different stages
        optimizer.track_latency("STT", 150.0)
        optimizer.track_latency("LLM", 300.0)
        optimizer.track_latency("TTS", 200.0)
        
        # Check metrics were recorded
        assert len(optimizer.metrics_history) > 0
        latest_metrics = optimizer.metrics_history[-1]
        
        assert latest_metrics.stt_latency == 150.0
        assert latest_metrics.llm_latency == 300.0
        assert latest_metrics.tts_latency == 200.0
        assert latest_metrics.total_latency == 650.0  # Sum of all stages
    
    def test_latency_warning_threshold(self, caplog):
        """Test warning when latency exceeds target"""
        optimizer = LatencyOptimizer(target_latency_ms=500.0)
        
        # Track latency that exceeds target
        optimizer.track_latency("STT", 200.0)
        optimizer.track_latency("LLM", 400.0)  # Total will be 600ms > 500ms target
        
        # Check warning was logged
        assert "Latency exceeded target" in caplog.text
    
    @pytest.mark.asyncio
    async def test_parallel_stt_llm_prep(self):
        """Test parallel STT and LLM context preparation"""
        optimizer = LatencyOptimizer()
        
        # Mock audio data and conversation context
        mock_audio = b"fake_audio_data"
        mock_context = {"workflow_state": "patient_verification"}
        
        # Test parallel processing
        start_time = time.time()
        result = await optimizer.parallel_stt_llm_prep(mock_audio, mock_context)
        end_time = time.time()
        
        # Verify result structure
        assert "transcript" in result
        assert "prepared_context" in result
        assert "system_prompt" in result["prepared_context"]
        assert "functions" in result["prepared_context"]
        
        # Should complete quickly due to mocked operations
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 200  # Should be very fast with mocks
    
    def test_prompt_optimization_caching(self):
        """Test that prompts are cached for performance"""
        optimizer = LatencyOptimizer()
        
        context1 = {"workflow_state": "greeting"}
        context2 = {"workflow_state": "greeting"}  # Same state
        context3 = {"workflow_state": "patient_verification"}  # Different state
        
        # Build prompts
        prompt1 = optimizer._build_optimized_prompt(context1)
        prompt2 = optimizer._build_optimized_prompt(context2)
        prompt3 = optimizer._build_optimized_prompt(context3)
        
        # Same state should return same cached prompt
        assert prompt1 == prompt2
        assert "prompt_greeting" in optimizer.response_cache
        
        # Different state should be different
        assert prompt1 != prompt3
        assert "prompt_patient_verification" in optimizer.response_cache
    
    def test_function_caching(self):
        """Test function result caching"""
        optimizer = LatencyOptimizer()
        
        # Cache a function result
        function_name = "search_patient_by_name"
        args = {"first_name": "John", "last_name": "Doe"}
        result = {"patient_id": "123", "name": "John Doe"}
        
        optimizer.cache_function_result(function_name, args, result)
        
        # Retrieve cached result
        cached_result = optimizer.get_cached_function_result(function_name, args)
        
        assert cached_result == result
        assert len(optimizer.function_cache) == 1
    
    def test_function_cache_expiration(self):
        """Test that function cache expires after time limit"""
        optimizer = LatencyOptimizer()
        
        # Cache a result
        function_name = "search_patient_by_name"
        args = {"first_name": "John", "last_name": "Doe"}
        result = {"patient_id": "123"}
        
        optimizer.cache_function_result(function_name, args, result)
        
        # Manually set old timestamp to simulate expiration
        cache_key = f"{function_name}_{hash(str(sorted(args.items())))}"
        optimizer.function_cache[cache_key]["cached_at"] = time.time() - 400  # 400 seconds ago
        
        # Should return None for expired cache
        cached_result = optimizer.get_cached_function_result(function_name, args)
        assert cached_result is None
        assert cache_key not in optimizer.function_cache  # Should be removed
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        optimizer = LatencyOptimizer(target_latency_ms=800.0)
        
        # Add some test metrics
        for i in range(5):
            optimizer.track_latency("STT", 100.0 + i * 10)
            optimizer.track_latency("LLM", 200.0 + i * 20)
            optimizer.track_latency("TTS", 150.0 + i * 5)
        
        summary = optimizer.get_performance_summary()
        
        # Check summary structure
        assert "target_latency_ms" in summary
        assert "average_total_latency_ms" in summary
        assert "performance_ratio" in summary
        assert summary["target_latency_ms"] == 800.0
        
        # Performance ratio should be total/target
        expected_avg_total = (450.0 + 490.0 + 530.0 + 570.0 + 610.0) / 5  # 530ms
        expected_ratio = expected_avg_total / 800.0
        assert abs(summary["performance_ratio"] - expected_ratio) < 0.01
    
    def test_state_based_optimization(self):
        """Test optimization settings vary by workflow state"""
        optimizer = LatencyOptimizer()
        
        greeting_opts = optimizer.optimize_for_state("greeting")
        verification_opts = optimizer.optimize_for_state("patient_verification")
        
        # Greeting should not use parallel processing or functions
        assert greeting_opts["enable_parallel_processing"] == False
        assert greeting_opts["functions_enabled"] == False
        assert greeting_opts["cache_responses"] == True
        
        # Patient verification should use parallel processing and functions
        assert verification_opts["enable_parallel_processing"] == True
        assert verification_opts["functions_enabled"] == True
        assert "search_patient_by_name" in verification_opts["priority_functions"]
    
    def test_relevant_functions_filtering(self):
        """Test that only relevant functions are returned for each state"""
        optimizer = LatencyOptimizer()
        
        # Test different workflow states
        greeting_context = {"workflow_state": "greeting"}
        verification_context = {"workflow_state": "patient_verification"}
        decision_context = {"workflow_state": "authorization_decision"}
        
        greeting_functions = optimizer._get_relevant_functions(greeting_context)
        verification_functions = optimizer._get_relevant_functions(verification_context)
        decision_functions = optimizer._get_relevant_functions(decision_context)
        
        # Greeting should have no functions
        assert len(greeting_functions) == 0
        
        # Verification should have search function
        assert len(verification_functions) == 1
        assert verification_functions[0]["name"] == "search_patient_by_name"
        
        # Decision should have update function
        assert len(decision_functions) == 1
        assert decision_functions[0]["name"] == "update_prior_auth_status"

class TestPipelineOptimization:
    
    @pytest.mark.asyncio
    async def test_optimized_pipeline_creation(self):
        """Test that optimized pipeline creates successfully"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key',
            'LIVEKIT_URL': 'ws://test:7880'
        }):
            pipeline = HealthcareAIPipeline(session_id="test_optimization")
            
            # Mock external services
            with patch('pipeline.LiveKitTransport'), \
                 patch('pipeline.DeepgramSTTService'), \
                 patch('pipeline.OpenAILLMService'), \
                 patch('pipeline.CartesiaTTSService'):
                
                result = pipeline.create_pipeline()
                
                assert result is not None
                assert pipeline.optimizer is not None
                assert pipeline.interruption_handler is not None
                assert pipeline.audio_processor is not None
    
    @pytest.mark.asyncio
    async def test_audio_processing_wrapper(self):
        """Test audio processing wrapper functionality"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Create mock STT service
            mock_stt = Mock()
            mock_stt.process_frame = AsyncMock()
            
            # Apply audio processing wrapper
            wrapped_stt = pipeline._add_audio_processing_wrapper(mock_stt)
            
            # Create mock frame with audio data
            mock_frame = Mock()
            mock_frame.audio = np.random.rand(1600).astype(np.float32).tobytes()  # 100ms of 16kHz audio
            
            # Test processing
            await wrapped_stt.process_frame(mock_frame)
            
            # Verify original STT was called
            mock_stt.process_frame.assert_called_once()
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions generation"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Mock performance metrics that exceed target
            pipeline.optimizer.track_latency("STT", 200.0)
            pipeline.optimizer.track_latency("LLM", 600.0)  # High latency
            pipeline.optimizer.track_latency("TTS", 300.0)
            
            suggestions = pipeline.get_optimization_suggestions()
            
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
            
            # Should suggest LLM optimization due to high latency
            suggestion_text = " ".join(suggestions).lower()
            assert "latency" in suggestion_text or "llm" in suggestion_text
    
    @pytest.mark.asyncio
    async def test_function_call_caching(self):
        """Test that function calls are cached for performance"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Mock function registry
            with patch('pipeline.FUNCTION_REGISTRY') as mock_registry:
                mock_function = Mock(return_value={"patient_id": "123", "name": "John Doe"})
                mock_registry.__getitem__.return_value = mock_function
                mock_registry.__contains__.return_value = True
                
                # First call should execute function
                result1 = pipeline._handle_function_call(
                    "search_patient_by_name", 
                    {"first_name": "John", "last_name": "Doe"}
                )
                
                # Second call with same args should use cache
                result2 = pipeline._handle_function_call(
                    "search_patient_by_name", 
                    {"first_name": "John", "last_name": "Doe"}
                )
                
                # Function should only be called once due to caching
                assert mock_function.call_count == 1
                assert result1 == result2
    
    def test_conversation_state_with_metrics(self):
        """Test that conversation state includes performance metrics"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Add some metrics
            pipeline.optimizer.track_latency("STT", 100.0)
            pipeline.optimizer.track_latency("LLM", 300.0)
            
            state = pipeline.get_conversation_state()
            
            # Verify performance metrics are included
            assert "performance_metrics" in state
            assert "interruption_analytics" in state
            assert "audio_quality" in state
            
            perf_metrics = state["performance_metrics"]
            assert "average_total_latency_ms" in perf_metrics
            assert "target_latency_ms" in perf_metrics

if __name__ == "__main__":
    pytest.main([__name__, "-v"])