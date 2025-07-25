import pytest
import time
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interruption_handler import InterruptionHandler, InterruptionType, InterruptionEvent
from pipeline import HealthcareAIPipeline
from workflow import HealthcareWorkflow, ConversationState

class TestInterruptionHandler:
    
    def test_handler_initialization(self):
        """Test interruption handler initializes correctly"""
        handler = InterruptionHandler(silence_timeout_ms=3000.0)
        
        assert handler.silence_timeout_ms == 3000.0
        assert len(handler.interruption_history) == 0
        assert handler.current_ai_response == ""
        assert handler.is_ai_speaking == False
    
    def test_ai_response_tracking(self):
        """Test AI response start/stop tracking"""
        handler = InterruptionHandler()
        
        # Start AI response
        test_response = "Hello, I'm MyRobot calling about prior authorization"
        handler.start_ai_response(test_response)
        
        assert handler.current_ai_response == test_response
        assert handler.is_ai_speaking == True
        assert handler.response_start_time > 0
    
    def test_clarification_interruption_detection(self):
        """Test detection of clarification requests"""
        handler = InterruptionHandler()
        
        # Start AI response
        handler.start_ai_response("I need to verify the patient information")
        
        # User asks for clarification
        context = {"workflow_state": "patient_verification"}
        interruption = handler.detect_interruption("Sorry, what did you say?", context)
        
        assert interruption is not None
        assert interruption.interruption_type == InterruptionType.CLARIFICATION_REQUEST
        assert interruption.user_input == "Sorry, what did you say?"
        assert handler.is_ai_speaking == False  # Should stop AI speaking
    
    def test_correction_interruption_detection(self):
        """Test detection of user corrections"""
        handler = InterruptionHandler()
        
        handler.start_ai_response("So the patient is Jane Smith, correct?")
        
        context = {"workflow_state": "patient_verification"}
        interruption = handler.detect_interruption("No, it's John Smith", context)
        
        assert interruption is not None
        assert interruption.interruption_type == InterruptionType.CORRECTION
        assert "No, it's John Smith" in interruption.user_input
    
    def test_topic_change_interruption_detection(self):
        """Test detection of topic changes"""
        handler = InterruptionHandler()
        
        handler.start_ai_response("Now let's discuss the procedure details")
        
        context = {"workflow_state": "procedure_collection"}
        interruption = handler.detect_interruption("But first, what about the insurance coverage?", context)
        
        assert interruption is not None
        assert interruption.interruption_type == InterruptionType.TOPIC_CHANGE
    
    def test_user_interruption_detection(self):
        """Test detection of general user interruptions"""
        handler = InterruptionHandler()
        
        # Start long AI response
        long_response = "I need to collect several pieces of information from you including the patient's full name, date of birth, insurance information, and the specific procedure that requires authorization"
        handler.start_ai_response(long_response)
        
        # User interrupts early (simulated by short delay)
        time.sleep(0.1)  # Brief pause to simulate interruption timing
        context = {"workflow_state": "procedure_collection"}
        interruption = handler.detect_interruption("Yes, I understand", context)
        
        assert interruption is not None
        assert interruption.interruption_type == InterruptionType.USER_INTERRUPTION
    
    def test_no_interruption_when_not_speaking(self):
        """Test that no interruption is detected when AI isn't speaking"""
        handler = InterruptionHandler()
        
        # AI is not speaking
        assert handler.is_ai_speaking == False
        
        context = {"workflow_state": "greeting"}
        interruption = handler.detect_interruption("Hello there", context)
        
        assert interruption is None
    
    def test_interruption_handling_strategies(self):
        """Test different interruption handling strategies"""
        handler = InterruptionHandler()
        
        # Test clarification strategy
        clarification_event = InterruptionEvent(
            interruption_type=InterruptionType.CLARIFICATION_REQUEST,
            timestamp=time.time(),
            user_input="Can you repeat that?",
            ai_response_truncated="I was saying...",
            context={"workflow_state": "patient_verification"}
        )
        
        strategy = handler.handle_interruption(clarification_event, {"workflow_state": "patient_verification"})
        
        assert strategy["should_acknowledge"] == True
        assert strategy["response_prefix"] == "Let me clarify that. "
        assert strategy["repeat_information"] == True
        assert strategy["clarification_needed"] == True
        
        # Test correction strategy
        correction_event = InterruptionEvent(
            interruption_type=InterruptionType.CORRECTION,
            timestamp=time.time(),
            user_input="No, that's wrong",
            ai_response_truncated="The patient is...",
            context={"workflow_state": "patient_verification"}
        )
        
        correction_strategy = handler.handle_interruption(correction_event, {"workflow_state": "patient_verification"})
        
        assert correction_strategy["should_acknowledge"] == True
        assert correction_strategy["response_prefix"] == "Thank you for the correction. "
        assert correction_strategy["adjust_workflow"] == True
    
    def test_recovery_response_generation(self):
        """Test generation of appropriate recovery responses"""
        handler = InterruptionHandler()
        
        # Clarification interruption
        event = InterruptionEvent(
            interruption_type=InterruptionType.CLARIFICATION_REQUEST,
            timestamp=time.time(),
            user_input="What information do you need?",
            ai_response_truncated="I need to verify the patient details",
            context={"workflow_state": "patient_verification"}
        )
        
        strategy = {
            "should_acknowledge": True,
            "response_prefix": "Let me clarify that. ",
            "repeat_information": True,
            "clarification_needed": True
        }
        
        workflow_context = {"workflow_state": "patient_verification"}
        response = handler.generate_recovery_response(event, strategy, workflow_context)
        
        assert "Let me clarify that" in response
        assert "patient's full name" in response.lower()
        assert len(response) > 10  # Should be a substantial response
    
    def test_silence_timeout_detection(self):
        """Test silence timeout detection"""
        handler = InterruptionHandler(silence_timeout_ms=1000.0)  # 1 second timeout
        
        # Simulate user input time
        old_time = time.time() - 2.0  # 2 seconds ago
        current_time = time.time()
        
        is_timeout = handler.check_silence_timeout(old_time)
        assert is_timeout == True
        
        # Recent input should not timeout
        is_timeout_recent = handler.check_silence_timeout(current_time)
        assert is_timeout_recent == False
    
    def test_silence_prompt_generation(self):
        """Test appropriate prompts for different workflow states during silence"""
        handler = InterruptionHandler()
        
        # Test different workflow states
        states_and_prompts = [
            ("greeting", "prior authorization"),
            ("patient_verification", "patient's full name"),
            ("procedure_collection", "procedure that needs authorization"),
            ("authorization_decision", "process the authorization"),
            ("completion", "anything else")
        ]
        
        for state, expected_content in states_and_prompts:
            context = {"workflow_state": state}
            prompt = handler.generate_silence_prompt(context)
            
            assert expected_content.lower() in prompt.lower()
            assert len(prompt) > 10  # Should be substantial
    
    def test_response_speed_adjustment(self):
        """Test response speed adjustments based on interruption frequency"""
        handler = InterruptionHandler()
        
        # Add multiple interruptions to simulate high interruption rate
        for i in range(5):
            event = InterruptionEvent(
                interruption_type=InterruptionType.USER_INTERRUPTION,
                timestamp=time.time(),
                user_input=f"Interruption {i}",
                ai_response_truncated="AI response",
                context={}
            )
            handler.interruption_history.append(event)
        
        adjustments = handler.adjust_response_speed(0.4)  # 40% interruption rate
        
        assert adjustments["speech_rate"] == "slower"
        assert adjustments["pause_duration"] == 1.5
        assert adjustments["chunk_size"] == "shorter"
        assert adjustments["confirmation_frequency"] == "higher"
    
    def test_interruption_analytics(self):
        """Test interruption analytics generation"""
        handler = InterruptionHandler()
        
        # Add various types of interruptions
        interruption_types = [
            InterruptionType.CLARIFICATION_REQUEST,
            InterruptionType.CORRECTION,
            InterruptionType.USER_INTERRUPTION,
            InterruptionType.CLARIFICATION_REQUEST
        ]
        
        for i, int_type in enumerate(interruption_types):
            event = InterruptionEvent(
                interruption_type=int_type,
                timestamp=time.time() - (len(interruption_types) - i),  # Spread over time
                user_input=f"Test input {i}",
                ai_response_truncated="AI response",
                context={}
            )
            handler.interruption_history.append(event)
        
        analytics = handler.get_interruption_analytics()
        
        assert analytics["total_interruptions"] == 4
        assert analytics["interruption_types"]["clarification"] == 2  # Two clarification requests
        assert analytics["interruption_types"]["correction"] == 1
        assert analytics["interruption_types"]["user_interruption"] == 1
        assert analytics["most_common_type"] == "clarification"

class TestPipelineInterruptionIntegration:
    
    @pytest.mark.asyncio
    async def test_interruption_wrapper_integration(self):
        """Test interruption handling integration in pipeline"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Create mock TTS service
            mock_tts = Mock()
            mock_tts.process_frame = AsyncMock()
            
            # Apply interruption wrapper
            wrapped_tts = pipeline._add_interruption_wrapper(mock_tts)
            
            # Create mock frame
            mock_frame = Mock()
            mock_frame.text = "Test response"
            
            # Test processing
            await wrapped_tts.process_frame(mock_frame)
            
            # Verify original TTS was called
            mock_tts.process_frame.assert_called_once()
    
    def test_workflow_interruption_integration(self):
        """Test that interruptions properly integrate with workflow"""
        workflow = HealthcareWorkflow()
        handler = InterruptionHandler()
        
        # Start in patient verification state
        workflow.state = ConversationState.PATIENT_VERIFICATION
        
        # Simulate interruption during patient verification
        handler.start_ai_response("Can you spell the patient's last name?")
        
        context = workflow.get_workflow_context()
        interruption = handler.detect_interruption("Sorry, can you repeat that?", context)
        
        assert interruption is not None
        assert interruption.interruption_type == InterruptionType.CLARIFICATION_REQUEST
        
        # Generate recovery response
        strategy = handler.handle_interruption(interruption, context)
        recovery_response = handler.generate_recovery_response(interruption, strategy, context)
        
        # Should stay in same workflow state for clarification
        assert "patient's full name" in recovery_response.lower()
        assert "clarify" in recovery_response.lower()
    
    def test_conversation_flow_preservation(self):
        """Test that interruptions don't break conversation flow"""
        workflow = HealthcareWorkflow()
        handler = InterruptionHandler()
        
        # Test progression through workflow states with interruptions
        states_to_test = [
            (ConversationState.GREETING, "Hello, this is about John Doe"),
            (ConversationState.PATIENT_VERIFICATION, "Sorry, what was that?"),
            (ConversationState.PATIENT_VERIFICATION, "John Doe, D-O-E"),
            (ConversationState.PROCEDURE_COLLECTION, "He needs an MRI scan"),
            (ConversationState.AUTHORIZATION_DECISION, "Yes, please approve it")
        ]
        
        for expected_state, user_input in states_to_test:
            workflow.state = expected_state
            context = workflow.get_workflow_context()
            
            # Simulate AI speaking before user input
            handler.start_ai_response("AI is asking something...")
            
            # Detect potential interruption
            interruption = handler.detect_interruption(user_input, context)
            
            if interruption:
                # Handle interruption gracefully
                strategy = handler.handle_interruption(interruption, context)
                recovery = handler.generate_recovery_response(interruption, strategy, context)
                
                # Recovery should be appropriate for the workflow state
                assert len(recovery) > 0
                assert recovery != "Error"
            
            # Workflow should continue functioning
            assert workflow.get_workflow_context() is not None
    
    def test_high_interruption_rate_handling(self):
        """Test system behavior under high interruption rates"""
        handler = InterruptionHandler()
        
        # Simulate many interruptions in short time
        for i in range(10):
            handler.start_ai_response(f"AI response {i}")
            
            interruption = handler.detect_interruption(
                f"User interruption {i}", 
                {"workflow_state": "procedure_collection"}
            )
            
            if interruption:
                strategy = handler.handle_interruption(interruption, {"workflow_state": "procedure_collection"})
                recovery = handler.generate_recovery_response(interruption, strategy, {"workflow_state": "procedure_collection"})
                
                # Should always generate valid recovery
                assert len(recovery) > 0
                assert "Error" not in recovery
        
        # System should adapt to high interruption rate
        adjustments = handler.adjust_response_speed(0.8)  # 80% interruption rate
        
        assert adjustments["speech_rate"] == "slower"
        assert adjustments["confirmation_frequency"] == "higher"
    
    def test_interruption_memory_integration(self):
        """Test that interruptions are properly tracked in conversation memory"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'DEEPGRAM_API_KEY': 'test-key',
            'CARTESIA_API_KEY': 'test-key'
        }):
            pipeline = HealthcareAIPipeline()
            
            # Add user message that could be an interruption
            pipeline.memory.add_message("user", "Sorry, can you repeat that?")
            pipeline.memory.add_message("assistant", "Let me clarify that for you...")
            
            # Check that conversation memory maintains context
            messages = pipeline.memory.get_messages_for_llm()
            
            assert len(messages) >= 2
            assert any("repeat" in msg.get("content", "").lower() for msg in messages)
            assert any("clarify" in msg.get("content", "").lower() for msg in messages)
    
    def test_no_infinite_interruption_loops(self):
        """Test that interruption handling doesn't create infinite loops"""
        handler = InterruptionHandler()
        workflow = HealthcareWorkflow()
        
        # Simulate rapid back-and-forth that could create loops
        context = {"workflow_state": "patient_verification"}
        
        for attempt in range(20):  # Many attempts
            handler.start_ai_response("Can you please provide the patient name?")
            
            interruption = handler.detect_interruption("What?", context)
            
            if interruption:
                strategy = handler.handle_interruption(interruption, context)
                recovery = handler.generate_recovery_response(interruption, strategy, context)
                
                # Each response should be different and meaningful
                assert len(recovery) > 10
                assert "Error" not in recovery
                
                # Should not get stuck in same response
                if attempt > 0:
                    # Allow some repetition but ensure system remains responsive
                    assert len(handler.interruption_history) == attempt + 1
        
        # System should continue functioning after many interruptions
        final_analytics = handler.get_interruption_analytics()
        assert final_analytics["total_interruptions"] > 0
        assert final_analytics["most_common_type"] == "clarification"

if __name__ == "__main__":
    pytest.main([__name__, "-v"])