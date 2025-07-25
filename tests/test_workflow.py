import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import HealthcareWorkflow, ConversationState
from memory import ConversationMemory, MemoryManager

class TestHealthcareWorkflow:
    
    def test_workflow_initialization(self):
        """Test workflow initializes in correct state"""
        workflow = HealthcareWorkflow()
        
        assert workflow.state == ConversationState.GREETING
        assert workflow.patient_data is None
        assert workflow.collected_info == {}
    
    def test_get_system_prompt_greeting(self):
        """Test system prompt for greeting state"""
        workflow = HealthcareWorkflow()
        prompt = workflow.get_system_prompt()
        
        assert "MyRobot" in prompt
        assert "patient's name" in prompt.lower()
        assert workflow.state == ConversationState.GREETING
    
    def test_get_system_prompt_verification(self):
        """Test system prompt for patient verification state"""
        workflow = HealthcareWorkflow()
        workflow.state = ConversationState.PATIENT_VERIFICATION
        prompt = workflow.get_system_prompt()
        
        assert "search_patient_by_name" in prompt
        assert "verify" in prompt.lower()
    
    def test_advance_state_progression(self):
        """Test state progression through workflow"""
        workflow = HealthcareWorkflow()
        
        # Test each state transition
        assert workflow.state == ConversationState.GREETING
        
        next_state = workflow.advance_state()
        assert next_state == ConversationState.PATIENT_VERIFICATION
        assert workflow.state == ConversationState.PATIENT_VERIFICATION
        
        next_state = workflow.advance_state()
        assert next_state == ConversationState.PROCEDURE_COLLECTION
        
        next_state = workflow.advance_state()
        assert next_state == ConversationState.AUTHORIZATION_DECISION
        
        next_state = workflow.advance_state()
        assert next_state == ConversationState.COMPLETION
        
        # Should stay at completion
        next_state = workflow.advance_state()
        assert next_state == ConversationState.COMPLETION
    
    def test_update_patient_data(self):
        """Test updating patient data"""
        workflow = HealthcareWorkflow()
        patient_data = {
            "_id": "507f1f77bcf86cd799439011",
            "patient_name": "John Doe",
            "date_of_birth": "1980-01-01"
        }
        
        workflow.update_patient_data(patient_data)
        
        assert workflow.patient_data == patient_data
        assert workflow.patient_data["patient_name"] == "John Doe"
    
    def test_add_collected_info(self):
        """Test adding collected information"""
        workflow = HealthcareWorkflow()
        
        workflow.add_collected_info("procedure_type", "MRI")
        workflow.add_collected_info("symptoms", "Back pain")
        
        assert workflow.collected_info["procedure_type"] == "MRI"
        assert workflow.collected_info["symptoms"] == "Back pain"
        assert len(workflow.collected_info) == 2
    
    def test_get_workflow_context(self):
        """Test getting workflow context"""
        workflow = HealthcareWorkflow()
        workflow.update_patient_data({"patient_name": "John Doe"})
        workflow.add_collected_info("test_key", "test_value")
        
        context = workflow.get_workflow_context()
        
        assert context["current_state"] == "greeting"
        assert context["patient_data"]["patient_name"] == "John Doe"
        assert context["collected_info"]["test_key"] == "test_value"
        assert "next_action" in context
    
    def test_should_use_function_verification_state(self):
        """Test function suggestion in verification state"""
        workflow = HealthcareWorkflow()
        workflow.state = ConversationState.PATIENT_VERIFICATION
        
        # Test positive cases
        assert workflow.should_use_function("My name is John Doe") == "search_patient_by_name"
        assert workflow.should_use_function("The patient is Jane Smith") == "search_patient_by_name"
        
        # Test negative case
        assert workflow.should_use_function("Hello there") is None
    
    def test_should_use_function_decision_state(self):
        """Test function suggestion in decision state"""
        workflow = HealthcareWorkflow()
        workflow.state = ConversationState.AUTHORIZATION_DECISION
        workflow.patient_data = {"patient_name": "John Doe"}
        workflow.collected_info = {"collected_info": "procedure data"}  # Match the logic in workflow.py
        
        result = workflow.should_use_function("Let's update the status")
        assert result == "update_prior_auth_status"
    
    def test_reset_workflow(self):
        """Test workflow reset functionality"""
        workflow = HealthcareWorkflow()
        
        # Modify workflow state
        workflow.state = ConversationState.COMPLETION
        workflow.update_patient_data({"patient_name": "John Doe"})
        workflow.add_collected_info("test", "value")
        
        # Reset workflow
        workflow.reset()
        
        assert workflow.state == ConversationState.GREETING
        assert workflow.patient_data is None
        assert workflow.collected_info == {}

class TestConversationMemory:
    
    def test_memory_initialization(self):
        """Test memory initializes correctly"""
        memory = ConversationMemory()
        
        assert len(memory.messages) == 0
        assert memory.max_messages == 20
        assert memory.metadata == {}
    
    def test_add_message(self):
        """Test adding messages to memory"""
        memory = ConversationMemory()
        
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        assert len(memory.messages) == 2
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "Hello"
        assert memory.messages[1]["role"] == "assistant"
        assert memory.messages[1]["content"] == "Hi there!"
        assert "timestamp" in memory.messages[0]
    
    def test_add_system_message(self):
        """Test adding and updating system messages"""
        memory = ConversationMemory()
        
        memory.add_system_message("You are a helpful assistant")
        assert len(memory.messages) == 1
        assert memory.messages[0]["role"] == "system"
        
        # Update system message
        memory.add_system_message("You are a healthcare assistant")
        assert len(memory.messages) == 1
        assert memory.messages[0]["content"] == "You are a healthcare assistant"
    
    def test_get_messages_for_llm(self):
        """Test formatting messages for LLM"""
        memory = ConversationMemory()
        
        memory.add_system_message("System prompt")
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi!")
        
        llm_messages = memory.get_messages_for_llm()
        
        assert len(llm_messages) == 3
        assert all("role" in msg and "content" in msg for msg in llm_messages)
        assert llm_messages[0]["role"] == "system"
        assert llm_messages[1]["role"] == "user"
        assert llm_messages[2]["role"] == "assistant"
    
    def test_message_limit(self):
        """Test message limit enforcement"""
        memory = ConversationMemory(max_messages=3)
        
        memory.add_system_message("System")
        memory.add_message("user", "Message 1")
        memory.add_message("assistant", "Response 1")
        memory.add_message("user", "Message 2")
        memory.add_message("assistant", "Response 2")
        
        # Should keep system message + recent messages
        assert len(memory.messages) == 3
        assert memory.messages[0]["role"] == "system"
        assert memory.messages[-1]["content"] == "Response 2"
    
    def test_get_recent_context(self):
        """Test getting recent conversation context"""
        memory = ConversationMemory()
        
        memory.add_system_message("System prompt")
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        context = memory.get_recent_context(2)
        
        assert "User: Hello" in context
        assert "Assistant: Hi there!" in context
        assert "System" not in context  # System messages excluded
    
    def test_conversation_summary(self):
        """Test conversation summary generation"""
        memory = ConversationMemory()
        
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi!")
        memory.update_metadata("patient_name", "John Doe")
        
        summary = memory.get_conversation_summary()
        
        assert "message_count" in summary
        assert summary["message_count"] == 2
        assert "metadata" in summary
        assert summary["metadata"]["patient_name"] == "John Doe"
        assert "last_user_message" in summary
        assert summary["last_user_message"] == "Hello"
    
    def test_clear_memory(self):
        """Test clearing conversation memory"""
        memory = ConversationMemory()
        
        memory.add_message("user", "Hello")
        memory.update_metadata("test", "value")
        
        memory.clear()
        
        assert len(memory.messages) == 0
        # Metadata should be preserved
        assert memory.metadata["test"] == "value"

class TestMemoryManager:
    
    def test_memory_manager_initialization(self):
        """Test memory manager initializes correctly"""
        manager = MemoryManager()
        
        assert len(manager.active_conversations) == 0
        assert manager.get_active_session_count() == 0
    
    def test_get_or_create_conversation(self):
        """Test creating and retrieving conversations"""
        manager = MemoryManager()
        
        # Create new conversation
        memory1 = manager.get_or_create_conversation("session1")
        assert isinstance(memory1, ConversationMemory)
        assert manager.get_active_session_count() == 1
        
        # Get existing conversation
        memory2 = manager.get_or_create_conversation("session1")
        assert memory1 is memory2  # Should be same instance
        assert manager.get_active_session_count() == 1
        
        # Create different conversation
        memory3 = manager.get_or_create_conversation("session2")
        assert memory3 is not memory1
        assert manager.get_active_session_count() == 2
    
    def test_end_conversation(self):
        """Test ending conversations"""
        manager = MemoryManager()
        
        # Create conversation
        memory = manager.get_or_create_conversation("session1")
        memory.add_message("user", "Hello")
        
        # End conversation
        summary = manager.end_conversation("session1")
        
        assert summary is not None
        assert "message_count" in summary
        assert manager.get_active_session_count() == 0
        
        # Try to end non-existent conversation
        result = manager.end_conversation("nonexistent")
        assert result is None

if __name__ == "__main__":
    pytest.main([__name__])