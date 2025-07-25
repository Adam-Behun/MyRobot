import logging
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Simple conversation states for prior authorization workflow"""
    GREETING = "greeting"
    PATIENT_VERIFICATION = "patient_verification"
    PROCEDURE_COLLECTION = "procedure_collection"
    AUTHORIZATION_DECISION = "authorization_decision"
    COMPLETION = "completion"

class HealthcareWorkflow:
    """Simple workflow manager for prior authorization conversations"""
    
    def __init__(self):
        self.state = ConversationState.GREETING
        self.patient_data = None
        self.collected_info = {}
        
    def get_system_prompt(self) -> str:
        """Get system prompt based on current conversation state"""
        
        base_prompt = """You are MyRobot, a healthcare AI assistant for prior authorization calls. 
You are professional, empathetic, and HIPAA-compliant. Keep responses concise for voice interaction."""
        
        state_prompts = {
            ConversationState.GREETING: """
Current task: Greet the caller and ask for the patient's name to begin verification.
Say something like: "Hello, I'm MyRobot calling about a prior authorization request. May I have the patient's full name please?"
""",
            
            ConversationState.PATIENT_VERIFICATION: """
Current task: You have the patient's name. Use the search_patient_by_name function to find them in the database.
If found, verify a few details like date of birth. If not found, politely ask them to double-check the spelling.
Once verified, move to collecting procedure information.
""",
            
            ConversationState.PROCEDURE_COLLECTION: """
Current task: Collect information about the medical procedure requiring authorization.
Ask about: procedure type, medical necessity, symptoms, and any supporting documentation.
Keep questions natural and conversational.
""",
            
            ConversationState.AUTHORIZATION_DECISION: """
Current task: Review the collected information and make an authorization decision.
Use update_prior_auth_status function to update the patient's record.
Inform the caller of the decision and next steps.
""",
            
            ConversationState.COMPLETION: """
Current task: Wrap up the call professionally.
Provide any reference numbers, next steps, or contact information if needed.
Thank them for their time.
"""
        }
        
        return base_prompt + state_prompts.get(self.state, "")
    
    def advance_state(self, trigger: str = None) -> ConversationState:
        """Advance to next conversation state based on trigger"""
        
        state_transitions = {
            ConversationState.GREETING: ConversationState.PATIENT_VERIFICATION,
            ConversationState.PATIENT_VERIFICATION: ConversationState.PROCEDURE_COLLECTION,
            ConversationState.PROCEDURE_COLLECTION: ConversationState.AUTHORIZATION_DECISION,
            ConversationState.AUTHORIZATION_DECISION: ConversationState.COMPLETION,
            ConversationState.COMPLETION: ConversationState.COMPLETION  # Stay at completion
        }
        
        previous_state = self.state
        self.state = state_transitions.get(self.state, self.state)
        
        logger.info(f"Workflow state: {previous_state.value} -> {self.state.value}")
        return self.state
    
    def update_patient_data(self, patient_data: Dict[str, Any]):
        """Update stored patient data"""
        self.patient_data = patient_data
        logger.info(f"Updated patient data for: {patient_data.get('patient_name', 'Unknown')}")
    
    def add_collected_info(self, key: str, value: Any):
        """Add collected information to workflow state"""
        self.collected_info[key] = value
        logger.info(f"Added to workflow: {key} = {value}")
    
    def get_workflow_context(self) -> Dict[str, Any]:
        """Get current workflow context for LLM"""
        return {
            "current_state": self.state.value,
            "patient_data": self.patient_data,
            "collected_info": self.collected_info,
            "next_action": self._get_next_action()
        }
    
    def _get_next_action(self) -> str:
        """Get description of what should happen next"""
        actions = {
            ConversationState.GREETING: "Ask for patient's full name",
            ConversationState.PATIENT_VERIFICATION: "Search for patient and verify details",
            ConversationState.PROCEDURE_COLLECTION: "Collect procedure and medical information",
            ConversationState.AUTHORIZATION_DECISION: "Make authorization decision and update status",
            ConversationState.COMPLETION: "End call professionally"
        }
        return actions.get(self.state, "Continue conversation")
    
    def should_use_function(self, user_message: str) -> Optional[str]:
        """Determine if a function should be called based on conversation state and user input"""
        
        # Simple keyword-based function triggering
        if self.state == ConversationState.PATIENT_VERIFICATION:
            if any(word in user_message.lower() for word in ["name is", "patient is", "my name"]):
                return "search_patient_by_name"
        
        if self.state == ConversationState.AUTHORIZATION_DECISION:
            if self.patient_data and "collected_info" in str(self.collected_info):
                return "update_prior_auth_status"
        
        return None
    
    def reset(self):
        """Reset workflow to initial state"""
        self.state = ConversationState.GREETING
        self.patient_data = None
        self.collected_info = {}
        logger.info("Workflow reset to initial state")