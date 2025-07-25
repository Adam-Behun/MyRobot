import time
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class InterruptionType(Enum):
    """Types of conversation interruptions"""
    USER_INTERRUPTION = "user_interruption"        # User speaks while AI is speaking
    CLARIFICATION_REQUEST = "clarification"        # User asks for clarification
    TOPIC_CHANGE = "topic_change"                  # User changes subject
    CORRECTION = "correction"                      # User corrects information
    SILENCE_TIMEOUT = "silence_timeout"            # User doesn't respond

@dataclass
class InterruptionEvent:
    """Represents an interruption event"""
    interruption_type: InterruptionType
    timestamp: float
    user_input: str
    ai_response_truncated: str
    context: Dict[str, Any]

class InterruptionHandler:
    """Handle conversation interruptions and dynamic response adjustments"""
    
    def __init__(self, silence_timeout_ms: float = 3000.0):
        self.silence_timeout_ms = silence_timeout_ms
        self.interruption_history: List[InterruptionEvent] = []
        self.current_ai_response = ""
        self.response_start_time = 0.0
        self.is_ai_speaking = False
        
        # Interruption detection keywords
        self.clarification_keywords = ["sorry", "pardon", "repeat", "again", "clarify"]
        self.correction_keywords = ["no", "wrong", "actually", "correct", "mistake"]
        self.topic_change_keywords = ["but", "however", "instead", "what about", "different"]
    
    def start_ai_response(self, response_text: str):
        """Mark the start of AI response"""
        self.current_ai_response = response_text
        self.response_start_time = time.time()
        self.is_ai_speaking = True
        logger.info(f"AI started speaking: {response_text[:50]}...")
    
    def detect_interruption(self, user_input: str, conversation_context: Dict[str, Any]) -> Optional[InterruptionEvent]:
        """Detect and classify user interruption"""
        
        if not self.is_ai_speaking:
            return None
        
        # Calculate how much of AI response was delivered
        response_duration = time.time() - self.response_start_time
        estimated_completion = len(self.current_ai_response) * 0.1  # Rough estimate: 100ms per character
        
        interruption_type = self._classify_interruption(user_input, response_duration, estimated_completion)
        
        if interruption_type:
            event = InterruptionEvent(
                interruption_type=interruption_type,
                timestamp=time.time(),
                user_input=user_input,
                ai_response_truncated=self.current_ai_response,
                context=conversation_context.copy()
            )
            
            self.interruption_history.append(event)
            self.is_ai_speaking = False
            
            logger.info(f"Interruption detected: {interruption_type.value} - '{user_input[:30]}'")
            return event
        
        return None
    
    def _classify_interruption(self, user_input: str, response_duration: float, estimated_completion: float) -> Optional[InterruptionType]:
        """Classify the type of interruption"""
        
        user_input_lower = user_input.lower()
        
        # Check for topic changes first (more specific phrases)
        if any(keyword in user_input_lower for keyword in self.topic_change_keywords):
            return InterruptionType.TOPIC_CHANGE
        
        # Check for corrections
        if any(keyword in user_input_lower for keyword in self.correction_keywords):
            return InterruptionType.CORRECTION
        
        # Check for clarification requests
        if any(keyword in user_input_lower for keyword in self.clarification_keywords):
            return InterruptionType.CLARIFICATION_REQUEST
        
        # Check for user interruption during AI speech
        if response_duration < estimated_completion * 0.7:  # Interrupted before 70% completion
            return InterruptionType.USER_INTERRUPTION
        
        return None
    
    def handle_interruption(self, event: InterruptionEvent, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interruption and generate appropriate response strategy"""
        
        response_strategy = {
            "should_acknowledge": False,
            "response_prefix": "",
            "adjust_workflow": False,
            "repeat_information": False,
            "clarification_needed": False,
            "continue_from": "current_point"
        }
        
        if event.interruption_type == InterruptionType.CLARIFICATION_REQUEST:
            response_strategy.update({
                "should_acknowledge": True,
                "response_prefix": "Let me clarify that. ",
                "repeat_information": True,
                "clarification_needed": True
            })
            
        elif event.interruption_type == InterruptionType.CORRECTION:
            response_strategy.update({
                "should_acknowledge": True,
                "response_prefix": "Thank you for the correction. ",
                "adjust_workflow": True,
                "continue_from": "corrected_point"
            })
            
        elif event.interruption_type == InterruptionType.TOPIC_CHANGE:
            response_strategy.update({
                "should_acknowledge": True,
                "response_prefix": "I understand. ",
                "adjust_workflow": True,
                "continue_from": "new_topic"
            })
            
        elif event.interruption_type == InterruptionType.USER_INTERRUPTION:
            response_strategy.update({
                "should_acknowledge": True,
                "response_prefix": "Yes, ",
                "continue_from": "user_input"
            })
        
        logger.info(f"Interruption handling strategy: {response_strategy}")
        return response_strategy
    
    def generate_recovery_response(self, event: InterruptionEvent, strategy: Dict[str, Any], workflow_context: Dict[str, Any]) -> str:
        """Generate appropriate recovery response after interruption"""
        
        base_response = strategy.get("response_prefix", "")
        
        if strategy.get("repeat_information"):
            # Extract key information that was being communicated
            key_info = self._extract_key_information(event.ai_response_truncated, workflow_context)
            base_response += f"{key_info} "
        
        if strategy.get("clarification_needed"):
            base_response += "Is there anything specific you'd like me to explain further? "
        
        # Add context-appropriate continuation
        workflow_state = workflow_context.get("workflow_state", "unknown")
        
        if workflow_state == "patient_verification":
            base_response += "What's the patient's full name? "
        elif workflow_state == "procedure_collection":
            base_response += "Could you tell me about the procedure that needs authorization? "
        elif workflow_state == "authorization_decision":
            base_response += "Let me update the authorization status for you. "
        
        return base_response.strip()
    
    def _extract_key_information(self, truncated_response: str, context: Dict[str, Any]) -> str:
        """Extract key information from truncated AI response"""
        
        # Simple extraction - in production, this could use NLP
        sentences = truncated_response.split('. ')
        
        if sentences:
            # Return the first complete sentence as key information
            return sentences[0] + "."
        
        return "As I was saying,"
    
    def adjust_response_speed(self, interruption_frequency: float) -> Dict[str, Any]:
        """Adjust AI response characteristics based on interruption patterns"""
        
        # Calculate recent interruption frequency
        recent_interruptions = [
            event for event in self.interruption_history 
            if time.time() - event.timestamp < 300  # Last 5 minutes
        ]
        
        adjustments = {
            "speech_rate": "normal",
            "pause_duration": 1.0,
            "chunk_size": "normal",
            "confirmation_frequency": "normal"
        }
        
        if interruption_frequency > 0.3:  # High interruption rate
            adjustments.update({
                "speech_rate": "slower",
                "pause_duration": 1.5,
                "chunk_size": "shorter",
                "confirmation_frequency": "higher"
            })
            logger.info("Adjusted to slower, more confirmatory speech pattern")
            
        elif interruption_frequency < 0.1:  # Low interruption rate
            adjustments.update({
                "speech_rate": "normal",
                "pause_duration": 0.8,
                "chunk_size": "longer",
                "confirmation_frequency": "lower"
            })
        
        return adjustments
    
    def check_silence_timeout(self, last_user_input_time: float) -> bool:
        """Check if user has been silent too long"""
        
        if time.time() - last_user_input_time > (self.silence_timeout_ms / 1000):
            logger.info("Silence timeout detected")
            return True
        
        return False
    
    def generate_silence_prompt(self, workflow_context: Dict[str, Any]) -> str:
        """Generate appropriate prompt when user is silent"""
        
        workflow_state = workflow_context.get("workflow_state", "unknown")
        
        prompts = {
            "greeting": "Hello? Are you there? I'm calling about a prior authorization request.",
            "patient_verification": "I need the patient's full name to continue. Are you still there?",
            "procedure_collection": "Could you please tell me about the procedure that needs authorization?",
            "authorization_decision": "I'm ready to process the authorization. Should I continue?",
            "completion": "Is there anything else I can help you with?"
        }
        
        return prompts.get(workflow_state, "Are you still there? How can I help you?")
    
    def get_interruption_analytics(self) -> Dict[str, Any]:
        """Get analytics about interruption patterns"""
        
        if not self.interruption_history:
            return {"status": "no_interruptions"}
        
        type_counts = {}
        for event in self.interruption_history:
            interruption_type = event.interruption_type.value
            type_counts[interruption_type] = type_counts.get(interruption_type, 0) + 1
        
        recent_events = [
            event for event in self.interruption_history 
            if time.time() - event.timestamp < 300
        ]
        
        return {
            "total_interruptions": len(self.interruption_history),
            "recent_interruptions": len(recent_events),
            "interruption_types": type_counts,
            "average_time_between_interruptions": self._calculate_average_time_between_interruptions(),
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }
    
    def _calculate_average_time_between_interruptions(self) -> float:
        """Calculate average time between interruptions"""
        
        if len(self.interruption_history) < 2:
            return 0.0
        
        time_diffs = []
        for i in range(1, len(self.interruption_history)):
            diff = self.interruption_history[i].timestamp - self.interruption_history[i-1].timestamp
            time_diffs.append(diff)
        
        return sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
    
    def reset_for_new_conversation(self):
        """Reset state for new conversation"""
        self.current_ai_response = ""
        self.response_start_time = 0.0
        self.is_ai_speaking = False
        # Keep interruption_history for analytics across conversations