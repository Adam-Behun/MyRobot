import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Simple in-memory conversation storage for MVP"""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self.session_start = datetime.utcnow()
        self.metadata = {}
    
    def add_message(self, role: str, content: str, function_call: Optional[Dict] = None):
        """Add a message to conversation history"""
        
        message = {
            "role": role,  # "user", "assistant", "system", "function"
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if function_call:
            message["function_call"] = function_call
        
        self.messages.append(message)
        
        # Keep only recent messages to avoid context overflow
        if len(self.messages) > self.max_messages:
            # Keep first message (system prompt) and recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
        
        logger.info(f"Added {role} message: {content[:50]}...")
    
    def add_system_message(self, content: str):
        """Add or update system message (always at the beginning)"""
        system_message = {
            "role": "system",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Replace existing system message or add as first message
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = system_message
        else:
            self.messages.insert(0, system_message)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for OpenAI API"""
        
        # Return messages in OpenAI format (role, content)
        formatted_messages = []
        
        for msg in self.messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Add function call if present
            if "function_call" in msg:
                formatted_msg["function_call"] = msg["function_call"]
                
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def get_recent_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context as a string"""
        
        recent_messages = self.messages[-num_messages:] if len(self.messages) > num_messages else self.messages
        
        context_parts = []
        for msg in recent_messages:
            if msg["role"] != "system":  # Skip system messages in context
                context_parts.append(f"{msg['role'].title()}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def update_metadata(self, key: str, value: Any):
        """Store metadata about the conversation"""
        self.metadata[key] = value
        logger.info(f"Updated conversation metadata: {key} = {value}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        
        return {
            "session_start": self.session_start.isoformat(),
            "duration_minutes": (datetime.utcnow() - self.session_start).total_seconds() / 60,
            "message_count": len(self.messages),
            "metadata": self.metadata,
            "last_user_message": self._get_last_message_by_role("user"),
            "last_assistant_message": self._get_last_message_by_role("assistant")
        }
    
    def _get_last_message_by_role(self, role: str) -> Optional[str]:
        """Get the most recent message from a specific role"""
        for msg in reversed(self.messages):
            if msg["role"] == role:
                return msg["content"]
        return None
    
    def clear(self):
        """Clear conversation history but keep metadata"""
        self.messages = []
        self.session_start = datetime.utcnow()
        logger.info("Conversation memory cleared")
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export full conversation for persistence (if needed later)"""
        return {
            "messages": self.messages,
            "metadata": self.metadata,
            "session_start": self.session_start.isoformat(),
            "exported_at": datetime.utcnow().isoformat()
        }

class MemoryManager:
    """Simple manager for conversation sessions"""
    
    def __init__(self):
        self.active_conversations: Dict[str, ConversationMemory] = {}
    
    def get_or_create_conversation(self, session_id: str) -> ConversationMemory:
        """Get existing conversation or create new one"""
        
        if session_id not in self.active_conversations:
            self.active_conversations[session_id] = ConversationMemory()
            logger.info(f"Created new conversation memory for session: {session_id}")
        
        return self.active_conversations[session_id]
    
    def end_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End conversation and return summary"""
        
        if session_id in self.active_conversations:
            conversation = self.active_conversations[session_id]
            summary = conversation.get_conversation_summary()
            del self.active_conversations[session_id]
            logger.info(f"Ended conversation session: {session_id}")
            return summary
        
        return None
    
    def get_active_session_count(self) -> int:
        """Get number of active conversations"""
        return len(self.active_conversations)