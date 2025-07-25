import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import uuid

from pipeline import HealthcareAIPipeline
from models import get_db_client, PatientRecord
from memory import MemoryManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare AI Agent", version="1.0.0")

# Global instances
db_client = get_db_client()
patient_db = PatientRecord(db_client)
memory_manager = MemoryManager()
active_pipelines = {}

class CallRequest(BaseModel):
    room_url: str
    token: str
    session_id: str = None

class ConversationStateResponse(BaseModel):
    session_id: str
    workflow_state: str
    message_count: int
    patient_found: bool = False

@app.post("/start-call")
async def start_call(request: CallRequest):
    """Start a healthcare AI call session"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Starting call for session: {session_id}")
        
        # Create pipeline for this session
        pipeline = HealthcareAIPipeline(session_id=session_id)
        active_pipelines[session_id] = pipeline
        
        # Start the pipeline
        await pipeline.run(request.room_url, request.token)
        
        return {"status": "success", "session_id": session_id, "message": "Call started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-state/{session_id}")
async def get_conversation_state(session_id: str):
    """Get current conversation state for a session"""
    try:
        if session_id not in active_pipelines:
            raise HTTPException(status_code=404, detail="Session not found")
        
        pipeline = active_pipelines[session_id]
        state = pipeline.get_conversation_state()
        
        return {
            "session_id": session_id,
            "workflow_state": state["workflow_state"],
            "message_count": state["conversation_summary"]["message_count"],
            "patient_found": state["workflow_context"]["patient_data"] is not None,
            "details": state
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end-call/{session_id}")
async def end_call(session_id: str):
    """End a call session and cleanup"""
    try:
        if session_id in active_pipelines:
            pipeline = active_pipelines[session_id]
            final_state = pipeline.get_conversation_state()
            
            # Cleanup
            del active_pipelines[session_id]
            memory_manager.end_conversation(session_id)
            
            logger.info(f"Ended call session: {session_id}")
            return {"status": "success", "final_state": final_state}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
        
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-sessions")
async def get_active_sessions():
    """Get list of active call sessions"""
    return {
        "active_sessions": list(active_pipelines.keys()),
        "session_count": len(active_pipelines)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "healthcare-ai-agent"}

@app.get("/")
async def root():
    return {"message": "Healthcare AI Agent API", "version": "1.0.0"}

if __name__ == "__main__":
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting Healthcare AI Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)