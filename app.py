import os
import logging
import traceback
import asyncio
import datetime
from livekit import api
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import uuid
from typing import Optional

from pipeline import HealthcareAIPipeline
from models import get_async_patient_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare AI Agent", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
active_pipelines = {}
patient_db = get_async_patient_db()

class CallRequest(BaseModel):
    patient_id: str  # Required: MongoDB ObjectId of the patient

@app.post("/start-call")
async def start_call(request: CallRequest):
    """Start a healthcare AI call session for a specific patient"""
    try:
        # Validate patient exists and fetch full record
        patient_data = await patient_db.find_patient_by_id(request.patient_id)
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient not found: {request.patient_id}")
        
        # Convert _id to string for serialization and LLM use
        patient_data['_id'] = str(patient_data['_id'])
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting healthcare call for session: {session_id}, patient: {patient_data.get('patient_name')}")
        
        # Initialize LiveKit API client
        lk_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL").replace("wss://", "https://"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )

        # Create a room
        room_name = f"healthcare-ai-session-{session_id[:8]}"
        room = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        base_url = os.getenv("LIVEKIT_URL")
        room_url = base_url

        # Generate bot token
        bot_token = api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET")
        ).with_identity("healthcare-bot") \
         .with_ttl(datetime.timedelta(seconds=3600)) \
         .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True
            )
        ).to_jwt()

        # Generate user token
        user_token = api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET")
        ).with_identity("user") \
         .with_ttl(datetime.timedelta(seconds=3600)) \
         .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True
            )
        ).to_jwt()

        # Close API client
        await lk_api.aclose()

        # Create and run pipeline with patient_id and full patient_data
        pipeline = HealthcareAIPipeline(session_id=session_id, patient_id=request.patient_id, patient_data=patient_data)
        active_pipelines[session_id] = pipeline
        asyncio.create_task(pipeline.run(base_url, bot_token, room_name))
        
        return {
            "status": "success",
            "session_id": session_id,
            "patient_id": request.patient_id,
            "patient_name": patient_data.get('patient_name'),
            "facility_name": patient_data.get('facility_name'),
            "room_url": room_url,
            "room_name": room_name,
            "user_token": user_token,
            "message": f"Call started for {patient_data.get('patient_name')} at {patient_data.get('facility_name')}. Use user_token to join the room."
        }
        
    except Exception as e:
        logger.error(f"Error starting call: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start call: {str(e)}")

@app.get("/patients")
async def get_patients():
    """Get list of patients available for calling"""
    try:
        # Get patients with pending authorization
        pending_patients = await patient_db.find_patients_pending_auth()
        
        patients = []
        for patient in pending_patients[:10]:
            patients.append({
                "patient_id": str(patient["_id"]),
                "patient_name": patient.get("patient_name"),
                "facility_name": patient.get("facility_name"),
                "insurance_company": patient.get("insurance_company_name"),
                "prior_auth_status": patient.get("prior_auth_status"),
                "appointment_time": patient.get("appointment_time")
            })
        
        return {
            "patients": patients,
            "total_count": len(pending_patients)
        }
        
    except Exception as e:
        logger.error(f"Error getting patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-flow/{session_id}")
async def test_flow_state(session_id: str):
    """Debug endpoint to check flow state"""
    if session_id not in active_pipelines:
        raise HTTPException(status_code=404, detail="Session not found")
    
    pipeline = active_pipelines[session_id]
    if pipeline.flow_manager:
        return {
            "current_node": pipeline.flow_manager.state.get("current_node"),
            "collected_info": pipeline.flow_manager.state.get("collected_info"),
            "patient_id": pipeline.patient_id,
            "transcripts_count": len(pipeline.transcripts)
        }
    return {"error": "No flow manager"}

@app.get("/conversation-state/{session_id}")
async def get_conversation_state(session_id: str):
    """Get current conversation state"""
    try:
        if session_id not in active_pipelines:
            raise HTTPException(status_code=404, detail="Session not found")
        
        pipeline = active_pipelines[session_id]
        state = pipeline.get_conversation_state()
        
        return {
            "session_id": session_id,
            "patient_id": pipeline.patient_id,
            "workflow_state": state["workflow_state"],
            "workflow_context": state["workflow_context"],
            "patient_data": state["patient_data"],
            "collected_info": state["collected_info"]
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
            patient_id = pipeline.patient_id
            
            # Cleanup
            del active_pipelines[session_id]
            
            logger.info(f"Ended healthcare call session: {session_id}")
            return {
                "status": "success", 
                "session_id": session_id,
                "patient_id": patient_id,
                "final_state": final_state["workflow_state"]
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
        
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-sessions")
async def get_active_sessions():
    """Get list of active call sessions"""
    sessions = []
    
    for session_id, pipeline in active_pipelines.items():
        state = pipeline.get_conversation_state()
        
        sessions.append({
            "session_id": session_id,
            "patient_id": pipeline.patient_id,
            "workflow_state": state["workflow_state"],
            "has_patient_data": state["patient_data"] is not None
        })
    
    return {
        "active_sessions": sessions,
        "session_count": len(active_pipelines)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with database connectivity"""
    try:
        # Test database connection
        db_status = "connected"
        patient_count = len(await patient_db.find_patients_pending_auth())
    except Exception as e:
        db_status = f"error: {str(e)}"
        patient_count = 0
    
    return {
        "status": "healthy",
        "service": "healthcare-ai-agent",
        "active_sessions": len(active_pipelines),
        "database_status": db_status,
        "pending_patients": patient_count
    }

@app.get("/")
async def root():
    """Serve the main application interface"""
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        # Fallback if file doesn't exist
        return HTMLResponse("""
        <html>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>Prior Authorization Voice Agent</h1>
                <p>Application is loading...</p>
                <p>If this persists, please contact support.</p>
            </body>
        </html>
        """)

if __name__ == "__main__":
    # Validate required environment variables
    required_vars = [
        "OPENAI_API_KEY", 
        "DEEPGRAM_API_KEY", 
        "LIVEKIT_API_KEY", 
        "LIVEKIT_API_SECRET", 
        "LIVEKIT_URL",
        "MONGO_URI"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting Healthcare AI Agent server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))