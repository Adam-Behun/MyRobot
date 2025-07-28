import os
import logging
import traceback
import asyncio
import datetime
from livekit import api
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
    room_url: Optional[str] = None
    token: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/start-call")
async def start_call(request: CallRequest):
    """Start a healthcare AI call session for a specific patient"""
    try:
        # Validate patient exists
        patient_data = await patient_db.find_patient_by_id(request.patient_id)
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient not found: {request.patient_id}")
        
        # Generate session ID
        session_id = request.session_id or str(uuid.uuid4())
        
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

        # Create and run pipeline with patient_id
        pipeline = HealthcareAIPipeline(session_id=session_id, patient_id=request.patient_id)
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
        for patient in pending_patients[:10]:  # Limit to 10 for demo
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

@app.get("/patient/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get detailed information about a specific patient"""
    try:
        patient = await patient_db.find_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Remove MongoDB ObjectId for JSON serialization
        patient_data = dict(patient)
        patient_data["_id"] = str(patient_data["_id"])
        
        return {
            "patient": patient_data,
            "ready_for_call": patient_data.get("prior_auth_status") == "Pending"
        }
        
    except Exception as e:
        logger.error(f"Error getting patient details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/start-call-quick/{patient_id}")
async def start_call_quick(patient_id: str):
    """Quick start call for a specific patient (convenience endpoint)"""
    request = CallRequest(patient_id=patient_id)
    return await start_call(request)

@app.get("/facilities")
async def get_facilities():
    """Get list of facilities with pending patients"""
    try:
        pending_patients = await patient_db.find_patients_pending_auth()
        
        # Group by facility
        facilities = {}
        for patient in pending_patients:
            facility_name = patient.get("facility_name", "Unknown")
            if facility_name not in facilities:
                facilities[facility_name] = {
                    "facility_name": facility_name,
                    "pending_count": 0,
                    "patients": []
                }
            
            facilities[facility_name]["pending_count"] += 1
            facilities[facility_name]["patients"].append({
                "patient_id": str(patient["_id"]),
                "patient_name": patient.get("patient_name"),
                "insurance_company": patient.get("insurance_company_name")
            })
        
        return {
            "facilities": list(facilities.values()),
            "total_facilities": len(facilities)
        }
        
    except Exception as e:
        logger.error(f"Error getting facilities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    return {
        "message": "Healthcare AI Agent API - Insurance Verification", 
        "version": "1.0.0",
        "features": [
            "insurance_verification", 
            "voice_pipeline", 
            "patient_data_integration",
            "function_calling",
            "workflow_management"
        ],
        "endpoints": {
            "start_call": "POST /start-call (requires patient_id)",
            "quick_start": "POST /start-call-quick/{patient_id}",
            "patients": "GET /patients (list pending patients)",
            "patient_details": "GET /patient/{patient_id}",
            "facilities": "GET /facilities (grouped by facility)",
            "conversation_state": "GET /conversation-state/{session_id}",
            "active_sessions": "GET /active-sessions",
            "end_call": "POST /end-call/{session_id}"
        }
    }

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
    uvicorn.run(app, host="0.0.0.0", port=8000)