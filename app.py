import os
import logging
import traceback
import asyncio
import datetime
from livekit import api
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import uuid
from urllib.parse import urlparse, parse_qs

from pipeline import HealthcareAIPipeline
from models import get_async_patient_db, AsyncPatientRecord
from memory import MemoryManager
from evaluations import HealthcareEvaluator

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
patient_db = get_async_patient_db()
memory_manager = MemoryManager()
evaluator = HealthcareEvaluator()
active_pipelines = {}

class CallRequest(BaseModel):
    room_url: str
    token: str
    session_id: str = None

@app.post("/start-call")
async def start_call(request: CallRequest = None):  # Make request optional for simplicity
    """Start a healthcare AI call session, auto-creating room and tokens"""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        logger.info(f"Starting optimized call for session: {session_id}")
        
        # Initialize LiveKit API client
        lk_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL").replace("wss://", "https://"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )

        # Create a room
        room_name = f"healthcare-ai-session-{session_id[:8]}"  # Unique per session
        room = await lk_api.room.create_room(api.CreateRoomRequest(name=room_name))
        base_url = os.getenv("LIVEKIT_URL")
        # Don't add room parameters to the URL - LiveKit client will handle this
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

        # Create and run pipeline
        pipeline = HealthcareAIPipeline(session_id=session_id)
        active_pipelines[session_id] = pipeline
        asyncio.create_task(pipeline.run(base_url, bot_token, room_name))
        
        return {
            "status": "success",
            "session_id": session_id,
            "room_url": room_url,
            "room_name": room_name,
            "user_token": user_token,
            "message": "Call started successfully. Use user_token to join the room."
        }
        
    except Exception as e:
        logger.error(f"Error starting call: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start call: {str(e)}")

@app.get("/conversation-state/{session_id}")
async def get_conversation_state(session_id: str):
    """Get current conversation state with performance metrics"""
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
            "performance_metrics": state.get("performance_metrics", {}),
            "optimization_suggestions": pipeline.get_optimization_suggestions()
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-session/{session_id}")
async def evaluate_session(session_id: str):
    """Run evaluation on a session"""
    try:
        if session_id not in active_pipelines:
            raise HTTPException(status_code=404, detail="Session not found")
        
        pipeline = active_pipelines[session_id]
        
        # Quick evaluation using default test case
        test_case = evaluator.test_cases[0]  # Use first test case
        results = await evaluator.run_comprehensive_evaluation(pipeline, test_case)
        
        return {
            "session_id": session_id,
            "evaluation_results": {k: v.score for k, v in results.items()},
            "detailed_metrics": {k: v.details for k, v in results.items()}
        }
        
    except Exception as e:
        logger.error(f"Error evaluating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance-report")
async def get_performance_report():
    """Get overall system performance report"""
    try:
        # Aggregate metrics from all active sessions
        total_metrics = {
            "active_sessions": len(active_pipelines),
            "average_latency": 0.0,
            "average_quality": 0.0,
            "total_interruptions": 0
        }
        
        if active_pipelines:
            latencies = []
            qualities = []
            interruptions = []
            
            for pipeline in active_pipelines.values():
                state = pipeline.get_conversation_state()
                perf_metrics = state.get("performance_metrics", {})
                interruption_metrics = state.get("interruption_analytics", {})
                audio_metrics = state.get("audio_quality", {})
                
                if perf_metrics.get("average_total_latency_ms"):
                    latencies.append(perf_metrics["average_total_latency_ms"])
                
                if audio_metrics.get("average_quality_score"):
                    qualities.append(audio_metrics["average_quality_score"])
                
                if interruption_metrics.get("total_interruptions"):
                    interruptions.append(interruption_metrics["total_interruptions"])
            
            total_metrics.update({
                "average_latency": sum(latencies) / len(latencies) if latencies else 0.0,
                "average_quality": sum(qualities) / len(qualities) if qualities else 0.0,
                "total_interruptions": sum(interruptions)
            })
        
        # Get evaluation report
        eval_report = evaluator.generate_evaluation_report()
        
        return {
            "system_metrics": total_metrics,
            "evaluation_summary": eval_report.get("performance_summary", {}),
            "recommendations": eval_report.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
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
            
            logger.info(f"Ended optimized call session: {session_id}")
            return {
                "status": "success", 
                "final_performance": final_state.get("performance_metrics", {}),
                "optimization_suggestions": pipeline.get_optimization_suggestions()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
        
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-sessions")
async def get_active_sessions():
    """Get list of active call sessions with performance summary"""
    sessions = []
    
    for session_id, pipeline in active_pipelines.items():
        state = pipeline.get_conversation_state()
        perf_metrics = state.get("performance_metrics", {})
        
        sessions.append({
            "session_id": session_id,
            "workflow_state": state["workflow_state"],
            "duration_minutes": state["conversation_summary"].get("duration_minutes", 0),
            "average_latency": perf_metrics.get("average_total_latency_ms", 0),
            "performance_ratio": perf_metrics.get("performance_ratio", 0)
        })
    
    return {
        "active_sessions": sessions,
        "session_count": len(active_pipelines)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    system_status = {
        "status": "healthy",
        "service": "healthcare-ai-agent",
        "active_sessions": len(active_pipelines),
        "optimization_enabled": True
    }
    
    # Check if any sessions are having performance issues
    if active_pipelines:
        high_latency_sessions = []
        for session_id, pipeline in active_pipelines.items():
            state = pipeline.get_conversation_state()
            perf_metrics = state.get("performance_metrics", {})
            if perf_metrics.get("performance_ratio", 0) > 1.5:  # 50% over target
                high_latency_sessions.append(session_id)
        
        if high_latency_sessions:
            system_status["warnings"] = f"High latency detected in {len(high_latency_sessions)} sessions"
    
    return system_status

@app.get("/")
async def root():
    return {
        "message": "Healthcare AI Agent API - Optimized MVP", 
        "version": "1.0.0",
        "features": ["function_calling", "workflow_management", "latency_optimization", "interruption_handling", "audio_processing"]
    }

if __name__ == "__main__":
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)
    
    logger.info("Starting Healthcare AI Agent server with optimizations...")
    uvicorn.run(app, host="0.0.0.0", port=8000)