import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from pipeline import HealthcareAIPipeline
from models import get_db_client, PatientRecord

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare AI Agent", version="1.0.0")

# Global instances
db_client = get_db_client()
patient_db = PatientRecord(db_client)
pipeline_instance = HealthcareAIPipeline()

class CallRequest(BaseModel):
    room_url: str
    token: str
    patient_id: str = None

@app.post("/start-call")
async def start_call(request: CallRequest):
    """Start a healthcare AI call session"""
    try:
        logger.info(f"Starting call for room: {request.room_url}")
        
        # Optional: Validate patient if provided
        if request.patient_id:
            patient = patient_db.find_patient(request.patient_id)
            if not patient:
                raise HTTPException(status_code=404, detail="Patient not found")
        
        # Start the pipeline
        await pipeline_instance.run(request.room_url, request.token)
        
        return {"status": "success", "message": "Call started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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