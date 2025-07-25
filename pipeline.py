from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams

import os
import time
import logging
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareAIPipeline:
    def __init__(self):
        self.transport = None
        self.pipeline = None
        self.runner = None
        
    def create_pipeline(self) -> Pipeline:
        """Create the core Pipecat pipeline with healthcare AI components"""
        
        # Transport configuration
        transport_params = LiveKitParams(
            url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            token="", # Will be set per call
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
        
        self.transport = LiveKitTransport(transport_params)
        
        # STT Service
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-2",
            language="en-US"
        )
        
        # LLM Service with healthcare system prompt
        system_prompt = """You are MyRobot, a healthcare AI assistant for prior authorization. 
        Your role is to:
        1. Verify patient details (name, DOB, insurance info)
        2. Extract procedure codes and medical information
        3. Gather symptoms and medical justification
        4. Be professional, empathetic, and HIPAA-compliant
        5. Keep responses concise for voice interaction
        
        Always confirm patient identity before proceeding with medical information."""
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}]
        )
        
        # TTS Service
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Professional female voice
            model_id="sonic-english"
        )
        
        # Create pipeline without VAD for now
        self.pipeline = Pipeline([
            self.transport.input(),
            stt,
            self._add_latency_logging(llm, "LLM"),
            tts,
            self.transport.output()
        ])
        
        return self.pipeline
    
    def _add_latency_logging(self, service, stage_name: str):
        """Add latency logging wrapper around service"""
        original_process = service.process_frame
        
        async def logged_process(frame):
            start_time = time.time()
            result = await original_process(frame)
            latency = (time.time() - start_time) * 1000
            logger.info(f"{stage_name} latency: {latency:.2f}ms")
            return result
        
        service.process_frame = logged_process
        return service
    
    async def run(self, room_url: str, token: str):
        """Run the pipeline with given LiveKit room and token"""
        if not self.pipeline:
            self.create_pipeline()
        
        # Update transport token
        self.transport._params.token = token
        
        task = PipelineTask(self.pipeline)
        self.runner = PipelineRunner()
        
        logger.info(f"Starting healthcare AI pipeline for room: {room_url}")
        await self.runner.run(task)