from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from deepgram import LiveOptions  # For explicit Deepgram configuration

import os
import sys
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging
logger = logging.getLogger(__name__)

class CustomPipelineRunner(PipelineRunner):
    def _setup_sigint(self):
        if sys.platform == 'win32':
            logger.warning("Signal handling not supported on Windows. Use task manager or endpoint to end sessions.")
            return
        super()._setup_sigint()

class AudioFrameLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._log_interval = 100  # Log every 100th frame to reduce verbosity

    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            if self._frame_count % self._log_interval == 0:
                logger.info(f"📬 Raw audio frame received: {len(frame.audio)} bytes, sample_rate={frame.sample_rate}, channels={frame.num_channels}")
        else:
            logger.info(f"📬 Non-audio frame: {type(frame).__name__}")
        # Explicitly forward the frame to the next processor
        return frame

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default"):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        
    def create_pipeline(self, url: str, token: str, room_name: str) -> Pipeline:
        """Create a simple, clean pipeline for Pipecat 0.0.70"""
        logger.info(f"Creating simple pipeline for room: {room_name}")
        
        # Transport configuration
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            audio_in_enabled=True,  # Ensure audio input is enabled
            vad_enabled=False  # Disable VAD to pass all audio frames
        )
        
        self.transport = LiveKitTransport(url, token, room_name, params=params)
        
        # STT Service - with explicit LiveOptions for real-time transcription
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-2",  # Explicitly set real-time model
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                interim_results=True,  # Enable partial transcripts
                endpointing=100,  # Lower to 100ms for faster detection
                vad_events=True,  # Enable Deepgram VAD for speech detection
                smart_format=True,
                punctuate=True,
                profanity_filter=False  # Disable for broader transcription capture
            )
        )
        
        # Wrap STT to see what it receives
        original_stt_process = stt.process_frame
        async def logged_stt_process(frame, direction):
            frame_type = type(frame).__name__
            if hasattr(frame, 'audio') and frame.audio:
                logger.info(f"🎵 STT got audio frame: {len(frame.audio)} bytes")
                # Log a sample of audio data for debugging
                logger.debug(f"STT audio frame sample: {frame.audio[:10].hex()}...")
            elif hasattr(frame, 'text') and frame.text:
                logger.info(f"📝 STT transcribed: '{frame.text}'")
            else:
                logger.info(f"📥 STT got frame: {frame_type}")
            result = await original_stt_process(frame, direction)
            return result
        stt.process_frame = logged_stt_process
        
        # Simple LLM
        initial_messages = [
            {"role": "system", "content": "You are a helpful healthcare AI assistant. Keep responses brief and clear. Always acknowledge what the user said."}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            messages=initial_messages
        )
        
        # Wrap LLM to see interactions
        original_llm_process = llm.process_frame
        async def logged_llm_process(frame, direction):
            if hasattr(frame, 'text') and frame.text:
                logger.info(f"🤖 LLM got text: '{frame.text}'")
            result = await original_llm_process(frame, direction)
            if hasattr(result, 'text') and result.text:
                logger.info(f"💬 LLM replied: '{result.text}'")
            return result
        llm.process_frame = logged_llm_process
        
        # TTS Service
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
            model_id="sonic-english"
        )
        
        # Wrap TTS to see what it outputs
        original_tts_process = tts.process_frame
        async def logged_tts_process(frame, direction):
            if hasattr(frame, 'text') and frame.text:
                logger.info(f"🔊 TTS got text: '{frame.text}'")
            result = await original_tts_process(frame, direction)
            if hasattr(result, 'audio') and result.audio:
                logger.info(f"🎤 TTS created audio: {len(result.audio)} bytes")
            return result
        tts.process_frame = logged_tts_process
        
        # Create the simplest possible pipeline with added logger for raw frames
        self.pipeline = Pipeline([
            self.transport.input(),
            AudioFrameLogger(),  # Insert logger to capture raw frames before STT
            stt,
            llm,
            tts,
            self.transport.output()
        ])
        
        logger.info("Simple pipeline created successfully")
        return self.pipeline
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get basic conversation state"""
        return {
            "session_id": self.session_id,
            "workflow_state": "active",
            "workflow_context": {},
            "conversation_summary": {"message_count": 0}
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get basic suggestions"""
        return ["Simple pipeline active"]
    
    async def run(self, url: str, token: str, room_name: str):
        """Run the simple pipeline"""
        if not self.pipeline:
            self.create_pipeline(url, token, room_name)
        
        task = PipelineTask(self.pipeline)
        self.runner = CustomPipelineRunner()
        
        logger.info(f"Starting simple pipeline for session: {self.session_id}")
        await self.runner.run(task)