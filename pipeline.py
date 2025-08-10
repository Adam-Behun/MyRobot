from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame, TextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat_flows import FlowManager
from deepgram import LiveOptions

# Local imports
from audio_processors import AudioResampler, DropEmptyAudio
from flow_nodes import create_greeting_node
from functions import PATIENT_FUNCTIONS

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def setup_logging(level=logging.INFO):
    """Configure logging for the pipeline"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logging(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)

class CustomPipelineRunner(PipelineRunner):
    def _setup_sigint(self):
        if sys.platform == 'win32':
            logger.warning("Signal handling not supported on Windows")
            return
        super()._setup_sigint()

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default", patient_id: str = None, 
                 patient_data: Optional[Dict[str, Any]] = None, debug_mode: bool = False):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        self.patient_id = patient_id
        self.patient_data = patient_data
        self.transcripts = []
        self.transcript_processor = TranscriptProcessor()
        self.flow_manager = None
        self.debug_mode = debug_mode
        self.llm = None
        self.context_aggregators = None
        
    def create_pipeline(self, url: str, token: str, room_name: str) -> Pipeline:
        logger.info(f"Creating healthcare pipeline for room: {room_name}")
        
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
        )
        
        self.transport = LiveKitTransport(url, token, room_name, params=params)
        
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-2",
                language="en-US",
                encoding="linear16",
                sample_rate=16000,
                channels=1,
                interim_results=True,
                endpointing=200,
                vad_events=True,
                smart_format=True,
                punctuate=True,
                filler_words=True,
                utterance_end_ms=1000,
            )
        )
        
        self.llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",  # Use standard model name
            tools=PATIENT_FUNCTIONS
        )

        self._setup_transcript_handler()

        llm_context = OpenAILLMContext(messages=[])
        self.context_aggregators = self.llm.create_context_aggregator(llm_context)
        
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="FGY2WhTYpPnrIDTdsKH5",
            model="eleven_turbo_v2_5"
        )

        # Build pipeline
        pipeline_components = [
            self.transport.input(),
            AudioResampler(),
            DropEmptyAudio(),
            stt,
            self.transcript_processor.user(),
            self.context_aggregators.user(),
            self.llm,
            tts,
            self.transcript_processor.assistant(),
            self.context_aggregators.assistant(),
            self.transport.output()
        ]

        self.pipeline = Pipeline(pipeline_components)
        logger.info("Healthcare pipeline created successfully with Pipecat Flows")
        return self.pipeline

    def _setup_transcript_handler(self):
        """Setup transcript event handler"""
        @self.transcript_processor.event_handler("on_transcript_update")
        async def handle_transcript_update(processor, frame):
            for message in frame.messages:
                transcript_entry = {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp or datetime.now().isoformat(),
                    "type": "transcript"
                }
                self.transcripts.append(transcript_entry)
                logger.info(f"Transcript: [{transcript_entry['timestamp']}] {transcript_entry['role']}: {transcript_entry['content'][:50]}...")
                
                # Send via transport if available
                transport_client = self.transport._output._client if self.transport._output else None
                if transport_client:
                    try:
                        data = json.dumps(transcript_entry)
                        await transport_client.send_data(data.encode('utf-8'))
                    except Exception as e:
                        logger.error(f"Error sending transcript: {e}")
    
    async def run(self, url: str, token: str, room_name: str):
        if not self.pipeline:
            self.create_pipeline(url, token, room_name)
        
        task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
            conversation_id=self.session_id
        )
        
        # Create FlowManager
        self.flow_manager = FlowManager(
            task=task,
            llm=self.llm,
            context_aggregator=self.context_aggregators
        )
        
        # Initialize flow with patient data
        if self.flow_manager and self.patient_data:
            # Store patient data in flow manager state FIRST
            self.flow_manager.state["patient_data"] = self.patient_data
            self.flow_manager.state["patient_id"] = self.patient_data.get('_id')
            self.flow_manager.state["collected_info"] = {
                "reference_number": None,
                "auth_status": None,
                "insurance_rep_name": None
            }
            
            # Create greeting node with patient data
            from flow_nodes import create_greeting_node
            greeting_node = create_greeting_node(self.patient_data)
            
            # Initialize with the greeting node
            await self.flow_manager.initialize(greeting_node)
            
            logger.info(f"Flow initialized with patient data for: {self.patient_data.get('patient_name')}")
        
        self.runner = CustomPipelineRunner()
        logger.info(f"Starting healthcare pipeline for session: {self.session_id}")
        
        try:
            await self.runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    def get_conversation_state(self):
        """Get current conversation state"""
        if self.flow_manager and hasattr(self.flow_manager, 'state'):
            return {
                "workflow_state": "active",
                "patient_data": self.flow_manager.state.get("patient_data", self.patient_data),
                "collected_info": self.flow_manager.state.get("collected_info", {})
            }
        return {
            "workflow_state": "inactive",
            "patient_data": self.patient_data,
            "collected_info": {}
        }