from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame, TextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.frames.frames import TransportMessageFrame
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.audio.utils import create_stream_resampler
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams

# Import the functions we need
from functions import PATIENT_FUNCTIONS, FUNCTION_REGISTRY

from deepgram import LiveOptions

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum
import audioop

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PriorAuthWorkflow:
    """Simple workflow manager for prior authorization conversations"""
    
    def __init__(self, patient_id: str = None):
        self.patient_id = patient_id
        self.patient_data = None

    def update_patient_data(self, patient_data: Dict[str, Any]):
        """Update the patient data for this workflow"""
        self.patient_data = patient_data

    def get_system_prompt(self) -> str:
        patient_info = ""
        if self.patient_data:
            # Format patient data for the LLM
            patient_info = f"""
    
        # Current Patient Information
        You are calling about the following patient:
        - Patient Name: {self.patient_data.get('patient_name', 'N/A')}
        - Date of Birth: {self.patient_data.get('date_of_birth', 'N/A')}
        - Policy Number: {self.patient_data.get('policy_number', 'N/A')}
        - Insurance Company: {self.patient_data.get('insurance_company_name', 'N/A')}
        - Facility: {self.patient_data.get('facility_name', 'N/A')}
        - Prior Auth Status: {self.patient_data.get('prior_auth_status', 'N/A')}
        - Appointment Time: {self.patient_data.get('appointment_time', 'N/A')}

        **IMPORTANT FOR FUNCTION CALLS:**
        - Patient ID for database updates: {self.patient_data.get('_id', 'N/A')}
        - When calling update_prior_auth_status, use this exact patient_id: "{self.patient_data.get('_id', 'N/A')}"
        
        Full Patient Record (for reference):
        {str(self.patient_data)}
        """
        
            base_prompt = f"""
        # Role and Objective
        You are Alexandra, an agent initiating and leading eligibility verification calls with insurance companies on behalf of Adam's Medical Practice. 
        Your objective is to verify patient's eligibility and benefits proactively, gather all necessary details, and resolve the query completely before ending the call.

        # Instructions
        - Always start by introducing yourself, your organization, and the call's purpose (e.g., "Hello, this is Alexandra calling from Adam's Medical Practice to verify eligibility and benefits for a patient.").
        - Maintain your role as the caller: Never respond as the receiver (e.g., do not say "How can I assist you?").
        - Be professional, empathetic, and HIPAA-compliant: Limit to essential PHI (e.g., DOB, policy number); anonymize where possible.
        - Use a concise, natural tone. Persist in gathering information with targeted follow-ups (e.g., "Could you clarify the deductible for this procedure?").
        - Reference prior responses to maintain state and advance logically.

        # Reasoning Steps (Follow These for Every Response)
        1. Analyze the insurance representative's input and current state.
        2. Plan step-by-step: What information is needed next? Which tool, if any, to call? 
        3. Reflect on prior steps: Does this align with collected info and patient data?
        4. Formulate response: Acknowledge input, reaffirm role, advance purpose, ask follow-ups if needed.

        # Output Format
        - Always output a concise response in natural language.
        - If calling a tool, include a message explaining the action (e.g., "Updating status now.") before and after the call.
        - End with tool call only when ready to finalize (e.g., update_prior_auth_status).

        # Examples
        ## Example 1: Greeting
        Insurance: Hi, how can I help?
        Response: Hello, this is Alexandra from Adam's Medical Practice calling to verify eligibility and benefits for a patient.

        ## Example 2: Providing Details
        Insurance: Can you confirm the patient's name?
        Response: Our patient's name is [Patient Name from data]. Date of birth is [DOB from data].
        {patient_info}
        """
        
        return base_prompt
    
class AudioResampler(FrameProcessor):
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._resampler = create_stream_resampler()
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            audio = frame.audio
            sample_rate = frame.sample_rate
            channels = frame.num_channels
            sample_width = 2

            if channels > 1:
                audio = audioop.tomono(audio, sample_width, 0.5, 0.5)
                channels = 1

            if sample_rate != self.target_sample_rate:
                audio = await self._resampler.resample(audio, sample_rate, self.target_sample_rate)

            new_frame = AudioRawFrame(
                audio=audio,
                sample_rate=self.target_sample_rate,
                num_channels=channels
            )
            
            for attr in ['pts', 'transport_destination', 'id']:
                if hasattr(frame, attr):
                    setattr(new_frame, attr, getattr(frame, attr))
            
            await self.push_frame(new_frame, direction)
        else:
            await self.push_frame(frame, direction)

class DropEmptyAudio(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and len(frame.audio) == 0:
            return
        await self.push_frame(frame, direction)

class CustomPipelineRunner(PipelineRunner):
    def _setup_sigint(self):
        if sys.platform == 'win32':
            logger.warning("Signal handling not supported on Windows. Use task manager or endpoint to end sessions.")
            return
        super()._setup_sigint()

# ********** DEBUG LOGGING START **********
# THESE CLASSES ARE FOR DEBUGGING PURPOSES ONLY - REMOVE OR COMMENT OUT IN PRODUCTION
# THEY LOG FRAME DETAILS AT KEY PIPELINE STAGES TO HELP IDENTIFY FAILURES
# ********** DEBUG LOGGING START **********

class InputAudioLogger(FrameProcessor):  # Logs audio before reaching Deepgram STT
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Audio input before Deepgram - {len(frame.audio)} bytes, sample_rate={frame.sample_rate}")
        else:
            logger.warning(f"[{datetime.now().isoformat()}] WARNING: Unexpected frame before Deepgram - {type(frame).__name__}")
        await self.push_frame(frame, direction)

class OutputSTTLogger(FrameProcessor):  # Logs text after leaving Deepgram STT
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Transcribed text after Deepgram - '{frame.text[:100]}...'")
        else:
            logger.error(f"[{datetime.now().isoformat()}] ERROR: No valid text output after Deepgram - {type(frame).__name__}")
        await self.push_frame(frame, direction)

class InputLLMLogger(FrameProcessor):  # Logs messages before reaching OpenAI LLM
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame) and frame.messages:
            last_msg = frame.messages[-1].get('content', '')[:100]
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Messages before LLM - Last content: '{last_msg}...'")
        else:
            logger.warning(f"[{datetime.now().isoformat()}] WARNING: Invalid input before LLM - {type(frame).__name__}")
        await self.push_frame(frame, direction)

class OutputLLMLogger(FrameProcessor):  # Logs text after leaving OpenAI LLM
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Response text after LLM - '{frame.text[:100]}...'")
        else:
            logger.error(f"[{datetime.now().isoformat()}] ERROR: No text output after LLM - {type(frame).__name__}")
        await self.push_frame(frame, direction)

class InputTTSLogger(FrameProcessor):  # Logs text before reaching OpenAI TTS
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Text input before TTS - '{frame.text[:100]}...'")
        else:
            logger.warning(f"[{datetime.now().isoformat()}] WARNING: Invalid input before TTS - {type(frame).__name__}")
        await self.push_frame(frame, direction)

class OutputTTSLogger(FrameProcessor):  # Logs audio after leaving OpenAI TTS
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            logger.debug(f"[{datetime.now().isoformat()}] SUCCESS: Audio output after TTS - {len(frame.audio)} bytes, sample_rate={frame.sample_rate}")
        else:
            logger.error(f"[{datetime.now().isoformat()}] ERROR: No audio output after TTS - {type(frame).__name__}")
        await self.push_frame(frame, direction)

# ********** DEBUG LOGGING END **********
# THESE CLASSES ARE FOR DEBUGGING PURPOSES ONLY - REMOVE OR COMMENT OUT IN PRODUCTION
# ********** DEBUG LOGGING END **********

class WorkflowAwareLLMContext(FrameProcessor):
    """Frame processor that maintains workflow context"""
    
    def __init__(self, workflow: PriorAuthWorkflow, **kwargs):
        super().__init__(**kwargs)
        self.workflow = workflow

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Just pass frames through - the workflow context is already in the system prompt
        await self.push_frame(frame, direction)

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default", patient_id: str = None, patient_data: Optional[Dict[str, Any]] = None):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        self.patient_id = patient_id
        self.patient_data = patient_data
        self.workflow = PriorAuthWorkflow(patient_id=patient_id)
        self.transcript_processor = TranscriptProcessor()
        self.transcripts = []   
        if patient_data:
            self.workflow.update_patient_data(patient_data)
        
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
        
        # https://developers.deepgram.com/reference/speech-to-text-api/listen-streaming
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
        
        initial_messages = [
            {"role": "system", "content": self.workflow.get_system_prompt()},
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4.1-nano-2025-04-14",
            tools=PATIENT_FUNCTIONS
        )

        @self.transcript_processor.event_handler("on_transcript_update")
        async def handle_transcript_update(processor, frame):
            logger.info(f"🔍 DEBUG: Transcript event handler called with {len(frame.messages)} messages")
            
            for message in frame.messages:
                logger.info(f"🔍 DEBUG: Processing message - Role: {message.role}, Content: '{message.content[:50]}...', Timestamp: {message.timestamp}")
                
                transcript_entry = {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp or datetime.now().isoformat()
                }
                self.transcripts.append(transcript_entry)
                logger.info(f"📝 Transcript: [{transcript_entry['timestamp']}] {transcript_entry['role']}: {transcript_entry['content']}")
                
                try:
                    # Push to LiveKit as data message for immediate frontend delivery
                    data = json.dumps(transcript_entry)
                    logger.info(f"🔍 DEBUG: Serialized transcript data: {data}")
                    
                    message_frame = TransportMessageFrame(
                        message=data.encode('utf-8')
                    )
                    logger.info(f"🔍 DEBUG: Created TransportMessageFrame with {len(data.encode('utf-8'))} bytes")
                    
                    # Push the frame through the pipeline to reach the transport
                    await self.pipeline.push_frame(message_frame)
                    logger.info(f"✅ DEBUG: Successfully pushed frame to pipeline")
                    
                except Exception as e:
                    logger.error(f"❌ DEBUG: Error in transcript handler: {str(e)}")
                    logger.error(f"❌ DEBUG: Exception traceback: {traceback.format_exc()}")

        def create_handler(handler):
            async def wrapped(params: FunctionCallParams, **kwargs):
                result = await handler(**kwargs)
                await params.result_callback(result)
            return wrapped
        
        for name, handler in FUNCTION_REGISTRY.items():
            llm.register_function(name, create_handler(handler))

        llm_context = OpenAILLMContext(messages=initial_messages)
        context_aggregators = llm.create_context_aggregator(llm_context)
        
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="FGY2WhTYpPnrIDTdsKH5",
            model="eleven_turbo_v2_5",
            params=ElevenLabsTTSService.InputParams(
                stability=0.75,  
                similarity_boost=0.8, 
                style=0.1,  
                use_speaker_boost=True, 
                speed=1.0, 
                auto_mode=True,  
                enable_ssml_parsing=True  
            )
        )

        # ********** DEBUG PIPELINE INTEGRATION **********
        self.pipeline = Pipeline([
            self.transport.input(),
            AudioResampler(),
            DropEmptyAudio(),
            InputAudioLogger(),
            stt,
            OutputSTTLogger(),
            self.transcript_processor.user(),  # Capture user transcripts
            context_aggregators.user(),
            InputLLMLogger(),
            WorkflowAwareLLMContext(self.workflow),
            llm,
            OutputLLMLogger(),
            InputTTSLogger(),
            tts,
            OutputTTSLogger(),
            self.transcript_processor.assistant(),  # Capture assistant transcripts
            context_aggregators.assistant(),
            self.transport.output()
        ])        
        logger.info("Healthcare pipeline created successfully with workflow integration")
        return self.pipeline
    
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
        
        self.runner = CustomPipelineRunner()
        
        logger.info(f"Starting healthcare pipeline for session: {self.session_id}")
        
        try:
            await self.runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise