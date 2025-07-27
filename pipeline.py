from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame, TextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.audio.utils import create_stream_resampler
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Simple OpenTelemetry setup (optional)
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Simple Langfuse integration
from langfuse import Langfuse

import audioop
from deepgram import LiveOptions
import wave
import os
import sys
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('websockets').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

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
            
            # Copy attributes from original frame
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

class AudioFrameLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._log_interval = 100
        self._audio_file = None
        try:
            self._audio_file = wave.open('debug_audio.wav', 'wb')
            self._audio_file.setnchannels(1)
            self._audio_file.setsampwidth(2)
            self._audio_file.setframerate(16000)
        except Exception as e:
            logger.warning(f"Could not create audio debug file: {e}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            if self._frame_count % self._log_interval == 0:
                logger.info(f"📬 Raw audio frame received: {len(frame.audio)} bytes, sample_rate={frame.sample_rate}, channels={frame.num_channels}")
            if self._audio_file:
                try:
                    self._audio_file.writeframes(frame.audio)
                except Exception as e:
                    logger.warning(f"Could not write audio frame: {e}")
        else:
            logger.info(f"📬 Non-audio frame: {type(frame).__name__}")
        await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._audio_file:
            try:
                self._audio_file.close()
            except Exception as e:
                logger.warning(f"Could not close audio file: {e}")
        await super().cleanup()

class IntermediateFrameLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        frame_type = type(frame).__name__
        text = getattr(frame, 'text', 'N/A')
        logger.info(f"🔍 Frame after STT: {frame_type}, text='{text}'")
        await self.push_frame(frame, direction)

class LLMInputLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesFrame):
            logger.info(f"🤖 LLM input messages: {frame.messages}")
        await self.push_frame(frame, direction)

class LLMOutputLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        frame_type = type(frame).__name__
        text = getattr(frame, 'text', 'N/A')
        logger.info(f"💬 LLM output frame: {frame_type}, text='{text}'")
        await self.push_frame(frame, direction)

class TTSInputLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            logger.info(f"🔊 TTS input text: '{frame.text}'")
        await self.push_frame(frame, direction)

class AudioOutputLogger(FrameProcessor):
    """Logger to track audio output frames and debug audio routing"""
    def __init__(self):
        super().__init__()
        self._audio_frame_count = 0
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TTSAudioRawFrame):
            self._audio_frame_count += 1
            logger.info(f"🔊 AUDIO OUTPUT: Frame #{self._audio_frame_count}, {len(frame.audio)} bytes, {frame.sample_rate}Hz, {frame.num_channels} channels")
            
            # Log first few audio frames in detail
            if self._audio_frame_count <= 3:
                logger.info(f"🎵 Audio data preview: {frame.audio[:20]}...")
                
        elif hasattr(frame, 'audio') and frame.audio:
            logger.info(f"🎵 Other audio frame: {type(frame).__name__}, {len(frame.audio)} bytes")
        else:
            logger.debug(f"📤 Non-audio output frame: {type(frame).__name__}")
            
        await self.push_frame(frame, direction)

class TransportOutputLogger(FrameProcessor):
    """Logger to track what's being sent to the transport output"""
    def __init__(self):
        super().__init__()
        self._output_frame_count = 0
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        self._output_frame_count += 1
        frame_type = type(frame).__name__
        
        if isinstance(frame, TTSAudioRawFrame):
            logger.info(f"🚀 TRANSPORT OUTPUT #{self._output_frame_count}: {frame_type} - {len(frame.audio)} bytes, {frame.sample_rate}Hz")
        elif hasattr(frame, 'audio'):
            logger.info(f"🚀 TRANSPORT OUTPUT #{self._output_frame_count}: {frame_type} - {len(frame.audio)} bytes")
        else:
            logger.info(f"🚀 TRANSPORT OUTPUT #{self._output_frame_count}: {frame_type}")
            
        await self.push_frame(frame, direction)

class SimpleLangfuseLogger(FrameProcessor):
    """Simple Langfuse logger that logs key events"""
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        try:
            self.langfuse = Langfuse()
            logger.info("Langfuse logging initialized")
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}")
            self.langfuse = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if self.langfuse:
            try:
                if isinstance(frame, TextFrame) and hasattr(frame, 'text') and frame.text:
                    # Log STT transcription or TTS input
                    event_data = {
                        "session_id": self.session_id,
                        "text": frame.text,
                        "frame_type": type(frame).__name__,
                        "direction": str(direction),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    if direction == FrameDirection.UPSTREAM:
                        # This is speech-to-text
                        self.langfuse.event(name="speech_to_text", **event_data)
                        logger.info(f"📝 Logged STT to Langfuse: '{frame.text}'")
                    elif direction == FrameDirection.DOWNSTREAM:
                        # This is text going to TTS
                        self.langfuse.event(name="text_to_speech", **event_data)
                        logger.info(f"🔊 Logged TTS to Langfuse: '{frame.text}'")
                
                elif isinstance(frame, LLMMessagesFrame):
                    # Log LLM conversation
                    messages = frame.messages if frame.messages else []
                    if messages:
                        self.langfuse.event(
                            name="llm_conversation",
                            session_id=self.session_id,
                            messages=messages,
                            timestamp=datetime.utcnow().isoformat()
                        )
                        logger.info(f"🤖 Logged LLM conversation to Langfuse")
            except Exception as e:
                logger.warning(f"Failed to log to Langfuse: {e}")
        
        await self.push_frame(frame, direction)

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default"):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        self.langfuse = None
        self._init_langfuse()
        
    def _init_langfuse(self):
        """Initialize Langfuse client"""
        try:
            self.langfuse = Langfuse()
            logger.info("Langfuse client initialized")
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}")
            self.langfuse = None
        
    def create_pipeline(self, url: str, token: str, room_name: str) -> Pipeline:
        logger.info(f"Creating simple pipeline for room: {room_name}")
        
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            audio_in_enabled=True,
            audio_out_enabled=True,  # Explicitly enable audio output
            vad_enabled=True
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
                endpointing=100,
                vad_events=True,
                smart_format=True,
                punctuate=True,
                profanity_filter=False
            )
        )
        
        initial_messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Keep responses brief and clear. Always acknowledge what the user said."}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o"
        )
        
        llm_context = OpenAILLMContext(messages=initial_messages)
        context_aggregators = llm.create_context_aggregator(llm_context)
        
        # Create TTS service with intelligent fallback strategy
        tts_service = None
        
        # Try Cartesia first (if you prefer it)
        if os.getenv("CARTESIA_API_KEY") and os.getenv("USE_CARTESIA", "false").lower() == "true":
            try:
                from pipecat.services.cartesia.tts import CartesiaTTSService
                tts_service = CartesiaTTSService(
                    api_key=os.getenv("CARTESIA_API_KEY"),
                    voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
                    model="sonic-2",  # Use latest model name
                    cartesia_version="2025-04-16"  # Use latest API version
                )
                logger.info("✅ Cartesia TTS service initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Cartesia TTS failed to initialize: {e}")
                tts_service = None
        
        # Fallback to OpenAI TTS (more reliable)
        if not tts_service:
            try:
                from pipecat.services.openai.tts import OpenAITTSService
                tts_service = OpenAITTSService(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    voice="alloy"  # or "echo", "fable", "onyx", "nova", "shimmer"
                )
                logger.info("✅ OpenAI TTS service initialized successfully (fallback)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize any TTS service: {e}")
                raise RuntimeError("No TTS service available")
        
        tts = tts_service
        
        self.pipeline = Pipeline([
            self.transport.input(),
            AudioResampler(),
            AudioFrameLogger(),
            DropEmptyAudio(),
            stt,
            IntermediateFrameLogger(),
            SimpleLangfuseLogger(self.session_id),
            context_aggregators.user(),
            LLMInputLogger(),
            llm,
            LLMOutputLogger(),
            TTSInputLogger(),
            tts,
            AudioOutputLogger(),  # Track TTS audio output
            context_aggregators.assistant(),
            TransportOutputLogger(),  # Track what goes to transport
            self.transport.output()
        ])
        
        logger.info("Simple pipeline created successfully")
        return self.pipeline
    
    def get_conversation_state(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workflow_state": "active",
            "workflow_context": {},
            "conversation_summary": {"message_count": 0}
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        return ["Simple pipeline active with basic Langfuse logging"]
    
    async def run(self, url: str, token: str, room_name: str):
        if not self.pipeline:
            self.create_pipeline(url, token, room_name)
        
        # Set up basic OpenTelemetry tracing if available and enabled
        if os.getenv("ENABLE_TRACING", "").lower() == "true" and OTEL_AVAILABLE:
            try:
                tracer_provider = TracerProvider()
                trace.set_tracer_provider(tracer_provider)
                
                exporter = OTLPSpanExporter(
                    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                    headers=dict(
                        item.split("=", 1) for item in 
                        os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").split(",") if item
                    )
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info("OpenTelemetry tracing enabled")
            except Exception as e:
                logger.warning(f"Failed to setup OpenTelemetry: {e}")
        
        task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
            enable_tracing=False,  # Disable pipecat's built-in tracing since the module doesn't exist
            conversation_id=self.session_id
        )
        
        self.runner = CustomPipelineRunner()
        
        # Log session start to Langfuse
        if self.langfuse:
            try:
                self.langfuse.event(
                    name="session_start",
                    session_id=self.session_id,
                    metadata={"room_name": room_name, "timestamp": datetime.utcnow().isoformat()}
                )
            except Exception as e:
                logger.warning(f"Failed to log session start: {e}")
        
        logger.info(f"Starting simple pipeline for session: {self.session_id}")
        
        try:
            await self.runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Log error to Langfuse
            if self.langfuse:
                try:
                    self.langfuse.event(
                        name="session_error",
                        session_id=self.session_id,
                        metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()}
                    )
                except Exception:
                    pass
            raise
        finally:
            # Flush Langfuse logs
            if self.langfuse:
                try:
                    self.langfuse.flush()
                except Exception:
                    pass