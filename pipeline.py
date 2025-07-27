from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.audio.utils import create_stream_resampler
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
import audioop
from deepgram import LiveOptions
import wave  # Added for audio saving
import os
import sys
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging
logging.getLogger('websockets').setLevel(logging.INFO)  # Suppress websockets debug logs
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
            sample_width = 2  # Assuming 16-bit audio

            # Downmix to mono if necessary
            if channels > 1:
                audio = audioop.tomono(audio, sample_width, 0.5, 0.5)
                channels = 1

            # Resample if sample rate does not match target
            if sample_rate != self.target_sample_rate:
                audio = await self._resampler.resample(audio, sample_rate, self.target_sample_rate)

            # Create and push the processed frame
            new_frame = AudioRawFrame(
                audio=audio,
                sample_rate=self.target_sample_rate,
                num_channels=channels
            )
            # Copy pts attribute if present on the original frame
            if hasattr(frame, 'pts'):
                new_frame.pts = frame.pts
            # Copy transport_destination attribute if present on the original frame
            if hasattr(frame, 'transport_destination'):
                new_frame.transport_destination = frame.transport_destination
            # Copy id attribute if present on the original frame (for observer compatibility)
            if hasattr(frame, 'id'):
                new_frame.id = frame.id
            await self.push_frame(new_frame, direction)
        else:
            await self.push_frame(frame, direction)

class DropEmptyAudio(FrameProcessor):
    """Filter to drop empty audio frames before STT processing"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize parent class to set up internal attributes

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and len(frame.audio) == 0:
            return  # Skip pushing empty frames
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
        self._log_interval = 100  # Log every 100th frame to reduce verbosity
        self._audio_file = wave.open('debug_audio.wav', 'wb')  # Open WAV file for writing
        self._audio_file.setnchannels(1)  # Mono
        self._audio_file.setsampwidth(2)  # 16-bit
        self._audio_file.setframerate(16000)  # 16000 Hz

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            if self._frame_count % self._log_interval == 0:
                logger.info(f"📬 Raw audio frame received: {len(frame.audio)} bytes, sample_rate={frame.sample_rate}, channels={frame.num_channels}")
            self._audio_file.writeframes(frame.audio)  # Save audio to file
        else:
            logger.info(f"📬 Non-audio frame: {type(frame).__name__}")
        await self.push_frame(frame, direction)

    async def cleanup(self):
        self._audio_file.close()  # Close WAV file on cleanup
        await super().cleanup()

class IntermediateFrameLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        frame_type = type(frame).__name__
        text = getattr(frame, 'text', 'N/A')
        logger.info(f"🔍 Frame after STT: {frame_type}, text='{text}'")
        await self.push_frame(frame, direction)

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default"):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        
    def create_pipeline(self, url: str, token: str, room_name: str) -> Pipeline:
        logger.info(f"Creating simple pipeline for room: {room_name}")
        
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            audio_in_enabled=True,
            vad_enabled=True  # Enable LiveKit VAD for better speech detection
        )
        
        self.transport = LiveKitTransport(url, token, room_name, params=params)
        
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
                vad_events=True,  # Enable Deepgram VAD to handle speech boundaries reliably
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
                pass  # Removed debug logging for audio frames
            elif hasattr(frame, 'text') and frame.text:
                logger.info(f"📝 STT transcribed: '{frame.text}'")
            else:
                logger.info(f"📥 STT got frame: {frame_type}")
            result = await original_stt_process(frame, direction)
            return result
        stt.process_frame = logged_stt_process
        
        # Enhanced logging for Deepgram WebSocket messages with flexible signature
        original_on_message = stt._on_message
        async def logged_on_message(self, *args, **kwargs):
            # Extract result from args or kwargs
            result = None
            if args:
                result = args[0]
            if 'result' in kwargs:
                result = kwargs['result']
            if result is not None:
                logger.debug(f"Deepgram WebSocket message (full): {result}")
            # Invoke original, avoiding duplicate 'result' if present
            if 'result' in kwargs and args and kwargs['result'] == args[0]:
                kwargs.pop('result')
            await original_on_message(self, *args, **kwargs)
        stt._on_message = logged_on_message.__get__(stt, DeepgramSTTService)
        
        initial_messages = [
            {"role": "system", "content": "You are a helpful healthcare AI assistant. Keep responses brief and clear. Always acknowledge what the user said."}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o"
        )
        
        # Create LLM context and aggregators for managing conversation history
        llm_context = OpenAILLMContext(messages=initial_messages)
        context_aggregators = llm.create_context_aggregator(llm_context)
        
        original_llm_process = llm.process_frame
        async def logged_llm_process(frame, direction):
            if hasattr(frame, 'text') and frame.text:
                logger.info(f"🤖 LLM got text: '{frame.text}'")
            try:
                result = await original_llm_process(frame, direction)
                if hasattr(result, 'text') and result.text:
                    logger.info(f"💬 LLM replied: '{result.text}'")
                return result
            except Exception as e:
                logger.error(f"LLM processing error: {str(e)}")
                raise  # Or handle as needed
        llm.process_frame = logged_llm_process
        
        # Temporarily bypass Cartesia TTS for testing
        # tts = CartesiaTTSService(
        #     api_key=os.getenv("CARTESIA_API_KEY"),
        #     voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
        #     model_id="sonic-english"
        # )
        # original_tts_process = tts.process_frame
        # async def logged_tts_process(frame, direction):
        #     if hasattr(frame, 'text') and frame.text:
        #         logger.info(f"🔊 TTS got text: '{frame.text}'")
        #     result = await original_tts_process(frame, direction)
        #     if hasattr(result, 'audio') and result.audio:
        #         logger.info(f"🎤 TTS created audio: {len(result.audio)} bytes")
        #     return result
        # tts.process_frame = logged_tts_process
        
        self.pipeline = Pipeline([
            self.transport.input(),
            AudioResampler(),  # Add resampling here to ensure correct format for downstream processors
            AudioFrameLogger(),
            DropEmptyAudio(),  # Add filter to drop empty frames before STT
            stt,
            IntermediateFrameLogger(),  # Insert here to monitor output from STT
            context_aggregators.user(),  # Aggregates user messages from STT into LLM context
            llm,
            # tts,  # Commented out to bypass TTS
            context_aggregators.assistant(),  # Aggregates assistant responses back into LLM context
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
        return ["Simple pipeline active"]
    
    async def run(self, url: str, token: str, room_name: str):
        if not self.pipeline:
            self.create_pipeline(url, token, room_name)
        
        task = PipelineTask(self.pipeline)
        self.runner = CustomPipelineRunner()
        
        logger.info(f"Starting simple pipeline for session: {self.session_id}")
        await self.runner.run(task)