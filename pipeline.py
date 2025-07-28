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
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('websockets').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Workflow State Management
class ConversationState(Enum):
    """Insurance verification call workflow states"""
    INITIAL_RESPONSE = "initial_response"
    PATIENT_INFO_REQUEST = "patient_info_request"
    ELIGIBILITY_VERIFICATION = "eligibility_verification"
    BENEFITS_INQUIRY = "benefits_inquiry"
    COMPLETION = "completion"

class HealthcareWorkflow:
    """Simple workflow manager for prior authorization conversations"""
    
    def __init__(self, patient_id: str = None):
        self.state = ConversationState.INITIAL_RESPONSE
        self.patient_id = patient_id  # Store patient ID for function calls
        self.patient_data = None
        self.collected_info = {}
        
    def get_system_prompt(self) -> str:
        """Get system prompt based on current conversation state"""
        
        base_prompt = """You are MyRobot, a healthcare AI assistant for prior authorization calls. 
                         You are professional, empathetic, and HIPAA-compliant. Keep responses concise for voice interaction."""
        
        state_prompts = {
            ConversationState.GREETING: """
Current task: You have just called about a prior authorization request. Start the conversation professionally.
Say something like: "Hello, I'm MyRobot calling about a prior authorization request. May I have the patient's full name please?"
Be direct and professional - you are the one calling them.
""",
            
            ConversationState.PATIENT_VERIFICATION: """
Current task: You have the patient's name. Acknowledge it and ask for verification details.
Ask for date of birth or other identifying information to verify this is the correct patient.
Example: "Thank you. Can you please confirm the patient's date of birth for verification?"
""",
            
            ConversationState.PROCEDURE_COLLECTION: """
Current task: Patient is verified. Now collect information about the medical procedure requiring authorization.
Ask about: specific procedure, medical necessity, symptoms, and when it's needed.
Keep questions natural and conversational. Example: "Great, now can you tell me what procedure needs authorization?"
""",
            
            ConversationState.AUTHORIZATION_DECISION: """
Current task: You have the procedure information. Make an authorization decision.
Provide a clear decision (approved/denied/needs more info) and explain next steps.
Example: "Based on the information provided, I can approve this authorization. Your reference number is..."
""",
            
            ConversationState.COMPLETION: """
Current task: Wrap up the call professionally.
Provide any reference numbers, confirm next steps, and thank them.
Example: "Is there anything else I can help you with regarding this authorization? Thank you for your time."
"""
        }
        
        return base_prompt + state_prompts.get(self.state, "")
    
    def advance_state(self, trigger: str = None) -> ConversationState:
        """Advance to next conversation state based on trigger"""
        
        state_transitions = {
            ConversationState.GREETING: ConversationState.PATIENT_VERIFICATION,
            ConversationState.PATIENT_VERIFICATION: ConversationState.PROCEDURE_COLLECTION,
            ConversationState.PROCEDURE_COLLECTION: ConversationState.AUTHORIZATION_DECISION,
            ConversationState.AUTHORIZATION_DECISION: ConversationState.COMPLETION,
            ConversationState.COMPLETION: ConversationState.COMPLETION  # Stay at completion
        }
        
        previous_state = self.state
        self.state = state_transitions.get(self.state, self.state)
        
        logger.info(f"Workflow state: {previous_state.value} -> {self.state.value}")
        return self.state
    
    def update_patient_data(self, patient_data: Dict[str, Any]):
        """Update stored patient data"""
        self.patient_data = patient_data
        logger.info(f"Updated patient data for: {patient_data.get('patient_name', 'Unknown')}")
    
    def add_collected_info(self, key: str, value: Any):
        """Add collected information to workflow state"""
        self.collected_info[key] = value
        logger.info(f"Added to workflow: {key} = {value}")
    
    def get_workflow_context(self) -> Dict[str, Any]:
        """Get current workflow context for LLM"""
        return {
            "current_state": self.state.value,
            "patient_data": self.patient_data,
            "collected_info": self.collected_info,
            "next_action": self._get_next_action()
        }
    
    def _get_next_action(self) -> str:
        """Get description of what should happen next"""
        actions = {
            ConversationState.INITIAL_RESPONSE: "Introduce yourself and state purpose",
            ConversationState.PATIENT_INFO_REQUEST: "Provide patient information when requested",
            ConversationState.ELIGIBILITY_VERIFICATION: "Verify patient eligibility status",
            ConversationState.BENEFITS_INQUIRY: "Inquire about specific benefits and coverage",
            ConversationState.COMPLETION: "End call professionally"
        }
        return actions.get(self.state, "Continue conversation")
    
    def should_use_function(self, user_message: str) -> Optional[str]:
        """Determine if a function should be called based on conversation state and user input"""
        
        # Simple keyword-based function triggering
        if self.state == ConversationState.PATIENT_VERIFICATION:
            if any(word in user_message.lower() for word in ["name is", "patient is", "my name"]):
                return "search_patient_by_name"
        
        if self.state == ConversationState.AUTHORIZATION_DECISION:
            if self.patient_data and "collected_info" in str(self.collected_info):
                return "update_prior_auth_status"
        
        return None
    
    def reset(self):
        """Reset workflow to initial state"""
        self.state = ConversationState.GREETING
        self.patient_data = None
        self.collected_info = {}
        logger.info("Workflow reset to initial state")

# Frame Processors (keeping all existing ones)
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
            logger.info(f"🤖 LLM input messages ({len(frame.messages)} total):")
            for i, msg in enumerate(frame.messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200] + "..." if len(msg.get('content', '')) > 200 else msg.get('content', '')
                logger.info(f"   [{i}] {role}: {content}")
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

class FunctionCallHandler(FrameProcessor):
    """Processor that handles function calls from the LLM"""
    def __init__(self, patient_id: str = None):
        super().__init__()
        self.patient_id = patient_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle function calls from LLM
        if hasattr(frame, 'function_calls') and frame.function_calls:
            logger.info(f"🔧 Processing function calls: {frame.function_calls}")
            
            for function_call in frame.function_calls:
                function_name = function_call.get('name')
                function_args = function_call.get('arguments', {})
                
                # Auto-inject patient_id if function needs it and it's not provided
                if function_name in ['get_facility_name', 'get_patient_demographics', 'get_patient_insurance_info', 'get_patient_medical_info', 'get_provider_info']:
                    if 'patient_id' not in function_args and self.patient_id:
                        function_args['patient_id'] = self.patient_id
                        logger.info(f"Auto-injected patient_id: {self.patient_id}")
                
                if function_name in FUNCTION_REGISTRY:
                    try:
                        # Call the function
                        result = await FUNCTION_REGISTRY[function_name](**function_args)
                        logger.info(f"✅ Function {function_name} returned: {result}")
                        
                        # Create a function result frame (if your pipecat version supports it)
                        # Otherwise, you might need to inject this back into the conversation context
                        
                    except Exception as e:
                        logger.error(f"❌ Function {function_name} failed: {e}")
                        result = f"Error: {str(e)}"
        
        await self.push_frame(frame, direction)
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

class SimpleLangfuseLogger(FrameProcessor):
    """Processor that injects workflow context into LLM messages and advances workflow"""
    def __init__(self, workflow: HealthcareWorkflow):
        super().__init__()
        self.workflow = workflow
        self.last_user_message = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame) and frame.messages:
            # Store user message for workflow advancement
            messages = frame.messages.copy()
            
            # Find the last user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    self.last_user_message = msg.get("content", "")
                    break
            
            # Check if we should advance workflow state based on user input
            self._maybe_advance_workflow(self.last_user_message)
            
            # Update system message with current workflow state
            system_prompt = self.workflow.get_system_prompt()
            workflow_context = self.workflow.get_workflow_context()
            
            enhanced_system_prompt = f"""{system_prompt}

IMPORTANT CONTEXT:
- Current state: {workflow_context['current_state']}
- Next action: {workflow_context['next_action']}
- Patient data: {workflow_context.get('patient_data', 'None found yet')}
- Collected info: {workflow_context.get('collected_info', 'None collected yet')}

Remember: You are calling THEM. Be professional and direct about your purpose."""
            
            # Find and replace system message
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i] = {"role": "system", "content": enhanced_system_prompt}
                    break
            else:
                # No system message found, add one
                messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
            
            # Create new frame with updated messages
            new_frame = LLMMessagesFrame(messages=messages)
            await self.push_frame(new_frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    def _maybe_advance_workflow(self, user_message: str):
        """Advance workflow based on user input patterns"""
        if not user_message:
            return
            
        user_lower = user_message.lower()
        current_state = self.workflow.state
        
        # Auto-advance workflow based on conversation patterns
        if current_state == ConversationState.INITIAL_RESPONSE:
            # If user asks for patient info or responds positively, move forward
            if any(word in user_lower for word in ["patient", "name", "member", "information", "verify", "sure", "go ahead"]):
                logger.info("User asking for patient info, advancing to patient info request")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.PATIENT_INFO_REQUEST:
            # If user acknowledges patient info or asks about eligibility
            if any(word in user_lower for word in ["found", "located", "active", "eligible", "coverage", "effective"]):
                logger.info("User confirmed patient, advancing to eligibility verification")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.ELIGIBILITY_VERIFICATION:
            # If user provides eligibility info or asks about benefits
            if any(word in user_lower for word in ["active", "covered", "benefits", "copay", "deductible", "procedure"]):
                logger.info("Moving to benefits inquiry")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.BENEFITS_INQUIRY:
            # After getting benefits info, wrap up
            if any(word in user_lower for word in ["covered", "approved", "that's all", "anything else", "help"]):
                logger.info("Benefits verified, moving to completion")
                self.workflow.advance_state()

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default", patient_id: str = None):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        self.patient_id = patient_id  # Store patient ID
        self.langfuse = None
        self.workflow = HealthcareWorkflow(patient_id=patient_id)  # Pass patient_id to workflow
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
        logger.info(f"Creating healthcare pipeline for room: {room_name}")
        
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            audio_in_enabled=True,
            audio_out_enabled=True,
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
        
        # Initial system message will be enhanced by workflow - START WITH GREETING
        initial_messages = [
            {"role": "system", "content": self.workflow.get_system_prompt()}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            tools=PATIENT_FUNCTIONS  # Enable function calling
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
                    model="sonic-2",
                    cartesia_version="2025-04-16"
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
                    voice="alloy"
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
            WorkflowAwareLLMContext(self.workflow),  # Inject workflow context
            LLMInputLogger(),
            llm,
            FunctionCallHandler(patient_id=self.patient_id),  # Handle function calls
            LLMOutputLogger(),
            TTSInputLogger(),
            tts,
            AudioOutputLogger(),
            context_aggregators.assistant(),
            TransportOutputLogger(),
            self.transport.output()
        ])
        
        logger.info("Healthcare pipeline created successfully with workflow integration")
        return self.pipeline
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state including workflow"""
        return {
            "session_id": self.session_id,
            "workflow_state": self.workflow.state.value,
            "workflow_context": self.workflow.get_workflow_context(),
            "patient_data": self.workflow.patient_data,
            "collected_info": self.workflow.collected_info
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        return [
            f"Healthcare pipeline active with workflow state: {self.workflow.state.value}",
            f"Next action: {self.workflow._get_next_action()}"
        ]
    
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
            enable_tracing=False,
            conversation_id=self.session_id
        )
        
        self.runner = CustomPipelineRunner()
        
        # Log session start to Langfuse
        if self.langfuse:
            try:
                self.langfuse.event(
                    name="session_start",
                    session_id=self.session_id,
                    metadata={
                        "room_name": room_name, 
                        "workflow_state": self.workflow.state.value,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log session start: {e}")
        
        logger.info(f"Starting healthcare pipeline for session: {self.session_id}")
        
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