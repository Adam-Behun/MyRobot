# MyRobot calls insurance companies to verify patient eligibility and benefits
# For demo purposes, I (human) simulate the insurance agent by going to this website: test_client.html
# I load available patient from MongoDB, select a patient, select "Start Call for Patient", then, i "Connect to Room"
# When I join the room, I start the conversation just like a real insurance agent would by introducing myself
# Something like "Hello, this is United Healthcare, my name is Jessica, how can I help?"
# After that, MyRobot navigates the conversation flow
# Introduces itself - "Hi Jessica, I'm My Robot, I'm calling on behalf of {facility name pulled from the database patient's record} to verify eligibility and benefits for a patient"
# Jessica from UHC says something like "Sure, I can help with that, what is the patient's name?"
# MyRobot already has the patient name as I selected it in the UI, so it says "The patient's name is John Doe" 
#   It is important that during one phone call MyRobot discusses only one patient
# Jessica says something like "Ok, can you provide the patient's date of birth for verification?"
# MyRobot retrieves the patient's DOB from MongoDB and says "Sure, the patient's date of birth is {DOB}"

from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame, TextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
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
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum
import audioop

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Workflow State Management
class ConversationState(Enum):
    """Insurance verification call workflow states"""
    GREETING = "greeting"
    PATIENT_VERIFICATION = "patient_verification"
    PROCEDURE_COLLECTION = "procedure_collection"
    ELIGIBILITY_VERIFICATION = "eligibility_verification"
    BENEFITS_INQUIRY = "benefits_inquiry"
    AUTHORIZATION_DECISION = "authorization_decision"
    COMPLETION = "completion"

class HealthcareWorkflow:
    """Simple workflow manager for prior authorization conversations"""
    
    def __init__(self, patient_id: str = None):
        self.state = ConversationState.GREETING
        self.patient_id = patient_id
        self.patient_data = None
        self.collected_info = {}
        
    def get_system_prompt(self) -> str:
        """Get system prompt based on current conversation state"""
        
        base_prompt = """You are Voice Agent, a healthcare AI assistant for eligibility verification calls. 
                         You are professional, empathetic, and HIPAA-compliant. Keep responses concise for voice interaction."""
        
        state_prompts = {
            ConversationState.GREETING: """
            Current task: You have just called about a prior authorization request. Start the conversation professionally.
            Say something like: "Hello, I'm Voice Agent calling to verify eligibility and benefits for a patient."
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
    
    def advance_state(self) -> ConversationState:
        """Advance to next conversation state"""
        
        state_transitions = {
            ConversationState.GREETING: ConversationState.PATIENT_VERIFICATION,
            ConversationState.PATIENT_VERIFICATION: ConversationState.PROCEDURE_COLLECTION,
            ConversationState.PROCEDURE_COLLECTION: ConversationState.AUTHORIZATION_DECISION,
            ConversationState.AUTHORIZATION_DECISION: ConversationState.COMPLETION,
            ConversationState.COMPLETION: ConversationState.COMPLETION
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
            ConversationState.GREETING: "Introduce yourself and ask for patient name",
            ConversationState.PATIENT_VERIFICATION: "Verify patient identity and search for patient",
            ConversationState.PROCEDURE_COLLECTION: "Collect procedure information",
            ConversationState.AUTHORIZATION_DECISION: "Make authorization decision",
            ConversationState.COMPLETION: "End call professionally"
        }
        return actions.get(self.state, "Continue conversation")

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
    """Processor that injects workflow context into LLM messages and advances workflow"""
    def __init__(self, workflow: HealthcareWorkflow):
        super().__init__()
        self.workflow = workflow
        self.last_user_message = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame) and frame.messages:
            messages = frame.messages.copy()
            
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    self.last_user_message = msg.get("content", "")
                    break
            
            self._maybe_advance_workflow(self.last_user_message)
            
            system_prompt = self.workflow.get_system_prompt()
            workflow_context = self.workflow.get_workflow_context()
            
            enhanced_system_prompt = f"""{system_prompt}

IMPORTANT CONTEXT:
- Current state: {workflow_context['current_state']}
- Next action: {workflow_context['next_action']}
- Patient data: {workflow_context.get('patient_data', 'None found yet')}
- Collected info: {workflow_context.get('collected_info', 'None collected yet')}

Remember: You are calling THEM. Be professional and direct about your purpose."""
            
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i] = {"role": "system", "content": enhanced_system_prompt}
                    break
            else:
                messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
            
            new_frame = LLMMessagesFrame(messages=messages)
            await self.push_frame(new_frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    def _maybe_advance_workflow(self, user_message: str):
        if not user_message:
            return
            
        user_lower = user_message.lower()
        current_state = self.workflow.state
        
        if current_state == ConversationState.GREETING:
            if any(word in user_lower for word in ["patient", "name", "is", "it's", "the patient"]):
                logger.info("User provided patient name, advancing to patient verification")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.PATIENT_VERIFICATION:
            if any(word in user_lower for word in ["found", "located", "active", "eligible", "coverage", "effective", "yes", "correct"]):
                logger.info("User confirmed patient, advancing to procedure collection")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.PROCEDURE_COLLECTION:
            if any(word in user_lower for word in ["procedure", "surgery", "treatment", "test", "scan", "therapy"]):
                logger.info("Moving to authorization decision")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.AUTHORIZATION_DECISION:
            if any(word in user_lower for word in ["approved", "denied", "thank", "anything else", "that's all"]):
                logger.info("Decision made, moving to completion")
                self.workflow.advance_state()

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default", patient_id: str = None):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        self.patient_id = patient_id
        self.workflow = HealthcareWorkflow(patient_id=patient_id)
        
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
                endpointing=300,
                vad_events=True,
                smart_format=True,
                punctuate=True,
                profanity_filter=False
            )
        )
        
        initial_messages = [
            {"role": "system", "content": self.workflow.get_system_prompt()}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            tools=PATIENT_FUNCTIONS
        )
        
        # Register function handlers with wrapper for result callback
        def create_handler(handler):
            async def wrapped(params: FunctionCallParams, **kwargs):
                result = await handler(**kwargs)
                await params.result_callback(result)
            return wrapped
        
        for name, handler in FUNCTION_REGISTRY.items():
            llm.register_function(name, create_handler(handler))
        
        llm_context = OpenAILLMContext(messages=initial_messages)
        context_aggregators = llm.create_context_aggregator(llm_context)
        
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="alloy"
        )
        
        # ********** DEBUG PIPELINE INTEGRATION **********
        # INSERTING DEBUG LOGGERS INTO THE PIPELINE FOR TRACING
        # THESE WILL OUTPUT TO TERMINAL FOR REAL-TIME DEBUGGING
        # REMOVE THESE IN PRODUCTION TO AVOID PERFORMANCE OVERHEAD
        # ********** DEBUG PIPELINE INTEGRATION **********
        self.pipeline = Pipeline([
            self.transport.input(),
            AudioResampler(),
            DropEmptyAudio(),
            InputAudioLogger(),
            stt,
            OutputSTTLogger(),
            context_aggregators.user(),
            InputLLMLogger(),
            WorkflowAwareLLMContext(self.workflow),  # Move this AFTER user context
            llm,
            OutputLLMLogger(),
            InputTTSLogger(),
            tts,
            OutputTTSLogger(),
            context_aggregators.assistant(),  # Move this AFTER TTS
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