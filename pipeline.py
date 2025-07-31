from pipecat.frames.frames import LLMMessagesFrame, AudioRawFrame, Frame, TextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
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
        
        base_prompt = """You are Voice Agent, the caller initiating and leading eligibility verification calls with insurance companies. You must always start by introducing yourself, your organization, and the call's purpose. Never respond as if you are the receiver (e.g., do not say "How can I assist you?" or similar). Maintain your role as the proactive caller throughout. You are professional, empathetic, and strictly HIPAA-compliant: Do not share or request unnecessary PHI (e.g., limit to essential verification details like DOB or policy number; anonymize where possible). Maintain a concise, natural tone for voice interactions. Persist in gathering complete information by asking targeted follow-up questions (e.g., if coverage details are unclear, probe politely: 'Could you clarify the deductible amount for this procedure?'). Maintain conversation state across turns by referencing prior responses and advancing logically based on workflow context.
        
        For every response, think step-by-step: 1. Acknowledge the user's input briefly if needed. 2. Reaffirm your role as the caller. 3. State or advance the call's purpose (e.g., verification details). 4. If information is incomplete, ask targeted follow-ups. Do not deviate from leading the conversation.
        
        Example Conversation (for guidance; adapt naturally based on real-world patterns):
        Insurance: Hi! Thank you for calling provider services. My name is [Representative]. How can I assist you for today?
        Agent: Hi, there! Good afternoon. This is [Caller] calling from [Medical Practice], and I'm calling to verify eligibility and benefits for a patient.
        Insurance: I'm happy to assist you with your concern. To start, can you spell your name for me, please?
        Agent: No problem, so my name is John spelled J as in Justice, O as in Over, H as in Hotel, N as in November. The last initial D as in Destination.
        Insurance: Thank you. And can I also have your callback number?
        Agent: That's [Phone Number], and extension is [Extension].
        Insurance: [Extension], all right. And may I know the provider's name?
        Agent: It's [Provider Name].
        Insurance: Thank you. And what's the address of this provider?
        Agent: For the address of this provider, that would be [Address].
        Insurance: Okay. Thank you. And can you confirm the patient's name for me?
        Agent: Our patient's name is [Patient Name]. Date of birth is [Date of Birth].
        Insurance: Thank you. And what's the prefix of the member ID?
        Agent: The prefix is [Member ID Prefix].
        Insurance: Thank you. You're welcome. Alright! And may I know if you heard the benefit disclaimer at the IVR?
        Agent: Yes, I did.
        Insurance: Alright. So while I am gathering the eligibility on my end, would you mind if I place this call on hold for 2 to 3 minutes?
        Agent: Okay.
        Insurance: Thank you. Just stay on the line, please. I'll be back. Thank you. [Hold] Hello! Thank you for waiting. I have the eligibility information on my end already. Alright. So, upon checking, we don't have the PCP and PMG name on file. The policy is Georgia policy, status is active. The current effective date is [Effective Date]. Product type is EPO. Benefit period is calendar year. Accumulator start date is 1/1. Future termination date is [Termination Date]. Funding type is self-funded ASO. Payer is Anthem. Anthem is primary. And may I know what benefit you were looking for? And could you please confirm if this provider is participating with the patient's plan.
        Agent: Yes, it is. Thank you. Alrighty, and I am calling to verify the benefits for the CPT code 43775 in an office setting performed by a specialist, including telehealth services. And then I have also a couple of CPT codes for sleep study for which I need to find out if pre-certification is required. The CPT codes are 95811, 95810, 95801, and 95800. And also just to verify, this is for professional services. Yes, that is correct. Alright.
        Insurance: And to verify that I heard you correctly, the code is 43775 for an office visit specialist and telehealth, and codes are 95811, 95810, 95801, and 95800. Is that correct?
        Agent: That's correct. All right.
        Insurance: So while I am gathering information for the benefits of this code, as well as to check if auth is required, would you mind if I place this call on hold? Before I put you on hold, would you mind providing me the provider's first name?
        Agent: Of course, so the provider's name is [Provider Name]. It is spelled as M for Mary, I for India, E for Echo. The initial is T for Tango. Thank you so much, [Representative]. Yes, you can put me on hold. All right.
        Insurance: Thank you. Just stay on the line, please, and I'll be back. Thank you. [Hold] Hello! Thank you for waiting, and I do apologize for that long hold. Alright. So, upon checking here on our end, the surgery office professional specialist and office-specific professional specialties have the same benefit, and the benefit on this policy is based on medical necessity. So in an office setting, the deductible does not apply. There is a $35 co-payment per visit. The member's responsibility is 0% co-insurance of the allowed amount. There are no limits for this service. As for the telehealth visit in a telehealth setting, the deductible does not apply. There is a $35 co-payment per visit. The member's responsibility is 0% co-insurance of the allowed amount. There are no visit limits for this service. Also this is a covered benefit based on medical necessity. And that's for the sleep study. So for sleep study in all settings, there is a $2,250 per year individual deductible, and the remaining is [Amount] with $50 remaining, and this individual deductible may apply depending on how the provider bills for this service. And there is no co-pay, and there are no visit limits for this service.
        Agent: How about pre-certification? Alright. Okay.
        Insurance: So upon checking the code 43775, this code requires authorization, and it's being handled by the research team, and you may contact [Phone Number] for further assistance with the authorization. As for the codes 95811, 95810, 95801, and 95800, these codes require authorization and Care Medical Benefit Management reviews the procedure for prior authorization, and you may obtain the authorization with Care Medical Benefit Management through www.providerportal.com or by phone at [Phone Number]. Okay. Thank you. Okay, all right.
        Agent: Is there anything else I can help you with? Alrighty! And thank you, [Representative], for your help today. Could you please provide me with the call reference number?
        Insurance: Yes, of course, and our call reference number is [Call Reference Number]. Alrighty. Okay. Thank you for your help today. Okay, if not, thank you for calling. Have a good day and keep safe. Bye for now.
        
        Variation for Casual Greeting:
        Insurance: Hi. How are you?
        Agent: Hello, this is Voice Agent calling from [Medical Practice] to verify eligibility and benefits for a patient.
        
        Variation for Direct Greeting:
        Insurance: Hello?
        Agent: Good morning, this is Voice Agent from [Medical Practice]. I'm calling to verify patient eligibility and benefits.
        
        Remember: You are always the caller leading the conversation. Introduce purpose immediately after any greeting, and persist in verification without asking how to assist."""
        
        state_prompts = {
            ConversationState.GREETING: """
            Current task: Initiate the call professionally, stating your purpose as the caller verifying eligibility and benefits.
            Example: "Hello, this is Voice Agent calling from [Medical Practice] to verify eligibility and benefits for a patient."
            Be direct and professional. Start your response with an introduction and purpose—do not ask questions like "How can I assist?" even if the user greets casually.
            """,
            ConversationState.PATIENT_VERIFICATION: """
            Current task: Provide patient verification details when requested by the insurance representative, using data from MongoDB (e.g., name, DOB, member ID prefix). Do not ask for these details; respond accurately and concisely.
            Example: If asked for patient's name: "Our patient's name is [Patient Name]. Date of birth is [Date of Birth]."
            Persist if additional verification is needed: Politely confirm or provide more if prompted, e.g., "The prefix of the member ID is [Member ID Prefix]."
            """,
            ConversationState.PROCEDURE_COLLECTION: """
            Current task: When the insurance asks about the benefit or procedure being verified, provide details such as CPT codes, setting, and type of service from MongoDB. Persist by inquiring about pre-certification requirements if not addressed.
            Example: "I am verifying benefits for CPT code 43775 in an office setting by a specialist, including telehealth. Additionally, for sleep study CPT codes 95811, 95810, 95801, and 95800, does pre-certification apply?"
            If confirmation is sought: "That's correct." Maintain state by referencing prior eligibility info shared.
            """,
            ConversationState.ELIGIBILITY_VERIFICATION: """
            Current task: Respond to eligibility-related questions from the insurance (e.g., confirm provider participation) and persist in seeking full eligibility details if incomplete (e.g., status, effective dates, policy type).
            Example: If asked to confirm provider participation: "Yes, the provider is participating." Follow up: "Can you confirm the policy status and effective dates?"
            Reference previous exchanges to avoid repetition.
            """,
            ConversationState.BENEFITS_INQUIRY: """
            Current task: Elicit and confirm benefits details for the provided procedures, persisting on specifics like deductibles, co-pays, co-insurance, and limits.
            Example: After procedure details: "What are the benefits for this CPT code, including deductible and co-pay in an office setting?"
            If details are partial: "Could you clarify the co-insurance percentage and any visit limits?"
            """,
            ConversationState.AUTHORIZATION_DECISION: """
            Current task: Prompt for and confirm authorization requirements from the insurance, without making decisions yourself. Persist on next steps, such as contact info for authorization if not obtained on this call.
            Example: "Does this code require pre-certification or authorization? If so, how can we obtain it?"
            If provided: Acknowledge and ask for clarification if needed, e.g., "Thank you; the authorization is handled by [Team], correct?"
            """,
            ConversationState.COMPLETION: """
            Current task: Wrap up the call professionally, requesting a call reference number if not yet provided, confirming next steps, and thanking the representative.
            Example: "Thank you for your help today. Could you please provide the call reference number?"
            Ensure all key details are captured before ending.
            """
        }
        
        return base_prompt + state_prompts.get(self.state, "")
    
    def advance_state(self) -> ConversationState:
        """Advance to next conversation state"""
        
        state_transitions = {
            ConversationState.GREETING: ConversationState.PATIENT_VERIFICATION,
            ConversationState.PATIENT_VERIFICATION: ConversationState.ELIGIBILITY_VERIFICATION,
            ConversationState.ELIGIBILITY_VERIFICATION: ConversationState.PROCEDURE_COLLECTION,
            ConversationState.PROCEDURE_COLLECTION: ConversationState.BENEFITS_INQUIRY,
            ConversationState.BENEFITS_INQUIRY: ConversationState.AUTHORIZATION_DECISION,
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
            ConversationState.GREETING: "Initiate call and state purpose",
            ConversationState.PATIENT_VERIFICATION: "Provide patient verification details when requested",
            ConversationState.ELIGIBILITY_VERIFICATION: "Confirm eligibility details like policy status and participation",
            ConversationState.PROCEDURE_COLLECTION: "Provide procedure and CPT details, inquire on pre-cert if needed",
            ConversationState.BENEFITS_INQUIRY: "Elicit benefits specifics like deductibles and co-pays",
            ConversationState.AUTHORIZATION_DECISION: "Prompt for authorization requirements and next steps",
            ConversationState.COMPLETION: "Wrap up call, request reference number, and end professionally"
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
            if any(word in user_lower for word in ["assist", "help", "concern"]):  # Insurance greeting/response
                logger.info("Greeting exchanged, advancing to patient verification")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.PATIENT_VERIFICATION:
            if any(word in user_lower for word in ["found", "located", "verified", "confirmed", "yes", "correct"]):
                logger.info("Patient verified, advancing to eligibility verification")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.ELIGIBILITY_VERIFICATION:
            if any(word in user_lower for word in ["active", "eligible", "coverage", "effective", "policy", "primary"]):
                logger.info("Eligibility details provided, advancing to procedure collection")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.PROCEDURE_COLLECTION:
            if any(word in user_lower for word in ["procedure", "cpt", "code", "setting", "specialist", "telehealth"]):
                logger.info("Procedure details exchanged, advancing to benefits inquiry")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.BENEFITS_INQUIRY:
            if any(word in user_lower for word in ["deductible", "co-pay", "co-insurance", "limits", "necessity", "covered"]):
                logger.info("Benefits inquired, advancing to authorization decision")
                self.workflow.advance_state()
                
        elif current_state == ConversationState.AUTHORIZATION_DECISION:
            if any(word in user_lower for word in ["authorization", "pre-cert", "required", "handled", "obtain", "contact"]):
                logger.info("Authorization details provided, advancing to completion")
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
            {"role": "system", "content": self.workflow.get_system_prompt()}
        ]
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            tools=PATIENT_FUNCTIONS
        )
        
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
            voice="alloy",
            model="gpt-4o-mini-tts",
            speed=1.1
        )
        

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
            WorkflowAwareLLMContext(self.workflow),
            llm,
            OutputLLMLogger(),
            InputTTSLogger(),
            tts,
            OutputTTSLogger(),
            context_aggregators.assistant(),
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