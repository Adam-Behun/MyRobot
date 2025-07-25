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
from typing import Callable, Dict, Any

from functions import PATIENT_FUNCTIONS, FUNCTION_REGISTRY
from workflow import HealthcareWorkflow
from memory import ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default"):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        
        # Initialize workflow and memory
        self.workflow = HealthcareWorkflow()
        self.memory = ConversationMemory()
        
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
        
        # Initialize conversation with system prompt
        system_prompt = self.workflow.get_system_prompt()
        self.memory.add_system_message(system_prompt)
        
        # LLM Service with function calling
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            messages=self.memory.get_messages_for_llm(),
            functions=PATIENT_FUNCTIONS,  # Enable function calling
            function_call="auto"
        )
        
        # Add function call handler
        llm.function_call_handler = self._handle_function_call
        
        # TTS Service
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Professional female voice
            model_id="sonic-english"
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            self.transport.input(),
            stt,
            self._add_latency_logging(llm, "LLM"),
            tts,
            self.transport.output()
        ])
        
        return self.pipeline
    
    def _handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Handle LLM function calls"""
        start_time = time.time()
        
        try:
            if function_name in FUNCTION_REGISTRY:
                logger.info(f"Calling function: {function_name} with args: {arguments}")
                
                # Execute the function
                result = FUNCTION_REGISTRY[function_name](**arguments)
                
                # Update workflow state based on function results
                if function_name == "search_patient_by_name" and result:
                    self.workflow.update_patient_data(result)
                    self.workflow.advance_state()
                elif function_name == "update_prior_auth_status":
                    self.workflow.add_collected_info("auth_status_updated", result)
                    self.workflow.advance_state()
                
                # Update system prompt for new workflow state
                new_prompt = self.workflow.get_system_prompt()
                self.memory.add_system_message(new_prompt)
                
                latency = (time.time() - start_time) * 1000
                logger.info(f"Function call latency: {latency:.2f}ms")
                
                return str(result) if result else "Function executed successfully"
            else:
                logger.error(f"Unknown function: {function_name}")
                return f"Error: Unknown function {function_name}"
                
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            return f"Error executing function: {str(e)}"
    
    def _add_latency_logging(self, service, stage_name: str):
        """Add latency logging wrapper around service"""
        original_process = service.process_frame
        
        async def logged_process(frame):
            start_time = time.time()
            
            # Add conversation memory handling for user messages
            if hasattr(frame, 'text') and frame.text:
                self.memory.add_message("user", frame.text)
                
                # Check if workflow suggests function call
                suggested_function = self.workflow.should_use_function(frame.text)
                if suggested_function:
                    logger.info(f"Workflow suggests function: {suggested_function}")
            
            result = await original_process(frame)
            
            # Add assistant response to memory
            if hasattr(result, 'text') and result.text:
                self.memory.add_message("assistant", result.text)
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"{stage_name} latency: {latency:.2f}ms")
            return result
        
        service.process_frame = logged_process
        return service
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state"""
        return {
            "workflow_state": self.workflow.state.value,
            "workflow_context": self.workflow.get_workflow_context(),
            "conversation_summary": self.memory.get_conversation_summary(),
            "session_id": self.session_id
        }
    
    async def run(self, room_url: str, token: str):
        """Run the pipeline with given LiveKit room and token"""
        if not self.pipeline:
            self.create_pipeline()
        
        # Update transport token
        self.transport._params.token = token
        
        task = PipelineTask(self.pipeline)
        self.runner = PipelineRunner()
        
        logger.info(f"Starting healthcare AI pipeline for session: {self.session_id}")
        await self.runner.run(task)