from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams

import os
import sys
import time
import logging
import asyncio
import numpy as np
from typing import Callable, Dict, Any, Optional, List

from functions import PATIENT_FUNCTIONS, FUNCTION_REGISTRY
from workflow import HealthcareWorkflow
from memory import ConversationMemory
from optimization import LatencyOptimizer
from interruption_handler import InterruptionHandler
from audio_processing import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPipelineRunner(PipelineRunner):
    def _setup_sigint(self):
        if sys.platform == 'win32':
            logger.warning("Signal handling not supported on Windows. Use task manager or endpoint to end sessions.")
            return
        super()._setup_sigint()

class HealthcareAIPipeline:
    def __init__(self, session_id: str = "default"):
        self.transport = None
        self.pipeline = None
        self.runner = None
        self.session_id = session_id
        
        # Initialize workflow and memory
        self.workflow = HealthcareWorkflow()
        self.memory = ConversationMemory()
        
        # Initialize optimization and interruption handling
        self.optimizer = LatencyOptimizer(target_latency_ms=800.0)
        self.interruption_handler = InterruptionHandler()
        self.audio_processor = AudioProcessor()
        
        # Track conversation state
        self.last_user_input_time = time.time()
        self.current_audio_chunk = None
        
    def create_pipeline(self, url: str, token: str, room_name: str) -> Pipeline:
        """Create the optimized Pipecat pipeline with healthcare AI components"""
        logger.debug(f"Creating pipeline with url: {url}, room_name: {room_name}, token: {token[:10]}...")
        
        # Transport configuration
        params = LiveKitParams(
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
        
        self.transport = LiveKitTransport(url, token, room_name, params=params)
        
        # STT Service
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            model="nova-2",
            language="en-US"
        )
        
        # Initialize conversation with system prompt
        system_prompt = self.workflow.get_system_prompt()
        self.memory.add_system_message(system_prompt)
        
        # LLM Service with function calling and optimization
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            messages=self.memory.get_messages_for_llm(),
            functions=PATIENT_FUNCTIONS,
            function_call="auto"
        )
        
        # Add function call handler
        llm.function_call_handler = self._handle_function_call
        
        # TTS Service
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
            model_id="sonic-english"
        )
        
        # Create optimized pipeline
        self.pipeline = Pipeline([
            self.transport.input(),
            self._add_audio_processing_wrapper(stt),
            self._add_optimization_wrapper(llm, "LLM"),
            self._add_interruption_wrapper(tts),
            self.transport.output()
        ])
        
        return self.pipeline
    
    def _add_audio_processing_wrapper(self, stt_service):
        """Add audio processing wrapper around STT service"""
        original_process = stt_service.process_frame
        
        async def audio_processed_stt(frame, direction):
            start_time = time.time()
            
            # Process audio if available
            if hasattr(frame, 'audio') and frame.audio is not None:
                try:
                    # Convert audio to numpy array
                    audio_data = np.frombuffer(frame.audio, dtype=np.float32)
                    
                    # Apply audio processing
                    processed_audio, audio_metrics = self.audio_processor.process_audio_chunk(audio_data)
                    
                    # Update frame with processed audio
                    frame.audio = processed_audio.tobytes()
                    
                    # Log audio quality
                    logger.debug(f"Audio quality: {audio_metrics.quality_score:.2f}")
                    
                except Exception as e:
                    logger.debug(f"Audio processing error: {e}")
            
            # Process with STT
            result = await original_process(frame, direction)
            
            # Track STT latency
            stt_latency = (time.time() - start_time) * 1000
            self.optimizer.track_latency("STT", stt_latency)
            
            return result
        
        stt_service.process_frame = audio_processed_stt
        return stt_service

    def _add_optimization_wrapper(self, llm_service, stage_name: str):
        """Add optimization wrapper around LLM service"""
        original_process = llm_service.process_frame
        
        async def optimized_llm_process(frame, direction):
            start_time = time.time()
            
            # Handle user input and conversation memory
            if hasattr(frame, 'text') and frame.text:
                user_input = frame.text
                self.last_user_input_time = time.time()
                
                # Check for interruptions
                interruption = self.interruption_handler.detect_interruption(
                    user_input, 
                    self.workflow.get_workflow_context()
                )
                
                if interruption:
                    # Handle interruption
                    strategy = self.interruption_handler.handle_interruption(
                        interruption, 
                        self.workflow.get_workflow_context()
                    )
                    
                    # Generate recovery response
                    recovery_response = self.interruption_handler.generate_recovery_response(
                        interruption, 
                        strategy, 
                        self.workflow.get_workflow_context()
                    )
                    
                    # Update memory with recovery response
                    self.memory.add_message("user", user_input)
                    self.memory.add_message("assistant", recovery_response)
                    
                    # Update system prompt based on interruption
                    updated_prompt = self.workflow.get_system_prompt()
                    self.memory.add_system_message(updated_prompt)
                    
                    logger.info(f"Interruption handled: {recovery_response[:50]}...")
                
                else:
                    # Normal processing
                    self.memory.add_message("user", user_input)
                
                # Check for function call suggestions
                suggested_function = self.workflow.should_use_function(user_input)
                if suggested_function:
                    logger.info(f"Workflow suggests function: {suggested_function}")
                
                # Get optimization settings for current state
                optimization_settings = self.optimizer.optimize_for_state(
                    self.workflow.state.value
                )
                
                # Apply optimization if enabled
                if optimization_settings.get("enable_parallel_processing"):
                    # Prepare LLM context in parallel
                    context_prep = await self.optimizer._prepare_llm_context(
                        self.workflow.get_workflow_context()
                    )
                    logger.debug("Applied parallel context preparation")
                
                # Update LLM messages with current conversation
                llm_service.messages = self.memory.get_messages_for_llm()
                
                # Check for cached responses
                if optimization_settings.get("cache_responses"):
                    cache_key = f"response_{self.workflow.state.value}_{hash(user_input[:20])}"
                    # Simplified caching logic
            
            # Process with LLM
            result = await original_process(frame, direction)
            
            # Add assistant response to memory
            if hasattr(result, 'text') and result.text:
                self.memory.add_message("assistant", result.text)
                
                # Mark AI as speaking for interruption detection
                self.interruption_handler.start_ai_response(result.text)
            
            # Track LLM latency
            llm_latency = (time.time() - start_time) * 1000
            self.optimizer.track_latency("LLM", llm_latency)
            
            return result
        
        llm_service.process_frame = optimized_llm_process
        return llm_service

    def _add_interruption_wrapper(self, tts_service):
        """Add interruption handling wrapper around TTS service"""
        original_process = tts_service.process_frame
        
        async def interruption_aware_tts(frame, direction):
            start_time = time.time()
            
            # Check for silence timeout
            if self.interruption_handler.check_silence_timeout(self.last_user_input_time):
                silence_prompt = self.interruption_handler.generate_silence_prompt(
                    self.workflow.get_workflow_context()
                )
                logger.info(f"Silence timeout - prompting: {silence_prompt}")
                
                # Update frame with silence prompt
                if hasattr(frame, 'text'):
                    frame.text = silence_prompt
            
            # Get response speed adjustments
            adjustments = self.interruption_handler.adjust_response_speed(
                len(self.interruption_handler.interruption_history)
            )
            
            # Apply TTS adjustments
            if adjustments.get("speech_rate") == "slower":
                logger.debug("Applying slower speech rate")
            
            # Process with TTS
            result = await original_process(frame, direction)
            
            # Track TTS latency
            tts_latency = (time.time() - start_time) * 1000
            self.optimizer.track_latency("TTS", tts_latency)
            
            return result
        
        tts_service.process_frame = interruption_aware_tts
        return tts_service
    
    async def _handle_function_call(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Handle LLM function calls with optimization and caching"""
        start_time = time.time()
        
        try:
            # Check for cached function results
            cached_result = self.optimizer.get_cached_function_result(function_name, arguments)
            if cached_result is not None:
                logger.info(f"Using cached result for {function_name}")
                return str(cached_result)
            
            if function_name in FUNCTION_REGISTRY:
                logger.info(f"Calling function: {function_name} with args: {arguments}")
                
                # Execute the function
                result = await FUNCTION_REGISTRY[function_name](**arguments)
                
                # Cache the result
                self.optimizer.cache_function_result(function_name, arguments, result)
                
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
                
                # Track function call latency
                function_latency = (time.time() - start_time) * 1000
                self.optimizer.track_latency("FUNCTION", function_latency)
                
                return str(result) if result else "Function executed successfully"
            else:
                logger.error(f"Unknown function: {function_name}")
                return f"Error: Unknown function {function_name}"
                
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            return f"Error executing function: {str(e)}"
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state with optimization metrics"""
        base_state = {
            "workflow_state": self.workflow.state.value,
            "workflow_context": self.workflow.get_workflow_context(),
            "conversation_summary": self.memory.get_conversation_summary(),
            "session_id": self.session_id
        }
        
        # Add optimization metrics
        base_state["performance_metrics"] = self.optimizer.get_performance_summary()
        base_state["interruption_analytics"] = self.interruption_handler.get_interruption_analytics()
        base_state["audio_quality"] = self.audio_processor.get_audio_quality_summary()
        
        return base_state
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for improving performance"""
        suggestions = []
        
        # Performance suggestions
        perf_summary = self.optimizer.get_performance_summary()
        if perf_summary.get("performance_ratio", 0) > 1.2:
            suggestions.append("Consider optimizing LLM calls - latency exceeds target")
        
        # Audio suggestions
        audio_suggestions = self.audio_processor.suggest_audio_improvements()
        suggestions.extend(audio_suggestions)
        
        # Interruption suggestions
        interruption_analytics = self.interruption_handler.get_interruption_analytics()
        if interruption_analytics.get("recent_interruptions", 0) > 3:
            suggestions.append("High interruption rate - consider adjusting response pace")
        
        return suggestions
    
    async def run(self, url: str, token: str, room_name: str):
        """Run the optimized pipeline with given LiveKit room and token"""
        if not self.pipeline:
            self.create_pipeline(url, token, room_name)
        
        task = PipelineTask(self.pipeline)
        self.runner = CustomPipelineRunner()
        
        logger.info(f"Starting optimized healthcare AI pipeline for session: {self.session_id}")
        await self.runner.run(task)