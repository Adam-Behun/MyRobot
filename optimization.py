import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Track latency metrics for optimization"""
    stt_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    total_latency: float = 0.0
    function_call_latency: float = 0.0
    timestamp: float = 0.0

class LatencyOptimizer:
    """Optimize pipeline latency through parallel processing and caching"""
    
    def __init__(self, target_latency_ms: float = 800.0):
        self.target_latency_ms = target_latency_ms
        self.metrics_history: List[LatencyMetrics] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Simple caching for common responses
        self.response_cache: Dict[str, str] = {}
        self.function_cache: Dict[str, Any] = {}
        
    def track_latency(self, stage: str, latency_ms: float, context: Dict[str, Any] = None):
        """Track latency for a specific stage"""
        
        # Update current metrics
        if not self.metrics_history or self.metrics_history[-1].timestamp < time.time() - 5:
            self.metrics_history.append(LatencyMetrics(timestamp=time.time()))
        
        current_metrics = self.metrics_history[-1]
        
        if stage == "STT":
            current_metrics.stt_latency = latency_ms
        elif stage == "LLM":
            current_metrics.llm_latency = latency_ms
        elif stage == "TTS":
            current_metrics.tts_latency = latency_ms
        elif stage == "FUNCTION":
            current_metrics.function_call_latency = latency_ms
        
        # Calculate total latency
        current_metrics.total_latency = (
            current_metrics.stt_latency + 
            current_metrics.llm_latency + 
            current_metrics.tts_latency +
            current_metrics.function_call_latency
        )
        
        # Log if exceeding target
        if current_metrics.total_latency > self.target_latency_ms:
            logger.warning(f"Latency exceeded target: {current_metrics.total_latency:.2f}ms > {self.target_latency_ms}ms")
        
        logger.info(f"{stage} latency: {latency_ms:.2f}ms | Total: {current_metrics.total_latency:.2f}ms")
    
    async def parallel_stt_llm_prep(self, audio_data: Any, conversation_context: Dict[str, Any]):
        """Prepare LLM context while STT is processing"""
        try:
            # Start both operations in parallel
            stt_task = asyncio.create_task(self._process_stt(audio_data))
            context_task = asyncio.create_task(self._prepare_llm_context(conversation_context))
            
            # Wait for both to complete
            stt_result, llm_context = await asyncio.gather(stt_task, context_task)
            
            return {
                "transcript": stt_result,
                "prepared_context": llm_context
            }
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            return None
    
    async def _process_stt(self, audio_data: Any) -> str:
        """Placeholder for STT processing"""
        # This would integrate with actual STT service
        await asyncio.sleep(0.1)  # Simulate STT processing
        return "placeholder_transcript"
    
    async def _prepare_llm_context(self, conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare LLM context while STT is running"""
        start_time = time.time()
        
        try:
            # Prepare system prompt based on conversation state
            system_prompt = self._build_optimized_prompt(conversation_context)
            
            # Prepare function definitions if needed
            functions = self._get_relevant_functions(conversation_context)
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"Context preparation latency: {latency:.2f}ms")
            
            return {
                "system_prompt": system_prompt,
                "functions": functions,
                "prepared_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error preparing LLM context: {e}")
            return {}
    
    def _build_optimized_prompt(self, context: Dict[str, Any]) -> str:
        """Build optimized system prompt based on conversation state"""
        
        # Cache common prompts to reduce processing
        state = context.get("workflow_state", "greeting")
        cache_key = f"prompt_{state}"
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Build prompt based on state (simplified)
        base_prompt = "You are MyRobot, a healthcare AI for prior authorization. "
        
        state_additions = {
            "greeting": "Greet the caller and ask for patient name.",
            "patient_verification": "Search for the patient and verify details.",
            "procedure_collection": "Collect procedure information efficiently.",
            "authorization_decision": "Make authorization decision and update status.",
            "completion": "Wrap up the call professionally."
        }
        
        optimized_prompt = base_prompt + state_additions.get(state, "Continue conversation naturally.")
        
        # Cache for future use
        self.response_cache[cache_key] = optimized_prompt
        
        return optimized_prompt
    
    def _get_relevant_functions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return only relevant functions based on conversation state"""
        
        state = context.get("workflow_state", "greeting")
        
        # Only provide functions relevant to current state
        if state == "patient_verification":
            return [{"name": "search_patient_by_name"}]
        elif state == "authorization_decision":
            return [{"name": "update_prior_auth_status"}]
        else:
            return []
    
    def cache_function_result(self, function_name: str, args: Dict[str, Any], result: Any):
        """Cache function results to avoid repeated database calls"""
        
        # Create cache key from function name and args
        cache_key = f"{function_name}_{hash(str(sorted(args.items())))}"
        
        self.function_cache[cache_key] = {
            "result": result,
            "cached_at": time.time()
        }
        
        logger.info(f"Cached function result: {function_name}")
    
    def get_cached_function_result(self, function_name: str, args: Dict[str, Any]) -> Optional[Any]:
        """Get cached function result if available and fresh"""
        
        cache_key = f"{function_name}_{hash(str(sorted(args.items())))}"
        
        if cache_key in self.function_cache:
            cached = self.function_cache[cache_key]
            
            # Check if cache is still fresh (5 minutes)
            if time.time() - cached["cached_at"] < 300:
                logger.info(f"Using cached result for: {function_name}")
                return cached["result"]
            else:
                # Remove stale cache
                del self.function_cache[cache_key]
        
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_total = sum(m.total_latency for m in recent_metrics) / len(recent_metrics)
        avg_stt = sum(m.stt_latency for m in recent_metrics) / len(recent_metrics)
        avg_llm = sum(m.llm_latency for m in recent_metrics) / len(recent_metrics)
        avg_tts = sum(m.tts_latency for m in recent_metrics) / len(recent_metrics)
        
        return {
            "target_latency_ms": self.target_latency_ms,
            "average_total_latency_ms": avg_total,
            "average_stt_latency_ms": avg_stt,
            "average_llm_latency_ms": avg_llm,
            "average_tts_latency_ms": avg_tts,
            "performance_ratio": avg_total / self.target_latency_ms,
            "cache_hit_rate": len(self.response_cache) / max(len(self.metrics_history), 1),
            "measurements_count": len(self.metrics_history)
        }
    
    def optimize_for_state(self, workflow_state: str) -> Dict[str, Any]:
        """Return optimization settings for specific workflow state"""
        
        optimizations = {
            "greeting": {
                "enable_parallel_processing": False,
                "cache_responses": True,
                "functions_enabled": False
            },
            "patient_verification": {
                "enable_parallel_processing": True,
                "cache_responses": False,
                "functions_enabled": True,
                "priority_functions": ["search_patient_by_name"]
            },
            "procedure_collection": {
                "enable_parallel_processing": True,
                "cache_responses": False,
                "functions_enabled": False
            },
            "authorization_decision": {
                "enable_parallel_processing": True,
                "cache_responses": False,
                "functions_enabled": True,
                "priority_functions": ["update_prior_auth_status"]
            },
            "completion": {
                "enable_parallel_processing": False,
                "cache_responses": True,
                "functions_enabled": False
            }
        }
        
        return optimizations.get(workflow_state, {
            "enable_parallel_processing": True,
            "cache_responses": False,
            "functions_enabled": True
        })