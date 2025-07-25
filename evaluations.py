import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """Types of evaluation metrics"""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    NATURALNESS = "naturalness"
    CONVERSATION_FLOW = "conversation_flow"
    FUNCTION_ACCURACY = "function_accuracy"
    AUDIO_QUALITY = "audio_quality"

@dataclass
class EvaluationResult:
    """Individual evaluation result"""
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: float
    test_case_id: str
    notes: str = ""

@dataclass
class ConversationTestCase:
    """Test case for conversation evaluation"""
    test_id: str
    scenario: str
    expected_patient_name: str
    expected_procedure: str
    expected_outcome: str
    user_inputs: List[str]
    expected_flow_states: List[str]
    max_latency_ms: float = 800.0

class HealthcareEvaluator:
    """Comprehensive evaluation framework for healthcare AI agent"""
    
    def __init__(self, target_latency_ms: float = 800.0):
        self.target_latency_ms = target_latency_ms
        self.evaluation_results: List[EvaluationResult] = []
        self.test_cases: List[ConversationTestCase] = []
        
        # Load default test cases
        self._load_default_test_cases()
    
    def _load_default_test_cases(self):
        """Load standard test cases for healthcare prior authorization"""
        
        self.test_cases = [
            ConversationTestCase(
                test_id="basic_auth_001",
                scenario="Standard prior authorization request",
                expected_patient_name="John Doe",
                expected_procedure="MRI scan",
                expected_outcome="Approved",
                user_inputs=[
                    "Hi, I'm calling about John Doe",
                    "Yes, John Doe, D-O-E",
                    "He needs an MRI scan for lower back pain",
                    "The pain has been persistent for 3 months",
                    "Yes, please approve it"
                ],
                expected_flow_states=["greeting", "patient_verification", "procedure_collection", "authorization_decision", "completion"]
            ),
            ConversationTestCase(
                test_id="interruption_001",
                scenario="User interrupts AI response",
                expected_patient_name="Jane Smith",
                expected_procedure="CT scan",
                expected_outcome="Pending",
                user_inputs=[
                    "Jane Smith",
                    "Sorry, what did you say?",  # Interruption
                    "Jane Smith, S-M-I-T-H",
                    "CT scan for chest pain",
                    "Let me check with the doctor first"
                ],
                expected_flow_states=["greeting", "patient_verification", "patient_verification", "procedure_collection", "completion"]
            ),
            ConversationTestCase(
                test_id="clarification_001", 
                scenario="User requests clarification",
                expected_patient_name="Bob Johnson",
                expected_procedure="Physical therapy",
                expected_outcome="Approved",
                user_inputs=[
                    "Bob Johnson",
                    "What information do you need exactly?",  # Clarification request
                    "Physical therapy for shoulder injury",
                    "6 weeks of treatment needed",
                    "Yes, approve it please"
                ],
                expected_flow_states=["greeting", "patient_verification", "procedure_collection", "procedure_collection", "authorization_decision", "completion"]
            )
        ]
    
    async def evaluate_latency(self, pipeline_function: Callable, test_input: str) -> EvaluationResult:
        """Evaluate response latency"""
        
        latencies = []
        
        # Run multiple iterations for accurate measurement
        for i in range(3):
            start_time = time.time()
            
            try:
                await pipeline_function(test_input)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.error(f"Latency test error: {e}")
                latencies.append(float('inf'))
        
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Score: 1.0 if under target, decreasing linearly
        score = max(0.0, min(1.0, (self.target_latency_ms - avg_latency) / self.target_latency_ms + 0.5))
        
        return EvaluationResult(
            metric=EvaluationMetric.LATENCY,
            score=score,
            details={
                "average_latency_ms": avg_latency,
                "min_latency_ms": min_latency,
                "max_latency_ms": max_latency,
                "target_latency_ms": self.target_latency_ms,
                "measurements": latencies
            },
            timestamp=time.time(),
            test_case_id="latency_test",
            notes=f"Target: {self.target_latency_ms}ms, Actual: {avg_latency:.2f}ms"
        )
    
    def evaluate_function_accuracy(self, expected_functions: List[str], actual_functions: List[str], function_results: Dict[str, Any]) -> EvaluationResult:
        """Evaluate accuracy of function calls"""
        
        # Check if correct functions were called
        function_precision = len(set(expected_functions) & set(actual_functions)) / max(len(actual_functions), 1)
        function_recall = len(set(expected_functions) & set(actual_functions)) / max(len(expected_functions), 1)
        
        # Check function results accuracy
        result_accuracy = 0.0
        if function_results:
            # Simplified accuracy check - in production, use more sophisticated validation
            successful_calls = sum(1 for result in function_results.values() if result and result != "error")
            result_accuracy = successful_calls / max(len(function_results), 1)
        
        # Combined score
        score = (function_precision + function_recall + result_accuracy) / 3
        
        return EvaluationResult(
            metric=EvaluationMetric.FUNCTION_ACCURACY,
            score=score,
            details={
                "expected_functions": expected_functions,
                "actual_functions": actual_functions,
                "function_precision": function_precision,
                "function_recall": function_recall,
                "result_accuracy": result_accuracy,
                "function_results": function_results
            },
            timestamp=time.time(),
            test_case_id="function_accuracy_test"
        )
    
    def evaluate_conversation_flow(self, expected_states: List[str], actual_states: List[str]) -> EvaluationResult:
        """Evaluate conversation flow adherence"""
        
        # Calculate state sequence similarity
        min_length = min(len(expected_states), len(actual_states))
        correct_transitions = 0
        
        for i in range(min_length):
            if expected_states[i] == actual_states[i]:
                correct_transitions += 1
        
        # Penalize length differences
        length_penalty = abs(len(expected_states) - len(actual_states)) * 0.1
        
        # Score based on correct transitions
        base_score = correct_transitions / max(len(expected_states), 1)
        score = max(0.0, base_score - length_penalty)
        
        return EvaluationResult(
            metric=EvaluationMetric.CONVERSATION_FLOW,
            score=score,
            details={
                "expected_states": expected_states,
                "actual_states": actual_states,
                "correct_transitions": correct_transitions,
                "total_expected": len(expected_states),
                "length_penalty": length_penalty
            },
            timestamp=time.time(),
            test_case_id="conversation_flow_test"
        )
    
    def evaluate_naturalness(self, ai_responses: List[str], context: Dict[str, Any]) -> EvaluationResult:
        """Evaluate naturalness of AI responses"""
        
        # Simple heuristic-based naturalness evaluation
        # In production, use more sophisticated NLP models
        
        naturalness_scores = []
        
        for response in ai_responses:
            response_score = self._score_response_naturalness(response, context)
            naturalness_scores.append(response_score)
        
        avg_naturalness = statistics.mean(naturalness_scores) if naturalness_scores else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.NATURALNESS,
            score=avg_naturalness,
            details={
                "individual_scores": naturalness_scores,
                "average_score": avg_naturalness,
                "response_count": len(ai_responses),
                "context": context
            },
            timestamp=time.time(),
            test_case_id="naturalness_test"
        )
    
    def _score_response_naturalness(self, response: str, context: Dict[str, Any]) -> float:
        """Score individual response naturalness (simplified)"""
        
        score = 1.0
        
        # Check response length (not too short or long)
        word_count = len(response.split())
        if word_count < 3:
            score -= 0.3  # Too short
        elif word_count > 50:
            score -= 0.2  # Too long for voice
        
        # Check for healthcare appropriateness
        healthcare_terms = ["patient", "authorization", "procedure", "medical", "insurance"]
        if not any(term in response.lower() for term in healthcare_terms):
            score -= 0.2  # Not healthcare-focused
        
        # Check for politeness
        polite_terms = ["please", "thank you", "may i", "could you", "i understand"]
        if any(term in response.lower() for term in polite_terms):
            score += 0.1  # Bonus for politeness
        
        # Check for questions (interactive)
        if "?" in response:
            score += 0.1  # Bonus for engagement
        
        # Penalize robotic phrases
        robotic_phrases = ["i am an ai", "my programming", "i cannot", "error occurred"]
        if any(phrase in response.lower() for phrase in robotic_phrases):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def run_comprehensive_evaluation(self, pipeline_instance, test_case: ConversationTestCase) -> Dict[str, EvaluationResult]:
        """Run comprehensive evaluation on a test case"""
        
        results = {}
        
        try:
            # Simulate conversation
            conversation_log = await self._simulate_conversation(pipeline_instance, test_case)
            
            # Evaluate different metrics
            results["latency"] = await self.evaluate_latency(
                pipeline_instance.process_message, 
                test_case.user_inputs[0]
            )
            
            results["conversation_flow"] = self.evaluate_conversation_flow(
                test_case.expected_flow_states,
                conversation_log.get("actual_states", [])
            )
            
            results["function_accuracy"] = self.evaluate_function_accuracy(
                ["search_patient_by_name", "update_prior_auth_status"],
                conversation_log.get("functions_called", []),
                conversation_log.get("function_results", {})
            )
            
            results["naturalness"] = self.evaluate_naturalness(
                conversation_log.get("ai_responses", []),
                {"test_case": test_case.scenario}
            )
            
            # Store results
            for result in results.values():
                result.test_case_id = test_case.test_id
                self.evaluation_results.append(result)
            
        except Exception as e:
            logger.error(f"Evaluation error for test case {test_case.test_id}: {e}")
        
        return results
    
    async def _simulate_conversation(self, pipeline_instance, test_case: ConversationTestCase) -> Dict[str, Any]:
        """Simulate a conversation for testing"""
        
        # This is a simplified simulation
        # In production, integrate with actual pipeline
        
        conversation_log = {
            "actual_states": [],
            "ai_responses": [],
            "functions_called": [],
            "function_results": {}
        }
        
        # Simulate conversation flow
        for user_input in test_case.user_inputs:
            # Simulate pipeline processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Track simulated state transitions
            if "name" in user_input.lower():
                conversation_log["actual_states"].append("patient_verification")
                conversation_log["functions_called"].append("search_patient_by_name")
                conversation_log["function_results"]["search_patient_by_name"] = {"found": True}
            elif any(word in user_input.lower() for word in ["procedure", "scan", "therapy"]):
                conversation_log["actual_states"].append("procedure_collection")
            elif any(word in user_input.lower() for word in ["approve", "yes", "confirm"]):
                conversation_log["actual_states"].append("authorization_decision")
                conversation_log["functions_called"].append("update_prior_auth_status")
                conversation_log["function_results"]["update_prior_auth_status"] = True
            
            # Simulate AI response
            ai_response = f"Thank you for that information. {user_input}"
            conversation_log["ai_responses"].append(ai_response)
        
        return conversation_log
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not self.evaluation_results:
            return {"status": "no_evaluations"}
        
        # Group results by metric
        metric_scores = {}
        for result in self.evaluation_results:
            metric = result.metric.value
            if metric not in metric_scores:
                metric_scores[metric] = []
            metric_scores[metric].append(result.score)
        
        # Calculate averages
        metric_averages = {
            metric: statistics.mean(scores) 
            for metric, scores in metric_scores.items()
        }
        
        # Overall score
        overall_score = statistics.mean(metric_averages.values()) if metric_averages else 0.0
        
        # Performance summary
        performance_summary = {
            "overall_score": overall_score,
            "metric_scores": metric_averages,
            "total_evaluations": len(self.evaluation_results),
            "evaluation_date": time.time(),
            "target_latency_ms": self.target_latency_ms
        }
        
        # Recommendations
        recommendations = self._generate_recommendations(metric_averages)
        
        return {
            "performance_summary": performance_summary,
            "detailed_results": [asdict(result) for result in self.evaluation_results],
            "recommendations": recommendations,
            "test_cases_run": len(self.test_cases)
        }
    
    def _generate_recommendations(self, metric_averages: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        
        recommendations = []
        
        if metric_averages.get("latency", 1.0) < 0.7:
            recommendations.append("Latency exceeds target - consider optimizing STT/LLM parallel processing")
        
        if metric_averages.get("function_accuracy", 1.0) < 0.8:
            recommendations.append("Function calling accuracy is low - review function definitions and error handling")
        
        if metric_averages.get("conversation_flow", 1.0) < 0.8:
            recommendations.append("Conversation flow needs improvement - review workflow state transitions")
        
        if metric_averages.get("naturalness", 1.0) < 0.7:
            recommendations.append("Response naturalness is low - improve system prompts and response templates")
        
        if not recommendations:
            recommendations.append("Performance meets standards - consider advanced optimizations")
        
        return recommendations