import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class AudioMetrics:
    """Audio quality metrics"""
    noise_level: float = 0.0
    signal_strength: float = 0.0
    echo_detected: bool = False
    voice_activity: bool = False
    quality_score: float = 0.0
    timestamp: float = 0.0

class AudioProcessor:
    """Handle audio processing, noise reduction, and quality enhancement"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_threshold = 0.1
        self.echo_threshold = 0.3
        self.voice_activity_threshold = 0.15
        
        # Simple noise profile (in production, this would be more sophisticated)
        self.noise_profile = None
        self.background_noise_level = 0.0
        
        # Audio quality history
        self.quality_history = []
        
    def process_audio_chunk(self, audio_data: np.ndarray) -> Tuple[np.ndarray, AudioMetrics]:
        """Process audio chunk for noise reduction and quality enhancement"""
        
        start_time = time.time()
        
        try:
            # Calculate audio metrics
            metrics = self._analyze_audio_quality(audio_data)
            
            # Apply noise reduction if needed
            processed_audio = audio_data.copy()
            
            if metrics.noise_level > self.noise_threshold:
                processed_audio = self._reduce_noise(processed_audio, metrics)
            
            # Apply echo cancellation if detected
            if metrics.echo_detected:
                processed_audio = self._cancel_echo(processed_audio)
            
            # Normalize audio levels
            processed_audio = self._normalize_audio(processed_audio)
            
            # Update quality history
            metrics.timestamp = time.time()
            self.quality_history.append(metrics)
            
            # Keep only recent history
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-100:]
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Audio processing latency: {processing_time:.2f}ms")
            
            return processed_audio, metrics
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return audio_data, AudioMetrics()
    
    def _analyze_audio_quality(self, audio_data: np.ndarray) -> AudioMetrics:
        """Analyze audio quality and detect issues"""
        
        metrics = AudioMetrics()
        
        # Calculate RMS (signal strength)
        rms = np.sqrt(np.mean(audio_data ** 2))
        metrics.signal_strength = float(rms)
        
        # Estimate noise level (simplified)
        # In production, this would use more sophisticated algorithms
        sorted_samples = np.sort(np.abs(audio_data))
        noise_floor = np.mean(sorted_samples[:len(sorted_samples)//4])  # Bottom 25%
        metrics.noise_level = float(noise_floor)
        
        # Detect voice activity
        metrics.voice_activity = rms > self.voice_activity_threshold
        
        # Simple echo detection using autocorrelation
        metrics.echo_detected = self._detect_echo(audio_data)
        
        # Calculate overall quality score (0-1)
        signal_to_noise = rms / (noise_floor + 1e-10)
        quality_factors = [
            min(signal_to_noise / 10, 1.0),  # SNR factor
            1.0 - min(noise_floor / 0.5, 1.0),  # Noise factor
            0.8 if not metrics.echo_detected else 0.4,  # Echo factor
        ]
        
        metrics.quality_score = np.mean(quality_factors)
        
        return metrics
    
    def _detect_echo(self, audio_data: np.ndarray) -> bool:
        """Simple echo detection using autocorrelation"""
        
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks that might indicate echo
            # Check for correlation peaks at typical echo delays (50-300ms)
            echo_start = int(0.05 * self.sample_rate)  # 50ms
            echo_end = int(0.3 * self.sample_rate)     # 300ms
            
            if echo_end < len(autocorr):
                echo_region = autocorr[echo_start:echo_end]
                max_echo_corr = np.max(echo_region) / autocorr[0] if autocorr[0] > 0 else 0
                
                return max_echo_corr > self.echo_threshold
            
            return False
            
        except Exception as e:
            logger.debug(f"Echo detection error: {e}")
            return False
    
    def _reduce_noise(self, audio_data: np.ndarray, metrics: AudioMetrics) -> np.ndarray:
        """Apply basic noise reduction"""
        
        try:
            # Simple spectral subtraction approach
            # In production, use more advanced algorithms like Wiener filtering
            
            # Update noise profile if this is low-activity audio
            if not metrics.voice_activity:
                self._update_noise_profile(audio_data)
            
            # Apply gentle high-pass filter to reduce low-frequency noise
            filtered_audio = self._high_pass_filter(audio_data, cutoff_freq=80)
            
            # Apply noise gate
            noise_gate_threshold = self.background_noise_level * 2
            filtered_audio = np.where(
                np.abs(filtered_audio) > noise_gate_threshold,
                filtered_audio,
                filtered_audio * 0.1  # Reduce but don't completely eliminate
            )
            
            return filtered_audio
            
        except Exception as e:
            logger.debug(f"Noise reduction error: {e}")
            return audio_data
    
    def _cancel_echo(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic echo cancellation"""
        
        try:
            # Simple approach: apply a comb filter
            # In production, use adaptive echo cancellation
            
            delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
            
            if len(audio_data) > delay_samples:
                # Create delayed version
                delayed = np.zeros_like(audio_data)
                delayed[delay_samples:] = audio_data[:-delay_samples]
                
                # Subtract delayed signal to reduce echo
                echo_cancelled = audio_data - 0.3 * delayed
                
                return echo_cancelled
            
            return audio_data
            
        except Exception as e:
            logger.debug(f"Echo cancellation error: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio levels"""
        
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > 1e-10:  # Avoid division by zero
                # Target RMS level
                target_rms = 0.2
                
                # Calculate gain
                gain = target_rms / rms
                
                # Limit gain to avoid excessive amplification
                gain = min(gain, 3.0)
                
                # Apply gain
                normalized_audio = audio_data * gain
                
                # Clip to prevent distortion
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                
                return normalized_audio
            
            return audio_data
            
        except Exception as e:
            logger.debug(f"Audio normalization error: {e}")
            return audio_data
    
    def _high_pass_filter(self, audio_data: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply simple high-pass filter"""
        
        try:
            # Simple first-order high-pass filter
            # In production, use proper filter design
            
            alpha = 1.0 / (1.0 + 2.0 * np.pi * cutoff_freq / self.sample_rate)
            
            filtered = np.zeros_like(audio_data)
            filtered[0] = audio_data[0]
            
            for i in range(1, len(audio_data)):
                filtered[i] = alpha * (filtered[i-1] + audio_data[i] - audio_data[i-1])
            
            return filtered
            
        except Exception as e:
            logger.debug(f"High-pass filter error: {e}")
            return audio_data
    
    def _update_noise_profile(self, audio_data: np.ndarray):
        """Update background noise profile"""
        
        try:
            current_noise_level = np.sqrt(np.mean(audio_data ** 2))
            
            if self.background_noise_level == 0:
                self.background_noise_level = current_noise_level
            else:
                # Exponential moving average
                alpha = 0.1
                self.background_noise_level = (
                    alpha * current_noise_level + 
                    (1 - alpha) * self.background_noise_level
                )
                
        except Exception as e:
            logger.debug(f"Noise profile update error: {e}")
    
    def get_audio_quality_summary(self) -> Dict[str, Any]:
        """Get summary of audio quality over recent history"""
        
        if not self.quality_history:
            return {"status": "no_data"}
        
        recent_metrics = self.quality_history[-20:]  # Last 20 measurements
        
        avg_quality = np.mean([m.quality_score for m in recent_metrics])
        avg_noise = np.mean([m.noise_level for m in recent_metrics])
        avg_signal = np.mean([m.signal_strength for m in recent_metrics])
        echo_rate = sum(1 for m in recent_metrics if m.echo_detected) / len(recent_metrics)
        voice_activity_rate = sum(1 for m in recent_metrics if m.voice_activity) / len(recent_metrics)
        
        status = "good" if avg_quality > 0.8 else "fair" if avg_quality > 0.5 else "poor"
        
        return {
            "average_quality_score": float(avg_quality),
            "average_noise_level": float(avg_noise),
            "average_signal_strength": float(avg_signal),
            "echo_detection_rate": float(echo_rate),
            "voice_activity_rate": float(voice_activity_rate),
            "background_noise_level": float(self.background_noise_level),
            "chunk_count": len(self.quality_history),
            "quality_trend": self._calculate_quality_trend(),
            "status": status
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate if audio quality is improving or degrading"""
        
        if len(self.quality_history) < 10:
            return "insufficient_data"
        
        recent_half = self.quality_history[-5:]
        earlier_half = self.quality_history[-10:-5]
        
        recent_avg = np.mean([m.quality_score for m in recent_half])
        earlier_avg = np.mean([m.quality_score for m in earlier_half])
        
        if recent_avg > earlier_avg + 0.05:
            return "improving"
        elif recent_avg < earlier_avg - 0.05:
            return "degrading"
        else:
            return "stable"
    
    def suggest_audio_improvements(self) -> List[str]:
        """Suggest improvements based on audio analysis"""
        
        suggestions = []
        
        if not self.quality_history:
            return ["No audio data available for analysis"]
        
        recent_metrics = self.quality_history[-10:]
        avg_quality = np.mean([m.quality_score for m in recent_metrics])
        avg_noise = np.mean([m.noise_level for m in recent_metrics])
        avg_signal = np.mean([m.signal_strength for m in recent_metrics])
        echo_rate = sum(1 for m in recent_metrics if m.echo_detected) / len(recent_metrics)
        
        if avg_quality < 0.6:
            suggestions.append("Overall audio quality is low - consider improving microphone or environment")
        
        if avg_noise > 0.2:
            suggestions.append("High background noise detected - use noise cancellation or quieter environment")
        
        if echo_rate > 0.3:
            suggestions.append("Echo detected frequently - check speaker volume and room acoustics")
        
        if self.background_noise_level > 0.15:
            suggestions.append("Consistent background noise - consider using a headset")
        
        if avg_signal < 0.1:
            suggestions.append("Low volume detected - increase microphone volume")
        
        if not suggestions:
            suggestions.append("Audio quality is good")
        
        return suggestions