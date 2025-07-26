"""
Audio utility functions for device detection, format conversion, and processing.
"""

import sounddevice as sd
import numpy as np
import wave
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger


logger = get_logger("audio.utils")


def list_audio_devices() -> List[Dict[str, Any]]:
    """
    List all available audio devices.
    
    Returns:
        List of audio device dictionaries
    """
    try:
        devices = sd.query_devices()
        device_list = []
        
        for i, device in enumerate(devices):
            device_info = {
                'index': i,
                'name': device.get('name', 'Unknown'),
                'max_inputs': device.get('max_inputs', 0),
                'max_outputs': device.get('max_outputs', 0),
                'default_samplerate': device.get('default_samplerate', 44100),
                'is_blackhole': 'blackhole' in str(device).lower()
            }
            device_list.append(device_info)
        
        return device_list
        
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return []


def find_blackhole_device() -> Optional[int]:
    """
    Find BlackHole device index.
    
    Returns:
        Device index if found, None otherwise
    """
    try:
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            device_str = str(device).lower()
            if 'blackhole' in device_str:
                logger.info(f"Found BlackHole device at index {i}: {device}")
                return i
        
        logger.warning("BlackHole device not found")
        return None
        
    except Exception as e:
        logger.error(f"Error finding BlackHole device: {e}")
        return None


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get audio file information.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio file information
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Get basic file info
        file_size = os.path.getsize(file_path)
        
        # Get audio info using librosa
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'file_path': file_path,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'sample_rate': sr,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'channels': y.shape[1] if len(y.shape) > 1 else 1,
            'samples': len(y)
        }
        
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return {}


def convert_audio_format(input_path: str, output_path: str, 
                        target_sr: int = 16000, target_channels: int = 1) -> bool:
    """
    Convert audio file format and properties.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_sr: Target sample rate
        target_channels: Target number of channels
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info(f"Converting audio: {input_path} -> {output_path}")
        
        # Load audio with librosa
        y, sr = librosa.load(input_path, sr=target_sr)
        
        # Convert to mono if needed
        if target_channels == 1 and len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(target_channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(target_sr)
            wav_file.writeframes((y * 32767).astype(np.int16).tobytes())
        
        logger.info(f"Audio conversion completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return False


def normalize_audio(audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio_data: Input audio data
        target_db: Target dB level
        
    Returns:
        Normalized audio data
    """
    try:
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            # Convert target dB to linear scale
            target_rms = 10**(target_db / 20)
            
            # Calculate normalization factor
            normalization_factor = target_rms / rms
            
            # Apply normalization
            normalized_audio = audio_data * normalization_factor
            
            # Clip to prevent overflow
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
            return normalized_audio
        else:
            return audio_data
            
    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        return audio_data


def detect_silence(audio_data: np.ndarray, threshold: float = 0.01, 
                   min_silence_duration: float = 0.5, sample_rate: int = 16000) -> List[Tuple[float, float]]:
    """
    Detect silence periods in audio.
    
    Args:
        audio_data: Input audio data
        threshold: Silence threshold
        min_silence_duration: Minimum silence duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        List of (start_time, end_time) tuples for silence periods
    """
    try:
        # Calculate audio energy
        energy = np.abs(audio_data)
        
        # Find silence periods
        silence_mask = energy < threshold
        
        # Find silence segments
        silence_segments = []
        start_idx = None
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and start_idx is None:
                start_idx = i
            elif not is_silent and start_idx is not None:
                end_idx = i
                duration = (end_idx - start_idx) / sample_rate
                
                if duration >= min_silence_duration:
                    start_time = start_idx / sample_rate
                    end_time = end_idx / sample_rate
                    silence_segments.append((start_time, end_time))
                
                start_idx = None
        
        # Handle case where audio ends with silence
        if start_idx is not None:
            duration = (len(silence_mask) - start_idx) / sample_rate
            if duration >= min_silence_duration:
                start_time = start_idx / sample_rate
                end_time = len(silence_mask) / sample_rate
                silence_segments.append((start_time, end_time))
        
        return silence_segments
        
    except Exception as e:
        logger.error(f"Silence detection failed: {e}")
        return []


def split_audio_by_silence(audio_data: np.ndarray, sample_rate: int = 16000,
                          silence_threshold: float = 0.01, min_segment_duration: float = 1.0) -> List[np.ndarray]:
    """
    Split audio into segments based on silence detection.
    
    Args:
        audio_data: Input audio data
        sample_rate: Audio sample rate
        silence_threshold: Silence threshold
        min_segment_duration: Minimum segment duration in seconds
        
    Returns:
        List of audio segments
    """
    try:
        silence_periods = detect_silence(audio_data, silence_threshold, 0.5, sample_rate)
        
        segments = []
        last_end = 0
        
        for start_time, end_time in silence_periods:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Check if segment is long enough
            segment_duration = (start_sample - last_end) / sample_rate
            if segment_duration >= min_segment_duration:
                segment = audio_data[last_end:start_sample]
                segments.append(segment)
            
            last_end = end_sample
        
        # Add final segment
        if last_end < len(audio_data):
            final_segment = audio_data[last_end:]
            segment_duration = len(final_segment) / sample_rate
            if segment_duration >= min_segment_duration:
                segments.append(final_segment)
        
        return segments
        
    except Exception as e:
        logger.error(f"Audio splitting failed: {e}")
        return [audio_data]


def validate_audio_file(file_path: str) -> bool:
    """
    Validate audio file format and properties.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error("Audio file is empty")
            return False
        
        # Try to load with librosa
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) == 0:
            logger.error("Audio file has no samples")
            return False
        
        # Check sample rate
        if sr < 8000 or sr > 48000:
            logger.warning(f"Unusual sample rate: {sr}")
        
        logger.info(f"Audio file validation passed: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Audio file validation failed: {e}")
        return False
