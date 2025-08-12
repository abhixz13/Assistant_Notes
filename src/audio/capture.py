"""
Audio capture module using BlackHole for system audio recording.
"""

import sounddevice as sd
import numpy as np
import wave
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import threading
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import config
from utils.logger import LoggerMixin, get_logger


class AudioCapture:
    """Audio capture class using BlackHole for system audio recording."""
    
    def __init__(self, audio_config: Dict[str, Any]):
        """
        Initialize audio capture with configuration.
        
        Args:
            audio_config: Audio configuration dictionary
        """
        self.config = audio_config
        self.logger = get_logger("audio.capture")
        
        # Audio settings
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.chunk_size = audio_config.get('chunk_size', 1024)
        self.format = audio_config.get('format', 'wav')
        self.blackhole_device = audio_config.get('blackhole_device', 'BlackHole 2ch')
        
        # Find BlackHole device
        self.device_index = self._find_blackhole_device()
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.audio_buffer = []
        
        self.logger.info(f"AudioCapture initialized with device index: {self.device_index}")
    
    def _find_blackhole_device(self) -> int:
        """
        Find audio device index by name or default to BlackHole.
        
        Returns:
            Device index for the specified device
        """
        try:
            devices = sd.query_devices()
            
            # First, try to find the specified device name
            target_device = self.blackhole_device.lower()
            for i, device in enumerate(devices):
                device_str = str(device).lower()
                if target_device in device_str:
                    self.logger.info(f"Found device '{self.blackhole_device}' at index {i}: {device}")
                    return i
            
            # If not found, try to find by name attribute
            for i, device in enumerate(devices):
                if hasattr(device, 'name') and target_device in device.name.lower():
                    self.logger.info(f"Found device '{self.blackhole_device}' by name at index {i}: {device}")
                    return i
            
            # If still not found, try to find BlackHole as fallback
            for i, device in enumerate(devices):
                device_str = str(device).lower()
                if 'blackhole' in device_str:
                    self.logger.info(f"BlackHole device not found, using BlackHole at index {i}: {device}")
                    return i
            
            # Default to device 0 if nothing found
            self.logger.warning(f"Device '{self.blackhole_device}' not found, using default device 0")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error finding audio device: {e}")
            return 0
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict[str, Any], status: sd.CallbackFlags) -> None:
        """
        Callback function for audio recording.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Store audio data in buffer
        self.audio_buffer.append(indata.copy())
    
    def record_to_file(self, duration: Optional[int] = None, output_path: Optional[str] = None) -> str:
        """
        Record audio to file.
        
        Args:
            duration: Recording duration in seconds (None for continuous)
            output_path: Output file path (None for auto-generated)
            
        Returns:
            Path to the recorded audio file
        """
        if duration is None:
            duration = self.config.get('recording_duration', 300)
        
        if output_path is None:
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/transcripts/audio_{timestamp}.wav"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting audio recording for {duration} seconds to {output_path}")
        
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_index
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Save to WAV file
            self._save_wav_file(recording, output_path)
            
            self.logger.info(f"Audio recording completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            raise
    
    def record_continuous(self, output_path: str, max_duration: Optional[int] = None) -> None:
        """
        Record audio continuously until stopped.
        
        Args:
            output_path: Output file path
            max_duration: Maximum recording duration in seconds (None for unlimited)
        """
        self.logger.info(f"Starting continuous recording to {output_path}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = True
        self.audio_buffer = []
        self.output_path = output_path
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self._record_continuous_thread, args=(max_duration,))
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info("Continuous recording started in background thread")
    
    def _record_continuous_thread(self, max_duration: Optional[int] = None) -> None:
        """Background thread for continuous recording."""
        try:
            # Start recording stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_index,
                callback=self._audio_callback,
                blocksize=self.chunk_size
            ):
                start_time = time.time()
                
                while self.is_recording:
                    if max_duration and (time.time() - start_time) > max_duration:
                        self.logger.info("Maximum recording duration reached")
                        break
                    
                    time.sleep(0.1)  # Small delay to prevent CPU overuse
                
                # Save recorded audio
                if self.audio_buffer:
                    audio_data = np.concatenate(self.audio_buffer, axis=0)
                    self._save_wav_file(audio_data, self.output_path)
                    self.logger.info(f"Continuous recording saved: {self.output_path}")
                
        except Exception as e:
            self.logger.error(f"Continuous recording failed: {e}")
        finally:
            self.is_recording = False
    
    def stop_recording(self) -> None:
        """Stop continuous recording."""
        self.logger.info("Stopping continuous recording")
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread') and self.recording_thread and self.recording_thread.is_alive():
            self.logger.info("Waiting for recording thread to finish...")
            self.recording_thread.join(timeout=5)  # Wait up to 5 seconds
            if self.recording_thread.is_alive():
                self.logger.warning("Recording thread did not finish in time")
    
    def _save_wav_file(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Output file path
        """
        try:
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            self.logger.info(f"Audio saved to WAV file: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving WAV file: {e}")
            raise
    
    def get_audio_info(self) -> Dict[str, Any]:
        """
        Get audio device information.
        
        Returns:
            Dictionary with audio device information
        """
        try:
            devices = sd.query_devices()
            current_device = devices[self.device_index]
            
            return {
                'device_index': self.device_index,
                'device_name': current_device.get('name', 'Unknown'),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'max_inputs': current_device.get('max_inputs', 0),
                'max_outputs': current_device.get('max_outputs', 0),
                'is_blackhole': 'blackhole' in str(current_device).lower()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting audio info: {e}")
            return {}
    
    def test_recording(self, duration: int = 5) -> bool:
        """
        Test audio recording functionality.
        
        Args:
            duration: Test recording duration in seconds
            
        Returns:
            True if test successful, False otherwise
        """
        try:
            self.logger.info(f"Testing audio recording for {duration} seconds")
            
            # Record test audio
            test_file = self.record_to_file(duration=duration)
            
            # Check if file exists and has content
            if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                self.logger.info("Audio recording test successful")
                return True
            else:
                self.logger.error("Audio recording test failed - no file created")
                return False
                
        except Exception as e:
            self.logger.error(f"Audio recording test failed: {e}")
            return False


def create_audio_capture() -> AudioCapture:
    """
    Create audio capture instance with default configuration.
    
    Returns:
        AudioCapture instance
    """
    audio_config = config.get_audio_config()
    return AudioCapture(audio_config)
