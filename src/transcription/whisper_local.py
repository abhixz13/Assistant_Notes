"""
Real-time transcription using OpenAI Whisper.
"""

import whisper
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import queue
import sounddevice as sd
import soundfile as sf

from utils.config import config
from utils.logger import get_logger


logger = get_logger("transcription.whisper")


class WhisperTranscriber:
    """
    Real-time transcription using OpenAI Whisper model.
    """
    
    def __init__(self, model_size: str = "medium"): 
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.transcript_segments = []
        self.current_segment = ""
        self.start_time = None
        self.audio_config = config.get_audio_config()
        
        # Transcription settings
        self.chunk_duration = 5  # seconds per chunk
        self.sample_rate = self.audio_config.get('sample_rate', 16000)
        self.channels = self.audio_config.get('channels', 1)
        
        logger.info(f"Initializing WhisperTranscriber with model size: {model_size}")
    
    def load_model(self) -> bool:
        """
        Load the Whisper model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """
        Start real-time transcription streaming.
        
        Returns:
            bool: True if streaming started successfully
        """
        if not self.model:
            if not self.load_model():
                return False
        
        try:
            self.is_streaming = True
            self.start_time = time.time()
            self.transcript_segments = []
            self.current_segment = ""
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start transcription thread
            self.transcription_thread = threading.Thread(target=self._transcription_loop)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            logger.info("Real-time transcription streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self) -> Dict:
        """
        Stop real-time transcription and return final transcript.
        
        Returns:
            Dict: Complete transcript with metadata
        """
        self.is_streaming = False
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=5)
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=5)
        
        # Process any remaining audio
        self._process_remaining_audio()
        
        # Generate final transcript
        final_transcript = self._format_final_transcript()
        
        logger.info("Real-time transcription stopped")
        return final_transcript
    
    def _audio_capture_loop(self):
        """Audio capture loop for real-time streaming."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=int(self.sample_rate * self.chunk_duration)
            ) as stream:
                logger.info("Audio capture started")
                
                while self.is_streaming:
                    audio_chunk, _ = stream.read(int(self.sample_rate * self.chunk_duration))
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        self.audio_queue.put(audio_chunk)
                
                logger.info("Audio capture stopped")
                
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _transcription_loop(self):
        """Transcription processing loop."""
        try:
            logger.info("Transcription processing started")
            
            while self.is_streaming or not self.audio_queue.empty():
                try:
                    # Get audio chunk with timeout
                    audio_chunk = self.audio_queue.get(timeout=1)
                    
                    # Transcribe chunk
                    transcript = self._transcribe_chunk(audio_chunk)
                    
                    if transcript and transcript.strip():
                        self._add_transcript_segment(transcript)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Transcription processing error: {e}")
            
            logger.info("Transcription processing stopped")
            
        except Exception as e:
            logger.error(f"Transcription loop error: {e}")
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            str: Transcribed text
        """
        try:
            # Ensure audio is in the right format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_chunk,
                language="en",
                task="transcribe",
                fp16=False
            )
            
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Chunk transcription error: {e}")
            return ""
    
    def _add_transcript_segment(self, text: str):
        """
        Add a new transcript segment.
        
        Args:
            text: Transcribed text
        """
        if not text:
            return
        
        current_time = time.time() - self.start_time
        timestamp = self._format_timestamp(current_time)
        
        segment = {
            "timestamp": timestamp,
            "time_seconds": current_time,
            "text": text
        }
        
        self.transcript_segments.append(segment)
        self.current_segment = text
        
        logger.debug(f"Added transcript segment: {timestamp} - {text[:50]}...")
    
    def _process_remaining_audio(self):
        """Process any remaining audio in the queue."""
        try:
            while not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get_nowait()
                transcript = self._transcribe_chunk(audio_chunk)
                if transcript:
                    self._add_transcript_segment(transcript)
        except Exception as e:
            logger.error(f"Error processing remaining audio: {e}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into MM:SS timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _format_final_transcript(self) -> Dict:
        """
        Format the complete transcript.
        
        Returns:
            Dict: Structured transcript with metadata
        """
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        # Combine all segments
        full_text = " ".join([seg["text"] for seg in self.transcript_segments])
        
        # Extract key points and technical terms
        key_points = self._extract_key_points(full_text)
        technical_terms = self._extract_technical_terms(full_text)
        
        transcript = {
            "metadata": {
                "model_size": self.model_size,
                "duration_seconds": total_duration,
                "duration_formatted": self._format_timestamp(total_duration),
                "segments_count": len(self.transcript_segments),
                "created_at": datetime.now().isoformat()
            },
            "segments": self.transcript_segments,
            "full_text": full_text,
            "key_points": key_points,
            "technical_terms": technical_terms,
            "formatted_notes": self._create_formatted_notes()
        }
        
        return transcript
    
    def _extract_key_points(self, text: str) -> List[str]:
        """
        Extract key points from transcript text.
        
        Args:
            text: Full transcript text
            
        Returns:
            List[str]: Key points
        """
        # Simple key point extraction (can be enhanced later)
        sentences = text.split('.')
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                # Look for sentences with important keywords
                important_keywords = ['important', 'key', 'main', 'primary', 'essential', 
                                   'crucial', 'critical', 'significant', 'fundamental']
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    key_points.append(sentence)
        
        return key_points[:10]  # Limit to 10 key points
    
    def _extract_technical_terms(self, text: str) -> Dict[str, str]:
        """
        Extract technical terms from transcript text.
        
        Args:
            text: Full transcript text
            
        Returns:
            Dict[str, str]: Technical terms and their context
        """
        # Simple technical term extraction (can be enhanced later)
        technical_terms = {}
        
        # Common AI/ML terms
        ai_terms = ['neural network', 'machine learning', 'deep learning', 'algorithm',
                   'model', 'training', 'inference', 'backpropagation', 'gradient descent',
                   'overfitting', 'underfitting', 'validation', 'test set', 'feature']
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            for term in ai_terms:
                if term in sentence.lower():
                    # Extract the sentence containing the term
                    technical_terms[term] = sentence
        
        return technical_terms
    
    def _create_formatted_notes(self) -> str:
        """
        Create formatted, readable notes.
        
        Returns:
            str: Formatted notes
        """
        if not self.transcript_segments:
            return "No transcript available."
        
        notes = []
        notes.append("# AI Notes Assistant - Live Transcript\n")
        notes.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        notes.append(f"Duration: {self._format_timestamp(time.time() - self.start_time)}\n")
        notes.append(f"Model: Whisper {self.model_size}\n")
        notes.append("---\n\n")
        
        # Add transcript segments
        notes.append("## Transcript\n")
        for segment in self.transcript_segments:
            notes.append(f"[{segment['timestamp']}] {segment['text']}\n")
        
        # Add key points
        if self.transcript_segments:
            full_text = " ".join([seg["text"] for seg in self.transcript_segments])
            key_points = self._extract_key_points(full_text)
            if key_points:
                notes.append("\n## Key Points\n")
                for i, point in enumerate(key_points, 1):
                    notes.append(f"{i}. {point}\n")
        
        # Add technical terms
        if self.transcript_segments:
            full_text = " ".join([seg["text"] for seg in self.transcript_segments])
            technical_terms = self._extract_technical_terms(full_text)
            if technical_terms:
                notes.append("\n## Technical Terms\n")
                for term, context in technical_terms.items():
                    notes.append(f"- **{term.title()}**: {context}\n")
        
        return "".join(notes)
    
    def get_live_status(self) -> Dict:
        """
        Get current live transcription status.
        
        Returns:
            Dict: Current status information
        """
        current_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "is_streaming": self.is_streaming,
            "duration": self._format_timestamp(current_time),
            "segments_count": len(self.transcript_segments),
            "current_segment": self.current_segment,
            "queue_size": self.audio_queue.qsize()
        }

