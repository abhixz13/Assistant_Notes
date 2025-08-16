"""
Intelligent Chunked Audio Processor for Long Videos

Implements hybrid approach: 60-minute chunks with Whisper internal optimizations
and seamless transcript compilation.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import soundfile as sf
from datetime import datetime, timedelta

from utils.logger import get_logger
from .whisper_local import WhisperTranscriber

logger = get_logger("transcription.chunked")


class ChunkedAudioProcessor:
    """
    Processes long audio files using intelligent 60-minute chunks
    with overlap handling and seamless transcript compilation.
    """
    
    def __init__(self, 
                 chunk_duration_minutes: int = 60,
                 overlap_seconds: int = 30,
                 model_size: str = "small",
                 silence_threshold: float = 0.01,
                 min_silence_duration: float = 2.0):
        """
        Initialize the chunked processor.
        
        Args:
            chunk_duration_minutes: Target chunk size in minutes
            overlap_seconds: Overlap between chunks to prevent word splitting
            model_size: Whisper model size to use
            silence_threshold: Audio level below which is considered silence
            min_silence_duration: Minimum silence duration for smart chunking
        """
        self.chunk_duration_minutes = chunk_duration_minutes
        self.overlap_seconds = overlap_seconds
        self.model_size = model_size
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        
        # Calculate chunk size in samples (will be set when we load audio)
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.sample_rate = None
        self.transcriber = None
        
        logger.info(f"ChunkedAudioProcessor initialized: {chunk_duration_minutes}min chunks, {overlap_seconds}s overlap")
    
    def process_long_audio(self, audio_file_path: str, output_dir: str = None) -> Dict:
        """
        Process a long audio file using intelligent chunking.
        
        Args:
            audio_file_path: Path to the audio file
            output_dir: Directory to save chunk files (optional)
            
        Returns:
            Dict: Complete compiled transcript with metadata
        """
        try:
            logger.info(f"Starting chunked processing of: {audio_file_path}")
            
            # Load audio file
            audio_data, self.sample_rate = sf.read(audio_file_path)
            audio_data = audio_data.copy()  # Avoid stride issues
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            logger.info(f"Audio loaded: {len(audio_data)/self.sample_rate:.1f} seconds, {self.sample_rate}Hz")
            
            # Initialize transcriber
            self.transcriber = WhisperTranscriber(model_size=self.model_size)
            if not self.transcriber.load_model():
                raise Exception("Failed to load Whisper model")
            
            # Create intelligent chunks
            chunk_boundaries = self._create_intelligent_chunks(audio_data)
            logger.info(f"Created {len(chunk_boundaries)} intelligent chunks")
            
            # Process each chunk
            chunk_transcripts = []
            for i, (start_time, end_time) in enumerate(chunk_boundaries):
                logger.info(f"Processing chunk {i+1}/{len(chunk_boundaries)}: {start_time:.1f}s - {end_time:.1f}s")
                
                chunk_transcript = self._process_chunk(
                    audio_data, start_time, end_time, i+1, audio_file_path, output_dir
                )
                
                if chunk_transcript:
                    chunk_transcripts.append(chunk_transcript)
            
            # Compile final transcript
            final_transcript = self._compile_transcript(
                chunk_transcripts, audio_file_path, len(audio_data)/self.sample_rate
            )
            
            logger.info("Chunked processing completed successfully")
            return final_transcript
            
        except Exception as e:
            logger.error(f"Error in chunked processing: {e}")
            raise
    
    def _create_intelligent_chunks(self, audio_data: np.ndarray) -> List[Tuple[float, float]]:
        """
        Create intelligent chunk boundaries using silence detection.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        total_duration = len(audio_data) / self.sample_rate
        chunk_boundaries = []
        
        if total_duration <= self.chunk_duration_seconds:
            # Short audio, process as single chunk
            return [(0.0, total_duration)]
        
        current_start = 0.0
        
        while current_start < total_duration:
            # Calculate target end time
            target_end = min(current_start + self.chunk_duration_seconds, total_duration)
            
            if target_end >= total_duration:
                # Last chunk
                chunk_boundaries.append((current_start, total_duration))
                break
            
            # Look for silence near the target end time
            optimal_end = self._find_optimal_split_point(
                audio_data, current_start, target_end
            )
            
            chunk_boundaries.append((current_start, optimal_end))
            
            # Next chunk starts with overlap
            current_start = optimal_end - self.overlap_seconds
            current_start = max(current_start, 0.0)  # Don't go negative
        
        return chunk_boundaries
    
    def _find_optimal_split_point(self, audio_data: np.ndarray, 
                                  start_time: float, target_end: float) -> float:
        """
        Find the best place to split audio near the target end time.
        
        Args:
            audio_data: Audio data array
            start_time: Start time of current chunk
            target_end: Target end time
            
        Returns:
            Optimal split time in seconds
        """
        # Look for silence in a window around the target end
        search_window = 60  # Search 60 seconds around target
        search_start = max(target_end - search_window/2, start_time + 30*60)  # At least 30 min chunks
        search_end = min(target_end + search_window/2, len(audio_data)/self.sample_rate)
        
        start_sample = int(search_start * self.sample_rate)
        end_sample = int(search_end * self.sample_rate)
        
        if start_sample >= end_sample:
            return target_end
        
        # Analyze audio in search window
        window_audio = audio_data[start_sample:end_sample]
        
        # Find silence periods
        silence_periods = self._detect_silence_periods(
            window_audio, search_start
        )
        
        if silence_periods:
            # Choose the silence period closest to target end
            best_split = min(silence_periods, 
                           key=lambda x: abs(x[0] + (x[1] - x[0])/2 - target_end))
            return best_split[0] + (best_split[1] - best_split[0]) / 2
        
        # No good silence found, use target end
        return target_end
    
    def _detect_silence_periods(self, audio_data: np.ndarray, 
                               time_offset: float) -> List[Tuple[float, float]]:
        """
        Detect periods of silence in audio data.
        
        Args:
            audio_data: Audio data to analyze
            time_offset: Time offset to add to detected periods
            
        Returns:
            List of (start_time, end_time) tuples for silence periods
        """
        # Calculate RMS in small windows
        window_size = int(0.1 * self.sample_rate)  # 100ms windows
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        # Find silence periods
        silence_periods = []
        in_silence = False
        silence_start = 0
        
        for i, rms in enumerate(rms_values):
            time_pos = time_offset + (i * window_size) / self.sample_rate
            
            if rms < self.silence_threshold:
                if not in_silence:
                    silence_start = time_pos
                    in_silence = True
            else:
                if in_silence:
                    silence_duration = time_pos - silence_start
                    if silence_duration >= self.min_silence_duration:
                        silence_periods.append((silence_start, time_pos))
                    in_silence = False
        
        return silence_periods
    
    def _process_chunk(self, audio_data: np.ndarray, start_time: float, 
                      end_time: float, chunk_num: int, 
                      original_file: str, output_dir: str = None) -> Optional[Dict]:
        """
        Process a single audio chunk.
        
        Args:
            audio_data: Full audio data
            start_time: Chunk start time in seconds
            end_time: Chunk end time in seconds
            chunk_num: Chunk number
            original_file: Original audio file path
            output_dir: Output directory for chunk files
            
        Returns:
            Chunk transcript dictionary
        """
        try:
            # Extract chunk audio
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Resample to 16kHz if needed (Whisper requirement)
            if self.sample_rate != 16000:
                import librosa
                chunk_audio = librosa.resample(
                    chunk_audio, orig_sr=self.sample_rate, target_sr=16000
                )
                target_sample_rate = 16000
            else:
                target_sample_rate = self.sample_rate
            
            # Ensure audio is float32
            if chunk_audio.dtype != np.float32:
                chunk_audio = chunk_audio.astype(np.float32)
            
            logger.info(f"Transcribing chunk {chunk_num}: {end_time-start_time:.1f}s duration")
            
            # Transcribe chunk with optimized settings
            result = self.transcriber.model.transcribe(
                chunk_audio,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False,
                word_timestamps=True,
                condition_on_previous_text=True,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                best_of=3,
                beam_size=3
            )
            
            # Adjust timestamps to global time
            segments = result.get('segments', [])
            for segment in segments:
                segment['start'] += start_time
                segment['end'] += start_time
                
                # Adjust word timestamps if available
                if 'words' in segment:
                    for word in segment['words']:
                        if 'start' in word:
                            word['start'] += start_time
                        if 'end' in word:
                            word['end'] += start_time
            
            chunk_transcript = {
                'chunk_num': chunk_num,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'text': result.get('text', '').strip(),
                'segments': segments,
                'language': result.get('language', 'en')
            }
            
            # Save chunk file if output directory specified
            if output_dir:
                self._save_chunk_file(chunk_transcript, original_file, output_dir)
            
            logger.info(f"Chunk {chunk_num} completed: {len(chunk_transcript['text'])} characters")
            return chunk_transcript
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {e}")
            return None
    
    def _save_chunk_file(self, chunk_transcript: Dict, original_file: str, output_dir: str):
        """Save individual chunk transcript to file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(original_file).stem
            chunk_file = Path(output_dir) / f"{base_name}_chunk_{chunk_transcript['chunk_num']:02d}.json"
            
            with open(chunk_file, 'w') as f:
                json.dump(chunk_transcript, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving chunk file: {e}")
    
    def _compile_transcript(self, chunk_transcripts: List[Dict], 
                           original_file: str, total_duration: float) -> Dict:
        """
        Compile individual chunk transcripts into final unified transcript.
        
        Args:
            chunk_transcripts: List of chunk transcript dictionaries
            original_file: Original audio file path
            total_duration: Total audio duration in seconds
            
        Returns:
            Unified transcript dictionary
        """
        # Sort chunks by start time
        chunk_transcripts.sort(key=lambda x: x['start_time'])
        
        # Merge overlapping segments and remove duplicates
        all_segments = []
        full_text_parts = []
        
        for chunk in chunk_transcripts:
            # Handle overlap removal for text
            chunk_text = chunk['text']
            
            # For segments, we need to be smarter about overlaps
            chunk_segments = chunk['segments']
            
            if all_segments:
                # Remove overlapping segments
                last_end_time = all_segments[-1]['end']
                chunk_segments = [seg for seg in chunk_segments if seg['start'] >= last_end_time - 1.0]  # 1s tolerance
            
            all_segments.extend(chunk_segments)
            
            # For full text, handle overlap more intelligently
            if full_text_parts:
                # Try to merge overlapping text intelligently
                chunk_text = self._merge_overlapping_text(full_text_parts[-1], chunk_text)
            
            full_text_parts.append(chunk_text)
        
        # Create final transcript
        final_transcript = {
            'metadata': {
                'source_file': original_file,
                'processing_method': 'chunked_hybrid',
                'chunk_count': len(chunk_transcripts),
                'chunk_duration_minutes': self.chunk_duration_minutes,
                'overlap_seconds': self.overlap_seconds,
                'model_size': self.model_size,
                'total_duration_seconds': total_duration,
                'total_duration_formatted': str(timedelta(seconds=int(total_duration))),
                'created_at': datetime.now().isoformat(),
                'segments_count': len(all_segments)
            },
            'segments': all_segments,
            'full_text': ' '.join(full_text_parts).strip(),
            'chunks_info': [
                {
                    'chunk_num': chunk['chunk_num'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'duration': chunk['duration'],
                    'text_length': len(chunk['text'])
                }
                for chunk in chunk_transcripts
            ]
        }
        
        logger.info(f"Final transcript compiled: {len(final_transcript['full_text'])} characters, {len(all_segments)} segments")
        return final_transcript
    
    def _merge_overlapping_text(self, previous_text: str, current_text: str) -> str:
        """
        Intelligently merge overlapping text from consecutive chunks.
        
        Args:
            previous_text: Text from previous chunk
            current_text: Text from current chunk
            
        Returns:
            Merged text with overlap removed
        """
        # Simple approach: look for common ending/beginning phrases
        prev_words = previous_text.split()
        curr_words = current_text.split()
        
        if not prev_words or not curr_words:
            return current_text
        
        # Look for overlap in last/first 10 words
        max_overlap = min(10, len(prev_words), len(curr_words))
        
        for overlap_len in range(max_overlap, 0, -1):
            prev_ending = ' '.join(prev_words[-overlap_len:])
            curr_beginning = ' '.join(curr_words[:overlap_len])
            
            # If we find a match, remove the overlap from current text
            if prev_ending.lower() == curr_beginning.lower():
                remaining_words = curr_words[overlap_len:]
                return ' '.join(remaining_words) if remaining_words else ""
        
        # No overlap found, return current text as-is
        return current_text
