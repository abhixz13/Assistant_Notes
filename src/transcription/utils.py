"""
Utility functions for transcription module.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf
import numpy as np

from utils.config import config
from utils.logger import get_logger
from .whisper_local import WhisperTranscriber


logger = get_logger("transcription.utils")


def transcribe_audio_file(audio_file_path: str, model_size: str = "medium") -> Dict:
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_file_path: Path to the audio file
        model_size: Whisper model size
        
    Returns:
        Dict: Complete transcript with metadata
    """
    try:
        logger.info(f"Starting transcription of: {audio_file_path}")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size=model_size)
        
        # Load model
        if not transcriber.load_model():
            raise Exception("Failed to load Whisper model")
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_file_path)
        
        # Check audio quality
        logger.info(f"Audio file info: shape={audio_data.shape}, sample_rate={sample_rate}, dtype={audio_data.dtype}")
        
        # Check if audio has meaningful content
        audio_rms = np.sqrt(np.mean(audio_data**2))
        logger.info(f"Audio RMS level: {audio_rms}")
        
        if audio_rms < 0.001:  # Very low audio level
            logger.warning("Audio file has very low volume - may not contain speech")
        
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            logger.info("Audio normalized to prevent clipping")
        
        # Transcribe the entire file with improved settings
        logger.info("Transcribing audio file...")
        result = transcriber.model.transcribe(
            audio_data,
            language="en",
            task="transcribe",
            fp16=False,
            verbose=True,  # More detailed output
            word_timestamps=True,  # Get word-level timestamps
            condition_on_previous_text=True,  # Better context
            temperature=0.0,  # More deterministic
            compression_ratio_threshold=2.4,  # Better quality threshold
            logprob_threshold=-1.0,  # Lower threshold for more content
            no_speech_threshold=0.6  # Lower threshold to capture more speech
        )
        
        # Check if transcription produced meaningful content
        full_text = result.get('text', '').strip()
        segments = result.get('segments', [])
        
        # Check for common "no audio" patterns
        no_audio_patterns = ['you', 'um', 'uh', 'ah', 'hmm', '...', '']
        meaningful_content = False
        
        if full_text and len(full_text) > 10:
            # Check if text contains more than just filler words
            words = full_text.lower().split()
            meaningful_words = [w for w in words if w not in no_audio_patterns and len(w) > 2]
            meaningful_content = len(meaningful_words) > 5
        
        if not meaningful_content:
            logger.warning("No meaningful audio content detected")
            # Create a "no audio" transcript
            transcript = {
                "metadata": {
                    "audio_file": audio_file_path,
                    "model_size": model_size,
                    "duration_seconds": len(audio_data) / sample_rate if 'audio_data' in locals() else 0,
                    "segments_count": 0,
                    "created_at": datetime.now().isoformat(),
                    "duration_formatted": "00:00",
                    "youtube_url": None,
                    "title": "No Audio Content Detected",
                    "generated_at": datetime.now().isoformat(),
                    "error": "No meaningful audio content detected. Please check: 1) Audio is playing, 2) BlackHole is routing audio correctly, 3) Recording duration was sufficient"
                },
                "segments": [],
                "full_text": "",
                "key_points": [],
                "technical_terms": {},
                "formatted_notes": "# AI Notes Assistant - No Audio Detected\n\nNo meaningful audio content was detected in the recording.\n\n## Possible Issues:\n- YouTube video was not playing\n- Audio not routed through BlackHole\n- Recording duration too short\n- Audio quality issues\n\nPlease try again with audio playing."
            }
            return transcript
        
        # Create transcript structure
        transcript = _create_transcript_from_result(
            result, 
            audio_file_path, 
            model_size,
            transcriber
        )
        
        logger.info(f"Transcription completed: {len(transcript['segments'])} segments")
        return transcript
        
    except Exception as e:
        logger.error(f"Error transcribing file {audio_file_path}: {e}")
        raise


def transcribe_batch_files(audio_dir: str, model_size: str = "small") -> List[Dict]:
    """
    Transcribe all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        model_size: Whisper model size
        
    Returns:
        List[Dict]: List of transcripts
    """
    try:
        audio_dir_path = Path(audio_dir)
        if not audio_dir_path.exists():
            raise Exception(f"Audio directory does not exist: {audio_dir}")
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_dir_path.glob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in: {audio_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files to transcribe")
        
        # Initialize transcriber once
        transcriber = WhisperTranscriber(model_size=model_size)
        if not transcriber.load_model():
            raise Exception("Failed to load Whisper model")
        
        transcripts = []
        
        for audio_file in audio_files:
            try:
                logger.info(f"Transcribing: {audio_file.name}")
                
                # Load audio file
                audio_data, sample_rate = sf.read(str(audio_file))
                
                # Ensure audio is in the right format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Transcribe
                result = transcriber.model.transcribe(
                    audio_data,
                    language="en",
                    task="transcribe",
                    fp16=False
                )
                
                # Create transcript
                transcript = _create_transcript_from_result(
                    result, 
                    str(audio_file), 
                    model_size,
                    transcriber
                )
                
                transcripts.append(transcript)
                logger.info(f"Completed: {audio_file.name}")
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_file}: {e}")
                continue
        
        logger.info(f"Batch transcription completed: {len(transcripts)} files")
        return transcripts
        
    except Exception as e:
        logger.error(f"Error in batch transcription: {e}")
        raise


def _create_transcript_from_result(result: Dict, audio_file_path: str, model_size: str, transcriber: WhisperTranscriber) -> Dict:
    """
    Create structured transcript from Whisper result.
    
    Args:
        result: Whisper transcription result
        audio_file_path: Path to the original audio file
        model_size: Model size used
        transcriber: Transcriber instance for utilities
        
    Returns:
        Dict: Structured transcript
    """
    # Extract segments with timestamps
    segments = []
    for segment in result.get("segments", []):
        segments.append({
            "timestamp": transcriber._format_timestamp(segment["start"]),
            "time_seconds": segment["start"],
            "text": segment["text"].strip()
        })
    
    # Get full text
    full_text = result.get("text", "").strip()
    
    # Extract key points and technical terms
    key_points = transcriber._extract_key_points(full_text)
    technical_terms = transcriber._extract_technical_terms(full_text)
    
    # Create metadata
    metadata = {
        "audio_file": audio_file_path,
        "model_size": model_size,
        "duration_seconds": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
        "segments_count": len(segments),
        "created_at": datetime.now().isoformat()
    }
    
    # Format duration
    if metadata["duration_seconds"]:
        metadata["duration_formatted"] = transcriber._format_timestamp(metadata["duration_seconds"])
    else:
        metadata["duration_formatted"] = "00:00"
    
    transcript = {
        "metadata": metadata,
        "segments": segments,
        "full_text": full_text,
        "key_points": key_points,
        "technical_terms": technical_terms,
        "formatted_notes": _create_formatted_notes_from_transcript(
            segments, full_text, key_points, technical_terms, metadata
        )
    }
    
    return transcript


def _create_formatted_notes_from_transcript(segments: List[Dict], full_text: str, key_points: List[str], 
                                          technical_terms: Dict[str, str], metadata: Dict) -> str:
    """
    Create formatted notes from transcript data.
    
    Args:
        segments: Transcript segments with timestamps
        full_text: Complete transcript text
        key_points: Extracted key points
        technical_terms: Extracted technical terms
        metadata: Transcript metadata
        
    Returns:
        str: Formatted notes
    """
    notes = []
    notes.append("# AI Notes Assistant - Transcript\n")
    notes.append(f"Source: {Path(metadata['audio_file']).name}\n")
    notes.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    notes.append(f"Duration: {metadata.get('duration_formatted', 'Unknown')}\n")
    notes.append(f"Model: Whisper {metadata['model_size']}\n")
    notes.append("---\n\n")
    
    # Add transcript segments
    notes.append("## Transcript\n")
    for segment in segments:
        notes.append(f"[{segment['timestamp']}] {segment['text']}\n")
    
    # Add key points
    if key_points:
        notes.append("\n## Key Points\n")
        for i, point in enumerate(key_points, 1):
            notes.append(f"{i}. {point}\n")
    
    # Add technical terms
    if technical_terms:
        notes.append("\n## Technical Terms\n")
        for term, context in technical_terms.items():
            notes.append(f"- **{term.title()}**: {context}\n")
    
    return "".join(notes)


def save_transcript(transcript: Dict, output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save transcript to file.
    
    Args:
        transcript: Transcript dictionary
        output_dir: Output directory
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        str: Path to saved file
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.json"
        
        # Save JSON transcript
        json_file = output_path / filename
        with open(json_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        # Save formatted notes
        notes_filename = filename.replace('.json', '.md')
        notes_file = output_path / notes_filename
        with open(notes_file, 'w') as f:
            f.write(transcript['formatted_notes'])
        
        logger.info(f"Transcript saved: {json_file}")
        logger.info(f"Formatted notes saved: {notes_file}")
        
        return str(json_file)
        
    except Exception as e:
        logger.error(f"Error saving transcript: {e}")
        raise


def get_transcription_status() -> Dict:
    """
    Get transcription module status.
    
    Returns:
        Dict: Status information
    """
    try:
        # Check if Whisper model can be loaded
        transcriber = WhisperTranscriber()
        model_loaded = transcriber.load_model()
        
        return {
            "module": "transcription",
            "status": "ready" if model_loaded else "error",
            "model_loaded": model_loaded,
            "supported_formats": ['.wav', '.mp3', '.flac', '.m4a', '.aac'],
            "supported_models": ['tiny', 'base', 'small', 'medium', 'large']
        }
        
    except Exception as e:
        logger.error(f"Error getting transcription status: {e}")
        return {
            "module": "transcription",
            "status": "error",
            "error": str(e)
        }


def validate_audio_file(audio_file_path: str) -> bool:
    """
    Validate if audio file can be transcribed.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        bool: True if file is valid
    """
    try:
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file does not exist: {audio_file_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_file_path}")
            return False
        
        # Check file extension
        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        file_ext = Path(audio_file_path).suffix.lower()
        
        if file_ext not in supported_extensions:
            logger.error(f"Unsupported audio format: {file_ext}")
            return False
        
        # Try to load audio file
        try:
            audio_data, sample_rate = sf.read(audio_file_path)
            if len(audio_data) == 0:
                logger.error(f"Audio file has no data: {audio_file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return False
        
        logger.info(f"Audio file validated: {audio_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        return False
