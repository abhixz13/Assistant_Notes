"""
Processing Module Utilities

Utility functions for transcript processing, note generation,
and content enhancement.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from utils.config import ConfigManager
from utils.logger import get_logger
from .custom_model import CustomModelProcessor


def process_transcript(transcript_path: str, output_dir: Optional[str] = None, expertise_level: str = "moderate", model_override: Optional[str] = None) -> Dict:
    """
    Process a transcript file and generate enhanced notes.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_dir: Directory to save enhanced notes (optional)
        
    Returns:
        Enhanced transcript dictionary
    """
    logger = get_logger(__name__)
    
    try:
        # Load transcript
        logger.info(f"Loading transcript from: {transcript_path}")
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        # Initialize processor with model override
        processor = CustomModelProcessor(model_override)
        
        # Enhance transcript
        logger.info(f"Enhancing transcript with AI processing (expertise level: {expertise_level})")
        enhanced_transcript = processor.enhance_transcript(transcript, expertise_level)
        
        # Save enhanced notes
        if output_dir:
            save_enhanced_notes(enhanced_transcript, output_dir)
        
        logger.info("Transcript processing completed successfully")
        return enhanced_transcript
        
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        return {'error': str(e)}


def process_batch_transcripts(transcript_dir: str, output_dir: Optional[str] = None) -> List[Dict]:
    """
    Process all transcript files in a directory.
    
    Args:
        transcript_dir: Directory containing transcript files
        output_dir: Directory to save enhanced notes (optional)
        
    Returns:
        List of enhanced transcript dictionaries
    """
    logger = get_logger(__name__)
    
    try:
        transcript_dir_path = Path(transcript_dir)
        transcript_files = list(transcript_dir_path.glob("transcript_*.json"))
        
        if not transcript_files:
            logger.warning(f"No transcript files found in: {transcript_dir}")
            return []
        
        logger.info(f"Found {len(transcript_files)} transcript files to process")
        
        # Initialize processor once
        processor = CustomModelProcessor()
        
        enhanced_transcripts = []
        
        for transcript_file in transcript_files:
            try:
                logger.info(f"Processing: {transcript_file.name}")
                
                # Load transcript
                with open(transcript_file, 'r') as f:
                    transcript = json.load(f)
                
                # Enhance transcript
                enhanced_transcript = processor.enhance_transcript(transcript)
                enhanced_transcripts.append(enhanced_transcript)
                
                # Save enhanced notes
                if output_dir:
                    save_enhanced_notes(enhanced_transcript, output_dir)
                
            except Exception as e:
                logger.error(f"Error processing {transcript_file.name}: {e}")
                continue
        
        logger.info(f"Batch processing completed: {len(enhanced_transcripts)} files processed")
        return enhanced_transcripts
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return []


def save_enhanced_notes(enhanced_transcript: Dict, output_dir: str) -> str:
    """
    Save enhanced notes to files.
    
    Args:
        enhanced_transcript: Enhanced transcript dictionary
        output_dir: Directory to save notes
        
    Returns:
        Path to saved notes file
    """
    logger = get_logger(__name__)
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract metadata
        metadata = enhanced_transcript.get('metadata', {})
        original_filename = metadata.get('filename', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filenames
        base_name = original_filename.replace('.wav', '').replace('.mp3', '')
        txt_filename = f"notes_{base_name}_{timestamp}.txt"
        
        # Save TXT version
        txt_path = output_path / txt_filename
        txt_content = create_txt_notes(enhanced_transcript)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        logger.info(f"Enhanced notes saved: {txt_path}")
        return str(txt_path)
        
    except Exception as e:
        logger.error(f"Error saving enhanced notes: {e}")
        return ""


def create_markdown_notes(enhanced_transcript: Dict) -> str:
    """
    Create formatted Markdown notes from enhanced transcript.
    
    Args:
        enhanced_transcript: Enhanced transcript dictionary
        
    Returns:
        Formatted Markdown content
    """
    try:
        # Extract components
        enhanced_notes = enhanced_transcript.get('enhanced_notes', {})
        metadata = enhanced_transcript.get('metadata', {})
        
        # Build Markdown content
        markdown_parts = []
        
        # Title
        title = enhanced_notes.get('title', 'YouTube Video Notes')
        markdown_parts.append(f"# {title}\n")
        
        # Video Information
        markdown_parts.append("## Video Information")
        markdown_parts.append(f"- **Video:** {title}")
        markdown_parts.append(f"- **Video Transcript ID:** {metadata.get('filename', 'unknown')}")
        markdown_parts.append(f"- **Duration:** {metadata.get('duration_formatted', 'Unknown')}")
        markdown_parts.append(f"- **Model:** {enhanced_notes.get('metadata', {}).get('model_used', 'Unknown')}")
        markdown_parts.append("")
        
        # Summary
        summary = enhanced_notes.get('summary', '')
        if summary:
            markdown_parts.append("## Key Topics")
            markdown_parts.append("")
            markdown_parts.append("### 1. Summary")
            markdown_parts.append(summary)
            markdown_parts.append("")
        
        # Key Discussion Points
        key_discussion_points = enhanced_notes.get('key_discussion_points', {})
        if key_discussion_points.get('content'):
            markdown_parts.append("### 2. Key Discussion Points")
            markdown_parts.append(key_discussion_points['content'])
            markdown_parts.append("")
        
        # Additional Details
        additional_details = enhanced_notes.get('additional_details', {})
        if additional_details.get('content'):
            markdown_parts.append("### 3. Additional Details")
            markdown_parts.append(additional_details['content'])
            markdown_parts.append("")
        
        # Quick Reference
        topics = enhanced_notes.get('metadata', {}).get('topics', [])
        if topics:
            markdown_parts.append("---")
            markdown_parts.append("## Quick Reference")
            markdown_parts.append(f"- **Main Topics:** {', '.join(topics[:5])}")
            markdown_parts.append("- **Focus:** Key concepts and practical applications")
            markdown_parts.append("- **Format:** Simple, scannable structure")
            markdown_parts.append("")
        
        return "\n".join(markdown_parts)
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating Markdown notes: {e}")
        return f"# Error Creating Notes\n\nError: {str(e)}"


def create_txt_notes(enhanced_transcript: Dict) -> str:
    """
    Create formatted TXT notes from enhanced transcript.
    
    Args:
        enhanced_transcript: Enhanced transcript dictionary
        
    Returns:
        Formatted TXT content
    """
    try:
        # Extract components
        enhanced_notes = enhanced_transcript.get('enhanced_notes', {})
        metadata = enhanced_transcript.get('metadata', {})
        
        # Build TXT content
        txt_parts = []
        
        # Title
        title = enhanced_notes.get('title', 'YouTube Video Notes')
        txt_parts.append(f"{title}")
        txt_parts.append("=" * len(title))
        txt_parts.append("")
        
        # Video Information
        txt_parts.append("VIDEO INFORMATION")
        txt_parts.append("-" * 20)
        txt_parts.append(f"Video: {title}")
        txt_parts.append(f"Video Transcript ID: {metadata.get('filename', 'unknown')}")
        txt_parts.append(f"Duration: {metadata.get('duration_formatted', 'Unknown')}")
        txt_parts.append(f"Model: {enhanced_notes.get('metadata', {}).get('model_used', 'Unknown')}")
        txt_parts.append("")
        
        # Summary
        summary = enhanced_notes.get('summary', '')
        if summary:
            txt_parts.append("KEY TOPICS")
            txt_parts.append("-" * 20)
            txt_parts.append("")
            txt_parts.append("1. SUMMARY")
            txt_parts.append(summary)
            txt_parts.append("")
        
        # Key Discussion Points
        key_discussion_points = enhanced_notes.get('key_discussion_points', {})
        if key_discussion_points.get('content'):
            txt_parts.append("2. KEY DISCUSSION POINTS")
            txt_parts.append(key_discussion_points['content'])
            txt_parts.append("")
        
        # Additional Details
        additional_details = enhanced_notes.get('additional_details', {})
        if additional_details.get('content'):
            txt_parts.append("3. ADDITIONAL DETAILS")
            txt_parts.append(additional_details['content'])
            txt_parts.append("")
        
        # Quick Reference
        topics = enhanced_notes.get('metadata', {}).get('topics', [])
        if topics:
            txt_parts.append("QUICK REFERENCE")
            txt_parts.append("-" * 20)
            txt_parts.append(f"Main Topics: {', '.join(topics[:5])}")
            txt_parts.append("Focus: Key concepts and practical applications")
            txt_parts.append("Format: Simple, scannable structure")
            txt_parts.append("")
        
        return "\n".join(txt_parts)
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating TXT notes: {e}")
        return f"Error Creating Notes\n\nError: {str(e)}"


def get_processing_status() -> Dict:
    """
    Get the status of the processing module.
    
    Returns:
        Status dictionary
    """
    try:
        processor = CustomModelProcessor()
        status = processor.get_status()
        return {
            'module': 'processing',
            'model_loaded': status.get('status') == 'ready',
            'model_path': status.get('api_url', 'http://localhost:11434/v1'),
            'config_loaded': True,
            'status': status.get('status', 'unknown')
        }
    except Exception as e:
        return {
            'module': 'processing',
            'error': str(e),
            'status': 'error'
        }


def validate_transcript_file(transcript_path: str) -> bool:
    """
    Validate that a transcript file exists and has the correct format.
    
    Args:
        transcript_path: Path to transcript file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not Path(transcript_path).exists():
            return False
        
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        # Check required fields
        required_fields = ['segments', 'full_text', 'metadata']
        return all(field in transcript for field in required_fields)
        
    except Exception:
        return False
