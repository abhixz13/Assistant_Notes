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
from .llama_local import MixtralProcessor


def process_transcript(transcript_path: str, output_dir: Optional[str] = None) -> Dict:
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
        
        # Initialize processor
        processor = MixtralProcessor()
        
        if not processor.model:
            logger.warning("Mixtral model not loaded, using fallback processing")
            enhanced_transcript = processor._fallback_enhance_transcript(transcript)
        else:
            # Enhance transcript
            logger.info("Enhancing transcript with AI processing")
            enhanced_transcript = processor.enhance_transcript(transcript)
        
        # Enhance transcript
        logger.info("Enhancing transcript with AI processing")
        enhanced_transcript = processor.enhance_transcript(transcript)
        
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
        processor = MixtralProcessor()
        
        if not processor.model:
            logger.error("Mixtral model not loaded, cannot process transcripts")
            return []
        
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
        json_filename = f"enhanced_{base_name}_{timestamp}.json"
        md_filename = f"notes_{base_name}_{timestamp}.md"
        
        # Save JSON version
        json_path = output_path / json_filename
        with open(json_path, 'w') as f:
            json.dump(enhanced_transcript, f, indent=2)
        
        # Save Markdown version
        md_path = output_path / md_filename
        markdown_content = create_markdown_notes(enhanced_transcript)
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Enhanced notes saved: {json_path}, {md_path}")
        return str(json_path)
        
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


def get_processing_status() -> Dict:
    """
    Get the status of the processing module.
    
    Returns:
        Status dictionary
    """
    try:
        processor = MixtralProcessor()
        return {
            'module': 'processing',
            'model_loaded': processor.model is not None,
            'model_path': processor.model_path,
            'config_loaded': True,
            'status': 'ready' if processor.model else 'model_not_loaded'
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
