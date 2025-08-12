#!/usr/bin/env python3
"""
AI Video Summarization Pipeline
Focused on AI/ML content with start/stop functionality.
"""

import click
import sys
import signal
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from audio.capture import AudioCapture
from transcription.utils import transcribe_audio_file, save_transcript
from summarization.custom_model import CustomModelProcessor
from utils.config import ConfigManager
from utils.logger import get_logger

logger = get_logger("ai_pipeline")

class AIVideoPipeline:
    """AI-focused video summarization pipeline with start/stop functionality."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.audio_capture = None
        self.is_recording = False
        self.recording_start_time = None
        self.audio_file_path = None
        
    def start_recording(self, youtube_url: Optional[str] = None, title: Optional[str] = None):
        """Start recording system audio."""
        try:
            click.echo("🎬 Starting AI Video Summarization Pipeline...")
            click.echo("📹 Recording system audio (YouTube video should be playing)")
            click.echo("⏹️  Press Ctrl+C to stop and generate notes")
            click.echo("---")
            
            # Initialize audio capture
            audio_config = self.config.get_audio_config()
            self.audio_capture = AudioCapture(audio_config)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if title:
                # Clean title for filename
                clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title.replace(' ', '_')[:50]  # Limit length
                filename = f"{clean_title}_{timestamp}.wav"
            else:
                filename = f"AI_Video_{timestamp}.wav"
            
            self.audio_file_path = f"data/transcripts/{filename}"
            
            # Start recording
            self.is_recording = True
            self.recording_start_time = time.time()
            
            # Start recording in background
            self.audio_capture.record_continuous(output_path=self.audio_file_path)
            
            click.echo(f"✅ Recording started: {filename}")
            click.echo("🎧 Listening to system audio...")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            click.echo(f"❌ Error starting recording: {e}")
            raise
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return the audio file path."""
        if not self.is_recording:
            click.echo("❌ No recording in progress")
            return None
        
        try:
            # Stop recording
            if self.audio_capture:
                self.audio_capture.stop_recording()
            
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time
            
            click.echo(f"⏹️  Recording stopped after {recording_duration:.1f} seconds")
            click.echo(f"📁 Audio saved: {self.audio_file_path}")
            
            return self.audio_file_path
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            click.echo(f"❌ Error stopping recording: {e}")
            return None
    
    def process_recording(self, audio_file_path: str, youtube_url: Optional[str] = None, title: Optional[str] = None):
        """Process the recorded audio through the full pipeline."""
        try:
            # Step 1: Transcribe
            click.echo("🔤 Transcribing audio...")
            from utils.config import ConfigManager
            # Read model size from config with 'medium' fallback
            cfg = ConfigManager()
            model_size = cfg.get_transcription_config().get('model_size', 'medium')
            transcript = transcribe_audio_file(audio_file_path, model_size=model_size)
            
            # Check if transcript has error or no content
            if 'error' in transcript.get('metadata', {}):
                error_msg = transcript['metadata']['error']
                click.echo(f"❌ {error_msg}")
                click.echo("💡 Troubleshooting tips:")
                click.echo("   1. Make sure YouTube video is playing and you can hear audio")
                click.echo("   2. Check System Preferences → Sound → Output → BlackHole 2ch")
                click.echo("   3. Try recording for longer (2-3 minutes minimum)")
                click.echo("   4. Check if BlackHole is properly installed")
                return None, None
            
            if not transcript['full_text'].strip():
                click.echo("❌ No audio content detected! Please check:")
                click.echo("   1. YouTube video is playing")
                click.echo("   2. System audio is enabled")
                click.echo("   3. BlackHole audio driver is working")
                click.echo("   4. Recording duration was sufficient")
                return None, None
            
            # Generate title from content if not provided
            if not title:
                if transcript['full_text'].strip():
                    title = self._generate_title_from_content(transcript['full_text'])
                else:
                    title = "AI Video Summary"  # Fallback for empty transcripts
            
            # Add metadata
            transcript['metadata']['youtube_url'] = youtube_url
            transcript['metadata']['title'] = title
            transcript['metadata']['generated_at'] = datetime.now().isoformat()
            
            # Save transcript
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_title = clean_title.replace(' ', '_')[:50]
            
            transcript_filename = f"{clean_title}_{timestamp}.json"
            transcript_path = f"data/transcripts/{transcript_filename}"
            
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)
            
            click.echo(f"✅ Transcript saved: {transcript_filename}")
            
            # Step 2: Generate AI notes
            click.echo("🧠 Generating AI-focused notes...")
            processor = CustomModelProcessor()
            enhanced_transcript = processor.enhance_transcript(transcript)
            
            # Step 3: Save notes in TXT format
            notes_filename = f"{clean_title}_{timestamp}.txt"
            notes_path = f"data/notes/{notes_filename}"
            
            notes_content = self._format_notes_as_txt(enhanced_transcript, title, youtube_url)
            
            Path("data/notes").mkdir(parents=True, exist_ok=True)
            with open(notes_path, 'w', encoding='utf-8') as f:
                f.write(notes_content)
            
            click.echo(f"✅ AI notes saved: {notes_filename}")
            click.echo("---")
            click.echo("🎉 Pipeline completed successfully!")
            click.echo(f"📄 Transcript: data/transcripts/{transcript_filename}")
            click.echo(f"📝 Notes: data/notes/{notes_filename}")
            
            return transcript_path, notes_path
            
        except Exception as e:
            logger.error(f"Failed to process recording: {e}")
            click.echo(f"❌ Error processing recording: {e}")
            raise
    
    def _generate_title_from_content(self, text: str) -> str:
        """Generate a title from the transcript content."""
        try:
            # Use the first few sentences to generate a title
            sentences = text.split('.')[:3]
            title_text = '. '.join(sentences) + '.'
            
            processor = CustomModelProcessor()
            prompt = f"""
            Generate a concise, descriptive title (5-8 words) for this AI/ML content.
            The title should capture the main AI topic or theme.
            
            Content: {title_text[:500]}...
            
            Title:
            """
            
            messages = [
                {"role": "system", "content": "You are an AI/ML expert. Generate concise, descriptive titles for AI content."},
                {"role": "user", "content": prompt}
            ]
            
            response = processor._make_api_request(messages, 100)
            
            if response:
                return response.strip().strip('"').strip("'")
            
            return "AI Video Summary"
            
        except Exception as e:
            logger.error(f"Failed to generate title: {e}")
            return "AI Video Summary"
    
    def _format_notes_as_txt(self, enhanced_transcript: dict, title: str, youtube_url: Optional[str] = None) -> str:
        """Format enhanced notes as plain text."""
        notes = enhanced_transcript.get('enhanced_notes', {})
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"AI VIDEO SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Title: {title}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if youtube_url:
            lines.append(f"Source: {youtube_url}")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        if notes.get('summary'):
            lines.append("SUMMARY")
            lines.append("-" * 40)
            lines.append(notes['summary'])
            lines.append("")
        
        # Additional Details
        additional_details = notes.get('additional_details', {})
        if additional_details.get('related_technologies'):
            lines.append("RELATED TECHNOLOGIES")
            lines.append("-" * 40)
            for tech in additional_details['related_technologies']:
                lines.append(f"• {tech}")
            lines.append("")
        
        if additional_details.get('learning_resources'):
            lines.append("LEARNING RESOURCES")
            lines.append("-" * 40)
            for resource in additional_details['learning_resources']:
                lines.append(f"• {resource}")
            lines.append("")
        
        if additional_details.get('best_practices'):
            lines.append("BEST PRACTICES")
            lines.append("-" * 40)
            for practice in additional_details['best_practices']:
                lines.append(f"• {practice}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("End of AI Video Summary")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Global pipeline instance
pipeline = None

def signal_handler(signum, frame):
    """Handle Ctrl+C to stop recording."""
    global pipeline
    if pipeline and pipeline.is_recording:
        click.echo("\n⏹️  Stopping recording...")
        audio_file = pipeline.stop_recording()
        if audio_file:
            pipeline.process_recording(audio_file)
    sys.exit(0)


@click.group()
def cli():
    """AI Video Summarization Pipeline - Focused on AI/ML content."""
    pass


@cli.command()
@click.option('--youtube-url', help='YouTube URL for metadata')
@click.option('--title', help='Video title (auto-generated if not provided)')
def start(youtube_url: Optional[str], title: Optional[str]):
    """Start recording system audio for AI video summarization."""
    global pipeline
    
    try:
        pipeline = AIVideoPipeline()
        pipeline.start_recording(youtube_url=youtube_url, title=title)
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep running until interrupted
        while pipeline.is_recording:
            time.sleep(1)
            
    except KeyboardInterrupt:
        click.echo("\n⏹️  Interrupted by user")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--youtube-url', help='YouTube URL for metadata')
@click.option('--title', help='Video title (auto-generated if not provided)')
def process(audio_file: str, youtube_url: Optional[str], title: Optional[str]):
    """Process an existing audio file through the AI pipeline."""
    try:
        pipeline = AIVideoPipeline()
        pipeline.process_recording(audio_file, youtube_url=youtube_url, title=title)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli() 