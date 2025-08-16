"""
Main CLI interface for AI Notes Assistant.
Provides command-line interface for all modules.
"""

import click
import sys
import json
import glob
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import config
from utils.config import ConfigManager
from utils.logger import get_logger


logger = get_logger("cli")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config-file', type=click.Path(exists=True), help='Custom config file path')
def cli(debug: bool, config_file: Optional[str]):
    """
    AI Notes Assistant - Personal AI Tutor for YouTube Content
    
    Capture YouTube audio, transcribe it, and generate structured notes.
    """
    if debug:
        config._config['development']['debug_mode'] = True
        logger.setLevel('DEBUG')
        logger.debug("Debug mode enabled")
    
    if config_file:
        config.config_path = Path(config_file)
        config.reload()
        logger.info(f"Using custom config file: {config_file}")


@cli.command()
@click.option('--duration', type=int, help='Recording duration in seconds (0 for continuous)')
@click.option('--output', type=click.Path(), help='Output audio file path')
@click.option('--device', help='Audio device name')
def record(duration: Optional[int], output: Optional[str], device: Optional[str]):
    """
    Record system audio using BlackHole.
    
    This command captures system audio and saves it to a file.
    """
    try:
        from audio.capture import AudioCapture
        
        # Override config with CLI options
        audio_config = config.get_audio_config()
        if duration is not None:
            audio_config['recording_duration'] = duration
        if device:
            audio_config['blackhole_device'] = device
        
        capture = AudioCapture(audio_config)
        
        if output:
            capture.record_to_file(duration=duration, output_path=output)
        else:
            capture.record_to_file(duration=duration)
            
    except ImportError:
        click.echo("Error: Audio module not implemented yet.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output transcript file path')
@click.option('--model-size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), 
              default='small', help='Whisper model size')
def transcribe(audio_file: str, output: Optional[str], model_size: str):
    """
    Transcribe audio file using OpenAI Whisper.
    
    AUDIO_FILE: Path to the audio file to transcribe
    """
    try:
        from transcription.utils import transcribe_audio_file, save_transcript
        
        logger.info(f"Transcribing audio file: {audio_file}")
        
        # Transcribe the audio file
        transcript = transcribe_audio_file(audio_file, model_size=model_size)
        
        # Save transcript
        if output:
            output_path = Path(output)
            output_dir = output_path.parent
            filename = output_path.name
        else:
            output_dir = "data/transcripts"
            filename = None
        
        saved_file = save_transcript(transcript, output_dir, filename)
        
        click.echo(f"✅ Transcription completed!")
        click.echo(f"📄 Transcript saved: {saved_file}")
        click.echo(f"📝 Formatted notes saved: {saved_file.replace('.json', '.md')}")
        click.echo(f"⏱️  Duration: {transcript['metadata']['duration_formatted']}")
        click.echo(f"📊 Segments: {transcript['metadata']['segments_count']}")
        
    except ImportError:
        click.echo("Error: Transcription module not available.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model-size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), 
              default='small', help='Whisper model size')
@click.option('--output-dir', type=click.Path(), default='data/transcripts', 
              help='Output directory for transcripts')
def transcribe_live(model_size: str, output_dir: str):
    """
    Start real-time transcription from system audio.
    
    This command captures live audio and transcribes it in real-time.
    Press Ctrl+C to stop transcription.
    """
    try:
        from transcription.whisper_local import WhisperTranscriber
        import signal
        import time
        
        logger.info("Starting real-time transcription...")
        click.echo("🎧 Starting real-time transcription...")
        click.echo("📝 Press Ctrl+C to stop and save transcript")
        click.echo("---")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size=model_size)
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            click.echo("\n🛑 Stopping transcription...")
            transcript = transcriber.stop_streaming()
            
            # Save transcript
            from transcription.utils import save_transcript
            saved_file = save_transcript(transcript, output_dir)
            
            click.echo(f"✅ Transcription completed!")
            click.echo(f"📄 Transcript saved: {saved_file}")
            click.echo(f"📝 Formatted notes saved: {saved_file.replace('.json', '.md')}")
            click.echo(f"⏱️  Duration: {transcript['metadata']['duration_formatted']}")
            click.echo(f"📊 Segments: {transcript['metadata']['segments_count']}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start streaming
        if not transcriber.start_streaming():
            click.echo("Error: Failed to start real-time transcription")
            sys.exit(1)
        
        # Show live status
        click.echo("🎤 Listening for audio...")
        
        while transcriber.is_streaming:
            status = transcriber.get_live_status()
            click.echo(f"\r⏱️  {status['duration']} | 📊 {status['segments_count']} segments | 🎵 {status['queue_size']} chunks", nl=False)
            time.sleep(1)
            
    except ImportError:
        click.echo("Error: Transcription module not available.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Real-time transcription failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('audio_dir', type=click.Path(exists=True))
@click.option('--model-size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), 
              default='small', help='Whisper model size')
@click.option('--output-dir', type=click.Path(), default='data/transcripts', 
              help='Output directory for transcripts')
def transcribe_batch(audio_dir: str, model_size: str, output_dir: str):
    """
    Transcribe all audio files in a directory.
    
    AUDIO_DIR: Directory containing audio files to transcribe
    """
    try:
        from transcription.utils import transcribe_batch_files, save_transcript
        
        logger.info(f"Starting batch transcription of directory: {audio_dir}")
        click.echo(f"📁 Processing audio files in: {audio_dir}")
        
        # Transcribe all files
        transcripts = transcribe_batch_files(audio_dir, model_size=model_size)
        
        if not transcripts:
            click.echo("❌ No audio files found or transcription failed")
            sys.exit(1)
        
        # Save all transcripts
        saved_files = []
        for transcript in transcripts:
            saved_file = save_transcript(transcript, output_dir)
            saved_files.append(saved_file)
        
        click.echo(f"✅ Batch transcription completed!")
        click.echo(f"📄 Processed {len(transcripts)} files")
        click.echo(f"📁 Transcripts saved to: {output_dir}")
        
        for i, saved_file in enumerate(saved_files, 1):
            click.echo(f"  {i}. {Path(saved_file).name}")
            
    except ImportError:
        click.echo("Error: Transcription module not available.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('transcript_file', type=click.Path(exists=True))
@click.option('--output', type=click.Path(), help='Output notes file path')
@click.option('--format', type=click.Choice(['markdown', 'json']), default='markdown',
              help='Output format')
def summarize(transcript_file: str, output: Optional[str], format: str):
    """
    Generate structured notes from transcript using LLaMA 2.
    
    TRANSCRIPT_FILE: Path to the transcript file
    """
    try:
        from summarization.llama_local import LLaMASummarizer
        
        summarizer = LLaMASummarizer(config.get_summarization_config())
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        notes = summarizer.generate_notes(transcript)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                if format == 'markdown':
                    f.write(notes.to_markdown())
                else:
                    f.write(notes.to_json())
            click.echo(f"Notes saved to: {output}")
        else:
            if format == 'markdown':
                click.echo(notes.to_markdown())
            else:
                click.echo(notes.to_json())
                
    except ImportError:
        click.echo("Error: Summarization module not implemented yet.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('notes_file', type=click.Path(exists=True))
@click.option('--google-docs', is_flag=True, help='Export to Google Docs')
@click.option('--notion', is_flag=True, help='Export to Notion')
@click.option('--title', help='Document title')
def export(notes_file: str, google_docs: bool, notion: bool, title: Optional[str]):
    """
    Export notes to Google Docs and/or Notion.
    
    NOTES_FILE: Path to the notes file to export
    """
    try:
        from export.google_docs import GoogleDocsExporter
        from export.notion import NotionExporter
        
        with open(notes_file, 'r', encoding='utf-8') as f:
            notes_content = f.read()
        
        if google_docs:
            exporter = GoogleDocsExporter(config.get_export_config()['google_docs'])
            doc_url = exporter.export_notes(notes_content, title)
            click.echo(f"Exported to Google Docs: {doc_url}")
        
        if notion:
            exporter = NotionExporter(config.get_export_config()['notion'])
            page_url = exporter.export_notes(notes_content, title)
            click.echo(f"Exported to Notion: {page_url}")
            
    except ImportError:
        click.echo("Error: Export modules not implemented yet.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--duration', type=int, default=300, help='Recording duration in seconds')
@click.option('--youtube-url', help='YouTube URL for metadata')
@click.option('--title', help='Document title')
@click.option('--tags', help='Comma-separated tags')
@click.option('--export', is_flag=True, help='Export notes after generation')
def pipeline(duration: int, youtube_url: Optional[str], title: Optional[str], 
            tags: Optional[str], export: bool):
    """
    Run the complete pipeline: record -> transcribe -> summarize -> export.
    
    This is the main command that runs the entire workflow.
    """
    try:
        click.echo("Starting AI Notes Assistant pipeline...")
        
        # Step 1: Record audio
        click.echo("1. Recording audio...")
        from audio.capture import AudioCapture
        capture = AudioCapture(config.get_audio_config())
        audio_file = capture.record_to_file(duration=duration)
        click.echo(f"   Audio recorded: {audio_file}")
        
        # Step 2: Transcribe
        click.echo("2. Transcribing audio...")
        from transcription.whisper_local import WhisperTranscriber
        transcriber = WhisperTranscriber(config.get_transcription_config())
        transcript = transcriber.transcribe_file(audio_file)
        click.echo("   Transcription completed")
        
        # Step 3: Generate notes
        click.echo("3. Generating structured notes...")
        from summarization.llama_local import LLaMASummarizer
        summarizer = LLaMASummarizer(config.get_summarization_config())
        notes = summarizer.generate_notes(transcript, title=title, tags=tags, youtube_url=youtube_url)
        click.echo("   Notes generated")
        
        # Step 4: Export (optional)
        if export:
            click.echo("4. Exporting notes...")
            from export.google_docs import GoogleDocsExporter
            from export.notion import NotionExporter
            
            export_config = config.get_export_config()
            
            if export_config['google_docs']['enabled']:
                exporter = GoogleDocsExporter(export_config['google_docs'])
                doc_url = exporter.export_notes(notes.to_markdown(), title)
                click.echo(f"   Exported to Google Docs: {doc_url}")
            
            if export_config['notion']['enabled']:
                exporter = NotionExporter(export_config['notion'])
                page_url = exporter.export_notes(notes.to_markdown(), title)
                click.echo(f"   Exported to Notion: {page_url}")
        
        click.echo("Pipeline completed successfully!")
        
    except ImportError as e:
        click.echo(f"Error: Module not implemented yet: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show system status and configuration."""
    click.echo("AI Notes Assistant Status")
    click.echo("=" * 30)
    
    # Configuration status
    click.echo(f"Config file: {config.config_path}")
    click.echo(f"Debug mode: {config.is_debug_mode()}")
    click.echo(f"Test mode: {config.is_test_mode()}")
    
    # Audio configuration
    audio_config = config.get_audio_config()
    click.echo(f"Audio device: {audio_config.get('blackhole_device', 'Not configured')}")
    click.echo(f"Sample rate: {audio_config.get('sample_rate', 'Not configured')}")
    
    # Model paths
    llama_path = config.get_env('LLAMA_MODEL_PATH')
    whisper_path = config.get_env('WHISPER_MODEL_PATH')
    click.echo(f"LLaMA model: {llama_path}")
    click.echo(f"Whisper model: {whisper_path}")
    
    # API configuration
    google_enabled = bool(config.get_env('GOOGLE_CLIENT_ID'))
    notion_enabled = bool(config.get_env('NOTION_TOKEN'))
    click.echo(f"Google Docs API: {'Configured' if google_enabled else 'Not configured'}")
    click.echo(f"Notion API: {'Configured' if notion_enabled else 'Not configured'}")


@cli.command()
def test():
    """Run system tests."""
    click.echo("Running system tests...")
    
    # Test configuration
    try:
        config.reload()
        click.echo("✓ Configuration loaded successfully")
    except Exception as e:
        click.echo(f"✗ Configuration error: {e}")
        return
    
    # Test audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        blackhole_found = any('blackhole' in str(device).lower() for device in devices)
        if blackhole_found:
            click.echo("✓ BlackHole audio device found")
        else:
            click.echo("✗ BlackHole audio device not found")
    except ImportError:
        click.echo("✗ sounddevice not installed")
    except Exception as e:
        click.echo(f"✗ Audio device test failed: {e}")
    
    # Test model files
    llama_path = config.get_env('LLAMA_MODEL_PATH')
    if Path(llama_path).exists():
        click.echo("✓ LLaMA model file found")
    else:
        click.echo("✗ LLaMA model file not found")
    
    whisper_path = config.get_env('WHISPER_MODEL_PATH')
    if Path(whisper_path).exists():
        click.echo("✓ Whisper model file found")
    else:
        click.echo("✗ Whisper model file not found")
    
    click.echo("System tests completed")


@cli.command()
@click.option('--file', '-f', required=True, help='Path to enhanced transcript JSON file')
def show_topics(file):
    """Show extracted topics from processed transcript."""
    import json
    from pathlib import Path
    
    try:
        file_path = Path(file)
        if not file_path.exists():
            click.echo(f"❌ File not found: {file}")
            return
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        enhanced_notes = data.get('enhanced_notes', {})
        metadata = enhanced_notes.get('metadata', {})
        topics = metadata.get('topics', [])
        topic_summaries = metadata.get('topic_summaries', {})
        
        if not topics:
            click.echo("No topics found in this file.")
            return
            
        click.echo(f"📚 Extracted Topics ({len(topics)}):")
        click.echo("=" * 40)
        
        for i, topic in enumerate(topics, 1):
            click.echo(f"{i}. {topic}")
            if topic in topic_summaries:
                summary = topic_summaries[topic][:100] + "..." if len(topic_summaries[topic]) > 100 else topic_summaries[topic]
                click.echo(f"   💡 {summary}")
            click.echo("")
            
    except Exception as e:
        click.echo(f"❌ Error reading file: {e}")


@cli.command()
@click.option('--transcript', '-t', required=True, help='Path to transcript JSON file')
@click.option('--output', '-o', help='Output directory for enhanced notes')
@click.option('--model', type=click.Choice(['deepseek', 'gpt-4o-mini', 'mixtral'], case_sensitive=False),
              default=None, help='AI model for summarization (overrides config)')
@click.option('--expertise', type=click.Choice(['beginner', 'moderate', 'expert'], case_sensitive=False),
              default='moderate', help='Expertise level for summary generation')
@click.option('--model-size', default='8x7b', help='Model size (legacy parameter)')
def process(transcript, output, model, expertise, model_size):
    """Process a transcript and generate enhanced AI scholar notes."""
    from summarization.utils import process_transcript, validate_transcript_file
    
    if not validate_transcript_file(transcript):
        click.echo(f"❌ Invalid transcript file: {transcript}")
        return
    
    click.echo(f"🧠 Processing transcript: {transcript}")
    
    # Set output directory
    if not output:
        config = ConfigManager()
        output = config.get_storage_config().get('notes_path', 'data/notes')
    
    try:
        enhanced_transcript = process_transcript(transcript, output, expertise, model)
        
        if 'error' in enhanced_transcript:
            click.echo(f"❌ Processing failed: {enhanced_transcript['error']}")
            return
        
        click.echo("✅ Transcript processing completed successfully!")
        click.echo(f"📁 Enhanced notes saved to: {output}")
        
        # Show summary
        enhanced_notes = enhanced_transcript.get('enhanced_notes', {})
        if enhanced_notes:
            title = enhanced_notes.get('title', 'AI/ML Learning Notes')
            topics = enhanced_notes.get('metadata', {}).get('topics', [])
            click.echo(f"📚 Title: {title}")
            click.echo(f"🏷️ Topics: {', '.join(topics[:5])}")
            
    except Exception as e:
        click.echo(f"❌ Processing error: {e}")


@cli.command()
@click.option('--transcript-dir', '-d', required=True, help='Directory containing transcript files')
@click.option('--output', '-o', help='Output directory for enhanced notes')
@click.option('--model-size', default='8x7b', help='Mixtral model size (8x7b)')
def process_batch(transcript_dir, output, model_size):
    """Process all transcript files in a directory."""
    from summarization.utils import process_batch_transcripts, validate_transcript_file
    
    if not Path(transcript_dir).exists():
        click.echo(f"❌ Directory not found: {transcript_dir}")
        return
    
    click.echo(f"🧠 Processing transcripts in: {transcript_dir}")
    
    # Set output directory
    if not output:
        config = ConfigManager()
        output = config.get_storage_config().get('notes_path', 'data/notes')
    
    try:
        enhanced_transcripts = process_batch_transcripts(transcript_dir, output)
        
        if not enhanced_transcripts:
            click.echo("❌ No transcripts were processed successfully")
            return
        
        click.echo(f"✅ Batch processing completed: {len(enhanced_transcripts)} files processed")
        click.echo(f"📁 Enhanced notes saved to: {output}")
        
        # Show summary
        for transcript in enhanced_transcripts[:3]:  # Show first 3
            enhanced_notes = transcript.get('enhanced_notes', {})
            if enhanced_notes:
                title = enhanced_notes.get('title', 'AI/ML Learning Notes')
                click.echo(f"📚 {title}")
        
        if len(enhanced_transcripts) > 3:
            click.echo(f"... and {len(enhanced_transcripts) - 3} more")
            
    except Exception as e:
        click.echo(f"❌ Batch processing error: {e}")


@cli.command()
def process_status():
    """Check the status of the processing module."""
    from summarization.utils import get_processing_status
    
    status = get_processing_status()
    
    click.echo("🧠 Processing Module Status:")
    click.echo(f"   Model Loaded: {'✅' if status.get('model_loaded') else '❌'}")
    click.echo(f"   Model Path: {status.get('model_path', 'Not set')}")
    click.echo(f"   Status: {status.get('status', 'Unknown')}")
    
    if 'error' in status:
        click.echo(f"   Error: {status['error']}")


@cli.command()
def smart_process():
    """Interactive processing - choose between YouTube or live audio."""
    from cli.video_selector import VideoSelector
    
    selector = VideoSelector()
    choice = selector.show_processing_choice()
    
    if choice == 'youtube':
        youtube_workflow()
    else:
        live_workflow()

def youtube_workflow():
    """Complete YouTube processing workflow."""
    from cli.video_selector import VideoSelector
    from audio.youtube_downloader import YouTubeDownloader
    
    downloader = YouTubeDownloader()
    selector = VideoSelector()
    
    # Option to download new videos
    if click.confirm("📥 Download new YouTube videos?"):
        while True:
            url = click.prompt("🔗 Enter YouTube URL (or 'done' to finish)", type=str)
            if url.lower() == 'done':
                break
            
            try:
                base_name, metadata = downloader.download_audio(url)
                click.echo(f"✅ Downloaded: {metadata['title']}")
            except Exception as e:
                click.echo(f"❌ Download failed: {e}")
    
    # Show and select videos for processing
    videos = downloader.list_downloaded_videos()
    selected_videos = selector.select_videos_for_processing(videos)
    
    if not selected_videos:
        click.echo("No videos selected for processing.")
        return
    
    # Process each selected video sequentially
    process_videos_sequentially(selected_videos)

def process_videos_sequentially(videos: List[Dict]):
    """Process selected videos one by one through the full pipeline."""
    from summarization.utils import process_transcript
    
    total = len(videos)
    
    for i, video in enumerate(videos, 1):
        base_name = video['base_name']
        click.echo(f"\n🔄 Processing video {i}/{total}: {video['title']}")
        
        try:
            # Step 1: Transcribe audio
            click.echo("📝 Transcribing audio...")
            audio_file = video['audio_file']
            
            transcript_path = f"data/transcripts/{base_name}_transcribed.json"
            
            # Use chunked processor for long videos
            from transcription.chunked_processor import ChunkedAudioProcessor
            
            # Check audio duration to decide processing method
            import soundfile as sf
            audio_info = sf.info(audio_file)
            duration_minutes = audio_info.duration / 60
            
            if duration_minutes > 30:  # Use chunked processing for videos longer than 30 minutes
                click.echo(f"📊 Long video detected ({duration_minutes:.1f} minutes), using intelligent chunking...")
                processor = ChunkedAudioProcessor(
                    chunk_duration_minutes=60,
                    overlap_seconds=30,
                    model_size="small"  # Use smaller model for efficiency
                )
                transcript = processor.process_long_audio(audio_file)
            else:
                click.echo(f"📊 Short video ({duration_minutes:.1f} minutes), using direct processing...")
                from transcription.utils import transcribe_audio_file
                transcript = transcribe_audio_file(audio_file)
            
            # Save with custom filename
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2)
            
            click.echo(f"✅ Transcript saved: {transcript_path}")
            
            # Step 2: Generate AI summary
            click.echo("🧠 Generating AI summary...")
            enhanced_transcript = process_transcript(
                transcript_path,
                output_dir="data/notes"
            )
            
            # Step 3: Save topics separately
            click.echo("🏷️ Extracting topics...")
            save_topics_file(base_name, enhanced_transcript)
            
            # Step 4: Rename summary file to match naming convention
            rename_summary_file(base_name)
            
            click.echo(f"✅ Completed: {video['title']}")
            
        except Exception as e:
            click.echo(f"❌ Failed to process {video['title']}: {e}")
            continue
    
    click.echo(f"\n🎉 Processing complete! Processed {total} videos.")

def save_topics_file(base_name: str, enhanced_transcript: Dict):
    """Save extracted topics to separate file."""
    topics_path = f"data/transcripts/{base_name}_topics_extracted.json"
    
    enhanced_notes = enhanced_transcript.get('enhanced_notes', {})
    metadata = enhanced_notes.get('metadata', {})
    topics = metadata.get('topics', [])
    topic_summaries = metadata.get('topic_summaries', {})
    
    topics_data = {
        'base_name': base_name,
        'extraction_timestamp': datetime.now().isoformat(),
        'total_topics': len(topics),
        'topics': topics,
        'topic_summaries': topic_summaries,
        'title': enhanced_notes.get('title', 'Unknown')
    }
    
    with open(topics_path, 'w') as f:
        json.dump(topics_data, f, indent=2)

def rename_summary_file(base_name: str):
    """Rename the generated summary file to match naming convention."""
    # Find the most recent notes file
    notes_pattern = "data/notes/notes_*"
    notes_files = glob.glob(notes_pattern)
    
    if notes_files:
        # Get the most recent file
        latest_file = max(notes_files, key=lambda x: Path(x).stat().st_mtime)
        
        # Rename to our convention
        new_name = f"data/notes/{base_name}_summary.txt"
        Path(latest_file).rename(new_name)

def live_workflow():
    """Live audio processing workflow."""
    click.echo("🎤 Live Audio Processing")
    duration = click.prompt("⏱️ Recording duration (seconds)", type=int, default=300)
    
    click.echo(f"🔴 Starting {duration} second recording...")
    click.echo("📢 Make sure audio is playing and will be captured by BlackHole")
    
    # Call existing record command
    from audio.capture import AudioCapture
    
    config = ConfigManager()
    capture = AudioCapture(config.get_audio_config())
    audio_file = capture.record(duration)
    
    click.echo(f"✅ Recording saved: {audio_file}")
    
    # Ask if user wants to transcribe immediately
    if click.confirm("📝 Transcribe the recording now?"):
        click.echo("Processing with existing live audio workflow...")

@cli.command()
@click.option('--url', required=True, help='YouTube video URL')
def download_youtube(url):
    """Download YouTube video audio for later processing."""
    from audio.youtube_downloader import YouTubeDownloader
    
    downloader = YouTubeDownloader()
    
    try:
        base_name, metadata = downloader.download_audio(url)
        click.echo(f"✅ Downloaded: {metadata['title']}")
        click.echo(f"📁 Saved as: {base_name}")
        click.echo(f"📍 Location: data/downloads/{base_name}.*")
    except Exception as e:
        click.echo(f"❌ Download failed: {e}")

@cli.command()
def list_videos():
    """List all downloaded YouTube videos."""
    from cli.video_selector import VideoSelector
    
    selector = VideoSelector()
    videos = selector.downloader.list_downloaded_videos()
    
    selector.show_downloaded_videos(videos)
    
    if videos:
        click.echo(f"\n📊 Total: {len(videos)} downloaded videos")
        total_size = sum(v.get('file_size_mb', 0) for v in videos)
        click.echo(f"💾 Total size: {total_size:.1f} MB")

@cli.command() 
def youtube_status():
    """Show YouTube processing status and folder information."""
    from audio.youtube_downloader import YouTubeDownloader
    
    downloader = YouTubeDownloader()
    
    click.echo("📊 YouTube Processing Status")
    click.echo("=" * 40)
    
    # Check directories
    for name, path in [
        ("Downloads", downloader.downloads_dir),
        ("Transcripts", downloader.transcripts_dir), 
        ("Notes", downloader.notes_dir)
    ]:
        exists = "✅" if path.exists() else "❌"
        click.echo(f"{exists} {name}: {path}")
    
    # Count files
    videos = downloader.list_downloaded_videos()
    transcripts = len(list(downloader.transcripts_dir.glob("*_transcribed.json")))
    summaries = len(list(downloader.notes_dir.glob("*_summary.txt")))
    topics = len(list(downloader.transcripts_dir.glob("*_topics_extracted.json")))
    
    click.echo("\n📈 File Counts:")
    click.echo(f"   📺 Downloaded videos: {len(videos)}")
    click.echo(f"   📝 Transcripts: {transcripts}")
    click.echo(f"   📄 Summaries: {summaries}")
    click.echo(f"   🏷️  Topics files: {topics}")


if __name__ == '__main__':
    cli()
