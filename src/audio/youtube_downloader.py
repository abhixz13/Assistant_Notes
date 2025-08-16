"""
YouTube Video Downloader Module

Downloads audio from YouTube videos for transcription and AI processing.
Handles file naming, folder organization, and metadata extraction.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yt_dlp

from utils.config import ConfigManager
from utils.logger import get_logger


class YouTubeDownloader:
    """Downloads audio from YouTube videos with organized file management."""
    
    def __init__(self):
        """Initialize YouTube downloader with configuration."""
        self.logger = get_logger(__name__)
        self.config = ConfigManager()
        
        # Create directory structure
        self.downloads_dir = Path("data/downloads")
        self.transcripts_dir = Path("data/transcripts") 
        self.notes_dir = Path("data/notes")
        
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.downloads_dir, self.transcripts_dir, self.notes_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory ready: {directory}")
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize video title for filename use."""
        # Remove special characters and limit length
        sanitized = re.sub(r'[^\w\s-]', '', title)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized[:50]  # Limit to 50 characters
    
    def _generate_base_name(self, video_info: Dict) -> str:
        """Generate base filename from video metadata."""
        title = self._sanitize_filename(video_info.get('title', 'Unknown'))
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{title}_{date_str}"
    
    def download_audio(self, url: str) -> Tuple[str, Dict]:
        """
        Download audio from YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (base_filename, video_info)
        """
        try:
            self.logger.info(f"Starting download from: {url}")
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.downloads_dir / '%(title)s_temp.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'no_warnings': False,
            }
            
            # Download video info and audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                video_info = ydl.extract_info(url, download=False)
                self.logger.info(f"Video found: {video_info.get('title')}")
                
                # Generate consistent filename
                base_name = self._generate_base_name(video_info)
                
                # Update output template with our naming convention
                ydl_opts['outtmpl'] = str(self.downloads_dir / f"{base_name}.%(ext)s")
                
                # Download audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])
                
                # Find the downloaded file
                downloaded_files = list(self.downloads_dir.glob(f"{base_name}.*"))
                audio_files = [f for f in downloaded_files if not f.name.endswith('_metadata.json')]
                
                if audio_files:
                    audio_file = audio_files[0]
                    self.logger.info(f"Audio downloaded: {audio_file}")
                    
                    # Save metadata
                    metadata = {
                        'url': url,
                        'title': video_info.get('title'),
                        'duration': video_info.get('duration'),
                        'upload_date': video_info.get('upload_date'),
                        'uploader': video_info.get('uploader'),
                        'base_name': base_name,
                        'download_timestamp': datetime.now().isoformat()
                    }
                    
                    metadata_file = self.downloads_dir / f"{base_name}_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return base_name, metadata
                else:
                    raise Exception("No audio file found after download")
                    
        except Exception as e:
            self.logger.error(f"Failed to download audio from {url}: {e}")
            raise
    
    def list_downloaded_videos(self) -> List[Dict]:
        """
        List all downloaded videos with metadata.
        
        Returns:
            List of video metadata dictionaries
        """
        videos = []
        
        for metadata_file in self.downloads_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Check if audio file still exists
                base_name = metadata['base_name']
                audio_files = list(self.downloads_dir.glob(f"{base_name}.*"))
                audio_files = [f for f in audio_files if not f.name.endswith('_metadata.json')]
                
                if audio_files:
                    metadata['audio_file'] = str(audio_files[0])
                    metadata['file_size_mb'] = round(audio_files[0].stat().st_size / (1024*1024), 2)
                    videos.append(metadata)
                    
            except Exception as e:
                self.logger.error(f"Error reading metadata from {metadata_file}: {e}")
        
        # Sort by download timestamp (newest first)
        videos.sort(key=lambda x: x.get('download_timestamp', ''), reverse=True)
        return videos
    
    def get_file_paths(self, base_name: str) -> Dict[str, str]:
        """
        Get all file paths for a given base name.
        
        Args:
            base_name: Base filename without extension
            
        Returns:
            Dictionary with file paths for each type
        """
        return {
            'audio': str(self.downloads_dir / f"{base_name}.wav"),
            'transcript': str(self.transcripts_dir / f"{base_name}_transcribed.json"),
            'transcript_md': str(self.transcripts_dir / f"{base_name}_transcribed.md"),
            'summary': str(self.notes_dir / f"{base_name}_summary.txt"),
            'topics': str(self.transcripts_dir / f"{base_name}_topics_extracted.json"),
            'metadata': str(self.downloads_dir / f"{base_name}_metadata.json")
        }
