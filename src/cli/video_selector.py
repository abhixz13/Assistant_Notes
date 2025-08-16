"""
Interactive Video Selection Interface

Provides user-friendly selection of downloaded videos for processing.
"""

import click
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import json

from utils.logger import get_logger
from audio.youtube_downloader import YouTubeDownloader


class VideoSelector:
    """Interactive interface for selecting videos to process."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.downloader = YouTubeDownloader()
    
    def show_processing_choice(self) -> str:
        """
        Ask user to choose between YouTube and live audio processing.
        
        Returns:
            Choice: 'youtube' or 'live'
        """
        click.echo("\n" + "="*50)
        click.echo("🎵 AI Notes Assistant - Processing Options")
        click.echo("="*50)
        click.echo("1. 📺 Process YouTube Videos")
        click.echo("2. 🎤 Process Live System Audio")
        click.echo("="*50)
        
        while True:
            choice = click.prompt("Select processing method (1 or 2)", type=int)
            if choice == 1:
                return 'youtube'
            elif choice == 2:
                return 'live'
            else:
                click.echo("❌ Please enter 1 or 2")
    
    def show_downloaded_videos(self, videos: List[Dict]) -> None:
        """Display list of downloaded videos in a formatted table."""
        if not videos:
            click.echo("📂 No downloaded videos found.")
            return
        
        click.echo("\n" + "="*80)
        click.echo("📺 Downloaded Videos")
        click.echo("="*80)
        
        for i, video in enumerate(videos, 1):
            title = video.get('title', 'Unknown')[:50]
            duration = self._format_duration(video.get('duration'))
            size = video.get('file_size_mb', 0)
            date = video.get('download_timestamp', '')[:10]
            
            click.echo(f"{i:2d}. {title}")
            click.echo(f"    ⏱️  Duration: {duration} | 💾 Size: {size}MB | 📅 Downloaded: {date}")
            click.echo()
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration from seconds to MM:SS format."""
        if not seconds:
            return "Unknown"
        
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def select_videos_for_processing(self, videos: List[Dict]) -> List[Dict]:
        """
        Allow user to select which videos to process.
        
        Args:
            videos: List of available videos
            
        Returns:
            List of selected videos
        """
        if not videos:
            return []
        
        self.show_downloaded_videos(videos)
        
        click.echo("🔢 Enter video numbers to process (e.g., 1,3,5 or 1-3 or 'all'):")
        selection = click.prompt("Your selection", type=str).strip().lower()
        
        if selection == 'all':
            return videos
        
        selected_videos = []
        
        try:
            # Parse selection (handle ranges and individual numbers)
            indices = self._parse_selection(selection, len(videos))
            
            for index in indices:
                if 1 <= index <= len(videos):
                    selected_videos.append(videos[index - 1])
                else:
                    click.echo(f"❌ Invalid index: {index}")
            
            if selected_videos:
                click.echo(f"\n✅ Selected {len(selected_videos)} video(s) for processing:")
                for video in selected_videos:
                    click.echo(f"   📺 {video.get('title', 'Unknown')}")
            
        except ValueError as e:
            click.echo(f"❌ Invalid selection format: {e}")
            return []
        
        return selected_videos
    
    def _parse_selection(self, selection: str, max_videos: int) -> List[int]:
        """Parse user selection string into list of indices."""
        indices = []
        
        for part in selection.split(','):
            part = part.strip()
            
            if '-' in part:
                # Handle ranges (e.g., "1-3")
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                # Handle individual numbers
                indices.append(int(part))
        
        return list(set(indices))  # Remove duplicates
