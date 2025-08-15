"""
Configuration management module for AI Notes Assistant.
Handles loading settings from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manages application configuration from YAML and environment variables."""
    
    def __init__(self, config_path: str = "config/settings.yaml", env_path: str = "config/.env"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to environment variables file
        """
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._config: Optional[Dict[str, Any]] = None
        self._env_vars: Optional[Dict[str, str]] = None
        
        # Load configuration
        self._load_config()
        self._load_env()
    
    def _load_config(self) -> None:
        """Load YAML configuration file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                print(f"Warning: Config file {self.config_path} not found. Using defaults.")
                self._config = self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self._config = self._get_default_config()
    
    def _load_env(self) -> None:
        """Load environment variables."""
        try:
            if self.env_path.exists():
                load_dotenv(self.env_path)
            
            # Load environment variables into dict
            self._env_vars = {
                'GOOGLE_CLIENT_ID': os.getenv('GOOGLE_CLIENT_ID', ''),
                'GOOGLE_CLIENT_SECRET': os.getenv('GOOGLE_CLIENT_SECRET', ''),
                'GOOGLE_REDIRECT_URI': os.getenv('GOOGLE_REDIRECT_URI', ''),
                'NOTION_TOKEN': os.getenv('NOTION_TOKEN', ''),
                'NOTION_DATABASE_ID': os.getenv('NOTION_DATABASE_ID', ''),
                'LLAMA_MODEL_PATH': os.getenv('LLAMA_MODEL_PATH', 'models/llama-2-7b.gguf'),
                'WHISPER_MODEL_PATH': os.getenv('WHISPER_MODEL_PATH', 'models/whisper-base.gguf'),
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
                'DEBUG_MODE': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
                'TEST_MODE': os.getenv('TEST_MODE', 'false').lower() == 'true',
                'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
                'AUDIO_DEVICE': os.getenv('AUDIO_DEVICE', 'BlackHole 2ch'),
                'SAMPLE_RATE': int(os.getenv('SAMPLE_RATE', '16000')),
                'DATA_PATH': os.getenv('DATA_PATH', 'data'),
                'BACKUP_ENABLED': os.getenv('BACKUP_ENABLED', 'true').lower() == 'true',
            }
        except Exception as e:
            print(f"Error loading environment variables: {e}")
            self._env_vars = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not found."""
        return {
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024,
                'format': 'wav',
                'blackhole_device': 'BlackHole 2ch',
                'recording_duration': 300
            },
            'transcription': {
                'model': 'whisper.cpp',
                'model_size': 'base',
                'language': 'en',
                'temperature': 0.0,
                'best_of': 5,
                'beam_size': 5,
                'chunk_length': 30,
                'overlap': 5
            },
            'summarization': {
                'model': 'deepseek',
                'model_path': 'models/llama-2-7b.gguf',
                'max_tokens': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'context_length': 4096,
                'section_break_threshold': 300
            },
            'export': {
                'google_docs': {
                    'enabled': True,
                    'template_id': '',
                    'folder_id': ''
                },
                'notion': {
                    'enabled': True,
                    'database_id': '',
                    'page_id': ''
                }
            },
            'storage': {
                'base_path': 'data',
                'transcripts_path': 'data/transcripts',
                'notes_path': 'data/notes',
                'metadata_path': 'data/metadata',
                'backup_enabled': True,
                'max_file_age_days': 30
            },
            'notes': {
                'sections': [
                    'Key Points',
                    'Tools Mentioned',
                    'Concepts Explained',
                    'Q&A Section',
                    'Additional Resources'
                ],
                'metadata_fields': [
                    'title',
                    'tags',
                    'topics',
                    'youtube_url',
                    'timestamp',
                    'duration',
                    'model_used'
                ]
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/app.log',
                'max_size_mb': 10,
                'backup_count': 5
            },
            'development': {
                'debug_mode': False,
                'test_mode': False,
                'mock_apis': False
            },
            'custom_model': {
                'name': 'Deepseek-coder',
                'api': {
                    'type': 'openai-compatible',
                    'url': 'http://localhost:11434/v1'
                },
                'chat': {
                    'model': 'deepseek-coder:latest'
                }
            },
            'openai': {
                'api_key': 'sk-placeholder-key-loaded-from-env',
                'api_url': 'https://api.openai.com/v1',
                'model': 'gpt-4o-mini',
                'timeout': 120
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_env(self, key: str, default: str = '') -> str:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return self._env_vars.get(key, default) if self._env_vars else default
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration."""
        return self._config.get('audio', {})
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get transcription configuration."""
        return self._config.get('transcription', {})
    
    def get_summarization_config(self) -> Dict[str, Any]:
        """Get summarization configuration."""
        return self._config.get('summarization', {})
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration."""
        return self._config.get('export', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self._config.get('storage', {})
    
    def get_notes_config(self) -> Dict[str, Any]:
        """Get notes structure configuration."""
        return self._config.get('notes', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    def get_custom_model_config(self) -> Dict[str, Any]:
        """Get custom model configuration."""
        return self._config.get('custom_model', {})
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return self._config.get('openai', {})
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self._config.get('development', {}).get('debug_mode', False)
    
    def is_test_mode(self) -> bool:
        """Check if test mode is enabled."""
        return self._config.get('development', {}).get('test_mode', False)
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()
        self._load_env()


# Global configuration instance
config = ConfigManager()
