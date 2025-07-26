"""
Logging utility module for AI Notes Assistant.
Provides consistent logging configuration across all modules.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from .config import config


class LoggerManager:
    """Manages application logging configuration."""
    
    def __init__(self, name: str = "ai_notes_assistant"):
        """
        Initialize logger manager.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with file and console handlers."""
        # Get logging configuration
        log_config = config.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'logs/app.log')
        max_size_mb = log_config.get('max_size_mb', 10)
        backup_count = log_config.get('backup_count', 5)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        try:
            # Create logs directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
        
        # Debug mode: add more detailed logging
        if config.is_debug_mode():
            self.logger.setLevel(logging.DEBUG)
            for handler in self.logger.handlers:
                handler.setLevel(logging.DEBUG)
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """
        Get logger instance for a specific module.
        
        Args:
            module_name: Optional module name to append to logger name
            
        Returns:
            Logger instance
        """
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger
    
    def set_level(self, level: str) -> None:
        """
        Set logging level.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        for handler in self.logger.handlers:
            handler.setLevel(log_level)


# Global logger manager
logger_manager = LoggerManager()


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance for a module.
    
    Args:
        module_name: Optional module name
        
    Returns:
        Logger instance
    """
    return logger_manager.get_logger(module_name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    return wrapper


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
