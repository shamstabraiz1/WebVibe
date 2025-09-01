"""Configuration management for Urdu Sentiment Analysis project."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Main configuration class for the project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CACHE_DIR = PROJECT_ROOT / "cache"
    
    # Model configuration
    MODEL_NAME = "bert-base-multilingual-cased"
    ALTERNATIVE_MODEL = "xlm-roberta-base"
    MAX_LENGTH = 512
    NUM_CLASSES = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_BATCH_SIZE = 1000
    REQUEST_TIMEOUT = 30
    
    # Training configuration
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
    
    # GPU configuration
    USE_GPU = True
    GPU_MEMORY_FRACTION = 0.8
    
    # Preprocessing configuration
    CLEAN_TEXT = True
    NORMALIZE_SCRIPT = True
    HANDLE_DIALECTS = True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device for model training/inference."""
        import torch
        if cls.USE_GPU and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "model_name": cls.MODEL_NAME,
            "max_length": cls.MAX_LENGTH,
            "num_classes": cls.NUM_CLASSES,
            "batch_size": cls.BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE,
            "num_epochs": cls.NUM_EPOCHS,
            "warmup_steps": cls.WARMUP_STEPS,
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration dictionary."""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "max_batch_size": cls.MAX_BATCH_SIZE,
            "timeout": cls.REQUEST_TIMEOUT,
        }

# Create directories on import
Config.create_directories()