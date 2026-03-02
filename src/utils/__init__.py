"""Utility functions for whispered speech recognition."""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
import warnings

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_audio(
    path: str,
    sample_rate: int = 16000,
    normalize: bool = True,
    mono: bool = True
) -> torch.Tensor:
    """Load audio file and return waveform tensor.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        mono: Whether to convert to mono
        
    Returns:
        Audio waveform tensor of shape (channels, samples)
    """
    waveform, orig_sr = torchaudio.load(path)
    
    # Convert to mono if requested
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Normalize if requested
    if normalize:
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    
    return waveform


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 16000
) -> None:
    """Save waveform tensor to audio file.
    
    Args:
        waveform: Audio waveform tensor
        path: Output file path
        sample_rate: Sample rate for saving
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, waveform, sample_rate)


def anonymize_text(text: str) -> str:
    """Anonymize text by removing potential PII.
    
    Args:
        text: Input text to anonymize
        
    Returns:
        Anonymized text
    """
    # Simple anonymization - replace common PII patterns
    import re
    
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Replace phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Replace SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    return text


def create_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        output_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(config, output_path)


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score
            model: Model to potentially restore weights
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
