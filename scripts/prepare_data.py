"""Data preparation script for whispered speech recognition."""

import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Any
import logging
import argparse
from omegaconf import DictConfig, OmegaConf

from src.utils import setup_logging, set_seed, create_output_dir, load_config


def create_synthetic_dataset(
    output_dir: str,
    num_samples: int = 1000,
    sample_rate: int = 16000,
    min_duration: float = 1.0,
    max_duration: float = 10.0
) -> None:
    """Create synthetic whispered speech dataset for demonstration.
    
    Args:
        output_dir: Output directory for dataset
        num_samples: Number of samples to generate
        sample_rate: Sample rate for audio
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    # Create directories
    audio_dir = Path(output_dir) / "wav"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Synthetic transcripts
    transcripts = [
        "hello world this is a test",
        "whispered speech recognition system",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning",
        "speech processing technology research",
        "natural language processing applications",
        "deep learning neural networks",
        "computer vision and audio processing",
        "machine learning algorithms and models",
        "data science and analytics research"
    ]
    
    metadata = []
    
    for i in range(num_samples):
        # Generate random duration
        duration = np.random.uniform(min_duration, max_duration)
        num_samples_audio = int(duration * sample_rate)
        
        # Create synthetic whispered speech
        # Whispered speech characteristics: lower amplitude, different frequency content
        t = torch.linspace(0, duration, num_samples_audio)
        
        # Base frequency (lower for whispered speech)
        base_freq = np.random.uniform(80, 200)
        
        # Generate harmonic content (simplified)
        audio = torch.zeros(num_samples_audio)
        
        # Add fundamental frequency
        audio += 0.2 * torch.sin(2 * torch.pi * base_freq * t)
        
        # Add harmonics (reduced compared to normal speech)
        for harmonic in range(2, 6):
            freq = base_freq * harmonic
            amplitude = 0.1 / harmonic  # Decreasing amplitude
            audio += amplitude * torch.sin(2 * torch.pi * freq * t)
        
        # Add noise (whispered speech has more noise)
        noise_level = np.random.uniform(0.05, 0.15)
        audio += noise_level * torch.randn_like(audio)
        
        # Apply envelope (simulate speech rhythm)
        envelope = torch.exp(-0.5 * ((t - duration/2) / (duration/4))**2)
        audio *= envelope
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Add some silence at the beginning and end
        silence_samples = int(0.1 * sample_rate)  # 100ms silence
        audio_with_silence = torch.cat([
            torch.zeros(silence_samples),
            audio,
            torch.zeros(silence_samples)
        ])
        
        # Save audio file
        audio_path = audio_dir / f"synthetic_{i:04d}.wav"
        torchaudio.save(str(audio_path), audio_with_silence.unsqueeze(0), sample_rate)
        
        # Select transcript
        transcript = transcripts[i % len(transcripts)]
        
        # Add some variation to transcripts
        if np.random.random() < 0.3:  # 30% chance of variation
            words = transcript.split()
            if len(words) > 2:
                # Remove a random word
                word_to_remove = np.random.randint(0, len(words))
                words.pop(word_to_remove)
                transcript = " ".join(words)
        
        # Determine split
        if i < num_samples * 0.7:
            split = "train"
        elif i < num_samples * 0.85:
            split = "validation"
        else:
            split = "test"
        
        # Add to metadata
        metadata.append({
            "id": f"synthetic_{i:04d}",
            "path": str(audio_path),
            "transcript": transcript,
            "duration": duration + 0.2,  # Include silence
            "split": split,
            "speaker_id": f"speaker_{i % 10}",  # 10 different speakers
            "language": "en",
            "accent": "general"
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = Path(output_dir) / "meta.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    logger.info(f"Dataset created successfully:")
    logger.info(f"  - Audio files: {len(metadata)}")
    logger.info(f"  - Train samples: {len(metadata_df[metadata_df['split'] == 'train'])}")
    logger.info(f"  - Validation samples: {len(metadata_df[metadata_df['split'] == 'validation'])}")
    logger.info(f"  - Test samples: {len(metadata_df[metadata_df['split'] == 'test'])}")
    logger.info(f"  - Metadata saved to: {metadata_path}")


def download_real_dataset(dataset_name: str, output_dir: str) -> None:
    """Download real whispered speech dataset.
    
    Args:
        dataset_name: Name of dataset to download
        output_dir: Output directory
    """
    logger = logging.getLogger(__name__)
    
    if dataset_name == "whispered_speech_corpus":
        logger.info("Downloading Whispered Speech Corpus...")
        # This would implement actual dataset download
        # For now, we'll create synthetic data
        logger.warning("Real dataset download not implemented. Creating synthetic data instead.")
        create_synthetic_dataset(output_dir)
    
    elif dataset_name == "common_voice_whispered":
        logger.info("Downloading Common Voice whispered subset...")
        # This would filter Common Voice for whispered speech
        logger.warning("Common Voice whispered subset not available. Creating synthetic data instead.")
        create_synthetic_dataset(output_dir)
    
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_data(config: DictConfig) -> None:
    """Prepare dataset according to configuration.
    
    Args:
        config: Data preparation configuration
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    create_output_dir(config.data_dir)
    
    # Check if dataset already exists
    metadata_path = Path(config.data_dir) / "meta.csv"
    if metadata_path.exists() and not config.force_recreate:
        logger.info(f"Dataset already exists at {config.data_dir}")
        return
    
    # Prepare dataset
    if config.dataset_type == "synthetic":
        create_synthetic_dataset(
            output_dir=config.data_dir,
            num_samples=config.num_samples,
            sample_rate=config.sample_rate,
            min_duration=config.min_duration,
            max_duration=config.max_duration
        )
    
    elif config.dataset_type == "real":
        download_real_dataset(config.dataset_name, config.data_dir)
    
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")
    
    # Validate dataset
    validate_dataset(config.data_dir)


def validate_dataset(data_dir: str) -> None:
    """Validate prepared dataset.
    
    Args:
        data_dir: Dataset directory
    """
    logger = logging.getLogger(__name__)
    
    metadata_path = Path(data_dir) / "meta.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Check required columns
    required_columns = ["id", "path", "transcript", "duration", "split"]
    missing_columns = [col for col in required_columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check audio files exist
    missing_files = []
    for _, row in metadata.iterrows():
        if not Path(row["path"]).exists():
            missing_files.append(row["path"])
    
    if missing_files:
        logger.warning(f"Missing audio files: {len(missing_files)}")
        logger.warning("First 5 missing files:")
        for file in missing_files[:5]:
            logger.warning(f"  - {file}")
    
    # Check splits
    splits = metadata["split"].unique()
    logger.info(f"Dataset splits: {list(splits)}")
    
    for split in splits:
        split_data = metadata[metadata["split"] == split]
        logger.info(f"{split}: {len(split_data)} samples")
    
    # Check duration distribution
    logger.info(f"Duration statistics:")
    logger.info(f"  Min: {metadata['duration'].min():.2f}s")
    logger.info(f"  Max: {metadata['duration'].max():.2f}s")
    logger.info(f"  Mean: {metadata['duration'].mean():.2f}s")
    logger.info(f"  Std: {metadata['duration'].std():.2f}s")
    
    logger.info("Dataset validation completed successfully")


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare whispered speech dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate dataset")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override force recreate
    config.force_recreate = args.force_recreate
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    # Set seed
    set_seed(config.seed)
    
    # Prepare data
    prepare_data(config)


if __name__ == "__main__":
    main()
