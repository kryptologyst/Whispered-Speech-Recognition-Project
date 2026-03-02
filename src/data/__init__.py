"""Data loading and preprocessing for whispered speech recognition."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from ..utils import load_audio, anonymize_text
from ..features import LogMelSpectrogram, SpecAugment, CMVN


class WhisperedSpeechDataset(Dataset):
    """Dataset for whispered speech recognition."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        normalize_audio: bool = True,
        feature_type: str = "log_mel",
        use_augmentation: bool = False,
        anonymize: bool = True
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing data
            split: Data split (train, validation, test)
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            normalize_audio: Whether to normalize audio
            feature_type: Type of features to extract
            use_augmentation: Whether to use data augmentation
            anonymize: Whether to anonymize text
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_audio = normalize_audio
        self.feature_type = feature_type
        self.use_augmentation = use_augmentation
        self.anonymize = anonymize
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Initialize feature extractor
        self.feature_extractor = self._init_feature_extractor()
        
        # Initialize augmentation
        if use_augmentation and split == "train":
            self.augmentation = SpecAugment()
        else:
            self.augmentation = None
        
        # Initialize CMVN
        self.cmvn = CMVN()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata.
        
        Returns:
            DataFrame with metadata
        """
        metadata_path = self.data_dir / "meta.csv"
        
        if not metadata_path.exists():
            # Create synthetic metadata for demo purposes
            self.logger.warning(f"Metadata file not found at {metadata_path}. Creating synthetic data.")
            return self._create_synthetic_metadata()
        
        metadata = pd.read_csv(metadata_path)
        
        # Filter by split
        if "split" in metadata.columns:
            metadata = metadata[metadata["split"] == self.split]
        
        # Filter by duration
        if "duration" in metadata.columns:
            metadata = metadata[
                (metadata["duration"] >= self.min_duration) &
                (metadata["duration"] <= self.max_duration)
            ]
        
        return metadata.reset_index(drop=True)
    
    def _create_synthetic_metadata(self) -> pd.DataFrame:
        """Create synthetic metadata for demo purposes.
        
        Returns:
            Synthetic metadata DataFrame
        """
        # Create synthetic audio files and metadata
        audio_dir = self.data_dir / "wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        synthetic_data = []
        transcripts = [
            "hello world this is a test",
            "whispered speech recognition",
            "the quick brown fox jumps",
            "artificial intelligence research",
            "speech processing technology"
        ]
        
        for i in range(50):  # Create 50 synthetic samples
            # Generate synthetic audio (sine wave with noise)
            duration = torch.randint(
                int(self.min_duration * self.sample_rate),
                int(self.max_duration * self.sample_rate),
                (1,)
            ).item()
            
            # Create synthetic whispered speech (lower amplitude, different frequency content)
            t = torch.linspace(0, duration / self.sample_rate, duration)
            freq = torch.randint(100, 500, (1,)).item()  # Lower frequency for whispered speech
            audio = 0.3 * torch.sin(2 * torch.pi * freq * t)  # Lower amplitude
            audio += 0.1 * torch.randn_like(audio)  # Add noise
            
            # Save audio file
            audio_path = audio_dir / f"synthetic_{self.split}_{i:03d}.wav"
            torchaudio.save(str(audio_path), audio.unsqueeze(0), self.sample_rate)
            
            # Add to metadata
            synthetic_data.append({
                "id": f"synthetic_{i:03d}",
                "path": str(audio_path),
                "transcript": transcripts[i % len(transcripts)],
                "duration": duration / self.sample_rate,
                "split": self.split,
                "speaker_id": f"speaker_{i % 5}"
            })
        
        return pd.DataFrame(synthetic_data)
    
    def _init_feature_extractor(self):
        """Initialize feature extractor.
        
        Returns:
            Feature extractor instance
        """
        if self.feature_type == "log_mel":
            return LogMelSpectrogram(sample_rate=self.sample_rate)
        elif self.feature_type == "mfcc":
            from ..features import MFCC
            return MFCC(sample_rate=self.sample_rate)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples in dataset
        """
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing features and labels
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = row["path"]
        waveform = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            normalize=self.normalize_audio
        )
        
        # Extract features
        features = self.feature_extractor(waveform)
        
        # Apply augmentation if training
        if self.augmentation is not None:
            features = self.augmentation(features)
        
        # Get transcript
        transcript = row["transcript"]
        if self.anonymize:
            transcript = anonymize_text(transcript)
        
        return {
            "features": features,
            "transcript": transcript,
            "audio_path": audio_path,
            "duration": row.get("duration", 0.0),
            "speaker_id": row.get("speaker_id", "unknown")
        }


class WhisperedSpeechDataModule:
    """Data module for whispered speech recognition."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **dataset_kwargs
    ):
        """Initialize data module.
        
        Args:
            data_dir: Directory containing data
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            persistent_workers: Whether to use persistent workers
            **dataset_kwargs: Additional arguments for dataset
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_kwargs = dataset_kwargs
        
        self.logger = logging.getLogger(__name__)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/validation/testing.
        
        Args:
            stage: Current stage (fit, validate, test, or None)
        """
        if stage == "fit" or stage is None:
            self.train_dataset = WhisperedSpeechDataset(
                data_dir=self.data_dir,
                split="train",
                use_augmentation=True,
                **self.dataset_kwargs
            )
            
            self.val_dataset = WhisperedSpeechDataset(
                data_dir=self.data_dir,
                split="validation",
                use_augmentation=False,
                **self.dataset_kwargs
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = WhisperedSpeechDataset(
                data_dir=self.data_dir,
                split="test",
                use_augmentation=False,
                **self.dataset_kwargs
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader.
        
        Returns:
            Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader.
        
        Returns:
            Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader.
        
        Returns:
            Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for batching.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # Pad features to same length
        features = [sample["features"] for sample in batch]
        features = torch.nn.utils.rnn.pad_sequence(
            features, batch_first=True, padding_value=0
        )
        
        # Get transcripts
        transcripts = [sample["transcript"] for sample in batch]
        
        # Get metadata
        audio_paths = [sample["audio_path"] for sample in batch]
        durations = [sample["duration"] for sample in batch]
        speaker_ids = [sample["speaker_id"] for sample in batch]
        
        return {
            "features": features,
            "transcripts": transcripts,
            "audio_paths": audio_paths,
            "durations": durations,
            "speaker_ids": speaker_ids
        }
