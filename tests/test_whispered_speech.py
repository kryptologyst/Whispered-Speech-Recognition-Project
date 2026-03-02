"""Tests for whispered speech recognition."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.models import Wav2Vec2ConformerModel, WhisperedSpeechRecognizer
from src.data import WhisperedSpeechDataset, WhisperedSpeechDataModule
from src.features import LogMelSpectrogram, SpecAugment, CMVN
from src.metrics import ASRMetrics, ConfidenceMetrics, PerformanceMetrics
from src.utils import get_device, set_seed, load_audio, save_audio


class TestModels:
    """Test model functionality."""
    
    def test_wav2vec2_conformer_model(self):
        """Test Wav2Vec2-Conformer model initialization."""
        model = Wav2Vec2ConformerModel(
            model_name="facebook/wav2vec2-base-960h",  # Use smaller model for testing
            vocab_size=32
        )
        
        assert model is not None
        assert model.vocab_size == 32
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = Wav2Vec2ConformerModel(
            model_name="facebook/wav2vec2-base-960h",
            vocab_size=32
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 1000
        input_values = torch.randn(batch_size, seq_len)
        
        # Forward pass
        outputs = model(input_values)
        
        assert "logits" in outputs
        assert outputs["logits"].shape[0] == batch_size
        assert outputs["logits"].shape[-1] == 32
    
    def test_whispered_speech_recognizer(self):
        """Test WhisperedSpeechRecognizer initialization."""
        recognizer = WhisperedSpeechRecognizer(
            model_name="facebook/wav2vec2-base-960h",
            device="cpu"
        )
        
        assert recognizer is not None
        assert recognizer.device.type == "cpu"


class TestData:
    """Test data functionality."""
    
    def test_whispered_speech_dataset(self):
        """Test WhisperedSpeechDataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy metadata
            metadata = {
                "id": ["test_001", "test_002"],
                "path": ["dummy1.wav", "dummy2.wav"],
                "transcript": ["hello world", "test transcription"],
                "duration": [2.0, 3.0],
                "split": ["train", "train"],
                "speaker_id": ["speaker_1", "speaker_2"]
            }
            
            import pandas as pd
            metadata_df = pd.DataFrame(metadata)
            metadata_path = Path(temp_dir) / "meta.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            # Create dummy audio files
            audio_dir = Path(temp_dir) / "wav"
            audio_dir.mkdir()
            
            for i, path in enumerate(metadata["path"]):
                audio_path = audio_dir / path
                # Create dummy audio
                dummy_audio = torch.randn(1, 16000)  # 1 second at 16kHz
                torchaudio.save(str(audio_path), dummy_audio, 16000)
            
            # Test dataset
            dataset = WhisperedSpeechDataset(
                data_dir=temp_dir,
                split="train",
                use_augmentation=False
            )
            
            assert len(dataset) == 2
            
            # Test getting item
            item = dataset[0]
            assert "features" in item
            assert "transcript" in item
            assert "audio_path" in item
    
    def test_data_module(self):
        """Test WhisperedSpeechDataModule."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data structure
            audio_dir = Path(temp_dir) / "wav"
            audio_dir.mkdir()
            
            # Create dummy metadata with all splits
            metadata = []
            for split in ["train", "validation", "test"]:
                for i in range(5):
                    audio_path = audio_dir / f"{split}_{i:03d}.wav"
                    dummy_audio = torch.randn(1, 16000)
                    torchaudio.save(str(audio_path), dummy_audio, 16000)
                    
                    metadata.append({
                        "id": f"{split}_{i:03d}",
                        "path": str(audio_path),
                        "transcript": f"test {split} {i}",
                        "duration": 1.0,
                        "split": split,
                        "speaker_id": f"speaker_{i}"
                    })
            
            import pandas as pd
            metadata_df = pd.DataFrame(metadata)
            metadata_path = Path(temp_dir) / "meta.csv"
            metadata_df.to_csv(metadata_path, index=False)
            
            # Test data module
            data_module = WhisperedSpeechDataModule(
                data_dir=temp_dir,
                batch_size=2,
                num_workers=0  # Use 0 workers for testing
            )
            
            data_module.setup()
            
            # Test dataloaders
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            assert len(test_loader) > 0


class TestFeatures:
    """Test feature extraction."""
    
    def test_log_mel_spectrogram(self):
        """Test LogMelSpectrogram feature extractor."""
        extractor = LogMelSpectrogram(sample_rate=16000)
        
        # Create dummy audio
        audio = torch.randn(1, 16000)  # 1 second
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 1  # Batch dimension
        assert features.shape[1] == 80  # Number of mel bins
        assert features.shape[2] > 0  # Time dimension
    
    def test_spec_augment(self):
        """Test SpecAugment."""
        augment = SpecAugment()
        
        # Create dummy spectrogram
        spec = torch.randn(2, 80, 100)  # Batch, mel, time
        
        # Apply augmentation
        augmented = augment(spec)
        
        assert augmented.shape == spec.shape
    
    def test_cmvn(self):
        """Test CMVN normalization."""
        cmvn = CMVN()
        
        # Create dummy features
        features = torch.randn(2, 80, 100)
        
        # Fit and transform
        normalized = cmvn.fit_transform(features)
        
        assert normalized.shape == features.shape


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_asr_metrics(self):
        """Test ASR metrics computation."""
        metrics = ASRMetrics()
        
        predictions = ["hello world", "test transcription"]
        references = ["hello world", "test transcription"]
        
        # Test WER
        wer_results = metrics.compute_wer(predictions, references)
        assert "wer" in wer_results
        assert wer_results["wer"] == 0.0  # Perfect match
        
        # Test CER
        cer_results = metrics.compute_cer(predictions, references)
        assert "cer" in cer_results
        assert cer_results["cer"] == 0.0  # Perfect match
        
        # Test accuracy
        acc_results = metrics.compute_accuracy(predictions, references)
        assert "token_accuracy" in acc_results
        assert "char_accuracy" in acc_results
    
    def test_confidence_metrics(self):
        """Test confidence metrics."""
        metrics = ConfidenceMetrics()
        
        predictions = ["hello world", "test transcription"]
        references = ["hello world", "test transcription"]
        confidences = [0.9, 0.8]
        
        results = metrics.compute_confidence_calibration(
            predictions, references, confidences
        )
        
        assert "ece" in results
        assert "mce" in results
    
    def test_performance_metrics(self):
        """Test performance metrics."""
        metrics = PerformanceMetrics()
        
        audio_durations = [1.0, 2.0, 3.0]
        inference_times = [0.5, 1.0, 1.5]
        
        rtf_results = metrics.compute_rtf(audio_durations, inference_times)
        assert "rtf_mean" in rtf_results
        assert rtf_results["rtf_mean"] == 0.5  # All RTF = 0.5


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device is not None
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        rand1 = torch.rand(1)
        set_seed(42)
        rand2 = torch.rand(1)
        
        # Should be the same with same seed
        assert torch.allclose(rand1, rand2)
    
    def test_load_save_audio(self):
        """Test audio loading and saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy audio
            original_audio = torch.randn(1, 16000)
            audio_path = Path(temp_dir) / "test.wav"
            
            # Save audio
            save_audio(original_audio, str(audio_path))
            
            # Load audio
            loaded_audio = load_audio(str(audio_path))
            
            # Should be approximately the same (allowing for some precision loss)
            assert torch.allclose(original_audio, loaded_audio, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
