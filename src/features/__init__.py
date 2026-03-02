"""Feature extraction utilities for audio processing."""

import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple


class LogMelSpectrogram:
    """Log mel spectrogram feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = False
    ):
        """Initialize log mel spectrogram extractor.
        
        Args:
            sample_rate: Sample rate of input audio
            n_fft: Size of FFT window
            hop_length: Number of samples between successive frames
            win_length: Size of window function
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency
            power: Power of the magnitude spectrogram
            normalized: Whether to normalize the mel spectrogram
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            power=power,
            normalized=normalized
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log mel spectrogram features.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Log mel spectrogram features
        """
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        return log_mel_spec


class MFCC:
    """MFCC feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        norm: str = "ortho"
    ):
        """Initialize MFCC extractor.
        
        Args:
            sample_rate: Sample rate of input audio
            n_mfcc: Number of MFCC coefficients
            n_fft: Size of FFT window
            hop_length: Number of samples between successive frames
            win_length: Size of window function
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency
            f_max: Maximum frequency
            norm: Normalization mode
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": n_mels,
                "f_min": f_min,
                "f_max": self.f_max
            },
            norm=norm
        )
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            MFCC features
        """
        return self.mfcc(waveform)


class SpecAugment:
    """SpecAugment data augmentation for speech recognition."""
    
    def __init__(
        self,
        time_mask_param: int = 27,
        freq_mask_param: int = 12,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        time_mask_prob: float = 1.0,
        freq_mask_prob: float = 1.0
    ):
        """Initialize SpecAugment.
        
        Args:
            time_mask_param: Maximum length of time mask
            freq_mask_param: Maximum length of frequency mask
            num_time_masks: Number of time masks to apply
            num_freq_masks: Number of frequency masks to apply
            time_mask_prob: Probability of applying time mask
            freq_mask_prob: Probability of applying frequency mask
        """
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Augmented spectrogram
        """
        augmented = spectrogram.clone()
        
        # Apply time masking
        if torch.rand(1) < self.time_mask_prob:
            for _ in range(self.num_time_masks):
                augmented = self._time_mask(augmented)
        
        # Apply frequency masking
        if torch.rand(1) < self.freq_mask_prob:
            for _ in range(self.num_freq_masks):
                augmented = self._freq_mask(augmented)
        
        return augmented
    
    def _time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Time-masked spectrogram
        """
        batch_size, n_mels, time_steps = spectrogram.shape
        
        for b in range(batch_size):
            mask_length = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            mask_start = torch.randint(0, max(1, time_steps - mask_length), (1,)).item()
            
            spectrogram[b, :, mask_start:mask_start + mask_length] = 0
        
        return spectrogram
    
    def _freq_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Frequency-masked spectrogram
        """
        batch_size, n_mels, time_steps = spectrogram.shape
        
        for b in range(batch_size):
            mask_length = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            mask_start = torch.randint(0, max(1, n_mels - mask_length), (1,)).item()
            
            spectrogram[b, mask_start:mask_start + mask_length, :] = 0
        
        return spectrogram


class CMVN:
    """Cepstral Mean and Variance Normalization."""
    
    def __init__(self, norm_vars: bool = True):
        """Initialize CMVN.
        
        Args:
            norm_vars: Whether to normalize variances
        """
        self.norm_vars = norm_vars
        self.mean = None
        self.std = None
    
    def fit(self, features: torch.Tensor) -> None:
        """Fit CMVN parameters.
        
        Args:
            features: Training features
        """
        # Compute mean and std across time dimension
        self.mean = torch.mean(features, dim=-1, keepdim=True)
        if self.norm_vars:
            self.std = torch.std(features, dim=-1, keepdim=True)
    
    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply CMVN normalization.
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        if self.mean is None:
            raise ValueError("CMVN not fitted. Call fit() first.")
        
        normalized = features - self.mean
        
        if self.norm_vars and self.std is not None:
            normalized = normalized / (self.std + 1e-8)
        
        return normalized
    
    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Fit CMVN and transform features.
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        self.fit(features)
        return self.transform(features)
