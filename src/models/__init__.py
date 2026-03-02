"""Model architectures for whispered speech recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Any, Optional, Tuple
import logging

from ..utils import get_device


class ConformerBlock(nn.Module):
    """Conformer block implementation."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        """Initialize Conformer block.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            conv_kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv_module = ConformerConvModule(
            d_model, d_ff, conv_kernel_size, dropout
        )
        self.conv_norm = nn.LayerNorm(d_model)
        
        # Feed-forward module
        self.feed_forward = ConformerFeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Self-attention
        residual = x
        x = self.attention_norm(x)
        x, _ = self.self_attention(x, x, x)
        x = self.dropout(x) + residual
        
        # Convolution
        residual = x
        x = self.conv_norm(x)
        x = self.conv_module(x)
        x = self.dropout(x) + residual
        
        # Feed-forward
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual
        
        return x


class ConformerConvModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int,
        dropout: float = 0.1
    ):
        """Initialize convolution module.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.depthwise_conv = nn.Conv1d(
            d_ff, d_ff, kernel_size,
            padding=kernel_size // 2, groups=d_ff
        )
        self.pointwise_conv2 = nn.Conv1d(d_ff, d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Transpose for conv1d: (batch, seq, dim) -> (batch, dim, seq)
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: (batch, dim, seq) -> (batch, seq, dim)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward module.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Wav2Vec2ConformerModel(nn.Module):
    """Wav2Vec2 with Conformer architecture for whispered speech recognition."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        vocab_size: int = 32,
        conformer_num_blocks: int = 16,
        conformer_conv_kernel_size: int = 31,
        conformer_ff_expansion_factor: int = 4,
        conformer_self_attention_dropout: float = 0.1,
        conformer_conv_dropout: float = 0.1,
        conformer_feed_forward_dropout: float = 0.1,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = True
    ):
        """Initialize Wav2Vec2-Conformer model.
        
        Args:
            model_name: Pre-trained Wav2Vec2 model name
            vocab_size: Vocabulary size for CTC
            conformer_num_blocks: Number of Conformer blocks
            conformer_conv_kernel_size: Convolution kernel size
            conformer_ff_expansion_factor: Feed-forward expansion factor
            conformer_self_attention_dropout: Self-attention dropout
            conformer_conv_dropout: Convolution dropout
            conformer_feed_forward_dropout: Feed-forward dropout
            ctc_loss_reduction: CTC loss reduction method
            ctc_zero_infinity: CTC zero infinity setting
        """
        super().__init__()
        
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            ctc_loss_reduction=ctc_loss_reduction,
            ctc_zero_infinity=ctc_zero_infinity
        )
        
        # Get model dimensions
        d_model = self.wav2vec2.config.hidden_size
        d_ff = d_model * conformer_ff_expansion_factor
        n_heads = self.wav2vec2.config.num_attention_heads
        
        # Replace transformer layers with Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conformer_conv_kernel_size,
                dropout=conformer_self_attention_dropout
            )
            for _ in range(conformer_num_blocks)
        ])
        
        # CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Forward pass.
        
        Args:
            input_values: Input audio features
            attention_mask: Attention mask
            labels: Target labels for training
            
        Returns:
            Dictionary containing logits and loss
        """
        # Get Wav2Vec2 features
        outputs = self.wav2vec2.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Apply Conformer blocks
        for conformer_block in self.conformer_blocks:
            hidden_states = conformer_block(hidden_states)
        
        # CTC prediction
        logits = self.ctc_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.ctc_loss(
                logits.transpose(0, 1),  # (time, batch, vocab)
                labels,
                input_lengths=None,
                target_lengths=None,
                reduction=self.ctc_loss_reduction,
                zero_infinity=self.ctc_zero_infinity
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states
        }
    
    def generate(
        self,
        input_values: torch.Tensor,
        processor,
        beam_size: int = 5,
        use_lm: bool = False,
        lm_weight: float = 0.3,
        word_score: float = 0.0
    ) -> str:
        """Generate transcription.
        
        Args:
            input_values: Input audio features
            processor: Wav2Vec2 processor
            beam_size: Beam search size
            use_lm: Whether to use language model
            lm_weight: Language model weight
            word_score: Word score for beam search
            
        Returns:
            Generated transcription
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_values)
            logits = outputs["logits"]
            
            # Simple greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode with processor
            transcription = processor.decode(predicted_ids[0])
        
        return transcription


class WhisperedSpeechRecognizer:
    """High-level interface for whispered speech recognition."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        device: str = "auto"
    ):
        """Initialize recognizer.
        
        Args:
            model_name: Pre-trained model name
            device: Device to use
        """
        self.device = get_device(device)
        self.model_name = model_name
        
        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Initialize model
        self.model = Wav2Vec2ConformerModel(model_name=model_name)
        self.model.to(self.device)
        
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
        use_lm: bool = False
    ) -> str:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            beam_size: Beam search size
            use_lm: Whether to use language model
            
        Returns:
            Transcription text
        """
        from ..utils import load_audio
        
        # Load audio
        waveform = load_audio(audio_path)
        
        # Process with Wav2Vec2 processor
        inputs = self.processor(
            waveform.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        
        # Generate transcription
        transcription = self.model.generate(
            input_values,
            self.processor,
            beam_size=beam_size,
            use_lm=use_lm
        )
        
        return transcription
