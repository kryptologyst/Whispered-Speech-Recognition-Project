"""Training script for whispered speech recognition."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb

from src.models import Wav2Vec2ConformerModel
from src.data import WhisperedSpeechDataModule
from src.metrics import EvaluationReport
from src.utils import (
    setup_logging, set_seed, get_device, create_output_dir,
    EarlyStopping, load_config, save_config
)


class Trainer:
    """Trainer for whispered speech recognition model."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup logging
        self.logger = setup_logging(config.log_level)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Setup device
        self.device = get_device(config.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Create output directories
        create_output_dir(config.output_dir)
        create_output_dir(config.checkpoint_dir)
        create_output_dir(config.log_dir)
        
        # Initialize model
        self.model = Wav2Vec2ConformerModel(**config.model)
        self.model.to(self.device)
        
        # Initialize data module
        self.data_module = WhisperedSpeechDataModule(**config.data)
        self.data_module.setup()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.trainer.lr_patience,
            verbose=True
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.trainer.early_stopping_patience,
            min_delta=config.trainer.early_stopping_min_delta
        )
        
        # Initialize evaluation
        self.evaluator = EvaluationReport()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
        
        self.logger.info("Trainer initialized successfully")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            features = batch["features"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Note: In a real implementation, you would need to convert
            # transcripts to token IDs for CTC loss computation
            # For now, we'll use a placeholder
            outputs = self.model(features)
            loss = outputs["loss"]
            
            if loss is not None:
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.max_grad_norm
                )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Log progress
            if batch_idx % self.config.trainer.log_interval == 0:
                self.logger.info(
                    f"Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item() if loss is not None else 0:.4f}"
                )
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {"train_loss": avg_loss}
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        references = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                features = batch["features"].to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = outputs["loss"]
                
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
                
                # Collect predictions and references for evaluation
                # Note: In a real implementation, you would decode logits to text
                predictions.extend(batch["transcripts"])
                references.extend(batch["transcripts"])  # Placeholder
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute evaluation metrics
        eval_metrics = self.evaluator.generate_report(
            predictions=predictions,
            references=references
        )
        
        return {
            "val_loss": avg_loss,
            "wer": eval_metrics["wer"]["wer"],
            "cer": eval_metrics["cer"]["cer"]
        }
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("Starting training...")
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        best_val_loss = float("inf")
        train_start_time = time.time()
        
        for epoch in range(self.config.trainer.max_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_loss"])
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.trainer.max_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"WER: {val_metrics['wer']:.4f}, "
                f"CER: {val_metrics['cer']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Wandb logging
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "wer": val_metrics["wer"],
                    "cer": val_metrics["cer"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time
                })
            
            # Save checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, "best.ckpt"),
                    epoch, val_metrics
                )
            
            # Save latest checkpoint
            self.save_checkpoint(
                os.path.join(self.config.checkpoint_dir, "latest.ckpt"),
                epoch, val_metrics
            )
            
            # Early stopping
            if self.early_stopping(val_metrics["val_loss"], self.model):
                self.logger.info("Early stopping triggered")
                break
        
        total_time = time.time() - train_start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        if self.config.use_wandb:
            wandb.finish()
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint["epoch"], checkpoint["metrics"]


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train whispered speech recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
