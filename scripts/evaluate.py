"""Evaluation script for whispered speech recognition."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from src.models import WhisperedSpeechRecognizer
from src.data import WhisperedSpeechDataModule
from src.metrics import EvaluationReport
from src.utils import (
    setup_logging, set_seed, get_device, create_output_dir,
    load_config, save_config
)


class Evaluator:
    """Evaluator for whispered speech recognition model."""
    
    def __init__(self, config: DictConfig):
        """Initialize evaluator.
        
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
        
        # Initialize recognizer
        self.recognizer = WhisperedSpeechRecognizer(
            model_name=config.model.model_name,
            device=self.device
        )
        
        # Initialize data module
        self.data_module = WhisperedSpeechDataModule(**config.data)
        self.data_module.setup("test")
        
        # Initialize evaluation
        self.evaluator = EvaluationReport()
        
        self.logger.info("Evaluator initialized successfully")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.recognizer.load_checkpoint(checkpoint_path)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate_dataset(
        self,
        dataloader,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            dataloader: Test data loader
            save_predictions: Whether to save predictions
            
        Returns:
            Evaluation results
        """
        self.logger.info("Starting evaluation...")
        
        predictions = []
        references = []
        confidences = []
        audio_paths = []
        durations = []
        inference_times = []
        
        total_samples = len(dataloader.dataset)
        
        for batch_idx, batch in enumerate(dataloader):
            batch_predictions = []
            batch_confidences = []
            batch_inference_times = []
            
            for i in range(len(batch["audio_paths"])):
                audio_path = batch["audio_paths"][i]
                duration = batch["durations"][i]
                
                # Measure inference time
                start_time = time.time()
                
                try:
                    # Transcribe audio
                    prediction = self.recognizer.transcribe(
                        audio_path,
                        beam_size=self.config.eval.beam_size,
                        use_lm=self.config.eval.use_lm
                    )
                    
                    # Compute confidence (placeholder - would need actual confidence scores)
                    confidence = np.random.uniform(0.7, 0.95)  # Placeholder
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {audio_path}: {e}")
                    prediction = ""
                    confidence = 0.0
                
                inference_time = time.time() - start_time
                
                batch_predictions.append(prediction)
                batch_confidences.append(confidence)
                batch_inference_times.append(inference_time)
            
            # Collect batch results
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
            inference_times.extend(batch_inference_times)
            references.extend(batch["transcripts"])
            audio_paths.extend(batch["audio_paths"])
            durations.extend(batch["durations"])
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Processed {batch_idx * dataloader.batch_size}/{total_samples} samples"
                )
        
        # Generate evaluation report
        report = self.evaluator.generate_report(
            predictions=predictions,
            references=references,
            confidences=confidences,
            audio_durations=durations,
            inference_times=inference_times
        )
        
        # Save predictions if requested
        if save_predictions:
            self.save_predictions(
                predictions, references, confidences,
                audio_paths, durations, inference_times
            )
        
        self.logger.info("Evaluation completed")
        return report
    
    def save_predictions(
        self,
        predictions: List[str],
        references: List[str],
        confidences: List[float],
        audio_paths: List[str],
        durations: List[float],
        inference_times: List[float]
    ) -> None:
        """Save predictions to CSV file.
        
        Args:
            predictions: List of predictions
            references: List of references
            confidences: List of confidence scores
            audio_paths: List of audio file paths
            durations: List of audio durations
            inference_times: List of inference times
        """
        results_df = pd.DataFrame({
            "audio_path": audio_paths,
            "prediction": predictions,
            "reference": references,
            "confidence": confidences,
            "duration": durations,
            "inference_time": inference_times,
            "rtf": np.array(inference_times) / np.array(durations)
        })
        
        output_path = os.path.join(self.config.output_dir, "predictions.csv")
        results_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to {output_path}")
    
    def evaluate_single_file(self, audio_path: str) -> Dict[str, Any]:
        """Evaluate model on single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Evaluation results for single file
        """
        start_time = time.time()
        
        # Transcribe audio
        prediction = self.recognizer.transcribe(
            audio_path,
            beam_size=self.config.eval.beam_size,
            use_lm=self.config.eval.use_lm
        )
        
        inference_time = time.time() - start_time
        
        # Load reference if available
        reference = ""  # Would need to load from metadata
        
        return {
            "audio_path": audio_path,
            "prediction": prediction,
            "reference": reference,
            "inference_time": inference_time
        }
    
    def generate_leaderboard(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Generate model leaderboard.
        
        Args:
            report: Evaluation report
            
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_data = {
            "Model": [self.config.experiment_name],
            "WER": [report["wer"]["wer"]],
            "CER": [report["cer"]["cer"]],
            "Token Accuracy": [report["accuracy"]["token_accuracy"]],
            "Character Accuracy": [report["accuracy"]["char_accuracy"]],
            "RTF Mean": [report.get("rtf", {}).get("rtf_mean", 0.0)],
            "RTF Std": [report.get("rtf", {}).get("rtf_std", 0.0)],
            "Total Samples": [report["summary"]["total_samples"]]
        }
        
        if "confidence" in report:
            leaderboard_data["ECE"] = [report["confidence"]["ece"]]
            leaderboard_data["MCE"] = [report["confidence"]["mce"]]
            leaderboard_data["Avg Confidence"] = [report["confidence"]["avg_confidence"]]
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Save leaderboard
        output_path = os.path.join(self.config.output_dir, "leaderboard.csv")
        leaderboard_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Leaderboard saved to {output_path}")
        return leaderboard_df
    
    def run_evaluation(self) -> None:
        """Run complete evaluation."""
        # Load checkpoint
        checkpoint_path = self.config.eval.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.load_checkpoint(checkpoint_path)
        
        # Get test dataloader
        test_loader = self.data_module.test_dataloader()
        
        # Run evaluation
        report = self.evaluate_dataset(test_loader)
        
        # Print report
        self.evaluator.print_report(report)
        
        # Generate leaderboard
        leaderboard = self.generate_leaderboard(report)
        print("\nLEADERBOARD:")
        print(leaderboard.to_string(index=False))
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "evaluation_report.json")
        import json
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate whispered speech recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", type=str, help="Path to single audio file for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override checkpoint path if provided
    config.eval.checkpoint_path = args.checkpoint
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    if args.audio:
        # Evaluate single file
        result = evaluator.evaluate_single_file(args.audio)
        print(f"Prediction: {result['prediction']}")
        print(f"Inference time: {result['inference_time']:.3f}s")
    else:
        # Run full evaluation
        evaluator.run_evaluation()


if __name__ == "__main__":
    main()
