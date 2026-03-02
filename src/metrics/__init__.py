"""Evaluation metrics for whispered speech recognition."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict
import editdistance


class ASRMetrics:
    """Metrics for Automatic Speech Recognition evaluation."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize ASR metrics.
        
        Args:
            vocab: Vocabulary list for character-level metrics
        """
        self.vocab = vocab or list("abcdefghijklmnopqrstuvwxyz ")
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        
        self.logger = logging.getLogger(__name__)
    
    def compute_wer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute Word Error Rate (WER).
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Dictionary containing WER metrics
        """
        total_words = 0
        total_substitutions = 0
        total_insertions = 0
        total_deletions = 0
        
        wer_details = []
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Compute edit distance
            distance = editdistance.eval(pred_words, ref_words)
            
            # Estimate substitutions, insertions, deletions
            # This is an approximation based on edit distance
            len_diff = len(pred_words) - len(ref_words)
            
            if len_diff > 0:
                insertions = len_diff
                deletions = 0
            else:
                insertions = 0
                deletions = -len_diff
            
            substitutions = max(0, distance - insertions - deletions)
            
            total_words += len(ref_words)
            total_substitutions += substitutions
            total_insertions += insertions
            total_deletions += deletions
            
            wer_details.append({
                "pred": pred,
                "ref": ref,
                "distance": distance,
                "substitutions": substitutions,
                "insertions": insertions,
                "deletions": deletions,
                "wer": distance / max(len(ref_words), 1)
            })
        
        wer = (total_substitutions + total_insertions + total_deletions) / max(total_words, 1)
        
        return {
            "wer": wer,
            "substitutions": total_substitutions,
            "insertions": total_insertions,
            "deletions": total_deletions,
            "total_words": total_words,
            "details": wer_details
        }
    
    def compute_cer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute Character Error Rate (CER).
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Dictionary containing CER metrics
        """
        total_chars = 0
        total_errors = 0
        
        cer_details = []
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())
            
            # Compute edit distance
            distance = editdistance.eval(pred_chars, ref_chars)
            
            total_chars += len(ref_chars)
            total_errors += distance
            
            cer_details.append({
                "pred": pred,
                "ref": ref,
                "distance": distance,
                "cer": distance / max(len(ref_chars), 1)
            })
        
        cer = total_errors / max(total_chars, 1)
        
        return {
            "cer": cer,
            "total_errors": total_errors,
            "total_chars": total_chars,
            "details": cer_details
        }
    
    def compute_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute token and character accuracy.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Dictionary containing accuracy metrics
        """
        total_tokens = 0
        correct_tokens = 0
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            pred_chars = list(pred.lower())
            ref_chars = list(ref.lower())
            
            # Token accuracy
            min_len = min(len(pred_tokens), len(ref_tokens))
            for i in range(min_len):
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
            total_tokens += len(ref_tokens)
            
            # Character accuracy
            min_len = min(len(pred_chars), len(ref_chars))
            for i in range(min_len):
                if pred_chars[i] == ref_chars[i]:
                    correct_chars += 1
            total_chars += len(ref_chars)
        
        token_accuracy = correct_tokens / max(total_tokens, 1)
        char_accuracy = correct_chars / max(total_chars, 1)
        
        return {
            "token_accuracy": token_accuracy,
            "char_accuracy": char_accuracy,
            "correct_tokens": correct_tokens,
            "total_tokens": total_tokens,
            "correct_chars": correct_chars,
            "total_chars": total_chars
        }


class ConfidenceMetrics:
    """Metrics for model confidence evaluation."""
    
    def __init__(self):
        """Initialize confidence metrics."""
        self.logger = logging.getLogger(__name__)
    
    def compute_confidence_calibration(
        self,
        predictions: List[str],
        references: List[str],
        confidences: List[float]
    ) -> Dict[str, float]:
        """Compute confidence calibration metrics.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            confidences: List of confidence scores
            
        Returns:
            Dictionary containing calibration metrics
        """
        # Compute correctness for each prediction
        correctness = []
        for pred, ref in zip(predictions, references):
            # Simple correctness based on exact match
            is_correct = pred.lower().strip() == ref.lower().strip()
            correctness.append(float(is_correct))
        
        correctness = np.array(correctness)
        confidences = np.array(confidences)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = correctness[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            "ece": ece,
            "mce": mce,
            "avg_confidence": confidences.mean(),
            "avg_accuracy": correctness.mean()
        }


class PerformanceMetrics:
    """Metrics for model performance evaluation."""
    
    def __init__(self):
        """Initialize performance metrics."""
        self.logger = logging.getLogger(__name__)
    
    def compute_rtf(
        self,
        audio_durations: List[float],
        inference_times: List[float]
    ) -> Dict[str, float]:
        """Compute Real-Time Factor (RTF).
        
        Args:
            audio_durations: List of audio durations in seconds
            inference_times: List of inference times in seconds
            
        Returns:
            Dictionary containing RTF metrics
        """
        audio_durations = np.array(audio_durations)
        inference_times = np.array(inference_times)
        
        rtf_values = inference_times / audio_durations
        
        return {
            "rtf_mean": rtf_values.mean(),
            "rtf_std": rtf_values.std(),
            "rtf_min": rtf_values.min(),
            "rtf_max": rtf_values.max(),
            "rtf_median": np.median(rtf_values)
        }
    
    def compute_throughput(
        self,
        batch_sizes: List[int],
        inference_times: List[float]
    ) -> Dict[str, float]:
        """Compute throughput metrics.
        
        Args:
            batch_sizes: List of batch sizes
            inference_times: List of inference times in seconds
            
        Returns:
            Dictionary containing throughput metrics
        """
        batch_sizes = np.array(batch_sizes)
        inference_times = np.array(inference_times)
        
        throughput = batch_sizes / inference_times
        
        return {
            "throughput_mean": throughput.mean(),
            "throughput_std": throughput.std(),
            "throughput_min": throughput.min(),
            "throughput_max": throughput.max(),
            "throughput_median": np.median(throughput)
        }


class EvaluationReport:
    """Comprehensive evaluation report generator."""
    
    def __init__(self):
        """Initialize evaluation report."""
        self.asr_metrics = ASRMetrics()
        self.confidence_metrics = ConfidenceMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self,
        predictions: List[str],
        references: List[str],
        confidences: Optional[List[float]] = None,
        audio_durations: Optional[List[float]] = None,
        inference_times: Optional[List[float]] = None,
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            confidences: List of confidence scores
            audio_durations: List of audio durations
            inference_times: List of inference times
            batch_sizes: List of batch sizes
            
        Returns:
            Comprehensive evaluation report
        """
        report = {}
        
        # ASR metrics
        report["wer"] = self.asr_metrics.compute_wer(predictions, references)
        report["cer"] = self.asr_metrics.compute_cer(predictions, references)
        report["accuracy"] = self.asr_metrics.compute_accuracy(predictions, references)
        
        # Confidence metrics
        if confidences is not None:
            report["confidence"] = self.confidence_metrics.compute_confidence_calibration(
                predictions, references, confidences
            )
        
        # Performance metrics
        if audio_durations is not None and inference_times is not None:
            report["rtf"] = self.performance_metrics.compute_rtf(
                audio_durations, inference_times
            )
        
        if batch_sizes is not None and inference_times is not None:
            report["throughput"] = self.performance_metrics.compute_throughput(
                batch_sizes, inference_times
            )
        
        # Summary statistics
        report["summary"] = {
            "total_samples": len(predictions),
            "avg_prediction_length": np.mean([len(pred.split()) for pred in predictions]),
            "avg_reference_length": np.mean([len(ref.split()) for ref in references])
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted evaluation report.
        
        Args:
            report: Evaluation report dictionary
        """
        print("\n" + "="*60)
        print("WHISPERED SPEECH RECOGNITION EVALUATION REPORT")
        print("="*60)
        
        # WER/CER
        print(f"\nWord Error Rate (WER): {report['wer']['wer']:.4f}")
        print(f"Character Error Rate (CER): {report['cer']['cer']:.4f}")
        print(f"Token Accuracy: {report['accuracy']['token_accuracy']:.4f}")
        print(f"Character Accuracy: {report['accuracy']['char_accuracy']:.4f}")
        
        # Confidence metrics
        if "confidence" in report:
            print(f"\nExpected Calibration Error (ECE): {report['confidence']['ece']:.4f}")
            print(f"Maximum Calibration Error (MCE): {report['confidence']['mce']:.4f}")
            print(f"Average Confidence: {report['confidence']['avg_confidence']:.4f}")
            print(f"Average Accuracy: {report['confidence']['avg_accuracy']:.4f}")
        
        # Performance metrics
        if "rtf" in report:
            print(f"\nReal-Time Factor (RTF):")
            print(f"  Mean: {report['rtf']['rtf_mean']:.4f}")
            print(f"  Std:  {report['rtf']['rtf_std']:.4f}")
            print(f"  Min:  {report['rtf']['rtf_min']:.4f}")
            print(f"  Max:  {report['rtf']['rtf_max']:.4f}")
        
        if "throughput" in report:
            print(f"\nThroughput (samples/sec):")
            print(f"  Mean: {report['throughput']['throughput_mean']:.2f}")
            print(f"  Std:  {report['throughput']['throughput_std']:.2f}")
        
        # Summary
        print(f"\nTotal Samples: {report['summary']['total_samples']}")
        print(f"Average Prediction Length: {report['summary']['avg_prediction_length']:.2f} words")
        print(f"Average Reference Length: {report['summary']['avg_reference_length']:.2f} words")
        
        print("\n" + "="*60)
