"""Main entry point for whispered speech recognition."""

import argparse
import logging
from pathlib import Path

from src.utils import setup_logging, load_config
from src.models import WhisperedSpeechRecognizer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Whispered Speech Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train --config configs/config.yaml
  
  # Evaluate model
  python main.py eval --config configs/config.yaml --checkpoint checkpoints/best.ckpt
  
  # Transcribe single file
  python main.py transcribe --audio path/to/audio.wav --checkpoint checkpoints/best.ckpt
  
  # Run demo
  python main.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", required=True, help="Path to config file")
    train_parser.add_argument("--resume", help="Path to checkpoint to resume from")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--config", required=True, help="Path to config file")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    transcribe_parser.add_argument("--audio", required=True, help="Path to audio file")
    transcribe_parser.add_argument("--checkpoint", help="Path to model checkpoint")
    transcribe_parser.add_argument("--beam-size", type=int, default=5, help="Beam search size")
    transcribe_parser.add_argument("--use-lm", action="store_true", help="Use language model")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo application")
    
    # Prepare data command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare dataset")
    data_parser.add_argument("--config", required=True, help="Path to data config file")
    data_parser.add_argument("--force-recreate", action="store_true", help="Force recreate dataset")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from scripts.train import main as train_main
        train_main()
    
    elif args.command == "eval":
        from scripts.evaluate import main as eval_main
        eval_main()
    
    elif args.command == "transcribe":
        # Load config if available
        config = None
        if hasattr(args, 'config') and args.config:
            config = load_config(args.config)
        
        # Initialize recognizer
        recognizer = WhisperedSpeechRecognizer()
        
        # Load checkpoint if provided
        if args.checkpoint:
            recognizer.load_checkpoint(args.checkpoint)
        
        # Transcribe audio
        transcription = recognizer.transcribe(
            args.audio,
            beam_size=args.beam_size,
            use_lm=args.use_lm
        )
        
        print(f"Transcription: {transcription}")
    
    elif args.command == "demo":
        import subprocess
        import sys
        
        # Run Streamlit demo
        demo_path = Path(__file__).parent / "demo" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)])
    
    elif args.command == "prepare-data":
        from scripts.prepare_data import main as prepare_main
        prepare_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
