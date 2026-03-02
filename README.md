# Whispered Speech Recognition Project

## PRIVACY DISCLAIMER

**IMPORTANT: This is a research and educational demonstration project. This software is NOT intended for production use in biometric identification, voice cloning, or any form of personal identification systems.**

### Prohibited Uses:
- **Voice cloning or impersonation** of individuals without explicit consent
- **Biometric identification** in production systems
- **Surveillance or monitoring** applications
- **Deepfake generation** or manipulation
- **Any form of identity theft** or fraud

### Ethical Guidelines:
- Use only for research, education, and legitimate speech processing applications
- Respect privacy rights and obtain proper consent for any voice data
- Do not use this technology to deceive or harm others
- Comply with all applicable laws and regulations regarding voice data

By using this software, you agree to these terms and acknowledge that misuse may have serious legal and ethical consequences.

---

## Overview

This project implements a modern whispered speech recognition system using transfer learning and advanced ASR techniques. Whispered speech presents unique challenges due to its lack of vocal cord vibration, different acoustic properties, and reduced volume compared to normal speech.

## Features

- **Modern ASR Architecture**: CTC/Attention hybrid with Conformer backbone
- **Transfer Learning**: Fine-tuned Wav2Vec2 models for whispered speech
- **Data Augmentation**: SpecAugment, speed/pitch perturbation, noise injection
- **Language Model Fusion**: Shallow fusion with n-gram and Transformer LMs
- **Comprehensive Evaluation**: WER/CER metrics with detailed analysis
- **Interactive Demo**: Streamlit app for real-time whispered speech recognition
- **Production Ready**: Proper configs, logging, and reproducible experiments

## Quick Start

1. **Installation**:
```bash
pip install -r requirements.txt
```

2. **Download/Prepare Data**:
```bash
python scripts/prepare_data.py --config configs/data_config.yaml
```

3. **Train Model**:
```bash
python scripts/train.py --config configs/train_config.yaml
```

4. **Run Demo**:
```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model definitions
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── decoding/          # Decoding algorithms
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Data directory
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated artifacts
├── demo/                  # Demo application
└── docs/                  # Documentation
```

## Dataset Schema

The project expects audio data in the following format:

- **Audio files**: WAV format, 16kHz sampling rate
- **Metadata**: CSV with columns: `id`, `path`, `transcript`, `speaker_id`, `split`
- **Optional**: JSON annotations for timestamps and speaker diarization

## Training

```bash
# Basic training
python scripts/train.py --config configs/train_config.yaml

# With specific model
python scripts/train.py --config configs/train_config.yaml model.name=conformer

# Resume from checkpoint
python scripts/train.py --config configs/train_config.yaml trainer.resume_from_checkpoint=checkpoints/last.ckpt
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --config configs/eval_config.yaml

# Generate detailed metrics
python scripts/evaluate.py --config configs/eval_config.yaml --detailed
```

## Demo Application

The Streamlit demo provides:
- Audio upload and recording
- Real-time transcription
- Confidence scores and timestamps
- Error analysis and visualization
- Model comparison

## Metrics

- **WER (Word Error Rate)**: Primary metric for ASR
- **CER (Character Error Rate)**: Character-level accuracy
- **RTF (Real-Time Factor)**: Inference speed
- **Confidence Calibration**: Model uncertainty estimation

## Known Limitations

- Performance may vary significantly across different whispered speech styles
- Requires substantial whispered speech data for optimal performance
- May struggle with very quiet or heavily accented whispered speech
- Real-time performance depends on hardware capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{whispered_speech_recognition,
  title={Whispered Speech Recognition},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Whispered-Speech-Recognition-Project}
}
```
# Whispered-Speech-Recognition-Project
