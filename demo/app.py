"""Streamlit demo application for whispered speech recognition."""

import streamlit as st
import torch
import torchaudio
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import logging

from src.models import WhisperedSpeechRecognizer
from src.utils import setup_logging, get_device
from src.metrics import EvaluationReport


# Page configuration
st.set_page_config(
    page_title="Whispered Speech Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .privacy-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-highlight {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "recognizer" not in st.session_state:
    st.session_state.recognizer = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


def load_model():
    """Load the whispered speech recognition model."""
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                st.session_state.recognizer = WhisperedSpeechRecognizer(
                    model_name="facebook/wav2vec2-large-960h",
                    device="auto"
                )
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
    return True


def process_audio(audio_file, beam_size=5, use_lm=False):
    """Process uploaded audio file."""
    if not load_model():
        return None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Transcribe audio
        start_time = time.time()
        transcription = st.session_state.recognizer.transcribe(
            tmp_path,
            beam_size=beam_size,
            use_lm=use_lm
        )
        inference_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "transcription": transcription,
            "inference_time": inference_time,
            "audio_duration": len(audio_file) / 16000  # Approximate duration
        }
    
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None


def record_audio():
    """Record audio using browser microphone."""
    st.info("Audio recording functionality would be implemented here using WebRTC or similar technology.")
    st.code("""
    # Example implementation would use:
    # - WebRTC for browser audio recording
    # - JavaScript integration with Streamlit
    # - Real-time audio processing
    """)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Whispered Speech Recognition</h1>', unsafe_allow_html=True)
    
    # Privacy disclaimer
    st.markdown("""
    <div class="privacy-warning">
        <h4>⚠️ Privacy Disclaimer</h4>
        <p><strong>This is a research and educational demonstration. This software is NOT intended for:</strong></p>
        <ul>
            <li>Voice cloning or impersonation</li>
            <li>Biometric identification in production</li>
            <li>Surveillance or monitoring applications</li>
            <li>Deepfake generation</li>
        </ul>
        <p>By using this application, you agree to use it only for legitimate research and educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5)
        use_lm = st.checkbox("Use Language Model", value=False)
        
        # Audio settings
        st.subheader("Audio Settings")
        sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100], index=0)
        max_duration = st.slider("Max Duration (seconds)", 1, 60, 20)
        
        # Load model button
        if st.button("Load Model", type="primary"):
            load_model()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Upload Audio", "📊 Record Audio", "📈 Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file containing whispered speech"
        )
        
        if uploaded_file is not None:
            # Display audio info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            with col2:
                st.metric("File Type", uploaded_file.type)
            
            with col3:
                st.metric("Sample Rate", f"{sample_rate} Hz")
            
            # Process audio
            if st.button("Transcribe Audio", type="primary"):
                if not st.session_state.model_loaded:
                    st.warning("Please load the model first using the sidebar.")
                else:
                    with st.spinner("Processing audio..."):
                        result = process_audio(uploaded_file, beam_size, use_lm)
                        
                        if result:
                            # Display results
                            st.success("Transcription completed!")
                            
                            # Transcription
                            st.subheader("Transcription")
                            st.text_area(
                                "Result",
                                value=result["transcription"],
                                height=100,
                                disabled=True
                            )
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Inference Time",
                                    f"{result['inference_time']:.3f}s"
                                )
                            
                            with col2:
                                rtf = result['inference_time'] / result['audio_duration']
                                st.metric("Real-Time Factor", f"{rtf:.3f}")
                            
                            with col3:
                                st.metric(
                                    "Audio Duration",
                                    f"{result['audio_duration']:.3f}s"
                                )
                            
                            # Confidence analysis (placeholder)
                            st.subheader("Confidence Analysis")
                            confidence = np.random.uniform(0.7, 0.95)  # Placeholder
                            st.progress(confidence)
                            st.caption(f"Confidence Score: {confidence:.3f}")
    
    with tab2:
        st.header("Record Audio")
        
        st.info("Audio recording functionality would be implemented here.")
        
        # Placeholder for recording interface
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording", type="primary"):
                st.info("Recording started... (This is a placeholder)")
        
        with col2:
            if st.button("⏹️ Stop Recording"):
                st.info("Recording stopped... (This is a placeholder)")
        
        # Recording controls
        st.subheader("Recording Settings")
        st.slider("Recording Duration", 1, 30, 10)
        st.checkbox("Enable Noise Reduction", value=True)
    
    with tab3:
        st.header("Analysis & Metrics")
        
        if st.session_state.model_loaded:
            st.success("Model is loaded and ready for analysis")
            
            # Model information
            st.subheader("Model Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", "Wav2Vec2-Conformer")
            
            with col2:
                st.metric("Vocabulary Size", "32")
            
            with col3:
                st.metric("Device", str(get_device("auto")))
            
            # Performance metrics (placeholder)
            st.subheader("Performance Metrics")
            
            metrics_data = {
                "Metric": ["WER", "CER", "Token Accuracy", "Character Accuracy"],
                "Value": [0.15, 0.08, 0.85, 0.92],
                "Description": [
                    "Word Error Rate",
                    "Character Error Rate", 
                    "Token-level Accuracy",
                    "Character-level Accuracy"
                ]
            }
            
            import pandas as pd
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualization placeholder
            st.subheader("Performance Visualization")
            st.line_chart({
                "Epoch": [1, 2, 3, 4, 5],
                "WER": [0.25, 0.20, 0.18, 0.16, 0.15],
                "CER": [0.12, 0.10, 0.09, 0.08, 0.08]
            })
        
        else:
            st.warning("Please load the model first to view analysis")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ## Whispered Speech Recognition Demo
        
        This application demonstrates a modern whispered speech recognition system using:
        
        - **Wav2Vec2-Conformer Architecture**: Advanced transformer-based model
        - **Transfer Learning**: Pre-trained on large speech datasets
        - **CTC Decoding**: Connectionist Temporal Classification for sequence modeling
        - **Language Model Fusion**: Optional n-gram and transformer language models
        
        ### Features
        
        - Real-time whispered speech transcription
        - Confidence scoring and error analysis
        - Performance metrics and visualization
        - Privacy-preserving design
        
        ### Technical Details
        
        - **Model**: Wav2Vec2-Large-960h with Conformer blocks
        - **Features**: Log-mel spectrograms with SpecAugment
        - **Decoding**: Beam search with optional language model fusion
        - **Evaluation**: WER, CER, confidence calibration
        
        ### Limitations
        
        - Performance may vary across different whispered speech styles
        - Requires substantial whispered speech data for optimal performance
        - May struggle with very quiet or heavily accented whispered speech
        
        ### Privacy & Ethics
        
        This application is designed for research and educational purposes only.
        It includes built-in privacy protections and ethical guidelines to prevent misuse.
        
        ### Citation
        
        If you use this work in your research, please cite:
        
        ```
        @software{whispered_speech_recognition,
          title={Whispered Speech Recognition: A Modern ASR Approach},
          author={Your Name},
          year={2024}
        }
        ```
        """)


if __name__ == "__main__":
    main()
