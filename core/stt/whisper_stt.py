''' 
Description: This is the subclass of the BaseSTT class. 
It uses the Whisper STT model to transcribe the audio.
'''

import whisper
from core.stt.base_stt import BaseSTT
import os
import numpy as np
import torch

class WhisperSTT(BaseSTT):
    def __init__(self, model_size="medium"):
        self.model = whisper.load_model(model_size)
        self.detected_language = None
        
    def preprocess_audio(self, audio):
        """Preprocess audio for Whisper model."""
        # Pad/trim audio to 30 seconds
        audio = whisper.pad_or_trim(audio)
        # Convert to mel spectrogram
        mel = whisper.log_mel_spectrogram(audio)
        return mel
        
    def detect_language(self, audio):
        """Detect the language of the audio using Whisper."""
        # Preprocess audio
        mel = self.preprocess_audio(audio)
        # Get language probabilities
        _, probs = self.model.detect_language(mel)
        # Get the most likely language
        language = max(probs.items(), key=lambda x: x[1])[0]
        return language
        
    def transcribe(self, audio_path):
        print(f"[DEBUG] Transcribing: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
        
        audio = whisper.load_audio(audio_path)
        print(f"Audio array range: {np.min(audio)} to {np.max(audio)}")
        
        # Detect language if not already detected or if it's a new conversation
        self.detected_language = self.detect_language(audio)
        print(f"Detected language: {self.detected_language}")
        
        result = self.model.transcribe(
            audio,
            language=self.detected_language,
            fp16=False
        )
        
        return result["text"].strip()
