''' 
Description: This is the subclass of the BaseSTT class. 
It uses the Whisper STT model to transcribe the audio.
'''

import whisper
from core.stt.base_stt import BaseSTT
import os
import numpy as np

class WhisperSTT(BaseSTT):
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path):
        print(f"[DEBUG] Transcribing: {audio_path}")  # Verify path received
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
        
        audio = whisper.load_audio(audio_path)
        print(f"Audio array range: {np.min(audio)} to {np.max(audio)}")  # Shouldn't be 0
        
        result = self.model.transcribe(
            audio,  # Use loaded audio array instead of file path
            language='en',
            fp16=False
        )
        return result["text"].strip()
