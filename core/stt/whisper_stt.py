''' 
Description: This is the subclass of the BaseSTT class. 
It uses the Whisper STT model to transcribe the audio.
'''

import whisper
from core.stt.base_stt import BaseSTT

class WhisperSTT(BaseSTT):
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]
