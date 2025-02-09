'''
This file contains the implementation of the RecruiterPipeline class. 
This class is responsible for orchestrating the entire conversational pipeline. 
It records audio input, transcribes it using the Speech-to-Text (STT) module, 
generates a response using the Language Model (LLM) module, and synthesizes the response 
using the Text-to-Speech (TTS) module. The conversation is carried out in a loop until 
the user exits the application.
'''

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os

from core.stt.whisper_stt import WhisperSTT
from core.llm.openai_llm import OpenAILLM
from core.tts.google_tts import GoogleTTS

class RecruiterPipeline:
    def __init__(self, stt: WhisperSTT, llm: OpenAILLM, tts: GoogleTTS):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        
    def record_audio(self, duration=5, fs=16000):
        print(f"Using audio device: {sd.query_devices(1)}")  # Using MacBook Air Microphone (index 1)
        recording = sd.rec(int(duration * fs),
                          samplerate=fs,
                          channels=1,
                          dtype='int16',
                          device=1)  # Changed from 2 to 1
        sd.wait()
        write("input.wav", fs, recording)
        print(f"Saved audio to: {os.path.abspath('input.wav')}")  # Show full path
        return os.path.abspath("input.wav")  # Return absolute path
    
    def _load_audio(self, file_path):
        """Load an audio file and return the data and sample rate."""
        from scipy.io import wavfile
        sample_rate, data = wavfile.read(file_path)
        return data, sample_rate

    def run_conversation(self):
        while True:
            # STT
            audio_path = self.record_audio()
            text = self.stt.transcribe(audio_path)
            
            # LLM
            response = self.llm.generate_response(text)
            
            # TTS
            speech_file = self.tts.synthesize(response)
            
            # Playback
            sd.play(*self._load_audio(speech_file))
            sd.wait()

# Correct: instantiate each module
stt = WhisperSTT()       # Create an instance, not just a reference to the class
llm = OpenAILLM()
tts = GoogleTTS()

pipeline = RecruiterPipeline(stt, llm, tts)
pipeline.run_conversation()
