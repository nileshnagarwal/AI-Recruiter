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

from core.stt.base_stt import BaseSTT
from core.llm.base_llm import BaseLLM
from core.tts.base_tts import BaseTTS

class RecruiterPipeline:
    def __init__(self, stt: BaseSTT, llm: BaseLLM, tts: BaseTTS):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        
    def record_audio(self, duration=5, fs=16000):
        print("Listening...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("input.wav", fs, recording)
        return "input.wav"
    
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
