"""
This module provides the GoogleTTS class for text-to-speech synthesis using Google Cloud's Text-to-Speech API.
Classes:
    GoogleTTS: A class that inherits from BaseTTS and provides methods to synthesize speech from text.
Methods:
    __init__(): Initializes the GoogleTTS class and creates a TextToSpeechClient instance.
    synthesize(text, language="hi-IN"): Synthesizes speech from the provided text and saves it as a WAV file.
Usage:
    google_tts = GoogleTTS()
    audio_file_path = google_tts.synthesize("Hello, world!", language="en-US")

"""

from google.cloud import texttospeech
from core.tts.base_tts import BaseTTS
import os
from dotenv import load_dotenv
load_dotenv()

class GoogleTTS(BaseTTS):
    def __init__(self):
        # load_dotenv() reads .env file and adds variables to os.environ
        load_dotenv()  # looks for .env file in current/parent directories
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        if not project_id:
            raise ValueError("Missing GOOGLE_CLOUD_PROJECT_ID in .env file")
            
        self.client = texttospeech.TextToSpeechClient()
        
    def synthesize(self, text, language="hi-IN"):
        # Configure the voice request
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice params - using WaveNet for higher quality
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=f"{language}-Wavenet-B"
        )
        
        # Select the audio encoding
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        # Perform the synthesis
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save the audio file
        output_path = "response.wav"
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            
        return output_path
