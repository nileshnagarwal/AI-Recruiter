"""
GoogleTTS: Text-to-speech synthesis using Google Cloud's Text-to-Speech API with WaveNet voices.

Attributes:
    client: Google Cloud Text-to-Speech client
    temp_dir: Directory for temporary audio files

Methods:
    synthesize(text, language="hi-IN", voice_gender="FEMALE", output_path=None):
        Synthesizes speech from text using specified language and voice parameters.
"""

import os
import tempfile
import logging
from google.cloud import texttospeech
from google.api_core import retry
from core.tts.base_tts import BaseTTS

class GoogleTTS(BaseTTS):
    def __init__(self):
        try:
            self.client = texttospeech.TextToSpeechClient()
            self.temp_dir = tempfile.mkdtemp()
        except Exception as e:
            logging.error(f"Failed to initialize Google TTS: {str(e)}")
            raise
    
    @retry.Retry()
    def synthesize(self, text, language="hi-IN", voice_gender="FEMALE", output_path=None):
        """
        Synthesize text to speech using Google Cloud TTS.
        
        Args:
            text: Text to synthesize
            language: Language code (e.g. "hi-IN", "en-US")
            voice_gender: Voice gender ("FEMALE" or "MALE")
            output_path: Optional custom output path
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language,
                name=f"{language}-Wavenet-B",
                ssml_gender=getattr(texttospeech.SsmlVoiceGender, voice_gender)
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.0
            )
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            if not output_path:
                output_path = os.path.join(self.temp_dir, f"response_{os.getpid()}.wav")
                
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                
            return output_path
            
        except Exception as e:
            logging.error(f"Speech synthesis failed: {str(e)}")
            raise

    def __del__(self):
        """Cleanup temporary files on object destruction"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass