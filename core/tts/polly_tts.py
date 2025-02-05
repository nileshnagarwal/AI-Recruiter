"""
A class to interact with Amazon Polly Text-to-Speech service.
This class provides methods to synthesize speech from text using Amazon Polly.
It inherits from the BaseTTS class.
Attributes:
    client (boto3.client): The Boto3 Polly client used to interact with the Polly service.
Methods:
    __init__():
        Initializes the PollyTTS class and sets up the Polly client.
    synthesize(text, language="hi-IN"):
        Synthesizes speech from the given text using Amazon Polly.
        Saves the synthesized speech as an MP3 file.
        Args:
            text (str): The text to be synthesized.
            language (str): The language code for the voice to be used. Defaults to "hi-IN".
        Returns:
            str: The filename of the saved MP3 file.
"""

import boto3
from core.tts.base_tts import BaseTTS

class PollyTTS(BaseTTS):
    def __init__(self):
        self.client = boto3.client("polly", region_name="ap-south-1")
        
    def synthesize(self, text, language="hi-IN"):
        response = self.client.synthesize_speech(
            OutputFormat="mp3",
            Text=text,
            VoiceId="Aditi" if language == "hi-IN" else "Raveena"
        )
        with open("response.mp3", "wb") as f:
            f.write(response["AudioStream"].read())
        return "response.mp3"
