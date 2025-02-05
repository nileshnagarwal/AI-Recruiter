'''
This is the subclass of the BaseTTS class.
It uses the Coqui TTS model to synthesize the text.
It will generate the audio response and save it to a file.
'''

from TTS.api import TTS
from core.tts.base_tts import BaseTTS

class CoquiTTS(BaseTTS):
    def __init__(self):
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
    def synthesize(self, text, language="hi-IN"):
        output_path = "response.wav"
        self.model.tts_to_file(
            text=text,
            speaker_wav="reference_speaker.wav",
            language=language,
            file_path=output_path
        )
        return output_path
