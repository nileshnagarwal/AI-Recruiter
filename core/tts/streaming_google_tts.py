"""
This module provides a streaming version of the GoogleTTS class that can handle
sentence-by-sentence synthesis for real-time audio playback.
"""

import os
import re
import queue
import threading
import sounddevice as sd
from google.cloud import texttospeech
from core.tts.base_tts import BaseTTS
from scipy.io import wavfile
import numpy as np
import tempfile

class StreamingGoogleTTS(BaseTTS):
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.is_interrupted = False
        self.playback_thread = None
        self.interrupt_event = threading.Event()
        self.fs = 24000  # Standard sample rate for Google TTS
        
        # Set up interrupt detection
        self.silence_threshold = 0.1
        self.interrupt_detector = None
        
    def split_into_sentences(self, text):
        """Split text into sentences for streaming synthesis."""
        # Simple sentence splitting on common punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def synthesize_sentence(self, text, language="hi-IN"):
        """Synthesize a single sentence and return the audio data."""
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=f"{language}-Wavenet-B"
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.fs
        )
        
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save to temporary file and read as numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(response.audio_content)
            temp_file.flush()
            _, audio_data = wavfile.read(temp_file.name)
        
        os.unlink(temp_file.name)
        return audio_data
        
    def detect_interrupt(self, indata, frames, time, status):
        """Callback for interrupt detection."""
        if status:
            print(f'Interrupt detection status: {status}')
        if np.max(np.abs(indata)) > self.silence_threshold:
            self.interrupt_event.set()
            
    def start_interrupt_detection(self):
        """Start listening for interruptions."""
        self.interrupt_event.clear()
        self.interrupt_detector = sd.InputStream(
            channels=1,
            samplerate=16000,
            callback=self.detect_interrupt,
            device=1  # Use the same device as recording
        )
        self.interrupt_detector.start()
        
    def stop_interrupt_detection(self):
        """Stop listening for interruptions."""
        if self.interrupt_detector:
            self.interrupt_detector.stop()
            self.interrupt_detector.close()
            self.interrupt_detector = None
        
    def playback_worker(self):
        """Worker thread for continuous audio playback."""
        try:
            while self.is_playing and not self.is_interrupted:
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    # Start interrupt detection before playing
                    self.start_interrupt_detection()
                    
                    # Play audio
                    sd.play(audio_data, self.fs)
                    
                    # Wait for playback to finish or interrupt
                    while sd.get_stream().active and not self.interrupt_event.is_set():
                        sd.sleep(100)
                    
                    # If interrupted, stop playback
                    if self.interrupt_event.is_set():
                        sd.stop()
                        self.is_interrupted = True
                        print("\nInterrupted by user")
                        break
                    
                    # Stop interrupt detection after sentence
                    self.stop_interrupt_detection()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Playback error: {e}")
                    break
        finally:
            self.stop_interrupt_detection()
                
    def start_playback(self):
        """Start the audio playback thread."""
        self.is_playing = True
        self.is_interrupted = False
        self.playback_thread = threading.Thread(target=self.playback_worker)
        self.playback_thread.start()
        
    def stop_playback(self):
        """Stop the audio playback thread."""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join()
            self.playback_thread = None
        self.stop_interrupt_detection()
        
    def synthesize(self, text, language="hi-IN"):
        """
        Stream the synthesis of text sentence by sentence.
        
        Args:
            text: Text to synthesize
            language: Language code (e.g. "hi-IN", "en-US")
            
        Returns:
            bool: True if completed normally, False if interrupted
        """
        # Start playback thread if not already running
        if not self.is_playing:
            self.start_playback()
        
        # Split text into sentences and synthesize each one
        sentences = self.split_into_sentences(text)
        for sentence in sentences:
            if sentence and not self.is_interrupted:
                audio_data = self.synthesize_sentence(sentence, language)
                self.audio_queue.put(audio_data)
            elif self.is_interrupted:
                break
        
        return not self.is_interrupted
        
    def __del__(self):
        """Cleanup on object destruction."""
        self.stop_playback() 