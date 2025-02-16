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
from datetime import datetime
import time
import queue
import threading

from core.stt.whisper_stt import WhisperSTT
from core.llm.openai_llm import OpenAILLM
from core.tts.streaming_google_tts import StreamingGoogleTTS

class RecruiterPipeline:
    def __init__(self, stt: WhisperSTT, llm: OpenAILLM, tts: StreamingGoogleTTS):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.audio_dir = "recordings"
        self.conversation_history = []
        self.latency_history = []
        
        # Audio recording parameters
        self.fs = 16000  # Sample rate
        self.silence_threshold = 0.01  # Adjust this value based on your needs
        self.silence_duration = 2.0  # Seconds of silence before stopping
        self.min_duration = 1.0  # Minimum recording duration in seconds
        self.max_duration = 30.0  # Maximum recording duration in seconds
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        
    def is_silent(self, data):
        """Check if the audio chunk is silent."""
        return np.max(np.abs(data)) < self.silence_threshold
        
    def record_audio(self):
        """Record audio with automatic silence detection."""
        print("Recording... (speak now)")
        
        # Initialize variables for recording
        audio_chunks = []  # Stores raw audio data chunks
        silent_chunks = 0  # Counter for consecutive silent chunks
        # Calculate how many silent chunks we need before stopping (duration * samples per chunk)
        silent_chunk_threshold = int(self.silence_duration * self.fs / 1024)
        is_recording = True  # Flag to control recording state
        start_time = time.time()  # Track when recording started
        
        # This callback function is called automatically for each audio block
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status)  # Print any sounddevice errors/warnings
            if is_recording:
                # Make a copy of the incoming audio data and store it
                audio_chunks.append(indata.copy())
        
        # Start recording using sounddevice's InputStream
        with sd.InputStream(samplerate=self.fs, channels=1, callback=audio_callback,
                          blocksize=1024, device=1):
            while is_recording:
                # Safety check: don't record longer than maximum allowed duration
                if time.time() - start_time > self.max_duration:
                    print("\nMaximum recording duration reached")
                    is_recording = False  # Stop recording
                    break
                
                # Only check for silence if we have some audio data
                if len(audio_chunks) > 0:
                    latest_chunk = audio_chunks[-1]  # Get most recent audio chunk
                    
                    # Check if current chunk is silent
                    if self.is_silent(latest_chunk):
                        silent_chunks += 1  # Increment silent counter
                    else:
                        silent_chunks = 0  # Reset counter if noise is detected
                    
                    # Stop conditions: enough silence AND minimum recording time met
                    if silent_chunks >= silent_chunk_threshold and time.time() - start_time > self.min_duration:
                        print("\nSilence detected, stopping recording")
                        is_recording = False  # Stop recording
                        break
                
                # Pause briefly to avoid consuming too much CPU
                time.sleep(0.1)  # 100ms delay between checks

        is_recording = False  # Ensure flag is reset when exiting the stream
        
        # Combine all chunks
        recording = np.concatenate(audio_chunks, axis=0)
        
        # Save the recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.join(self.audio_dir, f"input_{timestamp}.wav")
        write(input_filename, self.fs, recording)
        print(f"Saved audio to: {os.path.abspath(input_filename)}")
        return os.path.abspath(input_filename)
    
    def save_conversation(self):
        """Save the conversation history and latencies to files with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_file = os.path.join(self.audio_dir, f"conversation_{timestamp}.txt")
        latency_file = os.path.join(self.audio_dir, f"latency_{timestamp}.txt")
        
        # Save conversation
        with open(conversation_file, "w", encoding="utf-8") as f:
            for turn in self.conversation_history:
                f.write(f"User: {turn['user']}\n")
                f.write(f"AI: {turn['ai']}\n\n")
        
        # Save latencies
        with open(latency_file, "w", encoding="utf-8") as f:
            f.write("Response Latencies (seconds):\n")
            for idx, latency in enumerate(self.latency_history, 1):
                f.write(f"Turn {idx}:\n")
                f.write(f"  STT Time: {latency['stt']:.2f}s\n")
                f.write(f"  First Response Time: {latency.get('first_response', 0):.2f}s\n")
                f.write(f"  Total Time: {latency['total']:.2f}s\n")
                if latency.get('interrupted', False):
                    f.write("  (Response was interrupted)\n")
                f.write("\n")
        
        print(f"Conversation saved to: {os.path.abspath(conversation_file)}")
        print(f"Latencies saved to: {os.path.abspath(latency_file)}")
        
    def run_conversation(self):
        try:
            while True:
                turn_start = time.time()
                
                # STT
                audio_path = self.record_audio()
                stt_start = time.time()
                text = self.stt.transcribe(audio_path)
                stt_time = time.time() - stt_start
                
                # LLM with streaming
                llm_start = time.time()
                first_response_time = None
                accumulated_response = ""
                was_interrupted = False
                
                for response_chunk in self.llm.generate_response(text, stream=True):
                    if not first_response_time:
                        first_response_time = time.time() - llm_start
                        
                    # Accumulate response for history
                    accumulated_response += response_chunk
                    
                    # Stream to TTS and check for interruption
                    completed = self.tts.synthesize(response_chunk)
                    if not completed:
                        was_interrupted = True
                        break
                
                # Save to conversation history
                self.conversation_history.append({
                    "user": text,
                    "ai": accumulated_response
                })
                
                # Calculate total latency
                total_time = time.time() - turn_start
                
                # Save latency information
                self.latency_history.append({
                    "stt": stt_time,
                    "first_response": first_response_time,
                    "total": total_time,
                    "interrupted": was_interrupted
                })
                
                # Print current turn latency
                print(f"\nTurn {len(self.latency_history)} Latencies:")
                print(f"  STT Time: {stt_time:.2f}s")
                print(f"  First Response Time: {first_response_time:.2f}s")
                print(f"  Total Time: {total_time:.2f}s")
                if was_interrupted:
                    print("  (Response was interrupted)")
                
                # Check if we should exit
                if hasattr(accumulated_response, 'get') and accumulated_response.get('should_exit', False):
                    print("\nInterview completed. Saving conversation history...")
                    self.save_conversation()
                    print("Thank you for participating in the interview!")
                    break
                
        except KeyboardInterrupt:
            print("\nSaving conversation history before exit...")
            self.save_conversation()
            print("Conversation ended.")
            
        finally:
            # Cleanup
            self.tts.stop_playback()

# Correct: instantiate each module
stt = WhisperSTT()
llm = OpenAILLM()
tts = StreamingGoogleTTS()

pipeline = RecruiterPipeline(stt, llm, tts)
pipeline.run_conversation()
