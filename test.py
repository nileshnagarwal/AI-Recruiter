import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Print all available audio devices
print("\nAll audio devices:")
print(sd.query_devices())

# Find default devices
print("\nDefault input device:")
print(sd.query_devices(kind='input'))
print("\nDefault output device:")
print(sd.query_devices(kind='output'))

duration = 5  # seconds
fs = 16000
print(sd.query_devices())
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=2)
sd.wait()
print("Peak amplitude:", np.max(np.abs(recording)))  # Should be > 0 if audio captured
write("input.wav", fs, recording)