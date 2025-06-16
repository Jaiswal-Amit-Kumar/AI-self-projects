import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

fs = 44100  # Sample rate
duration = 5  # Recording duration in seconds

myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int32')
print("Recording Audio")
sd.wait()
print("Audio recording complete")

# Save the recording as a WAV file
wav.write('output.wav', fs, myrecording)