import sounddevice as sd
import numpy as np

print('Testing passthrough...')
print('Input:  device 1 - Microfoon (Realtek Audio)')
print('Output: device 4 - Headphones (WH-CH720N Stereo)')
print('Speak into your mic - press Ctrl+C to stop')

def callback(indata, outdata, frames, time, status):
    outdata[:] = indata

with sd.Stream(device=(1, 4),
               samplerate=44100,
               blocksize=512,
               channels=1,
               callback=callback):
    input('Press ENTER to stop...')