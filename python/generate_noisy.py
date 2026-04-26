import numpy as np
import wave
import struct

def generate_noisy_audio(output_path: str, duration: float = 5.0):
    """
    Generates a synthetic noisy audio file.
    It simulates a voice signal mixed with background noise.
    """

    sample_rate = 48000  # DeepFilterNet expects 48kHz
    num_samples = int(sample_rate * duration)

    print(f"Generating {duration} seconds of noisy audio...")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Total samples: {num_samples}")

    # 1. Simulate a voice signal
    #    A simple sine wave at 200Hz mimics a human voice fundamental
    t = np.linspace(0, duration, num_samples)
    voice = 0.4 * np.sin(2 * np.pi * 200 * t)   # 200Hz — voice-like
    voice += 0.2 * np.sin(2 * np.pi * 400 * t)  # 400Hz — harmonic
    voice += 0.1 * np.sin(2 * np.pi * 800 * t)  # 800Hz — harmonic

    # 2. Simulate background noise
    #    White noise mimics fan noise, air conditioning, crowd noise
    noise = 0.3 * np.random.randn(num_samples)

    # 3. Mix voice and noise together
    noisy_signal = voice + noise

    # 4. Normalize to prevent clipping (keep within [-1.0, 1.0])
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

    # 5. Convert to 16-bit PCM format (standard .wav format)
    samples_int16 = (noisy_signal * 32767).astype(np.int16)

    # 6. Write to .wav file
    with wave.open(output_path, 'w') as wav_file:
        wav_file.setnchannels(1)         # Mono
        wav_file.setsampwidth(2)         # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples_int16.tobytes())

    print(f"Noisy audio saved to: {output_path}")
    print(f"You can open it in any audio player to hear the noise.")


if __name__ == "__main__":
    generate_noisy_audio("noisy_input.wav")