import zmq
import numpy as np
import torch
import torchaudio.transforms as T
from df.enhance import enhance, init_df

# -----------------------------------------------
# Configuration
# Must match what C++ sends
# -----------------------------------------------
SAMPLE_RATE_HW    = 44100  # What C++ captures at (MME default)
SAMPLE_RATE_MODEL = 48000  # What DeepFilterNet expects
FRAMES_HW         = 1536   # Must match C++ FRAMES constant

# -----------------------------------------------
# Resamplers — created once at startup
# -----------------------------------------------
resampler_up   = T.Resample(SAMPLE_RATE_HW, SAMPLE_RATE_MODEL)  # 44100 → 48000
resampler_down = T.Resample(SAMPLE_RATE_MODEL, SAMPLE_RATE_HW)  # 48000 → 44100

def load_model():
    """Load DeepFilterNet once at startup."""
    print("Loading DeepFilterNet model...")
    model, df_state, _ = init_df()
    print("Model loaded! Ready to process audio.")
    return model, df_state

def process_chunk(audio_chunk: np.ndarray, model, df_state) -> np.ndarray:
    """
    Takes a numpy array of noisy audio (44100 Hz float32)
    and returns a clean numpy array (44100 Hz float32).

    Internally:
    1. Resample 44100 → 48000 Hz
    2. Run DeepFilterNet
    3. Resample 48000 → 44100 Hz
    """

    # 1. Convert numpy → torch tensor [1, samples]
    audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

    # 2. Upsample 44100 → 48000 Hz
    audio_48k = resampler_up(audio_tensor)

    # 3. Run DeepFilterNet at 48000 Hz
    enhanced_48k = enhance(model, df_state, audio_48k)

    # 4. Downsample 48000 → 44100 Hz
    enhanced_44k = resampler_down(enhanced_48k)

    # 5. Trim or pad to exactly match input size
    #    Resampling can produce slightly different lengths
    target_length = audio_chunk.shape[0]
    enhanced_np   = enhanced_44k.squeeze(0).numpy()

    if len(enhanced_np) > target_length:
        enhanced_np = enhanced_np[:target_length]
    elif len(enhanced_np) < target_length:
        enhanced_np = np.pad(enhanced_np, (0, target_length - len(enhanced_np)))

    return enhanced_np

def run_server():
    """
    Main server loop.
    Waits for audio chunks from C++, processes them, sends back clean audio.
    """

    # 1. Load AI model once
    model, df_state = load_model()

    # 2. Set up ZeroMQ
    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    print("Python server listening on tcp://127.0.0.1:5555")
    print("Waiting for audio chunks from C++...\n")

    chunk_count = 0

    while True:
        # 3. Receive noisy audio from C++
        message = socket.recv()

        # 4. Deserialize bytes → numpy float32
        noisy_chunk = np.frombuffer(message, dtype=np.float32).copy()

        # 5. Process with DeepFilterNet
        clean_chunk = process_chunk(noisy_chunk, model, df_state)

        # 6. Send clean audio back to C++
        socket.send(clean_chunk.astype(np.float32).tobytes())

        chunk_count += 1
        if chunk_count % 50 == 0:
            duration = chunk_count * FRAMES_HW / SAMPLE_RATE_HW
            print(f"Processed {chunk_count} chunks ({duration:.1f} seconds of audio)")

if __name__ == "__main__":
    run_server()