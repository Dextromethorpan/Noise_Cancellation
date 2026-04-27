import zmq
import numpy as np
import torch
import torchaudio.transforms as T
from df.enhance import init_df, enhance
import threading
import queue
import time

# -----------------------------------------------
# Configuration
# -----------------------------------------------
SAMPLE_RATE_HW    = 44100
SAMPLE_RATE_MODEL = 48000
FRAMES            = 1536
QUEUE_SIZE        = 10  # max chunks buffered in queue

# -----------------------------------------------
# Shared queues between receiver and sender
# -----------------------------------------------
input_queue  = queue.Queue(maxsize=QUEUE_SIZE)
output_queue = queue.Queue(maxsize=QUEUE_SIZE)

def load_model():
    """Load DeepFilterNet once at startup."""
    print("Loading DeepFilterNet model...")
    model, df_state, _ = init_df()
    print("Model loaded!")
    return model, df_state

def process_chunk(audio_chunk: np.ndarray, model, df_state,
                  resampler_up, resampler_down) -> np.ndarray:
    """
    Full denoise pipeline.
    44100 Hz in → DeepFilterNet → 44100 Hz out
    """
    audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)
    audio_48k    = resampler_up(audio_tensor)
    enhanced_48k = enhance(model, df_state, audio_48k)
    enhanced_44k = resampler_down(enhanced_48k)

    # Trim or pad to match original length
    target      = audio_chunk.shape[0]
    result      = enhanced_44k.squeeze(0).numpy()
    if len(result) > target:
        result = result[:target]
    elif len(result) < target:
        result = np.pad(result, (0, target - len(result)))
    return result

def receiver_thread(pull_socket):
    """
    Continuously receives noisy audio chunks from C++
    and puts them in the input queue.
    Runs on its own thread — never blocks the processor.
    """
    print("Receiver thread started.")
    while True:
        message     = pull_socket.recv()
        noisy_chunk = np.frombuffer(message, dtype=np.float32).copy()
        try:
            input_queue.put_nowait(noisy_chunk)
        except queue.Full:
            # If queue is full drop the oldest chunk
            # and add the new one
            try:
                input_queue.get_nowait()
            except queue.Empty:
                pass
            input_queue.put_nowait(noisy_chunk)

def processor_thread(model, df_state, resampler_up, resampler_down):
    """
    Continuously takes noisy chunks from input queue,
    processes them, and puts clean chunks in output queue.
    Runs on its own thread.
    """
    print("Processor thread started.")
    chunk_count = 0

    while True:
        # Wait for a chunk to process
        noisy_chunk = input_queue.get()

        # Process with DeepFilterNet
        clean_chunk = process_chunk(
            noisy_chunk, model, df_state,
            resampler_up, resampler_down)

        # Put clean chunk in output queue
        try:
            output_queue.put_nowait(clean_chunk)
        except queue.Full:
            # Drop oldest clean chunk if queue is full
            try:
                output_queue.get_nowait()
            except queue.Empty:
                pass
            output_queue.put_nowait(clean_chunk)

        chunk_count += 1
        if chunk_count % 50 == 0:
            print(f"Processed {chunk_count} chunks "
                  f"({chunk_count * FRAMES / SAMPLE_RATE_HW:.1f}s) "
                  f"| Queue: in={input_queue.qsize()} "
                  f"out={output_queue.qsize()}")

def sender_thread(push_socket):
    """
    Continuously takes clean chunks from output queue
    and sends them back to C++.
    Runs on its own thread.
    """
    print("Sender thread started.")
    while True:
        # Wait for a clean chunk
        clean_chunk = output_queue.get()

        # Send back to C++
        push_socket.send(clean_chunk.astype(np.float32).tobytes())

def run_server():
    """
    Async server — three threads working in parallel:
    1. Receiver  — pulls noisy audio from C++
    2. Processor — runs DeepFilterNet
    3. Sender    — pushes clean audio to C++
    """

    # 1. Load model and resamplers
    model, df_state = load_model()
    resampler_up    = T.Resample(SAMPLE_RATE_HW, SAMPLE_RATE_MODEL)
    resampler_down  = T.Resample(SAMPLE_RATE_MODEL, SAMPLE_RATE_HW)

    # 2. Set up ZeroMQ PUSH/PULL sockets
    context     = zmq.Context()

    # C++ pushes noisy audio → Python pulls here
    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind("tcp://127.0.0.1:5555")

    # Python pushes clean audio → C++ pulls here
    push_socket = context.socket(zmq.PUSH)
    push_socket.bind("tcp://127.0.0.1:5556")

    print("Async server listening:")
    print("  PULL noisy audio on tcp://127.0.0.1:5555")
    print("  PUSH clean audio on tcp://127.0.0.1:5556")
    print("Waiting for audio chunks from C++...\n")

    # 3. Start all three threads
    t1 = threading.Thread(
        target=receiver_thread,
        args=(pull_socket,),
        daemon=True)

    t2 = threading.Thread(
        target=processor_thread,
        args=(model, df_state, resampler_up, resampler_down),
        daemon=True)

    t3 = threading.Thread(
        target=sender_thread,
        args=(push_socket,),
        daemon=True)

    t1.start()
    t2.start()
    t3.start()

    print("All threads running!")
    print("Press Ctrl+C to stop.\n")

    # 4. Main thread just keeps the process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    run_server()