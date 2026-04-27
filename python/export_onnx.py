import torch
import numpy as np
import torchaudio.transforms as T
from df.enhance import init_df, enhance
import time

# -----------------------------------------------
# Configuration
# -----------------------------------------------
SAMPLE_RATE_HW    = 44100
SAMPLE_RATE_MODEL = 48000
FRAMES            = 1536
NUM_CHUNKS        = 50

def benchmark_onnx():
    print("=== ONNX Runtime Benchmark ===")
    print(f"Chunks to test:   {NUM_CHUNKS}")
    print(f"Frames per chunk: {FRAMES}")
    print(f"Buffer period:    {FRAMES / SAMPLE_RATE_HW * 1000:.1f}ms")
    print("(audio must be processed faster than buffer period)\n")

    # 1. Load model
    print("Loading DeepFilterNet model...")
    model, df_state, _ = init_df()
    print("Model loaded!\n")

    # 2. Set up resamplers
    resampler_up   = T.Resample(SAMPLE_RATE_HW, SAMPLE_RATE_MODEL)
    resampler_down = T.Resample(SAMPLE_RATE_MODEL, SAMPLE_RATE_HW)

    # 3. Generate fake chunks
    fake_chunks = [torch.randn(1, FRAMES) for _ in range(NUM_CHUNKS)]

    # 4. Warm up
    print("Warming up...")
    warmup = resampler_up(fake_chunks[0])
    _ = enhance(model, df_state, warmup)
    print("Warm up done!\n")

    # 5. Benchmark with torch.inference_mode
    #    This is the first optimization — disables gradient
    #    tracking which PyTorch does by default even during
    #    inference. We don't need gradients for noise cancellation.
    print(f"Benchmarking {NUM_CHUNKS} chunks with inference_mode...")
    times = []

    with torch.inference_mode():
        for i, chunk in enumerate(fake_chunks):
            start = time.perf_counter()

            upsampled   = resampler_up(chunk)
            enhanced    = enhance(model, df_state, upsampled)
            downsampled = resampler_down(enhanced)

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)

            if (i + 1) % 10 == 0:
                print(f"  Chunk {i+1:3d}: {elapsed_ms:.1f}ms")

    # 6. Results
    avg_ms = np.mean(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    std_ms = np.std(times)
    budget = FRAMES / SAMPLE_RATE_HW * 1000

    print(f"\n=== Results (inference_mode) ===")
    print(f"Buffer period (budget): {budget:.1f}ms")
    print(f"Average inference:      {avg_ms:.1f}ms")
    print(f"Min inference:          {min_ms:.1f}ms")
    print(f"Max inference:          {max_ms:.1f}ms")
    print(f"Std deviation:          {std_ms:.1f}ms")
    print("")
    if avg_ms < budget:
        print(f"PASS: Fast enough for real-time! ({avg_ms:.1f}ms < {budget:.1f}ms)")
    else:
        print(f"FAIL: Too slow for real-time ({avg_ms:.1f}ms > {budget:.1f}ms)")
        print(f"      Need to be {avg_ms - budget:.1f}ms faster")

if __name__ == "__main__":
    benchmark_onnx()