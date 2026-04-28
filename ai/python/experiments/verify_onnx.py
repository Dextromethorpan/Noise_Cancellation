import onnxruntime as ort
import numpy as np
import time

def verify_onnx_model():
    model_path = "../../ai/python/models/deepfilternet3.onnx"

    # Try multiple paths since we might run from different locations
    import os
    possible_paths = [
        "ai/python/models/deepfilternet3.onnx",
        "../models/deepfilternet3.onnx",
        "../../ai/python/models/deepfilternet3.onnx",
        os.path.join(os.path.dirname(__file__), "../models/deepfilternet3.onnx")
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("ERROR: deepfilternet3.onnx not found!")
        print("Run ai/python/experiments/export_to_onnx.py first")
        return False

    print(f"Found model at: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print()

    # 1. Load model
    print("Loading ONNX model...")
    session = ort.InferenceSession(model_path)
    print("Model loaded!")
    print()

    # 2. Print model info
    print("=== Model Inputs ===")
    for inp in session.get_inputs():
        print(f"  Name:  {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type:  {inp.type}")
    print()

    print("=== Model Outputs ===")
    for out in session.get_outputs():
        print(f"  Name:  {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type:  {out.type}")
    print()

    # 3. Test inference with dummy audio
    print("=== Inference Test ===")
    dummy_input = np.random.randn(1, 1536).astype(np.float32)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Input dtype:  {dummy_input.dtype}")

    start = time.perf_counter()
    result = session.run(None, {"input": dummy_input})
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Output shape: {result[0].shape}")
    print(f"Output dtype: {result[0].dtype}")
    print(f"Inference time: {elapsed_ms:.1f}ms")
    print()

    # 4. Benchmark — 50 chunks
    print("=== Benchmark (50 chunks) ===")
    times = []
    for i in range(50):
        chunk = np.random.randn(1, 1536).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {"input": chunk})
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    max_ms = np.max(times)
    std_ms = np.std(times)
    budget = 1536 / 44100 * 1000

    print(f"Budget (chunk period): {budget:.1f}ms")
    print(f"Average inference:     {avg_ms:.1f}ms")
    print(f"Min inference:         {min_ms:.1f}ms")
    print(f"Max inference:         {max_ms:.1f}ms")
    print(f"Std deviation:         {std_ms:.1f}ms")
    print()

    if avg_ms < budget:
        print(f"PASS: Fast enough for real-time ({avg_ms:.1f}ms < {budget:.1f}ms)")
    else:
        print(f"FAIL: Too slow for real-time ({avg_ms:.1f}ms > {budget:.1f}ms)")
        print(f"      Need {avg_ms - budget:.1f}ms improvement")

    # 5. Verify output is different from input (model is actually doing something)
    test_input  = np.random.randn(1, 1536).astype(np.float32)
    test_output = session.run(None, {"input": test_input})[0]
    is_different = not np.allclose(test_input, test_output, atol=1e-3)
    print()
    print(f"Output differs from input: {is_different}")
    if not is_different:
        print("WARNING: Output is identical to input - model may be passthrough only!")
    else:
        print("Model is transforming the audio as expected.")

    return True

if __name__ == "__main__":
    verify_onnx_model()