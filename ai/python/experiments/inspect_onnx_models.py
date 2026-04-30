"""
inspect_onnx_models.py — run once before Phase 5 Rust build.

Prints exact input / output names and shapes for all three ONNX models.
Use the output to confirm the tensor name constants in main.rs match.

Usage:
    cd NoiseCancellation
    ai\python\venv\Scripts\activate
    python ai\python\experiments\inspect_onnx_models.py
"""

import onnxruntime as ort

MODELS = {
    "enc":     "ai/python/models/tmp/export/enc.onnx",
    "erb_dec": "ai/python/models/tmp/export/erb_dec.onnx",
    "df_dec":  "ai/python/models/tmp/export/df_dec.onnx",
}


def inspect(name: str, path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({path})")
    print(f"{'='*60}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("INPUTS:")
    for inp in sess.get_inputs():
        print(f"  {inp.name:30s}  shape={inp.shape}  dtype={inp.type}")
    print("OUTPUTS:")
    for out in sess.get_outputs():
        print(f"  {out.name:30s}  shape={out.shape}  dtype={out.type}")


if __name__ == "__main__":
    for name, path in MODELS.items():
        inspect(name, path)
    print("\nDone. Copy any mismatched tensor names into ai/rust/src/main.rs.")