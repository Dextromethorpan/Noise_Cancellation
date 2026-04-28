### E03 — Rust ONNX Server
- **Date:** 2026-04-28
- **File:** engine/experiments/passthrough_sleep_fix.cpp (C++ side)
- **Server:** ai/rust/src/main.rs (Rust side)
- **Results:** results/experiment/experiment_rust_cpp.txt, results/experiment/experiment_rust_server.txt
- **Frames:** 1536 | **Sleep:** microseconds (34829 us)
- **Drop rate:** 0%
- **Queue depth:** N/A (Rust handles synchronously)
- **Avg inference:** 0.95ms
- **Audio quality:** 1/5 — continuous white noise (shshshsh), voice not audible
- **Root cause:** ONNX export captured a frozen RNN state via torch.jit.trace
  instead of the full DeepFilterNet model. The 0.22 MB file is too small
  (real model should be ~15-20 MB). The pipeline itself works perfectly —
  0% drop rate proves C++ → ZeroMQ → Rust → ZeroMQ → C++ is correct.
- **Next step:** Use deepfilterlib (native Rust crate) instead of ONNX export

---

## Comparison Summary

| ID  | Experiment        | Drop rate | Inference | Audio quality | Notes                    |
|-----|-------------------|-----------|-----------|---------------|--------------------------|
| E01 | Baseline          | 47%       | ~25ms     | 2/5           | ms sleep truncation      |
| E02 | Sleep Fix         | ~1%       | ~25ms     | 3/5           | us sleep precision       |
| E03 | Rust ONNX         | 0%        | 0.95ms    | 1/5           | Invalid ONNX export      |