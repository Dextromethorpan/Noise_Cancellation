# Experiments

This file tracks every experiment run in the project.
Each entry records what was tested, the measured results,
and the subjective audio quality observed.

---

## Experiment Log

### E01 — Baseline (broken sleep)
- **Date:** 2026-04-27
- **File:** engine/experiments/passthrough_1536.cpp
- **Server:** ai/python/server/server_async.py
- **Results:** results/phase4/phase4_final_cpp.txt, results/phase4/phase4_final_python.txt
- **Frames:** 1536 | **Sleep:** milliseconds (integer truncation)
- **Drop rate:** 47%
- **Queue depth:** 10/10 (always full)
- **Avg inference:** ~25ms (Python PyTorch CPU)
- **Audio quality:** 2/5 — broken radio, words cut off mid-syllable
- **Root cause:** `FRAMES * 1000 / SAMPLE_RATE` truncates 34.8ms to 34ms
  causing C++ to send chunks ~47% faster than Python can process them.
  Queue fills instantly and stays full for entire session.

---

### E02 — Sleep Fix (microsecond precision)
- **Date:** 2026-04-28
- **File:** engine/experiments/passthrough_sleep_fix.cpp
- **Server:** ai/python/server/server_async.py
- **Results:** results/experiment/experiment_sleep_fix_cpp.txt, results/experiment/experiment_sleep_fix_python.txt
- **Frames:** 1536 | **Sleep:** microseconds (34829 us = 34.829ms)
- **Drop rate:** ~1%
- **Queue depth:** 0/10 (almost always empty)
- **Avg inference:** ~25ms (Python PyTorch CPU)
- **Audio quality:** 3/5 — works for slow speech, distortion at fast speech
- **Root cause of remaining distortion:** Python GIL pauses during heavy
  processing bursts cause occasional spikes above the 34.8ms budget.
  The microsecond sleep fix eliminated queue overflow but Python's
  non-deterministic scheduling still causes ~1% drops during GIL pauses.

---

### E03 — Rust + ONNX Runtime
- **Date:** 2026-04-28 / 2026-04-29
- **File:** engine/experiments/passthrough_sleep_fix.cpp (C++ side)
- **Server:** ai/rust/src/main.rs (Rust side)
- **Results:** results/experiment/experiment_rust_cpp.txt,
  results/experiment/experiment_rust_server.txt,
  results/experiment/experiment_rust_hearing.txt
- **Frames:** 1536 | **Sleep:** microseconds (34829 us)
- **Drop rate:** 0%
- **Queue depth:** N/A (synchronous Rust loop, no queue)
- **Avg inference:** 0.95ms (ONNX Runtime CPU)
- **Audio quality:** 1/5 — continuous white noise (shshshsh), voice not audible
- **What worked:**
  - C++ → ZeroMQ → Rust → ZeroMQ → C++ pipeline: perfect
  - 0% drop rate over 2700+ chunks: proven
  - Inference time: 0.95ms vs 25ms Python — 26x faster
  - No GIL, no garbage collector pauses
- **What failed:**
  - ONNX model export via torch.jit.trace produced a 0.22 MB file
  - Real DeepFilterNet3 should be ~8.5 MB (three models)
  - Traced model froze the RNN hidden state as constants
  - Output was the frozen state pattern = white noise
- **Key discovery:** DeepFilterNet3 is three separate ONNX models:
  - `enc.onnx` (1.9 MB) — encoder, analyzes frequency content
  - `erb_dec.onnx` (3.3 MB) — ERB decoder, predicts noise mask
  - `df_dec.onnx` (3.3 MB) — deep filter decoder, applies filtering
  - Official ONNX models downloaded from DeepFilterNet GitHub repository
- **Dependencies added:**
  - `ort = 2.0.0-rc.12` — ONNX Runtime Rust bindings
  - `deep_filter = 0.2.5` — DSP processing (STFT/ISTFT, feature extraction)
- **Next step (Phase 5):** Orchestrate the three official ONNX models
  through the deep_filter DSP pipeline in Rust for correct audio output.

---

## Comparison Summary

| ID  | Experiment      | Drop rate | Inference | Queue  | Audio  | Sleep     |
|-----|-----------------|-----------|-----------|--------|--------|-----------|
| E01 | Baseline        | 47%       | ~25ms     | 10/10  | 2/5    | ms (bug)  |
| E02 | Sleep Fix       | ~1%       | ~25ms     | 0/10   | 3/5    | us (fix)  |
| E03 | Rust ONNX       | 0%        | 0.95ms    | N/A    | 1/5    | us (fix)  |

**Key findings:**
- Sleep precision alone reduced drop rate from 47% to 1%
- Rust eliminated drops entirely (0%) and inference is 26x faster
- Audio quality regression in E03 is due to invalid model export, not the pipeline
- The pipeline architecture is proven correct — the model is the remaining challenge

---

## How to Run an Experiment

### Python server (E01, E02)
```cmd
cd ai/python
venv\Scripts\activate
python server/server_async.py
```

### Rust server (E03)
```cmd
cd C:\Users\Luciano Muratore\NoiseCancellation
ai\rust\target\debug\noise_server.exe
```

### C++ experiment
```cmd
cd engine/build/Debug
SleepFixTest.exe
```

### Save results
```cmd
chcp 65001
SleepFixTest.exe > ../../../results/experiment/experiment_name_cpp.txt
```

---

## Lessons Learned

**1. Sleep precision matters more than model speed**
The 0.8ms truncation error in `FRAMES * 1000 / SAMPLE_RATE` caused 47% packet
loss. Switching to microseconds fixed it to 1%. A trivial code change with
enormous impact on audio quality.

**2. Pipeline architecture vs model quality are separate problems**
E03 proved the Rust pipeline works perfectly (0% drops) even though the audio
quality was wrong. These are independent concerns — always test the pipeline
with known-good data before connecting the AI model.

**3. Model file size is a sanity check**
Our broken export was 0.22 MB. The real model is 8.5 MB. Checking file size
before running inference would have caught this immediately.

**4. Neural networks are not single files**
DeepFilterNet3 is three ONNX models that must be orchestrated together.
Assuming a complex model exports as a single file was wrong.

**5. Rust eliminates GIL completely**
The Python GIL caused ~1% drops even with perfect sleep timing. Rust has no
GIL and produced 0% drops. For real-time audio, Rust is the correct choice.