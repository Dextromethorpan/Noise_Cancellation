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

### E04 — Three-Model DeepFilterNet3 Pipeline (Phase 5)
- **Date:** 2026-04-30
- **File:** engine/experiments/passthrough_sleep_fix.cpp (C++ side)
- **Server:** ai/rust/src/main.rs (Rust side — three-model pipeline)
- **Results:** results/phase5/phase5_cpp_terminal.txt,
  results/phase5/phase5_rust_terminal.txt
- **Frames:** 1536 | **Sleep:** microseconds (34829 us)
- **Drop rate:** ~0.06% (3 drops in 1800 chunks)
- **Queue depth:** N/A (synchronous Rust loop)
- **Avg inference:** ~12ms (ONNX Runtime CPU, three models)
- **Audio quality:** 4/5 — voice clearly audible, background noise suppressed,
  perceptible delay (~50ms total latency)
- **What worked:**
  - All three ONNX models loading and running correctly
  - enc.onnx → erb_dec.onnx → df_dec.onnx orchestration: correct
  - DFState STFT analysis / ISTFT synthesis: correct
  - ERB mask applied to full spectrum (481 bins)
  - Deep filter coefficients applied to lower 96 bins
  - rubato resampling 44100 ↔ 48000 Hz: correct
  - Tested with YouTube audio playing simultaneously — voice remained
    clearly separable and audible in feedback, confirming real-time
    noise separation is working
  - Pipeline stable over 1800+ chunks (~62 seconds)
- **Key discovery:** ONNX models are stateless at the I/O level — GRU
  hidden states are internal to the graph, not exposed as inputs/outputs.
  No state management required in Rust.
- **Tensor shapes confirmed by inspect_onnx_models.py:**
  - `feat_erb` → `[1, 1, 1, 32]` | `feat_spec` → `[1, 2, 1, 96]`
  - `emb` → `[1, 1, 512]` | `e0-e3` → `[1, 64, 1, 32/16/8/8]`
  - `c0` → `[1, 64, 1, 96]` | `m` → `[1, 1, 1, 32]`
  - `coefs` → `[1, 1, 96, 10]` (10 = 5 taps × {re, im})
- **Remaining issues:**
  - Latency ~50ms (C++ buffer 34.8ms + inference 12ms + resampling ~2ms)
  - Chunk boundary glitches — shshshsh artifact at every 35ms boundary
  - ~15% voice loss due to resampling mismatch (zeros padding output)
- **Dependencies added:**
  - `rubato = 0.14.1` — high-quality resampling
  - `ndarray = 0.15.6` — array operations
  - `configparser = 2.1.0` — model config parsing

---

### E05 — Continuous Stream Pipeline (Phase 6)
- **Date:** 2026-04-30
- **File:** engine/experiments/passthrough_sleep_fix.cpp (C++ side)
- **Server:** ai/rust/src/stream_server.rs (Rust side — continuous stream)
- **Results:** results/phase6/phase6_cpp_terminal.txt,
  results/phase6/phase6_rust_terminal.txt
- **Frames:** 1536 | **Sleep:** microseconds (34829 us)
- **Drop rate:** ~0% (stable over 8250+ chunks)
- **Queue depth:** N/A (synchronous Rust loop)
- **Avg inference:** ~11ms (ONNX Runtime CPU, three models)
- **Audio quality:** 4/5 — chunk boundary glitches eliminated, voice
  continuous and complete, some residual background noise remains
- **What worked:**
  - Continuous ring buffer architecture: no resets at chunk boundaries
  - proc_buf_48 and output_buf_44 flow seamlessly across chunks
  - Long vowel test ("aaaaa" 2-3 seconds): all sound heard, no dropout
  - All five words heard clearly in "uno dos tres cuatro cinco" test
  - Pipeline stable over 8250+ chunks (~4.7 minutes)
  - proc_buf always 0 — no accumulation backlog
  - out_buf oscillates 600-1500 — healthy steady-state flow
- **Key discovery — lsnr signal analysis:**
  - Encoder `lsnr` output IS a reliable voice activity signal
  - Silence → lsnr clusters at -10 to -15 dB consistently
  - Speaking → lsnr clusters at +20 to +35 dB consistently
  - Earlier tests appeared random because room noise was present
  - Clean separation confirmed with 2-min silence / 2-min speech / 2-min
    silence test at chunk=8150-8250
  - Threshold of +10 dB cleanly separates voice from silence
- **Remaining issues:**
  - Residual background noise when not speaking (room noise leaks through)
  - High-frequency consonants (agudos: s, sh, f, t) are muffled
  - Low-frequency sounds (graves) reproduce correctly
  - Root cause: hand-rolled feat_erb / feat_cplx normalisation does not
    exactly match the training preprocessing — model receives imprecise
    features and produces weak masks
- **Next step (Phase 7):** Implement lsnr-based Voice Activity Detection
  — output silence when lsnr < 10 dB, pass model output when lsnr ≥ 10 dB

---

## Comparison Summary

| ID  | Experiment              | Drop rate | Inference | Queue  | Audio  | Sleep     |
|-----|-------------------------|-----------|-----------|--------|--------|-----------|
| E01 | Baseline                | 47%       | ~25ms     | 10/10  | 2/5    | ms (bug)  |
| E02 | Sleep Fix               | ~1%       | ~25ms     | 0/10   | 3/5    | us (fix)  |
| E03 | Rust ONNX               | 0%        | 0.95ms    | N/A    | 1/5    | us (fix)  |
| E04 | Three-Model Pipeline    | ~0.06%    | ~12ms     | N/A    | 4/5    | us (fix)  |
| E05 | Continuous Stream       | ~0%       | ~11ms     | N/A    | 4/5    | us (fix)  |

**Key findings:**
- Sleep precision alone reduced drop rate from 47% to 1%
- Rust eliminated drops entirely (0%) and inference is 26x faster
- Audio quality regression in E03 is due to invalid model export, not the pipeline
- E04 confirms the three-model orchestration is correct — real noise suppression achieved
- E05 eliminates chunk boundary glitches via continuous ring buffers
- lsnr output from encoder is a reliable VAD signal — threshold at +10 dB
- Remaining quality issues trace to feature extraction mismatch with training

---

## How to Run an Experiment

### Python server (E01, E02)
```cmd
cd ai/python
venv\Scripts\activate
python server/server_async.py
```

### Rust server — Phase 5 (E03, E04)
```cmd
cd C:\Users\Luciano Muratore\NoiseCancellation
ai\rust\target\release\noise_server.exe
```

### Rust server — Phase 6 (E05)
```cmd
cd C:\Users\Luciano Muratore\NoiseCancellation
ai\rust\target\release\stream_server.exe
```

### Rust server — debug mode (lsnr logging)
```cmd
set RUST_LOG=debug
ai\rust\target\release\stream_server.exe
```

### C++ engine
```cmd
cd engine/build/Debug
SleepFixTest.exe
```

### Save results
```cmd
chcp 65001
SleepFixTest.exe > ../../../results/phase6/phase6_cpp_terminal.txt
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

**6. Always inspect ONNX tensor names before writing inference code**
The tensor names and shapes in the official DeepFilterNet3 export do not match
the Python source variable names one-to-one. Running inspect_onnx_models.py
before writing any Rust inference code saved significant debugging time and
revealed that GRU states are internal to the graph — a fact not documented
anywhere in the DeepFilterNet repository.

**7. Chunk boundary resets corrupt stateful DSP**
The STFT overlap-add buffers inside DFState are stateful. Resampling and
processing audio in isolated per-chunk calls resets this state at every
boundary, producing glitches. The fix is a continuous ring buffer that
feeds the DSP pipeline without interruption.

**8. Diagnose model outputs before assuming feature extraction is wrong**
The lsnr signal appeared unreliable in short tests because room noise was
always present. A structured silence/speech/silence test over several minutes
revealed that lsnr cleanly separates voice from silence at a +10 dB threshold.
Always test with controlled conditions before concluding a signal is broken.