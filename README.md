# NoiseCancellation

Real-time noise cancellation engine using C++ for audio I/O and Rust for AI inference.

---

## Architecture

```
[Mic] → [C++ PortAudio] → [ZeroMQ PUSH] → [Rust Server] → [ZeroMQ PULL] → [C++ Speakers]
```

The project is split into two separate processes that communicate over ZeroMQ:

- **C++ engine** — captures audio from the microphone, sends chunks to the Rust server, receives clean audio and plays it back through the headphones
- **Rust server** — receives noisy audio chunks, runs DeepFilterNet3 through a continuous ring buffer pipeline with Voice Activity Detection, sends clean audio back

---

## Project Structure

```
NoiseCancellation/
├── engine/                          ← C++ audio engine
│   ├── src/
│   │   └── main.cpp
│   ├── experiments/
│   │   ├── passthrough_1536.cpp     ← baseline experiment (E01)
│   │   └── passthrough_sleep_fix.cpp ← sleep fix experiment (E02+)
│   └── CMakeLists.txt
├── ai/
│   ├── python/
│   │   ├── server/
│   │   │   └── server_async.py      ← Python async server (reference)
│   │   ├── experiments/
│   │   │   ├── inspect_onnx_models.py ← tensor shape diagnostic tool
│   │   │   ├── benchmark.py
│   │   │   ├── verify_onnx.py
│   │   │   └── export_to_onnx.py
│   │   └── models/
│   │       └── tmp/export/          ← official DeepFilterNet3 ONNX models
│   │           ├── config.ini
│   │           ├── enc.onnx         ← encoder (1.9 MB)
│   │           ├── erb_dec.onnx     ← ERB decoder (3.3 MB)
│   │           └── df_dec.onnx     ← deep filter decoder (3.3 MB)
│   └── rust/
│       ├── src/
│       │   ├── main.rs              ← Phase 5: per-chunk pipeline (reference)
│       │   ├── stream_server.rs     ← Phase 6: continuous ring buffer pipeline
│       │   └── stream_server_vad.rs ← Phase 7: ring buffer + VAD (current)
│       └── Cargo.toml
├── results/
│   ├── benchmark/                   ← inference timing benchmarks
│   ├── experiment/                  ← early experiment results
│   ├── phase4/                      ← Phase 4 terminal outputs
│   ├── phase5/                      ← Phase 5 terminal outputs
│   └── phase6/                      ← Phase 6 terminal outputs
├── docs/
│   ├── noise_server_reports.md      ← Rust ONNX build troubleshooting report
│   ├── project_assessment.md        ← technical assessment: improvements and issues
│   └── EXPERIMENTS.md               ← full experiment log
├── Noise_Monitor.html               ← real-time dual-channel audio visualizer
├── EXPERIMENTS.md                   ← experiment log (root copy)
└── .github/workflows/ci.yml        ← CI/CD pipeline
```

---

## Phases

### Phase 1 — C++ Audio I/O ✓
Real-time audio capture and playback using PortAudio.
- MME devices: Input `[1]` Microfoon (Realtek), Output `[4]` Headphones (WH-CH720N Stereo)
- Sample rate: 44100 Hz | Buffer: 1536 frames | Latency: 34.8ms

### Phase 2 — Python AI Server ✓
DeepFilterNet3 noise cancellation running in Python.
- Model: DeepFilterNet3 (PyTorch)
- Resampling: 44100 Hz ↔ 48000 Hz via torchaudio
- Drop rate: ~1% (GIL pauses) | Avg inference: ~25ms | Audio: 3/5

### Phase 3 — ZeroMQ Bridge ✓
Connecting C++ and Python via ZeroMQ PUSH/PULL sockets.
- Pattern: PUSH/PULL (non-blocking async)
- Ports: 5555 (noisy audio) / 5556 (clean audio)
- Sleep precision fix: milliseconds → microseconds (47% → 1% drop rate)

### Phase 4 — Rust ONNX Server ✓
First Rust inference server using ONNX Runtime.
- Drop rate: 0% | Avg inference: 0.95ms (26x faster than Python)
- Issue: broken single-model export produced white noise

### Phase 5 — Three-Model Pipeline ✓
Orchestrating all three official DeepFilterNet3 ONNX models.
- Models: `enc.onnx` → `erb_dec.onnx` → `df_dec.onnx`
- Tensor shapes confirmed via `inspect_onnx_models.py`
- Drop rate: ~0.06% | Avg inference: ~12ms | Audio: 4/5
- Issue: chunk boundary glitches, ~15% voice loss from resampling mismatch

### Phase 6 — Continuous Stream Pipeline ✓
Replacing per-chunk processing with continuous ring buffers.
- Eliminates chunk boundary glitches and voice loss completely
- `proc_buf_48` accumulates 48 kHz samples across chunk boundaries
- `output_buf_44` accumulates downsampled output until 1536 samples ready
- Drop rate: ~0% | Avg inference: ~11ms | Audio: 4/5
- Key discovery: `lsnr` encoder output is a reliable VAD signal

### Phase 7 — Voice Activity Detection ✓ (current)
Using the encoder's `lsnr` output as a VAD gate.
- Silence (lsnr < threshold) → output silence, skip decoders
- Voice (lsnr ≥ threshold) → run full pipeline
- Threshold: 3 dB | Hold-off: 75 frames (750 ms)
- Background noise during silence: eliminated
- Binary: `stream_server_vad.exe`

---

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Audio I/O | PortAudio (C++) | 19.7 |
| AI Model | DeepFilterNet3 | ONNX export |
| Messaging | ZeroMQ | 4.3.5 |
| Build | CMake + MSVC | 3.29 / VS2022 |
| Rust inference | ort (ONNX Runtime) | 2.0.0-rc.12 |
| Rust DSP | deep_filter | 0.2.5 |
| Rust resampling | rubato | 0.14.1 |

---

## Setup

### Prerequisites
- Windows 10/11
- Visual Studio 2022 (with C++ workload)
- CMake 3.20+
- Rust 1.95+
- PortAudio DLL at `C:\dev\portaudio\`
- ZeroMQ at `C:\dev\zeromq\`

### C++ Engine

```cmd
cd engine
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Debug
```

### Rust Server

```cmd
cd ai\rust
cargo build --release
```

This builds three binaries:
- `noise_server.exe` — Phase 5 reference
- `stream_server.exe` — Phase 6 continuous stream
- `stream_server_vad.exe` — Phase 7 with VAD (current)

### ONNX Models

Download the official DeepFilterNet3 ONNX models and place them at:
```
ai/python/models/tmp/export/enc.onnx
ai/python/models/tmp/export/erb_dec.onnx
ai/python/models/tmp/export/df_dec.onnx
```

To inspect tensor names and shapes:
```cmd
cd ai\python
venv\Scripts\activate
python experiments\inspect_onnx_models.py
```

---

## Running the Pipeline

**Terminal 1 — Rust server (Phase 7):**
```cmd
cd C:\Users\Luciano Muratore\NoiseCancellation
ai\rust\target\release\stream_server_vad.exe
```

**Terminal 2 — C++ engine:**
```cmd
cd C:\Users\Luciano Muratore\NoiseCancellation
engine\build\Debug\SleepFixTest.exe
```

**Debug mode (lsnr logging):**
```cmd
set RUST_LOG=debug
ai\rust\target\release\stream_server_vad.exe
```

---

## Real-Time Visualizer

Open `Noise_Monitor.html` in Chrome or Edge while the pipeline is running.

- **Left panel (red):** microphone input — the noisy signal
- **Right panel (green):** headphone output via Stereo Mix — the clean signal
- **VAD indicator:** gate open/closed state and energy level
- **Live metrics:** RMS levels, suppression in dB, peak frequency, gate state

**Setup:** Enable Stereo Mix in Control Panel → Sound → Recording → Show Disabled Devices → right-click Stereo Mix → Enable.

---

## Audio Devices (Windows)

```
Input:  [1] Microfoon (Realtek Audio)      ← laptop built-in mic
Output: [4] Headphones (WH-CH720N Stereo)  ← Bluetooth headphones (stereo mode)
```

**Important:** Using the Bluetooth headset mic forces Windows into Hands-Free Profile (HFP) mode which mutes the stereo output. Always use the Realtek mic as input.

---

## Experiment Results

See `EXPERIMENTS.md` for full experiment log.

| ID | Experiment | Drop rate | Inference | Audio |
|----|------------|-----------|-----------|-------|
| E01 | Baseline (ms sleep) | 47% | ~25ms | 2/5 |
| E02 | Sleep Fix (us sleep) | ~1% | ~25ms | 3/5 |
| E03 | Rust ONNX (broken model) | 0% | 0.95ms | 1/5 |
| E04 | Three-Model Pipeline | ~0.06% | ~12ms | 4/5 |
| E05 | Continuous Stream | ~0% | ~11ms | 4/5 |
| E06 | Voice Activity Detection | ~0% | ~11ms | 4/5+ |

---

## Known Issues

- **Feature extraction mismatch:** `feat_erb` and `feat_cplx` are hand-rolled approximations of the DeepFilterNet training preprocessing. High-frequency consonants (s, sh, f, t) are muffled as a result. Fix: port the exact preprocessing from `df/features.py`.
- **Fixed VAD threshold:** the 3 dB threshold was tuned for one specific room and microphone. It will need retuning in different environments.
- **Bluetooth HFP conflict:** using the Bluetooth headset mic triggers Windows to switch headphones to Hands-Free mode, muting stereo output.
- **Model files not in repo:** ONNX model files must be downloaded separately and placed in `ai/python/models/tmp/export/`.

---

## Repository

https://github.com/Dextromethorpan/Noise_Cancellation