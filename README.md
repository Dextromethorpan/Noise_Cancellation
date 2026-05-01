# NoiseCancellation

Real-time noise cancellation engine using C++ for audio I/O and Python/Rust for AI inference.

---

## Architecture

```
[Mic] → [C++ PortAudio] → [ZeroMQ PUSH] → [AI Server] → [ZeroMQ PULL] → [C++ Speakers]
```

The project is split into two separate processes that communicate over ZeroMQ:

- **C++ engine** — captures audio from the microphone, sends chunks to the AI server, receives clean audio and plays it back through the speakers
- **AI server** — receives noisy audio chunks, runs DeepFilterNet3 for noise cancellation, sends clean audio back

---

## Project Structure

```
NoiseCancellation/
├── engine/                        ← C++ audio engine
│   ├── src/
│   │   └── main.cpp               ← main async pipeline
│   ├── experiments/               ← experiment executables
│   │   ├── passthrough_test.cpp   ← basic audio passthrough
│   │   ├── passthrough_1536.cpp   ← baseline experiment (E01)
│   │   └── passthrough_sleep_fix.cpp ← sleep fix experiment (E02/E03)
│   └── CMakeLists.txt
├── ai/
│   ├── python/
│   │   ├── server/
│   │   │   ├── server.py          ← blocking REQ/REP server (Phase 3)
│   │   │   └── server_async.py    ← async PUSH/PULL server (Phase 4)
│   │   ├── experiments/
│   │   │   ├── benchmark.py       ← PyTorch inference benchmark
│   │   │   ├── verify_onnx.py     ← ONNX model verification
│   │   │   └── export_to_onnx.py  ← ONNX export script
│   │   ├── models/
│   │   │   └── tmp/export/        ← official DeepFilterNet3 ONNX models
│   │   │       ├── config.ini
│   │   │       ├── enc.onnx       ← encoder (1.9 MB)
│   │   │       ├── erb_dec.onnx   ← ERB decoder (3.3 MB)
│   │   │       └── df_dec.onnx    ← deep filter decoder (3.3 MB)
│   │   └── requirements.txt
│   └── rust/
│       ├── src/
│       │   └── main.rs            ← Rust ONNX inference server
│       └── Cargo.toml
├── results/
│   ├── benchmark/                 ← inference timing benchmarks
│   ├── phase4/                    ← Phase 4 terminal outputs
│   └── experiment/                ← experiment results
├── docs/
│   ├── README.md                  ← this file
│   └── noise_server_report.docx   ← Rust ONNX build report
├── scripts/                       ← build and run scripts
├── EXPERIMENTS.md                 ← experiment log and results
└── .github/workflows/ci.yml       ← CI/CD pipeline
```

---

## Phases

### Phase 1 — C++ Audio I/O ✅
Real-time audio capture and playback using PortAudio.
- MME devices: Input `[1]` Microfoon (Realtek), Output `[4]` Headphones (WH-CH720N Stereo)
- Sample rate: 44100 Hz | Buffer: 1536 frames | Latency: 34.8ms
- Blog: `docs/phase1-audio-io-portaudio.md`

### Phase 2 — Python AI Model ✅
DeepFilterNet3 noise cancellation running in Python.
- Model: DeepFilterNet3 (PyTorch, ~16MB)
- Resampling: 44100 Hz ↔ 48000 Hz via torchaudio
- Blog: `docs/phase2-deepfilternet-python.md`

### Phase 3 — ZeroMQ Bridge ✅
Connecting C++ and Python via ZeroMQ REQ/REP sockets.
- Pattern: REQ/REP (blocking)
- Ports: 5555
- Blog: `docs/phase3-zeromq-bridge.md`

### Phase 4 — Optimization ✅
Async pipeline with PUSH/PULL pattern and Rust ONNX server.
- Pattern: PUSH/PULL (non-blocking async)
- Ports: 5555 (noisy audio) / 5556 (clean audio)
- Sleep precision fix: milliseconds → microseconds (47% → 1% drop rate)
- Rust server: 0% drop rate, 0.95ms inference (26x faster than Python)
- See `EXPERIMENTS.md` for full results

### Phase 5 — Rust Full Pipeline 🔲 (next)
Orchestrate three official DeepFilterNet3 ONNX models through the
deep_filter DSP pipeline in Rust for correct audio output.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Audio I/O | PortAudio (C++) | 19.7 |
| AI Model | DeepFilterNet3 | 0.5.6 |
| Bridge | ZeroMQ | 4.3.5 |
| Build | CMake + MSVC | 3.29 / VS2022 |
| Python env | venv | Python 3.11 |
| Rust inference | ort (ONNX Runtime) | 2.0.0-rc.12 |
| Rust DSP | deep_filter | 0.2.5 |

---

## Setup

### Prerequisites
- Windows 10/11
- Visual Studio 2022 (with C++ workload)
- CMake 3.20+
- Python 3.11
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

### Python Server

```cmd
cd ai\python
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Rust Server

```cmd
cd ai\rust
cargo build --release
```

---

## Running the Pipeline

### With Python server (Phase 4)

**Terminal 1:**
```cmd
cd ai\python
venv\Scripts\activate
python server\server_async.py
```

**Terminal 2:**
```cmd
engine\build\Debug\SleepFixTest.exe
```

### With Rust server (Phase 4 E03)

**Terminal 1:**
```cmd
ai\rust\target\debug\noise_server.exe
```

**Terminal 2:**
```cmd
engine\build\Debug\SleepFixTest.exe
```

---

## Audio Devices (Windows)

The project uses MME shared mode devices to avoid Bluetooth HFP conflicts:

```
Input:  [1] Microfoon (Realtek Audio)       — laptop built-in mic
Output: [4] Headphones (WH-CH720N Stereo)   — Bluetooth headphones (stereo mode)
```

**Important:** Using the Bluetooth headset mic (`[2]`) forces Windows into
Hands-Free Profile (HFP) mode which mutes the stereo output. Always use
the Realtek mic as input to keep the headphones in stereo mode.

---

## Benchmark Results

| Method | Avg latency | Min | Max | Std | Budget | Result |
|---|---|---|---|---|---|---|
| PyTorch CPU | 24.6ms | 14.0ms | 74.1ms | 9.2ms | 34.8ms | PASS |
| inference_mode | 32.9ms | 21.6ms | 139.5ms | 23.2ms | 34.8ms | PASS |
| torch.compile | 32.4ms | 15.7ms | 90.7ms | 15.2ms | 34.8ms | PASS (Windows unsupported) |
| ONNX Runtime (Python) | 0.2ms | 0.1ms | 1.9ms | 0.3ms | 34.8ms | PASS |
| ONNX Runtime (Rust) | 0.95ms | — | — | — | 34.8ms | PASS |

**Key finding:** Plain PyTorch gives the best consistent results for the
Python server. ONNX Runtime is 125x faster but requires correct model export.

---

## Experiment Results

See `EXPERIMENTS.md` for detailed experiment log.

| ID | Experiment | Drop rate | Inference | Audio quality |
|---|---|---|---|---|
| E01 | Baseline (ms sleep) | 47% | ~25ms | 2/5 |
| E02 | Sleep Fix (us sleep) | ~1% | ~25ms | 3/5 |
| E03 | Rust ONNX | 0% | 0.95ms | 1/5 (invalid model) |

---

## CI/CD

GitHub Actions runs on every push and pull request:
- Builds C++ engine with CMake
- Tests Python imports and DeepFilterNet model loading
- Packages release artifacts on merge to main

See `.github/workflows/ci.yml` for details.

---

## Known Issues

- **ONNX export:** `torch.jit.trace` produces an invalid 0.22 MB model
  instead of the real 8.5 MB three-model architecture. Use official
  models from `ai/python/models/tmp/export/` instead.
- **Bluetooth HFP conflict:** Using the Bluetooth headset mic triggers
  Windows to switch headphones to Hands-Free mode, muting stereo output.
- **torch.compile:** Not supported on Windows as of PyTorch 2.0.1.
- **Device indices:** Windows shuffles PortAudio device indices when
  Bluetooth connects/disconnects. Use name-based device search in production.

---

## Repository

https://github.com/Dextromethorpan/Noise_Cancellation

---

