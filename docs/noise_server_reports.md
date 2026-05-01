# Rust ONNX Noise Cancellation Server
## Build Troubleshooting & Resolution Report

*Author: Luciano Muratore | April 2026*

---

## 1. Project Overview

The goal of this project is to build a real-time audio noise cancellation pipeline using three components that communicate over the local network:

- A C++ audio capture layer that reads raw audio from the microphone, splits it into fixed-size chunks, and sends them via ZeroMQ.
- A Rust inference server that receives each chunk, runs it through a DeepFilterNet3 ONNX model, and returns the cleaned audio.
- The C++ layer again receives the cleaned audio and routes it to the audio output.

**Key technology choices:**

| Component | Choice / Version |
|-----------|-----------------|
| Language | Rust (edition 2021) |
| ONNX runtime | ort 2.0.0-rc.12 |
| Array library | ndarray 0.16 |
| Messaging | ZeroMQ via zmq 0.10 — PULL on 5555, PUSH on 5556 |
| Model | DeepFilterNet3 (deepfilternet3.onnx) |
| Frame size | 1536 f32 samples per chunk (~34 ms at 44100 Hz) |
| Serialisation | Raw little-endian f32 bytes |

---

## 2. Architecture

The Rust server performs the following steps in a tight loop:

1. Receive 6144 bytes (1536 × 4) from the C++ PULL socket.
2. Deserialise the bytes into a `Vec<f32>`.
3. Build an ONNX input tensor with shape `[1, 1536]`.
4. Run inference through the DeepFilterNet3 model.
5. Extract the output tensor and re-serialise it to bytes.
6. Send the clean audio bytes back via the PUSH socket.
7. Log throughput statistics every 50 chunks.

---

## 3. Build Errors and Fixes

The original code was drafted against a draft of the `ort 2.0.0-rc.12` API. Four compiler errors appeared in sequence, each revealing the next. The table below summarises all of them.

| # | Error | Fix |
|---|-------|-----|
| E1 | `session.run([...])` — array literal not accepted by `SessionInputs` | Changed to `session.run(vec![...])` |
| E2 | `try_extract_raw_tensor::<f32>()` — method does not exist in this version | Changed to `try_extract_tensor::<f32>()` |
| E3 | `input_tensor.into()` — ambiguous: multiple `Into` impls, compiler cannot resolve | Added `.into_dyn()` to erase the typed `Value` to a plain `Value` before passing it |
| E4 | `session.run()` requires `&mut self` but session was not declared mutable | Changed `let session` to `let mut session` |

---

## 4. Detailed Error Analysis

### Error E1 — Wrong input type for session.run()

The original call passed a fixed-size array literal:

```rust
session.run([("input", input_tensor.into())])?;
```

The `ort 2.0.0-rc.12` `SessionInputs` type implements `From<Vec<(K,V)>>` and `From<HashMap<K,V>>` but not `From<[T; N]>` when `N` is inferred. Fix:

```rust
session.run(vec![("input", input_tensor.into())])?;
```

### Error E2 — Non-existent method try_extract_raw_tensor

The method name used in the original code does not exist in `ort 2.0.0-rc.12`. The compiler itself suggested the correct replacement. The fixed version returns a `(&Shape, &[f32])` tuple that is destructured directly:

```rust
let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
```

### Error E3 — Ambiguous .into() conversion

After fixing E2, the compiler could no longer resolve which `Into` impl to use for `Value<TensorValueType<f32>>`. Calling `.into_dyn()` first erases the type parameter, producing a plain `Value` that has exactly one unambiguous impl:

```rust
let input_tensor = Tensor::<f32>::from_array((shape, noisy.into_boxed_slice()))?
    .into_dyn();  // erase typed Value -> plain Value

let outputs = session.run(vec![("input", input_tensor)])?;
```

### Error E4 — session not declared mutable

The `ort 2.0.0-rc.12` `Session::run()` takes `&mut self`, which is unusual. The binding therefore requires `mut`:

```rust
let mut session = Session::builder()?.commit_from_file(MODEL_PATH)?;
```

---

## 5. Final Inference Block

After all four fixes, the inference section compiles and runs correctly:

```rust
let shape        = [1usize, FRAMES];
let input_tensor = Tensor::<f32>::from_array((shape, noisy.into_boxed_slice()))?
    .into_dyn();

let outputs = session.run(vec![("input", input_tensor)])?;

let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
```

---

## 6. Key Lessons

1. **RC crate versions change their APIs frequently.** Always verify method signatures against the exact version's source or changelog, not general documentation.

2. **When a generic `Into` impl is ambiguous**, prefer an explicit conversion method such as `.into_dyn()` over `.into()` to give the compiler a single unambiguous path.

3. **`ort 2.0.0-rc.12` requires `&mut self` on `Session::run()`.** Declare sessions with `let mut` from the start.

4. **The Rust compiler error messages in this case were accurate and actionable** — following them directly led to the correct fix each time.

---

`cargo build` → `Finished dev profile [OK]`