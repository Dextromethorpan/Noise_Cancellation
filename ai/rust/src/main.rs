use anyhow::Result;
use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

const FRAMES: usize    = 1536;
const PULL_ADDR: &str  = "tcp://127.0.0.1:5555";
const PUSH_ADDR: &str  = "tcp://127.0.0.1:5556";
const MODEL_PATH: &str = "ai/python/models/deepfilternet3.onnx";
const LOG_EVERY: u64   = 50;

fn main() -> Result<()> {
    env_logger::init();
    println!("=== Rust ONNX Noise Server ===");
    println!("Model:  {}", MODEL_PATH);
    println!("Frames: {}", FRAMES);
    println!("Pull:   {}", PULL_ADDR);
    println!("Push:   {}", PUSH_ADDR);
    println!();

    // 1. Load ONNX model
    println!("Loading ONNX model...");
    let start   = Instant::now();
    let mut session = Session::builder()?
        .commit_from_file(MODEL_PATH)?;
    println!("Model loaded in {:.1}ms\n", start.elapsed().as_millis());

    // 2. Set up ZeroMQ
    println!("Setting up ZeroMQ sockets...");
    let context     = zmq::Context::new();
    let pull_socket = context.socket(zmq::PULL)?;
    let push_socket = context.socket(zmq::PUSH)?;

    pull_socket.bind(PULL_ADDR)?;
    push_socket.bind(PUSH_ADDR)?;

    println!("PULL socket bound to {}", PULL_ADDR);
    println!("PUSH socket bound to {}", PUSH_ADDR);
    println!("\nWaiting for audio chunks from C++...");
    println!("Press Ctrl+C to stop.\n");

    // 3. Main processing loop
    let mut chunk_count: u64        = 0;
    let mut total_inference_ms: f64 = 0.0;

    loop {
        // a. Receive noisy audio from C++
        let message = pull_socket.recv_bytes(0)?;

        if message.len() != FRAMES * 4 {
            eprintln!(
                "Unexpected message size: {} bytes (expected {})",
                message.len(), FRAMES * 4
            );
            continue;
        }

        // b. Deserialize bytes to f32 Vec
        let noisy: Vec<f32> = message
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // c. Run ONNX inference
        let inference_start = Instant::now();

        // Build tensor and erase the type with into_dyn() so the compiler
        // can resolve the Into<SessionInputValue> impl unambiguously
        let shape        = [1usize, FRAMES];
        let input_tensor = Tensor::<f32>::from_array((shape, noisy.into_boxed_slice()))?
            .into_dyn();

        let outputs = session.run(vec![("input", input_tensor)])?;

        // try_extract_tensor::<f32>() returns (&Shape, &[f32])
        let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;

        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        total_inference_ms += inference_ms;

        // d. Serialize clean audio to bytes
        let clean: Vec<u8> = output_data
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        // e. Send clean audio to C++
        push_socket.send(&clean, 0)?;

        // f. Log progress
        chunk_count += 1;
        if chunk_count % LOG_EVERY == 0 {
            let avg_ms     = total_inference_ms / chunk_count as f64;
            let duration_s = chunk_count * FRAMES as u64 / 44100;
            println!(
                "Chunks: {} | Duration: {}s | Avg inference: {:.2}ms",
                chunk_count, duration_s, avg_ms
            );
        }
    }
}