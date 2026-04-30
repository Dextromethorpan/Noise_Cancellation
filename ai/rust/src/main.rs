//! Phase 5 — DeepFilterNet3 three-model ONNX pipeline in Rust
//!
//! Tensor shapes confirmed by inspect_onnx_models.py:
//!
//!   enc inputs:
//!     feat_erb   [1, 1, S, 32]
//!     feat_spec  [1, 2, S, 96]
//!   enc outputs:
//!     e0         [1, 64, S, 32]
//!     e1         [1, 64, S, 16]
//!     e2         [1, 64, S, 8]
//!     e3         [1, 64, S, 8]
//!     emb        [1, S, 512]
//!     c0         [1, 64, S, 96]
//!     lsnr       [1, S, 1]
//!
//!   erb_dec inputs:  emb, e3, e2, e1, e0
//!   erb_dec outputs: m [batch, 1, S, 32]
//!
//!   df_dec inputs:   emb, c0
//!   df_dec outputs:  coefs [batch, S, nb_df, 10]
//!                    (10 = DF_ORDER*2 = 5 taps × {re, im})
//!                    235   (ignored — internal sigmoid output)
//!
//! NOTE: No RNN state inputs/outputs — GRU states are internal to the graph.

use anyhow::{Context, Result};
use df::{Complex32, DFState};
use log::{debug, info, warn};
use ort::{session::Session, value::Tensor};
use rubato::{FftFixedIn, Resampler};
use std::collections::VecDeque;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Frames per ZeroMQ chunk from C++ engine (@ 44100 Hz)
const FRAMES_44: usize = 1536;

/// DeepFilterNet3 sample rate
const SR_48: usize = 48_000;
/// FFT window size (20 ms @ 48 kHz)
const FFT_SIZE: usize = 960;
/// Hop / frame size (10 ms @ 48 kHz)
const HOP_SIZE: usize = 480;
/// Complex frequency bins = FFT_SIZE / 2 + 1
const FREQ_SIZE: usize = FFT_SIZE / 2 + 1; // 481
/// ERB bands
const NB_ERB: usize = 32;
/// DF frequency bins (lower ~4.8 kHz)
const NB_DF: usize = 96;
/// DF filter taps
const DF_ORDER: usize = 5;
/// Lookahead frames for rolling buffer
const LOOKAHEAD: usize = 2;

/// ZeroMQ addresses
const PULL_ADDR: &str = "tcp://127.0.0.1:5555";
const PUSH_ADDR: &str = "tcp://127.0.0.1:5556";

/// Model paths — relative to NoiseCancellation/ workspace root
const ENC_PATH:     &str = "ai/python/models/tmp/export/enc.onnx";
const ERB_DEC_PATH: &str = "ai/python/models/tmp/export/erb_dec.onnx";
const DF_DEC_PATH:  &str = "ai/python/models/tmp/export/df_dec.onnx";

/// Print stats every N chunks
const LOG_EVERY: u64 = 50;

// ──────────────────────────────────────────────────────────────────────────────
// Resampler helpers
// ──────────────────────────────────────────────────────────────────────────────

fn make_resampler(from_sr: usize, to_sr: usize, chunk: usize) -> Result<FftFixedIn<f32>> {
    FftFixedIn::<f32>::new(from_sr, to_sr, chunk, 2, 1)
        .context("Failed to create resampler")
}

fn resample(r: &mut FftFixedIn<f32>, samples: &[f32]) -> Result<Vec<f32>> {
    let out = r.process(&[samples.to_vec()], None)?;
    Ok(out.into_iter().next().unwrap_or_default())
}

// ──────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Exponential mean normalisation (used for ERB log-power features).
/// decay_frames ≈ SR / HOP_SIZE = 100 frames per second.
struct MeanNorm {
    mean: f32,
    alpha: f32,
}
impl MeanNorm {
    fn new(decay_frames: f32) -> Self {
        Self { mean: 0.0, alpha: (-1.0_f32 / decay_frames).exp() }
    }
    fn norm(&mut self, x: f32) -> f32 {
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x;
        x - self.mean
    }
}

/// Exponential unit normalisation (used for complex DF features).
struct UnitNorm {
    scale: f32,
    alpha: f32,
}
impl UnitNorm {
    fn new(decay_frames: f32) -> Self {
        Self { scale: 1.0, alpha: (-1.0_f32 / decay_frames).exp() }
    }
    fn norm(&mut self, x: f32) -> f32 {
        self.scale = self.alpha * self.scale + (1.0 - self.alpha) * x.abs();
        x / self.scale.max(1e-12)
    }
}

/// ERB log-power features → [NB_ERB] normalised values.
fn feat_erb(
    spec: &[Complex32],
    erb_bins: &[usize],  // last freq bin for each ERB band
    norm: &mut MeanNorm,
    out: &mut [f32; NB_ERB],
) {
    let mut bin = 0usize;
    for (b, &end) in erb_bins.iter().enumerate() {
        let start = bin;
        while bin <= end && bin < FREQ_SIZE {
            bin += 1;
        }
        let width = (bin - start).max(1) as f32;
        let power: f32 = spec[start..bin.min(FREQ_SIZE)]
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f32>() / width;
        let logp = (power.max(1e-12)).log10() * 10.0; // dB
        out[b] = norm.norm(logp);
    }
}

/// Complex features → [NB_DF] real, [NB_DF] imag (unit-norm each).
fn feat_cplx(
    spec: &[Complex32],
    norm: &mut UnitNorm,
    out_re: &mut [f32; NB_DF],
    out_im: &mut [f32; NB_DF],
) {
    for i in 0..NB_DF {
        out_re[i] = norm.norm(spec[i].re);
        out_im[i] = norm.norm(spec[i].im);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tensor construction helpers (ort 2.0.0-rc.12)
// ──────────────────────────────────────────────────────────────────────────────

fn make_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>> {
    Tensor::from_array((shape.to_vec(), data.into_boxed_slice()))
        .context("Tensor::from_array failed")
}

fn extract_f32(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Vec<f32>> {
    let (_, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .with_context(|| format!("extract '{name}'"))?;
    Ok(data.to_vec())
}

// ──────────────────────────────────────────────────────────────────────────────
// Spectrum post-processing
// ──────────────────────────────────────────────────────────────────────────────

/// Apply ERB-scale mask to full spectrum.
/// `m`: [NB_ERB] gains in [0,1].  `erb_bins`: last freq bin per band.
fn apply_erb_mask(spec: &mut [Complex32], m: &[f32], erb_bins: &[usize]) {
    let mut bin = 0usize;
    for (b, &end) in erb_bins.iter().enumerate() {
        let gain = m[b].clamp(0.0, 1.0);
        while bin <= end && bin < FREQ_SIZE {
            spec[bin] = Complex32::new(spec[bin].re * gain, spec[bin].im * gain);
            bin += 1;
        }
    }
}

/// Apply DF filter to lower NB_DF bins.
///
/// `coefs`: flat [NB_DF × DF_ORDER × 2] — layout matches ONNX output
///          `[batch=1, S=1, NB_DF, 10]` ravelled with NB_DF outer.
/// `rolling`: ring buffer of past noisy spectra, newest at back.
fn apply_df_filter(
    spec: &mut [Complex32],
    coefs: &[f32],
    rolling: &VecDeque<Vec<Complex32>>,
) {
    let n = rolling.len();
    for f in 0..NB_DF {
        let mut out = Complex32::new(0.0, 0.0);
        for tap in 0..DF_ORDER {
            // tap 0 = most recent frame
            if tap >= n { break; }
            let buf_idx = n - 1 - tap;
            let frame = &rolling[buf_idx];
            // coefs layout: [NB_DF, DF_ORDER, 2]  →  index = (f * DF_ORDER + tap) * 2
            let ci = (f * DF_ORDER + tap) * 2;
            if ci + 1 >= coefs.len() { break; }
            let w = Complex32::new(coefs[ci], coefs[ci + 1]);
            out += w * frame[f];
        }
        spec[f] = out;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main processor
// ──────────────────────────────────────────────────────────────────────────────

struct NoiseProcessor {
    df_state:   DFState,
    enc:        Session,
    erb_dec:    Session,
    df_dec:     Session,
    erb_norm:   MeanNorm,
    cplx_norm:  UnitNorm,
    /// Ring buffer of NOISY spectra (NB_DF bins each), newest at back.
    rolling:    VecDeque<Vec<Complex32>>,
    erb_bins:   Vec<usize>,
    up:         FftFixedIn<f32>,
    down:       FftFixedIn<f32>,
    leftover:   Vec<f32>,
}

impl NoiseProcessor {
    fn new() -> Result<Self> {
        info!("Loading enc.onnx …");
        let enc = Session::builder()?.commit_from_file(ENC_PATH)
            .with_context(|| format!("Cannot open {ENC_PATH}"))?;
        info!("Loading erb_dec.onnx …");
        let erb_dec = Session::builder()?.commit_from_file(ERB_DEC_PATH)
            .with_context(|| format!("Cannot open {ERB_DEC_PATH}"))?;
        info!("Loading df_dec.onnx …");
        let df_dec = Session::builder()?.commit_from_file(DF_DEC_PATH)
            .with_context(|| format!("Cannot open {DF_DEC_PATH}"))?;
        info!("All models loaded.");

        // DFState: sr=48000, fft=960, hop=480, nb_erb=32, min_nb_freqs=1
        let df_state = DFState::new(SR_48, FFT_SIZE, HOP_SIZE, NB_ERB, 1);
        let erb_bins = df_state.erb.clone();

        // 100 frames ≈ 1 second of decay
        let erb_norm  = MeanNorm::new(100.0);
        let cplx_norm = UnitNorm::new(100.0);

        let rolling = VecDeque::with_capacity(DF_ORDER + LOOKAHEAD);

        // Up-sampler: 1536 samples at 44100 → ~1672 samples at 48000
        let up   = make_resampler(44_100, SR_48,  FRAMES_44)?;
        // Down-sampler: 480 samples at 48000 → ~436 samples at 44100
        let down = make_resampler(SR_48,  44_100, HOP_SIZE)?;

        Ok(Self { df_state, enc, erb_dec, df_dec,
                  erb_norm, cplx_norm, rolling, erb_bins,
                  up, down, leftover: Vec::new() })
    }

    /// Process one 1536-sample @ 44100 Hz chunk → 1536-sample @ 44100 Hz.
    fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // 1. Upsample 44100 → 48000
        let upsampled = resample(&mut self.up, input)?;
        let mut buf = std::mem::take(&mut self.leftover);
        buf.extend_from_slice(&upsampled);

        // 2. Process complete 480-sample hops
        let mut out_48: Vec<f32> = Vec::with_capacity(buf.len());
        let mut pos = 0;
        while pos + HOP_SIZE <= buf.len() {
            let frame_out = self.process_frame(&buf[pos..pos + HOP_SIZE])?;
            out_48.extend_from_slice(&frame_out);
            pos += HOP_SIZE;
        }
        self.leftover = buf[pos..].to_vec();

        if out_48.is_empty() {
            return Ok(vec![0.0f32; FRAMES_44]);
        }

        // 3. Downsample 48000 → 44100, frame-by-frame
        let mut out_44: Vec<f32> = Vec::new();
        for hop in out_48.chunks(HOP_SIZE) {
            if hop.len() == HOP_SIZE {
                out_44.extend_from_slice(&resample(&mut self.down, hop)?);
            }
        }
        out_44.resize(FRAMES_44, 0.0);
        Ok(out_44)
    }

    /// Process one 480-sample frame at 48 kHz → 480-sample enhanced frame.
    fn process_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        // ── a. STFT analysis ───────────────────────────────────────────────
        let mut spec = vec![Complex32::new(0.0, 0.0); FREQ_SIZE];
        self.df_state.analysis(frame, &mut spec);

        // Save noisy lower bins for DF rolling buffer
        {
            let noisy_slice: Vec<Complex32> = spec[..NB_DF].to_vec();
            if self.rolling.len() >= DF_ORDER + LOOKAHEAD {
                self.rolling.pop_front();
            }
            self.rolling.push_back(noisy_slice);
        }

        // ── b. ERB features → [1, 1, 1, 32] ──────────────────────────────
        let mut erb_feat = [0.0f32; NB_ERB];
        feat_erb(&spec, &self.erb_bins, &mut self.erb_norm, &mut erb_feat);
        let feat_erb_t = make_tensor(erb_feat.to_vec(), &[1, 1, 1, NB_ERB])?;

        // ── c. Complex features → [1, 2, 1, 96] ──────────────────────────
        let mut re = [0.0f32; NB_DF];
        let mut im = [0.0f32; NB_DF];
        feat_cplx(&spec, &mut self.cplx_norm, &mut re, &mut im);
        let mut spec_data = re.to_vec();
        spec_data.extend_from_slice(&im);
        let feat_spec_t = make_tensor(spec_data, &[1, 2, 1, NB_DF])?;

        // ── d. Encoder ────────────────────────────────────────────────────
        let enc_out = self.enc.run(ort::inputs![
            "feat_erb"  => feat_erb_t,
            "feat_spec" => feat_spec_t,
        ])?;

        // emb [1, S=1, 512] — flatten to [1, 1, 512] for decoders
        let emb  = extract_f32(&enc_out, "emb")?;   // len = 512
        let e0   = extract_f32(&enc_out, "e0")?;    // len = 1*64*1*32 = 2048
        let e1   = extract_f32(&enc_out, "e1")?;    // len = 1*64*1*16 = 1024
        let e2   = extract_f32(&enc_out, "e2")?;    // len = 1*64*1*8  = 512
        let e3   = extract_f32(&enc_out, "e3")?;    // len = 1*64*1*8  = 512
        let c0   = extract_f32(&enc_out, "c0")?;    // len = 1*64*1*96 = 6144
        let lsnr = extract_f32(&enc_out, "lsnr")?;  // len = 1

        let lsnr_val = lsnr.first().copied().unwrap_or(0.0);
        debug!("lsnr = {lsnr_val:.1} dB");

        // ── e. ERB decoder ────────────────────────────────────────────────
        // Inputs must match inspector shapes exactly:
        //   emb  [1, S=1, 512]
        //   e3   [1, 64, S=1, 8]
        //   e2   [1, 64, S=1, 8]
        //   e1   [1, 64, S=1, 16]
        //   e0   [1, 64, S=1, 32]
        let erb_out = self.erb_dec.run(ort::inputs![
            "emb" => make_tensor(emb.clone(), &[1, 1, 512])?,
            "e3"  => make_tensor(e3,          &[1, 64, 1, 8])?,
            "e2"  => make_tensor(e2,          &[1, 64, 1, 8])?,
            "e1"  => make_tensor(e1,          &[1, 64, 1, 16])?,
            "e0"  => make_tensor(e0,          &[1, 64, 1, 32])?,
        ])?;

        // m [batch, 1, S=1, 32] — we only need the 32 gain values
        let m = extract_f32(&erb_out, "m")?;

        // ── f. DF decoder ─────────────────────────────────────────────────
        // Inputs:
        //   emb  [1, S=1, 512]
        //   c0   [1, 64, S=1, 96]
        let df_out = self.df_dec.run(ort::inputs![
            "emb" => make_tensor(emb, &[1, 1, 512])?,
            "c0"  => make_tensor(c0,  &[1, 64, 1, 96])?,
        ])?;

        // coefs [batch, S=1, NB_DF, 10]
        // 10 = DF_ORDER * 2 (5 taps × {re, im})
        // We need flat layout [NB_DF, DF_ORDER, 2] for apply_df_filter:
        //   coefs[f, tap, ri] = flat[ (f * DF_ORDER + tap) * 2 + ri ]
        // ONNX gives us [NB_DF, 10] already in that order (NB_DF outer, 10 inner)
        // because shape is [1, 1, NB_DF, 10] → ravelled = NB_DF * 10 values.
        let coefs = extract_f32(&df_out, "coefs")?; // len = 96 * 10 = 960

        // ── g. Apply ERB mask to full spectrum ────────────────────────────
        // m has length 32 (from the last dim of [batch, 1, 1, 32])
        apply_erb_mask(&mut spec, &m[..NB_ERB.min(m.len())], &self.erb_bins);

        // ── h. Apply DF filter to lower NB_DF bins ────────────────────────
        if self.rolling.len() >= DF_ORDER {
            apply_df_filter(&mut spec[..NB_DF], &coefs, &self.rolling);
        }

        // ── i. ISTFT synthesis ────────────────────────────────────────────
        let mut out_frame = vec![0.0f32; HOP_SIZE];
        self.df_state.synthesis(&mut spec, &mut out_frame);

        Ok(out_frame)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ZeroMQ server loop
// ──────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    ).init();

    info!("Phase 5 — DeepFilterNet3 three-model Rust server");
    info!("  enc:     {ENC_PATH}");
    info!("  erb_dec: {ERB_DEC_PATH}");
    info!("  df_dec:  {DF_DEC_PATH}");

    let mut proc = NoiseProcessor::new()?;

    let ctx    = zmq::Context::new();
    let puller = ctx.socket(zmq::PULL)?;
    puller.bind(PULL_ADDR)?;
    let pusher = ctx.socket(zmq::PUSH)?;
    pusher.bind(PUSH_ADDR)?;
    info!("ZeroMQ ready — PULL {PULL_ADDR}  PUSH {PUSH_ADDR}");

    let mut chunk_n: u64 = 0;
    let mut total_ms = 0.0f64;

    loop {
        let bytes = puller.recv_bytes(0)?;
        if bytes.len() != FRAMES_44 * 4 {
            warn!("Bad chunk size {} (expected {})", bytes.len(), FRAMES_44 * 4);
            continue;
        }

        let input: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        let t0 = Instant::now();
        let output = proc.process_chunk(&input).unwrap_or_else(|e| {
            warn!("process_chunk error: {e:#}");
            vec![0.0f32; FRAMES_44]
        });
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        chunk_n  += 1;
        total_ms += ms;

        if chunk_n % LOG_EVERY == 0 {
            info!("chunk={chunk_n}  avg={:.2}ms  last={ms:.2}ms",
                  total_ms / chunk_n as f64);
        }

        let out_bytes: Vec<u8> = output.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        pusher.send(out_bytes, 0)?;
    }
}