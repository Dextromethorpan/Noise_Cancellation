//! Phase 7 — DeepFilterNet3 continuous stream server with Voice Activity Detection
//!
//! Builds on Phase 6 (continuous ring buffers) and adds lsnr-based VAD.
//!
//! Key finding from Phase 6 lsnr analysis (E05):
//!   - Silence → lsnr clusters at -10 to -15 dB
//!   - Speaking → lsnr clusters at +20 to +35 dB
//!   - Clean separation at threshold: +10 dB
//!
//! New behaviour (Phase 7 addition):
//!   - lsnr >= VAD_THRESHOLD_DB → voice detected, output model result
//!   - lsnr <  VAD_THRESHOLD_DB → no voice, output silence
//!
//! Everything else is identical to Phase 6.

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

/// Rubato resampler chunk sizes
const UP_CHUNK: usize = FRAMES_44; // 1536 @ 44100 → ~1672 @ 48000
const DOWN_CHUNK: usize = HOP_SIZE; // 480 @ 48000 → ~441 @ 44100

/// ZeroMQ addresses
const PULL_ADDR: &str = "tcp://127.0.0.1:5555";
const PUSH_ADDR: &str = "tcp://127.0.0.1:5556";

/// Model paths — relative to NoiseCancellation/ workspace root
const ENC_PATH:     &str = "ai/python/models/tmp/export/enc.onnx";
const ERB_DEC_PATH: &str = "ai/python/models/tmp/export/erb_dec.onnx";
const DF_DEC_PATH:  &str = "ai/python/models/tmp/export/df_dec.onnx";

/// Print stats every N chunks
const LOG_EVERY: u64 = 50;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  PHASE 7 — NEW: Voice Activity Detection threshold                          ║
// ║                                                                              ║
// ║  Derived from the structured silence/speech/silence test in E05:             ║
// ║    Silence  → lsnr in [-15, -10] dB                                         ║
// ║    Speaking → lsnr in [+20, +35] dB                                         ║
// ║                                                                              ║
// ║  +10 dB sits cleanly between both clusters with ample margin.                ║
// ║  Raise this value to be more aggressive (less background noise but           ║
// ║  risk clipping quiet speech). Lower it to be more permissive.               ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
const VAD_THRESHOLD_DB: f32 = 3.0;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  PHASE 7 — NEW: VAD hold-off frames                                         ║
// ║                                                                              ║
// ║  When voice is detected (lsnr >= threshold), we keep the gate open for      ║
// ║  this many additional frames after lsnr drops below threshold again.         ║
// ║  This prevents the gate from snapping shut mid-word during short pauses      ║
// ║  between syllables (e.g. the gap in "uno dos").                              ║
// ║                                                                              ║
// ║  Each frame = 10 ms @ 48 kHz.  30 frames = 300 ms hold-off.                ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
const VAD_HOLDOFF_FRAMES: u32 = 75;

// ──────────────────────────────────────────────────────────────────────────────
// Resampler helpers
// ──────────────────────────────────────────────────────────────────────────────

fn make_resampler(from_sr: usize, to_sr: usize, chunk: usize) -> Result<FftFixedIn<f32>> {
    FftFixedIn::<f32>::new(from_sr, to_sr, chunk, 2, 1)
        .context("Failed to create resampler")
}

// ──────────────────────────────────────────────────────────────────────────────
// Feature extraction (unchanged from Phase 6)
// ──────────────────────────────────────────────────────────────────────────────

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

fn feat_erb(
    spec: &[Complex32],
    erb_bins: &[usize],
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
        let logp = power.max(1e-12).log10() * 10.0;
        out[b] = norm.norm(logp);
    }
}

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
// Tensor helpers (unchanged from Phase 6)
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
// Spectrum post-processing (unchanged from Phase 6)
// ──────────────────────────────────────────────────────────────────────────────

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

fn apply_df_filter(
    spec: &mut [Complex32],
    coefs: &[f32],
    rolling: &VecDeque<Vec<Complex32>>,
) {
    let n = rolling.len();
    for f in 0..NB_DF {
        let mut out = Complex32::new(0.0, 0.0);
        for tap in 0..DF_ORDER {
            if tap >= n { break; }
            let buf_idx = n - 1 - tap;
            let frame = &rolling[buf_idx];
            let ci = (f * DF_ORDER + tap) * 2;
            if ci + 1 >= coefs.len() { break; }
            let w = Complex32::new(coefs[ci], coefs[ci + 1]);
            out += w * frame[f];
        }
        spec[f] = out;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Stream processor
// ──────────────────────────────────────────────────────────────────────────────

struct StreamProcessor {
    // ONNX sessions
    enc:     Session,
    erb_dec: Session,
    df_dec:  Session,

    // DSP state — never reset between chunks
    df_state:  DFState,
    erb_bins:  Vec<usize>,
    erb_norm:  MeanNorm,
    cplx_norm: UnitNorm,
    rolling:   VecDeque<Vec<Complex32>>,

    // Continuous resamplers — never recreated between chunks
    up:   FftFixedIn<f32>,
    down: FftFixedIn<f32>,

    // Ring buffers
    proc_buf_48:   Vec<f32>,
    output_buf_44: Vec<f32>,

    // ── PHASE 7 — NEW: VAD hold-off counter ─────────────────────────────────
    // Counts down from VAD_HOLDOFF_FRAMES after voice is last detected.
    // While > 0, the gate stays open even if lsnr drops below threshold.
    // This prevents clipping the tail of words during brief silent gaps.
    vad_holdoff: u32,
    // ────────────────────────────────────────────────────────────────────────
}

impl StreamProcessor {
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

        let df_state  = DFState::new(SR_48, FFT_SIZE, HOP_SIZE, NB_ERB, 1);
        let erb_bins  = df_state.erb.clone();
        let erb_norm  = MeanNorm::new(100.0);
        let cplx_norm = UnitNorm::new(100.0);
        let rolling   = VecDeque::with_capacity(DF_ORDER + LOOKAHEAD);

        let up   = make_resampler(44_100, SR_48,  UP_CHUNK)?;
        let down = make_resampler(SR_48,  44_100, DOWN_CHUNK)?;

        Ok(Self {
            enc, erb_dec, df_dec,
            df_state, erb_bins, erb_norm, cplx_norm, rolling,
            up, down,
            proc_buf_48:   Vec::with_capacity(HOP_SIZE * 8),
            output_buf_44: Vec::with_capacity(FRAMES_44 * 2),
            // ── PHASE 7 — NEW ───────────────────────────────────────────────
            vad_holdoff: 0,
            // ────────────────────────────────────────────────────────────────
        })
    }

    /// Accept one 1536-sample chunk @ 44100 Hz.
    /// Returns exactly 1536 samples @ 44100 Hz.
    fn push_chunk(&mut self, input: &[f32]) -> Result<Option<Vec<f32>>> {
        // ── 1. Upsample 44100 → 48000 (continuous) ──────────────────────────
        let up_out = self.up.process(&[input.to_vec()], None)?;
        let upsampled: &[f32] = &up_out[0];
        self.proc_buf_48.extend_from_slice(upsampled);

        // ── 2. Process all complete 480-sample hops ──────────────────────────
        while self.proc_buf_48.len() >= HOP_SIZE {
            let hop: Vec<f32> = self.proc_buf_48.drain(..HOP_SIZE).collect();
            let enhanced = self.process_frame(&hop)?;
            let down_out = self.down.process(&[enhanced], None)?;
            self.output_buf_44.extend_from_slice(&down_out[0]);
        }

        // ── 3. Return exactly FRAMES_44 samples when ready ──────────────────
        if self.output_buf_44.len() >= FRAMES_44 {
            let out: Vec<f32> = self.output_buf_44.drain(..FRAMES_44).collect();
            Ok(Some(out))
        } else {
            debug!(
                "Output buffer not full yet ({} / {FRAMES_44}), returning silence",
                self.output_buf_44.len()
            );
            Ok(Some(vec![0.0f32; FRAMES_44]))
        }
    }

    /// Process one 480-sample frame at 48 kHz → 480-sample enhanced frame.
    fn process_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        // ── a. STFT analysis ─────────────────────────────────────────────────
        let mut spec = vec![Complex32::new(0.0, 0.0); FREQ_SIZE];
        self.df_state.analysis(frame, &mut spec);

        let noisy_slice: Vec<Complex32> = spec[..NB_DF].to_vec();
        if self.rolling.len() >= DF_ORDER + LOOKAHEAD {
            self.rolling.pop_front();
        }
        self.rolling.push_back(noisy_slice);

        // ── b. ERB features → [1, 1, 1, 32] ─────────────────────────────────
        let mut erb_feat = [0.0f32; NB_ERB];
        feat_erb(&spec, &self.erb_bins, &mut self.erb_norm, &mut erb_feat);
        let feat_erb_t = make_tensor(erb_feat.to_vec(), &[1, 1, 1, NB_ERB])?;

        // ── c. Complex features → [1, 2, 1, 96] ──────────────────────────────
        let mut re = [0.0f32; NB_DF];
        let mut im = [0.0f32; NB_DF];
        feat_cplx(&spec, &mut self.cplx_norm, &mut re, &mut im);
        let mut spec_data = re.to_vec();
        spec_data.extend_from_slice(&im);
        let feat_spec_t = make_tensor(spec_data, &[1, 2, 1, NB_DF])?;

        // ── d. Encoder ───────────────────────────────────────────────────────
        let enc_out = self.enc.run(ort::inputs![
            "feat_erb"  => feat_erb_t,
            "feat_spec" => feat_spec_t,
        ])?;

        let emb  = extract_f32(&enc_out, "emb")?;
        let e0   = extract_f32(&enc_out, "e0")?;
        let e1   = extract_f32(&enc_out, "e1")?;
        let e2   = extract_f32(&enc_out, "e2")?;
        let e3   = extract_f32(&enc_out, "e3")?;
        let c0   = extract_f32(&enc_out, "c0")?;
        let lsnr = extract_f32(&enc_out, "lsnr")?;

        let lsnr_val = lsnr.first().copied().unwrap_or(0.0);

        // ╔══════════════════════════════════════════════════════════════════════╗
        // ║  PHASE 7 — NEW: Voice Activity Detection gate                        ║
        // ║                                                                        ║
        // ║  lsnr is the local signal-to-noise ratio estimated by the encoder.    ║
        // ║  When it exceeds VAD_THRESHOLD_DB (+10 dB) we consider voice active   ║
        // ║  and reset the hold-off counter.                                       ║
        // ║                                                                        ║
        // ║  When lsnr drops below the threshold, the hold-off counter counts     ║
        // ║  down to zero before the gate closes. This avoids clipping the tail   ║
        // ║  of words during brief inter-syllable pauses.                          ║
        // ║                                                                        ║
        // ║  If the gate is closed → return silence immediately, skipping the      ║
        // ║  decoders entirely. This suppresses background noise and instruments   ║
        // ║  that leak through the ERB mask when no voice is present.             ║
        // ╚══════════════════════════════════════════════════════════════════════╝
        if lsnr_val >= VAD_THRESHOLD_DB {
            // Voice detected — reset hold-off counter
            self.vad_holdoff = VAD_HOLDOFF_FRAMES;
            debug!("VAD: OPEN  lsnr={lsnr_val:.1} dB");
        } else if self.vad_holdoff > 0 {
            // Below threshold but still within hold-off window — keep gate open
            self.vad_holdoff -= 1;
            debug!("VAD: HOLD  lsnr={lsnr_val:.1} dB  holdoff={}", self.vad_holdoff);
        } else {
            // Gate closed — output silence, skip decoders
            debug!("VAD: SHUT  lsnr={lsnr_val:.1} dB");
            return Ok(vec![0.0f32; HOP_SIZE]);
        }
        // ════════════════════════════════════════════════════════════════════════

        // ── e. ERB decoder ───────────────────────────────────────────────────
        let erb_out = self.erb_dec.run(ort::inputs![
            "emb" => make_tensor(emb.clone(), &[1, 1, 512])?,
            "e3"  => make_tensor(e3,          &[1, 64, 1, 8])?,
            "e2"  => make_tensor(e2,          &[1, 64, 1, 8])?,
            "e1"  => make_tensor(e1,          &[1, 64, 1, 16])?,
            "e0"  => make_tensor(e0,          &[1, 64, 1, 32])?,
        ])?;
        let m = extract_f32(&erb_out, "m")?;

        // ── f. DF decoder ────────────────────────────────────────────────────
        let df_out = self.df_dec.run(ort::inputs![
            "emb" => make_tensor(emb, &[1, 1, 512])?,
            "c0"  => make_tensor(c0,  &[1, 64, 1, 96])?,
        ])?;
        let coefs = extract_f32(&df_out, "coefs")?;

        // ── g. Apply ERB mask ────────────────────────────────────────────────
        apply_erb_mask(&mut spec, &m[..NB_ERB.min(m.len())], &self.erb_bins);

        // ── h. Apply DF filter ───────────────────────────────────────────────
        if self.rolling.len() >= DF_ORDER {
            apply_df_filter(&mut spec[..NB_DF], &coefs, &self.rolling);
        }

        // ── i. ISTFT synthesis ───────────────────────────────────────────────
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

    info!("Phase 7 — DeepFilterNet3 continuous stream server with VAD");
    info!("  enc:     {ENC_PATH}");
    info!("  erb_dec: {ERB_DEC_PATH}");
    info!("  df_dec:  {DF_DEC_PATH}");
    info!("  VAD threshold: {VAD_THRESHOLD_DB} dB");
    info!("  VAD hold-off:  {VAD_HOLDOFF_FRAMES} frames ({} ms)",
          VAD_HOLDOFF_FRAMES * (HOP_SIZE as u32) * 1000 / SR_48 as u32);

    let mut proc = StreamProcessor::new()?;

    let ctx    = zmq::Context::new();
    let puller = ctx.socket(zmq::PULL)?;
    puller.bind(PULL_ADDR)?;
    let pusher = ctx.socket(zmq::PUSH)?;
    pusher.bind(PUSH_ADDR)?;
    info!("ZeroMQ ready — PULL {PULL_ADDR}  PUSH {PUSH_ADDR}");

    let mut chunk_n:  u64 = 0;
    let mut total_ms: f64 = 0.0;

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
        let output = match proc.push_chunk(&input) {
            Ok(Some(v)) => v,
            Ok(None)    => vec![0.0f32; FRAMES_44],
            Err(e) => {
                warn!("push_chunk error: {e:#}");
                vec![0.0f32; FRAMES_44]
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        chunk_n  += 1;
        total_ms += ms;

        if chunk_n % LOG_EVERY == 0 {
            info!(
                "chunk={chunk_n}  avg={:.2}ms  last={ms:.2}ms  \
                 proc_buf={}  out_buf={}  vad_holdoff={}",
                total_ms / chunk_n as f64,
                proc.proc_buf_48.len(),
                proc.output_buf_44.len(),
                proc.vad_holdoff,
            );
        }

        let out_bytes: Vec<u8> = output.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        pusher.send(out_bytes, 0)?;
    }
}