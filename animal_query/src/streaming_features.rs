// 59 total features for pre-classifer
// - Geometry: 4 (width, height, area, aspect_ratio)
// - FFT Channels: 4 × 11 = 44 (Luma, R, G, B)
// - Contrast k=3: 11
//
// 11 FFT derived stats per channel
// - Top 5 amplitudes
// - Top 5 frequencies (normalized)
// - Texture ratio (high frequency energy / total energy)

use ndarray::{s, ArrayView3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;
use std::time::Instant;

// Timing statistics (ns)
#[derive(Debug, Default, Clone)]
pub struct TimingStats {
    pub row_extraction_ns: u64,
    pub interpolation_ns: u64,
    pub fft_compute_ns: u64,
    pub peak_finding_ns: u64,
    pub total_ns: u64,
}

impl TimingStats {
    //  total time in microseconds as f64 for reporting.
    pub fn total_us(&self) -> f64 {
        self.total_ns as f64 / 1000.0
    }

    // breakdown in microseconds.
    pub fn breakdown_us(&self) -> (f64, f64, f64, f64) {
        (
            self.row_extraction_ns as f64 / 1000.0,
            self.interpolation_ns as f64 / 1000.0,
            self.fft_compute_ns as f64 / 1000.0,
            self.peak_finding_ns as f64 / 1000.0,
        )
    }
}

// extraction strategy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureExtractionStrategy {
    Baseline,
    Thirds,
    LumaOnly,
    Fft1024,
    MoreRows,
}

impl Default for FeatureExtractionStrategy {
    fn default() -> Self {
        Self::Thirds
    }
}

impl FeatureExtractionStrategy {
    pub fn fft_len(&self) -> usize {
        match self {
            Self::Baseline | Self::Thirds | Self::LumaOnly | Self::MoreRows => 512,
            Self::Fft1024 => 1024,
        }
    }
}

// FFT statistics for a single channel.
#[derive(Debug, Default, Clone, Copy)]
pub struct FftChannelStats {
    pub top_amps: [f32; 5],
    pub top_freqs: [f32; 5],
    pub texture_ratio: f32,
}

// Complete streaming feature vector.
#[derive(Debug, Default, Clone)]
pub struct StreamingFeatures {
    pub width: f32,
    pub height: f32,
    pub area: f32,
    pub aspect_ratio: f32,

    pub fft_luma: FftChannelStats,
    pub fft_r: FftChannelStats,
    pub fft_g: FftChannelStats,
    pub fft_b: FftChannelStats,
    pub fft_contrast_k3: FftChannelStats,
}

impl StreamingFeatures {
    // Convert to flat f32 array (59 features).
    pub fn to_array(&self) -> [f32; 59] {
        let mut arr = [0.0f32; 59];
        arr[0] = self.width;
        arr[1] = self.height;
        arr[2] = self.area;
        arr[3] = self.aspect_ratio;

        let mut idx = 4;
        for stats in [
            &self.fft_luma,
            &self.fft_r,
            &self.fft_g,
            &self.fft_b,
            &self.fft_contrast_k3,
        ] {
            for &a in &stats.top_amps {
                arr[idx] = a;
                idx += 1;
            }
            for &f in &stats.top_freqs {
                arr[idx] = f;
                idx += 1;
            }
            arr[idx] = stats.texture_ratio;
            idx += 1;
        }
        arr
    }
}


// reusable buffers for FFT computation.
pub struct FftBuffers {
    pub scratch: Vec<Complex<f32>>,
    // accumulated spectrum buffers (5 channels × FFT_LEN/2+1)
    pub spectrum_luma: Vec<f32>,
    pub spectrum_r: Vec<f32>,
    pub spectrum_g: Vec<f32>,
    pub spectrum_b: Vec<f32>,
    pub spectrum_contrast: Vec<f32>,
}

impl FftBuffers {
    pub fn new(fft_len: usize) -> Self {
        let spec_len = fft_len / 2 + 1;
        Self {
            scratch: vec![Complex::new(0.0, 0.0); fft_len],
            spectrum_luma: vec![0.0; spec_len],
            spectrum_r: vec![0.0; spec_len],
            spectrum_g: vec![0.0; spec_len],
            spectrum_b: vec![0.0; spec_len],
            spectrum_contrast: vec![0.0; spec_len],
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.spectrum_luma {
            *s = 0.0;
        }
        for s in &mut self.spectrum_r {
            *s = 0.0;
        }
        for s in &mut self.spectrum_g {
            *s = 0.0;
        }
        for s in &mut self.spectrum_b {
            *s = 0.0;
        }
        for s in &mut self.spectrum_contrast {
            *s = 0.0;
        }
    }
}


/// Extract streaming features from an RGB image.
///
/// # Arguments
/// * `img` - RGB image as (H, W, 3) array
/// * `fft_plan` - Pre-planned FFT (size = FFT_LEN)
/// * `buffers` - Pre-allocated buffers
/// * `timing` - Mutable reference to fill timing stats
///
/// # Const Generics
/// * `NUM_ROWS` - Number of rows to sample (default: 8)
/// * `FFT_LEN` - FFT size for interpolation (default: 512)
/// * `CONTRAST_K` - Contrast window size (default: 3)
pub fn extract_streaming<const NUM_ROWS: usize, const FFT_LEN: usize, const CONTRAST_K: usize>(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeatures {
    let total_start = Instant::now();

    let (h, w, _c) = img.dim();

    // Geometry
    let mut features = StreamingFeatures {
        width: w as f32,
        height: h as f32,
        area: (w * h) as f32,
        aspect_ratio: w as f32 / h as f32,
        ..Default::default()
    };

    buffers.reset();

    // set up row sampling
    let phase1_start = Instant::now();
    let row_indices = compute_row_indices::<NUM_ROWS>(h);
    timing.row_extraction_ns = phase1_start.elapsed().as_nanos() as u64;

    // row extraction + interpolation + FFT
    let mut total_interp_ns = 0u64;
    let mut total_fft_ns = 0u64;

    let inv_num_rows = 1.0 / NUM_ROWS as f32;

    for &row_idx in &row_indices {
        let (interp_ns, fft_ns) = process_row_timed::<FFT_LEN, CONTRAST_K>(
            img,
            row_idx,
            w,
            &mut buffers.spectrum_luma,
            &mut buffers.spectrum_r,
            &mut buffers.spectrum_g,
            &mut buffers.spectrum_b,
            &mut buffers.spectrum_contrast,
            &mut buffers.scratch,
            fft_plan,
            inv_num_rows,
        );
        total_interp_ns += interp_ns;
        total_fft_ns += fft_ns;
    }

    timing.interpolation_ns = total_interp_ns;
    timing.fft_compute_ns = total_fft_ns;

    // peak finding
    let phase3_start = Instant::now();

    features.fft_luma = extract_fft_stats(&buffers.spectrum_luma, FFT_LEN);
    features.fft_r = extract_fft_stats(&buffers.spectrum_r, FFT_LEN);
    features.fft_g = extract_fft_stats(&buffers.spectrum_g, FFT_LEN);
    features.fft_b = extract_fft_stats(&buffers.spectrum_b, FFT_LEN);
    features.fft_contrast_k3 = extract_fft_stats(&buffers.spectrum_contrast, FFT_LEN);

    timing.peak_finding_ns = phase3_start.elapsed().as_nanos() as u64;
    timing.total_ns = total_start.elapsed().as_nanos() as u64;

    features
}


fn compute_row_indices<const NUM_ROWS: usize>(height: usize) -> [usize; NUM_ROWS] {
    let start = (height as f32 * 0.1) as usize;
    let end = (height as f32 * 0.9) as usize;
    let range = end.saturating_sub(start);

    let mut indices = [0usize; NUM_ROWS];
    if NUM_ROWS == 1 {
        indices[0] = height / 2;
    } else {
        for i in 0..NUM_ROWS {
            indices[i] = start + (i * range) / (NUM_ROWS - 1);
        }
    }
    indices
}


// returns (interpolation_ns, fft_ns)
// otherwise outputs go in place into the argumetn buffers
fn process_row_timed<const FFT_LEN: usize, const CONTRAST_K: usize>(
    img: &ArrayView3<u8>,
    row_idx: usize,
    width: usize,
    spectrum_luma: &mut [f32],
    spectrum_r: &mut [f32],
    spectrum_g: &mut [f32],
    spectrum_b: &mut [f32],
    spectrum_contrast: &mut [f32],
    scratch: &mut [Complex<f32>],
    fft_plan: &Arc<dyn Fft<f32>>,
    inv_num_rows: f32,
) -> (u64, u64) {
    let mut interp_ns = 0u64;
    let mut fft_ns = 0u64;

    // extract row data
    let row = img.slice(s![row_idx, .., ..]);

    // temp buffers for this row
    let mut luma_row = vec![0.0f32; width];
    let mut r_row = vec![0.0f32; width];
    let mut g_row = vec![0.0f32; width];
    let mut b_row = vec![0.0f32; width];

    // extract channels
    for x in 0..width {
        let r = row[[x, 0]] as f32;
        let g = row[[x, 1]] as f32;
        let b = row[[x, 2]] as f32;
        r_row[x] = r;
        g_row[x] = g;
        b_row[x] = b;
        luma_row[x] = (r + g + b) / 3.0;
    }

    let contrast_len = width.saturating_sub(CONTRAST_K);
    let mut contrast_row = vec![0.0f32; contrast_len];
    for x in 0..contrast_len {
        contrast_row[x] = (luma_row[x + CONTRAST_K] - luma_row[x]).abs();
    }

    // process each channel with timing
    let (i, f) = process_channel_fft_timed::<FFT_LEN>(
        &luma_row,
        scratch,
        fft_plan,
        spectrum_luma,
        inv_num_rows,
    );
    interp_ns += i;
    fft_ns += f;

    let (i, f) =
        process_channel_fft_timed::<FFT_LEN>(&r_row, scratch, fft_plan, spectrum_r, inv_num_rows);
    interp_ns += i;
    fft_ns += f;

    let (i, f) =
        process_channel_fft_timed::<FFT_LEN>(&g_row, scratch, fft_plan, spectrum_g, inv_num_rows);
    interp_ns += i;
    fft_ns += f;

    let (i, f) =
        process_channel_fft_timed::<FFT_LEN>(&b_row, scratch, fft_plan, spectrum_b, inv_num_rows);
    interp_ns += i;
    fft_ns += f;

    let (i, f) = process_channel_fft_timed::<FFT_LEN>(
        &contrast_row,
        scratch,
        fft_plan,
        spectrum_contrast,
        inv_num_rows,
    );
    interp_ns += i;
    fft_ns += f;

    (interp_ns, fft_ns)
}

// returns (interpolation_ns, fft_ns)
// otherwise outputs go in place into the argument buffers
fn process_channel_fft_timed<const FFT_LEN: usize>(
    row: &[f32],
    scratch: &mut [Complex<f32>],
    fft_plan: &Arc<dyn Fft<f32>>,
    spectrum_accum: &mut [f32],
    inv_num_rows: f32,
) -> (u64, u64) {
    let interp_start = Instant::now();
    interpolate_linear(row, scratch, FFT_LEN);
    let interp_ns = interp_start.elapsed().as_nanos() as u64;

    let fft_start = Instant::now();
    fft_plan.process(scratch);
    let fft_ns = fft_start.elapsed().as_nanos() as u64;

    // accumulate magnitudes (first half only)
    let half = FFT_LEN / 2 + 1;
    for i in 0..half {
        spectrum_accum[i] += scratch[i].norm() * inv_num_rows;
    }

    (interp_ns, fft_ns)
}

fn interpolate_linear(src: &[f32], dst: &mut [Complex<f32>], dst_len: usize) {
    if src.is_empty() {
        for c in dst.iter_mut().take(dst_len) {
            *c = Complex::new(0.0, 0.0);
        }
        return;
    }

    let scale = (src.len() - 1) as f32 / (dst_len - 1).max(1) as f32;

    for i in 0..dst_len {
        let pos = i as f32 * scale;
        let idx = pos.floor() as usize;
        let t = pos - pos.floor();

        let p0 = src[idx.min(src.len() - 1)];
        let p1 = src[(idx + 1).min(src.len() - 1)];

        dst[i] = Complex::new(p0 * (1.0 - t) + p1 * t, 0.0);
    }
}

fn extract_fft_stats(spectrum: &[f32], fft_len: usize) -> FftChannelStats {
    let mut stats = FftChannelStats::default();

    if spectrum.is_empty() {
        return stats;
    }

    // make working copy, sanitize NaN
    let mut work: Vec<(usize, f32)> = spectrum
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let v = if v.is_nan() { f32::NEG_INFINITY } else { v };
            let v = if i == 0 { 0.0 } else { v };
            (i, v)
        })
        .collect();

    // sort by magnitude descending
    work.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // top 5 peaks
    for (i, &(freq_idx, amp)) in work.iter().take(5).enumerate() {
        stats.top_amps[i] = amp;
        stats.top_freqs[i] = freq_idx as f32 / fft_len as f32;
    }

    // texture ratio measures high frequency energy relative to total energy
    let total_energy: f32 = spectrum
        .iter()
        .skip(1)
        .filter(|&&v| v > 0.0 && !v.is_nan())
        .sum();
    let cutoff = fft_len / 20; // 10% of Nyquist
    let hf_energy: f32 = spectrum
        .iter()
        .skip(cutoff)
        .filter(|&&v| v > 0.0 && !v.is_nan())
        .sum();

    stats.texture_ratio = if total_energy > 0.0 {
        hf_energy / total_energy
    } else {
        0.0
    };

    stats
}

pub fn create_fft_plan(len: usize) -> Arc<dyn Fft<f32>> {
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(len)
}

// =============================================================================
// config-specific extraction functions
// (used when narrowing down features and pre-classifier strategy)
// =============================================================================

// original implementation, used as baseline
// (internal pre-classifier selection name: n8-m1-k3)
// 8 rows, no averaging, 512 FFT, k=3 contrast window
// standard extraction during original proof-of-concept implementation for animal query.
pub fn extract_baseline(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeatures {
    extract_streaming::<8, 512, 3>(img, fft_plan, buffers, timing)
}

// Thirds (current best pre-classifier feature set)
// (n9-m3-k3): 9 rows, average into 3 groups (top/middle/bottom thirds), 512 FFT, k=3 contrast
// Sampling rows into thirds was chosen based on cinematic rule of thirds, but we found that averaging different numbers of rows generally had a good impact over a single row.
// the exact best strategy is likely to rely more on the distribution in the data than any absolute truth about row counts
pub fn extract_thirds(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeatures {
    extract_streaming_with_averaging::<9, 3, 512, 3>(img, fft_plan, buffers, timing)
}

// Luma Only
// (n8-m1-k3): 8 rows, no averaging, 512 FFT, k=3 contrast
// only extracts brightness and contrast FFT (no separate r,g,b channels)
// Fewer FFTs = faster extraction.
pub fn extract_luma_only(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeaturesLumaOnly {
    extract_streaming_luma_only::<8, 512, 3>(img, fft_plan, buffers, timing)
}

// FFT-1024
// (n8-m1-k3): 8 rows, no averaging, 1024 FFT, k=3 contrast
// like original but with higher frequency resolution (slower, not definiteively better in a way that is guaranteed to be worth it, at least on the data that we tried).
pub fn extract_fft1024(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>, // Must be 1024-length plan
    buffers: &mut FftBuffers,     // Must be 1024-length buffers
    timing: &mut TimingStats,
) -> StreamingFeatures {
    extract_streaming::<8, 1024, 3>(img, fft_plan, buffers, timing)
}

// More Rows strategy
// (n16-m1-k3): 16 rows, no averaging, 512 FFT, k=3 contrast
// samples twice as many rows for potentially finer spatial detail. This is more expensive but did not yield consistently better results in a way that was worth that cost.
pub fn extract_more_rows(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeatures {
    extract_streaming::<16, 512, 3>(img, fft_plan, buffers, timing)
}


// Luma-only feature output (26 features instead of 59).
#[derive(Debug, Default, Clone)]
pub struct StreamingFeaturesLumaOnly {
    pub width: f32,
    pub height: f32,
    pub area: f32,
    pub aspect_ratio: f32,

    pub fft_luma: FftChannelStats,
    pub fft_contrast_k3: FftChannelStats,
}

impl StreamingFeaturesLumaOnly {
    // convert to flat f32 array (26 features).
    pub fn to_array(&self) -> [f32; 26] {
        let mut arr = [0.0f32; 26];
        arr[0] = self.width;
        arr[1] = self.height;
        arr[2] = self.area;
        arr[3] = self.aspect_ratio;

        let mut idx = 4;
        for stats in [&self.fft_luma, &self.fft_contrast_k3] {
            for &a in &stats.top_amps {
                arr[idx] = a;
                idx += 1;
            }
            for &f in &stats.top_freqs {
                arr[idx] = f;
                idx += 1;
            }
            arr[idx] = stats.texture_ratio;
            idx += 1;
        }
        arr
    }
}

// Luma-only extraction (only 2 FFT channels instead of 5).
fn extract_streaming_luma_only<
    const NUM_ROWS: usize,
    const FFT_LEN: usize,
    const CONTRAST_K: usize,
>(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeaturesLumaOnly {
    let total_start = Instant::now();
    let (h, w, _c) = img.dim();

    let mut features = StreamingFeaturesLumaOnly {
        width: w as f32,
        height: h as f32,
        area: (w * h) as f32,
        aspect_ratio: w as f32 / h as f32,
        ..Default::default()
    };

    buffers.reset();

    let phase1_start = Instant::now();
    let row_indices = compute_row_indices::<NUM_ROWS>(h);
    timing.row_extraction_ns = phase1_start.elapsed().as_nanos() as u64;

    let mut total_interp_ns = 0u64;
    let mut total_fft_ns = 0u64;
    let inv_num_rows = 1.0 / NUM_ROWS as f32;

    for &row_idx in &row_indices {
        let (interp_ns, fft_ns) = process_row_luma_only::<FFT_LEN, CONTRAST_K>(
            img,
            row_idx,
            w,
            &mut buffers.spectrum_luma,
            &mut buffers.spectrum_contrast,
            &mut buffers.scratch,
            fft_plan,
            inv_num_rows,
        );
        total_interp_ns += interp_ns;
        total_fft_ns += fft_ns;
    }

    timing.interpolation_ns = total_interp_ns;
    timing.fft_compute_ns = total_fft_ns;

    let phase3_start = Instant::now();
    features.fft_luma = extract_fft_stats(&buffers.spectrum_luma, FFT_LEN);
    features.fft_contrast_k3 = extract_fft_stats(&buffers.spectrum_contrast, FFT_LEN);

    timing.peak_finding_ns = phase3_start.elapsed().as_nanos() as u64;
    timing.total_ns = total_start.elapsed().as_nanos() as u64;

    features
}

fn process_row_luma_only<const FFT_LEN: usize, const CONTRAST_K: usize>(
    img: &ArrayView3<u8>,
    row_idx: usize,
    width: usize,
    spectrum_luma: &mut [f32],
    spectrum_contrast: &mut [f32],
    scratch: &mut [Complex<f32>],
    fft_plan: &Arc<dyn Fft<f32>>,
    inv_num_rows: f32,
) -> (u64, u64) {
    let mut interp_ns = 0u64;
    let mut fft_ns = 0u64;

    let row = img.slice(s![row_idx, .., ..]);
    let mut luma_row = vec![0.0f32; width];

    for x in 0..width {
        let r = row[[x, 0]] as f32;
        let g = row[[x, 1]] as f32;
        let b = row[[x, 2]] as f32;
        luma_row[x] = (r + g + b) / 3.0;
    }

    let contrast_len = width.saturating_sub(CONTRAST_K);
    let mut contrast_row = vec![0.0f32; contrast_len];
    for x in 0..contrast_len {
        contrast_row[x] = (luma_row[x + CONTRAST_K] - luma_row[x]).abs();
    }

    let (i, f) = process_channel_fft_timed::<FFT_LEN>(
        &luma_row,
        scratch,
        fft_plan,
        spectrum_luma,
        inv_num_rows,
    );
    interp_ns += i;
    fft_ns += f;

    let (i, f) = process_channel_fft_timed::<FFT_LEN>(
        &contrast_row,
        scratch,
        fft_plan,
        spectrum_contrast,
        inv_num_rows,
    );
    interp_ns += i;
    fft_ns += f;

    (interp_ns, fft_ns)
}

// row group averaging; introduces reslience against object placement by averaging differently placed rows
fn extract_streaming_with_averaging<
    const NUM_ROWS: usize,
    const M_AVG: usize,
    const FFT_LEN: usize,
    const CONTRAST_K: usize,
>(
    img: &ArrayView3<u8>,
    fft_plan: &Arc<dyn Fft<f32>>,
    buffers: &mut FftBuffers,
    timing: &mut TimingStats,
) -> StreamingFeatures {
    let total_start = Instant::now();
    let (h, w, _c) = img.dim();

    let mut features = StreamingFeatures {
        width: w as f32,
        height: h as f32,
        area: (w * h) as f32,
        aspect_ratio: w as f32 / h as f32,
        ..Default::default()
    };

    buffers.reset();

    let phase1_start = Instant::now();
    let row_indices = compute_row_indices::<NUM_ROWS>(h);
    timing.row_extraction_ns = phase1_start.elapsed().as_nanos() as u64;

    // extract all row data first
    let mut all_luma: Vec<Vec<f32>> = Vec::with_capacity(NUM_ROWS);
    let mut all_r: Vec<Vec<f32>> = Vec::with_capacity(NUM_ROWS);
    let mut all_g: Vec<Vec<f32>> = Vec::with_capacity(NUM_ROWS);
    let mut all_b: Vec<Vec<f32>> = Vec::with_capacity(NUM_ROWS);

    for &row_idx in &row_indices {
        let row = img.slice(s![row_idx, .., ..]);
        let mut luma_row = vec![0.0f32; w];
        let mut r_row = vec![0.0f32; w];
        let mut g_row = vec![0.0f32; w];
        let mut b_row = vec![0.0f32; w];

        for x in 0..w {
            let r = row[[x, 0]] as f32;
            let g = row[[x, 1]] as f32;
            let b = row[[x, 2]] as f32;
            r_row[x] = r;
            g_row[x] = g;
            b_row[x] = b;
            luma_row[x] = (r + g + b) / 3.0;
        }

        all_luma.push(luma_row);
        all_r.push(r_row);
        all_g.push(g_row);
        all_b.push(b_row);
    }

    // average into groups
    let n_groups = NUM_ROWS / M_AVG;
    let mut grouped_luma: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
    let mut grouped_r: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
    let mut grouped_g: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
    let mut grouped_b: Vec<Vec<f32>> = Vec::with_capacity(n_groups);

    for group_idx in 0..n_groups {
        let start = group_idx * M_AVG;
        let end = start + M_AVG;

        let mut avg_luma = vec![0.0f32; w];
        let mut avg_r = vec![0.0f32; w];
        let mut avg_g = vec![0.0f32; w];
        let mut avg_b = vec![0.0f32; w];

        for i in start..end {
            for x in 0..w {
                avg_luma[x] += all_luma[i][x];
                avg_r[x] += all_r[i][x];
                avg_g[x] += all_g[i][x];
                avg_b[x] += all_b[i][x];
            }
        }

        let inv_m = 1.0 / M_AVG as f32;
        for x in 0..w {
            avg_luma[x] *= inv_m;
            avg_r[x] *= inv_m;
            avg_g[x] *= inv_m;
            avg_b[x] *= inv_m;
        }

        grouped_luma.push(avg_luma);
        grouped_r.push(avg_r);
        grouped_g.push(avg_g);
        grouped_b.push(avg_b);
    }

    // process groups
    let mut total_interp_ns = 0u64;
    let mut total_fft_ns = 0u64;
    let inv_n_groups = 1.0 / n_groups as f32;

    for group_idx in 0..n_groups {
        // contrast from grouped brightness
        let contrast_len = w.saturating_sub(CONTRAST_K);
        let mut contrast_row = vec![0.0f32; contrast_len];
        for x in 0..contrast_len {
            contrast_row[x] =
                (grouped_luma[group_idx][x + CONTRAST_K] - grouped_luma[group_idx][x]).abs();
        }

        // process each channel
        let (i, f) = process_channel_fft_timed::<FFT_LEN>(
            &grouped_luma[group_idx],
            &mut buffers.scratch,
            fft_plan,
            &mut buffers.spectrum_luma,
            inv_n_groups,
        );
        total_interp_ns += i;
        total_fft_ns += f;

        let (i, f) = process_channel_fft_timed::<FFT_LEN>(
            &grouped_r[group_idx],
            &mut buffers.scratch,
            fft_plan,
            &mut buffers.spectrum_r,
            inv_n_groups,
        );
        total_interp_ns += i;
        total_fft_ns += f;

        let (i, f) = process_channel_fft_timed::<FFT_LEN>(
            &grouped_g[group_idx],
            &mut buffers.scratch,
            fft_plan,
            &mut buffers.spectrum_g,
            inv_n_groups,
        );
        total_interp_ns += i;
        total_fft_ns += f;

        let (i, f) = process_channel_fft_timed::<FFT_LEN>(
            &grouped_b[group_idx],
            &mut buffers.scratch,
            fft_plan,
            &mut buffers.spectrum_b,
            inv_n_groups,
        );
        total_interp_ns += i;
        total_fft_ns += f;

        let (i, f) = process_channel_fft_timed::<FFT_LEN>(
            &contrast_row,
            &mut buffers.scratch,
            fft_plan,
            &mut buffers.spectrum_contrast,
            inv_n_groups,
        );
        total_interp_ns += i;
        total_fft_ns += f;
    }

    timing.interpolation_ns = total_interp_ns;
    timing.fft_compute_ns = total_fft_ns;

    let phase3_start = Instant::now();
    features.fft_luma = extract_fft_stats(&buffers.spectrum_luma, FFT_LEN);
    features.fft_r = extract_fft_stats(&buffers.spectrum_r, FFT_LEN);
    features.fft_g = extract_fft_stats(&buffers.spectrum_g, FFT_LEN);
    features.fft_b = extract_fft_stats(&buffers.spectrum_b, FFT_LEN);
    features.fft_contrast_k3 = extract_fft_stats(&buffers.spectrum_contrast, FFT_LEN);

    timing.peak_finding_ns = phase3_start.elapsed().as_nanos() as u64;
    timing.total_ns = total_start.elapsed().as_nanos() as u64;

    features
}
