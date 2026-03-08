pub use spaghetto as devec;
pub mod async_operators;
pub mod async_query_builder;
pub mod basic_pooling;
pub mod caching;
pub mod chroma_utils;
pub mod expression;
pub mod global_logger;
pub mod operators;
pub mod pooling;
pub mod preclassifier_lang;
pub mod query_builder;
pub mod scheduler;
#[cfg(test)]
mod scheduler_tests;
pub mod ws_types;

pub mod map_api;

pub use operators::*;
pub use ws_types::*;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

const CHANNEL_TIMEOUT_MILLIS: u64 = 3;
const MAX_TIMEOUTS: usize = 500;

pub type TupleVec = smallvec::SmallVec<Tuple>;

#[derive(Debug, Deserialize)]
pub struct FDConfig {
    pub image_base_path: String,
    pub image_trimmed_path: Option<String>,
    pub model_base_path: String,
    pub log_folder: Option<String>,
    pub target_frametime: Option<u64>,
    pub detector_id: Option<i32>,
    pub embedder_id: Option<i32>,
    pub prediction_coefficients: Option<Vec<Vec<f64>>>,
    pub use_cuda: bool,
    pub adaptive_window_size: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub enum RemotePythonRequest {
    Request { bytes: Vec<u8> },
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RemotePythonResponse {
    pub bytes: Vec<u8>,
}

pub struct FrameReader {
    backing: Vec<Bytes>,
    current: usize,
    index_in_current: usize,
}

impl FrameReader {
    pub fn new(backing: Vec<Bytes>) -> Self {
        FrameReader {
            backing,
            current: 0,
            index_in_current: 0,
        }
    }
    pub fn push(&mut self, bytes: Bytes) {
        self.backing.push(bytes);
    }
}

use std::io::Read;
impl Read for FrameReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut amount_to_read = buf.len();
        let mut bytes_read = 0;
        while amount_to_read > 0 {
            if self.current == self.backing.len() {
                break;
            }
            let current_buffer = &self.backing[self.current];
            let bytes_taken = Read::read(&mut &current_buffer[self.index_in_current..], buf)?;
            bytes_read += bytes_taken;
            amount_to_read -= bytes_taken;
            self.index_in_current += bytes_taken;
            if self.index_in_current >= current_buffer.len() {
                self.current += 1;
                self.index_in_current = 0;
                continue;
            }
            if bytes_taken == 0 {
                // our source is empty
                break;
            }
        }
        Ok(bytes_read)
    }
}

// we need a rust version of this python code
/*
import numpy as np
from scipy.signal import argrelextrema
from numpy.fft import fft, ifft
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import time

def complete_signal_to_features_freq_per_amp_buckets(signal_matrix, sequence_len, num_buckets=5, noise_floor_threshold=0.05, minimum_noise_filtered_len=0.05):
  signal_fft = np.abs(fft(signal_matrix.T))
  features = []
  for c in range(signal_fft.shape[0]):
    signal_fft_channel = signal_fft[c, :sequence_len//2]
    signal_fft_culled = signal_fft_channel[signal_fft_channel >= (max(signal_fft_channel)*noise_floor_threshold)]
    if signal_fft_culled.size <= minimum_noise_filtered_len*sequence_len:
      signal_fft_culled = signal_fft_channel[signal_fft_channel >= (max(signal_fft_channel)*0.01)]
    hist = np.histogram(signal_fft_culled, bins=num_buckets)
    # print(hist)
    features.append([hist[0], hist[1][1:]])

  return np.array(features)
 */

// OnceLock of a DashMap<usize, Arc<dyn Fft<T>>>
// use std::sync::OnceLock;
// use dashmap::DashMap;
// use std::sync::Arc;
// static cached_ffts: OnceLock<DashMap<usize, Arc<dyn Fft<T>>>> = OnceLock::new();

// fn init_fft_cache() -> Arc<dyn Fft<T>> {
//     cached_ffts.get_or_init(||{todo!()});
//     let mut cached_ffts = cached_ffts.lock();
//     let fft = Arc::new(RealFft::<f32>::new(sequence_len));
//     cached_ffts.insert(sequence_len, fft.clone());
//     fft
// }

// fn complete_signal_to_features_freq_per_amp_buckets(signal_matrix: ndarray::ArrayD<f32>, sequence_len: usize, num_buckets: usize, noise_floor_threshold: f32, minimum_noise_filtered_len: f32) -> ndarray::ArrayD<f32> {
//     let signal_fft = ndarray::Array::from_iter(signal_matrix.axis_iter(ndarray::Axis(1)).map(|v| {
//         let mut signal_fft_channel = ndarray::Array::from_iter(fft(&v.to_vec()).iter().map(|v| v.norm()));
//         signal_fft_channel.slice_mut(s![..sequence_len/2]);
//         signal_fft_channel
//     }));
//     let mut features = Vec::new();
//     for c in 0..signal_fft.shape()[0] {
//         let signal_fft_channel = signal_fft.slice(s![c, ..]);
//         let signal_fft_culled = signal_fft_channel.iter().filter(|v| *v >= signal_fft_channel.max().unwrap()*noise_floor_threshold).collect::<Vec<_>>();
//         let signal_fft_culled = if signal_fft_culled.len() <= (minimum_noise_filtered_len*sequence_len) as usize {
//             signal_fft_channel.iter().filter(|v| *v >= signal_fft_channel.max().unwrap()*0.01).collect::<Vec<_>>()
//         } else {
//             signal_fft_culled
//         };
//         let hist = ndarray::histogram(&signal_fft_culled, num_buckets);
//         features.push(hist);
//     }
//     ndarray::Array::from(features)
// }

// wrapper type to allow any std::fmt::Write to be used as a std::io::Write
pub struct WriteWrapper<T>(pub T);

impl<T: std::fmt::Write> std::io::Write for WriteWrapper<T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let string_form = std::str::from_utf8(buf);
        let string_err_mapped =
            string_form.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.0
            .write_str(string_err_mapped)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// wrapper type to use serde_json to serialize a type
pub struct SerdeJson<T>(pub T);

impl<T> std::fmt::Debug for SerdeJson<T>
where
    T: Serialize,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        serde_json::to_writer(WriteWrapper(f), &self.0).map_err(|_e| std::fmt::Error)?;
        Ok(())
    }
}
