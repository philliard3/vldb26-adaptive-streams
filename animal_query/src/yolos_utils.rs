use crate::animal_fields::{
    FACES_DETECTED_FIELD, FACE_COUNT_FIELD, ORIGINAL_IMAGE_FIELD, ORIGINAL_IMAGE_SHAPE_FIELD,
};
use crate::streaming_features::{FftBuffers, TimingStats};
use base64::{engine::general_purpose, Engine as _};
use ndarray::s;
use ordered_float::OrderedFloat;
use rustfft::Fft;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use watershed_shared::basic_pooling::{get_tuple_vec, return_tuple};
use watershed_shared::{HabValue, Tuple};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

const YOLO_INPUT_SIZE: u32 = 640;
pub const DEFAULT_YOLO_CONF_THRESHOLD: f32 = 0.25;
pub const YOLO_IOU_THRESHOLD: f32 = 0.45;

static YOLO_CONF_THRESHOLD_BITS: AtomicU32 = AtomicU32::new(DEFAULT_YOLO_CONF_THRESHOLD.to_bits());
static YOLO_MAX_DET: AtomicI32 = AtomicI32::new(300);
static NMS_AGNOSTIC: AtomicBool = AtomicBool::new(true);

static INPUT_DUMPED: AtomicBool = AtomicBool::new(false);
static OUTPUT_DUMPED: AtomicBool = AtomicBool::new(false);

pub fn set_yolo_conf_threshold(conf: f32) {
    let conf = conf.clamp(0.0, 1.0);
    YOLO_CONF_THRESHOLD_BITS.store(conf.to_bits(), Ordering::Relaxed);
}

pub fn get_yolo_conf_threshold() -> f32 {
    f32::from_bits(YOLO_CONF_THRESHOLD_BITS.load(Ordering::Relaxed))
}

pub fn set_yolo_max_det(max_det: usize) {
    let max_det = (max_det as i32).max(1);
    YOLO_MAX_DET.store(max_det, Ordering::Relaxed);
}

pub fn set_nms_agnostic(agnostic: bool) {
    NMS_AGNOSTIC.store(agnostic, Ordering::Relaxed);
}

// ============================================================================
// Core NMS Utilities
// ============================================================================

pub fn area_of(boxes: &ndarray::ArrayView2<'_, f32>) -> ndarray::Array1<f32> {
    let left_top = boxes.slice(s![.., ..2]);
    let right_bottom = boxes.slice(s![.., 2..]);
    let hw = &right_bottom - &left_top;
    let hw = hw.mapv(|x| x.max(0.0));
    let areas = hw.slice(s![.., 0]).to_owned() * hw.slice(s![.., 1]).to_owned();
    areas
}

pub fn iou_matrix(
    boxes0: &ndarray::ArrayView2<'_, f32>,
    boxes1: &ndarray::ArrayView2<'_, f32>,
    eps: f32,
) -> ndarray::Array2<f32> {
    let overlap_left_top = boxes0
        .slice(s![.., ..2])
        .to_owned()
        .insert_axis(ndarray::Axis(1))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0], 2))
        .unwrap()
        .to_owned();
    let overlap_left_top_comp = boxes1
        .slice(s![.., ..2])
        .to_owned()
        .insert_axis(ndarray::Axis(0))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0], 2))
        .unwrap()
        .to_owned();
    let overlap_left_top = ndarray::Zip::from(&overlap_left_top)
        .and(&overlap_left_top_comp)
        .map_collect(|a, b| a.max(*b));

    let overlap_right_bottom = boxes0
        .slice(s![.., 2..])
        .to_owned()
        .insert_axis(ndarray::Axis(1))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0], 2))
        .unwrap()
        .to_owned();
    let overlap_right_bottom_comp = boxes1
        .slice(s![.., 2..])
        .to_owned()
        .insert_axis(ndarray::Axis(0))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0], 2))
        .unwrap()
        .to_owned();
    let overlap_right_bottom = ndarray::Zip::from(&overlap_right_bottom)
        .and(&overlap_right_bottom_comp)
        .map_collect(|a, b| a.min(*b));

    let overlap_area = (&overlap_right_bottom - &overlap_left_top).mapv(|x| x.max(0.0));
    let overlap_area =
        overlap_area.slice(s![.., .., 0]).to_owned() * overlap_area.slice(s![.., .., 1]);
    let area0 = area_of(&boxes0.view());
    let area1 = area_of(&boxes1.view());
    let area0 = area0
        .insert_axis(ndarray::Axis(1))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0]))
        .unwrap()
        .to_owned();
    let area1 = area1
        .insert_axis(ndarray::Axis(0))
        .broadcast((boxes0.shape()[0], boxes1.shape()[0]))
        .unwrap()
        .to_owned();
    let iou = &overlap_area / (&area0 + &area1 - &overlap_area + eps);
    iou
}

pub fn iou_of(
    boxes0: &ndarray::ArrayView2<'_, f32>,
    boxes1: &ndarray::ArrayView2<'_, f32>,
    eps: f32,
) -> ndarray::Array1<f32> {
    iou_matrix(boxes0, boxes1, eps).diag().to_owned()
}

pub fn hard_nms(
    box_scores: &ndarray::Array2<f32>,
    iou_threshold: f32,
    top_k: i32,
    candidate_size: usize,
) -> ndarray::Array2<f32> {
    let scores = box_scores.slice(s![.., -1]).to_owned();
    let boxes = box_scores.slice(s![.., ..-1]).to_owned();
    let mut picked = Vec::new();
    let mut indexes = scores.iter().enumerate().collect::<Vec<_>>();
    indexes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    indexes.truncate(candidate_size);
    while !indexes.is_empty() {
        let current = indexes.last().unwrap().0;
        picked.push(current);
        if (top_k > 0 && picked.len() as i32 == top_k) || indexes.len() == 1 {
            break;
        }
        let current_box = boxes.slice(s![current, ..]);
        indexes.pop();
        let rest_boxes = boxes.select(
            ndarray::Axis(0),
            &indexes.iter().map(|x| x.0).collect::<Vec<_>>(),
        );
        let iou = iou_matrix(
            &rest_boxes.view(),
            &current_box.to_owned().insert_axis(ndarray::Axis(0)).view(),
            1e-5,
        );
        let iou = iou.column(0);
        // zip with mask
        let mut iou_mask_iter = iou.iter().map(|&x| x <= iou_threshold);
        indexes.retain(|&_x| iou_mask_iter.next().unwrap_or_default());
    }
    box_scores.select(ndarray::Axis(0), &picked)
}

// ============================================================================
// YOLO preprocessing
// ============================================================================

pub struct LetterboxInfo {
    pub ratio: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub new_width: u32,
    pub new_height: u32,
}

pub fn yolo_preprocess(
    orig_image: &[u8],
    old_width: u32,
    old_height: u32,
) -> (ndarray::Array4<f32>, LetterboxInfo) {
    type PixelType = fast_image_resize::pixels::U8x3;
    let orig_image: &[PixelType] = if orig_image.len() % 3 == 0 {
        // SAFETY: we checked that the length is a multiple of 3, so it must be composed of pixels
        unsafe { std::mem::transmute(orig_image) }
        // bytemuck::cast_slice(orig_image)
    } else {
        error!("Image length is not a multiple of 3");
        return (
            ndarray::Array4::zeros((1, 3, YOLO_INPUT_SIZE as usize, YOLO_INPUT_SIZE as usize)),
            LetterboxInfo {
                ratio: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
                new_width: YOLO_INPUT_SIZE,
                new_height: YOLO_INPUT_SIZE,
            },
        );
    };

    let Ok(orig_image) =
        fast_image_resize::images::ImageRef::from_pixels(old_width, old_height, orig_image)
    else {
        error!("Failed to create ImageRef");
        return (
            ndarray::Array4::zeros((1, 3, YOLO_INPUT_SIZE as usize, YOLO_INPUT_SIZE as usize)),
            LetterboxInfo {
                ratio: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
                new_width: YOLO_INPUT_SIZE,
                new_height: YOLO_INPUT_SIZE,
            },
        );
    };

    // letterbox resize
    let scale =
        (YOLO_INPUT_SIZE as f32 / old_width as f32).min(YOLO_INPUT_SIZE as f32 / old_height as f32);
    let new_width = ((old_width as f32) * scale).round().max(1.0) as u32;
    let new_height = ((old_height as f32) * scale).round().max(1.0) as u32;

    let mut resized_image = fast_image_resize::images::Image::new(
        new_width,
        new_height,
        fast_image_resize::PixelType::U8x3,
    );

    thread_local! {
        static RESIZER: std::cell::RefCell<fast_image_resize::Resizer> =
            std::cell::RefCell::new(fast_image_resize::Resizer::new());
    }

    RESIZER.with(|r| {
        let mut r = r.borrow_mut();
        let _ = r.resize(
            &orig_image,
            &mut resized_image,
            &Some(fast_image_resize::ResizeOptions {
                algorithm: fast_image_resize::ResizeAlg::Convolution(
                    fast_image_resize::FilterType::Box,
                ),
                ..Default::default()
            }),
        );
    });

    let resized_buf = resized_image.into_vec();

    // create padded canvas with gray background (114)
    let pad_x = ((YOLO_INPUT_SIZE - new_width) / 2) as usize;
    let pad_y = ((YOLO_INPUT_SIZE - new_height) / 2) as usize;
    let canvas_w = YOLO_INPUT_SIZE as usize;
    let canvas_h = YOLO_INPUT_SIZE as usize;
    let mut padded_buf = vec![114u8; canvas_w * canvas_h * 3];

    for y in 0..new_height as usize {
        let src_row = y * new_width as usize * 3;
        let dst_row = (y + pad_y) * canvas_w * 3 + pad_x * 3;
        let src_slice = &resized_buf[src_row..src_row + new_width as usize * 3];
        let dst_slice = &mut padded_buf[dst_row..dst_row + new_width as usize * 3];
        dst_slice.copy_from_slice(src_slice);
    }

    let mut arr_image = ndarray::Array3::<u8>::from_shape_vec(
        (YOLO_INPUT_SIZE as usize, YOLO_INPUT_SIZE as usize, 3),
        padded_buf,
    )
    .unwrap();

    // HWC to CHW
    arr_image.swap_axes(0, 2);
    arr_image.swap_axes(1, 2);

    // corce contiguous layout
    let arr_image = arr_image.as_standard_layout().into_owned();

    let arr_image = arr_image.insert_axis(ndarray::Axis(0));

    // convert to f32 and normalize to [0.0, 1.0] if enabled
    let normalize_input = crate::animal_utils::get_yolo_input_normalization();
    let arr_image = if normalize_input {
        // ultralytics YOLOv8 expects input normalization by dividing by 255.0
        arr_image.mapv(|x| x as f32 / 255.0)
    } else {
        // Pass raw [0, 255] values (for legacy/experimental models)
        arr_image.mapv(|x| x as f32)
    };

    // validate contiguousness for ORT
    if !(arr_image.is_standard_layout()) {
        error!(
            "Array is not contiguous. strides={:?}, shape={:?}",
            arr_image.strides(),
            arr_image.shape()
        );
        return (
            ndarray::Array4::zeros((1, 3, YOLO_INPUT_SIZE as usize, YOLO_INPUT_SIZE as usize)),
            LetterboxInfo {
                ratio: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
                new_width: YOLO_INPUT_SIZE,
                new_height: YOLO_INPUT_SIZE,
            },
        );
    }
    (
        arr_image,
        LetterboxInfo {
            ratio: scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            new_width,
            new_height,
        },
    )
}

// ============================================================================
// YOLO Postprocessing
// ============================================================================

const CLASS_PROBS_START: usize = 4;

/// YOLOv8 postprocessing for object detection
///
/// # Arguments
/// * `output_buf` - Raw model output buffer [1, 84, 8400] containing sigmoid-activated values
/// * `output_dims` - Shape of output [batch=1, features=84, anchors=8400]
/// * `original_width` - Original image width for coordinate scaling
/// * `original_height` - Original image height for coordinate scaling
/// * `conf_threshold` - Minimum confidence threshold for detections (uses max class score)
/// * `letterbox_ratio` - Letterbox resize ratio (input / original)
/// * `letterbox_pad_x` - Letterbox pad x (pixels in input space)
/// * `letterbox_pad_y` - Letterbox pad y (pixels in input space)
///
/// # Returns
/// Tuple of (boxes, objectness_scores, all_class_scores) where:
/// * `boxes` - [N, 4] bounding boxes in pixel coordinates [x1, y1, x2, y2]
/// * `objectness_scores` - [N] objectness values [0,1] (uses max class score)
/// * `all_class_scores` - [N, 80] all class probabilities for each detection (sigmoid-activated)
///
/// Filters with max(class_scores), then returns the full probability distribution.
pub fn yolo_postprocess(
    output_buf: &[f32],
    output_dims: &[usize],
    original_width: i32,
    original_height: i32,
    conf_threshold: f32,
    letterbox_ratio: f32,
    letterbox_pad_x: f32,
    letterbox_pad_y: f32,
) -> (
    ndarray::Array2<i32>, // boxes [N, 4]
    ndarray::Array1<f32>, // objectness scores [N]
    ndarray::Array2<f32>, // class scores [N, 80]
) {
    // INTERMEDIATE DATA LOGGING: Log raw output buffer stats for comparison with Python
    if crate::animal_utils::should_log_intermediate_yolo_data() {
        let buf_len = output_buf.len();
        let buf_sum: f32 = output_buf.iter().sum();
        let buf_mean = buf_sum / buf_len as f32;
        let buf_min = output_buf.iter().copied().fold(f32::INFINITY, f32::min);
        let buf_max = output_buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        debug!(
            "[YOLO_INTERMEDIATE] raw_output_buffer: len={}, dims={:?}, sum={:.4}, mean={:.6}, min={:.6}, max={:.6}",
            buf_len, output_dims, buf_sum, buf_mean, buf_min, buf_max
        );
        // log first 20 values for detailed comparison
        let preview: Vec<f32> = output_buf.iter().take(20).copied().collect();
        debug!(
            "[YOLO_INTERMEDIATE] raw_output_buffer first 20 values: {:?}",
            preview
        );
    }

    if !OUTPUT_DUMPED.load(Ordering::Relaxed) {
        log_error_context_for_repro(
            output_buf,
            "DEBUG_DUMP_OUTPUT_TENSOR",
            "First image model output (1x84x8400)",
        );
        OUTPUT_DUMPED.store(true, Ordering::Relaxed);
    }

    let output_dims: &[usize; 3] = match output_dims.try_into() {
        Ok(v) => v,
        Err(_) => {
            error!("Invalid output dims, expected 3D array");
            return (
                ndarray::Array2::zeros((0, 4)),
                ndarray::Array1::zeros(0),
                ndarray::Array2::zeros((0, 80)),
            );
        }
    };

    let output = match ndarray::ArrayView3::from_shape(*output_dims, output_buf) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to reshape output: {:?}", e);
            return (
                ndarray::Array2::zeros((0, 4)),
                ndarray::Array1::zeros(0),
                ndarray::Array2::zeros((0, 80)),
            );
        }
    };

    // slice and transpose to de-batch [1, 84, 8400] -> [8400, 84]
    let output = output.slice(s![0, .., ..]).reversed_axes();

    // extract bbox coordindates and class scores
    let bbox_raw = output.slice(s![.., ..4]);
    let class_scores_raw = output.slice(s![.., CLASS_PROBS_START..]);

    // auto-detect coordinate scale
    let coords_normalized = bbox_raw.iter().all(|&v| v <= 1.5);

    let bbox_scaled = if coords_normalized {
        let scale = YOLO_INPUT_SIZE as f32;
        bbox_raw.mapv(|v| v * scale)
    } else {
        bbox_raw.to_owned()
    };

    // convert from center format to corner format
    let x_center = bbox_scaled.slice(s![.., 0]);
    let y_center = bbox_scaled.slice(s![.., 1]);
    let width = bbox_scaled.slice(s![.., 2]);
    let height = bbox_scaled.slice(s![.., 3]);

    let x1 = &x_center - &width / 2.0;
    let y1 = &y_center - &height / 2.0;
    let x2 = &x_center + &width / 2.0;
    let y2 = &y_center + &height / 2.0;

    let mut bbox_corners = ndarray::stack![ndarray::Axis(1), x1, y1, x2, y2];

    // undo letterbox padding
    bbox_corners
        .slice_mut(s![.., 0])
        .mapv_inplace(|v| v - letterbox_pad_x);
    bbox_corners
        .slice_mut(s![.., 1])
        .mapv_inplace(|v| v - letterbox_pad_y);
    bbox_corners
        .slice_mut(s![.., 2])
        .mapv_inplace(|v| v - letterbox_pad_x);
    bbox_corners
        .slice_mut(s![.., 3])
        .mapv_inplace(|v| v - letterbox_pad_y);

    // undo letterbox scaling
    bbox_corners /= letterbox_ratio;

    // clamp to image boundaries
    bbox_corners
        .slice_mut(s![.., 0])
        .mapv_inplace(|v| v.max(0.0).min(original_width as f32));
    bbox_corners
        .slice_mut(s![.., 1])
        .mapv_inplace(|v| v.max(0.0).min(original_height as f32));
    bbox_corners
        .slice_mut(s![.., 2])
        .mapv_inplace(|v| v.max(0.0).min(original_width as f32));
    bbox_corners
        .slice_mut(s![.., 3])
        .mapv_inplace(|v| v.max(0.0).min(original_height as f32));

    // get max class score as objectness
    let max_class_scores: Vec<f32> = class_scores_raw
        .axis_iter(ndarray::Axis(0))
        .map(|row| {
            row.iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        })
        .collect();

    // (anchor_idx, class_id, class_score_after_sigmoid)
    let mut all_detections: Vec<(usize, usize, f32)> = Vec::new(); 
    for anchor_idx in 0..class_scores_raw.shape()[0] {
        let class_scores = class_scores_raw.slice(s![anchor_idx, ..]);
        for (class_id, &score) in class_scores.iter().enumerate() {
            // raw scores already sigmoid-activated
            if score >= conf_threshold {
                let x1 = bbox_corners[[anchor_idx, 0]];
                let y1 = bbox_corners[[anchor_idx, 1]];
                let x2 = bbox_corners[[anchor_idx, 2]];
                let y2 = bbox_corners[[anchor_idx, 3]];

                if (x2 - x1) > 1e-3 && (y2 - y1) > 1e-3 {
                    all_detections.push((anchor_idx, class_id, score));
                }
            }
        }
    }

    // check score distribution to confirm range [0, 1]
    if crate::animal_utils::should_log_intermediate_yolo_data() {
        if let Some((_, _, min_score)) = all_detections
            .iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        {
            if let Some((_, _, max_score)) = all_detections
                .iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            {
                debug!(
                    "[YOLO_POSTPROCESS] Score stats (post-filter): min={:.4}, max={:.4}, count={}",
                    min_score,
                    max_score,
                    all_detections.len()
                );
            }
        }
    }

    if all_detections.is_empty() {
        return (
            ndarray::Array2::zeros((0, 4)),
            ndarray::Array1::zeros(0),
            ndarray::Array2::zeros((0, 80)),
        );
    }

    let mut nms_indices: Vec<usize> = Vec::new();
    let max_det = YOLO_MAX_DET.load(Ordering::Relaxed);
    let num_anchors = bbox_corners.nrows();
    let nms_agnostic = NMS_AGNOSTIC.load(Ordering::Relaxed);

    if nms_agnostic {
        // agnostic NMS: filter by objectness (max score), ignore class during NMS

        let mut candidates: Vec<(usize, f32)> = Vec::new();
        let mut last_anchor = usize::MAX;

        // all_detections sorted by anchor_idx loops
        for (anchor_idx, _, _) in &all_detections {
            if *anchor_idx != last_anchor {
                candidates.push((*anchor_idx, max_class_scores[*anchor_idx]));
                last_anchor = *anchor_idx;
            }
        }

        // sort candidates by objectness score (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // limit candidates BEFORE NMS (in ultralytics max_nms=30000)
        const MAX_NMS_CANDIDATES: usize = 30000;
        if candidates.len() > MAX_NMS_CANDIDATES {
            candidates.truncate(MAX_NMS_CANDIDATES);
        }

        // scores for NMS
        let box_scores = {
            let mut combined = ndarray::Array2::zeros((candidates.len(), 5));
            for (row_idx, (anchor_idx, score)) in candidates.iter().enumerate() {
                if *anchor_idx >= num_anchors {
                    continue;
                }

                combined
                    .slice_mut(s![row_idx, ..4])
                    .assign(&bbox_corners.row(*anchor_idx));
                combined[[row_idx, 4]] = *score;
            }
            combined
        };

        // global NMS
        let cand_nms_indices = get_nms_indices(&box_scores, YOLO_IOU_THRESHOLD, max_det);

        // map back to anchor indices
        for idx in cand_nms_indices {
            if idx < candidates.len() {
                nms_indices.push(candidates[idx].0);
            }
        }
    } else {
        // class-aware NMS (old behavior)
        // groups detections by class and applies NMS per class
        let mut class_groups: std::collections::HashMap<usize, Vec<(usize, f32)>> =
            std::collections::HashMap::new();
        for (anchor_idx, class_id, score) in all_detections {
            class_groups
                .entry(class_id)
                .or_insert_with(Vec::new)
                .push((anchor_idx, score));
        }

        for (_class_id, mut class_detections) in class_groups {
            // sort by score descending
            class_detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // create scores array for NMS: [boxes | score]
            let box_scores = {
                let mut combined = ndarray::Array2::zeros((class_detections.len(), 5));
                for (row_idx, (anchor_idx, score)) in class_detections.iter().enumerate() {
                    // bounds check anchor_idx
                    if *anchor_idx >= num_anchors {
                        let msg = format!(
                            "[YOLO_ERROR] Invalid anchor_idx: {} >= num_anchors ({})",
                            anchor_idx, num_anchors
                        );
                        error!("{}", msg);
                        log_error_context_for_repro(
                            output_buf,
                            &msg,
                            &format!("{{\"class_id\": {}, \"score\": {}}}", _class_id, score),
                        );
                        continue;
                    }
                    combined
                        .slice_mut(s![row_idx, ..4])
                        .assign(&bbox_corners.row(*anchor_idx));
                    combined[[row_idx, 4]] = *score;
                }
                combined
            };

            // apply NMS to this class's detections
            let class_nms_indices = get_nms_indices(&box_scores, YOLO_IOU_THRESHOLD, max_det);

            // map back to original anchor indices
            for nms_idx in class_nms_indices {
                // bounds check nms_idx against class_detections
                if nms_idx >= class_detections.len() {
                    let msg = format!(
                        "[YOLO_ERROR] Invalid NMS index: {} >= class_detections.len() ({})",
                        nms_idx,
                        class_detections.len()
                    );
                    error!("{}", msg);
                    log_error_context_for_repro(
                        output_buf,
                        &msg,
                        &format!("{{\"class_id\": {}}}", _class_id),
                    );
                    continue;
                }
                 // original anchor_idx
                nms_indices.push(class_detections[nms_idx].0);
            }
        }
    }

    // remove duplicates and limit to max_det
    nms_indices.sort_unstable();
    nms_indices.dedup();

    // filter top-k by confidence score
    // map to (index, score)
    let mut final_indices_with_score: Vec<(usize, f32)> = Vec::with_capacity(nms_indices.len());

    for &idx in &nms_indices {
        // bounds check idx against max_class_scores
        if idx >= max_class_scores.len() {
            let msg = format!(
                "[YOLO_ERROR] Invalid index after NMS: {} >= max_class_scores.len() ({})",
                idx,
                max_class_scores.len()
            );
            error!("{}", msg);
            log_error_context_for_repro(output_buf, &msg, "Post-NMS filtering stage");
            continue;
        }
        final_indices_with_score.push((idx, max_class_scores[idx]));
    }

    // sort by score descending
    final_indices_with_score
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // truncate to max_det
    if final_indices_with_score.len() > max_det as usize {
        final_indices_with_score.truncate(max_det as usize);
    }

    // extract just the indices, score sorted
    let valid_nms_indices: Vec<usize> = final_indices_with_score
        .into_iter()
        .map(|(idx, _)| idx)
        .collect();

    // final bounds check before selection
    for &idx in &valid_nms_indices {
        if idx >= bbox_corners.nrows() {
            let msg = format!("[YOLO_ERROR] CRITICAL: Invalid box index in final selection: {} >= bbox_corners.nrows() ({})", idx, bbox_corners.nrows());
            error!("{}", msg);
            log_error_context_for_repro(output_buf, &msg, "Final selection stage");
        }
    }

    let final_boxes = bbox_corners
        .select(ndarray::Axis(0), &valid_nms_indices)
        .mapv(|v| v as i32);
    let final_objectness: Vec<f32> = valid_nms_indices
        .iter()
        .map(|&i| max_class_scores[i])
        .collect();
    let class_scores_2d = class_scores_raw.select(ndarray::Axis(0), &valid_nms_indices);

    // log post-NMS results
    if crate::animal_utils::should_log_intermediate_yolo_data() {
        debug!(
            "[YOLO_INTERMEDIATE] post_nms: num_detections={}, conf_threshold={}, iou_threshold={}, max_det={}",
            final_boxes.shape()[0], conf_threshold, YOLO_IOU_THRESHOLD, max_det
        );
        if final_boxes.shape()[0] > 0 {
            debug!(
                "[YOLO_INTERMEDIATE] post_nms: first_box={:?}",
                final_boxes.row(0)
            );
            debug!(
                "[YOLO_INTERMEDIATE] post_nms: first_objectness={:.4}",
                final_objectness[0]
            );
            let first_class_scores: Vec<f32> = class_scores_2d.row(0).iter().copied().collect();
            let top_5_classes: Vec<(usize, f32)> = first_class_scores
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect::<Vec<_>>()
                .into_iter()
                .fold(Vec::new(), |mut acc, x| {
                    acc.push(x);
                    acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    acc.truncate(5);
                    acc
                });
            debug!(
                "[YOLO_INTERMEDIATE] post_nms: first_detection_top_5_classes={:?}",
                top_5_classes
            );
        }
    }

    (
        final_boxes,
        ndarray::Array1::from_vec(final_objectness),
        class_scores_2d,
    )
}

fn log_error_context_for_repro(output_buf: &[f32], error_msg: &str, details: &str) {
    let byte_slice = bytemuck::cast_slice(output_buf);
    let b64 = general_purpose::STANDARD.encode(byte_slice);
    error!("[REPRO_DATA] Error: {}", error_msg);
    error!("[REPRO_DATA] Details: {}", details);
    error!("[REPRO_DATA] Base64OfFloatBuffer: {}", b64);
}

fn get_nms_indices(
    box_scores: &ndarray::Array2<f32>,
    iou_threshold: f32,
    top_k: i32,
) -> Vec<usize> {
    let scores = box_scores.slice(s![.., -1]).to_owned();
    let boxes = box_scores.slice(s![.., ..-1]).to_owned();
    let mut picked = Vec::new();
    let mut indexes: Vec<usize> = (0..scores.shape()[0]).collect();
    // sort *ascending* so pop() cheaply returns highest score
    indexes.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

    while let Some(current) = indexes.pop() {
        picked.push(current);

        if (top_k > 0 && picked.len() as i32 == top_k) || indexes.is_empty() {
            break;
        }

        let current_box = boxes.slice(s![current, ..]);
        let rest_indices = indexes.clone();
        let rest_boxes = boxes.select(ndarray::Axis(0), &rest_indices);

        let iou = iou_matrix(
            &rest_boxes.view(),
            &current_box.to_owned().insert_axis(ndarray::Axis(0)).view(),
            1e-5,
        );
        let iou = iou.column(0);

        indexes.retain(|&idx| {
            if let Some(pos) = rest_indices.iter().position(|&x| x == idx) {
                iou[pos] <= iou_threshold
            } else {
                false
            }
        });
    }

    picked
}

const DETECT_INPUT_FIELD: &str = "image_buffer";
const DETECT_SHAPE_FIELD: &str = "input_shape";

pub fn preprocess_image(
    mut t: Tuple,
    fft_buffers_mutex: &Mutex<FftBuffers>,
    timing_stats_mutex: &Mutex<TimingStats>,
    fft_planner: &Arc<dyn Fft<f32>>,
    strategy: crate::streaming_features::FeatureExtractionStrategy,
) -> Vec<Tuple> {
    let mut v = get_tuple_vec();

    let Some(original_image) = t.get(ORIGINAL_IMAGE_FIELD) else {
        error!("Failed to extract original image");
        return v;
    };

    let Some(original_image) = original_image.as_shared_array_u8() else {
        error!("Failed to extract original image");
        return v;
    };
    let original_image_shape = original_image.shape();
    let Some(original_image_buf) = original_image.as_slice() else {
        error!("original image buffer was not contiguous");
        return v;
    };

    let height = original_image_shape[0] as u32;
    let width = original_image_shape[1] as u32;

    let start_time = std::time::Instant::now();
    let (preprocessed_image, letterbox_info) = yolo_preprocess(original_image_buf, width, height);
    let img_proc_micros = start_time.elapsed().as_micros();
    debug!(
        "preprocessing of image for tuple {} (id={:?}, w={:?}, h={:?}) took {:?} micros",
        t.id(),
        t.get(crate::animal_fields::ORIGINAL_IMAGE_ID_FIELD),
        width,
        height,
        img_proc_micros
    );

    // extract features
    {
        let mut buffers = fft_buffers_mutex.lock().unwrap();
        let mut stats = timing_stats_mutex.lock().unwrap();

        let img_view = ndarray::ArrayView3::from_shape(
            (height as usize, width as usize, 3),
            original_image_buf,
        )
        .unwrap();

        use crate::streaming_features::FeatureExtractionStrategy::*;

        let features_list: Vec<HabValue> = match strategy {
            Baseline => {
                let features = crate::streaming_features::extract_baseline(
                    &img_view,
                    fft_planner,
                    &mut buffers,
                    &mut stats,
                );
                features
                    .to_array()
                    .iter()
                    .map(|&val| HabValue::Float(OrderedFloat(val as f64)))
                    .collect()
            }
            Thirds => {
                let features = crate::streaming_features::extract_thirds(
                    &img_view,
                    fft_planner,
                    &mut buffers,
                    &mut stats,
                );
                features
                    .to_array()
                    .iter()
                    .map(|&val| HabValue::Float(OrderedFloat(val as f64)))
                    .collect()
            }
            LumaOnly => {
                let features = crate::streaming_features::extract_luma_only(
                    &img_view,
                    fft_planner,
                    &mut buffers,
                    &mut stats,
                );
                features
                    .to_array()
                    .iter()
                    .map(|&val| HabValue::Float(OrderedFloat(val as f64)))
                    .collect()
            }
            Fft1024 => {
                let features = crate::streaming_features::extract_fft1024(
                    &img_view,
                    fft_planner,
                    &mut buffers,
                    &mut stats,
                );
                features
                    .to_array()
                    .iter()
                    .map(|&val| HabValue::Float(OrderedFloat(val as f64)))
                    .collect()
            }
            MoreRows => {
                let features = crate::streaming_features::extract_more_rows(
                    &img_view,
                    fft_planner,
                    &mut buffers,
                    &mut stats,
                );
                features
                    .to_array()
                    .iter()
                    .map(|&val| HabValue::Float(OrderedFloat(val as f64)))
                    .collect()
            }
        };

        t.insert("streaming_features".into(), HabValue::List(features_list));
    }

    let preprocessed_image_shape = preprocessed_image.shape().to_vec();
    let (preprocessed_image_buf, extra) = preprocessed_image.into_raw_vec_and_offset();
    if let Some(extra @ 1..) = extra {
        error!("Failed to extract preprocessed image buffer: {:?}", extra);
        return v;
    }

    if !INPUT_DUMPED.load(Ordering::Relaxed) {
        log_error_context_for_repro(
            &preprocessed_image_buf,
            "DEBUG_DUMP_INPUT_TENSOR",
            "First image preprocessed input (1x3x640x640)",
        );
        INPUT_DUMPED.store(true, Ordering::Relaxed);
    }

    t.insert(
        DETECT_INPUT_FIELD.into(),
        HabValue::IntBuffer(bytemuck::cast_vec(preprocessed_image_buf)),
    );
    t.insert(
        DETECT_SHAPE_FIELD.into(),
        HabValue::ShapeBuffer(preprocessed_image_shape),
    );
    t.insert(
        "letterbox_ratio".into(),
        HabValue::from(letterbox_info.ratio as f64),
    );
    t.insert(
        "letterbox_pad_x".into(),
        HabValue::from(letterbox_info.pad_x as f64),
    );
    t.insert(
        "letterbox_pad_y".into(),
        HabValue::from(letterbox_info.pad_y as f64),
    );

    v.push(t);
    v
}
