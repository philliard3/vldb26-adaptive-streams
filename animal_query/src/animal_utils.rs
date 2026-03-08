#![allow(unused_imports)]
use dashmap::DashMap;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::LazyLock;
use watershed_shared::{
    basic_pooling::{get_tuple, get_tuple_vec, return_tuple},
    caching::StrToKey,
    devec::DeVec as Queue,
    global_logger::LimitedHabValue,
    AggregationResult, HabValue, Tuple,
};

// extract_image_info UDF
pub(crate) fn extract_image_info_noop(input: Tuple) -> Vec<Tuple> {
    let mut outputs = get_tuple_vec();
    outputs.push(input);
    outputs
}

// apply_softmax UDF
const NO_OBJECT_FOUND_INDEX: usize = 0;
const NO_OBJECT_FOUND_VALUES: &[f32] = &[1.0, 0.0]; // assuming 5 classes total
const MINIMUM_LOGIT_THRESHOLD: f32 = 0.0001;
const MINIMUM_PROBABILITY_THRESHOLD: f64 = 0.01;
static YOLO_MIN_CLASS_PROB_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(MINIMUM_PROBABILITY_THRESHOLD.to_bits());
static YOLO_MAX_CLASSES_PER_DET: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(3);
static YOLO_NAME_TO_ID: LazyLock<DashMap<String, usize>> = LazyLock::new(DashMap::new);
static USE_DENSE_GROUND_TRUTH_LOGGING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static LOG_INTERMEDIATE_YOLO_DATA: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static YOLO_FILTER_ANIMALS_ONLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Normalization strategy for per-detection class probabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NormalizationStrategy {
    /// No normalization - raw model probabilities (default)
    None = 0,
    /// L1 normalization - divide by sum so classes sum to 1.0
    L1 = 1,
    /// Softmax - apply softmax function (exp then normalize)
    Softmax = 2,
}

impl NormalizationStrategy {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::L1,
            2 => Self::Softmax,
            _ => Self::None,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "l1" => Self::L1,
            "softmax" => Self::Softmax,
            "none" | "" => Self::None,
            _ => {
                warn!("Unknown normalization strategy '{}', using None", s);
                Self::None
            }
        }
    }
}

static YOLO_PROB_NORMALIZE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(NormalizationStrategy::None as u8);
static YOLO_INPUT_NORMALIZATION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

pub fn set_yolo_filter_animals_only(filter: bool) {
    YOLO_FILTER_ANIMALS_ONLY.store(filter, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_yolo_input_normalization(normalize: bool) {
    YOLO_INPUT_NORMALIZATION.store(normalize, std::sync::atomic::Ordering::Relaxed);
}

pub fn get_yolo_input_normalization() -> bool {
    YOLO_INPUT_NORMALIZATION.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_yolo_prob_normalize(strategy: NormalizationStrategy) {
    YOLO_PROB_NORMALIZE.store(strategy as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn get_yolo_prob_normalize() -> NormalizationStrategy {
    NormalizationStrategy::from_u8(YOLO_PROB_NORMALIZE.load(std::sync::atomic::Ordering::Relaxed))
}

pub(crate) fn set_yolo_name_to_id_map(map: &HashMap<String, usize>) {
    YOLO_NAME_TO_ID.clear();
    for (name, id) in map {
        YOLO_NAME_TO_ID.insert(name.clone(), *id);
    }
}

fn get_yolo_class_id_for_name(name: &str) -> Option<usize> {
    YOLO_NAME_TO_ID.get(name).map(|v| *v.value())
}

pub(crate) fn set_yolo_min_class_probability(min_prob: f64) {
    let min_prob = min_prob.clamp(0.0, 1.0);
    YOLO_MIN_CLASS_PROB_BITS.store(min_prob.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn set_yolo_max_classes_per_det(max_classes: usize) {
    let max_classes = max_classes.max(1);
    YOLO_MAX_CLASSES_PER_DET.store(max_classes, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn set_use_dense_ground_truth_logging(use_dense: bool) {
    USE_DENSE_GROUND_TRUTH_LOGGING.store(use_dense, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn set_log_intermediate_yolo_data(log_intermediate: bool) {
    LOG_INTERMEDIATE_YOLO_DATA.store(log_intermediate, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn should_log_intermediate_yolo_data() -> bool {
    LOG_INTERMEDIATE_YOLO_DATA.load(std::sync::atomic::Ordering::Relaxed)
}
pub(crate) fn apply_softmax_old(mut input: Tuple) -> Vec<Tuple> {
    let mut outputs = get_tuple_vec();
    // extract original_image_id
    let Some(original_image_id) = input.remove("original_image_id") else {
        error!(
            "Missing original_image_id field in tuple with id {} for apply_softmax UDF",
            input.id()
        );
        return_tuple(input);
        return outputs;
    };
    let Some(original_image_id) = original_image_id.as_string().cloned() else {
        error!(
            "original_image_id field is not a string in tuple with id {} for apply_softmax UDF",
            input.id()
        );
        return_tuple(input);
        return outputs;
    };

    const RAW_BOXES_BUFFER_FIELD: &str = "prediction_boxes";
    const RAW_BOXES_SHAPE_FIELD: &str = "prediction_boxes_shape";
    const EXPECTED_BOXES_SHAPE: &[usize] = &[1, 100, 4]; // [batch_size, num_boxes, 4]

    const SCORES_BUFFER_FIELD: &str = "prediction_logits";
    const SCORES_SHAPE_FIELD: &str = "prediction_logits_shape";
    const EXPECTED_SCORES_SHAPE: &[usize] = &[1, 100, 92]; // [batch_size, num_boxes, num_classes]

    let t: &Tuple = &input;
    let Some(box_buffer) = t.get(RAW_BOXES_BUFFER_FIELD) else {
        error!("Failed to extract boxes");
        return_tuple(input);
        return outputs;
    };
    let Some(boxes_shape) = t.get(RAW_BOXES_SHAPE_FIELD) else {
        error!("Failed to extract boxes shape");
        return_tuple(input);
        return outputs;
    };
    let Some(box_buffer) = box_buffer.as_int_buffer() else {
        error!("Failed to extract boxes");
        return_tuple(input);
        return outputs;
    };
    let box_buffer: &[f32] = bytemuck::cast_slice(box_buffer);
    let Some(boxes_shape) = boxes_shape.as_shape_buffer() else {
        error!("Failed to extract boxes shape");
        return_tuple(input);
        return outputs;
    };

    let Some(scores_buffer) = t.get(SCORES_BUFFER_FIELD) else {
        error!("Failed to extract scores");
        return_tuple(input);
        return outputs;
    };
    let Some(scores_shape) = t.get(SCORES_SHAPE_FIELD) else {
        error!("Failed to extract scores shape");
        return_tuple(input);
        return outputs;
    };
    let Some(scores_buffer) = scores_buffer.as_int_buffer() else {
        error!("Failed to extract scores");
        return_tuple(input);
        return outputs;
    };
    let scores_buffer: &[f32] = bytemuck::cast_slice(scores_buffer);
    let Some(scores_shape) = scores_shape.as_shape_buffer() else {
        error!("Failed to extract scores shape");
        return_tuple(input);
        return outputs;
    };

    if boxes_shape != EXPECTED_BOXES_SHAPE {
        error!(
            "Unexpected boxes shape: {:?}, expected: {:?}",
            boxes_shape, EXPECTED_BOXES_SHAPE
        );
        return_tuple(input);
        return outputs;
    }
    if scores_shape != EXPECTED_SCORES_SHAPE {
        error!(
            "Unexpected scores shape: {:?}, expected: {:?}",
            scores_shape, EXPECTED_SCORES_SHAPE
        );
        return_tuple(input);
        return outputs;
    }

    let boxes_arr: ndarray::ArrayView2<f32> =
        match ndarray::ArrayView2::from_shape((boxes_shape[1], boxes_shape[2]), box_buffer) {
            Ok(arr) => arr,
            Err(e) => {
                error!("Failed to create ndarray for boxes: {}", e);
                return_tuple(input);
                return outputs;
            }
        };
    let scores_ndarray: ndarray::ArrayView2<f32> =
        match ndarray::ArrayView2::from_shape((scores_shape[1], scores_shape[2]), scores_buffer) {
            Ok(arr) => arr,
            Err(e) => {
                error!("Failed to create ndarray for scores: {}", e);
                return_tuple(input);
                return outputs;
            }
        };

    for object_idx in 0..scores_shape[1] {
        // we do not actually use the boxes in this query. we only care about the count
        let _box_arr = boxes_arr.row(object_idx);
        let scores_arr = scores_ndarray.row(object_idx);
        if NO_OBJECT_FOUND_VALUES
            .iter()
            .any(|&v| (scores_arr[NO_OBJECT_FOUND_INDEX] - v).abs() < f32::EPSILON)
        {
            // no object found. skipping
            // debug!("No object found for object index {}", object_idx);
            continue;
        }
        // apply softmax to scores_arr
        let normalizing_factor: f64 = scores_arr
            .iter()
            // apply threshold to logits
            .filter(|&&x| x >= MINIMUM_LOGIT_THRESHOLD)
            .map(|&x| (x as f64).exp())
            .sum();
        for class_id in 0..scores_shape[2] {
            let probability = scores_arr[class_id].exp() as f64 / normalizing_factor;
            if probability < MINIMUM_PROBABILITY_THRESHOLD {
                // skip low probability classes
                continue;
            }
            let mut new_tuple = get_tuple();
            new_tuple.insert("class_id".into(), class_id.to_string().into());
            new_tuple.insert("probability".into(), probability.into());
            new_tuple.insert("original_image_id".into(), original_image_id.clone().into());
            outputs.push(new_tuple);
        }
    }

    return_tuple(input);
    outputs
}

// new alternative to apply_softmax. we take only the "outputs" and "outputs_shape" fields and pass them to the functions in yolos_utils.rs to get final class probabilities
// and then use that to split the input tuple into multiple tuples
pub(crate) fn split_tuple_yolos(mut input: Tuple) -> Vec<Tuple> {
    let mut output_tuples = get_tuple_vec();
    let original_tuple_id_for_feedback = input
        .get("original_tuple_id")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or_else(|| input.id());

    // extract original_image_id (img_id field)
    let Some(original_image_id) = input.remove("img_id") else {
        error!(
            "Missing img_id field in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };
    let Some(original_image_id) = original_image_id.as_string().cloned() else {
        error!(
            "img_id field is not a string in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };

    // extract outputs and outputs_shape (from ONNX model output)
    let Some(outputs_buffer) = input.remove("outputs") else {
        error!(
            "Missing outputs field in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };
    let Some(outputs_shape) = input.remove("outputs_shape") else {
        error!(
            "Missing outputs_shape field in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };

    // extract image dimensions for coordinate scaling
    let Some(original_width) = input.remove("original_width").and_then(|v| v.as_integer()) else {
        error!(
            "Missing original_width field in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };
    let Some(original_height) = input.remove("original_height").and_then(|v| v.as_integer()) else {
        error!(
            "Missing original_height field in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };

    let Some(outputs_int_buffer) = outputs_buffer.as_int_buffer() else {
        error!(
            "outputs field is not an int buffer in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };
    let outputs_f32_buffer: &[f32] = bytemuck::cast_slice(outputs_int_buffer);

    let Some(outputs_shape_buffer) = outputs_shape.as_shape_buffer() else {
        error!(
            "outputs_shape field is not a shape buffer in tuple with id {} for split_tuple_yolos UDF",
            input.id()
        );
        return_tuple(input);
        return output_tuples;
    };

    let letterbox_ratio = input
        .get("letterbox_ratio")
        .and_then(|v| v.as_float())
        .map(|v| v.0 as f32)
        .unwrap_or(1.0);
    let letterbox_pad_x = input
        .get("letterbox_pad_x")
        .and_then(|v| v.as_float())
        .map(|v| v.0 as f32)
        .unwrap_or(0.0);
    let letterbox_pad_y = input
        .get("letterbox_pad_y")
        .and_then(|v| v.as_float())
        .map(|v| v.0 as f32)
        .unwrap_or(0.0);

    let yolo_conf_threshold = crate::yolos_utils::get_yolo_conf_threshold();

    let (_boxes, _objectness_scores, all_class_scores) = crate::yolos_utils::yolo_postprocess(
        outputs_f32_buffer,
        &outputs_shape_buffer,
        original_width as i32,
        original_height as i32,
        yolo_conf_threshold,
        letterbox_ratio,
        letterbox_pad_x,
        letterbox_pad_y,
    );

    // Extract expected counts if available (embedded from metadata during image loading)
    // Format: "expected_count_<class_id>" → integer value
    let mut expected_counts_map: std::collections::HashMap<usize, i32> = input
        .keys()
        .filter_map(|key| {
            let key_str = key.to_string();
            if key_str.starts_with("expected_count_") {
                let class_id_str = key_str.strip_prefix("expected_count_")?;
                let class_id: usize = class_id_str.parse().ok()?;
                input
                    .get(key)
                    .and_then(|v| v.as_integer())
                    .map(|count| (class_id, count))
            } else {
                None
            }
        })
        .collect();

    if expected_counts_map.is_empty() {
        for key in input.keys() {
            let key_str = key.to_string();
            let Some(class_name) = key_str.strip_prefix("ground_truth_class_") else {
                continue;
            };
            let Some(class_id) = get_yolo_class_id_for_name(class_name) else {
                warn!(
                    "No YOLO class id for ground truth class name: {}",
                    class_name
                );
                continue;
            };
            if let Some(count) = input.get(key).and_then(|v| v.as_integer()) {
                expected_counts_map.insert(class_id, count);
            }
        }
    }
    if expected_counts_map.is_empty() {
        let available_keys: Vec<_> = input.keys().collect();
        warn!(
            "No expected counts found for tuple id {}. Available keys: {:?}",
            input.id(),
            available_keys
        );
    }
    let mut intra_image_detected_counts_debugging_map: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();

    let min_prob =
        f64::from_bits(YOLO_MIN_CLASS_PROB_BITS.load(std::sync::atomic::Ordering::Relaxed));
    let max_classes_per_det = YOLO_MAX_CLASSES_PER_DET.load(std::sync::atomic::Ordering::Relaxed);

    let filter_animals_only = YOLO_FILTER_ANIMALS_ONLY.load(std::sync::atomic::Ordering::Relaxed);
    let normalize_strategy = get_yolo_prob_normalize();

    for object_idx in 0..all_class_scores.dim().0 {
        let class_scores = all_class_scores.row(object_idx);
        let mut scored_classes: Vec<(usize, f64)> = Vec::with_capacity(class_scores.len());

        for (class_id, &score) in class_scores.iter().enumerate() {
            let mut score_f64 = score as f64;

            // Check threshold first to save work
            if score_f64 < min_prob {
                continue;
            }

            // Apply Animals-Only Filter
            if filter_animals_only {
                // COCO animal classes: 14..=23 (0-indexed)
                if class_id < 14 || class_id > 23 {
                    continue;
                }
            }

            scored_classes.push((class_id, score_f64));
        }

        // Sort descending
        scored_classes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to Top-K (max_classes_per_det)
        if scored_classes.len() > max_classes_per_det {
            scored_classes.truncate(max_classes_per_det);
        }

        // Apply Normalization based on strategy
        if !scored_classes.is_empty() {
            match normalize_strategy {
                NormalizationStrategy::None => {
                    // No normalization - keep raw probabilities
                }
                NormalizationStrategy::L1 => {
                    // L1 normalization: divide by sum so classes sum to 1.0
                    let sum_p: f64 = scored_classes.iter().map(|c| c.1).sum();
                    if sum_p > 1e-9 {
                        for c in &mut scored_classes {
                            c.1 /= sum_p;
                        }
                    }
                }
                NormalizationStrategy::Softmax => {
                    // Softmax: exp(x) / sum(exp(x))
                    // First find max for numerical stability
                    let max_score = scored_classes
                        .iter()
                        .map(|c| c.1)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let exp_scores: Vec<f64> = scored_classes
                        .iter()
                        .map(|c| (c.1 - max_score).exp())
                        .collect();
                    let sum_exp: f64 = exp_scores.iter().sum();
                    if sum_exp > 1e-9 {
                        for (i, c) in scored_classes.iter_mut().enumerate() {
                            c.1 = exp_scores[i] / sum_exp;
                        }
                    }
                }
            }
        }

        // Emit tuples
        for (class_id, probability) in scored_classes {
            let mut new_tuple = get_tuple();

            // Include expected count if available for logging
            new_tuple.insert("class_id".into(), class_id.to_string().into());
            new_tuple.insert("probability".into(), probability.into());
            new_tuple.insert("original_image_id".into(), original_image_id.clone().into());
            new_tuple.insert(
                "original_tuple_id".into(),
                HabValue::Integer(original_tuple_id_for_feedback as _),
            );

            *intra_image_detected_counts_debugging_map
                .entry(class_id)
                .or_default() += probability;

            output_tuples.push(new_tuple);
        }
    }

    // Always emit one dummy tuple for merge telemetry continuity.
    // This ensures merge callback can report completion keyed by original_tuple_id
    // even when no detection tuple is emitted.
    let mut dummy_tuple = get_tuple();
    dummy_tuple.insert("class_id".into(), "0".into());
    dummy_tuple.insert("probability".into(), 0.0.into());
    dummy_tuple.insert("original_image_id".into(), original_image_id.clone().into());
    dummy_tuple.insert(
        "original_tuple_id".into(),
        HabValue::Integer(original_tuple_id_for_feedback as _),
    );
    output_tuples.push(dummy_tuple);

    // STRING-BASED DEBUG LOGGING: Log the full expected/detected maps as formatted strings
    if !expected_counts_map.is_empty() {
        let expected_str: String = expected_counts_map
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        let detected_str: String = intra_image_detected_counts_debugging_map
            .iter()
            .map(|(k, v)| format!("{}:{:.2}", k, v))
            .collect::<Vec<_>>()
            .join(",");
        debug!(
            "[GROUND_TRUTH_DEBUG] image_id={}, tuple_id={}, expected=[{}], detected=[{}]",
            original_image_id,
            input.id(),
            expected_str,
            detected_str
        );
    }

    // log expected vs detected counts for debugging
    if !expected_counts_map.is_empty() {
        use watershed_shared::global_logger;

        let tuple_id = input.id();
        let log_location = "ground_truth_comparison".to_raw_key();

        let use_dense = USE_DENSE_GROUND_TRUTH_LOGGING.load(std::sync::atomic::Ordering::Relaxed);

        // Build aux_data with per-class comparison
        let mut aux_data_map = std::collections::HashMap::new();
        aux_data_map.insert(
            "image_id".to_raw_key(),
            LimitedHabValue::String(original_image_id.to_raw_key().into()),
        );

        if use_dense {
            // DENSE FORMAT: Log all 80 YOLO classes (0-79) with zeros for missing ones
            // This ensures uniform fields across all log entries
            const NUM_YOLO_CLASSES: usize = 80;
            for class_id in 0..NUM_YOLO_CLASSES {
                let expected_count = expected_counts_map.get(&class_id).copied().unwrap_or(0);
                let detected_count = intra_image_detected_counts_debugging_map
                    .get(&class_id)
                    .copied()
                    .unwrap_or(0.0);

                aux_data_map.insert(
                    format!("class_{}_expected", class_id).to_raw_key(),
                    LimitedHabValue::Integer(expected_count as _),
                );
                aux_data_map.insert(
                    format!("class_{}_detected", class_id).to_raw_key(),
                    LimitedHabValue::Float(detected_count.into()),
                );
            }
        } else {
            // SPARSE FORMAT: Only log classes that are present (original behavior)
            // Note: This may cause issues with object-of-arrays logging format
            for (class_id, expected_count) in expected_counts_map.iter() {
                let detected_count = intra_image_detected_counts_debugging_map
                    .get(class_id)
                    .cloned()
                    .unwrap_or(0.0);

                // Log both expected and detected for this class
                aux_data_map.insert(
                    format!("class_{}_expected", class_id).to_raw_key(),
                    LimitedHabValue::Integer(*expected_count as _),
                );
                aux_data_map.insert(
                    format!("class_{}_detected", class_id).to_raw_key(),
                    LimitedHabValue::Float(detected_count.into()),
                );
            }
        }

        if let Err(e) = global_logger::log_data(tuple_id, log_location, Some(aux_data_map)) {
            for err in e {
                error!(
                    "Failed to log ground truth comparison for image {}: {}",
                    original_image_id, err
                );
            }
        }
    }

    return_tuple(input);
    output_tuples
}

pub(crate) enum CountCondition {
    UseProbability, // uses probability to weight counts
    SimpleCount,    // uses just count of occurrences
}

pub(crate) struct LimitInfo {
    count: usize,
    duration_millis: Option<u64>,
}

impl LimitInfo {
    pub(crate) fn new(count: usize, duration_millis: Option<u64>) -> Self {
        LimitInfo {
            count,
            duration_millis,
        }
    }
}

// count_species UDF
// already grouped by class_id/species at this point
// we need to extract the probability and add that to our approximate count
pub(crate) fn aggregate_count_species(
    windowed_data: &mut Queue<Tuple>,
    count_condition: CountCondition,
    limit_info: LimitInfo,
) -> AggregationResult {
    // first cull data based on limit info if needed

    // sort by timestamp
    windowed_data.sort_by_key(|t| t.unix_time_created_ns());
    // compute cutoff time
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    if let Some(duration_millis) = limit_info.duration_millis {
        let cutoff_time = now.saturating_sub((duration_millis as u128) * 1_000_000);
        // remove older data
        while let Some(front) = windowed_data.first() {
            if front.unix_time_created_ns() < cutoff_time {
                windowed_data.pop_front();
            } else {
                break;
            }
        }
    }

    // limit remainder by count
    if windowed_data.len() >= limit_info.count {
        // TODO: can likely be made more efficient by using drain or similar
        for _ in 0..(windowed_data.len() - limit_info.count) {
            windowed_data.pop_front();
        }
    }

    if windowed_data.is_empty() {
        return AggregationResult {
            emit: None,
            is_finished: false,
        };
    }

    let Some(class_id) = windowed_data[0].get("class_id") else {
        error!(
            "Missing class_id field in tuple with id {} for count_species aggregation",
            windowed_data[0].id()
        );
        return AggregationResult {
            emit: None,
            is_finished: false,
        };
    };
    let Some(class_id_str) = class_id.as_string().cloned() else {
        error!(
            "class_id field is not a string in tuple with id {} for count_species aggregation",
            windowed_data[0].id()
        );
        return AggregationResult {
            emit: None,
            is_finished: false,
        };
    };

    let class_count = match count_condition {
        CountCondition::UseProbability => {
            let mut class_count = 0.0f64;
            for tuple in windowed_data.iter() {
                let Some(probability) = tuple.get("probability").and_then(|v| v.as_float()) else {
                    error!(
                        "Missing probability field in tuple with id {} for UseProbability count condition",
                        tuple.id()
                    );
                    // continue;
                    return AggregationResult {
                        emit: None,
                        is_finished: false,
                    };
                };
                class_count += probability.0;
            }
            class_count
        }
        CountCondition::SimpleCount => windowed_data.len() as f64,
    };

    // Aggregate expected count from tuples if available (for comparison/logging)
    let mut total_expected_count: i32 = 0;
    for tuple in windowed_data.iter() {
        if let Some(expected) = tuple.get("expected_count").and_then(|v| v.as_integer()) {
            total_expected_count = total_expected_count.max(expected);
        }
    }

    let mut output_tuple = get_tuple();
    output_tuple.insert("count".into(), class_count.into());
    output_tuple.insert("class_id".into(), class_id_str.into());

    // Include expected count for downstream logging/verification
    if total_expected_count > 0 {
        output_tuple.insert(
            "expected_count".into(),
            HabValue::Integer(total_expected_count),
        );
    }

    let mut output_vec = get_tuple_vec();
    output_vec.push(output_tuple);
    AggregationResult {
        emit: Some(output_vec),
        is_finished: false,
    }
}
