#![allow(unused_labels)]

mod animal_fields;
mod animal_utils;
// mod complexity_utils;
mod streaming_features;
mod yolos_utils;

use crate::streaming_features::{FftBuffers, TimingStats};
use rustfft::Fft;

use dashmap::DashMap;
use futures::StreamExt;
use image::RgbImage;
use rand::seq::SliceRandom;
use rand::Rng;
use watershed_shared::async_query_builder::FunctionKinds;
use watershed_shared::async_query_builder::PhysicalOperator;
use watershed_shared::async_query_builder::RuntimeState;
use watershed_shared::basic_pooling::get_tuple;
use watershed_shared::basic_pooling::get_tuple_vec;
use watershed_shared::basic_pooling::return_tuple_vec;
use watershed_shared::caching::StrToKey;
use watershed_shared::global_logger::LimitedHabValue;
use watershed_shared::query_builder::QueryDescriptor;
use watershed_shared::scheduler::AsyncPipeSendError;
use watershed_shared::scheduler::{
    self,
    basic_probability_forecast::{BasicCategory, History, PastData},
    AlgInputs, BinInfo, FutureWindowKind, ShareableArray,
};
use watershed_shared::start_python_with_modules;
use watershed_shared::{query_builder, AsyncPipe, Operator, UdfBolt};
use watershed_shared::{HabString, HabValue, Tuple};

#[allow(unused_imports)]
use polars::prelude::*;

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::collections::BTreeMap;

use std::collections::BTreeSet;
use std::env::args;

use std::sync::atomic::{self, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::watch;

use crate::animal_fields::{
    EXPECTED_MATCHES_FIELD, ORIGINAL_IMAGE_FIELD, ORIGINAL_IMAGE_ID_FIELD,
    ORIGINAL_IMAGE_ID_INT_FIELD, ORIGINAL_IMAGE_SHAPE_FIELD,
};

#[derive(Debug, Deserialize, Serialize)]
struct BlockingNoopBoltConfig {
    operator_name: String,
    delay_micros: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct RehydrateSpoutConfig {
    operator_name: String,
    rehydrate_path: String,
    #[serde(default = "RehydrateFormatOptions::default")]
    options: RehydrateFormatOptions,
    // enable this to allow for otherwise, it will use the relative times from the tuples' original run
    #[serde(default)]
    use_face_config_timestamps: RehydrateTimingOptions,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
enum RehydrateTimingOptions {
    #[serde(rename = "use_original_timestamps")]
    SimulateOriginalTiming,
    #[serde(rename = "use_face_config_timestamps")]
    FaceConfig {
        #[serde(default)]
        with_adjustment: i128,
    },
}

impl Default for RehydrateTimingOptions {
    fn default() -> Self {
        RehydrateTimingOptions::SimulateOriginalTiming
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
enum RehydrateFormatOptions {
    #[serde(rename = "flat_base64_json")]
    Base64JsonStringArray,
    // #[default]
    #[serde(rename = "log_base64_json")]
    NestedLogBase64Json { tuple_field: String },
}
impl Default for RehydrateFormatOptions {
    fn default() -> Self {
        RehydrateFormatOptions::NestedLogBase64Json {
            tuple_field: "binary".into(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(untagged)]
enum Delay {
    Fixed(u64),
    Variable(VariableDelay),
}

impl Delay {
    fn max(&self) -> u64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Variable(
                VariableDelay::Sinusoidal { upper, .. } | VariableDelay::Random { upper, .. },
            ) => *upper,
        }
    }
    fn min(&self) -> u64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Variable(
                VariableDelay::Sinusoidal { lower, .. } | VariableDelay::Random { lower, .. },
            ) => *lower,
        }
    }
    fn starting_delay(&mut self) -> u64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Variable(VariableDelay::Sinusoidal {
                start_at,
                current_value,
                ..
            }) => {
                *current_value = *start_at;
                *start_at
            }
            Self::Variable(VariableDelay::Random { upper, lower, .. }) => {
                rand::Rng::gen_range(&mut rand::thread_rng(), *lower..*upper)
            }
        }
    }
    fn max_streak(&self) -> usize {
        match self {
            Self::Fixed(_) => usize::MAX,
            Self::Variable(VariableDelay::Sinusoidal { maintain_for, .. }) => *maintain_for,
            Self::Variable(VariableDelay::Random { maintain_for, .. }) => *maintain_for,
        }
    }
    fn next_delay(&mut self) -> u64 {
        match self {
            Self::Fixed(v) => *v,
            Self::Variable(VariableDelay::Sinusoidal {
                increment_by,
                upper,
                lower,
                increasing,
                current_value,
                ..
            }) => {
                if *increasing {
                    let Some(next_value) = current_value.checked_add(*increment_by) else {
                        error!("Sinusoidal made it to u64 max!");
                        return u64::MAX;
                    };
                    if next_value >= *upper {
                        *increasing = false;
                        *current_value = *upper;
                        return *current_value;
                    }
                    *current_value = next_value;
                    next_value
                } else {
                    let Some(next_value) = current_value.checked_sub(*increment_by) else {
                        *increasing = true;
                        *current_value = 0;
                        return 0;
                    };
                    if next_value <= *lower {
                        *increasing = true;
                        *current_value = *lower;
                        return *current_value;
                    }
                    *current_value = next_value;
                    next_value
                }
            }
            Self::Variable(VariableDelay::Random { upper, lower, .. }) => {
                rand::Rng::gen_range(&mut rand::thread_rng(), *lower..=*upper)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(tag = "type")]
enum VariableDelay {
    Sinusoidal {
        #[serde(default)]
        current_value: u64,
        upper: u64,
        lower: u64,
        start_at: u64,
        #[serde(default = "variable_delay_fns::one")]
        maintain_for: usize,
        #[serde(default)]
        increasing: bool,
        #[serde(default = "variable_delay_fns::one_thousand_u64")]
        increment_by: u64,
    },
    Random {
        upper: u64,
        lower: u64,
        // we give a little more time to adjust when it's entirely random
        #[serde(default = "variable_delay_fns::ten")]
        maintain_for: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
enum RoutingOptions {
    Fixed(usize),
    Named(NamedRoutingOptions),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
enum NamedRoutingOptions {
    #[serde(rename = "huge")]
    AlwaysHuge,
    // default is removed here, handled in wrapper default
    #[serde(rename = "big")]
    AlwaysBig,
    #[serde(rename = "small")]
    AlwaysSmall,
    #[serde(rename = "tiny")]
    AlwaysTiny,
    #[serde(rename = "random")]
    Random,
    #[serde(rename = "eddies")]
    Eddies,
    #[serde(rename = "aquifer_greedy")]
    AquiferGreedy,
    #[serde(rename = "aquifer_optimal")]
    AquiferOptimal,
    #[serde(rename = "predictor_binary")]
    PredictorBinary,
    #[serde(rename = "predictor_probabilistic")]
    PredictorProbabilistic,
    #[serde(rename = "drop")]
    AlwaysDrop,
}

impl Default for RoutingOptions {
    fn default() -> Self {
        RoutingOptions::Named(NamedRoutingOptions::AlwaysBig)
    }
}

impl<'de> Deserialize<'de> for RoutingOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RoutingOptionsVisitor;

        impl<'de> serde::de::Visitor<'de> for RoutingOptionsVisitor {
            type Value = RoutingOptions;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an integer (pipe index) or a routing strategy string")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(RoutingOptions::Fixed(value as usize))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value < 0 {
                    return Err(E::custom(format!(
                        "routing index cannot be negative: {}",
                        value
                    )));
                }
                Ok(RoutingOptions::Fixed(value as usize))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                // Delegate to NamedRoutingOptions deserializer or manually match
                // Manual matching is cleaner given we want to merge them
                match value {
                    "huge" => Ok(RoutingOptions::Named(NamedRoutingOptions::AlwaysHuge)),
                    "big" => Ok(RoutingOptions::Named(NamedRoutingOptions::AlwaysBig)),
                    "small" => Ok(RoutingOptions::Named(NamedRoutingOptions::AlwaysSmall)),
                    "tiny" => Ok(RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny)),
                    "random" => Ok(RoutingOptions::Named(NamedRoutingOptions::Random)),
                    "eddies" => Ok(RoutingOptions::Named(NamedRoutingOptions::Eddies)),
                    "aquifer_greedy" => {
                        Ok(RoutingOptions::Named(NamedRoutingOptions::AquiferGreedy))
                    }
                    "aquifer_optimal" => {
                        Ok(RoutingOptions::Named(NamedRoutingOptions::AquiferOptimal))
                    }
                    "predictor_binary" => {
                        Ok(RoutingOptions::Named(NamedRoutingOptions::PredictorBinary))
                    }
                    "predictor_probabilistic" => Ok(RoutingOptions::Named(
                        NamedRoutingOptions::PredictorProbabilistic,
                    )),
                    "drop" => Ok(RoutingOptions::Named(NamedRoutingOptions::AlwaysDrop)),
                    _ => Err(E::custom(format!("unknown routing option: {}", value))),
                }
            }
        }

        deserializer.deserialize_any(RoutingOptionsVisitor)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct MergeInfo {
    tuple_id: usize,
    // person_id: usize,
    // sequence_id: u128,
    label: usize,
    pipeline_id: usize,
    time_merged: u128,
}

mod variable_delay_fns {
    pub(crate) const fn one() -> usize {
        1
    }
    pub(crate) const fn ten() -> usize {
        10
    }
    pub(crate) const fn one_thousand_u64() -> u64 {
        1_000
    }
}

// Temporary adapter struct for image loading pipeline
// TODO: Remove once paths_to_ndarrays_v2 is refactored for AnimalImageInfo
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageInfo {
    person_name: String,
    img_id: String,
    img_path: String,
}

// ===== ANIMAL QUERY STRUCTS =====

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
enum CountCondition {
    #[serde(rename = "probability")]
    UseProbability,
    #[serde(rename = "simple")]
    SimpleCount,
}

impl Default for CountCondition {
    fn default() -> Self {
        CountCondition::UseProbability
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AnimalImageInfo {
    image_id: usize,
    file_name: String,
    relative_path: String,
    animal_class_counts: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnimalImageMetadata {
    image_data: Vec<AnimalImageInfo>,
    #[serde(default)]
    coco_id_to_name_map: std::collections::HashMap<String, String>,
    #[serde(default)]
    coco_name_to_id_map: std::collections::HashMap<String, usize>,
    #[serde(default)]
    yolo_id_to_coco_id_map: std::collections::HashMap<String, usize>,
    #[serde(default)]
    yolo_name_to_id_map: std::collections::HashMap<String, usize>,
    #[serde(default)]
    coco_id_to_yolo_id_map: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnimalExperimentConfig {
    run_order_seed: Option<u64>,
    animal_metadata_path: String,
    preclassifier_path: Option<String>,
    animal_image_base_path: String,
    query_path: String,
    max_total_samples: Option<usize>,
    history_window_size: Option<usize>,
    greedy_lookahead_window_size: Option<usize>,
    optimal_lookahead_window_size: Option<usize>,
    lookahead_time_ms: Option<u64>,
    deadline_window_ms: Option<u64>,
    target_time_micros: Option<Delay>,
    input_delay_micros: Option<Delay>,
    overall_time_limit_ms: Option<u64>,
    initial_startup_delay_ms: Option<u64>,
    routing_strategy: Option<RoutingOptions>,
    log_folder: Option<HabString>,
    blocking_noops: Option<Vec<BlockingNoopBoltConfig>>,
    reyhydrate_spouts: Option<Vec<RehydrateSpoutConfig>>,
    // Animal-specific window params
    #[serde(default = "default_species_count_window_ms")]
    species_count_window_ms: u64,
    #[serde(default = "default_species_count_max_items")]
    species_count_max_items: usize,
    #[serde(default)]
    species_count_condition: CountCondition,
    #[serde(default)]
    debug_use_ground_truth: bool,
    max_det: Option<usize>,
    yolo_conf_threshold: Option<f32>,
    yolo_min_class_probability: Option<f64>,
    yolo_max_classes_per_det: Option<usize>,
    #[serde(default)]
    use_dense_ground_truth_logging: bool,
    #[serde(default)]
    log_intermediate_yolo_data: bool,
    #[serde(default)]
    exclude_purged_items_from_history: bool,

    use_agnostic_nms: Option<bool>,
    filter_animals_only: Option<bool>,
    /// Normalization strategy: "none", "l1", or "softmax"
    /// For backward compatibility, also accepts boolean (false="none", true="l1")
    #[serde(default, deserialize_with = "deserialize_normalization_strategy")]
    normalization_strategy: Option<String>,

    /// Whether to normalize input images by dividing by 255.0 (default: true)
    #[serde(default)]
    yolo_input_normalization: Option<bool>,

    /// Strategy for streaming feature extraction (default: Thirds)
    #[serde(default)]
    feature_extraction_strategy: Option<crate::streaming_features::FeatureExtractionStrategy>,

    /// Option to reroute traffic from pipe 0 (drop) to other pipes.
    /// Values: "ignored" (default), "first", "random"
    reroute_zero_option: Option<String>,

    /// Whether to cache image->ndarray conversions in memory for reuse.
    #[serde(default = "default_use_image_array_cache")]
    use_image_array_cache: bool,
}

fn default_species_count_window_ms() -> u64 {
    5000
}

fn default_use_image_array_cache() -> bool {
    true
}

/// Custom deserializer for normalization_strategy that accepts:
/// - String: "none", "l1", "softmax"
/// - Boolean: false="none", true="l1" (backward compatibility with normalize_probabilities)
fn deserialize_normalization_strategy<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrBool {
        String(String),
        Bool(bool),
    }

    match Option::<StringOrBool>::deserialize(deserializer)? {
        Some(StringOrBool::String(s)) => Ok(Some(s)),
        Some(StringOrBool::Bool(b)) => Ok(Some(if b {
            "l1".to_string()
        } else {
            "none".to_string()
        })),
        None => Ok(None),
    }
}

fn default_species_count_max_items() -> usize {
    1000
}

#[derive(Debug, Clone)]
struct ClassInfo {
    yolo_id: usize,
    coco_id: usize,
    name: String,
}

// These globals will be initialized by initialize_animal_globals()
// We use Arc<DashMap> which is already imported in the module
static YOLO_CLASS_INFO: std::sync::LazyLock<dashmap::DashMap<usize, ClassInfo>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

static IMAGE_GROUND_TRUTH: std::sync::LazyLock<
    dashmap::DashMap<usize, std::collections::HashMap<String, usize>>,
> = std::sync::LazyLock::new(dashmap::DashMap::new);

/// Load and parse animal metadata from JSON file
fn load_animal_metadata(metadata_path: &str) -> anyhow::Result<AnimalImageMetadata> {
    let metadata_json = std::fs::read_to_string(metadata_path)
        .with_context(|| format!("Failed to read animal metadata file: {}", metadata_path))?;
    let metadata: AnimalImageMetadata =
        serde_json::from_str(&metadata_json).with_context(|| {
            format!(
                "Failed to parse animal metadata JSON from: {}",
                metadata_path
            )
        })?;
    Ok(metadata)
}

/// Initialize global class info and ground truth from metadata
fn initialize_animal_globals(metadata: &AnimalImageMetadata) -> anyhow::Result<()> {
    // Build YOLO_CLASS_INFO from yolo_name_to_id_map and yolo_id_to_coco_id_map
    for (yolo_name, yolo_id) in &metadata.yolo_name_to_id_map {
        if let Some(coco_id) = metadata.yolo_id_to_coco_id_map.get(&yolo_id.to_string()) {
            let class_info = ClassInfo {
                yolo_id: *yolo_id,
                coco_id: *coco_id,
                name: yolo_name.clone(),
            };
            YOLO_CLASS_INFO.insert(*yolo_id, class_info);
        }
    }

    info!(
        "Initialized YOLO class info with {} entries",
        YOLO_CLASS_INFO.len()
    );

    // Build IMAGE_GROUND_TRUTH from image_data
    for img_info in &metadata.image_data {
        IMAGE_GROUND_TRUTH.insert(img_info.image_id, img_info.animal_class_counts.clone());
    }

    info!(
        "Initialized ground truth for {} images",
        IMAGE_GROUND_TRUTH.len()
    );

    Ok(())
}

fn main() {
    let handle = match std::thread::Builder::new()
        .name("main".into())
        // set new main thread stack size to 32 MB
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            if let Err(e) = async_main() {
                error!("Async Main error: {:?}", e);
                // time to write before exiting
                std::thread::sleep(Duration::from_millis(1500));
            }
        }) {
        Err(e) => {
            error!("Failed to spawn main thread: {:?}", e);
            return;
        }
        Ok(handle) => handle,
    };
    if let Err(e) = handle.join() {
        error!("Main thread panicked: {:?}", e);
    }
}

// use watershed_shared::scheduler::basic_probability_forecast::History to predict the future based on the past
//  and then schedule accordingly using watershed_shared::scheduler::aquifer_scheduler

// these are heuristics just to get an idea of how many items we should project using their history
// each algorithm can only handle so much future window before its complexity blows out of control
static OPTIMAL_MAX_COUNT: AtomicUsize = AtomicUsize::new(5);
// greedy can handle much more but this is usually good enough
static GREEDY_MAX_COUNT: AtomicUsize = AtomicUsize::new(10);

fn async_main() -> anyhow::Result<()> {
    start_python_with_modules(&["numpy"]);
    // print env vars
    debug!("Printing environment variables:");
    for (key, value) in std::env::vars() {
        debug!("{}: {}", key, value);
    }
    debug!("End of env vars");
    let mut args = args();
    let _this_file = args.next().context("no file name")?;
    let config_path = args.next().context("no config path provided")?;
    let config = std::fs::read_to_string(config_path)?;

    // use the log4rs file
    let log_path = args.next().context("no logger config path provided")?;
    log4rs::init_file(log_path, Default::default()).context("failed to initialize log4rs")?;

    let animal_config: AnimalExperimentConfig =
        serde_json::from_str(&config).context("unable to parse animal config")?;
    debug!(
        "proceeding with animal config:\n{}\n",
        serde_json::to_string_pretty(&animal_config)
            .with_context(|| "unable to JSON-pretty-print animal config: {animal_config:#?}")?
    );

    let AnimalExperimentConfig {
        run_order_seed: _run_order_seed,
        animal_metadata_path,
        preclassifier_path,
        animal_image_base_path,
        query_path,
        max_total_samples,
        history_window_size,
        greedy_lookahead_window_size,
        optimal_lookahead_window_size,
        lookahead_time_ms,
        deadline_window_ms,
        target_time_micros,
        input_delay_micros,
        overall_time_limit_ms,
        initial_startup_delay_ms,
        routing_strategy,
        log_folder,
        blocking_noops,
        reyhydrate_spouts,
        species_count_window_ms,
        species_count_max_items,
        species_count_condition,
        debug_use_ground_truth: _debug_use_ground_truth,
        max_det,
        yolo_conf_threshold,
        yolo_min_class_probability,
        yolo_max_classes_per_det,
        use_dense_ground_truth_logging,
        log_intermediate_yolo_data,

        use_agnostic_nms,
        filter_animals_only,
        normalization_strategy,
        yolo_input_normalization,
        feature_extraction_strategy,
        exclude_purged_items_from_history,
        reroute_zero_option,
        use_image_array_cache,
    } = animal_config;

    if let Some(reroute_opt) = reroute_zero_option {
        debug!("Setting REROUTE_ZERO_OPTION env var to: {}", reroute_opt);
        std::env::set_var("REROUTE_ZERO_OPTION", reroute_opt);
    }

    const DEFAULT_BATCH_SIZE: usize = 1;

    if let Some(ws) = greedy_lookahead_window_size {
        GREEDY_MAX_COUNT.store(ws, std::sync::atomic::Ordering::SeqCst);
    }
    if let Some(ws) = optimal_lookahead_window_size {
        OPTIMAL_MAX_COUNT.store(ws, std::sync::atomic::Ordering::SeqCst);
    }

    let _flush_handle = watershed_shared::global_logger::set_flush_frequency(u16::MAX as _);
    let log_folder = log_folder.unwrap_or("log_outputs".into());
    watershed_shared::global_logger::set_logging_directory(log_folder);

    let max_det = max_det.unwrap_or(300);
    crate::yolos_utils::set_yolo_max_det(max_det);
    debug!("YOLO max_det set to {max_det}");

    let yolo_conf_threshold =
        yolo_conf_threshold.unwrap_or(crate::yolos_utils::DEFAULT_YOLO_CONF_THRESHOLD);
    crate::yolos_utils::set_yolo_conf_threshold(yolo_conf_threshold);
    debug!("YOLO conf threshold set to {yolo_conf_threshold}");

    let use_agnostic_nms = use_agnostic_nms.unwrap_or(true);
    crate::yolos_utils::set_nms_agnostic(use_agnostic_nms);
    debug!("YOLO agnostic NMS set to {use_agnostic_nms}");

    let filter_animals_only = filter_animals_only.unwrap_or(true);
    crate::animal_utils::set_yolo_filter_animals_only(filter_animals_only);
    debug!("YOLO filter animals only set to {filter_animals_only}");

    let normalize_strategy = match normalization_strategy.as_deref() {
        Some(s) => crate::animal_utils::NormalizationStrategy::from_str(s),
        None => crate::animal_utils::NormalizationStrategy::None,
    };
    crate::animal_utils::set_yolo_prob_normalize(normalize_strategy);
    debug!(
        "YOLO normalization strategy set to {:?}",
        normalize_strategy
    );

    let yolo_input_normalization = yolo_input_normalization.unwrap_or(true);
    crate::animal_utils::set_yolo_input_normalization(yolo_input_normalization);
    debug!("YOLO input normalization set to {yolo_input_normalization}");

    let yolo_min_class_probability = yolo_min_class_probability.unwrap_or(0.01);
    animal_utils::set_yolo_min_class_probability(yolo_min_class_probability);
    debug!("YOLO min class probability set to {yolo_min_class_probability}");

    let yolo_max_classes_per_det = yolo_max_classes_per_det.unwrap_or(80);
    animal_utils::set_yolo_max_classes_per_det(yolo_max_classes_per_det);
    debug!("YOLO max classes per detection set to {yolo_max_classes_per_det}");

    animal_utils::set_use_dense_ground_truth_logging(use_dense_ground_truth_logging);
    debug!("Dense ground truth logging set to {use_dense_ground_truth_logging}");

    animal_utils::set_log_intermediate_yolo_data(log_intermediate_yolo_data);
    debug!("Intermediate YOLO data logging set to {log_intermediate_yolo_data}");

    let use_image_array_cache = std::env::var("AQUIFER_USE_IMAGE_ARRAY_CACHE")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(use_image_array_cache);
    debug!("Image ndarray cache enabled: {use_image_array_cache}");

    let deadline_window_ms = deadline_window_ms.unwrap_or(1_000);
    let lookahead_time_ms = lookahead_time_ms.unwrap_or(deadline_window_ms);
    // let target_time_micros =
    //     target_time_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let mut target_time_micros = target_time_micros.unwrap_or(Delay::Fixed(100_000));
    let max_target_time_ms = target_time_micros.max() / 1_000;

    // let input_delay_micros =
    //     input_delay_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let input_delay_micros = input_delay_micros.unwrap_or(Delay::Fixed(0));
    let _max_input_delay_ms = input_delay_micros.max() / 1_000;
    let max_total_samples = max_total_samples.unwrap_or(usize::MAX);
    // history window size is used by the router, not the main thread
    let history_window_size = history_window_size.unwrap_or(50);

    let initial_startup_delay: u64 = initial_startup_delay_ms.unwrap_or(10_000);
    debug!("g0: initialized config params");

    let query = std::fs::read_to_string(&query_path)?;
    let mut function_lookup = BTreeMap::<HabString, FunctionKinds>::new();
    let no_op_counts: Arc<DashMap<usize, usize>> = Arc::new(Default::default());
    let no_op_counter_operator_count: Arc<AtomicUsize> = Arc::new(Default::default());
    function_lookup.insert(
        "no_op_counter".into(),
        FunctionKinds::FlatMapUdf(Box::new({
            let no_op_counts = Arc::clone(&no_op_counts);
            let no_op_counter_operator_count = Arc::clone(&no_op_counter_operator_count);
            move || {
                let no_op_counts = Arc::clone(&no_op_counts);
                let my_id =
                    no_op_counter_operator_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Box::new(move |t| {
                    let mut val = no_op_counts.entry(my_id).or_default();
                    let val = val.value_mut();
                    *val += 1;
                    if *val % 1000 == 0 {
                        debug!("{val:?} items passed in no-op counter #{my_id}");
                    }
                    vec![t]
                })
            }
        })),
    );

    let (max_item_condition, stop_rx) = watch::channel(false);
    let all_items_produced_counter = Arc::new(AtomicUsize::new(0));
    let item_producer = Arc::clone(&all_items_produced_counter);

    // Load animal metadata and initialize global class mappings.
    // Keep this before runtime/thread startup so optional cache prefill can complete first.
    let animal_metadata =
        load_animal_metadata(&animal_metadata_path).context("Failed to load animal metadata")?;
    initialize_animal_globals(&animal_metadata).context("Failed to initialize animal globals")?;
    animal_utils::set_yolo_name_to_id_map(&animal_metadata.yolo_name_to_id_map);

    debug!("g2: loaded animal metadata and initialized globals");

    // Convert AnimalImageInfo to format compatible with image loading
    // This is a temporary adapter - the tuple format is still compatible
    let image_info_adapted: Vec<(bool, _)> = animal_metadata
        .image_data
        .iter()
        .map(|ainfo| {
            (
                true,
                ImageInfo {
                    person_name: ainfo.relative_path.clone(), // temp adapter
                    img_id: ainfo.image_id.to_string(),
                    img_path: ainfo.relative_path.clone(),
                },
            )
        })
        .collect();

    let prefilled_image_array_cache = if use_image_array_cache {
        info!(
            "Prefilling image ndarray cache before runtime/thread startup for {} metadata entries",
            image_info_adapted.len()
        );
        let prefill_start = Instant::now();
        let cache = prefill_image_array_cache(
            &animal_image_base_path,
            image_info_adapted.iter().map(|(_, image_info)| image_info),
        );
        info!(
            "Finished cache prefill with {} entries in {:.3}s",
            cache.len(),
            prefill_start.elapsed().as_secs_f64()
        );
        Some(cache)
    } else {
        None
    };

    // set stack size to 32 MB
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(32 * 1024 * 1024)
        .global_queue_interval(23) // default is 31
        .event_interval(47) // default is 61, choosing this smaller number so we check more frequently
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;
    debug!("g1: initialized tokio runtime");

    if let Some(rehydrate_spouts) = reyhydrate_spouts {
        for spout_info in rehydrate_spouts {
            let fname = spout_info.rehydrate_path;
            let file_content = std::fs::read_to_string(&fname)
                .with_context(|| format!("Failed to read rehydrate file at path: {}", &fname))?;
            let mut base64_rmpe_strings: Vec<HabString> = match spout_info.options {
                RehydrateFormatOptions::Base64JsonStringArray => {
                    serde_json::from_str(&file_content).with_context(|| {
                        format!(
                            "Failed to parse rehydrate file as JSON array of strings: {}",
                            &fname
                        )
                    })?
                }
                RehydrateFormatOptions::NestedLogBase64Json { tuple_field } => {
                    let mut log_record: std::collections::HashMap<
                        HabString,
                        watershed_shared::global_logger::LimitedHabValues,
                    > = serde_json::from_str(&file_content).with_context(|| {
                        format!(
                            "Failed to parse rehydrate file as JSON array of tuples: {}",
                            &fname
                        )
                    })?;
                    let record_value = log_record.remove(&*tuple_field).with_context(|| {
                        format!(
                            "could not find field {tuple_field:?} in log record of file {fname:?}"
                        )
                    })?;
                    use anyhow::bail;
                    match record_value {
                        watershed_shared::global_logger::LimitedHabValues::Integer(_items) => bail!("unsupported type Integer found for tuple rehydration in field {tuple_field:?} in log record of file {fname:?}"),
                        watershed_shared::global_logger::LimitedHabValues::UnsignedInteger(_items) => bail!("unsupported type UnsignedInteger found for tuple rehydration in field {tuple_field:?} in log record of file {fname:?}"),
                        watershed_shared::global_logger::LimitedHabValues::UnsignedLongLong(_items) => bail!("unsupported type UnsignedLongLong found for tuple rehydration in field {tuple_field:?} in log record of file {fname:?}"),
                        watershed_shared::global_logger::LimitedHabValues::Float(_items) => bail!("unsupported type Float found for tuple rehydration in field {tuple_field:?} in log record of file {fname:?}"),
                        watershed_shared::global_logger::LimitedHabValues::String(string_forms) => string_forms,
                    }
                }
            };
            base64_rmpe_strings.retain(|s| !s.is_empty());
            let takeable_base64_rmpe_strings = Mutex::new(base64_rmpe_strings);
            let handle = rt.handle().clone();
            function_lookup.insert(
                spout_info.operator_name.into(),
                FunctionKinds::SourceUdf(match spout_info.use_face_config_timestamps {
                    RehydrateTimingOptions::SimulateOriginalTiming => Box::new(move || {
                        let base_strings = std::mem::take(&mut *takeable_base64_rmpe_strings.lock().unwrap_or_else(|e| {
                            error!("mutex poisoned");
                            panic!("mutex poisoned: {:?}", e);
                        }));
                        watershed_shared::ws_types::BetterTuple::stream_from_strings_lazy_timestamp(
                            base_strings,
                            handle.clone(),
                        )
                    }) as _,
                    RehydrateTimingOptions::FaceConfig { with_adjustment } => {
                        let mut delay = target_time_micros;
                        let initial_startup_delay_ms = initial_startup_delay;
                        let initial_startup_delay_us = initial_startup_delay_ms * 1_000 ;
                        let mut current_delay_us = delay.starting_delay();
                        let mut total_delay_us = initial_startup_delay_us;
                        let item_producer = Arc::clone(&item_producer);
                        let timings_iter = std::iter::from_fn(move || {
                            let increment_us = current_delay_us;
                            total_delay_us += increment_us;
                            let current_total_ns = total_delay_us * 1_000;
                            current_delay_us = delay.next_delay();
                            let _old = item_producer.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            Some(watershed_shared::ws_types::RehydratedTupleIteratorInfo{ relative_emit_time_ns: current_total_ns as _, creation_time_adjustment_ns: with_adjustment })
                        });
                        let takeable_timing_adjustments = Mutex::new(Some(timings_iter));
                        Box::new(move || {
                            let mut timings_iter = takeable_timing_adjustments.lock().expect("mutex poisoned");
                            let Some(timings_iter) = timings_iter.take() else {
                                error!("timings_iter was already taken");
                                panic!("timings_iter was already taken");
                            };
                            let base_strings = std::mem::take(&mut *takeable_base64_rmpe_strings.lock().expect("mutex poisoned"));
                            watershed_shared::ws_types::BetterTuple::stream_from_strings_manual_timestamp_adjustment(
                                base_strings,
                                timings_iter,
                                handle.clone(),
                                watershed_shared::ws_types::SourceStreamBehavior::Finite { num_items: 25, cycles: <usize as std::cmp::Ord>::clamp(max_total_samples/25, 1, 1000) },
                            )
                        }) as _
                    },
                }),
            );
        }
    }

    if let Some(blocking_noops) = blocking_noops {
        for blocking_noop in blocking_noops.into_iter() {
            let delay_micros = blocking_noop.delay_micros;
            function_lookup.insert(
                blocking_noop.operator_name.into(),
                FunctionKinds::FlatMapUdf(Box::new(move || {
                    Box::new(move |t| {
                        std::thread::sleep(Duration::from_micros(delay_micros));
                        vec![t]
                    })
                })),
            );
        }
    }

    let batch_size = 'batch_size: {
        let batch_size = match std::env::var("BATCH_SIZE") {
            Ok(s) => s,
            Err(e) => {
                if e == std::env::VarError::NotPresent {
                    debug!("BATCH_SIZE env var not set, using default of {DEFAULT_BATCH_SIZE}");
                    break 'batch_size DEFAULT_BATCH_SIZE;
                } else {
                    error!("Failed to read BATCH_SIZE env var: {e}, using default of {DEFAULT_BATCH_SIZE}");
                    break 'batch_size DEFAULT_BATCH_SIZE;
                }
            }
        };
        let Ok(batch_size) = batch_size.parse::<usize>() else {
            error!("Failed to parse BATCH_SIZE env var as usize: {batch_size}, using default of {DEFAULT_BATCH_SIZE}");
            break 'batch_size DEFAULT_BATCH_SIZE;
        };
        if batch_size == 0 {
            error!("BATCH_SIZE cannot be zero, using default of {DEFAULT_BATCH_SIZE}");
            break 'batch_size DEFAULT_BATCH_SIZE;
        } else {
            debug!("Using environment-defined BATCH_SIZE of {batch_size}");
            batch_size
        }
    };

    let preclassifier_path = preclassifier_path.context("preclassifier_path is required")?;
    let model_info = std::fs::read_to_string(&preclassifier_path)?;
    let preclassifier =
        watershed_shared::preclassifier_lang::load_file_format(model_info.as_bytes())?;

    // input tuples
    // let mut img_tuple_iter = paths_to_ndarrays(imdb_image_path, testing_remainder.into_iter())
    //     .enumerate()
    //     .take(max_total_samples);

    // if we are ever more than 1k behind, then there is a serious issue anyway
    const EFFECTIVELY_INFINITE: usize = 1_000usize;
    let (img_send, img_recv) = tokio::sync::mpsc::channel::<Vec<Tuple>>(EFFECTIVELY_INFINITE);
    let completion_sender = Clone::clone(&max_item_condition);

    let img_future_halt_receiver = stop_rx.clone();
    let mut timer_ticker_halt_receiver = stop_rx.clone();

    let (timer_tick_sender, mut timer_tick_receiver) =
        tokio::sync::mpsc::channel::<()>(EFFECTIVELY_INFINITE);

    const SKIP_NEW_DATA_KEY: &str = "SKIP_NEW_DATA";
    let should_skip_new_data_thread = std::env::var(SKIP_NEW_DATA_KEY).is_ok();
    if !should_skip_new_data_thread {
        // is env var set
        const LOOP_ITEMS_KEY: &str = "LOOP_ITEMS";
        let should_loop_items = std::env::var(LOOP_ITEMS_KEY).is_ok();
        // TODO: add option for looping limited data if necessary
        let paths_to_ndarrays_settings = if should_loop_items {
            PathsToNdarrays2Settings::Loop {
                first_n: usize::MAX,
                num_times: usize::MAX,
                total_items: usize::MAX,
            }
        } else {
            PathsToNdarrays2Settings::Normal
        };
        let metadata_for_thread = animal_metadata.clone();
        let prefilled_image_array_cache = prefilled_image_array_cache;
        std::thread::spawn(move || {
            let (img_tuple_iter, mut converter) = paths_to_ndarrays_v2(
                animal_image_base_path,
                image_info_adapted.into_iter(),
                paths_to_ndarrays_settings,
                &metadata_for_thread,
                use_image_array_cache,
                prefilled_image_array_cache,
            );
            let mut img_tuple_iter = img_tuple_iter.enumerate().take(max_total_samples);

            let mut current_batch = get_tuple_vec();
            // allow for slower startup so that tuples will not be too old
            std::thread::sleep(Duration::from_millis(initial_startup_delay + 5_000));
            let mut timeout_amount = Duration::from_micros(target_time_micros.starting_delay());
            debug!(
                "background tuple creation thread starting with initial timeout of {:?}",
                timeout_amount
            );
            let mut deadline = Instant::now() + timeout_amount;
            debug!(
                "background tuple creation thread initial deadline at {:?}",
                deadline
            );
            let mut next_item;
            'outer: loop {
                // if we are waiting for more than a second, we break it up into smaller sleeps
                if timeout_amount > Duration::from_millis(100) {
                    let mut remaining_time = timeout_amount;
                    while remaining_time > Duration::from_millis(100) {
                        std::thread::sleep(Duration::from_millis(100));
                        if let Err(e) = timer_tick_sender.try_send(()) {
                            error!("failed to send timer tick: {e:?}");
                            break 'outer;
                        }
                        // remaining_time -= Duration::from_millis(100);
                        remaining_time = (deadline - Instant::now()).max(Duration::from_millis(0));
                        if let Ok(true) | Err(_) = img_future_halt_receiver.has_changed() {
                            break 'outer;
                        }
                    }
                    std::thread::sleep(remaining_time);
                } else {
                    // otherwise just sleep the whole amount since we won't need to wake up too soon
                    debug!(
                        "background tuple creation thread sleeping for timeout of {:?}",
                        timeout_amount
                    );
                    std::thread::sleep(timeout_amount);
                }
                if let Err(e) = timer_tick_sender.try_send(()) {
                    error!("failed to send timer tick: {e:?}");
                    break 'outer;
                }
                // the current deadline has been reached,
                timeout_amount = Duration::from_micros(target_time_micros.next_delay());
                // Match the scheduling behavior used in other query binaries:
                // each cycle's next deadline is relative to "now", not cumulative.
                deadline = Instant::now() + timeout_amount;
                debug!(
                    "background tuple creation thread woke up, setting new timeout of {:?} and new deadline at {:?}",
                    timeout_amount,deadline
                );

                // and then we send the current batch (if any)
                // 'get_batch_items: loop {
                'get_batch_items: {
                    if let Ok(true) | Err(_) = img_future_halt_receiver.has_changed() {
                        debug!("background tuple creation thread received halt signal");
                        break 'outer;
                    }
                    next_item = img_tuple_iter.next();
                    let Some((valid_img_index, read_values)) = next_item else {
                        // no more items, send the last batch
                        if !current_batch.is_empty() {
                            debug!("background tuple creation thread got to the end of the iterator. sending final batch of size {}", current_batch.len());
                            if let Err(e) = img_send.try_send(current_batch) {
                                error!("failed to send last batch: {e:?}");
                                break 'outer;
                            }
                        } else {
                            debug!("background tuple creation thread got to the end of the iterator. no final batch to send");
                        }
                        break 'outer;
                    };
                    if valid_img_index > 0 && valid_img_index % (batch_size * 10) == 0 {
                        info!("read {valid_img_index} images so far");
                    }
                    let mut tuple = converter(read_values);
                    tuple.reset_time_created();
                    // Increment global counter for each source tuple emitted
                    item_producer.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    current_batch.push(tuple);
                    debug!(
                        "background tuple creation thread added item to current batch, size now {}",
                        current_batch.len()
                    );
                    if current_batch.len() >= batch_size {
                        // send the batch
                        debug!(
                            "background tuple creation thread sending batch of size {}",
                            current_batch.len()
                        );
                        if let Err(e) = img_send.try_send(current_batch) {
                            error!("failed to send batch: {e:?}");
                            break 'outer;
                        }
                        current_batch = get_tuple_vec();
                        break 'get_batch_items;
                    }
                }
            }
            // sleep a little (at least 5 seconds, at most 60 seconds), signaling the end in the middle
            let max_amt = target_time_micros.max();
            let pre_completion_drain =
                Duration::from_micros((max_amt / 2).min(30_000_00_000).max(2_500_000));
            info!(
                "tuple source finished iterator; entering pre-completion drain sleep of {:?} (emitted_so_far={})",
                pre_completion_drain,
                item_producer.load(std::sync::atomic::Ordering::SeqCst)
            );
            std::thread::sleep(pre_completion_drain);
            debug!("background tuple creation thread sending completion sender");
            let completion_signal_sent_at = Instant::now();
            if let Err(e) = completion_sender.send(true) {
                error!("tuple creation thread failed to send completion signal: {e:?}");
            } else {
                info!(
                    "tuple source sent completion signal (emitted_total={})",
                    item_producer.load(std::sync::atomic::Ordering::SeqCst)
                );
            }
            let post_completion_drain = Duration::from_millis(deadline_window_ms).max(
                Duration::from_micros((max_amt / 2).min(30_000_00_000).max(2_500_000)),
            );
            info!(
                "tuple source entering post-completion drain sleep of {:?}",
                post_completion_drain
            );
            std::thread::sleep(post_completion_drain);
            // signal end
            info!(
                "tuple source dropping img_send after {:?} since completion signal (emitted_total={})",
                completion_signal_sent_at.elapsed(),
                item_producer.load(std::sync::atomic::Ordering::SeqCst)
            );
            drop(img_send);
        });
    }

    rt.spawn(async move {
        // while we haven't received the stop signal, we keep waiting for timer ticks
        loop {
            tokio::select! {
                _ = timer_ticker_halt_receiver.changed() => {
                    // Received stop signal, break the loop
                    break;
                }
                tick = timer_tick_receiver.recv() => {
                    if tick.is_none() {
                        // Channel closed, we are done
                        break;
                    }
                    // just receive the tick and do nothing, we just want to keep the channel moving
                    // this lets us create an artificial timer tick that we can use to trigger other events in the system
                    // since tokio will only check other tasks as long as the rest of the system is moving
                }
            }
        }
    });

    debug!("g3: background timer started");

    // let create_img_future = async move {
    //     let mut current_batch = get_tuple_vec();

    //     // startup delay
    //     tokio::time::sleep(Duration::from_millis(initial_startup_delay)).await;

    //     let mut timeout_amount = Duration::from_micros(target_time_micros.starting_delay());
    //     let mut next_item;
    //     loop {
    //         next_item = img_tuple_iter.next();
    //         if let Some((valid_img_index, tuple)) = next_item {
    //             if valid_img_index > 0 && valid_img_index % (batch_size * 10) == 0 {
    //                 info!("read {valid_img_index} images so far");
    //             }
    //             current_batch.push(tuple);
    //             if current_batch.len() >= batch_size {
    //                 // send the batch
    //                 if let Err(e) = img_send.send(current_batch).await {
    //                     error!("failed to send batch: {e:?}");
    //                     break;
    //                 }
    //                 current_batch = get_tuple_vec();
    //             }
    //         } else {
    //             // no more items, send the last batch
    //             if !current_batch.is_empty() {
    //                 if let Err(e) = img_send.send(current_batch).await {
    //                     error!("failed to send last batch: {e:?}");
    //                     break;
    //                 }
    //             }
    //             break;
    //         }
    //         let timeout_future = tokio::time::sleep(timeout_amount);
    //         let halt_future = img_future_halt_receiver.changed();
    //         // wait for either the timeout or the halt signal
    //         tokio::select! {
    //             _ = timeout_future => {
    //                 // continue
    //             }
    //             _ = halt_future => {
    //                 // halt signal received, break the loop
    //                 break;
    //             }
    //         }
    //         timeout_amount = Duration::from_micros(target_time_micros.next_delay());
    //     }
    //     // sleep a little (at least 5 seconds, at most 60 seconds), signaling the end in the middle
    //     let max_amt = target_time_micros.max();
    //     tokio::time::sleep(Duration::from_micros(
    //         (max_amt / 2).min(30_000_00_000).max(2_500_000),
    //     ))
    //     .await;
    //     if let Err(e) = completion_sender.send(true) {
    //         error!("failed to send completion signal: {e:?}");
    //     }
    //     tokio::time::sleep(Duration::from_micros(
    //         (max_amt / 2).min(30_000_00_000).max(2_500_000),
    //     ))
    //     .await;
    //     // signal end
    //     drop(img_send);
    // };
    let img_tuple_stream = tokio_stream::wrappers::ReceiverStream::new(img_recv);

    let img_tuple_stream = Mutex::new(Some(img_tuple_stream));
    let img_tuple_stream = FunctionKinds::SourceUdf(Box::new(move || {
        let Ok(mut img_tuple_stream) = img_tuple_stream.lock() else {
            error!("img_tuple_stream lock poisoned");
            panic!("img_tuple_stream lock poisoned");
        };
        let Some(img_tuple_stream) = img_tuple_stream.take() else {
            error!("img_tuple_stream was already taken");
            panic!("img_tuple_stream was already taken");
        };
        Box::new(move || Box::new(img_tuple_stream))
    }));
    function_lookup.insert("image_source".into(), img_tuple_stream);

    let (route_feedback_sender, route_feedback_receiver) = crossbeam::channel::unbounded();
    let routing_udf: FunctionKinds = FunctionKinds::RoutingUdf(Box::new(move || {
        match routing_strategy {
            Some(RoutingOptions::Named(option @ (
                | NamedRoutingOptions::AlwaysSmall
                | NamedRoutingOptions::AlwaysBig
            ))) => {
                let error_msg = format!(
                    "Routing option {:?} is not implemented because that model proved to not have a sufficiently powerful tradeoff curve. Please use a different routing strategy.",
                    option
                );
                error!("{}", error_msg);
                unimplemented!("{}", error_msg);
            }
            Some(RoutingOptions::Named(option @ (NamedRoutingOptions::AlwaysTiny
                | NamedRoutingOptions::AlwaysHuge
            ))) => Box::new(routing_fn_static(
                RoutingOptions::Named(option),
                // window_size,
                // deadline_window_ms,
                // route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::Fixed(idx)) => Box::new(routing_fn_static(
                RoutingOptions::Fixed(idx),
            )),
            Some(RoutingOptions::Named(NamedRoutingOptions::Random)) => unimplemented!(
                "Random routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::Named(NamedRoutingOptions::Eddies)) => Box::new(routing_fn_eddies(
                // window_size,
                // deadline_window_ms,
                // route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::Named(NamedRoutingOptions::AquiferGreedy)) => Box::new(aquifer_routing_fn(
                preclassifier.clone(),
                history_window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Greedy,
                exclude_purged_items_from_history,
            )),
            Some(RoutingOptions::Named(NamedRoutingOptions::AquiferOptimal)) => Box::new(aquifer_routing_fn(
                preclassifier.clone(),
                history_window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Optimal,
                exclude_purged_items_from_history,
            )),
            Some(RoutingOptions::Named(NamedRoutingOptions::AlwaysDrop)) => unimplemented!(
                "AlwaysDrop routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::Named(NamedRoutingOptions::PredictorBinary)) => unimplemented!(
                "PredictorBinary routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::Named(NamedRoutingOptions::PredictorProbabilistic)) => unimplemented!(
                "PredictorProbabilistic routing is not implemented yet. Please use a different routing strategy."
            ),
            None => Box::new(routing_fn_static(
                RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny),
                // route_feedback_receiver.clone(),
            )),
        }
    }));

    function_lookup.insert("routing_fn".into(), routing_udf);
    debug!("g5: routing function registered");

    // Register extract_image_info to call preprocess_image for YOLO preprocessing
    let feature_extraction_strategy = feature_extraction_strategy.unwrap_or_default();
    let extract_image_info_udf = FunctionKinds::FlatMapUdf(Box::new(move || {
        // Initialize state per thread
        let strategy = feature_extraction_strategy;
        let fft_len = strategy.fft_len();
        let buffers = Mutex::new(FftBuffers::new(fft_len));
        let timing = Mutex::new(TimingStats::default());
        let fft_plan = crate::streaming_features::create_fft_plan(fft_len);

        Box::new(move |t| yolos_utils::preprocess_image(t, &buffers, &timing, &fft_plan, strategy))
    }));
    function_lookup.insert("extract_image_info".into(), extract_image_info_udf);
    debug!("g5.5: extract_image_info (YOLO preprocessing) UDF registered");

    // Register animal query UDFs
    let split_tuple_yolos_udf = FunctionKinds::FlatMapUdf(Box::new(|| {
        Box::new(|t| animal_utils::split_tuple_yolos(t))
    }));
    function_lookup.insert("split_tuple_yolos".into(), split_tuple_yolos_udf);
    debug!("g6: split_tuple_yolos UDF registered");

    // Register aggregation UDF with config parameters
    let species_count_window_ms_copy = species_count_window_ms;
    let species_count_max_items_copy = species_count_max_items;
    let species_count_condition_copy = species_count_condition;
    let aggregate_count_species_udf = FunctionKinds::AggregationUdf(Box::new(move || {
        let window_ms = species_count_window_ms_copy;
        let max_items = species_count_max_items_copy;
        let count_condition = species_count_condition_copy;
        Box::new(move |window| {
            let limit_info = animal_utils::LimitInfo::new(max_items, Some(window_ms));
            let count_cond = match count_condition {
                CountCondition::UseProbability => animal_utils::CountCondition::UseProbability,
                CountCondition::SimpleCount => animal_utils::CountCondition::SimpleCount,
            };
            animal_utils::aggregate_count_species(window, count_cond, limit_info)
        })
    }));
    function_lookup.insert(
        "aggregate_count_species".into(),
        aggregate_count_species_udf,
    );
    debug!(
        "g7: aggregate_count_species UDF registered with config: window_ms={}, max_items={}",
        species_count_window_ms, species_count_max_items
    );

    // Register merge callback functions for tracking tuple flow through pipelines
    // Register merge callback functions for tracking tuple flow through pipelines.
    // Supports up to 16 models (v1..v16)
    let merge_feedback_seen: Arc<DashMap<(usize, usize), ()>> = Arc::new(DashMap::new());
    for my_pipeline_id in 0..16 {
        let callback_name = format!("merge_callback_animal_model_v{}", my_pipeline_id + 1);
        let callback_name_clone = callback_name.clone();
        let route_feedback_sender_clone = route_feedback_sender.clone();
        let merge_feedback_seen_clone = Arc::clone(&merge_feedback_seen);
        let merge_callback_fn: FunctionKinds = FunctionKinds::MergeCallbackUdf(Box::new(
            move || {
                let route_feedback_sender_clone = route_feedback_sender_clone.clone();
                let merge_feedback_seen_clone = Arc::clone(&merge_feedback_seen_clone);
                let my_pipeline_id = my_pipeline_id + 1;
                let callback_name = callback_name_clone.clone();
                Box::new(move |tuple| {
                    let feedback_tuple_id = tuple
                        .get("original_tuple_id")
                        .and_then(|v| v.as_integer())
                        .map(|v| v as usize)
                        .unwrap_or_else(|| tuple.id());

                    trace!(
                        "merge callback {callback_name:?} received tuple {} (feedback key {})",
                        tuple.id(),
                        feedback_tuple_id
                    );

                    // Split path can emit multiple child tuples for one routed source tuple.
                    // Keep one feedback event per (routed tuple id, pipeline id).
                    let dedupe_key = (feedback_tuple_id, my_pipeline_id);
                    if merge_feedback_seen_clone.insert(dedupe_key, ()).is_some() {
                        trace!(
                            "merge callback {callback_name:?} duplicate feedback ignored for key {:?}",
                            dedupe_key
                        );
                        return;
                    }

                    if let Err(e) = route_feedback_sender_clone.send(vec![(
                        feedback_tuple_id as _,
                        my_pipeline_id,
                        Instant::now(),
                    )]) {
                        error!("failed to send feedback to routing: {e}");
                    }
                })
            },
        ));
        function_lookup.insert(callback_name.into(), merge_callback_fn);
    }
    debug!("g7.5: merge callback functions registered");

    debug!("g8: creating operators from query info");
    let background_rt = rt.handle().clone();

    let RuntimeState {
        operators,
        runtime: _runtime,
        output_channels: _output_channels,
        stop_trigger,
    } = match watershed_shared::async_query_builder::json_descriptor_to_operators_with_runtime(
        &query,
        &function_lookup,
        Some(rt),
        // max age is in nanos
        // Some((deadline_window_ms * 1_000_000) as u128),
        None,
    ) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to build operators from query, possibly unable to parse: {e:?}");
            return Err(e)
                .context("failed to build operators from query, possibly unable to parse");
        }
    };
    debug!("g9: operators created from query info");
    let mut query_descriptor =
        serde_json::from_str::<QueryDescriptor>(&query).context("unable to parse query")?;
    query_descriptor.operators.sort_by_key(|op| op.id);
    let topology = query_builder::get_topology_simple(&query_descriptor.operators, 4);

    let mut early_poll_ids = std::collections::BTreeSet::new();
    for oid in 0..operators.len() {
        // let op_len = operators.len();
        let Some(PhysicalOperator::UserDefinedSource(source)) = operators.get(oid) else {
            continue;
        };
        // let parent_id = source
        //     .parent
        //     .with_context(|| format!("no parent found for build side of operator with id {oid}"))?;
        let Some(parent_id) = source.parent else {
            warn!("no parent found for build side of operator with id {oid}");
            continue;
        };
        let Some(PhysicalOperator::Join(watershed_shared::Join { right, .. })) =
            operators.get(parent_id)
        else {
            continue;
        };
        if *right == oid {
            early_poll_ids.insert(oid);
        }
    }
    debug!("g10: early poll ids determined");
    info!("early poll ids {early_poll_ids:?}");

    let log_udf_items_received = Arc::new(AtomicUsize::new(0));
    let log_udf_items_received_callback = Arc::clone(&log_udf_items_received);
    let log_udf_items_received_ending = Arc::clone(&log_udf_items_received);
    let all_items_read_logger = Arc::clone(&all_items_produced_counter);
    // continue as long as we haven't received the max amt and as long as the sequences aren't done

    let condition_bg_cutoff = Clone::clone(&max_item_condition);

    let log_udf: UdfBolt = UdfBolt {
        id: operators.len(),
        child: {
            let Some(op) = operators.last() else {
                error!("no operators found for log udf");
                panic!("no operators found for log udf");
            };
            op.get_id()
        },
        parent: None,
        process: Arc::new(move |tuple| {
            let tuple_id = tuple.id();
            trace!(
                "log udf tuple {tuple_id} has fields: {:?}",
                tuple.keys().collect::<Vec<_>>()
            );

            // Count all tuples - termination based on source counter (all_items_produced_counter)
            // which is incremented once per source tuple, regardless of fan-out
            log_udf_items_received_callback.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            // we don't need to do anything with the tuple, just count it
            drop(tuple);
            // vec![tuple]
            vec![]
        }),
    };

    debug!("g11: log udf created. starting background cutoff task");
    // let expected_items = (TOTAL_OBJS - SKIP_AMOUNT).min(max_total_samples.unwrap_or(usize::MAX));
    let expected_items = max_total_samples;
    // let _create_img_future = background_rt.spawn(create_img_future);
    let _background_cutoff = background_rt.spawn(async move {
        let delay_amount = initial_startup_delay + 5_000;
        tokio::time::sleep(Duration::from_millis(delay_amount)).await;
        let condition_start_time = Instant::now();
        loop{
            let completed_items = all_items_read_logger.load(atomic::Ordering::SeqCst);
            if completed_items >= expected_items {
                tokio::time::sleep(Duration::from_millis(max_target_time_ms)).await;
                let time_elapsed = condition_start_time.elapsed().as_millis();
                let expected_time = 1 + all_items_read_logger.load(atomic::Ordering::SeqCst) as u128 * max_target_time_ms as u128;
                // 20% grace period
                let grace_time = ((expected_time * 12)/10) + initial_startup_delay.max(1000) as u128;
                debug!("time elapsed: {:?}, expected time: {:?}, grace time: {:?}", time_elapsed, expected_time, grace_time);
                if time_elapsed < grace_time {
                    let diff = grace_time - time_elapsed;
                    info!("all sequences have been completed, but we are still within the grace period ({diff}ms remain). waiting for the grace period to end");
                    tokio::time::sleep(Duration::from_millis(diff as _)).await;
                }
                info!("sequence completion counter reached the expected count, notifying condition");
                let _ = condition_bg_cutoff.send(true);
                break;
            }
            tokio::time::sleep(Duration::from_millis(1_000)).await;
        }
        info!("background cutoff task finished");
    });
    let Some(_rt) = _runtime.as_ref() else {
        error!("no runtime found");
        // panic!("no runtime found");
        anyhow::bail!("no runtime found")
    };
    let metrics = _rt.metrics();
    debug!("rt workers: {:?}", metrics.num_workers());
    debug!("rt alive tasks {:?}", metrics.num_alive_tasks());

    let last_op = operators.len() - 1;

    info!("operators:");
    for op in operators.iter() {
        info!("#{}: {:?}", op.get_id(), op.get_op_type());
    }

    info!("topology:");
    for (thread_no, v) in topology.iter().enumerate() {
        info!("thread {:?}: {:?}", thread_no, v);
    }

    debug!("early poll ids:");
    for id in &early_poll_ids {
        debug!("{:?}", id);
    }

    debug!("g12: starting runner");
    let overall_time_limit_ms = overall_time_limit_ms.unwrap_or(1_000 * 60 * 30);
    watershed_shared::async_query_builder::runner_internal(
        operators,
        topology,
        early_poll_ids,
        last_op,
        Some(log_udf),
        Duration::from_millis(overall_time_limit_ms),
        _runtime.expect("runtime checked above"),
        move |task_index, runtime, v, a| {
            watershed_shared::async_query_builder::execute_for_while(
                task_index,
                runtime,
                stop_rx.clone(),
                stop_trigger.clone(),
                v,
                a,
            )
        },
    );
    debug!("g13: runner finished");
    let log_udf_items_received =
        log_udf_items_received_ending.load(std::sync::atomic::Ordering::SeqCst);
    info!("group by items received: {:?}", log_udf_items_received);

    info!("expected items: {:?}", expected_items);

    info!(
        "items read: {:?}",
        all_items_produced_counter.load(atomic::Ordering::SeqCst)
    );

    use std::io::Write;
    debug!("attempting first flush");
    if let Err(e) = std::io::stdout().flush() {
        error!("failed to flush stdout: {e}");
    }
    std::thread::sleep(Duration::from_millis(5000));

    debug!("attempting second flush");
    if let Err(e) = std::io::stdout().flush() {
        error!("failed to flush stdout: {e}");
    }
    std::thread::sleep(Duration::from_millis(1000));

    // TODO: allow the user to specify the file name and then this should flush any remaining data
    // binary encode the starting info
    // let starting_info = starting_info.lock().unwrap();
    // let starting_info_encoded = rmp_serde::to_vec(&*starting_info)?;
    // let mut starting_info_file = std::fs::File::create("starting_info.rmp")?;
    // starting_info_file.write_all(&starting_info_encoded)?;

    // let merge_info = merge_info.lock().unwrap();
    // let merge_info_encoded = rmp_serde::to_vec(&*merge_info)?;
    // let mut merge_info_file = std::fs::File::create("merge_info.rmp")?;
    // merge_info_file.write_all(&merge_info_encoded)?;

    info!("finished executing");
    info!("no op counter status: {no_op_counts:?}");
    Ok(())
}

fn paths_to_ndarrays(
    base_folder: String,
    image_paths: impl Send + Iterator<Item = ImageInfo>,
) -> impl Iterator<Item = Tuple> {
    let path = std::path::PathBuf::from(base_folder);
    // load all images first and then later make the tuples
    // this is to avoid the tuples having very early creation times that count against them later
    use rayon::prelude::*;
    let mut images: Vec<_> = image_paths
        .enumerate()
        .par_bridge()
        .flat_map(move |(img_idx, image_info)| {
            let mut path = path.clone();
            path.push(&image_info.img_path);
            let full_path: &std::path::Path = &path;

            let img = match image::open(full_path) {
                Ok(img) => img.to_rgb8(),
                Err(e) => {
                    error!("Failed to open image {:?} with error: {:?}", full_path, e);
                    // pop file
                    path.pop();
                    // pop person
                    path.pop();
                    return None;
                }
            };
            // pop file
            path.pop();
            // pop person
            path.pop();
            Some((img_idx, image_info, img))
        })
        .collect();
    // we have to correct the order because the parallel iteration did not preserve it
    images.sort_by_key(|(img_idx, _, _)| *img_idx);
    let mut last_emit_time = Instant::now();
    images
        .into_iter()
        .filter_map(move |(_img_idx, image_info, img)| {
            let (width, height) = img.dimensions();
            let mut tuple = get_tuple();
            let img_id_key = image_info.img_id.to_key();

            tuple.insert("img_id".into(), HabValue::String(image_info.img_id.into()));

            let buf = img.into_raw();
            debug!(
                "image buffer for {:?} with width {width} and height {height} has length {}",
                &image_info.img_path,
                buf.len()
            );
            // convert into python bytes and then into a numpy array
            // let py_arr: anyhow::Result<Py<PyAny>> = Python::with_gil(|py| {
            //     let b = pyo3::types::PyBytes::new(py, &buf);
            //     let np =
            //         pyo3::types::PyModule::import(py, "numpy").context("Failed to import numpy")?;
            //     let arr = np.call_method1("frombuffer", (b, "uint8"))?;
            //     let arr = arr
            //         .into_py_any(py)
            //         .context("Failed to convert to numpy array")?;
            //     let arr = arr
            //         .call_method1(py, "reshape", (height as usize, width as usize, 3))
            //         .context("Failed to reshape numpy array")?;
            //     let arr = arr.call_method0(py, "copy")?;
            //     Ok(arr)
            // });
            // let py_arr = match py_arr {
            //     Ok(p) => p,
            //     Err(e) => {
            //         error!(
            //             "Failed to convert image buffer {:?} to numpy array: {:?}",
            //             &image_info.img_path, e
            //         );
            //         return None;
            //     }
            // };
            // tuple.insert("python_array".into(), py_arr.into());

            tuple.insert("image".into(), HabValue::ByteBuffer(buf));

            tuple.insert(
                "img_path".into(),
                HabValue::String(image_info.img_path.into()),
            );
            tuple.insert(
                ORIGINAL_IMAGE_SHAPE_FIELD.into(),
                HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
            );
            tuple.insert("original_width".into(), HabValue::Integer(width as _));
            tuple.insert("original_height".into(), HabValue::Integer(height as _));
            tuple.insert("original_width_float".into(), HabValue::from(width as f64));
            tuple.insert(
                "original_height_float".into(),
                HabValue::from(height as f64),
            );
            tuple.insert(
                "original_area_float".into(),
                HabValue::from((height * width) as f64),
            );
            tuple.insert(
                "original_hw_ratio_float".into(),
                HabValue::from((height as f64) / (width as f64)),
            );
            let now = Instant::now();
            let since_last_emit = last_emit_time.elapsed().as_nanos() as f64 / 1_000_000.0;
            last_emit_time = now;
            debug!(
                "created tuple with id {} after {:.3} ms since the last tuple",
                tuple.id(),
                since_last_emit
            );
            'log_created_image_tuple: {
                let tuple_id = tuple.id();

                use watershed_shared::global_logger;
                let log_location = "create_tuple".to_raw_key();
                let aux_data = Some(std::collections::HashMap::from([(
                    "image_id".to_raw_key(),
                    LimitedHabValue::String(img_id_key),
                )]));
                if let Err(e) = global_logger::log_data(tuple_id, log_location, aux_data) {
                    for err in e {
                        error!("failed to log initial image tuple creation: {err}");
                    }
                    break 'log_created_image_tuple;
                }
            }

            Some(tuple)
        })
}

pub static NEXT_IMAGE_ID_INT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub static IMAGE_ID_INT_MAP: std::sync::LazyLock<dashmap::DashMap<HabString, usize>> =
    std::sync::LazyLock::new(dashmap::DashMap::new);

enum PathsToNdarrays2Settings {
    Normal,
    Loop {
        first_n: usize,
        num_times: usize,
        total_items: usize,
    },
}

struct BetterCycle<I: Iterator> {
    state: BetterCycleState<I>,
    max_iters: Option<usize>,
}
impl<I: Iterator> BetterCycle<I> {
    fn new(inner: I, max_iters: Option<usize>) -> Self {
        let size_hint = inner.size_hint();
        let expected_len = size_hint.1.or(Some(size_hint.0));
        const MAX_STARTING_LEN: usize = 10_000;
        let expected_len = expected_len.unwrap_or(0).min(MAX_STARTING_LEN);
        Self {
            state: BetterCycleState::Init {
                inner,
                cache: Vec::with_capacity(expected_len),
            },
            max_iters,
        }
    }
}
impl<I> Iterator for BetterCycle<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(&mut self.state, BetterCycleState::Done) {
            BetterCycleState::Init {
                mut inner,
                mut cache,
            } => {
                if let Some(item) = inner.next() {
                    cache.push(item.clone());
                    self.state = BetterCycleState::Init { inner, cache };
                    Some(item)
                } else {
                    if cache.is_empty() {
                        None
                    } else {
                        let item = cache[0].clone();
                        self.state = BetterCycleState::Cycling {
                            current_item: 1,
                            cache: cache,
                        };
                        Some(item)
                    }
                }
            }
            BetterCycleState::Cycling {
                mut current_item,
                cache,
            } => {
                if cache.is_empty() {
                    // already set to Done
                    // self.state = BetterCycleState::Done;
                    return None;
                }
                let current_iter = current_item / cache.len();
                let current_item_index = current_item % cache.len();
                if let Some(max_iters) = self.max_iters {
                    if current_iter >= max_iters {
                        self.state = BetterCycleState::Done;
                        return None;
                    }
                }
                let item = cache[current_item_index].clone();
                current_item += 1;
                self.state = BetterCycleState::Cycling {
                    current_item,
                    cache,
                };
                Some(item)
            }
            BetterCycleState::Done => None,
        }
    }
}
enum BetterCycleState<I>
where
    I: Iterator,
{
    Init {
        inner: I,
        cache: Vec<I::Item>,
    },
    Cycling {
        cache: Vec<I::Item>,
        current_item: usize,
    },
    Done,
}

fn paths_to_ndarrays_v2<'a>(
    base_folder: String,
    image_paths: impl 'a + Send + Iterator<Item = (bool, ImageInfo)>,
    options: PathsToNdarrays2Settings,
    metadata: &'a AnimalImageMetadata,
    use_image_array_cache: bool,
    prefilled_image_array_cache: Option<
        std::collections::HashMap<String, (usize, usize, HabValue)>,
    >,
) -> (
    impl 'a + Iterator<Item = (usize, bool, ImageInfo, (usize, usize, HabValue))>,
    impl FnMut((usize, bool, ImageInfo, (usize, usize, HabValue))) -> Tuple,
) {
    let path = std::path::PathBuf::from(base_folder);
    // load all images first and then later make the tuples
    // this is to avoid the tuples having very early creation times that count against them later
    let initial_iter = image_paths_to_iterator(
        image_paths,
        path,
        use_image_array_cache,
        prefilled_image_array_cache,
    );
    let images: Box<dyn Iterator<Item = (usize, bool, ImageInfo, (usize, usize, HabValue))>> =
        match options {
            PathsToNdarrays2Settings::Normal => Box::new(initial_iter),
            PathsToNdarrays2Settings::Loop {
                first_n,
                num_times,
                total_items,
            } => {
                let upper_limit = first_n.saturating_mul(num_times).min(total_items);
                // let rapid_load = initial_iter.par_iter()
                let initial_iter = BetterCycle::new(initial_iter.take(first_n), Some(num_times));
                Box::new(initial_iter.take(upper_limit))
            }
        };

    let mut last_emit_time = Instant::now();
    let metadata_clone = metadata.clone();
    let converter = move |(_img_idx, is_in_index, image_info, (width, height, arr3_buf)): (
        usize,
        bool,
        ImageInfo,
        // image::RgbImage,
        (usize, usize, HabValue),
    )| {
        let mut tuple = get_tuple();
        let original_tuple_id = tuple.id();
        let parsed_image_id = image_info.img_id.parse::<usize>().ok();
        let img_id_key = image_info.img_id.to_key();
        let img_id_int = parsed_image_id.unwrap_or_else(|| {
            let image_id = IMAGE_ID_INT_MAP
                .entry(img_id_key.clone().into())
                .or_insert_with(|| {
                    NEXT_IMAGE_ID_INT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                });
            *image_id.value()
        });

        tuple.insert(
            ORIGINAL_IMAGE_ID_FIELD.into(),
            HabValue::String(image_info.img_id.into()),
        );
        tuple.insert(
            ORIGINAL_IMAGE_ID_INT_FIELD.into(),
            HabValue::Integer(img_id_int as _),
        );
        tuple.insert(
            EXPECTED_MATCHES_FIELD.into(),
            HabValue::Integer(if is_in_index { 1 } else { 0 }),
        );
        tuple.insert(
            "original_tuple_id".into(),
            HabValue::Integer(original_tuple_id as _),
        );

        // Add ground truth data if available
        if let Some(image_data) = metadata_clone
            .image_data
            .iter()
            .find(|d| d.image_id == img_id_int as usize)
        {
            for (class_name, count) in &image_data.animal_class_counts {
                let field_name = format!("ground_truth_class_{}", class_name);
                tuple.insert(field_name.into(), HabValue::Integer(*count as i32));

                // Also insert expected_count_<yolo_class_id> so split_tuple_yolos can log comparisons
                let yolo_class_id = if let Some(class_id) =
                    metadata_clone.yolo_name_to_id_map.get(class_name)
                {
                    Some(*class_id)
                } else if let Some(coco_id) = metadata_clone.coco_name_to_id_map.get(class_name) {
                    metadata_clone
                        .coco_id_to_yolo_id_map
                        .get(&coco_id.to_string())
                        .cloned()
                } else {
                    None
                };

                if let Some(class_id) = yolo_class_id {
                    let expected_field = format!("expected_count_{}", class_id);
                    tuple.insert(expected_field.into(), HabValue::Integer(*count as i32));
                }
            }
        }

        // tuple.insert(ORIGINAL_IMAGE_FIELD.into(), HabValue::ByteBuffer(buf));
        tuple.insert(
            ORIGINAL_IMAGE_FIELD.into(),
            // HabValue::SharedArrayU8(watershed_shared::SharedU8Array(arr3)),
            arr3_buf,
        );

        tuple.insert(
            "img_path".into(),
            HabValue::String(image_info.img_path.into()),
        );
        tuple.insert(
            "shape".into(),
            HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
        );
        tuple.insert(
            "image_shape".into(),
            HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
        );
        tuple.insert("original_width".into(), HabValue::Integer(width as _));
        tuple.insert("original_height".into(), HabValue::Integer(height as _));
        tuple.insert("original_width_float".into(), HabValue::from(width as f64));
        tuple.insert(
            "original_height_float".into(),
            HabValue::from(height as f64),
        );
        tuple.insert(
            "original_area_float".into(),
            HabValue::from((height * width) as f64),
        );
        tuple.insert(
            "original_hw_ratio_float".into(),
            HabValue::from((height as f64) / (width as f64)),
        );
        let now = Instant::now();
        let since_last_emit = last_emit_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        last_emit_time = now;
        debug!(
            "created tuple with id {} after {:.3} ms since the last tuple",
            tuple.id(),
            since_last_emit
        );
        'log_created_image_tuple: {
            let tuple_id = tuple.id();

            use watershed_shared::global_logger;
            let log_location = "create_tuple".to_raw_key();
            let aux_data = Some(std::collections::HashMap::from([(
                "image_id".to_raw_key(),
                LimitedHabValue::String(img_id_key),
            )]));
            if let Err(e) = global_logger::log_data(tuple_id, log_location, aux_data) {
                for err in e {
                    error!("failed to log initial image tuple creation: {err}");
                }
                break 'log_created_image_tuple;
            }
        }

        tuple
    };
    (images, converter)
}

fn image_paths_to_iterator<'a>(
    image_paths: impl 'a + Send + Iterator<Item = (bool, ImageInfo)>,
    path: std::path::PathBuf,
    use_image_array_cache: bool,
    prefilled_image_array_cache: Option<
        std::collections::HashMap<String, (usize, usize, HabValue)>,
    >,
) -> impl 'a + Iterator<Item = (usize, bool, ImageInfo, (usize, usize, HabValue))> {
    let mut image_paths = image_paths.enumerate();
    let mut image_array_cache = if use_image_array_cache {
        prefilled_image_array_cache.unwrap_or_default()
    } else {
        std::collections::HashMap::<String, (usize, usize, HabValue)>::new()
    };

    if use_image_array_cache {
        debug!(
            "image ndarray cache initialized with {} prefilled entries",
            image_array_cache.len()
        );
    }

    Box::new(std::iter::from_fn(move || loop {
        let (img_idx, (is_in_index, image_info)) = image_paths.next()?;
        let image_data = if use_image_array_cache {
            get_image_array_cached(&path, &image_info, &mut image_array_cache)
        } else {
            get_image_array_direct(&path, &image_info)
        };

        if let Some(image_data) = image_data {
            return Some((img_idx, is_in_index, image_info, image_data));
        }
    }))
}

fn prefill_image_array_cache<'a>(
    base_folder: &str,
    image_infos: impl IntoIterator<Item = &'a ImageInfo>,
) -> std::collections::HashMap<String, (usize, usize, HabValue)> {
    let base_path = std::path::PathBuf::from(base_folder);
    let mut cache = std::collections::HashMap::<String, (usize, usize, HabValue)>::new();
    let mut skipped = 0usize;

    for image_info in image_infos {
        if cache.contains_key(&image_info.img_path) {
            continue;
        }
        match get_image_array_direct(&base_path, image_info) {
            Some(image_data) => {
                cache.insert(image_info.img_path.clone(), image_data);
            }
            None => {
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        warn!(
            "image ndarray cache prefill skipped {} images due to load/convert failures",
            skipped
        );
    }

    cache
}

fn get_image_array_cached(
    base_path: &std::path::Path,
    image_info: &ImageInfo,
    image_array_cache: &mut std::collections::HashMap<String, (usize, usize, HabValue)>,
) -> Option<(usize, usize, HabValue)> {
    if let Some((width, height, arr3_buf)) = image_array_cache.get(&image_info.img_path) {
        return Some((*width, *height, arr3_buf.clone()));
    }

    let loaded = get_image_array_direct(base_path, image_info)?;
    image_array_cache.insert(image_info.img_path.clone(), loaded.clone());
    Some(loaded)
}

fn get_image_array_direct(
    base_path: &std::path::Path,
    image_info: &ImageInfo,
) -> Option<(usize, usize, HabValue)> {
    let full_path = base_path.join(&image_info.img_path);
    let img = match image::open(&full_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            error!("Failed to open image {:?} with error: {:?}", full_path, e);
            return None;
        }
    };
    let (width, height) = img.dimensions();

    let buf = img.into_raw();
    debug!(
        "image buffer for {:?} with width {width} and height {height} has length {}",
        &image_info.img_path,
        buf.len()
    );

    let buf_len = buf.len();
    let Ok(arr3) = watershed_shared::ws_types::ArcArrayD::from_shape_vec(
        &[height as usize, width as usize, 3][..],
        buf,
    ) else {
        error!("failed to create ndarray from image buffer for image {:?} with width {width} and height {height} with buffer length {}",
            &image_info.img_path,
            buf_len
        );
        return None;
    };
    let arr3_buf = HabValue::SharedArrayU8(watershed_shared::SharedU8Array(arr3));
    Some((width as usize, height as usize, arr3_buf))
}

fn tuple_to_bucket(
    tuple: &Tuple,
    preclassifier: &watershed_shared::preclassifier_lang::RealBucketLookup,
) -> BinInfo<watershed_shared::preclassifier_lang::PreclassifierLangClass> {
    let Some(streaming_features) = tuple.get("streaming_features") else {
        // If features are missing, we can't route dynamically.
        // Log error and panic/return a default bucket?
        // Panicking is consistent with previous behavior for missing critical fields.
        let err = format!(
            "streaming_features not found in tuple with id {} and img_id={:?}",
            tuple.id(),
            tuple.get("img_id")
        );
        error!("{err}");
        panic!("{err}");
    };

    let HabValue::List(features_list) = streaming_features else {
        let err = format!(
            "streaming_features was not a List in tuple with id {} and img_id={:?}",
            tuple.id(),
            tuple.get("img_id")
        );
        error!("{err}");
        panic!("{err}");
    };

    // Convert HabValue::List<HabValue::Float> to Vec<f32>
    // Use a smallvec or similar if performance is critical, but Vec is fine for now.
    let features: Vec<f32> = features_list
        .iter()
        .map(|v| {
            v.as_float()
                .expect("streaming_features items must be floats")
                .into_inner() as f32
        })
        .collect();

    let bucket =
        watershed_shared::preclassifier_lang::map_inputs_to_bucket(&features, preclassifier);
    // trace!("mapped tuple {} to bucket {:?}", tuple.id(), bucket);
    info!("mapped tuple {} to bucket {:?}", tuple.id(), bucket);
    bucket
}

fn aquifer_routing_fn(
    preclassifier: watershed_shared::preclassifier_lang::RealBucketLookup,
    keep_n_history_items: usize,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
    deadline_ms: u64,
    lookahead_ms: u64,
    strategy: scheduler::Strategy,
    exclude_purged_items_from_history: bool,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    let budget_to_lookahead_ratio = lookahead_ms as f64 / deadline_ms as f64;
    let optimal_max_count = OPTIMAL_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let greedy_max_count = GREEDY_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let discrete_bins = preclassifier.buckets.to_vec();
    let mut history = watershed_shared::scheduler::basic_probability_forecast::History::new(
        keep_n_history_items,
        back_channel,
        discrete_bins,
    );
    history.exclude_purged_items_from_history = exclude_purged_items_from_history;
    let binning_fn = move |tuple: &Tuple| -> BinInfo<
        watershed_shared::preclassifier_lang::PreclassifierLangClass,
    > { tuple_to_bucket(tuple, &preclassifier) };
    let forecast_function =
        History::<watershed_shared::preclassifier_lang::PreclassifierLangClass>::forecast_fn();
    let deadline_ms = deadline_ms as u128;

    const DEFAULT_BATCH_SIZE: usize = 16;
    const SCHEDULING_BATCH_ENV_VAR: &str = "AQUIFER_SCHEDULING_BATCH_SIZE";
    let scheduling_batch_size = match std::env::var(SCHEDULING_BATCH_ENV_VAR) {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n @ 1..) => {
                info!("using scheduling batch size of {n} from env var {SCHEDULING_BATCH_ENV_VAR}");
                n
            }
            _ => {
                warn!("unrecognized value {v:?} for {SCHEDULING_BATCH_ENV_VAR}, using default scheduling batch size of {DEFAULT_BATCH_SIZE}");
                DEFAULT_BATCH_SIZE
            }
        },
        Err(_) => {
            info!(
                "{SCHEDULING_BATCH_ENV_VAR} not present, using default scheduling batch size of {DEFAULT_BATCH_SIZE}"
            );
            DEFAULT_BATCH_SIZE
        }
    };

    #[allow(unused)]
    #[derive(Debug, Clone, Copy)]
    enum LargeExcessPunishment {
        Ignore = 0,
        OverageRatio = 1,
        SqrtOverage = 2,
    }
    const DEFAULT_LARGE_EXCESS_PUNISHMENT: LargeExcessPunishment =
        LargeExcessPunishment::SqrtOverage;
    const EXCESS_ENV_VAR: &str = "AQUIFER_LARGE_EXCESS_PUNISHMENT";
    let large_excess_punishment = 'excess: {
        let Ok(mut setting) = std::env::var(EXCESS_ENV_VAR) else {
            info!(
                "{EXCESS_ENV_VAR} not present, using default excess punishment strategy of {DEFAULT_LARGE_EXCESS_PUNISHMENT:?}"
            );
            break 'excess DEFAULT_LARGE_EXCESS_PUNISHMENT;
        };
        setting.make_ascii_lowercase();
        let setting = setting.trim();
        match setting {
            "0" | "ignore" | "ignore_excess" | "ignoreexcess" | "ignore excess" => {
                info!("using ignore excess punishment strategy from env var {EXCESS_ENV_VAR}");
                LargeExcessPunishment::Ignore
            }
            "1" | "overage" | "overageratio" | "overage ratio" => {
                info!(
                    "using overage ratio excess punishment strategy from env var {EXCESS_ENV_VAR}"
                );
                LargeExcessPunishment::OverageRatio
            }
            "2" | "sqrt" | "squareroot" | "sqrt overage" | "sqrt overage ratio" | "sqrtoverage" => {
                info!(
                    "using square root of overage excess punishment strategy from env var {EXCESS_ENV_VAR}"
                );
                LargeExcessPunishment::SqrtOverage
            }
            _ => {
                warn!("unrecognized value {setting:?} for {EXCESS_ENV_VAR}, using default excess punishment strategy of {DEFAULT_LARGE_EXCESS_PUNISHMENT:?}");
                DEFAULT_LARGE_EXCESS_PUNISHMENT
            }
        }
    };

    #[derive(Debug, Clone, Copy)]
    enum BudgetCalculationVersion {
        V1 = 1,
        V2 = 2,
        V3 = 3,
    }
    const DEFAULT_BUDGET_CALCULATION_VERSION: BudgetCalculationVersion =
        BudgetCalculationVersion::V2;
    const BUDGET_CALCULATION_ENV_VAR: &str = "AQUIFER_BUDGET_CALCULATION_VERSION";
    let budget_calculation_version = match std::env::var(BUDGET_CALCULATION_ENV_VAR) {
        Ok(v) => match v.trim().parse::<u8>() {
            Ok(n @ 1..=3) => {
                info!(
                    "using budget calculation version {n} from env var {BUDGET_CALCULATION_ENV_VAR}"
                );
                match n {
                    1 => BudgetCalculationVersion::V1,
                    2 => BudgetCalculationVersion::V2,
                    3 => BudgetCalculationVersion::V3,
                    _ => unreachable!(),
                }
            }
            _ => {
                warn!("unrecognized value {v:?} for {BUDGET_CALCULATION_ENV_VAR}, using default budget calculation version of {DEFAULT_BUDGET_CALCULATION_VERSION:?}");
                DEFAULT_BUDGET_CALCULATION_VERSION
            }
        },
        Err(_) => {
            info!(
                "{BUDGET_CALCULATION_ENV_VAR} not present, using default budget calculation version of {DEFAULT_BUDGET_CALCULATION_VERSION:?}"
            );
            DEFAULT_BUDGET_CALCULATION_VERSION
        }
    };

    let mut last_mean_pending_age_ns: Option<f64> = None;
    let mut last_pending_count: Option<usize> = None;
    let mut last_weighted_delta_mean_pending_age_ns: f64 = 0.0;

    // --- PD STRATEGY CONFIGURATION (Loaded once at startup) ---
    const CULLING_RATIO_VAR: &str = "AQUIFER_CULLING_RATIO";
    let culling_ratio = std::env::var(CULLING_RATIO_VAR)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(1.05); // Default 1.05 (Tight culling)
    info!("Using {CULLING_RATIO_VAR}: {culling_ratio}");

    const PD_GAIN_P_VAR: &str = "AQUIFER_PD_GAIN_P";
    let pd_gain_p = std::env::var(PD_GAIN_P_VAR)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(1.05); // Default 1.05 (calibA)
    info!("Using {PD_GAIN_P_VAR}: {pd_gain_p}");

    const PD_GAIN_D_VAR: &str = "AQUIFER_PD_GAIN_D";
    let pd_gain_d = std::env::var(PD_GAIN_D_VAR)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(1.0); // Default 1.0
    info!("Using {PD_GAIN_D_VAR}: {pd_gain_d}");

    const DELTA_LENIENCY_CAP_VAR: &str = "AQUIFER_DELTA_LENIENCY_CAP";
    let delta_leniency_cap = std::env::var(DELTA_LENIENCY_CAP_VAR)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(1.05); // Default 1.05 (Max 5% reward)
    info!("Using {DELTA_LENIENCY_CAP_VAR}: {delta_leniency_cap}");

    move |mut tuples, senders| {
        // Ensure history is fresh
        history.update();
        // Safety net: clean up any items that have been pending for too much relative to the deadline
        // This handles cases where items are dropped without ack (though we fixed the main leak),
        // or other edge cases.
        // Feb 16: Updated to use configurable `culling_ratio` (default 1.05) for tighter bounds
        let safety_limit_ns = (deadline_ms as f64 * culling_ratio) as u128 * 1_000_000;
        history.cleanup_stale_pending(safety_limit_ns);

        let now = std::time::Instant::now();
        let mut assignment_predictions: Vec<usize> = Vec::with_capacity(tuples.len());

        debug!("received {:?} tuples in routing function", tuples.len());
        // Dynamic channel check: Just ensure we have at least Drop + 1 Model
        if senders.len() < 2 {
            error!(
                "Expected at least 2 channels [drop, model1, ...], got {:?} channels",
                senders.len()
            );
            return None;
        }
        trace!("closure g0");

        let time_of_scheduling = Instant::now();
        let time_of_scheduling_ns = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(v) => v.as_nanos(),
            Err(_) => {
                error!("time went backwards from the epoch to now");
                return None;
            }
        };

        // filter out tuples that have been alive for too long
        let mut i = 0;
        let mut too_old = get_tuple_vec();
        while i < tuples.len() {
            let tuple = &tuples[i];
            let time_created_ns = tuple.unix_time_created_ns();
            let diff = time_of_scheduling_ns - time_created_ns;
            let diff_ms = diff / 1_000_000;
            if diff_ms > deadline_ms {
                history.record_ingress(1);
                history.add_past_data(PastData {
                    tuple_id: tuple.id() as usize,
                    category: binning_fn(tuple).id,
                    age_when_scheduling_ns: diff,
                    time_of_creation_ns: time_created_ns,
                    time_of_scheduling: time_of_scheduling,
                    time_merged: time_of_scheduling,
                    time_elapsed_ms: 0.0,
                    pipeline_id: 0,
                });
                warn!(
                    "tuple {} has been alive for {}ms, removing",
                    tuple.id(),
                    diff_ms
                );
                too_old.push(tuples.remove(i));
            } else {
                i += 1;
            }
        }

        if too_old.len() > 0 {
            warn!(
                "dropping {} tuples that have been alive for too long",
                too_old.len()
            );
            if let Err(e) = senders[0].send(too_old) {
                error!("failed to send to pipe 0: {e:?}");
            }
        }

        if tuples.is_empty() {
            return Some(0);
        }
        if tuples.len() > 1 {
            warn!("found more than 1 tuple in the routing function during tests designed for 1 at a time:\n{:#?}", tuples);
        }
        // simple computation: we don't get to use the time that was already exhausted earlier in the pipeline
        let mean_age_of_items = tuples.iter().fold(0.0, |acc, tuple| {
            let time_created = tuple.unix_time_created_ns();
            let diff = time_of_scheduling_ns - time_created;
            let diff_ms = diff as f64 / 1_000_000.0;
            acc + diff_ms
        }) / tuples.len() as f64;

        trace!("closure g1");
        let mut i = 0;
        for tuple in &tuples {
            trace!("closure g2-{i}");
            i += 1;
            let time_created = tuple.unix_time_created_ns();
            let diff = time_of_scheduling_ns - time_created;
            debug!(
                "micro diff for the tuple at the time of routing is {:?}",
                diff / 1_000
            );
        }

        let mut out_total: Option<usize> = None;
        let tuples_to_route = tuples.len();
        let overall_route_start_time = Instant::now();
        let mut tuple_iter = tuples.into_iter();
        let mut current_batch = get_tuple_vec();
        loop {
            if let Some(t) = tuple_iter.next() {
                current_batch.push(t);
                if current_batch.len() < scheduling_batch_size {
                    continue;
                }
            } else {
                // we have already done our last batch and we're ready to exit
                if current_batch.is_empty() {
                    break;
                }
            }
            // if we made it here then either the current batch is the max length or we have reached the end
            let mut tuples = current_batch;
            current_batch = get_tuple_vec();
            current_batch.clear();

            let current_tuple = &tuples[0];

            let deadline_ns = deadline_ms as f64 * 1_000_000.0;
            let mut rate_per_item_ns = history
                .fine_ingress_rate_ns_per_item()
                .unwrap_or(deadline_ns);
            let mut mean_age_per_item_ns = history.mean_final_age_ns().unwrap_or(0.0);
            // we are on average this much over budget
            if rate_per_item_ns <= 0.0 {
                warn!("rate per item {rate_per_item_ns:.2} is 0 or negative, using abs or deadline as rate");
                rate_per_item_ns = rate_per_item_ns.abs().min(deadline_ns);
            }
            if mean_age_per_item_ns <= 0.0 {
                warn!("mean age per item {mean_age_per_item_ns:.2} is 0 or negative, using abs or deadline as mean age");
                mean_age_per_item_ns = mean_age_per_item_ns.abs().min(deadline_ns);
            }
            let mean_overage_ratio = mean_age_per_item_ns / rate_per_item_ns;
            let budget_per_item_ns = if mean_overage_ratio > 1.0 {
                // we are on average over budget, so we need to reduce our budget per item
                let val = rate_per_item_ns / mean_overage_ratio;
                info!("mean overage ratio is {mean_overage_ratio:.2}, reducing budget per item to {val:.2}ns");
                val
            } else {
                // we are on average under budget, so we can use the rate per item as is
                rate_per_item_ns
            };
            let items_per_ms = 1_000_000.0 / rate_per_item_ns;
            let lookahead_ms = f64::max(lookahead_ms as f64, 1.0);
            let items_in_lookahead = items_per_ms * lookahead_ms;
            let budget_in_lookahead_ns = items_in_lookahead * budget_per_item_ns;

            let budget_ms = ((budget_in_lookahead_ns / 1_000_000.0) as f64).max(0.0);

            // ADDED LOGGING FOR DEPLOYMENT DEBUGGING
            info!("animal_query: sending batch of size {} to scheduler. Rate per item: {:.2}ns, Mean age per item: {:.2}ns, budget_ms: {:.2}", 
                tuples.len(), rate_per_item_ns, mean_age_per_item_ns, budget_ms);
            // We calculated the budget based on the requirements for this current tuple,
            // but we want to allow the algorithm to look ahead a bit, so we scale the budget
            // by a ratio that allows it to look ahead for the desired lookahead time.
            let budget_ms_v1 = budget_ms * budget_to_lookahead_ratio;
            debug!(
                "v1 budget per item ns: {budget_per_item_ns:.2}, items per ms: {items_per_ms:.2}, lookahead ms: {lookahead_ms:.2}, items in lookahead: {items_in_lookahead:.2}, budget in lookahead ns: {budget_in_lookahead_ns:.2}, budget ms: {budget_ms_v1:.2}"
            );
            debug!(
                "v1 version of budget_per_item_ns: {rate_per_item_ns:.2}ns, or adjusted to {}ns",
                rate_per_item_ns / mean_overage_ratio
            );

            // now let's make a new version based on how much the queues are lagging
            // we are scheduling items every `rate_per_item_ns` nanoseconds,
            // and if that has negative consequences, then the average time elapsed for the items
            // in the history will be increasing over time.
            let _mean_elapsed_increase_ms = history.mean_elapsed_increase_ms();
            // we will care more about how that's been going recently, so we will use the weighted version
            let weighted_mean_elapsed_increase_ms =
                history.recent_weighted_mean_elapsed_increase_ms();

            // what about the age of the items themselves? are they continuing to get too old?
            let _mean_final_age_ms = history.mean_age_when_merging_increase_ms();
            let weighted_mean_final_age_increase_ms =
                history.recent_weighted_mean_age_when_merging_increase_ms();
            debug!(
                "Recent History consequences: mean elapsed increase: {weighted_mean_elapsed_increase_ms:.2} ms, recent-weighted mean elapsed increase: {weighted_mean_elapsed_increase_ms:.2} ms, mean final age increase: {weighted_mean_final_age_increase_ms:.2} ms, recent-weighted mean final age increase: {weighted_mean_final_age_increase_ms:.2} ms",
            );
            // --- PENDING AGE & QUEUE METRICS (Early Detection) ---
            let current_mean_pending_age_ns = history.mean_pending_age_ns();
            let current_max_pending_age_ns = history.max_pending_age_ns();
            let current_pending_count = history.pending_count();

            // Calculate Deltas
            let delta_mean_pending_age_ns =
                match (current_mean_pending_age_ns, last_mean_pending_age_ns) {
                    (Some(curr), Some(last)) => curr - last,
                    _ => 0.0,
                };

            // Calculate Exponential Moving Average (EMA) of Delta
            let ema_alpha = 0.1;
            let weighted_delta_mean_pending_age_ns = ema_alpha * delta_mean_pending_age_ns
                + (1.0 - ema_alpha) * last_weighted_delta_mean_pending_age_ns;

            let delta_pending_count = match last_pending_count {
                Some(last) => current_pending_count as isize - last as isize,
                None => 0,
            };

            // Update State
            last_mean_pending_age_ns = current_mean_pending_age_ns;
            last_pending_count = Some(current_pending_count);
            last_weighted_delta_mean_pending_age_ns = weighted_delta_mean_pending_age_ns;

            let mean_pending_age_ms = current_mean_pending_age_ns.unwrap_or(0.0) / 1_000_000.0;
            let max_pending_age_ms = current_max_pending_age_ns.unwrap_or(0.0) / 1_000_000.0;

            // --- PENALTY STRATEGIES ---
            let allowed_time_ms = deadline_ms as f64;

            // 1. Linear Mean Penalty: Reduce budget by how much items are already waiting on average
            let strategy_linear_mean_penalty = mean_pending_age_ms;

            // 2. Linear Max Penalty: Reduce budget by the OLDEST item (Conservative - catch outliers)
            let strategy_linear_max_penalty = max_pending_age_ms;

            // 3. Proportional Mean: Scale down based on % of deadline consumed
            // If mean pending is 500ms and deadline is 1000ms, budget is halved.
            let ratio_mean = mean_pending_age_ms / allowed_time_ms.max(1.0);
            let strategy_prop_mean_budget = lookahead_ms * (1.0 - ratio_mean).max(0.0);

            // 4. Queue Delta Penalty: If queue is growing, penalize.
            // Heuristic: If queue grew by 10 items, reduce budget by 10ms per item? (Tuning needed)
            let strategy_queue_delta_penalty = if delta_pending_count > 0 {
                delta_pending_count as f64 * 10.0 // arbitrary 10ms cost per new item
            } else {
                0.0
            };

            // 5. PD Mean Strategy (Multistage Proportional-Derivative)
            // P-Stage: Penalize based on current Mean Age
            let ratio_p = (mean_pending_age_ms / allowed_time_ms.max(1.0)) * pd_gain_p;
            let budget_base = lookahead_ms;
            let budget_p = budget_base * (1.0 - ratio_p);

            // D-Stage: Penalize/Reward based on Rate of Change (Delta EMA)
            let ratio_d =
                (weighted_delta_mean_pending_age_ns / 1_000_000.0 / allowed_time_ms.max(1.0))
                    * pd_gain_d;

            // Calculate Multiplier from D-term (1.0 - Ratio_D)
            // If Ratio_D is negative (queue clearing), this > 1.0 (Reward)
            // If Ratio_D is positive (queue growing), this < 1.0 (Penalty)
            let raw_multiplier = 1.0 - ratio_d;

            // Apply Leniency Cap (don't reward too much for clearing)
            let clipped_multiplier = if raw_multiplier > delta_leniency_cap {
                delta_leniency_cap
            } else {
                raw_multiplier
            };

            // Calculate Final Budget
            let strategy_pd_mean_budget = (budget_p * clipped_multiplier).max(0.0);

            // 6. PD Max Strategy (Conservative P-Term + Trend D-Term)
            // P-Stage: Penalize based on Max Age (Conservative)
            let ratio_p_max = (max_pending_age_ms / allowed_time_ms.max(1.0)) * pd_gain_p;
            let budget_base_max = lookahead_ms;
            let budget_p_max = budget_base_max * (1.0 - ratio_p_max);

            // D-Stage: Same Delta Trend logic as PD Mean
            // We reuse `clipped_multiplier` from above as the derivative term is identical
            let strategy_pd_max_budget = (budget_p_max * clipped_multiplier).max(0.0);

            // --- SELECT ACTIVE STRATEGY ---
            let active_strategy = std::env::var("ADAPTIVE_PENALTY_STRATEGY")
                .unwrap_or_else(|_| "pd_mean".to_string());
            let log_shadow_budgets = std::env::var("AQUIFER_LOG_SHADOW_BUDGETS")
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false);

            debug!(
                "Pending Metrics: MeanAge={:.2}ms, MaxAge={:.2}ms, Count={}, DeltaMean={:.2}ms, WeightedDeltaMean={:.2}ms, DeltaCount={}",
                mean_pending_age_ms,
                max_pending_age_ms,
                current_pending_count,
                delta_mean_pending_age_ns / 1_000_000.0,
                weighted_delta_mean_pending_age_ns / 1_000_000.0,
                delta_pending_count
            );

            // Logging "Shadow" Budgets only if requested or for the active strategy
            if log_shadow_budgets {
                let shadow_budget_linear_mean =
                    (allowed_time_ms - strategy_linear_mean_penalty).max(0.0);
                let shadow_budget_linear_max =
                    (allowed_time_ms - strategy_linear_max_penalty).max(0.0);

                info!(
                    "Shadow Budgets (Pending): LinearMean={:.2}ms, LinearMax={:.2}ms, PropMean={:.2}ms, QueueDeltaPenalty={:.2}ms",
                    shadow_budget_linear_mean,
                    shadow_budget_linear_max,
                    strategy_prop_mean_budget,
                    strategy_queue_delta_penalty
                );
            }

            if active_strategy == "pd_mean" || log_shadow_budgets {
                info!(
                    "PD Mean Analysis (Active={}): BudgetBase={:.2}ms, RatioP={:.2}, BudgetP={:.2}ms, RatioD={:.2}, Multiplier={:.2}, FinalBudget={:.2}ms",
                    active_strategy == "pd_mean",
                    lookahead_ms,
                    ratio_p,
                    budget_p,
                    ratio_d,
                    clipped_multiplier,
                    strategy_pd_mean_budget
                );
            }

            if active_strategy == "pd_max" || log_shadow_budgets {
                info!(
                    "PD Max Analysis (Active={}): BudgetBase={:.2}ms, RatioP_Max={:.2}, BudgetP_Max={:.2}ms, RatioD={:.2}, Multiplier={:.2}, FinalBudget={:.2}ms",
                    active_strategy == "pd_max",
                    lookahead_ms,
                    ratio_p_max,
                    budget_p_max,
                    ratio_d,
                    clipped_multiplier,
                    strategy_pd_max_budget
                );
            }

            let (penalty_metric_ms_v4, is_v4_active) = match active_strategy.as_str() {
                "linear_mean" => (strategy_linear_mean_penalty, true),
                "linear_max" => (strategy_linear_max_penalty, true),
                "prop_mean" => (0.0, true), // Special handling downstream
                "pd_mean" => (0.0, true),   // Special handling downstream
                "pd_max" => (0.0, true),    // Special handling downstream
                "none" | "legacy" => (weighted_mean_final_age_increase_ms, false),
                _ => (0.0, true), // Default to pd_mean handling
            };

            // If we selected a V4 strategy, use it. Otherwise fall back to V3 logic (Survivor Bias prone)
            let penalty_metric_ms = if is_v4_active {
                if active_strategy == "prop_mean" {
                    // Special case: Proportional doesn't map cleanly to "penalty ms" in the same linear way
                    // but we can approximate: penalty = lookahead - prop_budget?
                    // Or just override `excess_ratio` logic below.
                    // For now, let's map it: penalty = allowed * ratio
                    mean_pending_age_ms
                } else if active_strategy == "pd_mean" {
                    // Map PD Mean Budget back to an effective "penalty" for consistent downstream logic
                    // penalty = lookahead - budget
                    (lookahead_ms - strategy_pd_mean_budget).max(0.0)
                } else if active_strategy == "pd_max" {
                    (lookahead_ms - strategy_pd_max_budget).max(0.0)
                } else {
                    penalty_metric_ms_v4
                }
            } else {
                weighted_mean_final_age_increase_ms
            };

            // Old Logic, but now `penalty_metric_ms` might be derived from Pending Age
            // let penalty_metric_ms = weighted_mean_final_age_increase_ms;

            let excess_ratio = penalty_metric_ms / allowed_time_ms;
            debug!(
            "penalty metric is {penalty_metric_ms:.2}ms, allowed time is {allowed_time_ms:.2}ms, excess ratio is {excess_ratio:.2}"
        );
            if excess_ratio > 0.0 {
                info!("penalty metric is {penalty_metric_ms:.2}ms, allowed time is {allowed_time_ms:.2}ms, excess ratio is {excess_ratio:.2}");
            }
            let history_penalized_lookahead_ms = if excess_ratio > 1.0 {
                error!(
                "we are more than 100% over budget ({excess_ratio:.2}), so we will not schedule any items"
            );
                0.0
            } else if excess_ratio > 0.1 {
                info!("we are more than 10% over budget ({excess_ratio:.2}), so we will greatly reduce our budget per item");
                let penalized = match large_excess_punishment {
                    LargeExcessPunishment::Ignore => lookahead_ms,
                    LargeExcessPunishment::OverageRatio => lookahead_ms * (1.0 - excess_ratio),
                    LargeExcessPunishment::SqrtOverage => {
                        lookahead_ms * (1.0 - excess_ratio.sqrt())
                    }
                };
                info!(
                    "we are over budget by {excess_ratio:.2}, reducing budget to {penalized:.2}ms"
                );
                penalized
            } else if excess_ratio > 0.0 {
                if let LargeExcessPunishment::Ignore = large_excess_punishment {
                    debug!("ignoring small excess ratio of {excess_ratio:.2}, using full lookahead of {lookahead_ms:.2}ms");
                    lookahead_ms
                } else {
                    info!(
                        "we are slightly over budget by {excess_ratio:.2}, so we will reduce our budget per item"
                    );
                    // we are over budget, but not by much, so we will reduce our budget per item
                    let reduced_budget_ms = lookahead_ms * (1.0 - excess_ratio);
                    info!(
                        "we are over budget by {excess_ratio:.2}, reducing budget to {reduced_budget_ms:.2}ms"
                    );
                    reduced_budget_ms
                }
            } else {
                // we are under budget, so we can use the full budget
                lookahead_ms
            };
            debug!(
                "budget method v2 (history penalized with setting {large_excess_punishment:?} produced budget of: {history_penalized_lookahead_ms:.2}ms for a maximum of greedy={greedy_max_count} and optimal={optimal_max_count} items",
            );
            let budget_ms_v2 = history_penalized_lookahead_ms;

            let initial_budget_adjustment = mean_age_of_items / allowed_time_ms;
            let adjusted_budget_ms = lookahead_ms * (1.0 - initial_budget_adjustment);
            debug!(
                "mean age of items is {mean_age_of_items:.2}ms, max age is {allowed_time_ms:.2}ms, initial budget adjustment is {initial_budget_adjustment:.3}, adjusted budget is {adjusted_budget_ms:.2}ms",
            );
            // we can scale based on the earlier penalty
            const MAX_OVERAGE_RECOVERY_RATIO: f64 = 1.0 + 0.05; // 5% recovery
                                                                // we are allowed to lose a lot of the budget, but we can only recover a little bit in order to be conservative with how wrong we can be
            let budget_ms_v3 = (adjusted_budget_ms
                * f64::clamp(
                    1.0 - (penalty_metric_ms / allowed_time_ms),
                    0.0,
                    MAX_OVERAGE_RECOVERY_RATIO,
                ))
            .max(0.0);
            // TODO: additional adjustments based on the history of the items and based on the drop rates/rejection rates from the channels
            info!(
                "Budget decision v3: final={:.2}ms (penalty={:.2}ms, allowed={:.2}ms, ratio={:.3})",
                budget_ms_v3,
                penalty_metric_ms,
                allowed_time_ms,
                penalty_metric_ms / allowed_time_ms.max(1.0)
            );

            trace!("closure g3");
            let tuples_len = tuples.len();
            let alg_inputs = AlgInputs {
                binning_function: &binning_fn,
                forecast_function: &forecast_function,
                send_function: History::send,
            };

            let budget_ms = match budget_calculation_version {
                BudgetCalculationVersion::V1 => budget_ms_v1,
                BudgetCalculationVersion::V2 => budget_ms_v2,
                BudgetCalculationVersion::V3 => budget_ms_v3,
            };
            let tuple_id = current_tuple.id();
            let start_scheduling_time = Instant::now();
            let out = scheduler::lookahead_problem_scheduler(
                tuples,
                senders,
                &mut history,
                alg_inputs,
                budget_ms,
                match strategy {
                    scheduler::Strategy::Greedy => FutureWindowKind::TimeWithMaximumCount {
                        // time_ms: budget_ms.floor() as u128,
                        // time_ms: deadline_ms,
                        // we use the lookahead time instead of the budget time, and it will be scaled down if we don't have enough time for it all
                        time_ms: lookahead_ms as _,
                        max_count: greedy_max_count,
                    },
                    scheduler::Strategy::Optimal => FutureWindowKind::TimeWithMaximumCount {
                        // time_ms: budget_ms.floor() as u128,
                        // time_ms: deadline_ms,
                        // we use the lookahead time instead of the budget time, and it will be scaled down if we don't have enough time for it all
                        time_ms: lookahead_ms as _,
                        max_count: optimal_max_count,
                    },
                },
                strategy,
            );

            if let Some(count) = out {
                info!(
                    "Scheduler mapped batch of {} tuples: {} items sent to model, {} dropped/other",
                    tuples_len,
                    count,
                    tuples_len - count
                );
            } else {
                warn!("Scheduler returned None for batch of {} tuples", tuples_len);
            }

            let end_scheduling_time = Instant::now();
            let scheduling_duration = end_scheduling_time - start_scheduling_time;
            let scheduling_duration_ms = scheduling_duration.as_secs_f64() * 1_000.0;

            'log_scheduling_time: {
                // use global logger
                let log_location = "aquifer_routing_time".to_raw_key();
                let aux_data = [(
                    "scheduling_duration_ms".to_raw_key(),
                    LimitedHabValue::Float(scheduling_duration_ms),
                )];
                if let Err(e) = watershed_shared::global_logger::log_data(
                    tuple_id,
                    log_location,
                    Some(aux_data),
                ) {
                    for err in e {
                        error!("failed to log time to schedule: {err}");
                    }
                }
            }

            trace!("closure g5");
            match (&mut out_total, out) {
                (Some(total), Some(count)) => *total += count,
                (None, Some(count)) => out_total = Some(count),
                // if we didn't send anything, then we just keep it as None
                (Some(_total), None) => {}
                (None, None) => {}
            }

            // individual_scheduling_durations.push(individual_scheduling_start_time.elapsed());
        } // end for current_tuple in tuples
        let overall_route_end_time = Instant::now();
        let overall_route_duration_micros: f64 =
            (overall_route_end_time - overall_route_start_time).as_nanos() as f64 / 1_000.0;
        let gross_mean_route_duration_micros = if tuples_to_route > 0 {
            overall_route_duration_micros / tuples_to_route as f64
        } else {
            0.0
        };
        debug!(
            "overall routing duration for {tuples_to_route} items: {overall_route_duration_micros:.2} us, gross mean per item: {gross_mean_route_duration_micros:.2} us",
        );
        trace!("closure g6");
        out_total
    }
}

fn routing_fn_static(
    routing_option: RoutingOptions,
    // window_size: usize,
    // deadline_window_ms: u64,
    // back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    move |tuples, senders| {
        let amount = tuples.len();
        for t in tuples {
            let tuple_id = t.id();
            let pipe = match routing_option {
                // new query options where we only use tiny or huge
                RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny) => 1,
                RoutingOptions::Named(NamedRoutingOptions::AlwaysSmall) => 2,
                RoutingOptions::Named(NamedRoutingOptions::AlwaysBig) => 3,
                RoutingOptions::Named(NamedRoutingOptions::AlwaysHuge) => {
                    // Try to send to the last pipe if possible
                    if senders.len() > 1 {
                        senders.len() - 1
                    } else {
                        // Fallback to pipe 2 if it exists, else 1
                        2
                    }
                }
                RoutingOptions::Fixed(n) => n,
                _ => {
                    error!("incorrect static routing option {routing_option:?}");
                    continue;
                }
            };
            if pipe >= senders.len() {
                error!("static routing option {routing_option:?} mapped to pipe {pipe} which does not exist (len={})", senders.len());
                continue;
            }
            // it will be automatically dropped if it is able to be sent. no need to worry
            if let Err(e) = senders[pipe].send(vec![t]) {
                error!("failed to send tuple with id {tuple_id} to pipe {pipe}: {e:?}");
            }
        }
        Some(amount)
    }
}

fn routing_fn_eddies0(
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    // This looks unused or older version, but updated for consistency if called
    move |tuples, senders| {
        let amount = tuples.len();
        let mut rng = rand::thread_rng();
        // Dynamic construction of available routes
        let mut available_routes = vec![];
        if senders.len() > 1 {
            available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny));
        }
        if senders.len() > 2 {
            available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysSmall));
        }
        if senders.len() > 3 {
            available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysBig));
        }
        if senders.len() > 1 {
            available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysHuge));
        }

        let mut current_tuple_vec = Some(get_tuple_vec());
        for t in tuples {
            let mut current_vec = current_tuple_vec.take().unwrap_or_else(get_tuple_vec);
            current_vec.push(t);
            // randomly retry sending until we find a route that's available
            // if we run out of options then we drop
            'next_tuple: {
                loop {
                    if available_routes.is_empty() {
                        break;
                    }
                    let routing_option_index = rng.gen_range(0..available_routes.len());
                    let routing_option = available_routes.remove(routing_option_index);
                    let pipe = match routing_option {
                        RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny) => 1,
                        RoutingOptions::Named(NamedRoutingOptions::AlwaysSmall) => 2,
                        RoutingOptions::Named(NamedRoutingOptions::AlwaysBig) => 3,
                        RoutingOptions::Named(NamedRoutingOptions::AlwaysHuge) => senders.len() - 1,
                        _ => {
                            error!("incorrect eddies routing option {routing_option:?}");
                            break 'next_tuple;
                        }
                    };
                    if pipe >= senders.len() {
                        continue;
                    }

                    // it will be automatically dropped if it is able to be sent. no need to worry
                    match &senders[pipe] {
                        AsyncPipe::Active(bounded_async_sender) => {
                            match bounded_async_sender.try_send_and_return(current_vec) {
                                Ok(_success) => break 'next_tuple, // nothing further to do
                                Err((leftover, Ok(_) | Err(AsyncPipeSendError::Full))) => {
                                    // we have to send the leftover back to the dummy pipe
                                    current_vec = leftover;
                                    continue;
                                } // try again with the next one
                                Err((
                                    _unusable_leftover,
                                    Err(AsyncPipeSendError::Disconnected),
                                )) => {
                                    // we can't continue if it's disconnected. time to escape
                                    return_tuple_vec(_unusable_leftover);
                                    break 'next_tuple;
                                }
                            }
                        }
                        dummy @ AsyncPipe::Dummy => {
                            if let Err(e) = dummy.send(current_vec) {
                                error!("failed to send to pipe 0: {e:?}");
                            }
                            break 'next_tuple;
                        }
                    };
                }
                // we didn't find any available one. send it through the dummy pipe 0
                if let Err(e) = senders[0].send(current_vec) {
                    error!("failed to send to pipe 0: {e:?}");
                }
            }
            available_routes.clear();
            // Re-populate for next tuple
            if senders.len() > 1 {
                available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny));
            }
            if senders.len() > 2 {
                available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysSmall));
            }
            if senders.len() > 3 {
                available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysBig));
            }
            if senders.len() > 1 {
                available_routes.push(RoutingOptions::Named(NamedRoutingOptions::AlwaysHuge));
            }
        }
        Some(amount)
    }
}

fn routing_fn_eddies(
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    const BASELINE_MAX_PIPELINES: usize = 16;
    move |tuples, senders| {
        let amount = tuples.len();
        let mut rng = rand::thread_rng();

        // Dynamically build route options based on actual senders length
        // We assume senders[0] is Drop, senders[1..] are models
        // We just iterate all available models
        let _model_count = if senders.len() > 1 {
            senders.len() - 1
        } else {
            0
        }; // We want to map these to RoutingOptions just for the sake of the struct, OR
           // just construct the available routes using indices directly if we refactored.
           // But to keep structure:
        let mut available_routes: smallvec::SmallVec<[_; BASELINE_MAX_PIPELINES]> =
            smallvec::SmallVec::new();

        // This zip/map logic was complex. Let's simplify.
        // We want a list of (index, pipe_len, pipe_cap, pipe_ref) for all valid model pipes.
        for i in 1..senders.len() {
            let pipe = &senders[i];
            // We can use a dummy RoutingOption as a placeholder if needed, or just remove it from tuple
            // The original code used it. Let's just use AlwaysTiny as placeholder.
            if pipe.cap() > pipe.len() {
                available_routes.push((
                    i,
                    RoutingOptions::Named(NamedRoutingOptions::AlwaysTiny),
                    pipe.len(),
                    pipe.cap(),
                    pipe,
                ));
            }
        }

        let mut outputs: watershed_shared::ws_types::ArrayMap<
            usize,
            (Vec<Tuple>, &AsyncPipe),
            BASELINE_MAX_PIPELINES,
        > = Default::default();
        'next_tuple: for t in tuples {
            // Filter again for capacity
            available_routes.retain(|(_i, _o, l, c, _p)| *c > *l);

            if available_routes.is_empty() {
                outputs
                    .entry(0)
                    .or_insert_with(|| (get_tuple_vec(), &senders[0]))
                    .0
                    .push(t);
                continue 'next_tuple;
            }
            let total: f64 = available_routes
                .iter()
                .map(|(_i, _o, l, c, _p)| 1.0 - ((*l as f64) / (*c as f64)))
                .sum();
            let mut routing_weight: f64 = rng.gen::<f64>() * total;

            for (i, _o, l, c, p) in &mut available_routes {
                routing_weight -= 1.0 - ((*l as f64) / (*c as f64));
                if routing_weight <= 0.0 {
                    outputs
                        .entry(*i)
                        .or_insert_with(|| (get_tuple_vec(), p))
                        .0
                        .push(t);
                    *l += 1;
                    continue 'next_tuple;
                }
            }
            warn!("we ran out of weights (remaining routing weight={routing_weight:?}), sending tuple {} to backup pipeline 0", t.id());
            outputs
                .entry(0)
                .or_insert_with(|| (get_tuple_vec(), &senders[0]))
                .0
                .push(t);
        }
        for (p_idx, (v, p)) in outputs {
            let num_to_send = v.len();
            if let Err(e) = p.send(v) {
                error!("Failed to send {num_to_send} tuples to pipeline {p_idx}: {e:?}");
            }
        }
        Some(amount)
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use ordered_float::OrderedFloat;
    use watershed_shared::preclassifier_lang;

    #[test]
    fn test_tuple_to_bucket_integration() {
        // 1. Load Preclassifier
        // Path relative to Cargo.toml (animal_query/)
        let preclassifier_path =
            "../animal_preclassifiers_feb6_611pm/test/nano_xlarge/inference/mean/thirds.json";

        let path = std::path::Path::new(preclassifier_path);
        if !path.exists() {
            eprintln!(
                "Preclassifier not found at {:?}",
                std::fs::canonicalize(".")
                    .ok()
                    .map(|p| p.join(preclassifier_path))
            );
        }

        let model_info =
            std::fs::read_to_string(preclassifier_path).expect("Failed to read preclassifier file");
        let preclassifier = preclassifier_lang::load_file_format(model_info.as_bytes())
            .expect("Failed to parse preclassifier");

        // 2. Create Mock Tuple with 59 features
        let mut tuple = get_tuple();
        // Create 59 dummy features
        let features: Vec<HabValue> = (0..59)
            .map(|i| HabValue::Float(OrderedFloat(i as f64)))
            .collect();
        tuple.insert("streaming_features".into(), HabValue::List(features));
        // Add required fields to avoid panic in logging if error occurs
        tuple.insert("img_id".into(), HabValue::String("test_img".into()));

        // 3. Call tuple_to_bucket
        let bucket = tuple_to_bucket(&tuple, &preclassifier);

        // 4. Verify result
        println!("Mapped to bucket: {:?}", bucket);
        // We expect a valid bucket ID (likely 0 or 1)
        assert!(bucket.id.is_some());
    }
}
