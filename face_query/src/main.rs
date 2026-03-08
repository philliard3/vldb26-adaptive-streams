#![allow(unused_labels)]

mod face_utils;
mod omz_utils;

use dashmap::DashMap;
use face_utils::decode_embeddings_inline;
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

use crate::face_utils::INDIVIDUAL_BOX_BOUND_FIELD;
use crate::face_utils::{
    EXPECTED_MATCHES_FIELD, ORIGINAL_IMAGE_FIELD, ORIGINAL_IMAGE_ID_FIELD,
    ORIGINAL_IMAGE_ID_INT_FIELD,
};

#[derive(Debug, Deserialize, Serialize)]
struct FaceExperimentConfig {
    run_order_seed: Option<u64>,
    imdb_split_info_path: String,
    preclassifier_path: String,
    imdb_image_path: String,
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
    in_memory_embedding_index: Option<String>,
    blocking_noops: Option<Vec<BlockingNoopBoltConfig>>,
    reyhydrate_spouts: Option<Vec<RehydrateSpoutConfig>>,
}

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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
enum RoutingOptions {
    #[serde(rename = "huge")]
    AlwaysHuge,
    #[default]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageInfo {
    person_name: String,
    img_id: String,
    img_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImdbSplitInfo {
    // description: String,
    // training: Vec<ImageInfo>,
    testing_index: Vec<ImageInfo>,
    testing_remainder: Vec<ImageInfo>,
    // training_index_people: HashSet<String>,
    // testing_index_people: HashSet<String>,
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

    let face_config: FaceExperimentConfig =
        serde_json::from_str(&config).context("unable to parse face config")?;
    debug!(
        "proceeding with face config:\n{}\n",
        serde_json::to_string_pretty(&face_config)
            .with_context(|| "unable to JSON-pretty-print face config: {face_config:#?}")?
    );
    let FaceExperimentConfig {
        run_order_seed,
        imdb_split_info_path,
        preclassifier_path,
        imdb_image_path,
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
        in_memory_embedding_index,
        reyhydrate_spouts,
        blocking_noops,
    } = face_config;

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

    let window_size = history_window_size.unwrap_or(50);
    let deadline_window_ms = deadline_window_ms.unwrap_or(1_000);
    let lookahead_time_ms = lookahead_time_ms.unwrap_or(deadline_window_ms);
    // let target_time_micros =
    //     target_time_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let mut target_time_micros = target_time_micros.unwrap_or(Delay::Fixed(100_000));
    let max_target_time_ms = target_time_micros.max() / 1_000;

    // let input_delay_micros =
    //     input_delay_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let input_delay_micros = input_delay_micros.unwrap_or(Delay::Fixed(0));
    let max_input_delay_ms = input_delay_micros.max() / 1_000;
    let max_total_samples = max_total_samples.unwrap_or(usize::MAX);
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

    let model_info = std::fs::read_to_string(&preclassifier_path)?;
    let preclassifier =
        watershed_shared::preclassifier_lang::load_file_format(model_info.as_bytes())?;

    let ImdbSplitInfo {
        // training,
        testing_index,
        testing_remainder,
    } = serde_json::from_str(
        &std::fs::read_to_string(imdb_split_info_path).context("Failed to read imdb split info")?,
    )?;

    debug!("g2: read image split info");
    let testing_index_people: BTreeSet<HabString> = testing_index
        .iter()
        .map(|info| info.person_name.clone().into())
        .collect();
    let mut testing_remainder: Vec<(bool, ImageInfo)> = testing_remainder
        .into_iter()
        .map(|info| {
            (
                testing_index_people.contains(info.person_name.as_str()),
                info,
            )
        })
        .collect();
    info!(
        "There are {} people in the testing index",
        testing_index_people.len()
    );

    if let Some(run_order_seed) = run_order_seed {
        let mut order_rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(run_order_seed);
        testing_remainder.shuffle(&mut order_rng);
    }

    // build side of the join
    let id_to_person_name: Vec<Tuple> = testing_index
        .into_iter()
        .map(|image_info| {
            let mut tuple = get_tuple();
            tuple.insert(
                PERSON_NAME_FIELD.into(),
                HabValue::String(image_info.person_name.into()),
            );
            tuple.insert("img_id".into(), HabValue::String(image_info.img_id.into()));
            tuple
        })
        .collect();

    let id_to_person_name = Mutex::new(Some(id_to_person_name));
    let id_to_person_source_udf = FunctionKinds::SourceUdf(Box::new(move || {
        let mut id_to_person_name = id_to_person_name.lock().expect("mutex poisoned");
        let Some(id_to_person_name) = id_to_person_name.take() else {
            error!("id_to_person_name was already taken");
            panic!("id_to_person_name was already taken");
        };

        // let id_to_person_name = id_to_person_name.into_iter();
        // let id_to_person_name_stream = tokio_stream::iter(id_to_person_name);
        let id_to_person_name_stream = tokio_stream::once(id_to_person_name);
        Box::new(move || Box::new(id_to_person_name_stream))
    }));
    function_lookup.insert("person_info_source".into(), id_to_person_source_udf);

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
        let paths_to_ndarrays_settings = if should_loop_items {
            PathsToNdarrays2Settings::Loop {
                first_n: usize::MAX,
                num_times: usize::MAX,
                total_items: usize::MAX,
            }
        } else {
            PathsToNdarrays2Settings::Normal
        };
        std::thread::spawn(move || {
            let (img_tuple_iter, mut converter) = paths_to_ndarrays_v2(
                imdb_image_path,
                testing_remainder.into_iter(),
                paths_to_ndarrays_settings,
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
            std::thread::sleep(Duration::from_micros(
                (max_amt / 2).min(30_000_00_000).max(2_500_000),
            ));
            debug!("background tuple creation thread sending completion sender");
            if let Err(e) = completion_sender.send(true) {
                error!("tuple creation thread failed to send completion signal: {e:?}");
            }
            std::thread::sleep(Duration::from_micros(
                (max_amt / 2).min(30_000_00_000).max(2_500_000),
            ));
            // signal end
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

    function_lookup.insert(
        "preprocess_image".into(),
        FunctionKinds::FlatMapUdf(Box::new(move || Box::new(omz_utils::preprocess_image))),
    );
    // function_lookup.insert(
    //     "postprocess_boxes".into(),
    //     FunctionKinds::FlatMapUdf(Box::new(move || Box::new(omz_utils::postprocess_boxes))),
    // );
    function_lookup.insert(
        "postprocess_boxes".into(),
        FunctionKinds::FlatMapUdf(Box::new(move || {
            Box::new(|t| {
                omz_utils::postprocess_boxes(t)
                    .into_iter()
                    .flat_map(face_utils::split_bbs_before_scheduling)
                    .collect()
            })
        })),
    );

    // this stream is always empty because the data has already been cached
    const EMPTY_STREAM_DELAY: u64 = 25;
    let always_empty_stream = FunctionKinds::SourceUdf(Box::new(move || {
        let empty_item_future = async {
            tokio::time::sleep(Duration::from_millis(EMPTY_STREAM_DELAY)).await;
        };
        use futures::future::FutureExt;

        let empty_stream = empty_item_future
            .into_stream()
            .flat_map(|v| futures::stream::empty());
        Box::new(move || Box::new(empty_stream))
    }));

    function_lookup.insert("known_faces".into(), always_empty_stream);

    let detect_encode_fn = FunctionKinds::EncodeRemotePythonUdf(Box::new(|| {
        Box::new(face_utils::encode_image_for_detection)
    }));
    let detect_decode_fn = FunctionKinds::DecodeRemotePythonUdf(Box::new(|| {
        Box::new(face_utils::decode_bounding_boxes_from_detection)
    }));
    let embed_encode_fn = FunctionKinds::EncodeRemotePythonUdf(Box::new(|| {
        Box::new(face_utils::encode_for_embedding)
    }));
    let embed_decode_fn =
        FunctionKinds::DecodeRemotePythonUdf(Box::new(|| Box::new(face_utils::decode_embedding)));

    function_lookup.insert("encode_recognize_faces_input".into(), detect_encode_fn);
    function_lookup.insert("decode_recognize_faces_output".into(), detect_decode_fn);
    function_lookup.insert("encode_embedding_input".into(), embed_encode_fn);
    function_lookup.insert("decode_embedding_output".into(), embed_decode_fn);

    let shutdown_sequence_fn: FunctionKinds =
        FunctionKinds::ShutdownRemotePythonUdf(Box::new(|| {
            Box::new(face_utils::shutdown_sequence_detect)
        }));
    function_lookup.insert("shutdown_sequence_detect".into(), shutdown_sequence_fn);

    let shutdown_sequence_fn: FunctionKinds =
        FunctionKinds::ShutdownRemotePythonUdf(Box::new(|| {
            Box::new(face_utils::shutdown_sequence_embed)
        }));
    function_lookup.insert("shutdown_sequence_embed".into(), shutdown_sequence_fn);
    debug!("g4: shutdown function registered");

    let (route_feedback_sender, route_feedback_receiver) = crossbeam::channel::unbounded();
    let routing_udf: FunctionKinds = FunctionKinds::RoutingUdf(Box::new(move || {
        match routing_strategy {
            Some(option @ (
                | RoutingOptions::AlwaysSmall
                | RoutingOptions::AlwaysBig
            )) => {
                let error_msg = format!(
                    "Routing option {:?} is not implemented because that model proved to not have a sufficiently powerful tradeoff curve. Please use a different routing strategy.",
                    option
                );
                error!("{}", error_msg);
                unimplemented!("{}", error_msg);
            }
            Some(option @ (RoutingOptions::AlwaysTiny
                | RoutingOptions::AlwaysHuge
            )
             ) => Box::new(routing_fn_static(
                option,
                // window_size,
                // deadline_window_ms,
                // route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::Random) => unimplemented!(
                "Random routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::Eddies) => Box::new(routing_fn_eddies(
                // window_size,
                // deadline_window_ms,
                // route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::AquiferGreedy) => Box::new(aquifer_routing_fn(
                preclassifier.clone(),
                window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Greedy,
            )),
            Some(RoutingOptions::AquiferOptimal) => Box::new(aquifer_routing_fn(
                preclassifier.clone(),
                window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Optimal,
            )),
            Some(RoutingOptions::AlwaysDrop) => unimplemented!(
                "AlwaysDrop routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::PredictorBinary) => unimplemented!(
                "PredictorBinary routing is not implemented yet. Please use a different routing strategy."
            ),
            Some(RoutingOptions::PredictorProbabilistic) => unimplemented!(
                "PredictorProbabilistic routing is not implemented yet. Please use a different routing strategy."
            ),
            None => Box::new(routing_fn_static(
                RoutingOptions::AlwaysTiny,
                // route_feedback_receiver.clone(),
            )),
        }
    }));

    function_lookup.insert("routing_fn".into(), routing_udf);
    debug!("g5: routing function registered");
    // this is expected to be a (4 x n x 512) file with all the embeddings we ever need
    const IN_MEMORY_EMBEDDING_INDEX_FILE: &str = "in_memory_embedding_index.npy";
    let in_memory_embedding_index =
        in_memory_embedding_index.unwrap_or(IN_MEMORY_EMBEDDING_INDEX_FILE.into());
    let in_memory_embedding_index: ndarray::Array3<f32> =
        ndarray_npy::read_npy(&in_memory_embedding_index).with_context(|| {
            format!("unable to read embedding index file {in_memory_embedding_index:?}")
        })?;
    let in_memory_embedding_index = in_memory_embedding_index.into_shared();
    // const SINGLE_OVERALL_THRESHOLD: f32 = 1.15;

    // top 1 match. we can worry about more details later
    for (size, model_size_lookup_index, model_size_threshold) in [
        ("tiny", 0usize, 0.95),
        ("small", 1usize, 1.01),
        ("large", 2usize, 0.98),
        ("huge", 3usize, 0.79),
    ] {
        // let model_size_threshold = SINGLE_OVERALL_THRESHOLD;
        let local_embeddings = in_memory_embedding_index
            .slice(ndarray::s![model_size_lookup_index, .., ..])
            .to_shared();
        let mut known_bad_indices = BTreeSet::<usize>::new();
        if local_embeddings.shape().len() < 1 {
            let msg = format!(
                "in_memory_embedding_index had invalid shape: {:?}",
                local_embeddings.shape()
            );
            error!("{}", msg);
            return Err(anyhow::anyhow!(msg));
        }
        for local_index in 0..local_embeddings.shape()[0] {
            let embedding = local_embeddings.slice(ndarray::s![local_index, ..]);
            let mag = embedding
                .iter()
                .map(|&v| v as f64)
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt() as f32;
            // we only keep already-normalized embeddings
            if (1.0 - mag).abs() > 0.05 {
                known_bad_indices.insert(local_index);
            }
        }

        let v = FunctionKinds::FlatMapUdf(Box::new(move || {
            let local_embeddings = local_embeddings.clone();
            let known_bad_indices = known_bad_indices.clone();
            Box::new(move |mut t| {
                let local_embeddings: ndarray::ArrayView2<f32> = local_embeddings.view();
                let Some(embedding_buffer) = t.get("embedding") else {
                    error!("size {size}: tuple {} did not have any \"embedding\" field. available fields: {:?}", t.id(), t.keys().collect::<Vec<_>>());
                    return vec![];
                };
                let Some(embedding) = embedding_buffer.as_int_buffer() else {
                    error!("size {size}: embedding field was not an int buffer");
                    return vec![];
                };
                let closest_start = Instant::now();
                let embedding =
                    ndarray::ArrayView1::from(bytemuck::cast_slice::<i32, f32>(embedding));
                let mag = embedding
                    .iter()
                    .map(|&v| v as f64)
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt() as f32;

                // is it faster to let it clone everything or to try to do it incrementally?
                // where do I get the most simd? figure it out later
                // let diff  = local_embeddings.to_owned() - embedding.to_owned();

                let mut min_diff_index = usize::MAX;
                let mut min_diff = f64::MAX;
                for local_index in 0..local_embeddings.shape()[0] {
                    if known_bad_indices.contains(&local_index) {
                        continue;
                    }
                    let sum: f64 = local_embeddings
                        .slice(ndarray::s![local_index, ..])
                        .iter()
                        .zip(&embedding)
                        .map(|(a, b)| (a - (b / mag)).powi(2) as f64)
                        .sum();
                    // we don't need to sqrt all of them, only the minimum
                    // because the smallest doesn't change when we sqrt
                    // let diff = sum.sqrt() as f32;
                    let diff = sum;
                    if diff < min_diff {
                        min_diff = diff;
                        min_diff_index = local_index;
                    }
                }
                let min_diff = min_diff.sqrt() as f32;

                let closest_elapsed_micros: f64 =
                    closest_start.elapsed().as_nanos() as f64 / 1_000.0;
                debug!(
                    "knn {size}: tuple {} closest distance took {:.2} micros",
                    t.id(),
                    closest_elapsed_micros
                );
                let image_id = t.get(ORIGINAL_IMAGE_ID_FIELD);
                let image_id_int = t.get(ORIGINAL_IMAGE_ID_INT_FIELD);
                let (index_to_store, dist_to_store) = if min_diff > model_size_threshold {
                    warn!("knn {size}: tuple {} (image_id={:?}, image_id_int={:?}) rejected closest distance of {min_diff} with index embedding {min_diff_index}", t.id(), image_id, image_id_int);
                    (
                        HabValue::List(Default::default()),
                        (HabValue::List(Default::default())),
                    )
                } else {
                    info!("knn {size}: tuple {} (image_id={:?}, image_id_int={:?}) accepted closest distance of {min_diff} with index embedding {min_diff_index}", t.id(), image_id, image_id_int);
                    // ( HabValue::Integer(min_diff_index as _), HabValue::from(min_diff))
                    (
                        HabValue::from(min_diff_index.to_string()),
                        HabValue::from(min_diff as f64),
                    )
                };
                'log_accept_reject: {
                    let [x1, y1, x2, y2] =
                        crate::face_utils::get_bbox(&t, &INDIVIDUAL_BOX_BOUND_FIELD.into());
                    let expected_matches = match t.get(EXPECTED_MATCHES_FIELD) {
                        Some(HabValue::Integer(n)) if *n >= 0 => *n as usize,
                        Some(HabValue::UnsignedLongLong(n)) => *n as usize,
                        _ => {
                            // not present or not a nonnegative integer
                            warn!("tuple {} did not have a nonnegative integer \"expected_matches\" field, so we cannot log accept/reject properly. available fields: {:?}", t.id(), t.keys().collect::<smallvec::SmallVec<[_; 16]>>());
                            0
                        }
                    };
                    let image_id_int = match image_id_int {
                        Some(HabValue::Integer(n)) => *n as i64,
                        Some(HabValue::UnsignedLongLong(n)) => *n as i64,
                        _ => {
                            // not present or not an integer
                            warn!("tuple {} did not have an integer \"image_id_int\" field, so we cannot log accept/reject properly. available fields: {:?}", t.id(), t.keys().collect::<smallvec::SmallVec<[_; 16]>>());
                            i64::MIN
                        }
                    };
                    // currently we store bools as u64s
                    let did_accept = (min_diff <= model_size_threshold) as u64;
                    let aux_data = std::collections::HashMap::from([
                        (
                            "model_size".to_raw_key(),
                            LimitedHabValue::Integer(model_size_lookup_index as _),
                        ),
                        (
                            "image_id_int".to_raw_key(),
                            LimitedHabValue::Integer(image_id_int),
                        ),
                        (
                            "expected_matches".to_raw_key(),
                            (expected_matches as u64).into(),
                        ),
                        ("did_accept".to_raw_key(), did_accept.into()),
                        ("closest_distance".to_raw_key(), (min_diff as f64).into()),
                        ("closest_index".to_raw_key(), (min_diff_index as u64).into()),
                        (
                            "threshold".to_raw_key(),
                            (model_size_threshold as f64).into(),
                        ),
                        ("bbox_x1".to_raw_key(), LimitedHabValue::Integer(x1)),
                        ("bbox_y1".to_raw_key(), LimitedHabValue::Integer(y1)),
                        ("bbox_x2".to_raw_key(), LimitedHabValue::Integer(x2)),
                        ("bbox_y2".to_raw_key(), LimitedHabValue::Integer(y2)),
                    ]);
                    if let Err(e) = watershed_shared::global_logger::log_data(
                        t.id() as _,
                        "face_recognition_knn_decision".to_raw_key(),
                        Some(aux_data),
                    ) {
                        error!("failed to log knn decision: {e:?}");
                    }
                }
                t.insert("closest_distances".into(), dist_to_store);
                t.insert("match_ids".into(), index_to_store);

                let mut v = get_tuple_vec();
                v.push(t);
                v
            })
        }));
        function_lookup.insert(format!("in_memory_lookup_udf_{size}").into(), v);
    }
    debug!("g6: embedding knn functions registered");

    function_lookup.insert(
        "decode_embeddings_inline".into(),
        FunctionKinds::InlinePythonDecoder(Box::new(|| Box::new(decode_embeddings_inline))),
    );

    function_lookup.insert(
        "split_bbs_before_scheduling".into(),
        FunctionKinds::FlatMapUdf(Box::new(|| {
            Box::new(face_utils::split_bbs_before_scheduling)
        })),
    );
    function_lookup.insert(
        "preprocess_bb_after_scheduling_facenet".into(),
        FunctionKinds::FlatMapUdf(Box::new(|| Box::new(face_utils::preprocess_bb_facenet))),
    );

    function_lookup.insert(
        "preprocess_bb_after_scheduling_insightface".into(),
        FunctionKinds::FlatMapUdf(Box::new(|| Box::new(face_utils::preprocess_bb_insightface))),
    );

    function_lookup.insert(
        "split_bbs_facenet".into(),
        FunctionKinds::FlatMapUdf(Box::new(|| Box::new(face_utils::split_bbs_facenet))),
    );
    function_lookup.insert(
        "split_bbs_insightface".into(),
        FunctionKinds::FlatMapUdf(Box::new(|| Box::new(face_utils::split_bbs_insightface))),
    );

    let merge_info = Arc::new(std::sync::Mutex::<Vec<MergeInfo>>::new(Vec::new()));

    for (my_pipeline_id, callback_name) in [
        "merge_callback_buffalo_sc",
        "merge_callback_buffalo_l",
        "merge_callback_antelope_v2",
        "merge_callback_facenet",
    ]
    .into_iter()
    .enumerate()
    {
        let merge_info_clone = Arc::clone(&merge_info);
        let route_feedback_sender_clone = route_feedback_sender.clone();
        let merge_callback_fn: FunctionKinds =
            FunctionKinds::MergeCallbackUdf(Box::new(move || {
                let merge_info = Arc::clone(&merge_info_clone);
                let route_feedback_sender_clone = route_feedback_sender_clone.clone();
                let my_pipeline_id = my_pipeline_id + 1;
                Box::new(move |tuple| {
                    trace!(
                        "merge callback {callback_name:?} received tuple {}",
                        tuple.id()
                    );
                    if let Err(e) = route_feedback_sender_clone.send(vec![(
                        tuple.id() as _,
                        my_pipeline_id,
                        Instant::now(),
                    )]) {
                        error!("failed to send feedback to routing: {e}");
                    }
                })
            }));
        function_lookup.insert(callback_name.into(), merge_callback_fn);
    }
    debug!("g7: merge callback functions registered");

    function_lookup.insert(
        "get_first_id".into(),
        FunctionKinds::ComputationExpressionUdf(Box::new(move || {
            Box::new(|args| {
                let Some(match_ids) = args.get(0) else {
                    error!("get_first_id: no match_ids found");
                    return HabValue::Null;
                };
                let Some(match_ids) = match_ids.as_list() else {
                    error!("get_first_id: match_ids is not a list");
                    return HabValue::Null;
                };
                let Some(first_id) = match_ids.get(0) else {
                    error!("get_first_id: no first id found");
                    return HabValue::Null;
                };
                first_id.clone()
            })
        })),
    );

    let blur_known_faces: FunctionKinds = FunctionKinds::AggregationUdf(Box::new(move || {
        Box::new(|window| {
            face_utils::aggregate_blur_regions(window, face_utils::BlurCondition::BlurKnown)
        })
    }));
    function_lookup.insert("blur_known_faces".into(), blur_known_faces);

    let blur_unknown_faces: FunctionKinds = FunctionKinds::AggregationUdf(Box::new(move || {
        Box::new(|window| {
            face_utils::aggregate_blur_regions(window, face_utils::BlurCondition::BlurUnknown)
        })
    }));
    function_lookup.insert("blur_unknown_faces".into(), blur_unknown_faces);

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

            log_udf_items_received_callback.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if log_udf_items_received_callback.load(std::sync::atomic::Ordering::SeqCst)
                >= max_total_samples
            {
                warn!("log udf received the max items, notifying condition");
                let _ = max_item_condition.send(true);
            }
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
            let completed_items = log_udf_items_received.load(atomic::Ordering::SeqCst);
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

    info!("finished executing");
    info!("no op counter status: {no_op_counts:?}");
    Ok(())
}

const PERSON_NAME_FIELD: &str = "name";

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
                face_utils::ORIGINAL_IMAGE_SHAPE_FIELD.into(),
                // HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
                HabValue::ShapeBuffer(vec![width as usize, height as usize, 3]),
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
                current_item = current_item + 1;
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
) -> (
    impl 'a + Iterator<Item = (usize, bool, ImageInfo, (usize, usize, HabValue))>,
    impl FnMut((usize, bool, ImageInfo, (usize, usize, HabValue))) -> Tuple,
) {
    let path = std::path::PathBuf::from(base_folder);
    // load all images first and then later make the tuples
    // this is to avoid the tuples having very early creation times that count against them later
    let initial_iter = image_paths_to_iterator(image_paths, path);
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
    let converter = move |(_img_idx, is_in_index, image_info, (width, height, arr3_buf)): (
        usize,
        bool,
        ImageInfo,
        // image::RgbImage,
        (usize, usize, HabValue),
    )| {
        let mut tuple = get_tuple();
        let img_id_key = image_info.img_id.to_key();
        let image_id = IMAGE_ID_INT_MAP
            .entry(img_id_key.clone().into())
            .or_insert_with(|| NEXT_IMAGE_ID_INT.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
        let img_id_int = *image_id.value();

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
            // HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
            HabValue::ShapeBuffer(vec![width as usize, height as usize, 3]),
        );
        tuple.insert(
            "image_shape".into(),
            // HabValue::ShapeBuffer(vec![height as usize, width as usize, 3]),
            HabValue::ShapeBuffer(vec![width as usize, height as usize, 3]),
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
    mut path: std::path::PathBuf,
) -> impl 'a + Iterator<Item = (usize, bool, ImageInfo, (usize, usize, HabValue))> {
    Box::new(
        image_paths
            .enumerate()
            .flat_map(move |(img_idx, (is_in_index, image_info))| {
                // let mut path = path.clone();
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
                let (width, height) = img.dimensions();

                let buf = img.into_raw();
                debug!(
                    "image buffer for {:?} with width {width} and height {height} has length {}",
                    &image_info.img_path,
                    buf.len()
                );

                let buf_len = buf.len();
                let Ok(arr3) = watershed_shared::ws_types::ArcArrayD::from_shape_vec(
                    &[width as usize, height as usize, 3][..],
                    buf,
                ) else {
                    error!("failed to create ndarray from image buffer for image {:?} with width {width} and height {height} with buffer length {}",
                        &image_info.img_path,
                        buf_len
                    );
                    return None;
                };
                let arr3_buf = HabValue::SharedArrayU8(watershed_shared::SharedU8Array(arr3));
                Some((
                    img_idx,
                    is_in_index,
                    image_info,
                    (width as usize, height as usize, arr3_buf),
                ))
            }),
    )
}

fn aquifer_routing_fn(
    preclassifier: watershed_shared::preclassifier_lang::RealBucketLookup,
    keep_n_history_items: usize,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
    deadline_ms: u64,
    lookahead_ms: u64,
    strategy: scheduler::Strategy,
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
    let binning_fn = move |tuple: &Tuple| -> BinInfo<
        watershed_shared::preclassifier_lang::PreclassifierLangClass,
    > {
        // "features": [
        //     "boxes_detected",
        //     "width",
        //     "height"
        //   ],
        // let Some(boxes_detected) = tuple.get("boxes_detected") else {
        //     let available_keys = tuple.keys().collect::<Vec<_>>();
        //     let err = format!("boxes detected not found in tuple with id {} and img_id={:?}. avilable fields are {available_keys:?}",
        //             tuple.id(),
        //             tuple.get("img_id")
        //         );
        //     error!("{err}");
        //     panic!("{err}");
        // };
        // let Some(boxes_detected) = boxes_detected.as_integer() else {
        //     let err = format!(
        //         "boxes detected was not stored as an int in tuple with id {} and img_id={:?}",
        //         tuple.id(),
        //         tuple.get("img_id")
        //     );
        //     error!("{err}");
        //     panic!("{err}");
        // };
        // let original_width = match tuple.get("original_width") {
        //     Some(f) => f,
        //     None => {
        //         let available_keys = tuple.keys().collect::<Vec<_>>();
        //         error!("original width not found in tuple with id {} and img_id={:?}. avilable fields are {available_keys:?}",
        //             tuple.id(),
        //             tuple.get("img_id")
        //         );
        //         panic!("original width not found in tuple with id {} and img_id={:?}. avilable fields are {available_keys:?}",
        //             tuple.id(),
        //             tuple.get("img_id")
        //         );
        //     }
        // };
        // let original_width = match original_width.as_integer() {
        //     Some(f) => f,
        //     None => {
        //         error!(
        //             "original width was not stored as an int in tuple with id {} and img_id={:?}",
        //             tuple.id(),
        //             tuple.get("img_id")
        //         );
        //         panic!(
        //             "original width was not stored as an int in tuple with id {} and img_id={:?}",
        //             tuple.id(),
        //             tuple.get("img_id")
        //         );
        //     }
        // };
        // let original_height = match tuple.get("original_height") {
        //     Some(f) => f,
        //     None => {
        //         error!("original height not found");
        //         panic!("original height not found");
        //     }
        // };
        // let original_height = match original_height.as_integer() {
        //     Some(f) => f,
        //     None => {
        //         error!("original height was not stored as an int");
        //         panic!("original height was not stored as an int");
        //     }
        // };

        // let features = [
        //     boxes_detected as _,
        //     original_width as _,
        //     original_height as _,
        // ];

        // "args": [
        //     {
        //         "kind": "field",
        //         "name": "new_height_float"
        //     },
        //     {
        //         "kind": "field",
        //         "name": "new_width_float"
        //     },
        //     {
        //         "kind": "field",
        //         "name": "new_area_float"
        //     },
        //     {
        //         "kind": "field",
        //         "name": "new_hw_ratio_float"
        //     }
        // ]
        let Some(new_height_float) = tuple.get("new_height_float") else {
            error!(
                "new height float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new height float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_height_float) = new_height_float.as_float() else {
            error!(
                "new height float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new height float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_width_float) = tuple.get("new_width_float") else {
            error!(
                "new width float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new width float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_width_float) = new_width_float.as_float() else {
            error!(
                "new width float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new width float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_area_float) = tuple.get("new_area_float") else {
            error!(
                "new area float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new area float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_area_float) = new_area_float.as_float() else {
            error!(
                "new area float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new area float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_hw_ratio_float) = tuple.get("new_hw_ratio_float") else {
            error!(
                "new hw ratio float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new hw ratio float not found in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };
        let Some(new_hw_ratio_float) = new_hw_ratio_float.as_float() else {
            error!(
                "new hw ratio float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
            panic!(
                "new hw ratio float was not stored as a float in tuple with id {} and img_id={:?}",
                tuple.id(),
                tuple.get("img_id")
            );
        };

        let features = [
            new_height_float.0 as _,
            new_width_float.0 as _,
            new_area_float.0 as _,
            new_hw_ratio_float.0 as _,
        ];

        watershed_shared::preclassifier_lang::map_inputs_to_bucket(&features, &preclassifier)
    };
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

    move |mut tuples, senders| {
        debug!("received {:?} tuples in routing function", tuples.len());
        if !matches!(
            senders,
            [
                _drop_channel,
                _insightface_small,
                // _insightface_medium,
                // _insightface_large,
                _facenet
            ]
        ) {
            // error!("Expected channels were not present. Expected 5 channels [drop, insightface_small, insightface_medium, insightface_large, facenet], got {:?} channels", senders.len());
            error!("Expected channels were not present. Expected 3 channels [drop, insightface_small, facenet], got {:?} channels", senders.len());
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
                debug!("mean overage ratio is {mean_overage_ratio:.2}, reducing budget per item to {val:.2}ns");
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
            let mean_elapsed_increase_ms = history.mean_elapsed_increase_ms();
            // we will care more about how that's been going recently, so we will use the weighted version
            let weighted_mean_elapsed_increase_ms =
                history.recent_weighted_mean_elapsed_increase_ms();

            // what about the age of the items themselves? are they continuing to get too old?
            let mean_final_age_ms = history.mean_age_when_merging_increase_ms();
            let weighted_mean_final_age_increase_ms =
                history.recent_weighted_mean_age_when_merging_increase_ms();
            debug!(
                "Recent History consequences: mean elapsed increase: {mean_elapsed_increase_ms:.2} ms , recent-weighted mean elapsed increase: {weighted_mean_elapsed_increase_ms:.2} ms, mean final age increase: {mean_final_age_ms:.2} ms, recent-weighted mean final age increase: {weighted_mean_final_age_increase_ms:.2} ms",
            );
            // let penalty_metric_ms = weighted_mean_elapsed_increase_ms;
            let penalty_metric_ms = weighted_mean_final_age_increase_ms;
            // how much of a delay were we allowed to begin with?
            // if it delays by a little, but we still have plenty of time, then we won't give as big of a penalty
            let allowed_time_ms = deadline_ms as f64;
            let excess_ratio = penalty_metric_ms / allowed_time_ms;
            debug!(
            "penalty metric is {penalty_metric_ms:.2}ms, allowed time is {allowed_time_ms:.2}ms, excess ratio is {excess_ratio:.2}"
        );
            let history_penalized_lookahead_ms = if excess_ratio > 1.0 {
                error!(
                "we are more than 100% over budget ({excess_ratio:.2}), so we will not schedule any items"
            );
                0.0
            } else if excess_ratio > 0.1 {
                warn!("we are more than 10% over budget ({excess_ratio:.2}), so we will greatly reduce our budget per item");
                let penalized = match large_excess_punishment {
                    LargeExcessPunishment::Ignore => lookahead_ms,
                    LargeExcessPunishment::OverageRatio => lookahead_ms * (1.0 - excess_ratio),
                    LargeExcessPunishment::SqrtOverage => {
                        lookahead_ms * (1.0 - excess_ratio.sqrt())
                    }
                };
                debug!(
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
                    debug!(
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

            let max_allowed_age_ms = allowed_time_ms;
            let initial_budget_adjustment = mean_age_of_items / max_allowed_age_ms;
            let adjusted_budget_ms = lookahead_ms * (1.0 - initial_budget_adjustment);
            debug!(
                "mean age of items is {mean_age_of_items:.2}ms, max age is {max_allowed_age_ms:.2}ms, initial budget adjustment is {initial_budget_adjustment:.3}, adjusted budget is {adjusted_budget_ms:.2}ms",
            );
            // we can scale based on the earlier penalty
            const MAX_OVERAGE_RECOVERY_RATIO: f64 = 1.0 + 0.05; // 5% recovery
                                                                // we are allowed to lose a lot of the budget, but we can only recover a little bit in order to be conservative with how wrong we can be
            let budget_ms_v3 = adjusted_budget_ms
                * f64::clamp(
                    1.0 + (penalty_metric_ms / max_allowed_age_ms),
                    0.0,
                    MAX_OVERAGE_RECOVERY_RATIO,
                );
            // TODO: additional adjustments based on the history of the items and based on the drop rates/rejection rates from the channels
            debug!(
                "budget method v3: final budget after adjustments is {budget_ms:.2}ms after apploying the penalty metric {penalty_metric_ms:.2}ms/{max_allowed_age_ms:.2}ms",
            );

            trace!("closure g3");
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
            trace!("closure g4");
            let tuple_id = current_tuple.id();
            let start_scheduling_time = Instant::now();
            // use aquifer_scheduler
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
        // let mut max_individual_duration_micros: f64 = 0.0;
        // let mut sum_individual_durations_micros: f64 = 0.0;
        // let mut min_individual_duration_micros: f64 = f64::MAX;
        // for d in &individual_scheduling_durations {
        //     let d_micros = d.as_nanos() as f64 / 1_000.0;
        //     max_individual_duration_micros = max_individual_duration_micros.max(d_micros);
        //     sum_individual_durations_micros += d_micros;
        //     min_individual_duration_micros = min_individual_duration_micros.min(d_micros);
        // }
        // let mean_individual_duration_micros = if !individual_scheduling_durations.is_empty() {
        //     sum_individual_durations_micros / individual_scheduling_durations.len() as f64
        // } else {
        //     0.0
        // };
        // if min_individual_duration_micros == f64::MAX {
        //     min_individual_duration_micros = f64::NAN;
        // }
        // debug!(
        //     "overall routing duration for {tuples_to_route} items: {overall_route_duration_micros:.2} us, gross mean per item: {gross_mean_route_duration_micros:.2} us, individual scheduling durations: min: {min_individual_duration_micros:.2} us, mean: {mean_individual_duration_micros:.2} us, max: {max_individual_duration_micros:.2} us",
        // );
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
                // RoutingOptions::AlwaysTiny => 1,
                // RoutingOptions::AlwaysSmall => 2,
                // RoutingOptions::AlwaysBig => 3,
                // RoutingOptions::AlwaysHuge => 4,

                // new query options where we only use tiny or huge
                RoutingOptions::AlwaysTiny => 1,
                RoutingOptions::AlwaysSmall => {
                    error!("static routing option AlwaysSmall is not supported, use AlwaysHuge instead");
                    continue;
                }
                RoutingOptions::AlwaysBig => {
                    error!(
                        "static routing option AlwaysBig is not supported, use AlwaysHuge instead"
                    );
                    continue;
                }
                RoutingOptions::AlwaysHuge => 2,
                _ => {
                    error!("incorrect static routing option {routing_option:?}");
                    continue;
                }
            };
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
    move |tuples, senders| {
        let amount = tuples.len();
        let mut rng = rand::thread_rng();
        let mut available_routes = vec![
            // RoutingOptions::AlwaysTiny,
            // RoutingOptions::AlwaysSmall,
            // RoutingOptions::AlwaysBig,
            // RoutingOptions::AlwaysHuge,
            RoutingOptions::AlwaysTiny,
            RoutingOptions::AlwaysHuge,
        ];
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
                        // RoutingOptions::AlwaysTiny => 1,
                        // RoutingOptions::AlwaysSmall => 2,
                        // RoutingOptions::AlwaysBig => 3,
                        // RoutingOptions::AlwaysHuge => 4,

                        // new query options where we only use tiny or huge
                        RoutingOptions::AlwaysTiny => 1,
                        RoutingOptions::AlwaysSmall => {
                            error!("eddies routing option AlwaysSmall is not supported, use AlwaysHuge instead");
                            break 'next_tuple;
                        }
                        RoutingOptions::AlwaysBig => {
                            error!("eddies routing option AlwaysBig is not supported, use AlwaysHuge instead");
                            break 'next_tuple;
                        }
                        RoutingOptions::AlwaysHuge => 2,
                        _ => {
                            error!("incorrect eddies routing option {routing_option:?}");
                            break 'next_tuple;
                        }
                    };
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
            available_routes.extend([
                // RoutingOptions::AlwaysTiny,
                // RoutingOptions::AlwaysSmall,
                // RoutingOptions::AlwaysBig,
                // RoutingOptions::AlwaysHuge,
                RoutingOptions::AlwaysTiny,
                RoutingOptions::AlwaysHuge,
            ]);
        }
        Some(amount)
    }
}

fn routing_fn_eddies(
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    const BASELINE_MAX_PIPELINES: usize = 4;
    move |tuples, senders| {
        let amount = tuples.len();
        let mut rng = rand::thread_rng();
        let mut available_routes: smallvec::SmallVec<[_; BASELINE_MAX_PIPELINES]> = [
            // RoutingOptions::AlwaysTiny,
            // RoutingOptions::AlwaysSmall,
            // RoutingOptions::AlwaysBig,
            // RoutingOptions::AlwaysHuge,
            RoutingOptions::AlwaysTiny,
            RoutingOptions::AlwaysHuge,
        ]
        .into_iter()
        .zip(&senders[1..])
        .zip(1..usize::MAX)
        .map(|((opt, pipe), i)| (i, opt, pipe.len(), pipe.cap(), pipe))
        .filter(|(_i, _o, l, c, _p)| *c > *l)
        .collect();
        let mut outputs: watershed_shared::ws_types::ArrayMap<
            usize,
            (Vec<Tuple>, &AsyncPipe),
            BASELINE_MAX_PIPELINES,
        > = Default::default();
        'next_tuple: for t in tuples {
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
