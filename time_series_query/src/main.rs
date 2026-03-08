use anyhow::Context;
// use futures_batch::ChunksTimeoutStreamExt;
use log::trace;
#[allow(unused_imports)]
use log::{debug, error, info, warn};

use ndarray::s;
use ndarray::ArrayView2;
use polars::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::Fft;
use rustfft::FftPlanner;
use serde::Serialize;
use tokio::signal;
use tokio::sync::watch;
use tokio_stream::StreamExt;
use watershed_shared::async_query_builder::RuntimeState;
use watershed_shared::basic_pooling::get_tuple;
use watershed_shared::basic_pooling::get_tuple_vec;
use watershed_shared::caching::StrToKey;
use watershed_shared::global_logger::LimitedHabValue;
use watershed_shared::global_logger::NO_AUX_DATA;
use watershed_shared::scheduler;
use watershed_shared::scheduler::basic_probability_forecast::PastData;
use watershed_shared::scheduler::{
    basic_probability_forecast::{BasicCategory, History},
    AlgInputs, BinInfo, FutureWindowKind, ShareableArray,
};
use watershed_shared::AsyncPipe;
use watershed_shared::Operator;
use watershed_shared::UdfBolt;

use dashmap::DashMap;
use serde::Deserialize;
use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap};
use std::env::args;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic, Arc};
use std::time::{Duration, Instant};

use watershed_shared::query_builder::{self, QueryDescriptor};
use watershed_shared::{HabString, HabValue, Tuple};

use watershed_shared::devec::DeVec as Queue;
use watershed_shared::operators::AggregationResult;

#[derive(Debug, Deserialize)]
struct TimeSeriesExperimentConfig {
    model_info_path: String,
    sequence_info_path: String,
    label_info_path: String,
    patient_data_path: String,
    query_path: String,
    max_samples_per_sequence: Option<usize>,
    max_total_samples: Option<usize>,
    history_window_size: Option<usize>,
    greedy_lookahead_window_size: Option<usize>,
    optimal_lookahead_window_size: Option<usize>,
    lookahead_time_ms: Option<u64>,
    deadline_window_ms: Option<u64>,
    target_time_micros: Option<Delay>,
    input_delay_micros: Option<Delay>,
    overall_time_limit_ms: Option<u64>,
    sequence_read_batch_size: Option<usize>,
    initial_startup_delay_ms: Option<u64>,
    sequence_count: Option<usize>,
    routing_strategy: Option<RoutingOptions>,
    per_stream_read_delay: Option<DelayType>,
    log_folder: Option<HabString>,
    //  TODO: add the file name and rate to flush the data for the experiment info for time of creation, time of merging, etc
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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
#[derive(Serialize, Deserialize, Debug, Clone)]
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

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type")]
enum DelayType {
    Fixed { value: u64 },
    Random { value: u64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
enum RoutingOptions {
    #[default]
    #[serde(rename = "big")]
    AlwaysBig,
    #[serde(rename = "small")]
    AlwaysSmall,
    #[serde(rename = "random")]
    Random,
    #[serde(rename = "eddies")]
    Eddies,
    #[serde(rename = "aquifer_greedy")]
    AquiferGreedy,
    #[serde(rename = "aquifer_optimal")]
    AquiferOptimal,
}

#[derive(Serialize, Deserialize)]
struct StartingInfo {
    tuple_id: usize,
    person_id: usize,
    sequence_id: u128,
    label: usize,
    time_created: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct MergeInfo {
    tuple_id: usize,
    person_id: usize,
    sequence_id: u128,
    label: usize,
    pipeline_id: usize,
    time_merged: u128,
}

// TODO: log sequence data
#[derive(Debug, Serialize, Deserialize)]
struct SequenceDetectedInfo {
    tuple_ids: Vec<u64>,
    person_id: usize,
    label: usize,
}

fn main() -> anyhow::Result<()> {
    async_main().inspect_err(|e| {
        error!("Async Main error: {:?}", e);
        // time to write before exiting
        std::thread::sleep(Duration::from_millis(1500));
    })
}

const UNIQUE_PERSON_COUNT: usize = 50;
const SEQUENCE_LEN: usize = 100;
const SAMPLE_LEN: usize = 2_000;
const FFT_FEATURE_FIELD: &'static str = "fft_features_output";

const USE_COMPUTED: bool = false;

fn async_main() -> anyhow::Result<()> {
    use watershed_shared::async_query_builder::FunctionKinds;
    use watershed_shared::async_query_builder::PhysicalOperator;
    // print env vars
    debug!("Printing environment variables:");
    for (key, value) in std::env::vars() {
        debug!("{}: {}", key, value);
    }
    debug!("End of env vars");

    let per_sequence_default_delay = 100;
    let mut args = args();
    let _this_file = args.next().context("no file name")?;
    let config_path = args.next().context("no config path provided")?;
    let config = std::fs::read_to_string(config_path)?;

    // use the log4rs file
    let log_path = args.next().context("no logger config path provided")?;
    log4rs::init_file(log_path, Default::default()).context("failed to initialize log4rs")?;

    let TimeSeriesExperimentConfig {
        model_info_path,
        sequence_info_path,
        patient_data_path,
        label_info_path,
        query_path,
        max_samples_per_sequence,
        max_total_samples,
        history_window_size,
        greedy_lookahead_window_size,
        optimal_lookahead_window_size,
        lookahead_time_ms,
        deadline_window_ms,
        target_time_micros,
        input_delay_micros,
        overall_time_limit_ms,
        sequence_read_batch_size,
        initial_startup_delay_ms: initial_startup_delay,
        sequence_count,
        routing_strategy,
        per_stream_read_delay: read_delay,
        log_folder,
    } = serde_json::from_str(&config)?;

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
    let target_time_micros =
        target_time_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let max_target_time_ms = target_time_micros.max() / 1_000;
    if let (Some(max_items), Some(sequence_count)) = (max_samples_per_sequence, sequence_count) {
        let max_items = max_items * sequence_count;
        if max_items as u64 * max_target_time_ms
            > overall_time_limit_ms.unwrap_or(usize::MAX as u64)
        {
            warn!(
                "max items * target time exceeds overall time limit and is not expected to finish"
            );
        }
    }
    let input_delay_micros =
        input_delay_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let initial_startup_delay: u64 = initial_startup_delay.unwrap_or(10_000);

    // let model_info_path = "./model_info.json";
    let model_info = std::fs::read_to_string(&model_info_path)?;
    let model_info = watershed_shared::preclassifier_lang::load_file_format(model_info.as_bytes())?;

    use ndarray::{Array2, Array5};

    let sequences: Array5<f32> = ndarray_npy::read_npy(sequence_info_path)?;
    info!("sequences shape: {:?}", sequences.shape());

    let labels: Array2<i64> = ndarray_npy::read_npy(label_info_path)?;
    info!("labels shape: {:?}", labels.shape());

    info!(
        "records found: {:?}",
        read_patient_data_from_csv(&patient_data_path)?.count()
    );

    // let query = std::fs::read_to_string(r"./test_queries/time_series_query.json")?;
    let query = std::fs::read_to_string(&query_path)?;
    let mut function_lookup = BTreeMap::<HabString, FunctionKinds>::new();

    let sequence_read_batch_size = sequence_read_batch_size.unwrap_or(64);
    let shared_sequences = Arc::new(sequences);
    debug!("shared sequences shape: {:?}", shared_sequences.shape());
    let shared_labels = Arc::new(labels);
    assert_eq!(shared_sequences.shape()[0], shared_labels.shape()[0]);
    assert_eq!(shared_sequences.shape()[1], shared_labels.shape()[1]);
    assert_eq!(shared_sequences.shape()[1], SEQUENCE_LEN);
    let max_samples_per_sequence = max_samples_per_sequence.unwrap_or(SAMPLE_LEN);

    // for each sequence, we will start a new task that will run in the background, pushing to a channel for the input to the whole system
    let sequence_count = sequence_count.unwrap_or(shared_sequences.shape()[0]);

    // set stack size to 32 MB
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(32 * 1024 * 1024)
        .enable_all()
        .build()?;
    let (max_item_condition, stop_rx) = watch::channel(false);

    let (tx, rx) = tokio::sync::broadcast::channel::<(usize, u128, usize, Array2<f32>)>(
        sequence_count * SEQUENCE_LEN,
    );
    let sequence_completion_counter = Arc::new(AtomicUsize::new(0));
    let all_items_produced_counter = Arc::new(AtomicUsize::new(0));
    let mut spawner_futures = Vec::new();
    let num_input_sequences = UNIQUE_PERSON_COUNT.min(sequence_count);
    spawner_futures.push({
        let max_item_condition = max_item_condition.clone();
        let sequence_completion_counter = Arc::clone(&sequence_completion_counter);
        let shared_sequences = Arc::clone(&shared_sequences);
        let shared_labels = Arc::clone(&shared_labels);
        let mut stop_rx = stop_rx.clone();
        let mut input_delay_micros = input_delay_micros.clone();
        async move {
            let initial_read_delay = match read_delay {
                Some(DelayType::Fixed { value: t }) => t,
                Some(DelayType::Random { value: t }) => rand::random::<u64>() % t,
                None => 0,
            };
            tokio::time::sleep(Duration::from_millis(
                initial_read_delay + initial_startup_delay,
            )).await;

            let mut current_delay_streak = 0;
            let max_streak = input_delay_micros.max_streak();
            let mut current_input_delay_micros = input_delay_micros.starting_delay();
            'produce_loop: for sequence_id in 0..num_input_sequences {
                debug!("reading sequence {sequence_id}");
                let person_id = sequence_id % UNIQUE_PERSON_COUNT;
                for input_index in 0..SEQUENCE_LEN{
                    if all_items_produced_counter.load(std::sync::atomic::Ordering::SeqCst) >= max_total_samples.unwrap_or(usize::MAX) {
                        debug!("reached max total samples, notifying condition");
                        let _ = max_item_condition.send(true);
                        break 'produce_loop;
                    }
                    if input_index >= max_samples_per_sequence{
                        break; // move onto next sequence
                    }
                    // debug!("reading sequence {sequence_id}, index {input_index}");
                    // TODO: implement pooling for the sequences so they aren't duplicating all this memory
                    let sequence = shared_sequences.slice(s![sequence_id, input_index, 0, .., ..]);
                    let sequence: Array2<f32> = sequence.to_owned();
                    let label = shared_labels[[sequence_id, input_index]];
                    let sequence_id: u128 = (person_id as u128) << 64 | input_index as u128;
                    if let Err(e) = tx
                        .send((person_id, sequence_id, label as usize, sequence)) {
                            error!(
                                "failed to send data for sequence {} sample {}: {e}",
                                sequence_id, input_index
                            );
                            continue;
                    }
                    all_items_produced_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    match tokio::time::timeout(Duration::from_micros(
                        current_input_delay_micros,
                    ), stop_rx.changed()).await {
                        Ok(Ok(_)) => {
                            debug!("received stop signal, stopping production");
                            break 'produce_loop;
                        }
                        Ok(Err(e)) => {
                            error!("failed to read from stop signal: {e}");
                            break 'produce_loop;
                        }
                        Err(_elapsed) => {
                            trace!("timed out waiting for stop signal, continuing production");
                            current_delay_streak += 1;
                            if current_delay_streak >= max_streak {
                                current_input_delay_micros = input_delay_micros.next_delay();
                                current_delay_streak = 0;
                            }
                            continue;
                        }
                    }
                }
                debug!("finished sending data for sequence {}", sequence_id);
            }
            sequence_completion_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let outstanding_items = tx.len();
            let max_duration = Duration::from_millis((max_target_time_ms.max(1) * outstanding_items as u64).min(overall_time_limit_ms.unwrap_or(usize::MAX as u64)));
            debug!("finished sending data for, but there are still {} items in the channel. waiting for {}ms or until done", outstanding_items, max_duration.as_millis());
            let wait_on_change = stop_rx.changed();
            if let Err(_e) = tokio::time::timeout(max_duration, wait_on_change).await {
                // timed out
                warn!("timed out waiting for condition to be met. max duration was reached when done sending data");
            }
            debug!("finished sending data, waiting for the target time before exiting");
            tokio::time::sleep(Duration::from_millis(max_target_time_ms)).await;
        }
    });

    let expected_sequence_completions = spawner_futures.len();
    // now that all of those are spawned, we can make a source udf that reads from the rx side
    let max_amt_to_read = max_total_samples.unwrap_or(usize::MAX);
    let rx = std::sync::Mutex::new(Some(rx));
    let all_items_read_counter = Arc::new(AtomicUsize::new(0));
    let all_items_read_counter_source = Arc::clone(&all_items_read_counter);

    debug!("max amt to read: {}", max_amt_to_read);
    debug!("sequence read batch size: {}", sequence_read_batch_size);

    let starting_info = Arc::new(std::sync::Mutex::<Vec<StartingInfo>>::new(
        // pre-allocate space, but not too much if not necessary
        Vec::with_capacity(
            max_amt_to_read
                .min(max_samples_per_sequence * sequence_count)
                .min(1_000),
        ),
    ));
    let data_source_starting_info = Arc::clone(&starting_info);
    let data_source_udf = FunctionKinds::SourceUdf(Box::new(move || {
        let rx = rx
            .lock()
            .unwrap()
            .take()
            .expect("source using this rx channel can only be initialized once");
        let rx = tokio_stream::wrappers::BroadcastStream::new(rx);
        let counter = Arc::clone(&all_items_read_counter_source);
        let starting_info = Arc::clone(&data_source_starting_info);
        Box::new(move || {
            let starting_info = Arc::clone(&starting_info);
            Box::new(
                futures::StreamExt::take(rx, max_amt_to_read)
                    .filter_map(|v| match v {
                        Ok((person_id, sequence_id, label, sequence)) => {
                            Some((person_id, sequence_id, label, sequence))
                        }
                        Err(e) => {
                            error!("failed to read from background: {e}");
                            None
                        }
                    })
                    // .chunks(sequence_read_batch_size)
                    .chunks_timeout(
                        sequence_read_batch_size,
                        Duration::from_millis(max_target_time_ms.max(1) + 1),
                    )
                    // .filter(|v| std::future::ready(v.len() > 0))
                    .map(move |v| {
                        let mut tuples = get_tuple_vec();
                        for (person_id, sequence_id, label, sequence) in v {
                            let mut tuple = get_tuple();
                            let tuple_id = tuple.id();
                            tuple.insert("label".to_key(), HabValue::Integer(label as i32));
                            use bytemuck::cast_vec;
                            let dims = sequence.shape().to_vec();
                            let (sequence_backing, _offset) = sequence.into_raw_vec_and_offset();
                            let sequence_backing = cast_vec(sequence_backing);
                            tuple
                                .insert("patient_id".to_key(), HabValue::Integer(person_id as i32));
                            tuple.insert(
                                "sequence_id".to_key(),
                                HabValue::UnsignedLongLong(sequence_id),
                            );
                            tuple.insert(
                                SEQUENCE_FIELD_NAME.to_key(),
                                HabValue::IntBuffer(sequence_backing),
                            );
                            tuple.insert(SHAPE_FIELD_NAME.to_key(), HabValue::ShapeBuffer(dims));

                            let time_created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .expect("time went backwards")
                                .as_millis();
                            tuple.insert(
                                "time_created".to_key(),
                                HabValue::UnsignedLongLong(time_created),
                            );

                            // global log that we created the tuple
                            let log_location = "data_source_tuple_creation".to_raw_key();
                            let aux_data = NO_AUX_DATA;
                            if let Err(e) = watershed_shared::global_logger::log_data(
                                tuple_id,
                                log_location,
                                aux_data,
                            ) {
                                for err in e {
                                    error!("failed to log tuple creation: {err}");
                                }
                            }

                            let info = StartingInfo {
                                tuple_id,
                                person_id,
                                sequence_id,
                                label,
                                time_created: time_created,
                            };
                            starting_info.lock().unwrap().push(info);

                            tuples.push(tuple);
                        }
                        counter.fetch_add(tuples.len(), std::sync::atomic::Ordering::SeqCst);
                        trace!("read {} tuples from background", tuples.len());
                        tuples
                    }),
            )
        })
    }));

    function_lookup.insert("sequence_inputs".into(), data_source_udf);

    let encode_sequence_fn: FunctionKinds = FunctionKinds::EncodeRemotePythonUdf(Box::new(|| {
        Box::new(time_series_python_udfs::encode_sequence)
    }));
    function_lookup.insert("encode_sequence".into(), encode_sequence_fn);

    let decode_fft_features_fn: FunctionKinds =
        FunctionKinds::DecodeRemotePythonUdf(Box::new(|| {
            Box::new(time_series_python_udfs::decode_fft_features)
        }));
    function_lookup.insert("decode_fft_features".into(), decode_fft_features_fn);

    let fft_features_fn: FunctionKinds = FunctionKinds::ComputationExpressionUdf(Box::new({
        || {
            let f = make_fft_analysis();
            let f = std::sync::Mutex::new(f);
            Box::new(move |v| {
                let mut f = f.lock().unwrap();
                f(v)
            })
        }
    }));
    function_lookup.insert("fft_features".into(), fft_features_fn);

    let shutdown_sequence_fn: FunctionKinds =
        FunctionKinds::ShutdownRemotePythonUdf(Box::new(|| {
            Box::new(time_series_python_udfs::shutdown_sequence)
        }));
    function_lookup.insert("shutdown_sequence".into(), shutdown_sequence_fn);

    let (route_feedback_sender, route_feedback_receiver) = crossbeam::channel::unbounded();
    let routing_udf: FunctionKinds =
        FunctionKinds::RoutingUdf(Box::new(move || match routing_strategy {
            Some(RoutingOptions::AlwaysBig) => Box::new(routing_fn_static(
                false,
                window_size,
                deadline_window_ms,
                route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::AlwaysSmall) => Box::new(routing_fn_static(
                true,
                window_size,
                deadline_window_ms,
                route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::Random) => {
                unimplemented!("random routing not used for these experiments")
            }
            Some(RoutingOptions::Eddies) => Box::new(routing_fn_eddies(
                window_size,
                deadline_window_ms,
                route_feedback_receiver.clone(),
            )),
            Some(RoutingOptions::AquiferGreedy) => Box::new(aquifer_routing_fn(
                model_info.clone(),
                window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Greedy,
            )),
            Some(RoutingOptions::AquiferOptimal) => Box::new(aquifer_routing_fn(
                model_info.clone(),
                window_size,
                route_feedback_receiver.clone(),
                deadline_window_ms,
                lookahead_time_ms,
                scheduler::Strategy::Optimal,
            )),
            None => Box::new(routing_fn_static(
                false,
                window_size,
                deadline_window_ms,
                route_feedback_receiver.clone(),
            )),
        }));

    function_lookup.insert("routing_fn".into(), routing_udf);

    let decode_label_fn: FunctionKinds = FunctionKinds::DecodeRemotePythonUdf(Box::new(|| {
        Box::new(time_series_python_udfs::decode_label)
    }));
    function_lookup.insert("decode_label".into(), decode_label_fn);

    // let buffer_len = 10;
    let buffer_len = 3;
    let minimum_emit_streak: usize = 3;
    let label_field = "label";
    let group_by_items_received = Arc::new(AtomicUsize::new(0));
    let group_by_items_received_clone = Arc::clone(&group_by_items_received);
    let diagnosis_streaks_fn: FunctionKinds = FunctionKinds::AggregationUdf(Box::new(move || {
        let group_by_items_received = Arc::clone(&group_by_items_received_clone);
        Box::new(diagnosis_streaks(
            group_by_items_received,
            buffer_len,
            minimum_emit_streak,
            label_field.into(),
        ))
    }));
    function_lookup.insert("diagnosis_streak".into(), diagnosis_streaks_fn);

    let merge_info = Arc::new(std::sync::Mutex::<Vec<MergeInfo>>::new(Vec::new()));
    let merge_info_clone = Arc::clone(&merge_info);

    let route_feedback_sender_clone = route_feedback_sender.clone();
    let merge_callback_left_fn: FunctionKinds =
        FunctionKinds::MergeCallbackUdf(Box::new(move || {
            let merge_info = Arc::clone(&merge_info_clone);
            let route_feedback_sender_clone = route_feedback_sender_clone.clone();
            let my_pipeline_id = 1;
            Box::new(move |tuple| {
                trace!("merge callback left received tuple {}", tuple.id());
                if let Err(e) = route_feedback_sender_clone.send(vec![(
                    tuple.id() as _,
                    my_pipeline_id,
                    Instant::now(),
                )]) {
                    error!("failed to send feedback to routing: {e}");
                }
            })
        }));
    function_lookup.insert("merge_callback_left".into(), merge_callback_left_fn);

    let merge_info_clone = Arc::clone(&merge_info);
    let merge_callback_right_fn: FunctionKinds =
        FunctionKinds::MergeCallbackUdf(Box::new(move || {
            let merge_info = Arc::clone(&merge_info_clone);
            let my_pipeline_id = 2;
            let route_feedback_sender_clone = route_feedback_sender.clone();
            Box::new(move |tuple| {
                trace!("merge callback right received tuple {}", tuple.id());
                if let Err(e) = route_feedback_sender_clone.send(vec![(
                    tuple.id() as _,
                    my_pipeline_id,
                    Instant::now(),
                )]) {
                    error!("failed to send feedback to routing: {e}");
                }
            })
        }));
    function_lookup.insert("merge_callback_right".into(), merge_callback_right_fn);

    let person_info_source_fn = FunctionKinds::SourceUdf(Box::new(move || {
        let iter = read_patient_data_from_csv(&patient_data_path)
            .expect("failed to read patient data from file");
        Box::new(move || {
            use futures::stream::StreamExt;
            Box::new(futures::stream::iter(iter).chunks(1))
        })
    }));
    function_lookup.insert("person_info_source".into(), person_info_source_fn);

    // let always_true_join = FunctionKinds::ComputationExpressionUdf(Box::new(|| Box::new(|_data| HabValue::Bool(true))));
    let always_true_join = FunctionKinds::JoinFilterUdf(|_t1, _t2| (true));
    function_lookup.insert("always_true_join".into(), always_true_join);

    let background_rt = rt.handle().clone();

    let RuntimeState {
        operators,
        runtime: _runtime,
        output_channels: _output_channels,
        stop_trigger,
    } = watershed_shared::async_query_builder::json_descriptor_to_operators_with_runtime(
        &query,
        &function_lookup,
        Some(rt),
        // max age is in nanos
        Some((deadline_window_ms * 1_000_000) as u128),
    )
    .expect("unable to parse query");

    let mut query_descriptor =
        serde_json::from_str::<QueryDescriptor>(&query).expect("unable to parse query");
    query_descriptor.operators.sort_by_key(|op| op.id);
    let topology = query_builder::get_topology_simple(&query_descriptor.operators, 4);

    let mut early_poll_ids = std::collections::BTreeSet::new();
    for oid in 0..operators.len() {
        // let op_len = operators.len();
        let PhysicalOperator::UserDefinedSource(source) = &operators[oid] else {
            continue;
        };
        let parent_id = source.parent.expect("no parent found for build side");
        let PhysicalOperator::Join(watershed_shared::Join { right, .. }) = &operators[parent_id]
        else {
            continue;
        };
        if *right == oid {
            early_poll_ids.insert(oid);
        }
    }
    info!("early poll ids {early_poll_ids:?}");

    let log_udf_items_received = Arc::new(AtomicUsize::new(0));
    let log_udf_items_received_callback = Arc::clone(&log_udf_items_received);
    let all_items_read_logger = Arc::clone(&all_items_read_counter);
    // continue as long as we haven't received the max amt and as long as the sequences aren't done

    let condition_bg_cutoff = Clone::clone(&max_item_condition);

    let log_udf: UdfBolt = UdfBolt {
        id: operators.len(),
        child: operators.last().unwrap().get_id(),
        parent: None,
        process: Arc::new(move |tuple| {
            let tuple_id = tuple.id();
            let patient_id = tuple.get("patient_id").unwrap().as_integer().unwrap();
            let label = tuple.get("label").unwrap().as_integer().unwrap();
            info!(
                "received tuple {} for patient {} with label {} in log udf bolt",
                tuple_id, patient_id, label
            );
            trace!("log udf fields: {:?}", tuple.keys().collect::<Vec<_>>());
            trace!(
                "log udf streak length {:?}",
                tuple.get("streak_length").unwrap().as_integer().unwrap()
            );
            log_udf_items_received_callback.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if log_udf_items_received_callback.load(std::sync::atomic::Ordering::SeqCst)
                >= max_amt_to_read
            {
                warn!("log udf received the max items, notifying condition");
                let _ = max_item_condition.send(true);
            }
            vec![tuple]
        }),
    };

    let sequence_completion_counter_cutoff = Arc::clone(&sequence_completion_counter);
    for fut in spawner_futures {
        background_rt.spawn(fut);
    }
    let _background_cutoff = background_rt.spawn(async move {
        let delay_amount = match read_delay {
            Some(DelayType::Fixed { value } | DelayType::Random { value }) => value,
            None => 10_000
        };
        tokio::time::sleep(Duration::from_millis(delay_amount)).await;
        let condition_start_time = Instant::now();
        loop{
            let completed_sequence_count = sequence_completion_counter_cutoff.load(atomic::Ordering::SeqCst);
            if completed_sequence_count >= expected_sequence_completions {
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
    let metrics = _runtime.as_ref().unwrap().metrics();
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

    let overall_time_limit_ms = overall_time_limit_ms.unwrap_or(1_000 * 60 * 30);
    watershed_shared::async_query_builder::runner_internal(
        operators,
        topology,
        early_poll_ids,
        last_op,
        Some(log_udf),
        Duration::from_millis(overall_time_limit_ms),
        _runtime.unwrap(),
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
    let group_by_items_received = group_by_items_received.load(std::sync::atomic::Ordering::SeqCst);
    info!("group by items received: {:?}", group_by_items_received);

    info!(
        "expected sequence completions: {:?}",
        expected_sequence_completions
    );
    let sequences_completed = sequence_completion_counter.load(atomic::Ordering::SeqCst);
    info!("sequences completed: {:?}", sequences_completed);

    info!(
        "items read: {:?}",
        all_items_read_counter.load(atomic::Ordering::SeqCst)
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
    Ok(())
}

fn record_merged_data(
    tuple: &watershed_shared::BetterTuple,
    merge_info: &mut Vec<MergeInfo>,
    my_pipeline_id: usize,
) {
    let tuple_id = tuple.id();
    let person_id = tuple.get("patient_id").unwrap().as_integer().unwrap() as usize;
    let sequence_id = tuple
        .get("sequence_id")
        .unwrap()
        .as_unsigned_long_long()
        .unwrap();
    let label = tuple.get("label").unwrap().as_integer().unwrap() as usize;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time went backwards")
        .as_millis();
    merge_info.push(MergeInfo {
        tuple_id,
        person_id,
        sequence_id,
        label,
        pipeline_id: my_pipeline_id,
        time_merged: now,
    });
}

const SEQUENCE_FIELD_NAME: &str = "sequence";
const SHAPE_FIELD_NAME: &str = "sequence_dims";
pub(crate) mod time_series_python_udfs {
    use super::*;
    use serde::{Deserialize, Serialize};
    #[derive(Debug, Serialize, Deserialize)]
    struct TensorF32Message<'a> {
        tuple_id: u64,
        // dims: Vec<u64>,
        dims: Vec<u32>,
        #[serde(with = "serde_bytes")]
        tensor: &'a [u8],
    }
    pub(crate) fn encode_sequence(tuple_id: usize, tuple: &Tuple) -> zeromq::ZmqMessage {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let diff = now - tuple.unix_time_created_ns();
        trace!("encoding sequence for tuple {tuple_id} with time difference {diff} ns");
        let byte_buffer = tuple
            .get(SEQUENCE_FIELD_NAME)
            .expect("sequence field not found")
            .as_int_buffer()
            .expect("sequence field not a byte buffer");
        let byte_buffer = bytemuck::cast_slice(byte_buffer);
        let shape = tuple
            .get(SHAPE_FIELD_NAME)
            .expect("sequence field not found")
            .as_shape_buffer()
            .expect("sequence field not a shape buffer");
        let tensor_message = TensorF32Message {
            tuple_id: tuple_id as u64,
            dims: shape.iter().map(|x| *x as u32).collect(),
            tensor: byte_buffer,
        };
        zeromq::ZmqMessage::from(
            rmp_serde::to_vec(&tensor_message).expect("failed to serialize tensor message"),
        )
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct FftFeatures {
        tuple_id: usize,
        tensor: Vec<f64>, // python gives data back as f64
    }
    use bytemuck::allocation::cast_vec;
    pub(crate) fn decode_fft_features(
        msg: zeromq::ZmqMessage,
        tuple_map: &DashMap<usize, Tuple>,
    ) -> Vec<Tuple> {
        let msg = match rmp_serde::from_read::<_, FftFeatures>(watershed_shared::FrameReader::new(
            msg.into_vec(),
        )) {
            Ok(v) => v,
            Err(e) => {
                error!("failed to deserialize fft features: {e}");
                return Vec::new();
            }
        };
        let tuple_id = msg.tuple_id;
        let (_, mut tuple) = tuple_map.remove(&tuple_id).expect("tuple not found");
        tuple.insert(
            FFT_FEATURE_FIELD.into(),
            HabValue::IntBuffer(cast_vec(msg.tensor.into_iter().map(|x| x as f32).collect())),
        );

        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(now) => {
                let diff = now.as_nanos() - tuple.unix_time_created_ns();
                trace!("decoded fft features for tuple {tuple_id} with time difference {diff} ns");
            }
            Err(e) => {
                error!("time went backwards when decoding fft features for tuple {tuple_id}: {e}");
                return Vec::new();
            }
        }
        vec![tuple]
    }

    pub(crate) fn shutdown_sequence() -> zeromq::ZmqMessage {
        zeromq::ZmqMessage::from(
            rmp_serde::to_vec(&TensorF32Message {
                tuple_id: u64::MAX,
                dims: Default::default(),
                tensor: Default::default(),
            })
            .expect("failed to serialize shutdown message"),
        )
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct LabelMessage {
        tuple_id: usize,
        label: f32,
    }

    // const ADD_DELAY: bool = false;
    // const ADD_DELAY: bool = true;
    pub(crate) fn decode_label(
        msg: zeromq::ZmqMessage,
        tuple_map: &DashMap<usize, Tuple>,
    ) -> Vec<Tuple> {
        // if ADD_DELAY {
        //     std::thread::sleep(std::time::Duration::from_millis(60));
        // }
        let msg = rmp_serde::from_read::<_, LabelMessage>(watershed_shared::FrameReader::new(
            msg.into_vec(),
        ))
        .expect("failed to deserialize label");
        let (_tuple_id, mut tuple) = tuple_map.remove(&msg.tuple_id).expect("tuple not found");
        tuple.insert("label".into(), HabValue::Integer(msg.label as _));
        info!(
            "decoded label: {:?} for tuple {:?}",
            msg.label, msg.tuple_id
        );
        'log_decode_label: {
            use watershed_shared::global_logger;
            let log_location = "decode_label".to_raw_key();
            let aux_data = [(
                "label".to_raw_key(),
                global_logger::LimitedHabValue::UnsignedInteger(msg.label as _),
            )];
            let aux_data = Some(aux_data);
            if let Err(e) = global_logger::log_data(msg.tuple_id, log_location, aux_data) {
                for err in e {
                    error!("failed to log decode label: {err}");
                }
                break 'log_decode_label;
            }
        }
        vec![tuple]
    }
}

// aggregation udf for time series
// takes a given window and determines the longest streak of a each label, then gives the longest streak of the label with the longest streak
fn diagnosis_streaks(
    group_by_items_received: Arc<atomic::AtomicUsize>,
    buffer_len: usize,
    minimum_emit_streak: usize,
    label_field: HabString,
) -> impl 'static + Send + Sync + Fn(&mut Queue<Tuple>) -> AggregationResult {
    move |queue: &mut Queue<Tuple>| {
        if let Some(t) = queue.last() {
            let tuple_id = t.id();
            let time_created_ns = t.unix_time_created_ns();
            let Ok(current_time_ns) = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|v| v.as_nanos())
            else {
                error!("time went backwards for tuple {}", tuple_id);
                return AggregationResult {
                    emit: None,
                    is_finished: false,
                };
            };
            let Some(diff) = current_time_ns.checked_sub(time_created_ns) else {
                error!("time went backwards for tuple {}", tuple_id);
                return AggregationResult {
                    emit: None,
                    is_finished: false,
                };
            };
            debug!("received tuple {tuple_id} in diagnosis streaks with time difference {diff} ns");
        }
        group_by_items_received.fetch_add(1, atomic::Ordering::SeqCst);

        let patient_id = queue[0].get("patient_id").unwrap().as_integer().unwrap();
        if queue.len() < buffer_len {
            return AggregationResult {
                emit: None,
                is_finished: false,
            };
        }
        let mut tuples = Vec::new();
        let mut last_label = None;
        let mut current_streak = 0;
        let mut total_streaks: HashMap<i32, usize> = HashMap::new();
        for t in queue.iter().rev() {
            let label = t
                .get(&label_field)
                .expect("label field not found")
                .as_integer()
                .expect("label field not an int");
            if last_label == Some(label) {
                current_streak += 1;
                if current_streak >= minimum_emit_streak {
                    match total_streaks.entry(label) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if e.get() < &current_streak {
                                e.insert(current_streak);
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(current_streak);
                        }
                    }
                }
            } else if last_label.is_none() {
                last_label = Some(label);
                current_streak = 1;
                // do not insert if minimum is not met
                // total_streaks.insert(label, current_streak);
            } else {
                last_label = Some(label);
                current_streak = 1;
                // do not insert if minimum is not met
                // total_streaks.insert(label, current_streak);
            }
        }

        // remove excess items
        while queue.len() >= buffer_len {
            queue.pop_front();
        }

        if current_streak >= minimum_emit_streak {
            let (&label, &streak) = total_streaks
                .iter()
                .max_by_key(|(_, streak)| *streak)
                .unwrap();

            let tuple_ids_for_streak: Vec<_> = queue.iter().map(|t| t.id()).collect();

            let mut tuple = get_tuple();
            let tuple_id = tuple.id();
            tuple.insert("patient_id".into(), HabValue::Integer(patient_id));
            tuple.insert("streak_length".into(), HabValue::Integer(streak as i32));
            tuple.insert(label_field.clone(), HabValue::Integer(label));

            tuples.push(tuple);

            'log_streak: {
                use watershed_shared::global_logger;
                let log_location = "diagnosis_streak_match".to_raw_key();
                let aux_data = [
                    (
                        "streak_length".to_raw_key(),
                        global_logger::LimitedHabValue::UnsignedInteger(streak as _),
                    ),
                    (
                        "label".to_raw_key(),
                        global_logger::LimitedHabValue::UnsignedInteger(label as _),
                    ),
                    (
                        "patient_id".to_raw_key(),
                        global_logger::LimitedHabValue::UnsignedInteger(patient_id as _),
                    ),
                    (
                        "streak_tuple_ids".to_raw_key(),
                        global_logger::LimitedHabValue::String(
                            match serde_json::to_string(&tuple_ids_for_streak) {
                                Ok(v) => v.into(),
                                Err(e) => {
                                    error!("failed to serialize tuple ids for streak: {e}");
                                    break 'log_streak;
                                }
                            },
                        ),
                    ),
                ];
                let aux_data = Some(aux_data);
                if let Err(e) = global_logger::log_data(tuple_id, log_location, aux_data) {
                    for err in e {
                        error!("failed to log diagnosis streak: {err}");
                    }
                    break 'log_streak;
                }
            }

            AggregationResult {
                emit: Some(tuples),
                is_finished: false,
            }
        } else {
            AggregationResult {
                emit: None,
                is_finished: false,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PendingData {
    time_of_scheduling: Instant,
    age_when_scheduling_ns: u128,
    tuple_id: usize,
}

/// if at least HISTORY_DROP_PERCENTILE of the items are older than the deadline window, then we *must* drop the current items
const HISTORY_DROP_PERCENTILE: f64 = 0.05;

fn routing_fn_static(
    use_small: bool,
    window_size: usize,
    deadline_window_ms: u64,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    let deadline_window_ms = deadline_window_ms as u128;

    #[derive(Debug, Clone)]
    struct PastData {
        time_of_scheduling: Instant,
        tuple_id: usize,
        time_elapsed_ns: u128,
        total_age_ns: u128,
        pipeline_id: usize,
    }

    let mut history = VecDeque::<PastData>::with_capacity(10);
    let mut pending_data: nohash_hasher::IntMap<usize, PendingData> =
        nohash_hasher::IntMap::default();

    let mut age_heap: std::collections::BinaryHeap<u128> =
        std::collections::BinaryHeap::with_capacity(window_size);
    let mut age_heap_rev: std::collections::BinaryHeap<std::cmp::Reverse<u128>> =
        std::collections::BinaryHeap::with_capacity(window_size);

    let my_running_time_ms = if use_small {
        HARD_CLASS_BIN.costs[1]
    } else {
        HARD_CLASS_BIN.costs[2]
    };

    move |mut tuples, senders| {
        debug!(
            "received {:?} tuples in static routing function",
            tuples.len()
        );
        let time_of_scheduling = Instant::now();
        let Ok(time_of_scheduling_system_time) =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        else {
            error!("time went backwards");
            return None;
        };
        let time_of_scheduling_system_time_ns = time_of_scheduling_system_time.as_nanos();

        let current_system_time_ns = time_of_scheduling_system_time_ns;
        // loop backwards through tuples and remove any tuples that have been alive for too long
        let mut i = 0;
        let mut too_old = get_tuple_vec();
        while i < tuples.len() {
            let tuple = &tuples[i];
            let tuple_id = tuple.id();
            let time_created_ns = tuple.unix_time_created_ns();
            let time_alive_ns = current_system_time_ns - time_created_ns;
            let time_alive_ms = time_alive_ns / 1_000_000;
            if time_alive_ms > deadline_window_ms {
                warn!(
                    "tuple {} has been alive for {}ms, removing",
                    tuple_id, time_alive_ms
                );
                // add to history
                let past_data = PastData {
                    time_of_scheduling: Instant::now(),
                    tuple_id: tuple_id as _,
                    time_elapsed_ns: 0,
                    total_age_ns: time_alive_ns,
                    pipeline_id: 0,
                };

                history.push_back(past_data);
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

        'quit_filtering: {
            if window_size == 0 {
                error!("why is window size 0");
                break 'quit_filtering;
            }
            while let Ok(v) = back_channel.try_recv() {
                for (tuple_id, pipeline_id, time_received) in v {
                    if let Some(pending) = pending_data.remove(&tuple_id) {
                        let time_elapsed_ns = time_received
                            .duration_since(pending.time_of_scheduling)
                            .as_nanos();
                        history.push_back(PastData {
                            time_of_scheduling: pending.time_of_scheduling,
                            tuple_id,
                            time_elapsed_ns,
                            total_age_ns: pending.age_when_scheduling_ns + time_elapsed_ns,
                            pipeline_id,
                        });
                    }
                }
            }
            history
                .make_contiguous()
                .sort_by_key(|x| x.time_of_scheduling);
            while history.len() > window_size {
                let _stale_data = history.pop_front();
            }

            if history.len() < 2 {
                warn!("not enough history to calculate velocity");
                break 'quit_filtering;
            }

            debug!("static routing history: {:?}", history);

            trace!("g0");

            let earliest_time = history.front().map(|x| x.time_of_scheduling).unwrap();
            trace!("g1");
            let latest_time = history.back().map(|x| x.time_of_scheduling).unwrap();
            trace!("g2");
            let scheduling_time_elapsed_ms = latest_time.duration_since(earliest_time).as_millis();
            let incoming_item_ratio_to_keep = if scheduling_time_elapsed_ms == 0 {
                warn!("total time elapsed says it is 0. has time gone backwards?");
                None
            } else {
                let total_items = history.len();
                let incoming_items_per_ms =
                    (total_items as f64) / (scheduling_time_elapsed_ms as f64);
                let incoming_ms_per_item = 1.0 / incoming_items_per_ms;

                // if we're already keeping up, don't bother, but if we are taking longer, then we need to cut by at least that amount
                let ratio_to_keep = if incoming_ms_per_item < my_running_time_ms as f64 {
                    incoming_ms_per_item / my_running_time_ms as f64
                } else {
                    1.0
                };
                Some(ratio_to_keep)
            };

            debug!(
                "incoming item ratio to keep: {:?}",
                incoming_item_ratio_to_keep
            );

            // now for feedback ratio

            let sum_elapsed_ns: u128 = history.iter().map(|x| x.time_elapsed_ns).sum();
            // we know history has at least 2 elements, so it's not empty and we can divide by its length
            let mean_time_elapsed_ms =
                ((sum_elapsed_ns as f64) / (history.len() as f64)) / 1_000_000.0;
            let incoming_items_per_ms = 1.0 / mean_time_elapsed_ms;

            debug!(
                "mean time elapsed: {:?}ms, incoming items per ms: {:?}, my running time: {:?}ms",
                mean_time_elapsed_ms, incoming_items_per_ms, my_running_time_ms
            );

            // if mean time elapsed is fine then we don't need to drop anything more
            let back_channel_ratio_to_keep = if mean_time_elapsed_ms < my_running_time_ms as f64 {
                1.0
            } else {
                my_running_time_ms as f64 / mean_time_elapsed_ms
            };
            debug!(
                "back channel ratio to keep: {:?}",
                back_channel_ratio_to_keep,
            );

            let total_age_ns: u128 = history.iter().map(|x| x.total_age_ns).sum();
            let mean_total_age_ms = ((total_age_ns as f64) / (history.len() as f64)) / 1_000_000.0;
            debug!("mean total age: {:?}ms", mean_total_age_ms);
            // compare that to our deadline
            let (behind_schedule, deadline_ratio_to_keep) =
                if mean_total_age_ms < deadline_window_ms as f64 {
                    (false, 1.0)
                } else {
                    // extra punish having total ages that are over the deadline
                    (
                        true,
                        (deadline_window_ms as f64 / mean_total_age_ms).powi(2),
                    )
                };

            debug!(
                "deadline ratio to keep: {:?}, behind schedule: {:?}",
                deadline_ratio_to_keep, behind_schedule
            );

            let mut ratio_to_keep = match (incoming_item_ratio_to_keep, behind_schedule) {
                (Some(incoming_item_ratio_to_keep), true) => {
                    // if we're behind schedule, we want to drop more
                    debug!("we're behind schedule, dropping more");
                    incoming_item_ratio_to_keep
                        .min(back_channel_ratio_to_keep)
                        .min(deadline_ratio_to_keep) as f64
                }
                (Some(incoming_item_ratio_to_keep), false) => {
                    // if we're ahead of schedule, we want to drop less
                    debug!("we're ahead of schedule, dropping less");
                    incoming_item_ratio_to_keep.min(back_channel_ratio_to_keep) as f64
                }
                (None, true) => {
                    // if we have no time elapsed measurement, we're in a weird state
                    // so we just drop everything
                    debug!("no time elapsed measurement, dropping everything");
                    0.0f64
                }
                (None, false) => {
                    // all the history was muddled, but we can fall back to the elapsed time from the back channel
                    debug!("history had ties in time elapsed, falling back to back channel");
                    back_channel_ratio_to_keep as f64
                }
            };

            let history_drop_index = (HISTORY_DROP_PERCENTILE * history.len() as f64) as usize;
            // use a binary heap to find the nth value. We use the reverse order if that would be the faster
            let nth_total_age_ns = if HISTORY_DROP_PERCENTILE < 0.5 {
                let heap = &mut age_heap;
                heap.clear();
                heap.extend(history.iter().map(|x| x.total_age_ns));
                for _ in 0..history_drop_index {
                    heap.pop();
                }
                match heap.pop() {
                    Some(v) => v,
                    None => {
                        error!("history should have more than 2 items");
                        deadline_window_ms * 1_000_000
                    }
                }
            } else {
                let heap = &mut age_heap_rev;
                heap.clear();
                heap.extend(history.iter().map(|x| std::cmp::Reverse(x.total_age_ns)));

                for _ in 0..history.len() - history_drop_index {
                    heap.pop();
                }
                match heap.pop() {
                    Some(v) => v.0,
                    None => {
                        error!("history should have more than 2 items");
                        deadline_window_ms * 1_000_000
                    }
                }
            };

            debug!(
                "{}th total age: {:?}ns, deadline window: {:?}ms",
                history_drop_index, nth_total_age_ns, deadline_window_ms
            );
            // if we're over the deadline, we must drop everything now
            // we absolutely must not allow more than n% of items to be too old, so if we're over that, we must drop everything
            if nth_total_age_ns > deadline_window_ms * 1_000_000 {
                warn!("we're over the deadline, dropping everything");
                ratio_to_keep = 0.0;

                // TODO: can this be relaxed to instead say that we should drop according to
                //   the ratio of items that are too old by the time they're done?
            }

            debug!("final computed ratio to keep: {:?}", ratio_to_keep);

            if !USE_COMPUTED {
                let sender = if use_small { &senders[1] } else { &senders[2] };
                match sender {
                    AsyncPipe::Active(bounded_async_sender) => {
                        if bounded_async_sender.remaining_capacity() == 0 {
                            ratio_to_keep = 0.0;
                        } else {
                            ratio_to_keep = 1.0;
                        }
                    }
                    AsyncPipe::Dummy => {
                        error!("there should never be a dummy pipe in the small/large slot");
                    }
                }
            }

            // debug!("final computed ratio to keep: {:?}", ratio_to_keep);

            trace!("g3");
            // use rng to filter indices randomly
            let should_drop: Vec<usize> = (0..tuples.len())
                .filter(|_| {
                    let r = rand::random::<f64>();
                    // if we're above water then this will never be true
                    // because r is in [0, 1)
                    r > ratio_to_keep
                })
                .collect();
            let mut dropped = 0;
            let mut dropped_tuples = Vec::new();
            for i in should_drop {
                trace!("g4-{i}");
                let tuple = tuples.remove(i - dropped);
                let tuple_id = tuple.id() as _;
                let time_alive_ns = current_system_time_ns - tuple.unix_time_created_ns();
                let past_data = PastData {
                    time_of_scheduling: Instant::now(),
                    tuple_id,
                    time_elapsed_ns: 0,
                    total_age_ns: time_alive_ns,
                    pipeline_id: 0,
                };
                history.push_back(past_data);
                dropped_tuples.push(tuple);
                dropped += 1;
            }
            trace!("g5 - dropping {} tuples", dropped);
            if dropped_tuples.len() > 0 {
                if let Err(e) = senders[0].send(dropped_tuples) {
                    error!("failed to send to drop pipe 0: {e:?}");
                }
            }
        }

        for remaining_tuple in &tuples {
            let tuple_id = remaining_tuple.id() as _;
            let pending = PendingData {
                time_of_scheduling: Instant::now(),
                tuple_id,
                age_when_scheduling_ns: current_system_time_ns
                    - remaining_tuple.unix_time_created_ns(),
            };
            pending_data.insert(tuple_id, pending);
        }
        let total_items = tuples.len();
        if total_items > 0 {
            if use_small {
                trace!("g6 - sending {} tuples to small path", total_items);
                if let Err(e) = senders[1].send(tuples) {
                    error!("failed to send to pipe 1: {e:?}");
                }
            } else {
                trace!("g6 - sending {} tuples to big path", total_items);
                if let Err(e) = senders[2].send(tuples) {
                    error!("failed to send to pipe 2: {e:?}");
                }
            }
            Some(total_items)
        } else {
            trace!("g6 - no tuples");
            None
        }
    }
}

fn routing_fn_eddies(
    window_size: usize,
    deadline_window_ms: u64,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    let deadline_window_ms = deadline_window_ms as u128;

    struct PastData {
        time_of_scheduling: Instant,
        tuple_id: usize,
        time_elapsed_ns: u128,
        total_age_ns: u128,
        pipeline_id: usize,
    }
    let mut history = VecDeque::<PastData>::with_capacity(10);
    let mut pending_data: nohash_hasher::IntMap<usize, PendingData> =
        nohash_hasher::IntMap::default();

    let small_time_ms_per_item = SMALL_MODEL_RUNTIME;
    let big_time_ms_per_item = BIG_MODEL_RUNTIME;
    let big_little_ms_per_item_ratio = big_time_ms_per_item / small_time_ms_per_item;
    let little_big_ms_per_item_ratio = small_time_ms_per_item / big_time_ms_per_item;
    let little_big_ms_per_item_adjusted =
        (1.0 + (1.0 - little_big_ms_per_item_ratio)) * small_time_ms_per_item;
    let little_big_items_per_ms = 1.0 / little_big_ms_per_item_adjusted;

    let mut age_heap: std::collections::BinaryHeap<u128> =
        std::collections::BinaryHeap::with_capacity(window_size);

    let mut age_heap_rev: std::collections::BinaryHeap<std::cmp::Reverse<u128>> =
        std::collections::BinaryHeap::with_capacity(window_size);

    move |mut tuples, senders| {
        debug!(
            "received {:?} tuples in eddies routing function",
            tuples.len()
        );
        let time_of_scheduling = Instant::now();
        let Ok(time_of_scheduling_system_time) =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        else {
            error!("time went backwards");
            return None;
        };
        let time_of_scheduling_system_time_ns = time_of_scheduling_system_time.as_nanos();

        // loop backwards through tuples and remove any tuples that have been alive for too long
        let mut i = 0;
        let mut too_old = get_tuple_vec();
        while i < tuples.len() {
            let tuple = &tuples[i];
            let tuple_id = tuple.id();
            let time_created_ns = tuple.unix_time_created_ns();
            let current_system_time_ns = time_of_scheduling_system_time_ns;
            let time_alive_ns = current_system_time_ns - time_created_ns;
            let time_alive_ms = time_alive_ns / 1_000_000;
            if time_alive_ms > deadline_window_ms {
                warn!(
                    "tuple {} has been alive for {}ms, removing",
                    tuple_id, time_alive_ms
                );
                // add to history
                let past_data = PastData {
                    time_of_scheduling: Instant::now(),
                    tuple_id: tuple_id as _,
                    time_elapsed_ns: 0,
                    total_age_ns: time_alive_ns,
                    pipeline_id: 0,
                };
                history.push_back(past_data);
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

        let (big_path_tuples, small_path_tuples) = if USE_COMPUTED {
            'quit_filtering: {
                if window_size == 0 {
                    error!("why is window size 0");
                    break 'quit_filtering;
                }
                while let Ok(v) = back_channel.try_recv() {
                    for (tuple_id, pipeline_id, finish_time) in v {
                        if let Some(pending) = pending_data.remove(&tuple_id) {
                            let time_elapsed_ns = finish_time
                                .duration_since(pending.time_of_scheduling)
                                .as_nanos();
                            history.push_back(PastData {
                                time_of_scheduling: pending.time_of_scheduling,
                                tuple_id,
                                time_elapsed_ns,
                                total_age_ns: pending.age_when_scheduling_ns + time_elapsed_ns,
                                pipeline_id,
                            });
                        }
                    }
                }
                history
                    .make_contiguous()
                    .sort_by_key(|x| x.time_of_scheduling);
                while history.len() > window_size {
                    let _stale_data = history.pop_front();
                }
                if history.len() < 2 {
                    warn!("not enough history to calculate velocity");
                    break 'quit_filtering;
                }

                trace!("g0");
                let earliest_time = history.front().map(|x| x.time_of_scheduling).unwrap();
                trace!("g1");
                let latest_time = history.back().map(|x| x.time_of_scheduling).unwrap();
                trace!("g2");
                let Some(elapsed_duration) = latest_time.checked_duration_since(earliest_time)
                else {
                    error!(
                        "time went backwards with a history of size {}",
                        history.len()
                    );
                    return None;
                };
                let total_elapsed_ns = elapsed_duration.as_nanos();

                let incoming_item_ratio_to_keep = if total_elapsed_ns == 0 {
                    warn!("total time elapsed says it is 0. has time gone backwards?");
                    // break 'quit_filtering;
                    None
                } else {
                    let aggregate_avg_elapsed_ms_per_item =
                        ((total_elapsed_ns as f64) / (history.len() as f64)) / 1_000_000.0;
                    let keep_rate = if aggregate_avg_elapsed_ms_per_item
                        > little_big_ms_per_item_adjusted as f64
                    {
                        1.0
                    } else {
                        aggregate_avg_elapsed_ms_per_item / little_big_items_per_ms as f64
                    };
                    Some(keep_rate)
                };

                debug!(
                    "incoming item ratio to keep: {:?}",
                    incoming_item_ratio_to_keep
                );

                let sum_elapsed_ns: u128 = history.iter().map(|x| x.time_elapsed_ns).sum();
                // we know history has at least 2 elements, so it's not empty and we can divide by its length
                let mean_time_elapsed_ms =
                    ((sum_elapsed_ns as f64) / (history.len() as f64)) / 1_000_000.0;
                let incoming_items_per_ms = 1.0 / mean_time_elapsed_ms;

                // if we have extra time
                let back_channel_ratio_to_keep =
                    if mean_time_elapsed_ms < little_big_ms_per_item_adjusted as f64 {
                        1.0
                    } else {
                        little_big_ms_per_item_adjusted as f64 / mean_time_elapsed_ms
                    };

                debug!(
                "mean time elapsed: {:?}ms, incoming items per ms: {:?}, my running time: {:?}ms",
                mean_time_elapsed_ms, incoming_items_per_ms, little_big_ms_per_item_adjusted
            );

                let total_age_ns: u128 = history.iter().map(|x| x.total_age_ns).sum();
                let mean_total_age_ms =
                    ((total_age_ns as f64) / (history.len() as f64)) / 1_000_000.0;
                debug!("mean total age: {:?}ms", mean_total_age_ms);
                // compare that to our deadline
                let (behind_schedule, deadline_ratio_to_keep) =
                    if mean_total_age_ms < deadline_window_ms as f64 {
                        (false, 1.0)
                    } else {
                        // extra punish having total ages that are over the deadline
                        (
                            true,
                            (deadline_window_ms as f64 / mean_total_age_ms).powi(2),
                        )
                    };

                let mut ratio_to_keep = match (incoming_item_ratio_to_keep, behind_schedule) {
                    (Some(incoming_item_ratio_to_keep), true) => {
                        // if we're behind schedule, we want to drop more
                        incoming_item_ratio_to_keep
                            .min(back_channel_ratio_to_keep)
                            .min(deadline_ratio_to_keep) as f64
                    }
                    (Some(incoming_item_ratio_to_keep), false) => {
                        // if we're ahead of schedule, we want to drop less
                        incoming_item_ratio_to_keep.min(back_channel_ratio_to_keep) as f64
                    }
                    (None, true) => {
                        // if we have no time elapsed, we're in a weird state
                        // so we just drop everything
                        0.0f64
                    }
                    (None, false) => {
                        // if we have no distinction, but we can fall back to the elapsed time from the back channel
                        back_channel_ratio_to_keep as f64
                    }
                };

                let history_drop_index = (HISTORY_DROP_PERCENTILE * history.len() as f64) as usize;
                // use a binary heap to find the nth value. Reverse the order if that would be the faster case
                let nth_total_age_ns = if HISTORY_DROP_PERCENTILE < 0.5 {
                    let heap = &mut age_heap;
                    heap.clear();
                    heap.extend(history.iter().map(|x| x.total_age_ns));
                    for _ in 0..history_drop_index {
                        heap.pop();
                    }
                    match heap.pop() {
                        Some(v) => v,
                        None => {
                            error!("history should have more than 2 items");
                            deadline_window_ms * 1_000_000
                        }
                    }
                } else {
                    let heap = &mut age_heap_rev;
                    heap.clear();
                    heap.extend(history.iter().map(|x| std::cmp::Reverse(x.total_age_ns)));

                    for _ in 0..history.len() - history_drop_index {
                        heap.pop();
                    }
                    match heap.pop() {
                        Some(v) => v.0,
                        None => {
                            error!("history should have more than 2 items");
                            deadline_window_ms * 1_000_000
                        }
                    }
                };
                // if we're over the deadline, we must drop everything now
                // we absolutely must not allow more than n% of items to be too old, so if
                // we're over that, we must drop everything
                if nth_total_age_ns > deadline_window_ms * 1_000_000 {
                    warn!("we're over the deadline, dropping everything");
                    ratio_to_keep = 0.0;
                }

                debug!("final computed ratio to keep: {:?}", ratio_to_keep);
                trace!("g3");
                // use rng to filter indices randomly
                let should_drop: Vec<usize> = (0..tuples.len())
                    .filter(|_| {
                        let r = rand::random::<f64>();
                        // if we're above water then this will never be true
                        // because r is in [0, 1)
                        r > ratio_to_keep
                    })
                    .collect();
                let mut dropped = 0;
                let mut dropped_tuples = Vec::new();
                for i in should_drop {
                    trace!("g4-{i}");
                    let tuple = tuples.remove(i - dropped);
                    let tuple_id = tuple.id() as _;
                    let past_data = PastData {
                        time_of_scheduling,
                        tuple_id,
                        time_elapsed_ns: 0,
                        total_age_ns: time_of_scheduling_system_time_ns
                            - tuple.unix_time_created_ns(),
                        pipeline_id: 0,
                    };
                    history.push_back(past_data);
                    dropped_tuples.push(tuple);
                    dropped += 1;
                }
                trace!("g5 - dropping {} tuples", dropped);
                if dropped_tuples.len() > 0 {
                    if let Err(e) = senders[0].send(dropped_tuples) {
                        error!("failed to send to drop pipe 0: {e:?}");
                    }
                }
            }

            let total_items = tuples.len();
            if total_items == 0 {
                return None;
            }
            let mut big_path_tuples = Vec::new();
            let mut small_path_tuples = Vec::new();
            for remaining_tuple in tuples {
                let tuple_id = remaining_tuple.id() as _;
                let current_age =
                    time_of_scheduling_system_time_ns - remaining_tuple.unix_time_created_ns();
                let pending = PendingData {
                    time_of_scheduling: Instant::now(),
                    tuple_id,
                    age_when_scheduling_ns: current_age,
                };
                pending_data.insert(tuple_id, pending);
                let random_number = rand::random::<f64>();
                let odds = big_time_ms_per_item / (big_time_ms_per_item + small_time_ms_per_item);
                if random_number < odds {
                    big_path_tuples.push(remaining_tuple);
                } else {
                    small_path_tuples.push(remaining_tuple);
                }
            }
            (big_path_tuples, small_path_tuples)
        } else {
            let mut big_path_tuples = Vec::new();
            let mut small_path_tuples = Vec::new();
            let mut drop_tuples = Vec::new();
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for t in tuples {
                // randomly select a path to start, and if it is not free, spill over
                // if neither is available, then we must drop
                let (i1, path1, i2, path2) = if rng.gen_bool(0.5) {
                    (1, &mut small_path_tuples, 2, &mut big_path_tuples)
                } else {
                    (2, &mut big_path_tuples, 1, &mut small_path_tuples)
                };
                let AsyncPipe::Active(s1) = &senders[i1] else {
                    error!("pipe {i1} was a dummy when it shouldn't be");
                    drop_tuples.push(t);
                    continue;
                };
                if s1.remaining_capacity() > 0 {
                    debug!("pipe {i1} was free, sending tuple {} to it", t.id());
                    path1.push(t);
                    continue;
                }
                let AsyncPipe::Active(s2) = &senders[i2] else {
                    error!("pipe {i2} was a dummy when it shouldn't be");
                    drop_tuples.push(t);
                    continue;
                };
                if s2.remaining_capacity() > 0 {
                    debug!(
                        "pipe {i1} was full, but we were able to send tuple {} to pipe {i2}",
                        t.id()
                    );
                    path2.push(t);
                } else {
                    debug!("both pipes were full, dropping tuple {}", t.id());
                    drop_tuples.push(t);
                }
            }
            (big_path_tuples, small_path_tuples)
        };
        let total_items = big_path_tuples.len() + small_path_tuples.len();
        if big_path_tuples.len() > 0 {
            trace!("g6 - sending {} tuples to big path", big_path_tuples.len());
            if let Err(e) = senders[2].send(big_path_tuples) {
                error!("failed to send to pipe 1: {e:?}");
            }
        }
        if small_path_tuples.len() > 0 {
            trace!(
                "g6 - sending {} tuples to small path",
                small_path_tuples.len()
            );
            if let Err(e) = senders[1].send(small_path_tuples) {
                error!("failed to send to pipe 2: {e:?}");
            }
        }
        Some(total_items)
    }
}

const BIG_MODEL_RUNTIME: f64 = 39.20456180969874;
const SMALL_MODEL_RUNTIME: f64 = 19.500647087891895;

// as given in jan4_preds.npy based on features trained from jan4_fft_features.npy

// Training accuracy
const BIG_MODEL_REWARD_EASY_CLASS: f64 = 0.9118714359771902;
const SMALL_MODEL_REWARD_EASY_CLASS: f64 = 0.8228788664247452;
const BIG_MODEL_REWARD_HARD_CLASS: f64 = 0.8898761835396941;
const SMALL_MODEL_REWARD_HARD_CLASS: f64 = 0.756300072833212;

const HARD_CLASS_BIN: BinInfo<BasicCategory> = BinInfo {
    id: Some(BasicCategory::HardClass),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        SMALL_MODEL_REWARD_HARD_CLASS,
        BIG_MODEL_REWARD_HARD_CLASS,
    ]),
    costs: ShareableArray::Borrowed(&[0.0, SMALL_MODEL_RUNTIME, BIG_MODEL_RUNTIME]),
};
const EASY_CLASS_BIN: BinInfo<BasicCategory> = BinInfo {
    id: Some(BasicCategory::EasyClass),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        SMALL_MODEL_REWARD_EASY_CLASS,
        BIG_MODEL_REWARD_EASY_CLASS,
    ]),
    costs: ShareableArray::Borrowed(&[0.0, SMALL_MODEL_RUNTIME, BIG_MODEL_RUNTIME]),
};

// use watershed_shared::scheduler::basic_probability_forecast::History to predict the future based on the past
//  and then schedule accordingly using watershed_shared::scheduler::aquifer_scheduler

// these are heuristics just to get an idea of how many items we should project using their history
// each algorithm can only handle so much future window before its complexity blows out of control
static OPTIMAL_MAX_COUNT: AtomicUsize = AtomicUsize::new(5);
// greedy can handle much more but this is usually good enough
static GREEDY_MAX_COUNT: AtomicUsize = AtomicUsize::new(10);

fn aquifer_routing_fn(
    model_info: watershed_shared::preclassifier_lang::RealBucketLookup,
    keep_n_history_items: usize,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
    deadline_ms: u64,
    lookahead_ms: u64,
    strategy: scheduler::Strategy,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    let budget_to_lookahead_ratio = lookahead_ms as f64 / deadline_ms as f64;
    let optimal_max_count = OPTIMAL_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let greedy_max_count = GREEDY_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let mut history = watershed_shared::scheduler::basic_probability_forecast::History::new(
        keep_n_history_items,
        back_channel,
        model_info.buckets.to_vec(),
    );
    let max_cost = model_info
        .buckets
        .iter()
        .filter_map(|x| x.costs.last())
        .fold(0.0, |x, y| y.max(x));
    let binning_fn = move |tuple: &Tuple| -> BinInfo<
        watershed_shared::preclassifier_lang::PreclassifierLangClass,
    > {
        // TODO: use a default bucket when something goes wrong
        let features = match tuple.get(FFT_FEATURE_FIELD) {
            Some(f) => f,
            None => {
                let msg = format!("fft features field {FFT_FEATURE_FIELD} not found");
                error!("{msg}");
                panic!("{msg}");
            }
        };
        let features = match features.as_int_buffer() {
            Some(f) => f,
            None => {
                error!("fft features were not stored as an int buffer");
                panic!("fft features were not stored as an int buffer");
            }
        };
        let features = match bytemuck::try_cast_slice::<_, f32>(features.as_ref()) {
            Ok(f) => f,
            Err(e) => {
                let msg = format!("failed to cast features to f32 slice: {e}");
                error!("{msg}");
                panic!("{msg}");
            }
        };

        watershed_shared::preclassifier_lang::map_inputs_to_bucket(features, &model_info)
    };
    let forecast_function =
        History::<watershed_shared::preclassifier_lang::PreclassifierLangClass>::forecast_fn();
    let deadline_ms = deadline_ms as u128;
    move |mut tuples, senders| {
        debug!("received {:?} tuples in routing function", tuples.len());
        if !matches!(senders, [_drop_channel, _small_channel, _big_model_channel]) {
            error!("Expected channels were not present. Expected 3 channels [drop, small, big], got {:?} channels", senders.len());
            return None;
        }
        trace!("closure g0");

        let time_of_scheduling = Instant::now();
        let time_of_scheduling_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();

        // filter out tuples that have been alive for too long
        let mut i = 0;
        let mut too_old = get_tuple_vec();
        while i < tuples.len() {
            let tuple = &tuples[i];
            let time_created_ns = tuple.unix_time_created_ns();
            let diff = time_of_scheduling_ns - time_created_ns;
            let diff_ms = diff / 1_000_000;
            if diff_ms > deadline_ms {
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
            debug!("mean overage ratio is {mean_overage_ratio:.2}. we are not following it, but if we did, we woudl reduce our budget per item from {rate_per_item_ns} to {val:.2}ns");
            // val
            rate_per_item_ns
        } else {
            // we are on average under budget, so we can use the rate per item as is
            rate_per_item_ns
        };
        debug!(
            "rate per item is {rate_per_item_ns:.2}ns, mean age per item is {mean_age_per_item_ns:.2}ns, budget per item is {budget_per_item_ns:.2}ns"
        );
        let items_per_ms = 1_000_000.0 / rate_per_item_ns;
        let lookahead_ms = (lookahead_ms as f64).max(1.0);
        let items_in_lookahead = items_per_ms * lookahead_ms;
        debug!(
            "items per ms: {items_per_ms:.5}, expected items in lookahead of {lookahead_ms}ms: {items_in_lookahead:.2}"
        );
        let budget_in_lookahead_ns = items_in_lookahead * budget_per_item_ns;

        let budget_ms = ((budget_in_lookahead_ns / 1_000_000.0) as f64).max(0.0);
        // We calculated the budget based on the requirements for this current tuple,
        // but we want to allow the algorithm to look ahead a bit, so we scale the budget
        // by a ratio that allows it to look ahead for the desired lookahead time.
        let budget_ms = budget_ms * budget_to_lookahead_ratio;
        debug!(
            "old version of budget_per_item_ns: {rate_per_item_ns:.2}ns, or adjusted to {}ns",
            rate_per_item_ns / mean_overage_ratio
        );

        // now let's make a new version based on how much the queues are lagging
        // we are scheduling items every `rate_per_item_ns` nanoseconds,
        // and if that has negative consequences, then the average time elapsed for the items
        // in the history will be increasing over time.
        let mean_elapsed_increase_ms = history.mean_elapsed_increase_ms();
        // we will care more about how that's been going recently, so we will use the weighted version
        let weighted_mean_elapsed_increase_ms = history.recent_weighted_mean_elapsed_increase_ms();

        // what about the age of the items themselves? are they continuing to get too old?
        let mean_final_age_ms = history.mean_age_when_merging_increase_ms();
        let weighted_mean_final_age_ms =
            history.recent_weighted_mean_age_when_merging_increase_ms();
        debug!(
            "Recent History consequences: mean elapsed increase: {mean_elapsed_increase_ms:.2} ms , recent-weighted mean elapsed increase: {weighted_mean_elapsed_increase_ms:.2} ms, mean final age increase: {mean_final_age_ms:.2} ms, recent-weighted mean final age increase: {weighted_mean_final_age_ms:.2} ms",
        );
        let penalty_metric_ms = weighted_mean_elapsed_increase_ms;
        // how much of a delay were we allowed to begin with?
        // if it delays by a little, but we still have plenty of time, then we won't give as big of a penalty
        let allowed_time_ms = deadline_ms as f64;
        let excess_ratio = penalty_metric_ms / allowed_time_ms;
        debug!(
            "penalty metric is {penalty_metric_ms:.2}ms, allowed time is {allowed_time_ms:.2}ms, excess ratio is {excess_ratio:.2}"
        );
        #[allow(unused)]
        #[derive(Debug, Clone, Copy)]
        enum LargeExcessPunishment {
            Normal,
            Sqrt,
        }
        const LARGE_EXCESS_PUNISHMENT: LargeExcessPunishment = LargeExcessPunishment::Sqrt;
        let history_penalized_lookahead_ms = if excess_ratio > 1.0 {
            error!(
                "we are more than 100% over budget ({excess_ratio:.2}), so we will not schedule any items"
            );
            0.0
        } else if excess_ratio > 0.1 {
            warn!("we are more than 10% over budget ({excess_ratio:.2}), so we will greatly reduce our budget per item");
            let penalized = match LARGE_EXCESS_PUNISHMENT {
                LargeExcessPunishment::Normal => lookahead_ms * (1.0 - excess_ratio),
                LargeExcessPunishment::Sqrt => lookahead_ms * (1.0 - excess_ratio.sqrt()),
            };
            debug!("we are over budget by {excess_ratio:.2}, reducing budget to {penalized:.2}ms");
            penalized
        } else if excess_ratio > 0.0 {
            // we are over budget, but not by much, so we will reduce our budget per item
            let reduced_budget_ms = lookahead_ms * (1.0 - excess_ratio);
            debug!(
                "we are over budget by {excess_ratio:.2}, reducing budget to {reduced_budget_ms:.2}ms"
            );
            reduced_budget_ms
        } else {
            // we are under budget, so we can use the full budget
            lookahead_ms
        };
        debug!(
            "new method that produces a history penalized budget produced a new budget of: {history_penalized_lookahead_ms:.2}ms for a maximum of greedy={greedy_max_count} and optimal={optimal_max_count} items",
            greedy_max_count = greedy_max_count,
        );
        let budget_ms = history_penalized_lookahead_ms;

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
        trace!("closure g3");
        let alg_inputs = AlgInputs {
            binning_function: &binning_fn,
            forecast_function: &forecast_function,
            send_function: History::send,
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
            if let Err(e) =
                watershed_shared::global_logger::log_data(tuple_id, log_location, Some(aux_data))
            {
                for err in e {
                    error!("failed to log time to schedule: {err}");
                }
            }
        }

        trace!("closure g5");
        out
    }
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
struct LogisticModelInfo {
    model_coefficients: Vec<f64>,
    model_type: Option<String>,
    // concerned_classes: Option<Vec<u8>>,
    description: String,
    model_intercept: f64,
}

fn read_patient_data_from_csv(file_name: &str) -> anyhow::Result<impl Iterator<Item = Tuple>> {
    // use polars
    use polars::prelude::CsvReadOptions;
    let df: DataFrame = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_name.into()))?
        .finish()?;

    let mut idx = 0;
    let iter = std::iter::from_fn(move || {
        let item: Vec<_> = df.get(idx)?;
        idx += 1;
        use polars::datatypes::AnyValue;
        let &[AnyValue::Int64(patient_id), ref potential_string_val, AnyValue::Int64(age), AnyValue::Float64(minutes_since_last_checkup), ..] =
            item.as_slice()
        else {
            error!("not enough items or wrong types");
            panic!("not enough items or wrong types");
        };
        let patient_name = match potential_string_val {
            AnyValue::String(s) => s.to_string(),
            AnyValue::StringOwned(s) => s.to_string(),
            _ => {
                error!("wrong type on string val");
                panic!("wrong type on string val")
            }
        };
        let mut tuple_output = get_tuple();
        tuple_output.insert("patient_id".into(), HabValue::Integer(patient_id as i32));
        tuple_output.insert("patient_name".into(), HabValue::String(patient_name.into()));
        tuple_output.insert("age".into(), HabValue::Integer(age as i32));
        tuple_output.insert(
            "minutes_since_last_checkup".into(),
            HabValue::Float(From::from(minutes_since_last_checkup as f64)),
        );
        Some(tuple_output)
    });
    Ok(iter)
}

pub fn make_fft_analysis() -> impl 'static + FnMut(Vec<&HabValue>) -> HabValue {
    let mut planner = FftPlanner::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(SAMPLE_LEN);
    // scratch is different because it doesn't get reset every time
    let mut scratch = vec![Complex32::new(0.0, 0.0); SAMPLE_LEN];

    const HIST_BUCKETS: usize = 5;
    const NOISE_FLOOR: f32 = 0.05;
    const MINIMUM_NOISE_FILTERED_LEN: f32 = 0.05;
    let mut signal_buffer = vec![Default::default(); SAMPLE_LEN];
    // let mut magnitude_buffer = vec![Default::default(); SAMPLE_LEN / 2];
    let mut magnitude_buffer = vec![Default::default(); SAMPLE_LEN];
    let mut filtered_magnitude_buffer = vec![Default::default(); SAMPLE_LEN];
    move |values| {
        let Some(buf) = values[0].as_int_buffer() else {
            error!("expected int buffer");
            return HabValue::Null;
        };

        let Ok(buf_float) = bytemuck::try_cast_slice::<_, f32>(buf) else {
            error!("failed to cast buffer to f32 slice");
            return HabValue::Null;
        };

        let Ok(input) = ArrayView2::from_shape((2000, 12), buf_float) else {
            error!("failed to cast buffer to array view");
            return HabValue::Null;
        };

        magnitude_buffer.clear();
        filtered_magnitude_buffer.clear();
        let features = fft_analysis::<SAMPLE_LEN>(
            &*fft,
            &mut scratch,
            input.t(),
            &mut signal_buffer,
            &mut magnitude_buffer,
            &mut filtered_magnitude_buffer,
            // values.iter().map(|v| v.as_float().unwrap()).collect(),
            HIST_BUCKETS,
            NOISE_FLOOR,
            MINIMUM_NOISE_FILTERED_LEN,
        );
        let Ok(v) = bytemuck::try_cast_vec(features) else {
            error!("failed to cast features to bytes");
            return HabValue::Null;
        };
        HabValue::IntBuffer(v)
        // HabValue::IntBuffer(bytemuck::cast_vec(features))
    }
}

pub fn fft_analysis<const N: usize>(
    fft: &dyn Fft<f32>,
    scratch: &mut Vec<Complex32>,
    signal_matrix: ArrayView2<f32>,
    signal: &mut Vec<Complex32>,
    magnitude_buffer: &mut Vec<f32>,
    filtered_magnitude_buffer: &mut Vec<f32>,
    num_buckets: usize,
    noise_floor_threshold: f32,
    minimum_noise_filtered_len: f32,
) -> Vec<f32> {
    let mut features = Vec::new();
    for channel in signal_matrix.outer_iter() {
        signal.clear();
        signal.extend(channel.iter().map(|&x| Complex32::new(x, 0.0)));

        // Perform FFT
        fft.process_with_scratch(&mut *signal, &mut *scratch);

        // Compute magnitudes
        magnitude_buffer.clear();
        magnitude_buffer.extend(signal.iter().map(|c| c.norm()));

        // Filter by noise floor threshold
        let max_val = magnitude_buffer.iter().cloned().fold(f32::MIN, f32::max);
        filtered_magnitude_buffer.clear();
        filtered_magnitude_buffer.extend(
            magnitude_buffer
                .iter()
                .cloned()
                .filter(|&x| x >= max_val * noise_floor_threshold),
        );

        let filtered_len = filtered_magnitude_buffer.len();
        let culled_buffer = if filtered_len as f32 <= minimum_noise_filtered_len * N as f32 {
            filtered_magnitude_buffer.clear();
            filtered_magnitude_buffer.extend(
                magnitude_buffer
                    .iter()
                    .cloned()
                    .filter(|&x| x >= max_val * 0.01),
            );
            &mut *filtered_magnitude_buffer
        } else {
            &mut *filtered_magnitude_buffer
        };

        // Histogram calculation
        let min_val = culled_buffer.iter().cloned().fold(f32::MAX, f32::min);
        let max_val = culled_buffer.iter().cloned().fold(f32::MIN, f32::max);
        let bin_width = (max_val - min_val) / num_buckets as f32;

        let mut histogram = vec![0.0; num_buckets];
        for &val in culled_buffer.iter() {
            let bin = ((val - min_val) / bin_width).floor() as usize;
            if bin < num_buckets {
                histogram[bin] += 1.0;
            }
        }

        features.extend(histogram);
        let bin_edges = (1..=num_buckets).map(|i| min_val + i as f32 * bin_width);
        features.extend(bin_edges);
    }

    features
}
