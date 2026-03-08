#[allow(unused_imports)]
use polars::prelude::*;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use watershed_shared::async_query_builder::RuntimeState;
use watershed_shared::basic_pooling::get_tuple;
use watershed_shared::caching::StrToKey;
use watershed_shared::global_logger::LimitedHabValue;
use watershed_shared::scheduler::{self, AlgInputs, BinInfo, FutureWindowKind, ShareableArray};
use watershed_shared::{query_builder, AsyncPipe, Operator, UdfBolt};

#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::collections::{BTreeMap, VecDeque};

use std::io::BufRead;
use std::sync::atomic::{self, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use watershed_shared::query_builder::QueryDescriptor;
use watershed_shared::{HabString, HabValue, Tuple};

use futures::Stream;
use tokio::sync::watch;

#[derive(Debug, Deserialize)]
struct GptExperimentConfig {
    boolq_file: String,
    gpt_results_path: String,
    bucket_path: String,
    cached_embedding_path: String,
    query_path: String,
    max_total_samples: Option<usize>,
    history_window_size: Option<usize>,
    greedy_lookahead_window_size: Option<usize>,
    optimal_lookahead_window_size: Option<usize>,
    deadline_window_ms: Option<u64>,
    money_budget_per_deadline: f64,
    target_time_micros: Option<Delay>,
    input_delay_micros: Option<Delay>,
    overall_time_limit_ms: Option<u64>,
    initial_startup_delay_ms: Option<u64>,
    routing_strategy: Option<RoutingOptions>,
    log_folder: Option<HabString>,
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

#[derive(Debug, Clone, Copy, Deserialize, Default)]
enum RoutingOptions {
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
    person_id: usize,
    sequence_id: u128,
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

fn main() -> anyhow::Result<()> {
    async_main().inspect_err(|e| {
        error!("Async Main error: {:?}", e);
        println!("Async Main error: {:?}", e);
        // time to write before exiting
        std::thread::sleep(Duration::from_millis(1500));
    })
}

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
    let mut args = std::env::args();
    let _this_file = args.next().context("no file name")?;
    let config_path = args.next().context("no config path provided")?;
    let config = std::fs::read_to_string(config_path)?;

    // use the log4rs file
    let log_path = args.next().context("no logger config path provided")?;
    log4rs::init_file(log_path, Default::default()).context("failed to initialize log4rs")?;

    let GptExperimentConfig {
        boolq_file,
        gpt_results_path: _,
        bucket_path,
        cached_embedding_path,
        query_path,
        max_total_samples,
        history_window_size,
        greedy_lookahead_window_size,
        optimal_lookahead_window_size,
        deadline_window_ms,
        money_budget_per_deadline,
        target_time_micros,
        input_delay_micros,
        overall_time_limit_ms,
        initial_startup_delay_ms: initial_startup_delay,
        routing_strategy,
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
    let target_time_micros =
        target_time_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let max_target_time_ms = target_time_micros.max() / 1_000;
    if let Some(max_items) = max_total_samples {
        if max_items as u64 * max_target_time_ms
            > overall_time_limit_ms.unwrap_or(usize::MAX as u64)
        {
            warn!(
                "max items * target time exceeds overall time limit and is not expected to finish"
            );
        }
    }

    // let global_money_budget = Arc::new(Mutex::new(0.0f64));
    let global_money_budget = Arc::new(Mutex::new(money_budget_per_deadline));

    // let use_dummy = false;
    let tiny_cost_queue = VecDeque::with_capacity(window_size);
    let small_cost_queue = VecDeque::with_capacity(window_size);
    let big_cost_queue = VecDeque::with_capacity(window_size);
    // if use_dummy {
    //     small_cost_queue.push_back(SMALL_MODEL_DUMMY_COST);
    //     big_cost_queue.push_back(BIG_MODEL_DUMMY_COST);
    // }
    let tiny_cost_queue = Arc::new(Mutex::new(tiny_cost_queue));
    let small_cost_queue = Arc::new(Mutex::new(small_cost_queue));
    let big_cost_queue = Arc::new(Mutex::new(big_cost_queue));

    let input_delay_micros =
        input_delay_micros.unwrap_or(Delay::Fixed(per_sequence_default_delay * 1_000));
    let initial_startup_delay: u64 = initial_startup_delay.unwrap_or(10_000);

    let query = std::fs::read_to_string(&query_path)?;
    let mut function_lookup = BTreeMap::<HabString, FunctionKinds>::new();

    // set stack size to 32 MB
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_stack_size(32 * 1024 * 1024)
        .enable_all()
        .build()?;

    let (max_item_condition, stop_rx) = watch::channel(false);
    let all_items_produced_counter = Arc::new(AtomicUsize::new(0));

    let batch_size = 1;
    const TOTAL_OBJS: usize = 3270;
    const SKIP_RATIO: f64 = 0.5;
    const SKIP_AMOUNT: usize = (TOTAL_OBJS as f64 * SKIP_RATIO) as usize;
    let (budget_per_item, items_per_deadline) = {
        let ms_per_item = max_target_time_ms as f64;
        let ms_per_deadline = deadline_window_ms as f64;
        let budget_per_deadline = money_budget_per_deadline;
        let items_per_deadline = ms_per_deadline / ms_per_item;
        let budget_per_item = budget_per_deadline / items_per_deadline;
        (budget_per_item, items_per_deadline)
    };
    if items_per_deadline.fract() > 1e-6 {
        warn!("items per deadline is not a whole number: maximum {max_target_time_ms} ms per input item with {money_budget_per_deadline} budget per deadline window = {items_per_deadline} items per window. Is this intended?");
    }

    let boolq_src = rt.block_on(question_source(
        boolq_file,
        cached_embedding_path.clone(),
        global_money_budget.clone(),
        budget_per_item,
        initial_startup_delay,
        input_delay_micros,
        batch_size,
        SKIP_AMOUNT,
        max_total_samples.unwrap_or(usize::MAX),
        max_item_condition.clone(),
        stop_rx.clone(),
        all_items_produced_counter.clone(),
    ));

    let budget_insertion_future = {
        let global_money_budget = Arc::clone(&global_money_budget);
        let stop_rx = stop_rx.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(initial_startup_delay)).await;
            loop {
                match stop_rx.has_changed() {
                    Ok(true) => break,
                    Ok(false) => (), // continue as normal
                    Err(e) => {
                        error!("failed to check stop_rx in budget allotment task: {e}");
                        break;
                    }
                }
                {
                    // *global_money_budget.lock().unwrap() += 0.0;
                    *global_money_budget.lock().unwrap() += money_budget_per_deadline;
                }
                tokio::time::sleep(Duration::from_millis(deadline_window_ms)).await;
            }
        }
    };

    let boolq_src = Mutex::new(Some(boolq_src));
    let question_source_udf = FunctionKinds::SourceUdf(Box::new(move || {
        let Some(boolq_src) = boolq_src.lock().unwrap().take() else {
            error!("boolq_src already taken");
            return Box::new(|| {
                Box::new(futures::stream::empty()) as Box<dyn Stream<Item = Vec<Tuple>> + Send>
            });
        };
        Box::new(move || Box::new(boolq_src))
    }));

    function_lookup.insert("question_source".into(), question_source_udf);

    // all data is loaded from the folder so we don't need to load it at runtime
    let empty_source_udf = Box::new(|| {
        Box::new(|| Box::new(futures::stream::empty()) as Box<dyn Stream<Item = Vec<Tuple>> + Send>)
            as _
    });
    let knowledge_base_udf = empty_source_udf.clone();
    function_lookup.insert(
        "empty_source".into(),
        FunctionKinds::SourceUdf(empty_source_udf),
    );
    function_lookup.insert(
        "knowledge_base".into(),
        FunctionKinds::SourceUdf(knowledge_base_udf),
    );

    let finalize_question_udf = FunctionKinds::ComputationExpressionUdf(Box::new(|| {
        let f = finalize_question;
        Box::new(move |args: Vec<&HabValue>| f(args))
    }));

    function_lookup.insert("finalize_question".into(), finalize_question_udf);

    let bucket_lookup = watershed_shared::preclassifier_lang::load_file_format(
        &std::fs::read(&bucket_path).context("failed to open bucket file")?,
    )
    .context("failed to parse bucket file")?;

    let global_budget_clone = global_money_budget.clone();
    let small_cost_queue_clone = small_cost_queue.clone();
    let big_cost_queue_clone = big_cost_queue.clone();
    let (route_feedback_sender, route_feedback_receiver) = crossbeam::channel::unbounded();
    let routing_udf: FunctionKinds =
        FunctionKinds::RoutingUdf(Box::new(move || match routing_strategy {
            Some(
                option @ (RoutingOptions::AlwaysTiny
                | RoutingOptions::AlwaysSmall
                | RoutingOptions::AlwaysBig),
            ) => Box::new(routing_fn_static(option, global_budget_clone.clone())),
            Some(RoutingOptions::Random) => unimplemented!("Random not implemented"),
            Some(RoutingOptions::Eddies) => Box::new(routing_fn_eddies(
                window_size,
                deadline_window_ms,
                global_budget_clone.clone(),
                small_cost_queue_clone.clone(),
                big_cost_queue_clone.clone(),
            )),
            // keep_n_history_items: usize,
            // global_money_budget: Arc<Mutex<f64>>,
            // small_cost_queue: Arc<Mutex<VecDeque<f64>>>,
            // big_cost_queue: Arc<Mutex<VecDeque<f64>>>,
            // back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
            // deadline_ms: u64,
            // strategy: scheduler::Strategy,
            Some(RoutingOptions::AquiferGreedy) => Box::new(aquifer_routing_fn(
                window_size,
                global_budget_clone.clone(),
                route_feedback_receiver.clone(),
                deadline_window_ms,
                scheduler::Strategy::Greedy,
                bucket_lookup.clone(),
            )),
            Some(RoutingOptions::AquiferOptimal) => Box::new(aquifer_routing_fn(
                window_size,
                global_budget_clone.clone(),
                route_feedback_receiver.clone(),
                deadline_window_ms,
                scheduler::Strategy::Optimal,
                bucket_lookup.clone(),
            )),
            Some(RoutingOptions::AlwaysDrop) => Box::new(routing_fn_drop()),
            Some(RoutingOptions::PredictorBinary) => {
                unimplemented!("PredictorBinary not implemented")
            }
            Some(RoutingOptions::PredictorProbabilistic) => {
                unimplemented!("PredictorProbabilistic not implemented")
            }
            None => Box::new(routing_fn_static(
                RoutingOptions::AlwaysBig,
                global_budget_clone.clone(),
            )),
        }));

    function_lookup.insert("routing_fn".into(), routing_udf);

    let budget_clone = Arc::clone(&global_money_budget);
    let tiny_cost_queue_clone = Arc::clone(&tiny_cost_queue);
    let small_cost_queue_clone = Arc::clone(&small_cost_queue);
    let big_cost_queue_clone = Arc::clone(&big_cost_queue);
    let gpt_4o_fn = FunctionKinds::ComputationExpressionUdf(Box::new(move || {
        let global_money_budget = Arc::clone(&budget_clone);
        let tiny_cost_queue = Arc::clone(&tiny_cost_queue_clone);
        let small_cost_queue = Arc::clone(&small_cost_queue_clone);
        let big_cost_queue = Arc::clone(&big_cost_queue_clone);
        let model_kind = LanguageModel::Gpt4o;
        let by_id_lookup = results_lookup_by_id(
            model_kind,
            global_money_budget,
            tiny_cost_queue,
            small_cost_queue,
            big_cost_queue,
        );
        Box::new(move |items| by_id_lookup(items))
    }));
    function_lookup.insert("gpt-4o".into(), gpt_4o_fn);

    let budget_clone = Arc::clone(&global_money_budget);
    let tiny_cost_queue_clone = Arc::clone(&tiny_cost_queue);
    let small_cost_queue_clone = Arc::clone(&small_cost_queue);
    let big_cost_queue_clone = Arc::clone(&big_cost_queue);
    let gemini2_fn = FunctionKinds::ComputationExpressionUdf(Box::new(move || {
        let global_money_budget = Arc::clone(&budget_clone);
        let tiny_cost_queue = Arc::clone(&tiny_cost_queue_clone);
        let small_cost_queue = Arc::clone(&small_cost_queue_clone);
        let big_cost_queue = Arc::clone(&big_cost_queue_clone);
        let model_kind = LanguageModel::Gemini_2_0_flash;
        let by_id_lookup = results_lookup_by_id(
            model_kind,
            global_money_budget,
            tiny_cost_queue,
            small_cost_queue,
            big_cost_queue,
        );
        Box::new(move |items| by_id_lookup(items))
    }));
    function_lookup.insert("gemini-2.0-flash".into(), gemini2_fn);

    let budget_clone = Arc::clone(&global_money_budget);
    let tiny_cost_queue_clone = Arc::clone(&tiny_cost_queue);
    let small_cost_queue_clone = Arc::clone(&small_cost_queue);
    let big_cost_queue_clone = Arc::clone(&big_cost_queue);
    let gemini1_fn = FunctionKinds::ComputationExpressionUdf(Box::new(move || {
        let global_money_budget = Arc::clone(&budget_clone);
        let tiny_cost_queue = Arc::clone(&tiny_cost_queue_clone);
        let small_cost_queue = Arc::clone(&small_cost_queue_clone);
        let big_cost_queue = Arc::clone(&big_cost_queue_clone);
        let model_kind = LanguageModel::Gemini_1_5_flash_8b;
        let by_id_lookup = results_lookup_by_id(
            model_kind,
            global_money_budget,
            tiny_cost_queue,
            small_cost_queue,
            big_cost_queue,
        );
        Box::new(move |items| by_id_lookup(items))
    }));
    function_lookup.insert("gemini-1.5-flash-8b".into(), gemini1_fn);

    let merge_info = Arc::new(std::sync::Mutex::<Vec<MergeInfo>>::new(Vec::new()));
    let merge_info_clone = Arc::clone(&merge_info);

    let route_feedback_sender_clone = route_feedback_sender.clone();
    let merge_callback_one_fn: FunctionKinds =
        FunctionKinds::MergeCallbackUdf(Box::new(move || {
            let merge_info = Arc::clone(&merge_info_clone);
            let route_feedback_sender_clone = route_feedback_sender_clone.clone();
            let my_pipeline_id = 1;
            Box::new(move |tuple| {
                trace!("tiny merge callback received tuple {}", tuple.id());
                if let Err(e) = route_feedback_sender_clone.send(vec![(
                    tuple.id() as _,
                    my_pipeline_id,
                    Instant::now(),
                )]) {
                    error!("failed to send feedback to routing: {e}");
                }
                // TODO: push the merge info and then write it to a file if the flush threshold has been reached
                // record_merged_data(tuple, &mut *merge_info.lock().unwrap(), my_pipeline_id);
            })
        }));
    function_lookup.insert("merge_callback_one".into(), merge_callback_one_fn);

    let merge_info_clone = Arc::clone(&merge_info);
    let route_feedback_sender_clone = route_feedback_sender.clone();
    let merge_callback_two_fn: FunctionKinds =
        FunctionKinds::MergeCallbackUdf(Box::new(move || {
            let merge_info = Arc::clone(&merge_info_clone);
            let my_pipeline_id = 2;
            let route_feedback_sender = route_feedback_sender_clone.clone();
            Box::new(move |tuple| {
                trace!("small merge callback received tuple {}", tuple.id());
                if let Err(e) = route_feedback_sender.send(vec![(
                    tuple.id() as _,
                    my_pipeline_id,
                    Instant::now(),
                )]) {
                    error!("failed to send feedback to routing: {e}");
                }
                // TODO: push the merge info and then write it to a file if the flush threshold has been reached
                // record_merged_data(tuple, &mut *merge_info.lock().unwrap(), my_pipeline_id);
            })
        }));
    function_lookup.insert("merge_callback_two".into(), merge_callback_two_fn);

    let merge_info_clone = Arc::clone(&merge_info);
    let merge_callback_three_fn: FunctionKinds =
        FunctionKinds::MergeCallbackUdf(Box::new(move || {
            let merge_info = Arc::clone(&merge_info_clone);
            let my_pipeline_id = 3;
            let route_feedback_sender_clone = route_feedback_sender.clone();
            Box::new(move |tuple| {
                trace!("big merge callback received tuple {}", tuple.id());
                if let Err(e) = route_feedback_sender_clone.send(vec![(
                    tuple.id() as _,
                    my_pipeline_id,
                    Instant::now(),
                )]) {
                    error!("failed to send feedback to routing: {e}");
                }
                // TODO: push the merge info and then write it to a file if the flush threshold has been reached
                // record_merged_data(tuple, &mut *merge_info.lock().unwrap(), my_pipeline_id);
            })
        }));
    function_lookup.insert("merge_callback_three".into(), merge_callback_three_fn);

    let always_true_join = FunctionKinds::JoinFilterUdf(|_t1, _t2| (true));
    function_lookup.insert("always_true_join".into(), always_true_join);

    let is_yes_udf = FunctionKinds::ComputationExpressionUdf(Box::new(|| {
        let f = is_yes_udf;
        Box::new(f)
    }));
    function_lookup.insert("is_yes".into(), is_yes_udf);

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
    let log_udf_items_received_ending = Arc::clone(&log_udf_items_received);
    let all_items_read_logger = Arc::clone(&all_items_produced_counter);
    // continue as long as we haven't received the max amt and as long as the sequences aren't done

    let condition_bg_cutoff = Clone::clone(&max_item_condition);

    let log_udf: UdfBolt = UdfBolt {
        id: operators.len(),
        child: operators.last().unwrap().get_id(),
        parent: None,
        process: Arc::new(move |tuple| {
            let tuple_id = tuple.id();
            trace!(
                "log udf tuple {tuple_id} has fields: {:?}",
                tuple.keys().collect::<Vec<_>>()
            );
            let Some(row_id) = tuple.get("question_index") else {
                error!("no question index found in tuple");
                return vec![];
            };
            let Some(row_id) = row_id.as_integer() else {
                error!("question index not an integer");
                return vec![];
            };
            let Some(is_yes) = tuple.get("is_yes(llm_response)") else {
                error!("no is_yes found in tuple");
                return vec![];
            };
            let Some(is_yes) = is_yes.as_bool() else {
                error!("is_yes not a bool");
                return vec![];
            };
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let tuple_age = now_ns - tuple.unix_time_created_ns();
            let tuple_age_ms = tuple_age as f64 / 1_000_000.0;
            info!(
                "received tuple {} after {tuple_age_ms:.2}ms which is row {} with is_yes(llm_response)={}",
                tuple_id, row_id, is_yes
            );

            #[allow(unused_labels)]
            'log_question_answser: {
                let log_location = "receive_final_label".to_raw_key();
                let aux_data = [(
                    "is_yes".to_raw_key(),
                    LimitedHabValue::Integer(is_yes as u8 as _),
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

            log_udf_items_received_callback.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if log_udf_items_received_callback.load(std::sync::atomic::Ordering::SeqCst)
                >= max_total_samples.unwrap_or(usize::MAX)
            {
                warn!("log udf received the max items, notifying condition");
                let _ = max_item_condition.send(true);
            }
            vec![tuple]
        }),
    };

    let expected_items = max_total_samples.unwrap_or(TOTAL_OBJS - SKIP_AMOUNT);
    let _budget_insertion_future = background_rt.spawn(budget_insertion_future);
    let _background_cutoff = background_rt.spawn(async move {
        let delay_amount = 10_000;
        tokio::time::sleep(Duration::from_millis(delay_amount)).await;
        let condition_start_time = Instant::now();
        loop {
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
    Ok(())
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub(crate) struct BoolQEntry {
    pub(crate) title: String,
    pub(crate) question: String,
    pub(crate) answer: bool,
    pub(crate) passage: String,
}

async fn question_source(
    question_jsonl_file: String,
    question_embedding_file: String,
    global_money_budget: Arc<Mutex<f64>>,
    budget_per_item: f64,
    initial_delay_ms: u64,
    // input_delay_micros: u64,
    input_delay_micros: Delay,
    batch_size: usize,
    skip_amount: usize,
    overall_limit: usize,
    max_item_condition: watch::Sender<bool>,
    mut stop_rx: watch::Receiver<bool>,
    all_items_produced_counter: Arc<AtomicUsize>,
) -> impl 'static + Send + Sync + Stream<Item = Vec<Tuple>> {
    use tokio::io::AsyncBufReadExt;
    // use tokio_stream::StreamExt;
    use futures::StreamExt;
    use ndarray::{s, Array2};
    let sequences: Array2<f64> = ndarray_npy::read_npy(question_embedding_file).unwrap();
    let sequences: Array2<f32> = sequences.mapv(|v| v as f32);
    let mut lines = std::io::BufReader::new(std::fs::File::open(&question_jsonl_file).unwrap());
    let mut buffer = String::new();
    let line_channel = tokio::sync::mpsc::channel(1000);
    let line_insertion_future = async move {
        tokio::time::sleep(Duration::from_millis(initial_delay_ms)).await;
        let mut input_delay_micros = input_delay_micros;
        let mut current_delay = input_delay_micros.starting_delay();
        let mut row_id = 0;
        loop {
            buffer.clear();
            let _bytes_read = match lines.read_line(&mut buffer) {
                Ok(0) => {
                    info!("reached end of file, breaking");
                    // immediately close the channel so that the stream can finish
                    drop(line_channel.0);
                    break;
                }
                Err(e) => {
                    error!("failed to read line: {e}");
                    break;
                }
                Ok(b) => b,
            };

            let line = buffer.trim();
            if line.is_empty() {
                continue;
            }
            let BoolQEntry {
                title,
                question,
                answer,
                passage,
            } = match serde_json::from_str::<BoolQEntry>(line) {
                Ok(entry) => entry,
                Err(e) => {
                    error!("failed to parse line as BoolQEntry: {e}");
                    continue;
                }
            };
            let mut tuple = get_tuple();
            let question_index = row_id;
            row_id += 1;
            tuple.insert("question_index".into(), HabValue::Integer(question_index));
            tuple.insert("title".into(), HabValue::String(HabString::from(title)));
            tuple.insert(
                "question".into(),
                HabValue::String(HabString::from(question)),
            );
            tuple.insert("answer".into(), HabValue::Bool(answer));
            tuple.insert("passage".into(), HabValue::String(HabString::from(passage)));
            tuple.insert(
                "embedding".into(),
                HabValue::IntBuffer(bytemuck::cast_vec(
                    sequences.slice(s![question_index as usize, ..]).to_vec(),
                )),
            );
            if let Err(e) = line_channel.0.send(tuple).await {
                error!("failed to send tuple to channel: {e}");
                break;
            }
            let time_to_sleep = current_delay as u64;
            current_delay = input_delay_micros.next_delay();
            if time_to_sleep > 0 {
                tokio::time::sleep(Duration::from_micros(time_to_sleep)).await;
            }
        }
    };
    tokio::spawn(line_insertion_future);
    let tuple_stream = {
        use tokio_stream::wrappers::ReceiverStream;
        ReceiverStream::new(line_channel.1)
    };

    let items_produced_after_skip_and_take = Arc::clone(&all_items_produced_counter);
    let tuple_stream = tuple_stream
        .skip(skip_amount)
        .take(overall_limit)
        .inspect(move |_| {
            // this was removed in favor of using the more versatile fractional insertion in the background
            let _items_produced = items_produced_after_skip_and_take
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // *global_money_budget.lock().unwrap() += budget_per_item;
        });
    let tuple_stream = {
        use tokio_stream::StreamExt;
        tuple_stream
            // .throttle(Duration::from_micros(input_delay_micros))
            .chunks_timeout(batch_size, Duration::from_millis(1000))
    };
    // after everything is done, we notify the rest of the query
    let tuple_stream = tuple_stream.chain(futures::stream::once(async move {
        let items_produced = all_items_produced_counter.load(std::sync::atomic::Ordering::SeqCst);
        info!("question source finished after producing {items_produced} items");
        if items_produced >= overall_limit {
            tokio::time::sleep(Duration::from_millis(5000)).await;
            max_item_condition.send(true).unwrap();
        } else {
            warn!("question source finished but not all items were produced as expected. we produced {items_produced} items but expected up to {overall_limit}");
        }
        vec![]
    }));
    futures::stream::once(async move {
        tokio::time::sleep(Duration::from_millis(initial_delay_ms)).await;
        vec![]
    })
    .chain(tuple_stream)
    .filter(|v| std::future::ready(!v.is_empty()))
    .take_until(async move { stop_rx.changed().await })
}

// def finalize_question(original_question, helper_documents, model):
//   evidence_string = ""
//   for evidence in helper_documents:
//     evidence_string += evidence + "\n"

//   # prompt = f"Imagine you are reading an article to answer a question based on it. You are trying to answer the question \"{original_question}\" In order to do this please follow the instructions: {instructions}\n I will give evidence after the next time the word \"now\" is used, and then I will ask the question again on a new line after the word DELIMITER. Evidence starts now:\n\n{evidence_string}\nDELIMITER\n{original_question}\nPlease begin your answer with yes or no."
//   system_prompt = "You are an assistant for question-answering tasks. "\
//     "Use the following pieces of retrieved context to answer "\
//     "the question. Answer YES or NO. Then on the next line say REASON. Then on the next line state the reasoning behind your YES or NO answer."\
//     "\n\n"\
//     f"{evidence_string}"

//   # print(f"sending final prompt of length {len(prompt)}")
//   # print(system_prompt)

//   # Generate a response using the prompt
//   response = client.chat.completions.create(
//       model=model,
//       messages=[
//           {"role": "system", "content": system_prompt},
//           {"role": "user", "content": f"{original_question}?"}
//       ]
//     )

//   return response

fn client_chat_completions_create(system_prompt: String, original_question: &str) -> String {
    system_prompt + "\n\n" + original_question
}

fn finalize_question(mut args: Vec<&HabValue>) -> HabValue {
    let (passages, question) = (args.pop(), args.pop());
    let passages = passages
        .as_ref()
        .expect("passages not found")
        .as_list()
        .expect("passages not a list")
        .iter()
        .map(|v| &**v.as_string().expect("passage not a string"))
        .collect::<Vec<_>>();
    let original_question = question
        .as_ref()
        .expect("question not found")
        .as_string()
        .expect("question not a string");
    let evidence_string = passages.join("\n");

    let system_prompt = format!("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Answer YES or NO. Then on the next line say REASON. Then on the next line state the reasoning behind your YES or NO answer.\n\n{}", evidence_string);
    let response = client_chat_completions_create(system_prompt, original_question);

    HabValue::String(HabString::from(response))
}

const PYTHON_PUNCTUATION: &[char] = &[
    '+', '"', '|', '/', ']', '!', '{', '@', '~', '*', '>', '}', '[', '$', '(', '?', ',', '-', '=',
    '%', '<', '`', '#', '_', '&', '\\', '\'', '.', '^', ')', ':', ';',
];

fn punctuations(s: &str) -> usize {
    s.chars().filter(|c| PYTHON_PUNCTUATION.contains(c)).count()
}

fn sentence_count(s: &str) -> usize {
    s.split('.').count()
}

fn first_and_mean_distance<List>(set_distances: &[List]) -> Option<Vec<(f64, f64)>>
where
    for<'a> &'a List: IntoIterator<Item = &'a f64>,
{
    let mut results = Vec::with_capacity(set_distances.len());
    for distances in set_distances {
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut distances = distances.into_iter().copied();
        let Some(first) = distances.next() else {
            error!("empty distance list on index {}", results.len());
            return None;
        };
        sum += first;
        for distance in distances {
            sum += distance;
            count += 1;
        }
        results.push((first, sum / count as f64));
    }
    Some(results)
}

// 'name',
// 'input_tokens',
// 'output_tokens',
// 'total_tokens',
// 'response_texts',
// 'times'

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(transparent)]
struct GptResultsData(Vec<ModelResults>);

#[derive(Debug, Deserialize, Clone, Serialize)]
struct ModelResults {
    name: String,
    input_tokens: Vec<usize>,
    output_tokens: Vec<usize>,
    total_tokens: Vec<usize>,
    times: Vec<f64>,
    response_texts: Vec<String>,
}

fn fetch_llm_results_data() -> String {
    let path = "./viable_model_result_dicts.json";
    match std::fs::read_to_string(path) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to read llm_results json file {path:?}: {e:?}");
            String::new()
        }
    }
}

fn llm_results_data() -> GptResultsData {
    serde_json::from_str(&fetch_llm_results_data()).unwrap()
}

/*
[
  {
    "name": "gemini-1.5-flash-8b",
    "training_accuracy": 0.6295843520782396,
    "when_confident": 0.8562628336755647,
    "when_not_confident": 0.7039274924471299,
    "fields": [
      "byte_lengths",
      "sentence_counts",
      "first_matched_distances",
      "mean_matched_distances"
    ],
    "coefficients": [
      0.016442131470125246,
      -0.8101951190701397,
      -1.4904492495211734,
      -0.09041481105976343
    ],
    "intercept": 1.247539374833335
  },
  {
    "name": "gemini-2.0-flash",
    "training_accuracy": 0.5317848410757946,
    "when_confident": 0.8595238095238096,
    "when_not_confident": 0.8140703517587939,
    "fields": [
      "byte_lengths",
      "sentence_counts",
      "first_matched_distances",
      "mean_matched_distances"
    ],
    "coefficients": [
      0.014122972062964437,
      0.3625335219263725,
      -0.4891417474238937,
      -0.8224428901970617
    ],
    "intercept": 0.18796500006420394
  },
  {
    "name": "gpt-4o",
    "training_accuracy": 0.6149144254278729,
    "when_confident": 0.8931451612903226,
    "when_not_confident": 0.8136645962732919,
    "fields": [
      "byte_lengths",
      "sentence_counts",
      "first_matched_distances",
      "mean_matched_distances"
    ],
    "coefficients": [
      0.002638661062853303,
      0.3484577078528963,
      -1.1362603939916949,
      0.09143580113191349
    ],
    "intercept": 0.24750501676761358
  }
]
 */

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(transparent)]
struct LlmAccuracyBucketPredictors([LlmAccuracyBucketFunction; 3]);

#[derive(Debug, Deserialize, Clone, Serialize)]
struct LlmAccuracyBucketFunction {
    name: String,
    coefficients: Vec<f64>,
    intercept: f64,
}

fn fetch_llm_accuracy_bucket_predictors() -> String {
    let path = "./llm_accuracy_predictors.json";
    match std::fs::read_to_string(path) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to read bucket weights json file {path:?}: {e:?}");
            String::new()
        }
    }
}

fn llm_accuracy_bucket_predictors() -> LlmAccuracyBucketPredictors {
    serde_json::from_str(&fetch_llm_accuracy_bucket_predictors()).unwrap()
}

/**
{
    "gpt-4o": {
        "weights": [
            0.0016330926568945139,
            -0.00493868529643537,
            0.0008267837195833584,
            2.868087079775563e-05,
            -0.002487169327915593
        ],
        "intercept": 0.033421363487353535,
        "fields": [
            "first_distance",
            "mean_distance",
            "punctuations",
            "byte_lengths",
            "sentence_counts"
        ]
    },
    "gpt-4o-mini": {
        "weights": [
            0.0016330926568945139,
            -0.00493868529643537,
            0.0008267837195833584,
            2.868087079775563e-05,
            -0.002487169327915593
        ],
        "intercept": 0.033421363487353535,
        "fields": [
            "first_distance",
            "mean_distance",
            "punctuations",
            "byte_lengths",
            "sentence_counts"
        ]
    }
}
 */
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(transparent)]
struct CostPredictor([CostPredictorData; 3]);

#[derive(Debug, Deserialize, Clone, Serialize)]
struct CostPredictorData {
    coefficients: Vec<f64>,
    intercept: f64,
}

fn fetch_cost_predictor() -> String {
    let path = "./llm_cost_predictors.json";
    match std::fs::read_to_string(path) {
        Ok(v) => v,
        Err(e) => {
            error!("failed to read gpt cost predictor weight json file {path:?}: {e:?}");
            String::new()
        }
    }
}

fn cost_predictor() -> CostPredictor {
    serde_json::from_str(&fetch_cost_predictor()).unwrap()
}

fn logistic_fn(input: f64) -> f64 {
    1.0 / (1.0 + (-input).exp())
}

/*
Model: gemini-1.5-flash-8b, training step R-squared: 0.08691163293682425
Model: gemini-1.5-flash-8b, training step mean absolute error: 2.9382169895816764e-06
Model: gemini-1.5-flash-8b, training step mean relative error: 0.13664164524759148
Model: gemini-1.5-flash-8b, validation step R-squared: 0.05931801518223323
Model: gemini-1.5-flash-8b, validation step mean absolute error: 2.9316877952709583e-06
Model: gemini-1.5-flash-8b, validation step mean relative error: 0.1365283940940718
[ 9.07609451e-09 -2.66059280e-06  2.12286578e-06 -5.01474186e-06]

Model: gemini-2.0-flash, training step R-squared: 0.021720357869008167
Model: gemini-2.0-flash, training step mean absolute error: 7.458047436468947e-06
Model: gemini-2.0-flash, training step mean relative error: 0.12152160484026377
Model: gemini-2.0-flash, validation step R-squared: 0.06420058310002141
Model: gemini-2.0-flash, validation step mean absolute error: 7.428642956310901e-06
Model: gemini-2.0-flash, validation step mean relative error: 0.12439435192530222
[-9.43008807e-09 -2.92172360e-06  6.77580093e-06 -1.42992450e-05]

Model: gpt-4o, training step R-squared: -0.0006234701621237182
Model: gpt-4o, training step mean absolute error: 0.00019748868581200243
Model: gpt-4o, training step mean relative error: 0.12486663832028629
Model: gpt-4o, validation step R-squared: 0.020840114332444037
Model: gpt-4o, validation step mean absolute error: 0.00021015547640067323
Model: gpt-4o, validation step mean relative error: 0.13865331681265947
[ 1.56697988e-06 -4.97628355e-05  7.93434057e-05 -2.06454156e-04]

[
  {
    "name": "gemini-1.5-flash-8b",
    "coefficient": [
      9.076094510546505e-09,
      -2.6605928037475174e-06,
      2.1228657831025186e-06,
      -5.014741855572361e-06
    ],
    "intercept": 2.858242755934704e-05
  },
  {
    "name": "gemini-2.0-flash",
    "coefficient": [
      -9.430088073603097e-09,
      -2.9217236015707662e-06,
      6.775800930202539e-06,
      -1.4299245011637233e-05
    ],
    "intercept": 7.376417769896933e-05
  },
  {
    "name": "gpt-4o",
    "coefficient": [
      1.5669798801862512e-06,
      -4.9762835493103496e-05,
      7.934340570616295e-05,
      -0.00020645415641237704
    ],
    "intercept": 0.0016541700583714023
  }
]
 */

fn predict_bucket(
    bucket_predictor: &LlmAccuracyBucketPredictors,
    cost_predictor: &CostPredictor,
    first_distance: f64,
    mean_distance: f64,
    byte_lengths: f64,
    punctuations: f64,
    sentence_counts: f64,
) -> BinInfo<LlmInferenceCategory> {
    let mut confidences = [0.0; 3];
    let mut is_confident = [false; 3];
    for (i, bucket_predictor) in bucket_predictor.0.iter().enumerate() {
        let mut bucket_logit_sum = bucket_predictor.intercept;
        bucket_logit_sum += bucket_predictor.coefficients[0] * byte_lengths;
        bucket_logit_sum += bucket_predictor.coefficients[1] * sentence_counts;
        bucket_logit_sum += bucket_predictor.coefficients[2] * first_distance;
        bucket_logit_sum += bucket_predictor.coefficients[3] * mean_distance;
        let prob = logistic_fn(bucket_logit_sum);
        confidences[i] = prob;
        is_confident[i] = prob > 0.5;
        info!(
            "model {} predicted {} (easy class had a probability of {:.3})",
            &bucket_predictor.name,
            {
                if prob > 0.5 {
                    "easy"
                } else {
                    "hard"
                }
            },
            prob
        );
    }
    let mut baseline_bin = match is_confident {
        [false, false, false] => BIN_000,
        [false, false, true] => BIN_001,
        [false, true, false] => BIN_010,
        [false, true, true] => BIN_011,
        [true, false, false] => BIN_100,
        [true, false, true] => BIN_101,
        [true, true, false] => BIN_110,
        [true, true, true] => BIN_111,
    };
    let baseline_costs: &mut [f64] = baseline_bin.costs.as_mut();
    for (cost, predictor) in baseline_costs.iter_mut().skip(1).zip(&cost_predictor.0) {
        let mut cost_estimate_sum = predictor.intercept;
        cost_estimate_sum += predictor.coefficients[0] * byte_lengths;
        cost_estimate_sum += predictor.coefficients[1] * sentence_counts;
        cost_estimate_sum += predictor.coefficients[2] * first_distance;
        cost_estimate_sum += predictor.coefficients[3] * mean_distance;
        *cost = cost_estimate_sum
    }

    baseline_bin
}

fn predict_bucket_dynamic(
    bucket_predictor: &watershed_shared::preclassifier_lang::RealBucketLookup,
    // cost_predictor: &CostPredictor,
    first_distance: f64,
    mean_distance: f64,
    byte_lengths: f64,
    punctuations: f64,
    sentence_counts: f64,
) -> BinInfo<watershed_shared::preclassifier_lang::PreclassifierLangClass> {
    watershed_shared::preclassifier_lang::map_inputs_to_bucket(
        &[
            byte_lengths as _,
            sentence_counts as _,
            first_distance as _,
            mean_distance as _,
        ],
        bucket_predictor,
    )
}

fn make_bin_and_adjuster_function(
    bucket_predictor: watershed_shared::preclassifier_lang::RealBucketLookup,
) -> (
    impl 'static + Fn(&Tuple) -> BinInfo<watershed_shared::preclassifier_lang::PreclassifierLangClass>,
    impl 'static
        + Send
        + FnMut(&mut [BinInfo<watershed_shared::preclassifier_lang::PreclassifierLangClass>]),
) {
    const HISTORY_SIZE: usize = 50;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    let bin_history = Arc::new(Mutex::new(VecDeque::with_capacity(HISTORY_SIZE)));
    let binning_fn = {
        let bin_history = bin_history.clone();
        move |tuple: &Tuple| {
            debug!("binning tuple {}", tuple.id());
            let distances = {
                let map_fn = |opt: Option<&HabValue>| -> Option<Vec<f64>> {
                    let dist_list = opt?;
                    let dist_list = dist_list.as_list()?;
                    dist_list
                        .iter()
                        .map(|v| v.as_float().map(|f| f.into_inner()))
                        .collect()
                };
                map_fn(tuple.get("match_distances")).unwrap_or_else(|| {
                    error!("no distances not found");
                    vec![0.0]
                })
            };
            let (first_distance, mean_distance) = first_and_mean_distance(&[distances])
                .map(|v| v[0])
                .unwrap_or_else(|| {
                    error!("failed to calculate first and mean distance");
                    (0.0, 0.0)
                });

            let Some(question) = tuple.get("question") else {
                error!("question not found");
                return bucket_predictor.buckets[0].clone();
            };
            let Some(question) = question.as_string() else {
                error!("question not a string");
                return bucket_predictor.buckets[0].clone();
            };
            let byte_lengths = question.len() as f64;
            let punctuations = punctuations(question) as f64;
            let sentence_counts = sentence_count(question) as f64;
            let bin = predict_bucket_dynamic(
                &bucket_predictor,
                first_distance,
                mean_distance,
                byte_lengths,
                punctuations,
                sentence_counts,
            );
            bin_history.lock().unwrap().push_back(bin.clone());
            if bin_history.lock().unwrap().len() >= HISTORY_SIZE {
                bin_history.lock().unwrap().pop_front();
            }
            bin
        }
    };

    // let mut tiny_costs = Vec::with_capacity(HISTORY_SIZE);
    // let mut small_costs = Vec::with_capacity(HISTORY_SIZE);
    // let mut big_costs = Vec::with_capacity(HISTORY_SIZE);

    // let adjust_fn = move |bins: &mut [BinInfo<LlmInferenceCategory>]| {
    //     let history = bin_history.lock().unwrap();
    //     if history.is_empty() {
    //         return;
    //     }
    //     if history.len() == 1 {
    //         let old_bin = history[0].clone();
    //         for bin in bins.iter_mut() {
    //             bin.costs = old_bin.costs.clone();
    //         }
    //         return;
    //     }
    //     let (tiny_costs, small_costs, big_costs) = {
    //         tiny_costs.clear();
    //         small_costs.clear();
    //         big_costs.clear();
    //         for bin in history.iter() {
    //             let costs = &bin.costs;
    //             tiny_costs.push(costs[1]);
    //             small_costs.push(costs[2]);
    //             big_costs.push(costs[3]);
    //         }
    //         tiny_costs.sort_by(f64::total_cmp);
    //         small_costs.sort_by(f64::total_cmp);
    //         big_costs.sort_by(f64::total_cmp);
    //         (&mut tiny_costs, &mut small_costs, &mut big_costs)
    //     };
    //     // right now I just select a random previous cost, but there are other ways like
    //     //  - taking some percentile
    //     //  - taking a random number and getting the associated index, using the fractional component to interpolate between two indices
    //     //  - using the mean
    //     use rand::seq::SliceRandom;
    //     let mut rng = rand::thread_rng();
    //     for bin in bins.iter_mut() {
    //         let tiny_cost = tiny_costs.choose(&mut rng).unwrap();
    //         let small_cost = small_costs.choose(&mut rng).unwrap();
    //         let big_cost = big_costs.choose(&mut rng).unwrap();
    //         let costs = bin.costs.as_mut();
    //         costs[1] = *tiny_cost;
    //         costs[2] = *small_cost;
    //         costs[3] = *big_cost;
    //     }
    // };

    // no need to adjust for the dynamic binning
    let adjust_fn =
        |_: &mut [BinInfo<watershed_shared::preclassifier_lang::PreclassifierLangClass>]| {};
    (binning_fn, adjust_fn)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum LlmInferenceCategory {
    Llm000,
    Llm001,
    Llm010,
    Llm011,
    Llm100,
    Llm101,
    Llm110,
    Llm111,
}

impl scheduler::LabelCategory for LlmInferenceCategory {
    fn values() -> impl Iterator<Item = Self> {
        vec![
            Self::Llm000,
            Self::Llm001,
            Self::Llm010,
            Self::Llm011,
            Self::Llm100,
            Self::Llm101,
            Self::Llm110,
            Self::Llm111,
        ]
        .into_iter()
    }
}

// Model: gemini-1.5-flash-8b, training step testing Accuracy: 0.5609756097560976
// Model: gemini-1.5-flash-8b, validation step Accuracy: 0.6295843520782396
// Model: gemini-1.5-flash-8b, validation step Accuracy bucket A (preclassifier is confident): 0.8562628336755647 (59.54% of items)
// Model: gemini-1.5-flash-8b, validation step Accuracy bucket B (preclassifier is not confident): 0.7039274924471299 (40.46% of items)
const GEMINI_1_5_FLASH_8B_HIGH_CONFIDENCE_REWARD: f64 = 0.8562628336755647;
const GEMINI_1_5_FLASH_8B_LOW_CONFIDENCE_REWARD: f64 = 0.7039274924471299;

// Model: gemini-2.0-flash, training step testing Accuracy: 0.5914634146341463
// Model: gemini-2.0-flash, validation step Accuracy: 0.5317848410757946
// Model: gemini-2.0-flash, validation step Accuracy bucket A (preclassifier is confident): 0.8595238095238096 (51.34% of items)
// Model: gemini-2.0-flash, validation step Accuracy bucket B (preclassifier is not confident): 0.8140703517587939 (48.66% of items)
const GEMINI_2_0_FLASH_HIGH_CONFIDENCE_REWARD: f64 = 0.8595238095238096;
const GEMINI_2_0_FLASH_LOW_CONFIDENCE_REWARD: f64 = 0.8140703517587939;

// Model: gpt-4o, training step testing Accuracy: 0.5914634146341463
// Model: gpt-4o, validation step Accuracy: 0.6149144254278729
// Model: gpt-4o, validation step Accuracy bucket A (preclassifier is confident): 0.8931451612903226 (60.64% of items)
// Model: gpt-4o, validation step Accuracy bucket B (preclassifier is not confident): 0.8136645962732919 (39.36% of items)
const GPT_4O_HIGH_CONFIDENCE_REWARD: f64 = 0.8931451612903226;
const GPT_4O_LOW_CONFIDENCE_REWARD: f64 = 0.8136645962732919;

const TINY_MODEL_DUMMY_COST: f64 = 0.0005;
const SMALL_MODEL_DUMMY_COST: f64 = 0.002;
const BIG_MODEL_DUMMY_COST: f64 = 0.02;

const BIN_000: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm000),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_LOW_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_LOW_CONFIDENCE_REWARD,
        GPT_4O_LOW_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_001: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm001),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_LOW_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_LOW_CONFIDENCE_REWARD,
        GPT_4O_HIGH_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_010: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm010),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_LOW_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_HIGH_CONFIDENCE_REWARD,
        GPT_4O_LOW_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_011: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm011),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_LOW_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_HIGH_CONFIDENCE_REWARD,
        GPT_4O_HIGH_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_100: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm100),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_HIGH_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_LOW_CONFIDENCE_REWARD,
        GPT_4O_LOW_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_101: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm101),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_HIGH_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_LOW_CONFIDENCE_REWARD,
        GPT_4O_HIGH_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_110: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm110),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_HIGH_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_HIGH_CONFIDENCE_REWARD,
        GPT_4O_LOW_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

const BIN_111: BinInfo<LlmInferenceCategory> = BinInfo {
    id: Some(LlmInferenceCategory::Llm111),
    valid_pipelines: ShareableArray::Borrowed(&[0, 1, 2, 3]),
    rewards: ShareableArray::Borrowed(&[
        0.0,
        GEMINI_1_5_FLASH_8B_HIGH_CONFIDENCE_REWARD,
        GEMINI_2_0_FLASH_HIGH_CONFIDENCE_REWARD,
        GPT_4O_HIGH_CONFIDENCE_REWARD,
    ]),
    costs: ShareableArray::Borrowed(&[
        0.0,
        TINY_MODEL_DUMMY_COST,
        SMALL_MODEL_DUMMY_COST,
        BIG_MODEL_DUMMY_COST,
    ]),
};

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
enum LanguageModel {
    Gpt4o,
    // Gpt4oMini,
    #[allow(non_camel_case_types)]
    Gemini_1_5_flash_8b,
    #[allow(non_camel_case_types)]
    Gemini_2_0_flash,
}

// old rates
// costs = {
//     "gpt-4o": {
//         'input_rate': 5 * 10e-6,
//         'output_rate': 15 * 10e-6,
//     },
//     "gpt-4o-mini": {
//         'input_rate': 0.15 * 10e-6,
//         'output_rate': 0.60 * 10e-6,
//     }
// }

// new rates. GPT 4o has gotten cheaper
// costs = {
//     'gemini-1.5-flash-8b': {
//         'input': 0.0375 * 1e-6,
//         'output': 0.15 * 1e-6,
//     },
//     'gemini-2.0-flash': {
//         'input': 0.10 * 1e-6,
//         'output': 0.40 * 1e-6,
//     },
//     'gpt-4o': {
//         'input': 2.50 * 1e-6,
//         'output': 10.00 * 1e-6,
//     }
// }

// const GPT4O_INPUT_RATE: f64 = 5.0 * 10e-6;
// const GPT4O_OUTPUT_RATE: f64 = 15.0 * 10e-6;
// const GPT4OMINI_INPUT_RATE: f64 = 0.15 * 10e-6;
// const GPT4OMINI_OUTPUT_RATE: f64 = 0.60 * 10e-6;

const GPT4O_INPUT_RATE: f64 = 2.50 * 1e-6;
const GPT4O_OUTPUT_RATE: f64 = 10.00 * 1e-6;
const GEMINI_1_5_FLASH_8B_INPUT_RATE: f64 = 0.0375 * 1.0e-6;
const GEMINI_1_5_FLASH_8B_OUTPUT_RATE: f64 = 0.15 * 1.0e-6;
const GEMINI_2_0_FLASH_INPUT_RATE: f64 = 0.10 * 1.0e-6;
const GEMINI_2_0_FLASH_OUTPUT_RATE: f64 = 0.40 * 1.0e-6;
fn compute_cost(model_used: LanguageModel, prompt_tokens: usize, completion_tokens: usize) -> f64 {
    let (inp, out) = match model_used {
        LanguageModel::Gpt4o => (GPT4O_INPUT_RATE, GPT4O_OUTPUT_RATE),
        // LanguageModel::Gpt4oMini => (GPT4OMINI_INPUT_RATE, GPT4OMINI_OUTPUT_RATE),
        LanguageModel::Gemini_1_5_flash_8b => (
            GEMINI_1_5_FLASH_8B_INPUT_RATE,
            GEMINI_1_5_FLASH_8B_OUTPUT_RATE,
        ),
        LanguageModel::Gemini_2_0_flash => {
            (GEMINI_2_0_FLASH_INPUT_RATE, GEMINI_2_0_FLASH_OUTPUT_RATE)
        }
    };
    (inp * prompt_tokens as f64) + (out * completion_tokens as f64)
}

fn results_lookup_by_id(
    model_kind: LanguageModel,
    global_money_budget: Arc<Mutex<f64>>,
    tiny_cost_queue: Arc<Mutex<VecDeque<f64>>>,
    small_cost_queue: Arc<Mutex<VecDeque<f64>>>,
    big_cost_queue: Arc<Mutex<VecDeque<f64>>>,
) -> impl 'static + Send + Sync + Fn(Vec<&HabValue>) -> HabValue {
    let mut results_data = llm_results_data();
    let model_index = match model_kind {
        LanguageModel::Gpt4o => 2,
        LanguageModel::Gemini_2_0_flash => 1,
        LanguageModel::Gemini_1_5_flash_8b => 0,
    };
    let results_data = results_data.0.remove(model_index);
    let cost_queue = match model_kind {
        LanguageModel::Gpt4o => big_cost_queue,
        LanguageModel::Gemini_2_0_flash => small_cost_queue,
        LanguageModel::Gemini_1_5_flash_8b => tiny_cost_queue,
    };
    move |values| {
        debug!("in result lookup received values: {:?}", values);
        let index = values[0].as_integer().expect("index not an integer") as usize;
        let (input_tokens, output_tokens) = (
            results_data.input_tokens[index],
            results_data.output_tokens[index],
        );
        let cost = compute_cost(model_kind, input_tokens, output_tokens);
        debug!(
            "cost for index {} running model {model_kind:?}: {}",
            index, cost
        );
        *global_money_budget.lock().unwrap() -= cost;
        {
            let mut cost_queue = cost_queue.lock().unwrap();
            cost_queue.push_back(cost);
            if cost_queue.len() >= 50 {
                cost_queue.pop_front();
            }
        }
        results_data.response_texts[index].clone().into()
    }
}

fn is_yes(answer: &str) -> bool {
    match answer.get(..3) {
        Some(v) => v.eq_ignore_ascii_case("yes"),
        _ => false,
    }
}

fn is_yes_udf(string: Vec<&HabValue>) -> HabValue {
    let Some(answer) = string[0].as_string() else {
        error!("answer in field \"gpt_response\" was not a string");
        return HabValue::Null;
    };
    HabValue::Bool(is_yes(answer))
}

fn routing_fn_drop() -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize>
{
    move |tuples, senders| {
        debug!(
            "received {:?} tuples in drop routing function",
            tuples.len()
        );
        if let Err(e) = senders[0].send(tuples) {
            error!("failed to send to pipe 0: {e:?}");
        }
        Some(0)
    }
}

fn routing_fn_static(
    routing_option: RoutingOptions,
    global_money_budget: Arc<Mutex<f64>>,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    move |tuples, senders| {
        let budget = *global_money_budget.lock().unwrap();
        if budget < 0.0 {
            warn!("dropping {} tuples due to budget", tuples.len());
            senders[0].send(tuples).unwrap();
            return Some(0);
        }

        let amount = tuples.len();
        for t in tuples {
            let pipe = match routing_option {
                RoutingOptions::AlwaysTiny => 1,
                RoutingOptions::AlwaysSmall => 2,
                RoutingOptions::AlwaysBig => 3,
                _ => {
                    error!("incorrect static routing option {routing_option:?}");
                    continue;
                }
            };
            if let Err(e) = senders[pipe].send(vec![t]) {
                error!("failed to send to pipe {pipe}: {e:?}");
            }
        }
        Some(amount)
    }
}

fn routing_fn_eddies(
    _window_size: usize,
    _deadline_window_ms: u64,
    global_money_budget: Arc<Mutex<f64>>,
    small_cost_queue: Arc<Mutex<VecDeque<f64>>>,
    big_cost_queue: Arc<Mutex<VecDeque<f64>>>,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    move |tuples, senders| {
        let budget = *global_money_budget.lock().unwrap();
        // if we have no budget, we drop all the tuples
        // if this ends up being too much, the budget will be incremented later so it will recover when given enough time
        if budget < 0.0 {
            warn!("dropping {} tuples due to budget", tuples.len());
            senders[0].send(tuples).unwrap();
            return Some(0);
        }

        // minimum cost of any query to GPT-4o-mini
        // used as a bottom limit for the cost so that there can be no division by zero
        const MINIMUM_VALUE: f64 = 0.00045;
        let tiny_costs = small_cost_queue.lock().unwrap().iter().sum::<f64>() + MINIMUM_VALUE;
        let small_costs = small_cost_queue.lock().unwrap().iter().sum::<f64>() + MINIMUM_VALUE;
        let big_costs = big_cost_queue.lock().unwrap().iter().sum::<f64>() + MINIMUM_VALUE;
        // let small_chance = 1.0 - (small_costs / (small_costs + big_costs));
        let total = tiny_costs + small_costs + big_costs;
        let tiny_portion = tiny_costs / total;
        let small_portion = small_costs / total;
        let big_portion = big_costs / total;
        let inverted_tiny_portion = 1.0 - tiny_portion;
        let inverted_small_portion = 1.0 - small_portion;
        let inverted_big_portion = 1.0 - big_portion;
        let inverted_sum = inverted_tiny_portion + inverted_small_portion + inverted_big_portion;
        let small_threshold = inverted_tiny_portion / inverted_sum;
        let big_threshold = small_threshold + inverted_small_portion / inverted_sum;

        let mut rng = rand::thread_rng();
        use rand::Rng;
        let amount = tuples.len();
        let mut tiny_tuples = Vec::with_capacity(amount);
        let mut small_tuples = Vec::with_capacity(amount);
        let mut big_tuples = Vec::with_capacity(amount);
        for t in tuples {
            let chance = rng.gen::<f64>();
            if chance < small_threshold {
                tiny_tuples.push(t);
            } else if chance < big_threshold {
                small_tuples.push(t);
            } else {
                big_tuples.push(t);
            }
        }
        senders[1].send(tiny_tuples).unwrap();
        senders[2].send(small_tuples).unwrap();
        senders[3].send(big_tuples).unwrap();
        Some(amount)
    }
}

static OPTIMAL_MAX_COUNT: AtomicUsize = AtomicUsize::new(5);
static GREEDY_MAX_COUNT: AtomicUsize = AtomicUsize::new(5);

fn aquifer_routing_fn(
    keep_n_history_items: usize,
    global_money_budget: Arc<Mutex<f64>>,
    back_channel: crossbeam::channel::Receiver<Vec<(usize, usize, Instant)>>,
    deadline_ms: u64,
    strategy: scheduler::Strategy,
    bucket_predictor: watershed_shared::preclassifier_lang::RealBucketLookup,
) -> impl 'static + Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize> {
    use scheduler::basic_probability_forecast::PastData;

    let optimal_max_count = OPTIMAL_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let greedy_max_count = GREEDY_MAX_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let discrete_bins = bucket_predictor.buckets.iter().cloned().collect::<Vec<_>>();

    let (binning_fn, _adjust_fn) = make_bin_and_adjuster_function(bucket_predictor);
    use watershed_shared::scheduler::basic_probability_forecast::History;
    let mut history = History::new(keep_n_history_items, back_channel, discrete_bins);
    let forecast_function =
        History::<watershed_shared::preclassifier_lang::PreclassifierLangClass>::forecast_fn();

    move |tuples, senders| {
        let time_of_scheduling = Instant::now();
        let time_of_scheduling_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();

        let t0_keys = tuples[0]
            .keys() /*.map(|k| k.to_string())*/
            .collect::<Vec<_>>();
        debug!("received {:?} tuples in aquifer routing function", t0_keys);

        let budget = *global_money_budget.lock().unwrap();
        // if we have no budget, we drop all the tuples
        // if this ends up being too much, the budget will be incremented later so it will recover when given enough time
        if budget < 0.0 {
            warn!("dropping {} tuples due to budget", tuples.len());
            for t in &tuples {
                let time_created_ns = t.unix_time_created_ns();
                let diff = time_of_scheduling_ns - time_created_ns;
                history.add_past_data(PastData {
                    tuple_id: t.id(),
                    category: binning_fn(&t).id,
                    age_when_scheduling_ns: diff,
                    time_of_creation_ns: time_created_ns,
                    time_of_scheduling,
                    time_merged: time_of_scheduling,
                    time_elapsed_ms: 0.0,
                    pipeline_id: 0,
                });
            }
            senders[0].send(tuples).unwrap();
            return Some(0);
        }

        history.update();
        let ns_per_item = history.fine_ingress_rate_ns_per_item();
        let items_per_deadline = match ns_per_item {
            Some(t) => {
                let items_per_ms = 1.0 / (t / 1_000_000.0);
                let ms_per_deadline = deadline_ms as f64;
                items_per_ms * ms_per_deadline
            }
            None => 1.0,
        };
        // let budget = budget * items_per_deadline as f64;
        let current_tuple = &tuples[0];
        let tuple_id = current_tuple.id();
        let start_scheduling_time = Instant::now();
        let alg_inputs = AlgInputs {
            binning_function: &binning_fn,
            forecast_function: &forecast_function,
            send_function: History::send,
        };

        // use aquifer_scheduler
        let out = scheduler::lookahead_problem_scheduler(
            tuples,
            senders,
            &mut history,
            alg_inputs,
            budget,
            match strategy {
                scheduler::Strategy::Greedy => FutureWindowKind::TimeWithMaximumCount {
                    time_ms: deadline_ms as u128,
                    max_count: greedy_max_count,
                },
                scheduler::Strategy::Optimal => FutureWindowKind::TimeWithMaximumCount {
                    time_ms: deadline_ms as u128,
                    max_count: optimal_max_count,
                },
            },
            strategy,
        );

        let end_scheduling_time = Instant::now();
        let scheduling_duration = end_scheduling_time - start_scheduling_time;
        let scheduling_duration_ms = scheduling_duration.as_secs_f64() * 1_000.0;

        #[allow(unused_labels)]
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

        out
    }
}
