//! # Expression Builder
//!
//! This module contains the definition of various types used in the construction and running of operator pipelines.
//!

use std::collections::BTreeMap;

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::atomic::{self, AtomicUsize};
use std::sync::Arc;

use dashmap::{DashMap, DashSet};
#[allow(unused)]
use log::{debug, error, info, warn};

use serde::{Deserialize, Serialize};
use tap::Tap;

use crate::basic_pooling::{get_tuple, CollectTupleVec, CollectValueVec};
use crate::caching::StrToKey;
use crate::global_logger::LimitedHabValue;
// operators
use crate::operators::{ChannelRouter, ChannelSpout, Merge, MergeSpout};
use crate::operators::{Join, Project, Select};

use crate::operators::{UdfBolt, UdfSpout};

use crate::operators::{AsyncPythonRemoteTaskState, PythonInlineUdf, PythonRemoteUdf};

use crate::operators::{AggregationResult, BuiltinAggregator, DeriveValue};

use crate::operators::{ChromaJoin, ChromaJoinKind, DummyBolt, GroupBy, Union};

use crate::{HabString, HabValue, Operator, Tuple};

use crate::chroma_utils::DistanceMetric;
use crate::expression::{
    evaluate_computation_expression, BinaryOp, ComputationExpression, ComputationLiteral, UnaryOp,
};

use crate::devec::DeVec;

use crate::Queue;

// TODO: change this to have the actual fields needed to make it work
//   for now it just exists as a placeholder for the construction implementation
pub enum PhysicalOperator {
    Project(Project),
    Select(Select),
    Join(Join),
    GroupBy(GroupBy),
    DeriveValue(DeriveValue),
    ChromaJoin(ChromaJoin),
    ChannelRouter(ChannelRouter),
    ChannelSpout(ChannelSpout),
    Merge(Merge),
    MergeSpout(MergeSpout),
    UserDefinedFunction(UdfBolt),
    UserDefinedSource(UdfSpout),
    PythonInlineFunction(PythonInlineUdf),
    PythonRemoteFunction(PythonRemoteUdf),
    Union(Union),
    __PLACEHOLDER__(DummyBolt),
}

impl Operator for PhysicalOperator {
    fn get_id(&self) -> usize {
        match self {
            PhysicalOperator::Project(op) => op.id,
            PhysicalOperator::Select(op) => op.id,
            PhysicalOperator::Join(op) => op.id,
            PhysicalOperator::GroupBy(op) => op.id,
            PhysicalOperator::DeriveValue(op) => op.id,
            PhysicalOperator::ChromaJoin(op) => op.id,
            PhysicalOperator::ChannelRouter(op) => op.id,
            PhysicalOperator::ChannelSpout(op) => op.id,
            PhysicalOperator::Merge(op) => op.id,
            PhysicalOperator::MergeSpout(op) => op.id,
            PhysicalOperator::UserDefinedFunction(op) => op.id,
            PhysicalOperator::UserDefinedSource(op) => op.id,
            PhysicalOperator::PythonInlineFunction(op) => op.id,
            PhysicalOperator::PythonRemoteFunction(op) => op.id,
            PhysicalOperator::Union(op) => op.id,
            PhysicalOperator::__PLACEHOLDER__(op) => op.0,
        }
    }

    fn add_parent(&mut self, parent: usize) {
        match self {
            PhysicalOperator::Project(op) => op.add_parent(parent),
            PhysicalOperator::Select(op) => op.add_parent(parent),
            PhysicalOperator::Join(op) => op.add_parent(parent),
            PhysicalOperator::GroupBy(op) => op.add_parent(parent),
            PhysicalOperator::DeriveValue(op) => op.add_parent(parent),
            PhysicalOperator::ChromaJoin(op) => op.add_parent(parent),
            PhysicalOperator::ChannelRouter(op) => op.add_parent(parent),
            PhysicalOperator::ChannelSpout(op) => op.add_parent(parent),
            PhysicalOperator::Merge(op) => op.add_parent(parent),
            PhysicalOperator::MergeSpout(op) => op.add_parent(parent),
            PhysicalOperator::UserDefinedFunction(op) => op.add_parent(parent),
            PhysicalOperator::UserDefinedSource(op) => op.add_parent(parent),
            PhysicalOperator::PythonInlineFunction(op) => {
                // unimplemented!("no inline python functions at this time")
                op.add_parent(parent)
            }
            PhysicalOperator::PythonRemoteFunction(op) => op.add_parent(parent),
            PhysicalOperator::Union(op) => op.add_parent(parent),
            PhysicalOperator::__PLACEHOLDER__(op) => op.add_parent(parent),
        }
    }

    fn initialize(&mut self) {
        match self {
            PhysicalOperator::Project(op) => op.initialize(),
            PhysicalOperator::Select(op) => op.initialize(),
            PhysicalOperator::Join(op) => op.initialize(),
            PhysicalOperator::GroupBy(op) => op.initialize(),
            PhysicalOperator::DeriveValue(op) => op.initialize(),
            PhysicalOperator::ChromaJoin(op) => op.initialize(),
            PhysicalOperator::ChannelRouter(op) => op.initialize(),
            PhysicalOperator::ChannelSpout(op) => op.initialize(),
            PhysicalOperator::Merge(op) => op.initialize(),
            PhysicalOperator::MergeSpout(op) => op.initialize(),
            PhysicalOperator::UserDefinedFunction(op) => op.initialize(),
            PhysicalOperator::UserDefinedSource(op) => op.initialize(),
            PhysicalOperator::PythonInlineFunction(op) => op.initialize(),
            PhysicalOperator::PythonRemoteFunction(op) => op.initialize(),
            PhysicalOperator::Union(op) => op.initialize(),
            PhysicalOperator::__PLACEHOLDER__(op) => op.initialize(),
        }
    }
}

// TODO: change to use strum to derive variant enum
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperatorType {
    Project,
    Select,
    Join,
    GroupBy,
    ChromaJoin,
    ChannelRouter,
    ChannelSpout,
    Merge,
    MergeSpout,
    DeriveValue,
    UserDefinedFunction,
    UserDefinedSource,
    PythonInlineFunction,
    PythonRemoteFunction,
    Union,
}

impl PhysicalOperator {
    pub fn get_op_type(&self) -> OperatorType {
        match self {
            PhysicalOperator::Project(..) => OperatorType::Project,
            PhysicalOperator::Select(..) => OperatorType::Select,
            PhysicalOperator::Join(..) => OperatorType::Join,
            PhysicalOperator::DeriveValue(..) => OperatorType::DeriveValue,
            PhysicalOperator::UserDefinedFunction(..) => OperatorType::UserDefinedFunction,
            PhysicalOperator::UserDefinedSource(..) => OperatorType::UserDefinedSource,
            PhysicalOperator::PythonInlineFunction(..) => OperatorType::PythonInlineFunction,
            PhysicalOperator::PythonRemoteFunction(..) => OperatorType::PythonRemoteFunction,
            PhysicalOperator::Union(..) => OperatorType::Union,
            PhysicalOperator::__PLACEHOLDER__(..) => unimplemented!(
                "placeholder operator had its type asked for, which should not happen"
            ),
            PhysicalOperator::GroupBy(..) => OperatorType::GroupBy,
            PhysicalOperator::ChromaJoin(..) => OperatorType::ChromaJoin,
            PhysicalOperator::ChannelRouter(..) => OperatorType::ChannelRouter,
            PhysicalOperator::ChannelSpout(..) => OperatorType::ChannelSpout,
            PhysicalOperator::Merge(..) => OperatorType::Merge,
            PhysicalOperator::MergeSpout(..) => OperatorType::MergeSpout,
        }
    }
    pub fn is_spout(&self) -> bool {
        matches!(
            self,
            PhysicalOperator::UserDefinedSource(..)
                | PhysicalOperator::ChannelSpout(..)
                | PhysicalOperator::MergeSpout(..)
                | PhysicalOperator::PythonRemoteFunction(..)
        )
    }
    // most bolts are not spouts. the exceptions are ones where they have some external state that they send to and then pull from
    // one example of that is python remote fn
    pub fn is_bolt(&self) -> bool {
        !self.is_spout() || self.get_op_type() == OperatorType::PythonRemoteFunction
    }
}

fn split_async_python_state(
    mut state: AsyncPythonRemoteTaskState,
) -> (AsyncPythonRemoteTaskState, AsyncPythonRemoteTaskState) {
    let background_remote_task = AsyncPythonRemoteTaskState {
        port: state.port,
        should_stop: Arc::clone(&state.should_stop),
        runtime_handle: state.runtime_handle.clone(),
        input_to_background_thread: state.input_to_background_thread.clone(),
        input_from_main_thread: state.input_from_main_thread.take(),
        output_to_main_thread: state.output_to_main_thread.clone(),
        output_from_background_thread: None, // not needed
        script_name: state.script_name.clone(),
        scripts_dir_path: state.scripts_dir_path.clone(),
        pending_items: Arc::clone(&state.pending_items),
        encode: state.encode.take(),
        decode: state.decode.take(),
        shutdown: state.shutdown.take(),
    };
    debug!(
        "in split background has {}",
        background_remote_task.input_from_main_thread.is_some()
    );
    // foreground takes whatever is left
    let foreground_spout = AsyncPythonRemoteTaskState {
        port: state.port,
        should_stop: state.should_stop,
        runtime_handle: state.runtime_handle,
        input_to_background_thread: state.input_to_background_thread,
        input_from_main_thread: state.input_from_main_thread,
        output_to_main_thread: state.output_to_main_thread,
        output_from_background_thread: state.output_from_background_thread,
        script_name: state.script_name,
        scripts_dir_path: state.scripts_dir_path,
        pending_items: state.pending_items,
        encode: state.encode,
        decode: state.decode,
        shutdown: state.shutdown,
    };
    (background_remote_task, foreground_spout)
}

pub fn encode_tuple_with_ndarray(
    tuple: &Tuple,
    array_field: &str,
    shape_field: &str,
    tuple_id: usize,
) -> Vec<u8> {
    #[derive(Debug, Serialize, Deserialize)]
    struct TensorF32Message<'a> {
        tuple_id: u64,
        // dims: Vec<u64>,
        dims: Vec<usize>,
        #[serde(with = "serde_bytes")]
        tensor: &'a [u8],
    }

    let total_buffer = {
        // encode tuple_id into the message

        let shape = tuple
            .get(shape_field)
            .expect("shape field not found")
            .as_shape_buffer()
            .expect("shape field is not an array");
        let array_buf = tuple
            .get(array_field)
            .expect("field not found")
            .as_byte_buffer()
            .expect("field is not a byte buffer");

        let message = TensorF32Message {
            tuple_id: tuple_id as u64,
            // dims: shape.iter().map(|x| *x as u64).collect(),
            dims: shape.to_vec(),
            tensor: array_buf.as_ref(),
        };

        let mut total_buffer = Vec::<u8>::new();
        rmp_serde::encode::write(&mut total_buffer, &message).expect("Failed to encode message");
        total_buffer
    };
    total_buffer
}

pub fn tuple_with_ndarray_to_zmq_message(
    tuple: Tuple,
    array_field: &str,
    shape_field: &str,
    tuple_registry: &DashMap<usize, Tuple>,
    tuple_id_counter: &AtomicUsize,
) -> zeromq::ZmqMessage {
    let tuple_id = tuple_id_counter.fetch_add(1, atomic::Ordering::Relaxed);
    let total_buffer = encode_tuple_with_ndarray(&tuple, array_field, shape_field, tuple_id);

    let msg = zeromq::ZmqMessage::from(total_buffer);
    tuple_registry.insert(tuple_id, tuple);
    msg
}

#[allow(unused)]
#[cfg(test)]
// #[test]
fn test_save_tuple_with_ndarray_to_temp_file() {
    let temp_file_name = "./serialization_test_file";
    let array_field = "buffer";
    let shape_field = "shape";
    let mut tuple = get_tuple();
    let byte_arr = (0..32).flat_map(|v| (v as f32).to_ne_bytes()).collect();
    tuple.insert(array_field.to_key(), HabValue::ByteBuffer(byte_arr));
    let shape_arr = vec![2, 4, 4];
    tuple.insert(shape_field.to_key(), HabValue::ShapeBuffer(shape_arr));
    // let tuple_id_counter: AtomicUsize = AtomicUsize::new(0);
    let msg = encode_tuple_with_ndarray(&tuple, array_field, shape_field, 0);
    std::fs::write(temp_file_name, msg).expect("Failed to write to file");
}

static STRING_NUMBERS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

pub fn make_singleton_tuple(field: HabString, value: HabValue) -> Tuple {
    let mut tuple = get_tuple();
    tuple.insert(field, value);
    tuple
}

pub fn componentized_to_udf(
    derive_decision_key: ComputationExpression,
    should_emit: ComputationExpression,
    derive_eviction_key: ComputationExpression,
    should_evict: ComputationExpression,
) -> Box<dyn Send + Sync + Fn(&mut Queue<Tuple>) -> AggregationResult> {
    Box::new(move |state: &mut DeVec<Tuple>| {
        let decision_keys: Vec<HabValue> = state
            .iter()
            .map(|t| evaluate_computation_expression(t, &derive_decision_key))
            .collect_value_vec();
        let emit_decision = evaluate_computation_expression(
            &get_tuple().tap_mut(|v| {
                drop(v.insert(STRING_NUMBERS[0].to_key(), HabValue::List(decision_keys)))
            }),
            &should_emit,
        );
        let eviction_keys: Vec<HabValue> = state
            .iter()
            .map(|t| evaluate_computation_expression(t, &derive_eviction_key))
            .collect_value_vec();
        let mut eviction_tuple = get_tuple();
        eviction_tuple.insert(STRING_NUMBERS[0].to_key(), HabValue::List(eviction_keys));
        let emit_state = match emit_decision {
            HabValue::Bool(true) => Some(state.iter().cloned().collect_tuple_vec()),
            HabValue::Bool(false) => None,
            _ => unimplemented!(
                "Type Error in expression {:?} emit decision must be a boolean",
                emit_decision
            ),
        };
        state.extract_if(|index, _item| {
            // TODO: do we pass the item? do we add a map habvalue type?
            eviction_tuple.insert(STRING_NUMBERS[1].to_key(), HabValue::Integer(index as i32));
            let eviction_decision = evaluate_computation_expression(&eviction_tuple, &should_evict);
            match eviction_decision {
                HabValue::Bool(b) => b,
                _ => unimplemented!(
                    "Type Error in expression {:?} emit decision must be a boolean",
                    emit_decision
                ),
            }
        });
        AggregationResult {
            emit: emit_state,
            is_finished: false,
        }
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryDescriptor {
    pub name: HabString,
    pub description: HabString,
    pub sql_form: HabString,
    pub operators: Vec<OperatorDescriptor>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OperatorDescriptor {
    pub id: usize,
    #[serde(flatten)]
    pub operator: OperatorVariantDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum OperatorVariantDescriptor {
    #[serde(rename = "project")]
    Project {
        source: usize,
        fields: Vec<HabString>,
    },
    #[serde(rename = "filter")]
    Select {
        source: usize,
        predicate: ExpressionDescriptor,
    },
    #[serde(rename = "join")]
    HashJoin {
        left: usize,
        right: usize,
        predicate: ExpressionDescriptor,
        join_info: HashJoinKind,
    },
    #[serde(rename = "derive")]
    DeriveValue {
        source: usize,
        new_field_name: HabString,
        expression: ExpressionDescriptor,
    },
    #[serde(rename = "udf_source")]
    UdfSpout { name: HabString },
    #[serde(rename = "union")]
    Union { left: usize, right: usize },
    #[serde(rename = "group_by")]
    GroupBy {
        source: usize,
        fields: Vec<HabString>,
        aggregate: AggregationExpressionDescriptor,
    },
    #[serde(rename = "chroma_join")]
    ChromaJoin {
        index_stream: usize,
        lookup_stream: usize,
        metric: DistanceMetric,
        url: HabString,
        collection: HabString,
        distance_threshold: f32,
        join_info: ChromaJoinKind,
        // startup_info: ChromaLoadingData,
        // query_n_matches: usize,
        // keep_n_matches: usize,
    },
    #[serde(rename = "channel_router")]
    ChannelRouter {
        source: usize,
        routes: Vec<usize>,
        route_expression: ExpressionDescriptor,
        #[serde(default)]
        backup_route: BackupRouteDescriptor,
    },
    #[serde(rename = "channel_spout")]
    ChannelSpout {
        source: usize,
        timeout: usize,
        max_age_ns: Option<u64>,
    },
    #[serde(rename = "merge")]
    Merge {
        source: usize,
        parent: usize,
        on_merge_fn: Option<HabString>,
    },
    #[serde(rename = "merge_spout")]
    MergeSpout {
        sources: Vec<usize>,
        timeout: usize,
        max_age_ns: Option<u64>,
    },

    #[serde(rename = "udf")]
    UserDefinedFunction { source: usize, name: HabString },

    #[serde(rename = "python_udf")]
    RemotePythonUdf {
        scripts_dir_path: HabString,
        script_name: HabString,
        encode_fn: HabString,
        decode_fn: HabString,
        shutdown_fn: HabString,
        #[serde(rename = "source")]
        input: usize,
    },
    // TODO: make this optional later. it stays here for now to make sure we get all cases
    // alternatively we can turn on all features for IDE type checking
    // #[cfg(feature = "pyo3")]
    #[serde(rename = "inline_python_script")]
    InlinePythonUdf {
        scripts_dir_path: HabString,
        script_name: HabString,
        function_name: HabString,
        encode_fn: InlinePythonEncodeKind,
        decode_fn: InlinePythonDecodeKind,
        #[serde(rename = "source")]
        input: usize,
    },

    #[serde(rename = "onnx_inference")]
    OnnxInference {
        source: usize,
        model_path: HabString,
        args: Vec<NdArrayDescriptor>,
        outputs: Vec<NdArrayDescriptor>,
    },
}

#[derive(Default, Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind")]
pub enum BackupRouteDescriptor {
    #[default]
    #[serde(rename = "drop")]
    Drop,

    #[serde(rename = "forward_with_values")]
    ForwardWithValues {
        merge_spout_position: usize,
        values_to_set: Vec<FieldInfo>,
    },

    #[serde(rename = "custom")]
    CustomRoute(AsyncChannelDescriptor),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FieldInfo {
    pub key: HabString,
    pub value: Option<LimitedHabValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub struct AsyncChannelDescriptor {
    pub max_capacity: Option<usize>,
    pub spout_position: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind")]
pub enum NdArrayDescriptor {
    SplitParts {
        shape_field: HabString,
        buffer_field: HabString,
    }, // NdArray{
       //     ndarray_field: HabString,
       // },
}

// untagged enum
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum InlinePythonEncodeKind {
    #[serde(rename = "python_values")]
    PythonValues { fields: Vec<HabString> },
    #[serde(rename = "custom")]
    Custom {
        function_name: HabString,
        fields: Vec<HabString>,
    },
    // #[serde(rename = "ndarray")]
    // NdArray {
    //     array_field: HabString,
    //     shape_field: HabString,
    // },
    #[serde(rename = "default")]
    Default { fields: Vec<HabString> },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum InlinePythonDecodeKind {
    #[serde(rename = "python_value")]
    PythonValues { output_fields: Vec<HabString> },
    #[serde(rename = "custom")]
    Custom {
        output_fields: Vec<HabString>,
        function_name: HabString,
    },
    // #[serde(rename = "ndarray")]
    // NdArray {
    //     array_field: HabString,
    //     shape_field: HabString,
    // },
    #[serde(rename = "default")]
    Default { output_fields: Vec<HabString> },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
enum ChromaLoadingDataDescriptor {
    Remote {
        url: HabString,
    },
    ChildProcess {
        url: HabString,
        port: u16,
        persistent_folder: HabString,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HashJoinKind {
    #[serde(rename = "outer_build_right")]
    OuterTable { fields: Vec<HabString> },
    #[serde(rename = "inner_build_right")]
    InnerTable { fields: Vec<HabString> },
    #[serde(rename = "double_stream")]
    DoubleStream {
        eviction_policy: Option<ExpressionDescriptor>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ExpressionDescriptor {
    #[serde(rename = "field")]
    Field { name: HabString },
    #[serde(rename = "literal")]
    Literal {
        #[serde(flatten)]
        value: ComputationLiteral,
    },
    #[serde(rename = "unary")]
    UnaryOp {
        op: UnaryOp,
        expr: Box<ExpressionDescriptor>,
    },
    #[serde(rename = "binary")]
    BinaryOp {
        op: BinaryOp,
        left: Box<ExpressionDescriptor>,
        right: Box<ExpressionDescriptor>,
    },
    #[serde(rename = "ternary")]
    TernaryOp {
        cond: Box<ExpressionDescriptor>,
        left: Box<ExpressionDescriptor>,
        right: Box<ExpressionDescriptor>,
    },
    #[serde(rename = "udf")]
    UserDefinedFunction {
        name: HabString,
        args: Vec<ExpressionDescriptor>,
    },
}

pub struct ChannelState {
    pub senders: Option<crate::SyncPipe>,
    pub receivers: crossbeam::channel::Receiver<Vec<Tuple>>,
}

pub enum FunctionKinds {
    SourceUdf(Box<dyn Sync + (Fn() -> Box<dyn Send + Sync + Fn() -> Option<Vec<Tuple>>>)>),
    SelectFilterUdf(fn() -> Box<dyn Send + Sync + Fn(&Tuple) -> bool>),
    // JoinFilterUdf(fn() -> Box<dyn Send + Sync + Fn(&Tuple, &Tuple) -> bool>),
    JoinFilterUdf(fn(&Tuple, &Tuple) -> bool),
    // JoinEvictUdf(fn() -> Box<dyn Send + Sync + Fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool>),
    JoinEvictUdf(fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool),
    ComputationExpressionUdf(
        Box<dyn Fn() -> Box<dyn Send + Sync + Fn(Vec<&HabValue>) -> HabValue>>,
    ),
    MergeCallbackUdf(Box<dyn Fn() -> Box<dyn Send + Sync + Fn(&Tuple)>>),
    AggregationUdf(
        Box<dyn Fn() -> Box<dyn Send + Sync + Fn(&mut Queue<Tuple>) -> AggregationResult>>,
    ),
    RoutingUdf(
        Box<
            dyn Fn()
                -> Box<dyn Send + Sync + FnMut(Vec<Tuple>, &[crate::SyncPipe]) -> Option<usize>>,
        >,
    ),
    EncodeRemotePythonUdf(
        Box<dyn Fn() -> Box<dyn Send + Sync + Fn(usize, &Tuple) -> zeromq::ZmqMessage>>,
    ),
    DecodeRemotePythonUdf(
        Box<
            dyn Fn() -> Box<
                dyn Send + Sync + Fn(zeromq::ZmqMessage, &DashMap<usize, Tuple>) -> Vec<Tuple>,
            >,
        >,
    ),
    ShutdownRemotePythonUdf(Box<dyn Fn() -> Box<dyn Send + Sync + FnOnce() -> zeromq::ZmqMessage>>),
}

fn descriptor_is_spout(descriptor: &OperatorVariantDescriptor) -> bool {
    match descriptor {
        OperatorVariantDescriptor::UdfSpout { .. } => true,
        OperatorVariantDescriptor::MergeSpout { .. } => true,
        OperatorVariantDescriptor::ChannelSpout { .. } => true,
        OperatorVariantDescriptor::RemotePythonUdf { .. } => true,
        _ => false,
    }
}

// returns the threads that each operator needs to run on
pub fn get_topology_simple(operators: &[OperatorDescriptor], num_threads: u8) -> Vec<Vec<usize>> {
    let mut num_spouts = 0;
    for op in operators {
        if descriptor_is_spout(&op.operator) {
            num_spouts += 1;
        }
    }
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    let num_threads = num_threads.min(num_spouts) as usize;
    let mut thread = 0;
    'attempt_distribution: loop {
        let mut threads = vec![vec![]; num_threads];
        // round robin method
        for op in operators {
            if descriptor_is_spout(&op.operator) {
                threads[thread].push(op.id);
                thread = (thread + 1) % num_threads as usize;
            } else {
                // throw it on a random thread
                let thread = rng.gen_range(0..num_threads);
                threads[thread as usize].push(op.id);
            }
        }

        debug!("finished with thread distribution: {:?}", threads);
        let spouts_per_thread = threads
            .iter()
            .map(|thread| {
                thread
                    .iter()
                    .filter(|&x| descriptor_is_spout(&operators[*x].operator))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        debug!("spouts in each thread: {:?}", spouts_per_thread);

        for thread in threads.iter() {
            if thread.is_empty() {
                debug!("empty thread found, retrying");
                continue 'attempt_distribution;
            }
            if thread
                .iter()
                .all(|&x| !descriptor_is_spout(&operators[x].operator))
            {
                debug!("thread with no spouts found, retrying");
                continue 'attempt_distribution;
            }
        }

        debug!("finalized with thread distribution: {:?}", threads);
        debug!("spouts in each thread: {:?}", spouts_per_thread);

        // random method
        // for op in operators{
        //     if descriptor_is_spout(&op.operator) {
        //         let thread = rng.gen_range(0..num_threads);
        //         threads[thread].push(op.id);
        //     }
        // }

        return threads;
    }
}

// returns the threads that each operator needs to run on
pub fn get_topology(operators: &[OperatorDescriptor], num_threads: u8) -> Vec<Vec<usize>> {
    use OperatorVariantDescriptor::*;
    if operators.is_empty() {
        return vec![];
    }
    let mut sources: Vec<Vec<usize>> = operators
        .iter()
        .map(|op| match &op.operator {
            &Project { source, .. } => vec![source],
            &Select { source, .. } => vec![source],
            &HashJoin { left, right, .. } => vec![left, right],
            &DeriveValue { source, .. } => vec![source],
            &UdfSpout { .. } => vec![],
            &Union { left, right } => vec![left, right],
            &GroupBy { source, .. } => vec![source],
            &ChromaJoin {
                index_stream,
                lookup_stream,
                ..
            } => vec![index_stream, lookup_stream],
            &ChannelRouter { source, .. } => vec![source],
            &ChannelSpout { source, .. } => vec![source],
            &Merge { source, .. } => vec![source],
            MergeSpout { sources, .. } => sources.clone(),
            &UserDefinedFunction { source, .. } => vec![source],
            &RemotePythonUdf { input, .. } => vec![input],
            &InlinePythonUdf { input, .. } => vec![input],
            &OnnxInference { source, .. } => vec![source],
        })
        .collect::<Vec<_>>();

    let mut parents = Vec::new();
    for operator in operators {
        let mut op_parents = Vec::new();
        for (parent_id, _possible_parent) in operators.iter().enumerate() {
            if sources[parent_id].contains(&operator.id) {
                op_parents.push(parent_id);
            }
        }
        parents.push(op_parents);
    }
    // we start with one main thread that starts at the first operator
    let mut assigned_thread: Vec<Option<usize>> = operators.iter().map(|_| None).collect();
    // let mut topology = vec![];
    let mut topology = vec![vec![]; num_threads as usize];
    let mut path_stack = vec![];
    for (i, op) in operators.iter().enumerate().rev() {
        match op.operator {
            // add any other spouts that show up here
            // anything that doesn't have any inputs needs to be assigned to a new thread and added to the topology and path stack
            UdfSpout { .. } | MergeSpout { .. } | ChannelSpout { .. } => {
                // let thread = topology.len();
                // assigned_thread[i] = Some(thread);
                // topology.push(vec![i]);
                // path_stack.push(i);

                let thread = i % num_threads as usize;
                assigned_thread[i] = Some(thread);
                topology[thread].push(i);
                path_stack.push(i);
            }

            ChannelRouter { .. }
            | Merge { .. }
            | RemotePythonUdf { .. }
            // #[cfg(feature = "pyo3")]
            | InlinePythonUdf { .. }
            | OnnxInference { .. }
            | Union { .. }
            | ChromaJoin { .. }
            | GroupBy { .. }
            | HashJoin { .. }
            | DeriveValue { .. }
            | Select { .. }
            | Project { .. }
            | UserDefinedFunction { .. }=> {}
        }
    }
    // we follow all the sources for now, and we will follow more as we go
    debug!("sources at start: {:?}", sources);
    debug!("parents at start: {:?}", parents);
    loop {
        debug!("starting loop iteration with path stack: {:?}", path_stack);
        let Some(current_op_id) = path_stack.pop() else {
            break;
        };
        let current_thread = assigned_thread[current_op_id]
            .expect("the current operator must have been assigned to a thread already");
        // depth first to find a channel split point, then we split it into a depth first along both paths until they hit a Merge operator
        // then we continue depth first along the other path
        for &parent_id in parents[current_op_id].iter() {
            let parent_op = &operators[parent_id];
            match &parent_op.operator {
                Merge { .. } => {
                    // we've hit a merge operator, so we can continue along the other path
                    // remove it from the list of sources it is waiting on
                    let upstream_sources = &mut sources[parent_id];
                    upstream_sources.retain(|&x| x != current_op_id);
                    if upstream_sources.is_empty() {
                        // we've hit the end of the path, so we can re-use the thread for the parent here
                        assigned_thread[parent_id] = Some(current_thread);
                        // do not put the parent there because its parent will be a mergespout, which should already be covered
                        // path_stack.push(parent_id);
                        topology[current_thread].push(parent_id);
                        continue;
                    }
                }
                // we could possibly check that all the routes match up with the expected parents
                ChannelRouter { routes: _, .. } => {
                    // we've hit a channel router, and all the channel spouts are already on the stack so we don't have to add them
                    // let new_thread: usize = topology.len();
                    // assigned_thread[parent_id] = Some(new_thread);
                    assigned_thread[current_op_id] = Some(current_thread);
                    topology[current_thread].push(parent_id);
                }
                // when merging two streams using a binary operator, we pick a side to follow
                // in the case of a join, it's the index side
                // if there's no index, we choose the left side arbitrarily
                HashJoin { left, .. }
                | ChromaJoin {
                    index_stream: left, ..
                }
                | Union { left, .. }
                    if *left == current_op_id =>
                {
                    // we've hit a join operator, so we need to split the path by putting it on a new thread
                    let new_thread: usize = topology.len();
                    assigned_thread[parent_id] = Some(new_thread);
                    path_stack.push(parent_id);
                    topology.push(vec![parent_id]);
                }
                _ => {
                    // we haven't hit a merge operator yet, so we continue along the current path
                    assigned_thread[parent_id] = Some(current_thread);
                    path_stack.push(parent_id);
                    topology[current_thread].push(parent_id);
                }
            }
        }
    }
    topology
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AggregationExpressionDescriptor {
    #[serde(rename = "componentized")]
    Componentized {
        derive_decision_key: ExpressionDescriptor,
        should_emit: ExpressionDescriptor,
        derive_eviction_key: ExpressionDescriptor,
        should_evict: ExpressionDescriptor,
    },
    #[serde(rename = "udf")]
    Udf { name: HabString },
    #[serde(rename = "builtin")]
    Builtin {
        field: HabString,
        op: BuiltinAggregator,
    },
}

fn computation_descriptor_to_computation_expression(
    input: &ExpressionDescriptor,
    function_lookup: &BTreeMap<HabString, FunctionKinds>,
) -> ComputationExpression {
    match input {
        ExpressionDescriptor::Field { name } => ComputationExpression::Field(name.clone()),
        ExpressionDescriptor::Literal { value } => ComputationExpression::Literal(value.clone()),
        ExpressionDescriptor::UnaryOp { op, expr } => {
            let expr = computation_descriptor_to_computation_expression(expr, function_lookup);
            ComputationExpression::UnaryOp {
                op: *op,
                expr: Box::new(expr),
            }
        }
        ExpressionDescriptor::BinaryOp { op, left, right } => {
            let left = computation_descriptor_to_computation_expression(left, function_lookup);
            let right = computation_descriptor_to_computation_expression(right, function_lookup);
            ComputationExpression::BinaryOp {
                op: *op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        ExpressionDescriptor::TernaryOp { cond, left, right } => {
            let cond = computation_descriptor_to_computation_expression(cond, function_lookup);
            let left = computation_descriptor_to_computation_expression(left, function_lookup);
            let right = computation_descriptor_to_computation_expression(right, function_lookup);
            ComputationExpression::TernaryOp {
                cond: Box::new(cond),
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        ExpressionDescriptor::UserDefinedFunction { name, args } => {
            let args = args
                .iter()
                .map(|arg| computation_descriptor_to_computation_expression(arg, function_lookup))
                .collect();
            use FunctionKinds::*;
            let action = match function_lookup.get(name) {
                Some(ComputationExpressionUdf(f)) => f(),
                Some(SourceUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as udf source"),
                Some(SelectFilterUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as select filter"),
                Some(JoinFilterUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as join filter"),
                Some(JoinEvictUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as join evict"),
                Some(AggregationUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as aggregation"),
                Some(RoutingUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as routing"),
                Some(EncodeRemotePythonUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as encode remote python"),
                Some(DecodeRemotePythonUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as decode remote python"),
                Some(ShutdownRemotePythonUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as shutdown remote python"),
                Some(MergeCallbackUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as merge callback"),
                None => unreachable!("Function {name} not found in function lookup"),
            };
            ComputationExpression::UserDefinedFunction {
                name: name.clone(),
                args,
                action,
            }
        }
    }
}
