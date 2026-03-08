use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    sync::{
        atomic::{self, AtomicUsize},
        Arc, LazyLock, Mutex,
    },
    time::{Duration, Instant},
};

use dashmap::{DashMap, DashSet};
use either::Either::{self, Left, Right};
use itertools::Itertools;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use ort::session::input;
use serde::{Deserialize, Serialize};
use spaghetto::DeVec;
use tap::Tap;
use tokio::{runtime::Runtime, sync::watch};

use tokio::sync::mpsc::UnboundedReceiver as Receiver;

use pyo3::prelude::*;

use crate::{
    async_operators::{AsyncPythonRemoteTaskState, AsyncSpout, Bolt},
    basic_pooling::{get_tuple, get_tuple_vec, init_pools, CollectTupleVec, CollectValueVec},
    caching::StrToKey,
    expression::{evaluate_computation_expression, ComputationExpression},
    global_logger::{LimitedHabValue, NO_AUX_DATA as NoAuxData},
    query_builder::{
        AggregationExpressionDescriptor, AsyncChannelDescriptor, ExpressionDescriptor,
        HashJoinKind, OperatorDescriptor, OperatorVariantDescriptor, QueryDescriptor,
    },
    scheduler::{BackupChannel, BoundedAsyncReceiver},
    start_python_with_modules, AggregationExpression, AggregationResult, AsyncPipe,
    EncoderFunction, HabString, HabValue, JoinInner, Operator, OperatorOutput, Queue, Tuple,
};

use crate::async_operators::AsyncChannelRouter;
use crate::async_operators::AsyncChannelSpout;
use crate::async_operators::AsyncMergeSpout;
use crate::async_operators::Merge;
use crate::async_operators::OnnxInferenceOperator;
use crate::async_operators::PythonRemoteUdf;
use crate::async_operators::UdfSpout;
use crate::operators::ChromaJoin;
use crate::operators::DeriveValue;
use crate::operators::DummyBolt;
use crate::operators::GroupBy;
use crate::operators::Join;
use crate::operators::Project;
use crate::operators::PythonInlineUdf;
use crate::operators::Select;
use crate::operators::UdfBolt;
use crate::operators::Union;

// TODO: change this to have the actual fields needed to make it work
//   for now it just exists as a placeholder for the construction implementation
pub enum PhysicalOperator {
    Project(Project),
    Select(Select),
    Join(Join),
    GroupBy(GroupBy),
    DeriveValue(DeriveValue),
    ChromaJoin(ChromaJoin),
    ChannelRouter(AsyncChannelRouter),
    ChannelSpout(AsyncChannelSpout),
    Merge(Merge),
    MergeSpout(AsyncMergeSpout),
    UserDefinedFunction(UdfBolt),
    UserDefinedSource(UdfSpout),
    PythonInlineFunction(PythonInlineUdf),
    PythonRemoteFunction(PythonRemoteUdf),
    OnnxInferenceOperator(OnnxInferenceOperator),
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
            PhysicalOperator::OnnxInferenceOperator(op) => op.id,
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
                op.add_parent(parent);
            }
            PhysicalOperator::PythonRemoteFunction(op) => op.add_parent(parent),
            PhysicalOperator::OnnxInferenceOperator(op) => op.add_parent(parent),
            PhysicalOperator::Union(op) => op.add_parent(parent),
            PhysicalOperator::__PLACEHOLDER__(op) => op.add_parent(parent),
        }
    }

    fn initialize(&mut self) {
        match self {
            PhysicalOperator::Project(op) => Operator::initialize(op),
            PhysicalOperator::Select(op) => Operator::initialize(op),
            PhysicalOperator::Join(op) => Operator::initialize(op),
            PhysicalOperator::GroupBy(op) => Operator::initialize(op),
            PhysicalOperator::DeriveValue(op) => Operator::initialize(op),
            PhysicalOperator::ChromaJoin(op) => Operator::initialize(op),
            PhysicalOperator::ChannelRouter(op) => Operator::initialize(op),
            PhysicalOperator::ChannelSpout(op) => Operator::initialize(op),
            PhysicalOperator::Merge(op) => Operator::initialize(op),
            PhysicalOperator::MergeSpout(op) => Operator::initialize(op),
            PhysicalOperator::UserDefinedFunction(op) => Operator::initialize(op),
            PhysicalOperator::UserDefinedSource(op) => Operator::initialize(op),
            PhysicalOperator::PythonInlineFunction(op) => Operator::initialize(op),
            PhysicalOperator::PythonRemoteFunction(op) => Operator::initialize(op),
            PhysicalOperator::OnnxInferenceOperator(op) => Operator::initialize(op),
            PhysicalOperator::Union(op) => Operator::initialize(op),
            PhysicalOperator::__PLACEHOLDER__(op) => Operator::initialize(op),
        }
    }
}

// TODO: WIP
// This is intended to cache the allocations for futures of a given type, ignoring lifetime limitations
// Can we instead use bumpalo or something to improve the allocation time?

// mod physical_operator_cache {

//     use std::{future::Future, mem::MaybeUninit};
//     pub(super) static FUTURE_CACHE: std::sync::LazyLock<(
//         crossbeam::channel::Sender<CachedFuture>,
//         crossbeam::channel::Receiver<CachedFuture>,
//     )> = std::sync::LazyLock::new(|| {
//         let (tx, rx) = crossbeam::channel::bounded(128);
//         (tx, rx)
//     });

//     pub(super) struct CachedFuture {
//         // future: Box<dyn Future<Output = ()>>,
//         future_ptr: *mut (),
//     }
//     // SAFETY: the only way to make a CachedFuture is to provide a future that was Send
//     unsafe impl Send for CachedFuture {}

//     pub(super) fn create_cached_future<F: Future + Send>(future: Box<F>) -> CachedFuture {
//         let future_ptr = Box::into_raw(future) as *mut F;
//         let future_ptr = unsafe {
//             // drop in place
//             future_ptr.drop_in_place();
//             future_ptr as *mut ()
//         };
//         CachedFuture { future_ptr }
//     }

//     pub(super) struct RestoredFuture<F> {
//         future: std::mem::MaybeUninit<std::pin::Pin<Box<F>>>,
//     }

//     impl<F: Future> Future for RestoredFuture<F> {
//         type Output = F::Output;
//         fn poll(
//             mut self: std::pin::Pin<&mut Self>,
//             cx: &mut std::task::Context<'_>,
//         ) -> std::task::Poll<Self::Output> {
//             unsafe { self.future.assume_init_mut().as_mut().poll(cx) }
//         }
//     }

//     impl<F> Drop for RestoredFuture<F> {
//         fn drop(&mut self) {
//             // drop the future
//             // SAFETY: we are dropping the future,
//             // but that memory will be re-used with a different value later.
//             // This way, the values are only dropped once.
//             // If our pointer cannot be used again, we still need to de-allocate it,
//             // so we have to re
//             unsafe {
//                 let my_pin: &mut std::pin::Pin<Box<F>> = self.future.assume_init_mut();
//                 let my_pin: *mut std::pin::Pin<Box<F>> = my_pin as *mut _;
//                 let my_pin: std::pin::Pin<Box<F>> = std::ptr::read(my_pin);
//                 let my_box: Box<F> = std::pin::Pin::into_inner_unchecked(my_pin);
//                 let future_ptr: *mut F = Box::into_raw(my_box);
//                 // let mut future_ptr: Box<F> = todo!(); // std::ptr::read((&mut **self.future.assume_init_mut()) as *mut _);
//                 future_ptr.drop_in_place();
//                 // now we can return it to the cache
//                 let future_ptr = future_ptr as *mut ();
//                 match FUTURE_CACHE.0.send(CachedFuture { future_ptr }) {
//                     Ok(_) => (),
//                     Err(e) => {
//                         // we can de-allocate that memory now since the queue was full
//                         drop(Box::from_raw(
//                             e.into_inner().future_ptr as *mut std::mem::MaybeUninit<F>,
//                         ));
//                     }
//                 }
//             }
//         }
//     }

//     // there is no way to prevent this from being misused because we cannot enforce that the requested future is the correct type that was originally stored
//     // if we had type ids for the futures, we could verify with that, but the objective here is to allow Futures with references to be used
//     // so we cannot do that (type ids require 'static currently)
//     pub(super) unsafe fn restore_future<F: Future + Send>(future_ptr: *mut (), f: F) -> Box<F> {
//         let future_ptr = future_ptr as *mut F;
//         future_ptr.write(f);
//         let future = Box::from_raw(future_ptr);
//         future
//     }

//     pub(super) fn get_future_memory<F: Future + Send>(future: F) -> RestoredFuture<F> {
//         RestoredFuture {
//             future: std::mem::MaybeUninit::new(Box::into_pin(match FUTURE_CACHE.1.try_recv() {
//                 Ok(cached_future) => unsafe { restore_future(cached_future.future_ptr, future) },
//                 Err(_) => Box::new(future),
//             })),
//         }
//     }

//     #[derive(Clone, Copy)]
//     struct UnsafeFutureMonster {
//         // pd: std::marker::PhantomData<*mut ()>,
//         pd: std::marker::PhantomData<()>,
//     }
//     // unsafe impl Send for UnsafeFutureMonster{}
//     // unsafe impl Sync for UnsafeFutureMonster{}
//     impl std::fmt::Debug for UnsafeFutureMonster {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             write!(f, "UnsafeFutureMonster")
//         }
//     }

//     fn self_ref_fibonacci(
//         n: u32,
//         monster: UnsafeFutureMonster,
//     ) -> impl Send + Future<Output = u32> {
//         async move {
//             let output = if n == 0 {
//                 1
//             } else if n == 1 {
//                 1
//             } else {
//                 let a = get_future_memory(self_ref_fibonacci(n - 1, monster)).await;
//                 let b = get_future_memory(self_ref_fibonacci(n - 2, monster)).await;
//                 a + b
//             };
//             println!("{:?}", monster);
//             output
//         }
//     }

// }

impl Bolt for PhysicalOperator {
    fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> impl 'a + Send + Future<Output = OperatorOutput> {
        static LAZY_IS_BINARY_LOG_SET: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("TUPLE_BINARY_LOG").is_ok());
        const DEFAULT_MAXIMUM_TUPLES_TO_LOG: usize = 25;
        static MAXIMUM_TUPLES_TO_LOG: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            std::env::var("MAXIMUM_TUPLES_TO_LOG")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_MAXIMUM_TUPLES_TO_LOG)
        });
        static OPERATOR_TO_TUPLES_LOGGED: LazyLock<DashMap<usize, usize>> =
            LazyLock::new(|| DashMap::new());
        Box::pin(async move {
            if tuples.len() == 1 {
                debug!(
                    "received tuple {} in operator {} ({:?})",
                    tuples[0].id(),
                    self.get_id(),
                    self.get_op_type()
                );
            } else {
                debug!(
                    "received {} tuples in operator {} ({:?})",
                    tuples.len(),
                    self.get_id(),
                    self.get_op_type()
                );
            }

            // TODO: make default value version of early error return
            const MINIMUM_BATCH_SIZE: usize = 4;
            let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) else {
                error!("failed to get time since epoch");
                panic!("failed to get time since epoch");
            };
            let now = now.as_nanos();
            let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                smallvec::SmallVec::new();
            for tuple in tuples.iter() {
                let age = now - tuple.unix_time_created_ns();
                ages.push((tuple.id(), age as f64));
            }
            debug!(
                "Bolt operator #{} ({:?}) received {} tuples, ids and ages: {:?}",
                self.get_id(),
                self.get_op_type(),
                tuples.len(),
                ages
            );
            for t in &tuples {
                use crate::global_logger;
                use std::io::Write;
                #[allow(unused_labels)]
                'log_enter_physical_operator: {
                    let my_id = self.get_id();
                    let id_to_write = my_id;
                    let mut operator_string = *b"enter_physical_operator_xxxxxxxx";
                    let num_xs = 8;
                    let operator_string_len = operator_string.len();
                    let xs_idx = operator_string.len() - num_xs;
                    if let Err(e) = write!(&mut operator_string[xs_idx..], "{:08}", id_to_write) {
                        error!("Error creating operator string: {:?}", e);
                        break 'log_enter_physical_operator;
                    }
                    let Ok(operator_string) = std::str::from_utf8(&operator_string) else {
                        error!("only utf8 should be written to the limited operator name buffer");
                        panic!("only utf8 should be written to the limited operator name buffer");
                    };
                    let operator_string = &operator_string[..operator_string_len];
                    let tuple_id = t.id();
                    let log_location = operator_string.to_raw_key();

                    let logging_result = if *LAZY_IS_BINARY_LOG_SET {
                        let mut r = OPERATOR_TO_TUPLES_LOGGED.entry(my_id).or_insert(0);
                        *r += 1;
                        let current_count = *r;
                        let bytes_as_base64 = if current_count <= *MAXIMUM_TUPLES_TO_LOG {
                            let serializeable_input =
                                crate::ws_types::BetterTuple::into_serializable(t);
                            let serialized_to_bytes = match rmp_serde::to_vec(&serializeable_input)
                            {
                                Ok(b) => b,
                                Err(e) => {
                                    error!("failed to serialize tuple for binary logging: {:?}", e);
                                    break 'log_enter_physical_operator;
                                }
                            };
                            use base64::prelude::*;
                            let bytes_as_base64 = BASE64_STANDARD.encode(&serialized_to_bytes);
                            bytes_as_base64
                        } else {
                            String::new()
                        };
                        let aux_data = Some([(
                            "binary".to_raw_key(),
                            LimitedHabValue::String(HabString::Owned(bytes_as_base64)),
                        )]);
                        global_logger::log_data(tuple_id, log_location, aux_data)
                    } else {
                        let aux_data = NoAuxData;
                        global_logger::log_data(tuple_id, log_location, aux_data)
                    };
                    if let Err(errs) = logging_result {
                        for err in errs {
                            error!(
                                "Error logging tuple {} in operator {}: {:?}",
                                tuple_id, operator_string, err
                            );
                        }
                    }
                }
            }
            use PhysicalOperator as PO;
            match self {
                PO::Project(op) => op.process_tuples(tuples, source, bolts).await,
                PO::Select(op) => op.process_tuples(tuples, source, bolts).await,
                PO::Join(op) => op.process_tuples(tuples, source, bolts).await,
                PO::GroupBy(op) => op.process_tuples(tuples, source, bolts).await,
                PO::DeriveValue(op) => op.process_tuples(tuples, source, bolts).await,
                PO::ChromaJoin(op) => op.process_tuples(tuples, source, bolts).await,
                PO::ChannelRouter(op) => op.process_tuples(tuples, source, bolts).await,
                PO::ChannelSpout(_op) => {
                    error!("ChannelSpout is not a bolt");
                    panic!("ChannelSpout is not a bolt");
                }
                PO::Merge(op) => op.process_tuples(tuples, source, bolts).await,
                PO::MergeSpout(_op) => {
                    error!("MergeSpout is not a bolt");
                    panic!("MergeSpout is not a bolt")
                }
                PO::UserDefinedFunction(op) => op.process_tuples(tuples, source, bolts).await,
                PO::UserDefinedSource(_op) => {
                    panic!("UserDefinedSource is not a bolt")
                }
                PO::PythonInlineFunction(op) => op.process_tuples(tuples, source, bolts).await,
                PO::PythonRemoteFunction(op) => op.process_tuples(tuples, source, bolts).await,
                PO::OnnxInferenceOperator(op) => op.process_tuples(tuples, source, bolts).await,
                PO::Union(op) => op.process_tuples(tuples, source, bolts).await,
                PO::__PLACEHOLDER__(..) => {
                    let msg = format!(
                        "placeholder operator {} called as a bolt somehow",
                        self.get_id()
                    );
                    error!("{}", msg);
                    unimplemented!("{}", msg);
                }
            }
        })
    }
}

impl AsyncSpout for PhysicalOperator {
    fn initialize<'this>(
        &'this mut self,
        ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + std::future::Future<Output = ()> {
        let op_id = self.get_id();
        use AsyncSpout;
        use PhysicalOperator as PO;
        let fut: Either<_, Either<_, Either<_, Either<_, _>>>> = match self {
            PO::Project(_) => {
                error!("Project operator {op_id} should not be initialized as a spout");
                panic!("Project operator {op_id} should not be initialized as a spout");
            }
            PO::Select(_) => {
                error!("Select operator {op_id} should not be initialized as a spout");
                panic!("Select operator {op_id} should not be initialized as a spout");
            }
            PO::Join(_) => {
                error!("Join operator {op_id} should not be initialized as a spout");
                panic!("Join operator {op_id} should not be initialized as a spout");
            }
            PO::GroupBy(_) => {
                error!("GroupBy operator {op_id} should not be initialized as a spout");
                panic!("GroupBy operator {op_id} should not be initialized as a spout");
            }
            PO::DeriveValue(_) => {
                error!("DeriveValue operator {op_id} should not be initialized as a spout");
                panic!("DeriveValue operator {op_id} should not be initialized as a spout");
            }
            PO::ChromaJoin(_) => {
                error!("Chroma join operator {op_id} should not be initialized as a spout");
                panic!("Chroma join operator {op_id} should not be initialized as a spout");
            }
            PO::ChannelRouter(_) => {
                error!("channel router operator {op_id} should not be initialized as a spout");
                panic!("channel router operator {op_id} should not be initialized as a spout");
            }
            PO::ChannelSpout(op) => Left::<_, Either<_, Either<_, Either<_, _>>>>(
                AsyncSpout::initialize(op, ready_to_start),
            ),
            PO::Merge(_) => {
                error!("Merge operator {op_id} should not be initialized as a spout");
                panic!("Merge operator {op_id} should not be initialized as a spout");
            }
            PO::MergeSpout(op) => Right::<_, Either<_, Either<_, Either<_, _>>>>(Left(
                AsyncSpout::initialize(op, ready_to_start),
            )),
            PO::UserDefinedFunction(_) => {
                error!("Merge operator {op_id} should not be initialized as a spout");
                panic!("Merge operator {op_id} should not be initialized as a spout");
            }
            PO::UserDefinedSource(op) => Right::<_, Either<_, Either<_, Either<_, _>>>>(Right(
                Left(AsyncSpout::initialize(op, ready_to_start)),
            )),
            PO::PythonInlineFunction(_) => {
                // unimplemented!("inline python functions are not yet implemented");
                let err = format!(
                    "Python inline function operator {op_id} should not be initialized as a spout"
                );
                error!("{err}");
                panic!("{err}");
            }
            PO::PythonRemoteFunction(op) => Right::<_, Either<_, Either<_, Either<_, _>>>>(Right(
                Right(Left(AsyncSpout::initialize(op, ready_to_start))),
            )),
            PO::OnnxInferenceOperator(_) => {
                error!("onnx inference operator {op_id} should not be initialized as a spout");
                panic!("onnx inference operator {op_id} should not be initialized as a spout");
            }
            PO::Union(_) => {
                error!("Union operator {op_id} should not be initialized as a spout");
                panic!("Union operator {op_id} should not be initialized as a spout");
            }
            PO::__PLACEHOLDER__(op) => {
                // (this is a noop)
                Right::<_, Either<_, Either<_, Either<_, _>>>>(Right(Right(Right(
                    AsyncSpout::initialize(op, ready_to_start),
                ))))
            }
        };
        fut
    }

    fn produce<'this>(
        &'this mut self,
        bolts: std::sync::Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + std::future::Future<Output = OperatorOutput> {
        use PhysicalOperator as PO;
        let op_id = self.get_id();
        match self {
            PO::Project(_) => {
                error!("Project operator {op_id} should not be initialized as a spout");
                panic!("Project operator {op_id} should not be initialized as a spout");
            }
            PO::Select(_) => {
                error!("Select operator {op_id} should not be initialized as a spout");
                panic!("Select operator {op_id} should not be initialized as a spout");
            }
            PO::Join(_) => {
                error!("Join operator {op_id} should not be initialized as a spout");
                panic!("Join operator {op_id} should not be initialized as a spout");
            }
            PO::GroupBy(_) => {
                error!("GroupBy operator {op_id} should not be initialized as a spout");
                panic!("GroupBy operator {op_id} should not be initialized as a spout");
            }
            PO::DeriveValue(_) => {
                error!("DeriveValue operator {op_id} should not be initialized as a spout");
                panic!("DeriveValue operator {op_id} should not be initialized as a spout");
            }
            PO::ChromaJoin(_) => {
                error!("Chroma join operator {op_id} should not be initialized as a spout");
                panic!("Chroma join operator {op_id} should not be initialized as a spout");
            }
            PO::ChannelRouter(_) => {
                error!("channel router operator {op_id} should not be initialized as a spout");
                panic!("channel router operator {op_id} should not be initialized as a spout");
            }
            PO::ChannelSpout(op) => Left(AsyncSpout::produce(op, bolts)),
            PO::Merge(_) => {
                error!("Merge operator {op_id} should not be initialized as a spout");
                panic!("Merge operator {op_id} should not be initialized as a spout");
            }
            PO::MergeSpout(op) => Right(Left(AsyncSpout::produce(op, bolts))),
            PO::UserDefinedFunction(_) => {
                error!("Merge operator {op_id} should not be initialized as a spout");
                panic!("Merge operator {op_id} should not be initialized as a spout");
            }
            PO::UserDefinedSource(op) => Right(Right(Left(AsyncSpout::produce(op, bolts)))),
            PO::PythonInlineFunction(_) => {
                // unimplemented!("inline python functions are not yet implemented");
                let err = format!(
                    "Python inline function operator {op_id} should not be called to produce as a spout"
                );
                error!("{err}");
                panic!("{err}");
            }
            PO::PythonRemoteFunction(op) => {
                Right(Right(Right(Left(AsyncSpout::produce(op, bolts)))))
            }
            PO::OnnxInferenceOperator(_) => {
                error!("onnx inference operator {op_id} should not be initialized as a spout");
                panic!("onnx inference operator {op_id} should not be initialized as a spout");
            }
            PO::Union(_) => {
                error!("Union operator {op_id} should not be initialized as a spout");
                panic!("Union operator {op_id} should not be initialized as a spout");
            }
            PO::__PLACEHOLDER__(op) => {
                // (this is a noop)
                Right(Right(Right(Right(AsyncSpout::produce(op, bolts)))))
            }
        }
    }

    async fn finalize<'this>(&'this mut self) {
        match self {
            PhysicalOperator::Project(_) => {
                error!("Project operator should not be finalized as a spout");
                panic!("Project operator should not be finalized as a spout");
            }
            PhysicalOperator::Select(_) => {
                error!("Select operator should not be finalized as a spout");
                panic!("Select operator should not be finalized as a spout");
            }
            PhysicalOperator::Join(_) => {
                error!("Join operator should not be finalized as a spout");
                panic!("Join operator should not be finalized as a spout");
            }
            PhysicalOperator::GroupBy(_) => {
                error!("GroupBy operator should not be finalized as a spout");
                panic!("GroupBy operator should not be finalized as a spout");
            }
            PhysicalOperator::DeriveValue(_) => {
                error!("DeriveValue operator should not be finalized as a spout");
                panic!("DeriveValue operator should not be finalized as a spout");
            }
            PhysicalOperator::ChromaJoin(_) => {
                error!("Chroma join operator should not be finalized as a spout");
                panic!("Chroma join operator should not be finalized as a spout");
            }
            PhysicalOperator::ChannelRouter(_) => {
                error!("channel router operator should not be finalized as a spout");
                panic!("channel router operator should not be finalized as a spout");
            }
            PhysicalOperator::ChannelSpout(op) => op.finalize().await,
            PhysicalOperator::Merge(_) => {
                error!("Merge operator should not be finalized as a spout");
                panic!("Merge operator should not be finalized as a spout");
            }
            PhysicalOperator::MergeSpout(op) => op.finalize().await,
            PhysicalOperator::UserDefinedFunction(_) => {
                error!("Merge operator should not be finalized as a spout");
                panic!("Merge operator should not be finalized as a spout");
            }
            PhysicalOperator::UserDefinedSource(op) => op.finalize().await,
            PhysicalOperator::PythonInlineFunction(_) => {
                error!("inline python function operator should not be finalized as a spout");
                unimplemented!("inline python functions are not yet implemented");
            }
            PhysicalOperator::PythonRemoteFunction(op) => op.finalize().await,
            PhysicalOperator::OnnxInferenceOperator(_) => {
                error!("onnx inference operator should not be finalized as a spout");
                panic!("onnx inference operator should not be finalized as a spout");
            }
            PhysicalOperator::Union(_) => {
                error!("Union operator should not be finalized as a spout");
                panic!("Union operator should not be finalized as a spout");
            }
            PhysicalOperator::__PLACEHOLDER__(_op) => {}
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
    OnnxInferenceOperator,
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
            PhysicalOperator::OnnxInferenceOperator(..) => OperatorType::OnnxInferenceOperator,
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
    state: AsyncPythonRemoteTaskState,
) -> (AsyncPythonRemoteTaskState, AsyncPythonRemoteTaskState) {
    let background_remote_task = AsyncPythonRemoteTaskState {
        port: state.port,
        should_stop: state.should_stop.clone(),
        runtime_handle: state.runtime_handle.clone(),
        input_to_background_thread: state.input_to_background_thread.clone(),
        // input_from_main_thread: state.input_from_main_thread.take(),
        input_from_main_thread: None,
        output_to_main_thread: state.output_to_main_thread.clone(),
        output_from_background_thread: None, // not needed
        script_name: state.script_name.clone(),
        scripts_dir_path: state.scripts_dir_path.clone(),
        pending_items: Arc::clone(&state.pending_items),
        // encode: state.encode.take(),
        // decode: state.decode.take(),
        encode: None,
        decode: None,
        // shutdown: state.shutdown.take(),
        shutdown: None,
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

        let Some(shape) = tuple.get(shape_field) else {
            error!("shape field not found");
            panic!("shape field not found")
        };
        let Some(shape) = shape.as_shape_buffer() else {
            error!("shape field is not an array");
            panic!("shape field is not an array")
        };
        let Some(array_buf) = tuple.get(array_field) else {
            error!("field not found");
            panic!("field not found")
        };
        let Some(array_buf) = array_buf.as_byte_buffer() else {
            error!("field is not a byte buffer");
            panic!("field is not a byte buffer")
        };

        let message = TensorF32Message {
            tuple_id: tuple_id as u64,
            // dims: shape.iter().map(|x| *x as u64).collect(),
            dims: shape.to_vec(),
            tensor: array_buf.as_ref(),
        };

        let mut total_buffer = Vec::<u8>::new();
        if let Err(err) = rmp_serde::encode::write(&mut total_buffer, &message) {
            let msg = format!("Failed to encode message: {:?}", err);
            error!("{}", msg);
            panic!("{}", msg);
        }
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

static STRING_NUMBERS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

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

pub type FunctionLookup = BTreeMap<HabString, FunctionKinds>;
pub enum FunctionKinds {
    SourceUdf(
        Box<
            dyn Sync
                + (Fn() -> Box<
                    dyn Send
                        + Sync
                        + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>,
                >),
        >,
    ),
    SelectFilterUdf(fn() -> Box<dyn Send + Sync + Fn(&Tuple) -> bool>),
    // JoinFilterUdf(fn() -> Box<dyn Send + Sync + Fn(&Tuple, &Tuple) -> bool>),
    JoinFilterUdf(fn(&Tuple, &Tuple) -> bool),
    // JoinEvictUdf(fn() -> Box<dyn Send + Sync + Fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool>),
    JoinEvictUdf(fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool),
    ComputationExpressionUdf(
        // TODO: change this to take a reference to a slice so that the vec space can be reused
        Box<dyn Fn() -> Box<dyn Send + Sync + Fn(Vec<&HabValue>) -> HabValue>>,
    ),
    MergeCallbackUdf(Box<dyn Fn() -> Box<dyn Send + Sync + Fn(&Tuple)>>),
    AggregationUdf(
        Box<dyn Fn() -> Box<dyn Send + Sync + Fn(&mut Queue<Tuple>) -> AggregationResult>>,
    ),
    RoutingUdf(
        Box<dyn Fn() -> Box<dyn Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize>>>,
    ),
    InlinePythonEncoder(
        // Box<dyn Fn() -> Box<dyn Sync + Send + Fn(&Tuple) -> Option<Vec<Py<PyAny>>>>>,
        Box<dyn Fn() -> crate::operators::InlineEncodeFn>,
    ),
    InlinePythonDecoder(
        // Box<dyn Fn() -> Box<dyn Sync + Send + Fn(&Py<PyAny>) -> Option<Vec<HabValue>>>>,
        Box<dyn Fn() -> crate::operators::InlineDecodeFn>,
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
    FlatMapUdf(Box<dyn Fn() -> Box<dyn Send + Sync + Fn(Tuple) -> Vec<Tuple>>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionKindDiscriminant {
    SourceUdf,
    SelectFilterUdf,
    JoinFilterUdf,
    JoinEvictUdf,
    ComputationExpressionUdf,
    MergeCallbackUdf,
    AggregationUdf,
    RoutingUdf,
    InlinePythonEncoder,
    InlinePythonDecoder,
    EncodeRemotePythonUdf,
    DecodeRemotePythonUdf,
    ShutdownRemotePythonUdf,
    FlatMapUdf,
}
impl FunctionKinds {
    fn get_kind(&self) -> FunctionKindDiscriminant {
        match self {
            FunctionKinds::SourceUdf(_) => FunctionKindDiscriminant::SourceUdf,
            FunctionKinds::SelectFilterUdf(_) => FunctionKindDiscriminant::SelectFilterUdf,
            FunctionKinds::JoinFilterUdf(_) => FunctionKindDiscriminant::JoinFilterUdf,
            FunctionKinds::JoinEvictUdf(_) => FunctionKindDiscriminant::JoinEvictUdf,
            FunctionKinds::ComputationExpressionUdf(_) => {
                FunctionKindDiscriminant::ComputationExpressionUdf
            }
            FunctionKinds::MergeCallbackUdf(_) => FunctionKindDiscriminant::MergeCallbackUdf,
            FunctionKinds::AggregationUdf(_) => FunctionKindDiscriminant::AggregationUdf,
            FunctionKinds::RoutingUdf(_) => FunctionKindDiscriminant::RoutingUdf,
            FunctionKinds::InlinePythonEncoder(_) => FunctionKindDiscriminant::InlinePythonEncoder,
            FunctionKinds::InlinePythonDecoder(_) => FunctionKindDiscriminant::InlinePythonDecoder,
            FunctionKinds::EncodeRemotePythonUdf(_) => {
                FunctionKindDiscriminant::EncodeRemotePythonUdf
            }
            FunctionKinds::DecodeRemotePythonUdf(_) => {
                FunctionKindDiscriminant::DecodeRemotePythonUdf
            }
            FunctionKinds::ShutdownRemotePythonUdf(_) => {
                FunctionKindDiscriminant::ShutdownRemotePythonUdf
            }
            FunctionKinds::FlatMapUdf(_) => FunctionKindDiscriminant::FlatMapUdf,
        }
    }
}

pub struct ChannelState {
    pub senders: Option<AsyncPipe>,
    pub receivers: Option<BoundedAsyncReceiver>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ValueSetterState {
    UseDefaults(Vec<crate::query_builder::FieldInfo>),
    // TODO: allow custom UDF to set defaults
    SetDefaultsFn,
}

pub struct RuntimeState {
    pub operators: Vec<PhysicalOperator>,
    pub runtime: Option<Runtime>,
    pub output_channels: Vec<Receiver<()>>,
    pub stop_trigger: watch::Sender<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OnnxExecutorKind {
    Cpu = 0,
    #[default]
    Cuda = 1,
}

fn load_onnx_executor_from_env() -> OnnxExecutorKind {
    let Ok(s) = std::env::var("WATERSHED_ONNX_EXECUTOR") else {
        // info!("WATERSHED_ONNX_EXECUTOR not set, defaulting to CPU");
        // return OnnxExecutorKind::Cpu;
        info!("WATERSHED_ONNX_EXECUTOR not set, defaulting to Cuda");
        return OnnxExecutorKind::Cuda;
    };
    match s.to_lowercase().trim() {
        "" | "1" | "true" | "gpu" | "cuda" => {
            info!("WATERSHED_ONNX_EXECUTOR set to Cuda");
            OnnxExecutorKind::Cuda
        }
        "0" | "false" | "cpu" => {
            info!("WATERSHED_ONNX_EXECUTOR set to CPU");
            OnnxExecutorKind::Cpu
        }
        other => {
            // error!("Invalid WATERSHED_ONNX_EXECUTOR value: {}. Defaulting to CPU.", other);
            // OnnxExecutorKind::Cpu
            error!(
                "Invalid WATERSHED_ONNX_EXECUTOR value: {:?}. Defaulting to Cuda.",
                other
            );
            OnnxExecutorKind::Cuda
        }
    }
}

pub fn json_descriptor_to_operators_with_runtime(
    src: &str,
    function_lookup: &BTreeMap<HabString, FunctionKinds>,
    input_runtime: Option<Runtime>,
    max_tuple_age_ns: Option<u128>,
) -> anyhow::Result<RuntimeState> {
    let onnx_executor_kind = load_onnx_executor_from_env();
    json_descriptor_to_operators_with_runtime_and_onnx_executor(
        src,
        function_lookup,
        input_runtime,
        max_tuple_age_ns,
        onnx_executor_kind,
    )
}
pub fn json_descriptor_to_operators_with_runtime_and_onnx_executor(
    src: &str,
    function_lookup: &BTreeMap<HabString, FunctionKinds>,
    input_runtime: Option<Runtime>,
    max_tuple_age_ns: Option<u128>,
    onnx_executor_kind: OnnxExecutorKind,
) -> anyhow::Result<RuntimeState> {
    let execution_provider = match onnx_executor_kind {
        OnnxExecutorKind::Cuda => ort::execution_providers::CUDAExecutionProvider::default()
            .build()
            .error_on_failure(),
        OnnxExecutorKind::Cpu => ort::execution_providers::CPUExecutionProvider::default()
            .build()
            .error_on_failure(),
    };
    ort::init()
        .with_execution_providers([execution_provider])
        // .with_execution_providers([
        //     ort::execution_providers::CUDAExecutionProvider::default()
        //         .build()
        //         .error_on_failure(),
        //     // ort::execution_providers::TensorRTExecutionProvider::default().build(),
        //     // ort::execution_providers::CPUExecutionProvider::default().build(),
        // ])
        .commit()
        .map_err(|e| anyhow::anyhow!("Failed to initialize ONNX runtime: {e:?}"))?;
    let mut query = serde_json::from_str::<QueryDescriptor>(src)?;
    query.operators.sort_by_key(|op| op.id);
    let mut latest_index: Option<usize> = None;
    // check for contiguous indices
    for op in &query.operators {
        let index = op.id;
        if latest_index.is_none() && index != 0 {
            error!("First operator must have index 0");
            return Err(anyhow::anyhow!("First operator must have index 0"));
        }
        if let Some(latest_index) = latest_index {
            if index != latest_index + 1 {
                let fmtstr = format!(
                    "Operator indices must be contiguous. Expected index {} but got {}",
                    latest_index + 1,
                    index
                );
                error!("{}", fmtstr);
                return Err(anyhow::anyhow!(fmtstr));
            }
        }
        latest_index = Some(index);
    }

    // check that any referenced operators are within bounds
    let operator_count = query.operators.len();
    for op in &query.operators {
        match &op.operator {
            OperatorVariantDescriptor::Project { source, .. } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Project operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("Project operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::Select {
                source,
                predicate: _,
            } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Select operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("Select operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::DeriveValue {
                source,
                new_field_name: _,
                expression: _,
            } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "DeriveValue operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("DeriveValue operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::HashJoin {
                left,
                right,
                predicate: _,
                ..
            } => {
                if *left >= operator_count {
                    return Err(anyhow::anyhow!(
                        "HashJoin operator references left operator {} which is out of bounds",
                        left
                    ));
                }

                if *left >= op.id {
                    return Err(anyhow::anyhow!("HashJoin operator references left source ({}) with larger/later id than itself ({})", left, op.id));
                }

                if *right >= operator_count {
                    return Err(anyhow::anyhow!(
                        "HashJoin operator references right operator {} which is out of bounds",
                        right
                    ));
                }
                if *right >= op.id {
                    return Err(anyhow::anyhow!("HashJoin operator references right source ({}) with larger/later id than itself ({})", right, op.id));
                }
            }
            OperatorVariantDescriptor::GroupBy { source, .. } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "GroupBy operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("GroupBy operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::ChromaJoin {
                index_stream,
                lookup_stream,
                ..
            } => {
                if *index_stream >= operator_count {
                    return Err(anyhow::anyhow!("ChromaJoin operator references index_stream operator {} which is out of bounds", index_stream));
                }
                if *index_stream >= op.id {
                    return Err(anyhow::anyhow!("ChromaJoin operator references index stream ({}) with larger/later id than itself ({})", index_stream, op.id));
                }

                if *lookup_stream >= operator_count {
                    return Err(anyhow::anyhow!("ChromaJoin operator references lookup_stream operator {} which is out of bounds", lookup_stream));
                }
                if *lookup_stream >= op.id {
                    return Err(anyhow::anyhow!("ChromaJoin operator references lookup stream ({}) with larger/later id than itself ({})", lookup_stream, op.id));
                }
            }
            OperatorVariantDescriptor::ChannelRouter {
                source,
                routes: _,
                route_expression: _,
                backup_route: _,
            } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!("ChannelRouter operator references source operator {} which is out of bounds", source));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("ChannelRouter operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::ChannelSpout { source, .. } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "ChannelSpout operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("ChannelSpout operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::Merge {
                source,
                parent,
                on_merge_fn: _,
            } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Merge operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("Merge operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
                if *parent <= op.id {
                    return Err(anyhow::anyhow!("Merge operator references parent ({}) with smaller/earlier id than itself ({})", parent, op.id));
                }
            }
            OperatorVariantDescriptor::MergeSpout { sources, .. } => {
                for source in sources {
                    if *source >= operator_count {
                        return Err(anyhow::anyhow!(
                            "MergeSpout operator references source operator {} which is out of bounds",
                            source
                        ));
                    }
                    if *source >= op.id {
                        return Err(anyhow::anyhow!("MergeSpout operator references source ({}) with larger/later id than itself ({})", source, op.id));
                    }
                }
            }
            OperatorVariantDescriptor::UserDefinedFunction { source, .. } => {
                if *source >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Merge operator references source operator {} which is out of bounds",
                        source
                    ));
                }
                if *source >= op.id {
                    return Err(anyhow::anyhow!("Merge operator references source ({}) with larger/later id than itself ({})", source, op.id));
                }
            }
            OperatorVariantDescriptor::RemotePythonUdf {
                script_name: _,
                scripts_dir_path: _,
                encode_fn: _,
                decode_fn: _,
                shutdown_fn: _,
                input,
            } => {
                if *input >= operator_count {
                    return Err(anyhow::anyhow!(
                        "RemotePythonUdf operator references input operator {} which is out of bounds",
                        input
                    ));
                }
                if *input >= op.id {
                    return Err(anyhow::anyhow!("RemotePythonUdf operator references source ({}) with larger/later id than itself ({})", input, op.id));
                }
            }
            OperatorVariantDescriptor::InlinePythonUdf { input, .. } => {
                if *input >= operator_count {
                    return Err(anyhow::anyhow!(
                        "InlinePythonUdf operator references input operator {} which is out of bounds",
                        input
                    ));
                }
                if *input >= op.id {
                    return Err(anyhow::anyhow!("InlinePythonUdf operator references source ({}) with larger/later id than itself ({})", input, op.id));
                }
            }
            OperatorVariantDescriptor::OnnxInference { source: input, .. } => {
                if *input >= operator_count {
                    return Err(anyhow::anyhow!(
                        "OnnxInferenceOperator operator references input operator {} which is out of bounds",
                        input
                    ));
                }
                if *input >= op.id {
                    return Err(anyhow::anyhow!("OnnxInferenceOperator operator references source ({}) with larger/later id than itself ({})", input, op.id));
                }
            }
            OperatorVariantDescriptor::Union { left, right } => {
                if *left >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Union operator references left operator {} which is out of bounds",
                        left
                    ));
                }
                if *left >= op.id {
                    return Err(anyhow::anyhow!("Union operator references left source ({}) with larger/later id than itself ({})", left, op.id));
                }
                if *right >= operator_count {
                    return Err(anyhow::anyhow!(
                        "Union operator references right operator {} which is out of bounds",
                        right
                    ));
                }
                if *right >= op.id {
                    return Err(anyhow::anyhow!("Union operator references right source ({}) with larger/later id than itself ({})", right, op.id));
                }
            }
            OperatorVariantDescriptor::UdfSpout { .. } => {} // no references to check in a spout
        }
    }

    let mut operator_arena: Vec<PhysicalOperator> = vec![];
    let mut operator_channel_states = BTreeMap::<usize, ChannelState>::new();
    let mut operator_value_states = BTreeMap::<usize, ValueSetterState>::new();

    // let mut runtime = None;
    let mut runtime = input_runtime;
    // let mut port_counter: u16 = 5678;
    let mut port_counter: u16 = 5556;
    // let mut port_counter: u16 = 8489;
    let mut exit_channels = Vec::new();
    let (stop_signal_sender, should_stop) = watch::channel(false);
    for op in &query.operators {
        let op_id = op.id;
        let op = construct_operator_tree_from_descriptor(
            op,
            function_lookup,
            &mut operator_channel_states,
            &mut operator_value_states,
            &mut runtime,
            onnx_executor_kind,
            &mut port_counter,
            &mut exit_channels,
            should_stop.clone(),
            max_tuple_age_ns,
        );
        if operator_arena.len() != op_id {
            return Err(anyhow::anyhow!(
                "Expected operator index {} but got {}",
                operator_arena.len(),
                op_id,
            ));
        }
        operator_arena.push(op);
    }

    for op_id in 0..operator_count {
        if op_id >= operator_arena.len() {
            let msg = format!(
                "Expected operator index {} but got {}",
                operator_arena.len(),
                op_id,
            );
            error!("{}", msg);
            return Err(anyhow::anyhow!(msg));
        }

        let arena_len = operator_arena.len();
        fn report_oob(op1_id: usize, op2_id: usize, arena_len: usize) -> anyhow::Result<usize> {
            if op2_id < arena_len {
                return Ok(op2_id);
            }
            let msg = format!(
                "Operator {} (referenced by operator {}) is out of bounds for arena of size {}",
                op2_id, op1_id, arena_len
            );
            error!("{}", msg);
            Err(anyhow::anyhow!(msg))
        }
        match operator_arena[op_id] {
            PhysicalOperator::Project(Project { child: source, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, source, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::Select(Select { child: source, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, source, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::Join(Join { left, right, .. }) => {
                let left = &mut operator_arena[report_oob(op_id, left, arena_len)?];
                left.add_parent(op_id);
                let right = &mut operator_arena[report_oob(op_id, right, arena_len)?];
                right.add_parent(op_id);
            }
            PhysicalOperator::GroupBy(GroupBy { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::DeriveValue(DeriveValue { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::ChromaJoin(ChromaJoin {
                index_stream,
                lookup_stream,
                ..
            }) => {
                let index_stream = &mut operator_arena[
                    // index_stream];
                    report_oob(op_id,index_stream, arena_len)?
                ];
                index_stream.add_parent(op_id);
                let lookup_stream = &mut operator_arena[
                    // lookup_stream];
                    report_oob(op_id,lookup_stream, arena_len)?
                ];
                lookup_stream.add_parent(op_id);
            }
            PhysicalOperator::ChannelRouter(AsyncChannelRouter { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::ChannelSpout(AsyncChannelSpout { .. }) => {} // no parents to add for a spout
            PhysicalOperator::Merge(Merge { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::MergeSpout(AsyncMergeSpout { .. }) => {} // no parents to add for a spout
            PhysicalOperator::UserDefinedFunction(UdfBolt { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::UserDefinedSource(UdfSpout { .. }) => {} // no parents to add for a spout
            PhysicalOperator::PythonInlineFunction(PythonInlineUdf { input, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, input, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::PythonRemoteFunction(PythonRemoteUdf { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::OnnxInferenceOperator(OnnxInferenceOperator { child, .. }) => {
                let source = &mut operator_arena[report_oob(op_id, child, arena_len)?];
                source.add_parent(op_id);
            }
            PhysicalOperator::Union(Union { left, right, .. }) => {
                let left = &mut operator_arena[report_oob(op_id, left, arena_len)?];
                left.add_parent(op_id);
                let right = &mut operator_arena[report_oob(op_id, right, arena_len)?];
                right.add_parent(op_id);
            }
            PhysicalOperator::__PLACEHOLDER__(..) => {
                error!("PLACEHOLDER__ should not be in the operator arena");
                unimplemented!("PLACEHOLDER__ should not be in the operator arena")
            }
        }
    }

    // Ok((operator_arena, runtime, exit_channels, stop_signal_sender))
    Ok(RuntimeState {
        operators: operator_arena,
        runtime,
        output_channels: exit_channels,
        stop_trigger: stop_signal_sender,
    })
}

pub fn construct_operator_tree_from_descriptor(
    op: &OperatorDescriptor,
    function_lookup: &BTreeMap<HabString, FunctionKinds>,
    operator_channel_states: &mut BTreeMap<usize, ChannelState>,
    operator_value_states: &mut BTreeMap<usize, ValueSetterState>,
    runtime: &mut Option<tokio::runtime::Runtime>,
    onnx_executor_kind: OnnxExecutorKind,
    port_counter: &mut u16,
    exit_channels: &mut Vec<Receiver<()>>,
    should_stop: watch::Receiver<bool>,
    max_tuple_age_ns: Option<u128>,
) -> PhysicalOperator {
    match &op.operator {
        OperatorVariantDescriptor::Project { source, fields } => {
            PhysicalOperator::Project(Project {
                id: op.id,
                child: *source,
                parent: None,
                keep_list: fields.clone(),
            })
        }
        OperatorVariantDescriptor::Select { source, predicate } => {
            let pred = if let ExpressionDescriptor::UserDefinedFunction { name, .. } = predicate {
                match function_lookup.get(name) {
                    Some(FunctionKinds::SelectFilterUdf(f)) => Some(f()),
                    Some(FunctionKinds::ComputationExpressionUdf(..)) => None,
                    None => None,
                    _ => unreachable!(
                        "Expected selection filter udf {} but it was something else",
                        name
                    ),
                }
            } else {
                None
            };
            let pred = pred.unwrap_or_else(|| {
                let realized_predicate =
                    computation_descriptor_to_computation_expression(predicate, function_lookup);
                let op_id = op.id;
                Box::new(move |tuple| {
                    let result = evaluate_computation_expression(tuple, &realized_predicate);
                    match result {
                        HabValue::Bool(b) => b,
                        _ => unreachable!(
                            "Type Error in filter expression for operator expression {op_id:?} Select predicate must be a boolean"
                        ),
                    }
                })
            });
            PhysicalOperator::Select(Select {
                id: op.id,
                child: *source,
                parent: None,
                pred,
            })
        }
        OperatorVariantDescriptor::DeriveValue {
            source,
            new_field_name,
            expression,
        } => {
            let realized_expression =
                computation_descriptor_to_computation_expression(expression, function_lookup);
            PhysicalOperator::DeriveValue(DeriveValue {
                id: op.id,
                child: *source,
                parent: None,
                fields: vec![new_field_name.clone()],
                action: realized_expression,
                new_field_name: new_field_name.clone(),
            })
        }
        OperatorVariantDescriptor::HashJoin {
            left,
            right,
            predicate,
            // eviction_policy,
            join_info,
        } => {
            let pred_expr: Option<_> =
                if let ExpressionDescriptor::UserDefinedFunction { name, .. } = predicate {
                    match function_lookup.get(name) {
                        Some(FunctionKinds::JoinFilterUdf(f)) => Some(*f),
                        Some(FunctionKinds::ComputationExpressionUdf(..)) => None,
                        None => {
                            let msg = format!("name {name} was not found in function lookup");
                            error!("{}", msg);
                            panic!("{}", msg);
                        }
                        _ => {
                            let msg = format!(
                                "Expected join filter udf {} but it was something else",
                                name
                            );
                            error!("{}", msg);
                            unreachable!("{}", msg);
                        }
                    }
                } else {
                    None
                };
            // let realized_predicate = pred_expr.unwrap_or_else(|| {
            //     let realized_predicate = computation_descriptor_to_computation_expression(predicate, function_lookup);
            //     let op_id = op.id;
            //     Box::new(move |tuple| {
            //         let left = tuple;
            //         let right = tuple;
            //         evaluate_computation_expression(&tuple, &realized_predicate)
            //             .as_bool()
            //             .expect("Expected bool as output of predicate operator")
            //     })
            // });

            let Some(realized_predicate) = pred_expr else {
                error!("We only accept function pointers for now in joins.");
                panic!("We only accept function pointers for now in joins.");
            };

            let eviction_expr = match join_info {
                HashJoinKind::DoubleStream { eviction_policy } => {
                    if let Some(ExpressionDescriptor::UserDefinedFunction { name, .. }) =
                        eviction_policy
                    {
                        match function_lookup.get(name) {
                            Some(FunctionKinds::JoinEvictUdf(f)) => Some(*f),
                            Some(FunctionKinds::ComputationExpressionUdf(..)) => None,
                            None => {
                                let msg = format!("name {name} was not found in function lookup");
                                error!("{}", msg);
                                panic!("{}", msg);
                            }
                            _ => {
                                let msg = format!(
                                    "Expected join eviction udf {} but it was something else",
                                    name
                                );
                                error!("{}", msg);
                                unreachable!("{}", msg);
                            }
                        }
                    } else {
                        None
                    }
                }
                HashJoinKind::InnerTable { .. } | HashJoinKind::OuterTable { .. } => {
                    // noop
                    Some((|_, _| false) as fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool)
                }
            };
            let Some(realized_eviction_policy) = eviction_expr else {
                error!("We only accept function pointers for now in joins.");
                panic!("We only accept function pointers for now in joins.");
            };

            PhysicalOperator::Join(Join {
                id: op.id,
                left: *left,
                right: *right,
                parent: None,
                pred: realized_predicate,
                join_info: match join_info {
                    HashJoinKind::InnerTable { fields } => JoinInner::InnerTable {
                        fields: fields.clone(),
                        build_data: DashMap::new(),
                    },
                    HashJoinKind::OuterTable { fields } => JoinInner::OuterTable {
                        fields: fields.clone(),
                        build_data: DashMap::new(),
                        build_side_fields: DashSet::new(),
                    },
                    HashJoinKind::DoubleStream { .. } => JoinInner::DoublePipeline {
                        left_inputs: DashSet::new(),
                        right_inputs: DashSet::new(),
                        evict: realized_eviction_policy,
                    },
                },
            })
        }
        OperatorVariantDescriptor::GroupBy {
            source,
            fields,
            aggregate,
        } => {
            let aggregation_expression = match aggregate {
                AggregationExpressionDescriptor::Componentized {
                    derive_decision_key,
                    should_emit,
                    derive_eviction_key,
                    should_evict,
                } => {
                    let realized_derive_decision_key =
                        computation_descriptor_to_computation_expression(
                            derive_decision_key,
                            function_lookup,
                        );
                    let realized_should_emit = computation_descriptor_to_computation_expression(
                        should_emit,
                        function_lookup,
                    );
                    let realized_derive_eviction_key =
                        computation_descriptor_to_computation_expression(
                            derive_eviction_key,
                            function_lookup,
                        );
                    let realized_should_evict = computation_descriptor_to_computation_expression(
                        should_evict,
                        function_lookup,
                    );
                    AggregationExpression::Componentized {
                        derive_decision_key: realized_derive_decision_key,
                        should_emit: realized_should_emit,
                        derive_eviction_key: realized_derive_eviction_key,
                        should_evict: realized_should_evict,
                    }
                }
                AggregationExpressionDescriptor::Udf { name } => {
                    let realized_computation = match function_lookup.get(name) {
                        Some(FunctionKinds::AggregationUdf(f)) => f(),
                        _ => unreachable!("Function {} not found in function lookup", name),
                    };
                    AggregationExpression::Udf(realized_computation)
                }
                AggregationExpressionDescriptor::Builtin { field, op } => {
                    AggregationExpression::Builtin {
                        field: field.clone(),
                        op: op.clone(),
                    }
                }
            };
            PhysicalOperator::GroupBy(GroupBy {
                id: op.id,
                child: *source,
                parent: None,
                fields: fields.clone(),
                state: DashMap::new(),
                aggregate: aggregation_expression,
            })
        }
        OperatorVariantDescriptor::ChromaJoin {
            index_stream,
            lookup_stream,
            url,
            metric,
            collection,
            distance_threshold,
            // query_n_matches,
            // keep_n_matches,
            join_info,
        } => {
            let mut client: Option<reqwest::Client> = Some(reqwest::Client::new());
            let url_out = &mut String::new();
            let connection =
                crate::chroma_utils::get_collection_id(url, collection, url_out, &mut client);
            let Some(runtime) = runtime.as_mut() else {
                error!("No runtime found when trying to use chroma join operator");
                panic!("No runtime found when trying to use chroma join operator");
            };
            let collection_id = runtime.block_on(async {
                match connection.await {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Error loading collection {collection:?}: {:?}", e);
                        panic!("Error loading collection {collection:?}: {:?}", e);
                    }
                }
            });
            let embedding_method;
            #[cfg(feature = "bert")]
            {
                use rust_bert::pipelines::sentence_embeddings;
                embedding_method = match sentence_embeddings::SentenceEmbeddingsBuilder::remote(
                    sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2,
                )
                .create_model()
                {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Error creating sentence embeddings model: {:?}", e);
                        panic!("Error creating sentence embeddings model: {:?}", e);
                    }
                }
            }
            #[cfg(not(feature = "bert"))]
            {
                error!("Chroma join operator requires the 'bert' feature to be enabled");
                panic!("Chroma join operator requires the 'bert' feature to be enabled");
                embedding_method = crate::DummySentenceEmbeddingsModel;
            };
            PhysicalOperator::ChromaJoin(ChromaJoin {
                id: op.id,
                index_stream: *index_stream,
                lookup_stream: *lookup_stream,
                parent: None,
                metric: *metric,
                embedding_method: Mutex::new(embedding_method),
                client,
                chroma_url: url.clone(),
                collection_id: collection_id.into(),
                distance_threshold: *distance_threshold,
                join_info: join_info.clone(),
                // query_n_matches: *query_n_matches,
                // keep_n_matches: *keep_n_matches,
            })
        }
        OperatorVariantDescriptor::ChannelRouter {
            source,
            routes,
            route_expression,
            backup_route,
        } => {
            let (mut txs, backup_sender) = match backup_route {
                crate::query_builder::BackupRouteDescriptor::Drop => (vec![AsyncPipe::Dummy], None),
                crate::query_builder::BackupRouteDescriptor::ForwardWithValues {
                    merge_spout_position,
                    values_to_set,
                } => {
                    operator_value_states.insert(
                        *merge_spout_position,
                        ValueSetterState::UseDefaults(values_to_set.clone()),
                    );
                    let tx = match operator_channel_states.entry(*merge_spout_position) {
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            let state = entry.get_mut();
                            state.senders.clone().unwrap_or_else(|| {
                                error!(
                                    "No sender found for merge_spout_position key {merge_spout_position} of merge operator {}",
                                    op.id
                                );
                                panic!(
                                    "No sender found for merge_spout_position key {merge_spout_position} of merge operator {}",
                                    op.id
                                )
                            })
                        }
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            // let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Tuple>>();
                            let (tx, rx) = crate::scheduler::bounded_channel(
                                // crate::caching::DEFAULT_TUPLES,
                                // 4,
                                10_000,
                                max_tuple_age_ns.unwrap_or(u128::MAX), //1_000_000_000,
                            );
                            entry.insert(ChannelState {
                                senders: Some(tx.clone()),
                                receivers: Some(rx),
                            });
                            tx
                        }
                    };
                    let wrapped_tx = tx.clone();
                    let tx = match tx {
                        AsyncPipe::Active(bounded_async_sender) => {
                            BackupChannel::BoundedAndDrop(bounded_async_sender.extract_sender())
                        }
                        AsyncPipe::Dummy => BackupChannel::Dummy,
                    };
                    (vec![wrapped_tx], Some(tx))
                }
                crate::query_builder::BackupRouteDescriptor::CustomRoute(
                    AsyncChannelDescriptor {
                        max_capacity,
                        spout_position,
                    },
                ) => {
                    let (tx, rx) = crate::scheduler::bounded_channel(
                        max_capacity.unwrap_or(10_000),
                        u128::MAX,
                    );
                    let wrapped_tx = tx.clone();
                    let tx = match tx {
                        AsyncPipe::Active(bounded_async_sender) => {
                            BackupChannel::BoundedAndDrop(bounded_async_sender.extract_sender())
                        }
                        AsyncPipe::Dummy => BackupChannel::Dummy,
                    };
                    operator_channel_states.insert(
                        *spout_position,
                        ChannelState {
                            receivers: Some(rx),
                            senders: None,
                        },
                    );
                    (vec![wrapped_tx], Some(tx))
                }
            };
            let mut rxs = vec![];
            // let (tx, rx) = crossbeam::channel::unbounded::<Vec<Tuple>>();
            // let out_channels = vec![tx; routes.len()];
            for _ in routes {
                // let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Tuple>>();
                let (tx, rx) = crate::scheduler::bounded_backup_channel(
                    // we can use a small number to try to make the drop rate line up as needed
                    // 4,
                    // 1,
                    10_000,
                    // 100,
                    max_tuple_age_ns.unwrap_or(u128::MAX), //1_000_000_000,
                    backup_sender.clone(),
                );
                txs.push(tx);
                rxs.push(rx);
            }
            let route_function =
                if let ExpressionDescriptor::UserDefinedFunction { name, args: _ } =
                    route_expression
                {
                    // Box<dyn Send + Sync + FnMut(Vec<Tuple>, &[Sender<Vec<Tuple>>]) -> Option<usize>>
                    match function_lookup.get(name) {
                        Some(FunctionKinds::RoutingUdf(f)) => Some(f()),
                        None => unreachable!("Function {} not found in function lookup", name),
                        _ => None,
                    }
                } else {
                    None
                };
            let route_function = route_function.unwrap_or_else(|| {
                fn create_router_from_comp_expr(
                    computation_expression: ComputationExpression,
                ) -> Box<
                    dyn Send
                        + Sync
                        + FnMut(
                            Vec<Tuple>,
                            &[AsyncPipe],
                        ) -> Option<usize>,
                > {
                    Box::new(move |inputs, output_channels| {
                        let mut num_emitted = 0;
                        for tuple in inputs {
                            let result =
                                evaluate_computation_expression(&tuple, &computation_expression);
                            if let HabValue::Integer(i) = result {
                                if i < 0 {
                                    // drop the value by not sending it anywhere
                                    continue;
                                }
                                let index = i as usize;
                                if index < output_channels.len() {
                                    let mut tuple_vec = get_tuple_vec();
                                    tuple_vec.push(tuple);
                                    let Some(sender) =
                                        output_channels.get(index)
                                    else {
                                        let msg = format!("Channel routing expression tried to send on a non-existent channel at index {index}");
                                        error!("{}", msg);
                                        // panic!("{}", msg);
                                        continue;
                                    };
                                    if let Err(err) = sender.send(tuple_vec) {
                                        let msg = format!("unable to send on channel in router: {err:?}");
                                        error!("{}", msg);
                                        // panic!("{}", msg);
                                    }
                                    num_emitted += 1;
                                }
                            } else {
                                warn!("Channel routing expression must return an integer (use -1 if the decision is to drop)");
                            }
                        }
                        if num_emitted > 0 {
                            Some(num_emitted)
                        } else {
                            None
                        }
                    })
                }
                let computation_expression = computation_descriptor_to_computation_expression(
                    route_expression,
                    function_lookup,
                );
                create_router_from_comp_expr(computation_expression)
            });
            for (route_id, rx) in routes.iter().zip(rxs) {
                operator_channel_states.insert(
                    *route_id,
                    ChannelState {
                        receivers: Some(rx),
                        senders: None,
                    },
                );
            }
            let out_channels = txs;

            PhysicalOperator::ChannelRouter(AsyncChannelRouter {
                id: op.id,
                child: *source,
                parent_channels: Arc::new(out_channels),
                route: Arc::new(Mutex::new(route_function)),
            })
        }
        OperatorVariantDescriptor::ChannelSpout {
            source: _,
            timeout,
            max_age_ns,
        } => {
            // let id_to_lookup = *source;
            let id_to_lookup = op.id;
            let state = operator_channel_states
                .get_mut(&id_to_lookup)
                .unwrap_or_else(|| {
                    // panic!(
                    //     "No receiver found for source key {source} of channel spout operator {}",
                    //     op.id
                    // )
                    let msg = format!("No receiver found for channel spout operator {}", op.id);
                    error!("{}", msg);
                    panic!("{}", msg);
                });
            let Some(mut rx) = state.receivers.take() else {
                let msg = format!("No receiver found for channel spout operator {}", op.id);
                error!("{}", msg);
                panic!("{}", msg);
            };
            if let Some(max_age_ns) = max_age_ns {
                rx.max_age_ns = rx.max_age_ns.min(*max_age_ns as _);
            }
            PhysicalOperator::ChannelSpout(AsyncChannelSpout {
                id: op.id,
                parent: None,
                input: rx,
                timeouts: From::from(*timeout),
            })
        }
        OperatorVariantDescriptor::Merge {
            source,
            parent,
            on_merge_fn,
        } => {
            let tx = match operator_channel_states.entry(*parent) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let state = entry.get_mut();
                    state.senders.clone().unwrap_or_else(|| {
                        error!(
                            "No sender found for parent key {parent} of merge operator {}",
                            op.id
                        );
                        panic!(
                            "No sender found for parent key {parent} of merge operator {}",
                            op.id
                        )
                    })
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    // let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Tuple>>();
                    let (tx, rx) = crate::scheduler::bounded_channel(
                        // crate::caching::DEFAULT_TUPLES,
                        // 4,
                        10_000,
                        max_tuple_age_ns.unwrap_or(u128::MAX), //1_000_000_000,
                    );
                    entry.insert(ChannelState {
                        senders: Some(tx.clone()),
                        receivers: Some(rx),
                    });
                    tx
                }
            };
            let on_merge_fn = on_merge_fn.as_ref().map(|fn_name|{
                match function_lookup.get(&**fn_name) {
                    Some(FunctionKinds::MergeCallbackUdf(f)) => Box::new(f()) as _,
                    None => unreachable!("Function {} not found in function lookup", fn_name),
                    _ => unreachable!(
                        "Expected merge callback function {} to be an MergeCallbackUdf but it was something else",
                        fn_name
                    ),
                }
            });
            PhysicalOperator::Merge(Merge {
                id: op.id,
                child: *source,
                parent_channel: tx,
                on_merge_fn,
            })
        }
        OperatorVariantDescriptor::MergeSpout {
            sources: _,
            timeout,
            max_age_ns,
            // on_merge_fn,
        } => {
            // get the function to call when a tuple is received
            let Some(state) = operator_channel_states.get_mut(&op.id) else {
                error!("No receiver found for merge spout operator {}", op.id);
                panic!("No receiver found for merge spout operator {}", op.id);
            };
            let value_setter = operator_value_states.get(&op.id).cloned();

            let Some(mut rx) = state.receivers.take() else {
                let msg = format!("No receiver found for merge spout operator {}", op.id);
                error!("{}", msg);
                panic!("{}", msg);
            };
            if let Some(max_age_ns) = max_age_ns {
                rx.max_age_ns = rx.max_age_ns.min(*max_age_ns as _);
            }
            PhysicalOperator::MergeSpout(AsyncMergeSpout {
                id: op.id,
                parent: None,
                input: rx,
                timeouts: From::from(*timeout),
                value_setter,
            })
        }
        OperatorVariantDescriptor::UserDefinedFunction { source, name } => {
            let Some(func) = function_lookup.get(name) else {
                error!("failed to get UDF with name {name:?} from lookup");
                panic!("failed to get UDF with name {name:?} from lookup");
            };
            let process = match func {
                FunctionKinds::FlatMapUdf(f) => f(),
                _ => {
                    let kind = func.get_kind();
                    error!("UDF {name:?} was expected to be a flatmap for UDF operator but was instead {kind:?}");
                    panic!("UDF {name:?} was expected to be a flatmap for UDF operator but was instead {kind:?}");
                }
            };
            PhysicalOperator::UserDefinedFunction(UdfBolt {
                id: op.id,
                child: *source,
                parent: None,
                process: Arc::new(process),
            })
        }
        OperatorVariantDescriptor::RemotePythonUdf {
            script_name,
            scripts_dir_path,
            encode_fn,
            decode_fn,
            shutdown_fn,
            input,
        } => {
            // keep the queue having exactly one thing going out at a time.
            //  this will limit the parent operators from consuming more values
            //  which will in turn cause other queues upstream to be backed up if this operator cannot process items fast enough
            // TODO: should this be one of our custom pipes?
            // we can have at most 1 python operation running per operator, so we must block on that
            let (outgoing_tx, outgoing_rx) = tokio::sync::mpsc::channel::<Vec<Tuple>>(1);
            // let (outgoing_tx, outgoing_rx) = crate::scheduler::bounded_channel(
            //     1,
            //     1_000_000_000,
            // );
            // let (outgoing_tx, outgoing_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Tuple>>();
            let (incoming_tx, incoming_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<Tuple>>();

            let encode_fn = match function_lookup.get(encode_fn) {
                        Some(FunctionKinds::EncodeRemotePythonUdf(f)) => Some(f()),
                        None => unreachable!("Function {} not found in function lookup", encode_fn),
                        _ => unreachable!(
                            "Expected encode function {} to be an EncodeRemotePythonUdf but it was something else",
                            encode_fn
                        ),
            };

            let decode_fn = match function_lookup.get(decode_fn) {
                        Some(FunctionKinds::DecodeRemotePythonUdf(f)) => Some(f()),
                        None => unreachable!("Function {} not found in function lookup", decode_fn),
                        _ => unreachable!(
                            "Expected decode function {} to be an DecodeRemotePythonUdf but it was something else",
                            decode_fn
                        ),
            };

            let shutdown_fn = match function_lookup.get(shutdown_fn) {
                        Some(FunctionKinds::ShutdownRemotePythonUdf(f)) => Some(f()),
                        None => unreachable!("Function {} not found in function lookup", shutdown_fn),
                        _ => unreachable!(
                            "Expected shutdown function {} to be an ShutdownRemotePythonUdf but it was something else",
                            shutdown_fn
                        ),
            };
            let port = *port_counter;
            *port_counter += 1;
            let task_state = AsyncPythonRemoteTaskState {
                port,
                should_stop: should_stop.clone(),
                runtime_handle: runtime
                    .get_or_insert_with(|| {
                        match tokio::runtime::Builder::new_multi_thread()
                            .worker_threads(2)
                            .enable_all()
                            .build()
                        {
                            Ok(r) => r,
                            Err(e) => {
                                error!("Unable to create tokio runtime: {:?}", e);
                                panic!("Unable to create tokio runtime: {:?}", e);
                            }
                        }
                    })
                    .handle()
                    .clone(),
                input_to_background_thread: outgoing_tx,
                input_from_main_thread: Some(outgoing_rx),
                output_to_main_thread: incoming_tx,
                output_from_background_thread: Some(incoming_rx),
                script_name: script_name.clone(),
                scripts_dir_path: scripts_dir_path.clone(),
                pending_items: Arc::new(From::from(0)),
                encode: encode_fn.map(EncoderFunction::Single),
                decode: decode_fn,
                shutdown: shutdown_fn,
            };
            let (background_exit_sender, runtime_exit_receiver) =
                tokio::sync::mpsc::unbounded_channel();
            exit_channels.push(runtime_exit_receiver);
            PhysicalOperator::PythonRemoteFunction(PythonRemoteUdf {
                id: op.id,
                initialized: false,
                script_name: script_name.clone(),
                scripts_dir_path: scripts_dir_path.clone(),
                parent: None,
                child: *input,
                task_state,
                exit_channel: background_exit_sender,
            })
        }
        OperatorVariantDescriptor::InlinePythonUdf {
            scripts_dir_path,
            script_name,
            function_name,
            encode_fn,
            decode_fn,
            input,
        } => {
            let func_result = Python::with_gil(|py| -> PyResult<Py<_>> {
                use pyo3::types::PyModule;
                // other data should be imported and warmed up by the scripts themselves
                start_python_with_modules(&["os", "numpy"]);
                let mut script_path = std::path::PathBuf::from(scripts_dir_path.as_str());
                script_path.push(script_name.as_str());
                if !script_name.as_str().ends_with(".py") {
                    script_path.set_extension("py");
                }
                let code = match std::fs::read(&script_path) {
                    Ok(code) => code,
                    Err(e) => {
                        error!("Error reading script file {:?}: {:?}", script_path, e);
                        Err(e)?
                    }
                };

                let pymod = PyModule::from_code(
                    py,
                    &std::ffi::CString::new(code)?,
                    &std::ffi::CString::new(format!("inline_python_script_{}", op.id))?,
                    &std::ffi::CString::new(format!("inline_python_module_{}", op.id))?,
                )?;
                let udf: Py<PyAny> = pymod.getattr(function_name.as_str())?.unbind();
                Ok(udf)
            });
            let func = match func_result {
                Ok(func) => func,
                Err(e) => {
                    error!("Error creating python function: {:?}", e);
                    panic!("Error creating python function: {:?}", e);
                }
            };
            use crate::operators::{PythonDecodingMethod, PythonEncodingMethod};
            use crate::query_builder::{InlinePythonDecodeKind, InlinePythonEncodeKind};
            let encode_fn = match encode_fn {
                InlinePythonEncodeKind::Custom {
                    function_name,
                    fields,
                } => {
                    let function = match function_lookup.get(function_name) {
                        Some(FunctionKinds::InlinePythonEncoder(f)) => f(),
                        None => {
                            let err =
                                format!("Function {} not found in function lookup", function_name);
                            error!("{err}");
                            panic!("{err}");
                        }
                        _ => {
                            let err = format!(
                                "Expected encode function {} to be an EncodeRemotePythonUdf but it was something else",
                                function_name
                            );
                            error!("{err}");
                            panic!("{err}");
                        }
                    };
                    PythonEncodingMethod::CustomEncoder {
                        func: function,
                        fields: fields.clone(),
                    }
                }
                InlinePythonEncodeKind::Default { fields } => {
                    PythonEncodingMethod::HabValueToPyAny {
                        fields: fields.clone(),
                    }
                }
                InlinePythonEncodeKind::PythonValues { fields } => {
                    PythonEncodingMethod::PythonValues {
                        fields: fields.clone(),
                    }
                }
            };

            let decode_fn = match decode_fn {
                InlinePythonDecodeKind::Custom {
                    function_name,
                    output_fields,
                } => {
                    let function = match function_lookup.get(function_name) {
                        Some(FunctionKinds::InlinePythonDecoder(f)) => f(),
                        None => {
                            let err =
                                format!("Function {} not found in function lookup", function_name);
                            error!("{err}");
                            panic!("{err}");
                        }
                        Some(other) => {
                            let err = format!(
                                "Expected decode function {} to be an DecodeRemotePythonUdf but it was {:?}",
                                function_name, other.get_kind(),
                            );
                            error!("{err}");
                            panic!("{err}");
                        }
                    };
                    PythonDecodingMethod::CustomDecoder {
                        func: function,
                        fields: output_fields.clone(),
                    }
                }
                InlinePythonDecodeKind::Default { output_fields } => {
                    PythonDecodingMethod::PyAnyToHabValues {
                        fields: output_fields.clone(),
                    }
                }
                InlinePythonDecodeKind::PythonValues { output_fields } => {
                    PythonDecodingMethod::PythonValues {
                        fields: output_fields.clone(),
                    }
                }
            };
            PhysicalOperator::PythonInlineFunction({
                PythonInlineUdf {
                    id: op.id,
                    input: *input,
                    parent: None,
                    script_name: script_name.clone(),
                    scripts_dir_path: scripts_dir_path.clone(),
                    function_name: function_name.clone(),
                    func,
                    encoder: encode_fn,
                    decoder: decode_fn,
                }
            })
        }
        OperatorVariantDescriptor::OnnxInference {
            source,
            model_path,
            args,
            outputs,
        } => {
            debug!(
                "OnnxG0: creating ONNX inference operator {} with model path {}",
                op.id, model_path
            );
            let builder = match ort::session::Session::builder() {
                Ok(b) => b,
                Err(e) => {
                    error!(
                        "Error creating ONNX session builder for operator {}: {:?}",
                        op.id, e
                    );
                    panic!(
                        "Error creating ONNX session builder for operator {}: {:?}",
                        op.id, e
                    );
                }
            };
            // we can't set the execution provider more than once, which apparently is not done per-model
            // let builder = match builder.with_execution_providers([
            //     ort::execution_providers::CUDAExecutionProvider::default().build(),
            //     ort::execution_providers::CPUExecutionProvider::default().build(),
            // ]) {
            //     Ok(b) => b,
            //     Err(e) => {
            //         error!(
            //             "Error setting ONNX session execution provider for operator {}: {:?}",
            //             op.id, e
            //         );
            //         panic!(
            //             "Error setting ONNX session execution provider for operator {}: {:?}",
            //             op.id, e
            //         );
            //     }
            // };

            debug!(
                "OnnxG1: setting optimization level and intra threads for ONNX inference operator {} with model path {}",
                op.id, model_path
            );
            let builder = match builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            {
                Ok(b) => b,
                Err(e) => {
                    error!(
                        "Error setting ONNX session optimization level for operator {}: {:?}",
                        op.id, e
                    );
                    panic!(
                        "Error setting ONNX session optimization level for operator {}: {:?}",
                        op.id, e
                    );
                }
            };
            const THREADS_DEFAULT: usize = 4;
            let intra_threads = std::env::var("WATERSHED_ONNX_INTRA_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(THREADS_DEFAULT);
            let builder = match builder.with_intra_threads(intra_threads) {
                Ok(b) => b,
                Err(e) => {
                    error!(
                        "Error setting ONNX session intra threads for operator {}: {:?}",
                        op.id, e
                    );
                    panic!(
                        "Error setting ONNX session intra threads for operator {}: {:?}",
                        op.id, e
                    );
                }
            };

            use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
            let can_use_pinned_memory = onnx_executor_kind == OnnxExecutorKind::Cuda;
            let use_pinned_memory = 'use_pinned_memory: {
                let Ok(set_val) = std::env::var("WATERSHED_ONNX_USE_PINNED_MEMORY") else {
                    break 'use_pinned_memory false;
                };
                if !(set_val == "1" || set_val.eq_ignore_ascii_case("true")) {
                    break 'use_pinned_memory false;
                }
                // user wants to use pinned memory, but we can only do that if the executor kind is Cuda
                if !can_use_pinned_memory {
                    warn!("WATERSHED_ONNX_USE_PINNED_MEMORY was set to true, but the ONNX executor kind is not Cuda, so pinned memory cannot be used; ignoring the setting and using regular memory");
                    break 'use_pinned_memory false;
                }
                true
            };
            let builder = if use_pinned_memory {
                debug!(
                    "OnnxG2(A): setting allocator for ONNX inference operator {} with model path {}",
                    op.id, model_path
                );
                let allocator = match MemoryInfo::new(
                    AllocationDevice::CUDA_PINNED,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        error!(
                            "Error creating ONNX memory info for operator {}: {:?}",
                            op.id, e
                        );
                        panic!(
                            "Error creating ONNX memory info for operator {}: {:?}",
                            op.id, e
                        );
                    }
                };
                let builder = match builder.with_allocator(allocator) {
                    Ok(b) => b,
                    Err(e) => {
                        error!(
                            "Error setting ONNX session allocator for operator {}: {:?}",
                            op.id, e
                        );
                        panic!(
                            "Error setting ONNX session allocator for operator {}: {:?}",
                            op.id, e
                        );
                    }
                };
                builder
            } else {
                debug!(
                    "OnnxG2(B): not setting allocator for ONNX inference operator {} with model path {}",
                    op.id, model_path
                );
                builder
            };

            debug!(
                "OnnxG3: committing ONNX session from file for ONNX inference operator {} with model path {}",
                op.id, model_path
            );
            let mut model = match builder.commit_from_file(model_path.as_str()) {
                Ok(m) => m,
                Err(e) => {
                    error!(
                        "Error creating ONNX session for operator {}: {:?}",
                        op.id, e
                    );
                    panic!(
                        "Error creating ONNX session for operator {}: {:?}",
                        op.id, e
                    );
                }
            };
            debug!(
                "OnnxG4: preparing to warm up ONNX model for operator {} with path {}",
                op.id, model_path
            );
            let iters = ORT_DEFALT_ITERS.load(atomic::Ordering::Relaxed);
            const BATCH_SIZE_IF_UNKNOWN: usize = 1;
            let batch_size_if_unknown = 'batch_size_if_unknown: {
                if let Some(size) = std::env::var("WATERSHED_ONNX_BATCH_SIZE_IF_UNKNOWN")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    if size == 0 {
                        warn!("WATERSHED_ONNX_BATCH_SIZE_IF_UNKNOWN was set to 0, which is invalid, so using default of {}", BATCH_SIZE_IF_UNKNOWN);
                        break 'batch_size_if_unknown BATCH_SIZE_IF_UNKNOWN;
                    }
                    break 'batch_size_if_unknown size;
                }
                if let Some(sizes_json) =
                    std::env::var("WATERSHED_ONNX_BATCH_SIZES")
                        .ok()
                        .and_then(|s| {
                            serde_json::from_str::<std::collections::HashMap<usize, usize>>(&s).ok()
                        })
                {
                    if let Some(size) = sizes_json.get(&op.id) {
                        if *size == 0 {
                            warn!("WATERSHED_ONNX_BATCH_SIZES had an entry of 0 for operator {}, which is invalid, so using default of {}", op.id, BATCH_SIZE_IF_UNKNOWN);
                            break 'batch_size_if_unknown BATCH_SIZE_IF_UNKNOWN;
                        }
                        break 'batch_size_if_unknown *size;
                    }
                }
                BATCH_SIZE_IF_UNKNOWN
            };
            // batch size is the minimum of the arguments, or BATCH_SIZE_IF_UNKNOWN if none are known
            let mut known_batch_size = usize::MAX;
            for input in model.inputs.iter() {
                let ort::value::ValueType::Tensor {
                    ty,
                    shape: dimensions,
                    // dimensions: dimensions,
                    dimension_symbols: _,
                } = &input.input_type
                else {
                    let error_msg = format!("Expected a tensor input type for input {:?}", input);
                    error!("{error_msg}");
                    panic!("{error_msg}");
                };
                if ty != &ort::tensor::TensorElementType::Float32 {
                    let error_msg =
                        format!("Expected a Float32 tensor input type for input {:?}", input);
                    error!("{error_msg}");
                    panic!("{error_msg}");
                }
                if dimensions.len() < 1 {
                    let error_msg =
                        format!("Expected at least one dimension for input {:?}", input);
                    error!("{error_msg}");
                    panic!("{error_msg}");
                }
                let dim0 = dimensions[0];
                if dim0 > 0 {
                    known_batch_size = known_batch_size.min(dim0 as usize);
                }
            }
            if known_batch_size == usize::MAX {
                known_batch_size = batch_size_if_unknown;
            }
            debug!(
                "OnnxG5: using batch size {} for warming up ONNX model for operator {} with path {}",
                known_batch_size, op.id, model_path
            );
            let input_buffers = match warm_up_model(
                iters,
                &mut model,
                DimensionDefaults::Uniform(known_batch_size),
            ) {
                Ok(b) => b,
                Err(e) => {
                    let err = format!(
                        "Error warming up ONNX model for operator {}: {:?}",
                        op.id, e
                    );
                    error!("{err}");
                    panic!("{err}");
                }
            };
            debug!(
                "OnnxG6: completed warm up of ONNX model for operator {} with path {}, receiving {} cached input buffers",
                op.id, model_path, input_buffers.len(),
            );
            PhysicalOperator::OnnxInferenceOperator(OnnxInferenceOperator {
                id: op.id,
                child: *source,
                parent: None,
                model_path: model_path.clone(),
                batch_size: known_batch_size,
                onnx_session: std::sync::Arc::new(parking_lot::Mutex::new(model)),
                args: args
                    .iter()
                    .zip(input_buffers)
                    .map(|(arg_descriptor, buffer)| match arg_descriptor.clone() {
                        crate::query_builder::NdArrayDescriptor::SplitParts {
                            shape_field,
                            buffer_field,
                        } => crate::async_operators::OnnxValue::SplitParts {
                            shape_field,
                            buffer_field,
                            buffer: parking_lot::Mutex::new(Some(buffer)),
                        },
                    })
                    .collect(),
                outputs: outputs
                    .iter()
                    .map(|output_descriptor| match output_descriptor.clone() {
                        crate::query_builder::NdArrayDescriptor::SplitParts {
                            shape_field,
                            buffer_field,
                        } => crate::async_operators::OnnxValue::SplitParts {
                            shape_field: shape_field.clone(),
                            buffer_field: buffer_field.clone(),
                            buffer: parking_lot::Mutex::new(None),
                        },
                    })
                    .collect(),
            })
        }
        OperatorVariantDescriptor::Union { left, right } => PhysicalOperator::Union(Union {
            id: op.id,
            left: *left,
            right: *right,
            parent: None,
        }),
        OperatorVariantDescriptor::UdfSpout { name } => {
            PhysicalOperator::UserDefinedSource(UdfSpout {
                id: op.id,
                parent: None,
                init_state: crate::async_operators::InitializationState::NotInitialized {
                    background_task: match function_lookup.get(name) {
                        Some(FunctionKinds::SourceUdf(f)) => f(),
                        _ => unreachable!("Function {} not found in function lookup", name),
                    },
                },
            })
        }
    }
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
                Some(InlinePythonEncoder(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as inline python encoder"),
                Some(InlinePythonDecoder(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as inline python decoder"),
                Some(FlatMapUdf(..)) => panic!("Registered UDF was supposed to be a computation expression, but was a registered as a flatmap for a udf operation"),
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

fn split_bolts_and_spouts(
    operators: Vec<PhysicalOperator>,
) -> (Vec<PhysicalOperator>, Vec<PhysicalOperator>) {
    let mut bolts: Vec<PhysicalOperator> = Vec::with_capacity(operators.len());
    let mut spouts: Vec<PhysicalOperator> = Vec::with_capacity(operators.len());
    for op in operators {
        fn push_bolt(
            id: usize,
            bolt: PhysicalOperator,
            bolts: &mut Vec<PhysicalOperator>,
            spouts: &mut Vec<PhysicalOperator>,
        ) {
            bolts.push(bolt);
            spouts.push(PO::__PLACEHOLDER__(DummyBolt(id)));
        }
        fn push_spout(
            id: usize,
            spout: PhysicalOperator,
            bolts: &mut Vec<PhysicalOperator>,
            spouts: &mut Vec<PhysicalOperator>,
        ) {
            spouts.push(spout);
            bolts.push(PO::__PLACEHOLDER__(DummyBolt(id)));
        }

        use PhysicalOperator as PO;
        let op_id = op.get_id();
        match op {
            // normal bolts all get added into the shared list
            PO::Project(_)
            | PO::Select(_)
            | PO::Join(_)
            | PO::Union(_)
            | PO::GroupBy(_)
            | PO::DeriveValue(_)
            | PO::ChromaJoin(_)
            | PO::ChannelRouter(_)
            | PO::Merge(_)
            | PO::UserDefinedFunction(_)
            | PO::PythonInlineFunction(_)
            | PO::OnnxInferenceOperator(_) => push_bolt(op_id, op, &mut bolts, &mut spouts),
            PO::ChannelSpout(_) | PO::MergeSpout(_) | PO::UserDefinedSource(_) => {
                push_spout(op_id, op, &mut bolts, &mut spouts)
            }
            PO::PythonRemoteFunction(op) => {
                let (background_state, foreground_state) = split_async_python_state(op.task_state);
                // TODO: should these just be split into two different operators altogether?
                //  they currently have to try to tell if they are the input or the spout, whereas that could be baked in by construction
                let foreground = PythonRemoteUdf {
                    id: op.id,
                    // initialized: true,
                    initialized: false,
                    script_name: op.script_name.clone(),
                    scripts_dir_path: op.scripts_dir_path.clone(),
                    parent: op.parent,
                    child: op.child,
                    task_state: foreground_state,
                    exit_channel: op.exit_channel.clone(),
                };
                let background = PythonRemoteUdf {
                    id: op.id,
                    initialized: true,
                    script_name: op.script_name,
                    scripts_dir_path: op.scripts_dir_path,
                    parent: op.parent,
                    child: op.child,
                    task_state: background_state,
                    exit_channel: op.exit_channel,
                };

                debug!(
                    "in split bolts and spouts, background has {}",
                    background.task_state.input_from_main_thread.is_some()
                );
                spouts.push(PO::PythonRemoteFunction(foreground));
                bolts.push(PO::PythonRemoteFunction(background));
            }
            PO::__PLACEHOLDER__(_op) => {
                unimplemented!("there should not be a dummy bolt in the topology before starting")
            }
        }
    }
    (bolts, spouts)
}

pub static ORT_DEFALT_ITERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(10);

const WARM_UP_SEED: u64 = u64::MAX / 42;
enum DimensionDefaults {
    Uniform(usize),
    Individual(smallvec::SmallVec<[usize; 4]>),
}
impl Default for DimensionDefaults {
    fn default() -> Self {
        Self::Uniform(1)
    }
}

fn warm_up_model(
    iters: usize,
    model: &mut ort::session::Session,
    dimension_defaults: DimensionDefaults,
) -> Result<Vec<ort::value::Tensor<f32>>, anyhow::Error> {
    use anyhow::Context;
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(WARM_UP_SEED);
    let mut args: Vec<ort::value::Tensor<f32>> = Vec::new();
    for input in &model.inputs {
        debug!("found input during warmup: {:?}", input.input_type);
        let ort::value::ValueType::Tensor {
            ty,
            shape: dimensions,
            // dimensions: dimensions,
            dimension_symbols: _,
        } = &input.input_type
        else {
            anyhow::bail!("Expected a tensor input type for input {:?}", input);
        };
        if ty != &ort::tensor::TensorElementType::Float32 {
            anyhow::bail!("Expected a Float32 tensor input type for input {:?}", input);
        }

        let dimensions = match &dimension_defaults {
            DimensionDefaults::Uniform(default_val) => dimensions
                .iter()
                .map(|&dim| if dim < 0 { *default_val as _ } else { dim as _ })
                .collect::<Vec<_>>(),
            DimensionDefaults::Individual(defaults) => {
                let max_dims = defaults.len();
                dimensions
                    .iter()
                    .zip_longest(defaults.iter().take(max_dims))
                    .enumerate()
                    .map(|(_dimension_position, pair)| match pair {
                        itertools::EitherOrBoth::Both(dim, default_val) => {
                            if *dim < 0 {
                                *default_val
                            } else {
                                *dim as _
                            }
                        }
                        itertools::EitherOrBoth::Left(dim) => {
                            if *dim < 0 {
                                1
                            } else {
                                *dim as _
                            }
                        }
                        itertools::EitherOrBoth::Right(default_val) => *default_val,
                    })
                    .collect::<Vec<_>>()
            }
        };
        debug!("using dimensions {:?} for input {:?}", dimensions, input);
        let allocator = model.allocator();
        let mut tensor = ort::value::Tensor::<f32>::new(allocator, dimensions)
            .context("Failed to create tensor during warmup")?;
        let mut view_mut: ndarray::ArrayViewMutD<'_, f32> = tensor
            .try_extract_array_mut()
            .context("Failed to get mutable array view of tensor during warmup")?;
        // let mut arr = ndarray::ArrayD::<f32>::zeros(dimensions);
        // for val in arr.iter_mut() {
        for val in view_mut.iter_mut() {
            *val = rng.gen_range(0.0..1.0);
        }
        // let tensor_val =
        //     ort::value::Tensor::from_array(arr).context("Failed to convert array to tensor")?;
        // args.push(ort::session::SessionInputValue::Owned(
        //     tensor_val.into_dyn(),
        // ));
        // args.push(arr)
        args.push(tensor)
    }

    for warmup_iter in 0..iters {
        use rand::seq::SliceRandom;
        let args: Vec<_> = args
            .iter_mut()
            .map(|tensor| {
                // arr.as_slice_mut()
                //     .context("Failed to get mutable slice of array")?
                //     .shuffle(&mut rng);
                let mut arr: ndarray::ArrayViewMutD<'_, f32> = tensor
                    .try_extract_array_mut()
                    .context("Failed to get mutable array view of tensor during warmup")?;
                arr.as_slice_mut()
                    .context("Failed to get mutable slice of array")?
                    .shuffle(&mut rng);
                Ok::<_, anyhow::Error>(ort::session::SessionInputValue::ViewMut(
                    // ort::value::Tensor::from_array(arr.view())
                    ort::value::TensorRefMut::from_array_view_mut(arr)
                        .context("Failed to convert array to tensor")?
                        .into_dyn(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let input_builder: ort::session::SessionInputs<'_, '_, 1> =
            ort::session::SessionInputs::ValueSlice(args.as_slice());
        let first_run_start = std::time::Instant::now();
        let output = model
            .run(input_builder)
            .context("Error running ONNX model")?;
        let first_run_duration = first_run_start.elapsed().as_nanos();
        trace!("warmup iteration {warmup_iter} took {first_run_duration} nanos and produced:");
        for output in &output {
            trace!("    {output:?}",);
        }
    }
    Ok(args)
}

pub fn runner_internal<F>(
    mut operators: Vec<PhysicalOperator>,
    topology_threads: Vec<Vec<usize>>,
    early_poll_ids: BTreeSet<usize>,
    end_index: usize,
    output_logger: Option<UdfBolt>,
    running_time: Duration,
    runtime: tokio::runtime::Runtime,
    execute_fn: impl Clone + Fn(usize, Duration, Vec<PhysicalOperator>, Arc<[PhysicalOperator]>) -> F,
) where
    F: Future<Output = ()> + Send + 'static,
{
    init_pools();

    // put output logger at the end
    if let Some(output_logger) = output_logger {
        if let Some(last) = operators.get_mut(end_index) {
            last.add_parent(output_logger.id);
        } else {
            error!(
                "end index {end_index} is out of bounds for operators of length {}",
                operators.len()
            );
            // panic!("end index {end_index} is out of bounds for operators of length {}", operators.len());
        }
        operators.push(PhysicalOperator::UserDefinedFunction(output_logger));
    }
    info!("operators");
    for op in operators.iter() {
        info!("#{}: {:?}", op.get_id(), op.get_op_type());
    }
    let (mut bolts, mut spouts) = split_bolts_and_spouts(operators);
    for bolt in &mut bolts {
        Operator::initialize(bolt);
    }

    runtime.block_on(async {
        let (ready_to_start_signal, ready_to_start) = watch::channel(false);
        let init_set = futures::future::join_all(
            spouts
                .iter_mut()
                .map(|spout| AsyncSpout::initialize(spout, ready_to_start.clone())),
        );
        init_set.await;
        info!("Finished joining all initializations");
        // tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        let bolts: Arc<[PhysicalOperator]> = bolts.into();
        if let Err(e) = ready_to_start_signal.send(true) {
            error!("Error sending ready to start signal: {:?}", e);
        }
        for spout_id in early_poll_ids {
            let Some(spout) = spouts.get_mut(spout_id) else {
                error!(
                    "early poll spout id {spout_id} is out of bounds for spouts of length {}",
                    spouts.len()
                );
                continue;
            };
            let spout_output = spout.produce(bolts.clone()).await;
            debug!(
                "early poll of spout {spout_id:?} output: {:?}",
                spout_output
            );
        }

        let mut spouts_by_thread: Vec<Vec<PhysicalOperator>> = topology_threads
            .iter()
            .map(|v| Vec::with_capacity(v.len()))
            .collect();
        let spouts_by_thread_dims = (
            spouts_by_thread.len(),
            spouts_by_thread.iter().map(|v| v.len() as f32).sum::<f32>()
                / spouts_by_thread.len() as f32,
        );
        debug!("spout thread dims: {:?}", spouts_by_thread_dims);

        for (spout_id, spout) in spouts.into_iter().enumerate() {
            for (topology_thread, thread_spout_ids) in topology_threads.iter().enumerate() {
                if thread_spout_ids.contains(&spout_id) {
                    if let Some(spouts) = spouts_by_thread.get_mut(topology_thread){
                        spouts.push(spout);
                    } else {
                        error!("topology thread {topology_thread} is out of bounds for spouts_by_thread of length {}", spouts_by_thread.len());
                    }
                    break;
                }
            }
        }

        let mut threads = tokio::task::JoinSet::new();

        for (task_idx, spouts) in spouts_by_thread.into_iter().enumerate() {
            use futures::FutureExt;
            let bolts = bolts.clone();
            threads.spawn(
                tokio::task::spawn(execute_fn.clone()(
                    task_idx,
                    running_time,
                    spouts,
                    bolts,
                ))
                .then(move |e| async move {
                    (task_idx, e)
                })
            );
        }
        debug!("there are {} threads spawned", threads.len());
        for (task_no, exit_status) in threads.join_all().await.into_iter() {
            match exit_status {
                Ok(()) => {
                    info!("task {task_no} exited successfully");
                }
                Err(e) => {
                    error!("task {task_no} exited with error: {:?}", e);
                }
            }
        }
        info!("all threads have exited");
    });
    runtime.shutdown_background();
    info!("runtime has exited");
}

pub async fn execute_for_while(
    task_index: usize,
    duration: Duration,
    condition: watch::Receiver<bool>,
    signal: watch::Sender<bool>,
    mut spouts: Vec<PhysicalOperator>,
    bolts: Arc<[PhysicalOperator]>,
) {
    spouts.retain(|op| !matches!(op, PhysicalOperator::__PLACEHOLDER__(_)));
    let start = Instant::now();
    let mut spout_futures = Vec::with_capacity(spouts.len());
    for (spout_id, mut spout) in spouts.into_iter().enumerate() {
        let spout_operator_id = spout.get_id();
        let bolts = bolts.clone();
        let mut condition = condition.clone();
        // should they be pinned to this task on this thread?
        // spout_futures.push(Box::pin(async move {

        // or should we allow the tokio scheduler to move them around?
        spout_futures.push(tokio::spawn(async move {
            let mut total_timeout = Box::pin(tokio::time::sleep(duration));
            let mut last_wakeup_time = Instant::now();
            let mut avg_wakeup_time_micros = 0.0;
            let mut wakeup_count = 0usize;
            let mut micros_since_last_log = 0.0f64;
            loop {
                let output = spout.produce(bolts.clone());
                let total_timeout = std::pin::pin!(&mut total_timeout);
                let condition_changed = condition.changed();
                tokio::select! {
                    _ = total_timeout => {
                        warn!("timed out on spout {spout_id} (operator {spout_operator_id}) for task {task_index} (thread {:?})", std::thread::current().id());
                        break;
                    }
                    r = condition_changed => {
                        match r {
                            Ok(_) => {
                                info!("spout {spout_id} (operator {spout_operator_id}) received signal to exit on task {task_index} (thread {:?})", std::thread::current().id());
                                break;
                            }
                            Err(e) => {
                                error!("error receiving signal to exit on spout {spout_id} (operator {spout_operator_id}) for task {task_index} (thread {:?}): {e:?}", std::thread::current().id());
                                break;
                            }
                        }
                    }
                    output_status = output => {
                        match output_status {
                            OperatorOutput::Finished =>  {
                                info!("spout {spout_id} (operator {spout_operator_id}) finished on task {task_index} (thread {:?})", std::thread::current().id());
                                // return;
                                break;
                            }
                            OperatorOutput::Something(Some(_amt)) => {
                                let elapsed = last_wakeup_time.elapsed();
                                let current_wakeup_time_micros = elapsed.as_micros() as f64 / 1000.0;
                                avg_wakeup_time_micros = (avg_wakeup_time_micros * wakeup_count as f64 + current_wakeup_time_micros) / (wakeup_count as f64 + 1.0);
                                wakeup_count += 1;
                                micros_since_last_log += current_wakeup_time_micros;
                                // log every second
                                if micros_since_last_log > 1_000_000.0 {
                                    micros_since_last_log = 0.0;
                                    debug!("spout {spout_id} (operator {spout_operator_id}) on task {task_index} (thread {:?}) woke up {} times, average wakeup time: {:.2} micros", std::thread::current().id(), wakeup_count, avg_wakeup_time_micros);
                                    // reset stats so we have a rolling average to get the most recent data
                                    wakeup_count = 0;
                                    avg_wakeup_time_micros = 0.0;
                                }
                                last_wakeup_time = Instant::now();
                                continue
                            }
                            OperatorOutput::Something(None) | OperatorOutput::Nothing => {
                                debug!("spout {spout_id} (operator {spout_operator_id}) on task {task_index} (thread {:?}) says it produced nothing after processing its data", std::thread::current().id());
                                continue;
                            }
                        }
                    }
                }
            }

            spout.finalize().await;
            let elapsed = start.elapsed();
            debug!("spout {spout_id} (operator {spout_operator_id}) finalized on task {task_index} (thread {:?}) after {} seconds", std::thread::current().id(), elapsed.as_secs_f32());
        }));
    }
    let joinset_of_spouts = futures::future::join_all(spout_futures).await;
    for join_result in joinset_of_spouts {
        match join_result {
            Ok(_) => {
                debug!(
                    "a spout task finished successfully on task {task_index} (thread {:?})",
                    std::thread::current().id()
                );
            }
            Err(e) => {
                error!(
                    "a spout task finished with error on task {task_index} (thread {:?}): {e:?}",
                    std::thread::current().id()
                );
            }
        }
    }

    debug!(
        "sending signal to exit on task {task_index} (thread {:?})",
        std::thread::current().id()
    );
    if let Err(e) = signal.send(true) {
        error!(
            "error sending signal to exit on task {task_index} (thread {:?}): {e:?}",
            std::thread::current().id()
        );
    }
    // for mut spout in spouts {
    //     spout.finalize().await;
    // }
    debug!(
        "everything is finalized on task {task_index} (thread {:?}). exiting now",
        std::thread::current().id()
    );
}

pub fn runner_sample(
    operators: Vec<PhysicalOperator>,
    topology_threads: Vec<Vec<usize>>,
    early_poll_ids: BTreeSet<usize>,
    end_index: usize,
    output_logger: Option<UdfBolt>,
    running_time: Duration,
    runtime: tokio::runtime::Runtime,
    condition: watch::Receiver<bool>,
    signal: watch::Sender<bool>,
) {
    runner_internal(
        operators,
        topology_threads,
        early_poll_ids,
        end_index,
        output_logger,
        running_time,
        runtime,
        move |task_index, runtime, v, a| {
            execute_for_while(task_index, runtime, condition.clone(), signal.clone(), v, a)
        },
    );
}
