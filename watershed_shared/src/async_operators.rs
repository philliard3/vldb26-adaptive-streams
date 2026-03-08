use crate::async_query_builder::{PhysicalOperator, ValueSetterState};
use crate::basic_pooling::{get_tuple, get_tuple_vec};
use crate::basic_pooling::{return_tuple_vec, CollectTuple};
use crate::caching::StrToKey;
use crate::expression::evaluate_computation_expression;
use crate::global_logger::{LimitedHabValue, NO_AUX_DATA};
use crate::scheduler::BoundedAsyncReceiver;
use crate::ws_types::Operator;
use crate::ws_types::OperatorOutput;
use crate::AggregationExpression;
use crate::AggregationResult;
use crate::AsyncPipe;
use crate::BuiltinAggregator;
use crate::ChromaJoin;
use crate::ChromaJoinKind;
use crate::DeriveValue;
use crate::GroupBy;
use crate::HabValue;
use crate::Join;
use crate::JoinInner;
use crate::Project;
use crate::Queue;
use crate::Select;
use crate::Tuple;
use crate::UdfBolt;
use crate::Union;
use crate::{DummyBolt, EncoderFunction, HabString};
use dashmap::DashMap;
use futures::Stream;
use log::trace;
use log::{debug, error, info, warn};
use std::collections::BTreeSet;
use std::error;
use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::mpsc::{
    Receiver as BoundedReceiver, Sender as BoundedSender, UnboundedReceiver, UnboundedSender,
};
use tokio::sync::watch;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub trait AsyncSpout: Operator {
    // we communicate back from the spouts to the runtime that they are ready to procede by returning from the Future
    // but we need a way to communicate back to the spouts when they should start and when they should stop
    fn initialize<'this>(
        &'this mut self,
        ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()>;

    fn produce<'this>(
        &'this mut self,
        bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput>;

    // most spouts will not need to do anything here
    fn finalize<'this>(&'this mut self) -> impl 'this + Future<Output = ()> {
        async {}
    }
}

pub trait Bolt: Operator {
    fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> impl 'a + Send + Future<Output = OperatorOutput>;
}

impl Bolt for Select {
    async fn process_tuples<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        tuples.retain(&self.pred);
        if tuples.is_empty() {
            info!("Select operator {} filtered out all tuples", self.id);
            return OperatorOutput::Nothing;
        }
        let len = tuples.len();

        if let Some(p) = self.parent {
            let Some(p) = bolts.get(p) else {
                // TODO: should the bolt be able to return a Result?
                error!("Select could not find its parent {p} in list of bolts");
                panic!("Select could not find its parent {p} in list of bolts");
            };
            let _parent_results = p.process_tuples(tuples, self.id, bolts).await;
            info!("Select operator {} passed through {} tuples", self.id, len);
        } else {
            warn!(
                "Select operator {} has no parent so it did not pass its {len} tuples",
                self.id
            );
        }
        OperatorOutput::Something(Some(len))
    }
}

static INTERNAL_FIELDS: &[&str] = &["__uuid", "__provenance", crate::basic_pooling::UUID_FIELD];

impl Bolt for Project {
    async fn process_tuples<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        for t in tuples.iter_mut() {
            t.retain(|k, _| self.keep_list.contains(k) || INTERNAL_FIELDS.contains(&&**k));
        }
        if tuples.is_empty() {
            return OperatorOutput::Nothing;
        }
        let len = tuples.len();

        if let Some(p) = self.parent {
            let Some(p) = bolts.get(p) else {
                // TODO: should the bolt be able to return a Result?
                error!("Select could not find its parent {p} in list of bolts");
                panic!("Select could not find its parent {p} in list of bolts");
            };
            let _parent_results = p.process_tuples(tuples, self.id, bolts).await;
        }

        OperatorOutput::Something(Some(len))
    }
}

impl Bolt for UdfBolt {
    async fn process_tuples<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        use futures::StreamExt;
        let self_id = self.id;
        let mut outputs = get_tuple_vec();
        let mut udf_futures = futures::stream::FuturesOrdered::new();
        for t in tuples.drain(..) {
            let process = Arc::clone(&self.process);
            let start_waiting = std::time::Instant::now();
            let tuple_id = t.id();
            udf_futures.push_back(tokio::task::spawn_blocking(move || {
                let start_processing = std::time::Instant::now();
                let outputs = (process)(t);
                let processing_micros = start_processing.elapsed().as_micros();
                debug!(
                    "UdfBolt {} processed tuple {tuple_id} in {:?} us",
                    self_id, processing_micros
                );
                (tuple_id, start_waiting, outputs)
            }));
        }
        let mut udf_futures = udf_futures.enumerate();
        while let Some((input_no, future_result)) = udf_futures.next().await {
            let (tuple_id, start_waiting, mut new_tuples) = match future_result {
                Ok(v) => v,
                Err(e) => {
                    error!("UdfBolt {self_id} failed to join on input number {input_no} due to join error: {e:?}");
                    continue;
                }
            };
            let elapsed_micros = start_waiting.elapsed().as_micros();
            debug!("UdfBolt {self_id} total time waiting + processing tuple {tuple_id} in {elapsed_micros:?} us");
            outputs.extend(new_tuples.drain(..));
            return_tuple_vec(new_tuples);
        }

        return_tuple_vec(tuples);
        if outputs.is_empty() {
            return OperatorOutput::Nothing;
        }

        if let Some(p) = self.parent {
            let Some(p) = bolts.get(p) else {
                panic!("Select could not find its parent {p} in list of bolts");
            };

            let len = outputs.len();
            p.process_tuples(outputs, self.id, bolts).await;
            OperatorOutput::Something(Some(len))
        } else {
            OperatorOutput::Something(None)
        }
    }
}

impl Bolt for DeriveValue {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        let Some(parent) = self.parent else {
            warn!("DeriveValue operator missing parent");
            return OperatorOutput::Nothing;
        };
        let mut tuples_to_process = get_tuple_vec();
        for mut tuple in tuples.into_iter() {
            let new_field_value = evaluate_computation_expression(&tuple, &self.action);
            tuple.insert(self.new_field_name.clone(), new_field_value);
            tuples_to_process.push(tuple);
        }
        let num_tuples = tuples_to_process.len();
        let emitted_any = num_tuples > 0;
        bolts[parent]
            .process_tuples(tuples_to_process, self.id, bolts)
            .await;
        if emitted_any {
            OperatorOutput::Something(Some(num_tuples))
        } else {
            OperatorOutput::Nothing
        }
    }
}

impl Bolt for Join {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        info!(
            "Join {} received {} tuples from source {}",
            self.id,
            tuples.len(),
            source
        );

        fn handle_table_build_side(
            tuples: Vec<Tuple>,
            fields: &[HabString],
            build_data: &DashMap<Vec<HabValue>, Vec<Tuple>>,
        ) -> OperatorOutput {
            for t in tuples {
                let key = fields
                    .iter()
                    .map(|f| {
                        t.get(f).unwrap_or_else(|| {
                            error!(
                                "{}:{}:{} : Field {f} not found in tuple",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("field \"{f}\" not found in tuple")
                        })
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                build_data.entry(key).or_default().push(t);
            }
            OperatorOutput::Nothing
        }

        match &self.join_info {
            JoinInner::InnerTable { build_data, fields } => {
                // right is the index side
                if source == self.right {
                    handle_table_build_side(tuples, fields, build_data)
                } else if source == self.left {
                    let mut outputs = Vec::new();
                    for t in tuples {
                        let key = fields
                            .iter()
                            .map(|f| {
                                t.get(f).unwrap_or_else(|| {
                                    error!(
                                        "{}:{}:{} : Field {} not found in tuple",
                                        file!(),
                                        line!(),
                                        column!(),
                                        f
                                    );
                                    panic!("field {f} not found in tuple");
                                })
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        if let Some(cached) = build_data.get(&key) {
                            for l in &*cached {
                                let mut output_tuple = l.clone();
                                output_tuple.extend(t.iter().map(|(x, y)| (x.clone(), y.clone())));
                                outputs.push(output_tuple);
                            }
                        }
                    }
                    if outputs.is_empty() {
                        OperatorOutput::Nothing
                    } else {
                        let Some(p) = self.parent else {
                            return OperatorOutput::Something(None);
                        };
                        let Some(p) = bolts.get(p) else {
                            error!(
                                "{}:{}:{} : parent {p} of join not found within bolts",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("parent {p} of join not found within bolts");
                        };
                        let len = outputs.len();
                        p.process_tuples(outputs, self.id, bolts).await;
                        OperatorOutput::Something(Some(len))
                    }
                } else {
                    error!("{}:{}:{} : Join received data from unknown source {source}. expected left was {}. expected right was {}.", file!(), line!(), column!(), self.left, self.right);
                    panic!("join received data from unknown source {source}. expected left was {}. expected right was {}.", self.left, self.right);
                }
            }
            JoinInner::OuterTable {
                build_data,
                fields,
                build_side_fields,
            } => {
                // right is the index side
                if source == self.right {
                    let new_fields = tuples
                        .iter()
                        .flat_map(|t| t.keys())
                        .collect::<BTreeSet<_>>();

                    for new_field in new_fields {
                        build_side_fields.insert(new_field.clone());
                    }
                    handle_table_build_side(tuples, fields, build_data)
                } else if source == self.left {
                    debug!("build side fields: {build_side_fields:?}");
                    let mut outputs = Vec::new();
                    for t in tuples {
                        let key = fields
                            .iter()
                            .map(|f| {
                                t.get(f).unwrap_or_else(|| {
                                    error!(
                                        "{}:{}:{} : Field {} not found in tuple",
                                        file!(),
                                        line!(),
                                        column!(),
                                        f
                                    );
                                    panic!("field {f} not found in tuple")
                                })
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        if let Some(cached) = build_data.get(&key) {
                            if !cached.is_empty() {
                                let mut outputs_detected = false;
                                for l in &*cached {
                                    if !(self.pred)(l, &t) {
                                        continue;
                                    }
                                    debug!("matched build side {l:?} with lookup side {t:?}");
                                    let mut output_tuple = l.clone();
                                    output_tuple
                                        .extend(t.iter().map(|(x, y)| (x.clone(), y.clone())));
                                    outputs.push(output_tuple);
                                    outputs_detected = true;
                                }

                                if !outputs_detected {
                                    // TODO: handle provenance when UUID changes to a provenance tree
                                    let mut output_tuple = build_side_fields
                                        .iter()
                                        .map(|f| (HabString::clone(&f), HabValue::Null))
                                        .collect_tuple();
                                    output_tuple
                                        .extend(t.iter().map(|(x, y)| (x.clone(), y.clone())));
                                    outputs.push(output_tuple);
                                }
                            } else {
                                // TODO: handle provenance when UUID changes to a provenance tree
                                let mut output_tuple = build_side_fields
                                    .iter()
                                    .map(|f| (HabString::clone(&f), HabValue::Null))
                                    .collect_tuple();
                                output_tuple.extend(t.iter().map(|(x, y)| (x.clone(), y.clone())));
                                outputs.push(output_tuple);
                            }
                        } else {
                            // TODO: handle provenance when UUID changes to a provenance tree
                            let mut output_tuple = build_side_fields
                                .iter()
                                .map(|f| (HabString::clone(&f), HabValue::Null))
                                .collect_tuple();
                            output_tuple.extend(t.iter().map(|(x, y)| (x.clone(), y.clone())));
                            outputs.push(output_tuple);
                        }
                    }
                    if outputs.is_empty() {
                        OperatorOutput::Nothing
                    } else {
                        let Some(p) = self.parent else {
                            return OperatorOutput::Something(None);
                        };
                        let Some(p) = bolts.get(p) else {
                            panic!("parent {p} of join not found within bolts");
                        };
                        let len = outputs.len();
                        p.process_tuples(outputs, self.id, bolts).await;
                        OperatorOutput::Something(Some(len))
                    }
                } else {
                    error!("join received data from unknown source {source}. expected left was {}. expected right was {}.", self.left, self.right);
                    panic!("join received data from unknown source {source}. expected left was {}. expected right was {}.", self.left, self.right);
                }
            }
            JoinInner::DoublePipeline {
                left_inputs,
                right_inputs,
                evict,
            } => {
                // TODO: handle other times we might want to evict, like in a background thread
                (evict)(left_inputs, right_inputs);
                let outputs: Vec<Tuple> = if source == self.left {
                    let outputs = tuples
                        .iter()
                        .flat_map(|l| right_inputs.iter().map(move |r| (l, r)))
                        .filter(|(l, r)| (self.pred)(l, r))
                        .map(|(x, y)| {
                            let mut z = x.clone();
                            z.extend(y.iter().map(|(x, y)| (x.clone(), y.clone())));
                            z
                        })
                        .collect();
                    for t in tuples {
                        left_inputs.insert(t);
                    }
                    outputs
                } else if source == self.right {
                    let outputs = tuples
                        .iter()
                        .flat_map(|r| left_inputs.iter().map(move |l| (l, r)))
                        .filter(|(l, r)| (self.pred)(l, r))
                        .map(|(y, x)| {
                            let mut z = x.clone();
                            z.extend(y.iter().map(|(x, y)| (x.clone(), y.clone())));
                            z
                        })
                        .collect();

                    for t in tuples {
                        right_inputs.insert(t);
                    }
                    outputs
                } else {
                    error!(
                        "{}:{}:{} : join operator {} received data from unknown source {source}. expected left was {}. expected right was {}.",
                        file!(), line!(), column!(),
                        self.id, self.left, self.right);
                    panic!("join operator {} received data from unknown source {source}. expected left was {}. expected right was {}.", self.id, self.left, self.right);
                };
                // TODO: handle end of stream
                if outputs.is_empty() {
                    OperatorOutput::Nothing
                } else {
                    let Some(p) = self.parent else {
                        return OperatorOutput::Something(None);
                    };
                    let Some(p) = bolts.get(p) else {
                        error!(
                            "{}:{}:{} : parent {p} of join operator {} not found within bolts",
                            file!(),
                            line!(),
                            column!(),
                            self.id
                        );
                        panic!(
                            "parent {p} of join operator {} not found within bolts",
                            self.id
                        );
                    };
                    let len = outputs.len();
                    p.process_tuples(outputs, self.id, bolts).await;
                    OperatorOutput::Something(Some(len))
                }
            }
        }
    }
}

impl Bolt for DummyBolt {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        _bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        warn!(
            "DummyBolt {} received {} tuples from source {}",
            self.0,
            tuples.len(),
            source
        );
        OperatorOutput::Finished
    }
}

impl Bolt for crate::operators::PythonInlineUdf {
    async fn process_tuples<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        let num_to_output = tuples.len();
        if tuples.is_empty() {
            return OperatorOutput::Something(None);
        }
        let my_id = self.id;
        // map each using our function
        let mut outputs = get_tuple_vec();
        outputs.reserve(num_to_output.saturating_sub(outputs.len()));
        // let mut inputs = Vec::with_capacity(self.fields.len());
        use crate::PythonDecodingMethod::*;
        let (CustomDecoder { fields, .. }
        | PythonValues { fields, .. }
        | PyAnyToHabValues { fields, .. }) = &self.decoder;

        let op_id = self.id;
        for mut t in tuples.drain(..) {
            let tuple_id = t.id();
            'log_python_start: {
                let mut location_buffer =
                    *b"start_inline_python_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                let write_start = "start_inline_python_".len();
                let max_script_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                let write_end = write_start + max_script_len;
                use std::io::Write;
                // write the script name
                if let Err(e) = write!(
                    location_buffer[write_start..write_end].as_mut(),
                    "{}",
                    self.script_name
                ) {
                    error!("Failed to write to location buffer script name for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                    break 'log_python_start;
                }

                let write_start = location_buffer.len() - 8;
                if let Err(e) = write!(location_buffer[write_start..].as_mut(), "{op_id}") {
                    error!("Failed to write to location buffer op id for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                    break 'log_python_start;
                }
                let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                    error!("Failed to convert location buffer to string for sending tuple {tuple_id} in python remote udf operator {op_id}");
                    break 'log_python_start;
                };

                if let Err(e) = crate::global_logger::log_data(
                    tuple_id,
                    location_buffer.to_raw_key(),
                    NO_AUX_DATA,
                ) {
                    error!("Failed to log send to python in python remote udf operator {op_id} with error {:?}", e);
                }
            }
            let t_id = t.id();
            let Some(args) = self.encoder.encode_tuple(&t) else {
                error!(
                    "PythonInlineUdf {my_id} failed to encode tuple with id {t_id} to python values"
                );
                for output_field in fields.iter() {
                    t.insert(output_field.clone(), HabValue::Null);
                }
                continue;
            };

            let py_output = Python::with_gil(|ctx| {
                let py_args = PyTuple::new(ctx, args)?;
                self.func.call1(ctx, py_args)
            });
            let py_output = match py_output {
                Ok(v) => v,
                Err(e) => {
                    error!("PythonInlineUdf {my_id} failed to call python function with error {e}");
                    for output_field in fields.iter() {
                        t.insert(output_field.clone(), HabValue::Null);
                    }
                    continue;
                }
            };
            let Some(num_decoded) = self
                .decoder
                .decode_tuple(&py_output, t, &fields, &mut outputs)
            else {
                error!(
                    "PythonInlineUdf {my_id} failed to decode python output tuple with id {t_id}"
                );
                continue;
            };
            // success
            debug!("PythonInlineUdf {my_id} successfully processed tuple with id {t_id} into {} new tuples", num_decoded);
            'log_python_complete: {
                let mut location_buffer =
                    *b"complete_inline_python_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                let write_start = "complete_inline_python_".len();
                let max_script_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                let write_end = write_start + max_script_len;
                use std::io::Write;
                // write the script name
                if let Err(e) = write!(
                    location_buffer[write_start..write_end].as_mut(),
                    "{}",
                    self.script_name
                ) {
                    error!("Failed to write to location buffer script name for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                    break 'log_python_complete;
                }

                let write_start = location_buffer.len() - 8;
                if let Err(e) = write!(location_buffer[write_start..].as_mut(), "{op_id}") {
                    error!("Failed to write to location buffer op id for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                    break 'log_python_complete;
                }
                let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                    error!("Failed to convert location buffer to string for sending tuple {tuple_id} in python remote udf operator {op_id}");
                    break 'log_python_complete;
                };

                if let Err(e) = crate::global_logger::log_data(
                    tuple_id,
                    location_buffer.to_raw_key(),
                    NO_AUX_DATA,
                ) {
                    error!("Failed to log send to python in python remote udf operator {op_id} with error {:?}", e);
                }
            }
        }
        return_tuple_vec(tuples);
        let len = outputs.len();
        let Some(p) = self.parent else {
            warn!(
                "Select operator {} has no parent so it did not pass its {len} tuples",
                self.id
            );
            return OperatorOutput::Nothing;
        };
        let Some(p) = bolts.get(p) else {
            error!("Select could not find its parent {p} in list of bolts");
            return OperatorOutput::Finished;
        };

        let _parent_results = p.process_tuples(outputs, self.id, bolts).await;
        info!("Select operator {} passed through {} tuples", self.id, len);
        OperatorOutput::Something(Some(len))
    }
}

#[cfg(test)]
mod inline_python_tests {
    use pyo3::{
        types::{PyAnyMethods, PyModule},
        IntoPyObjectExt,
    };

    use crate::{
        async_operators::Bolt, async_query_builder::PhysicalOperator, OperatorOutput,
        PythonDecodingMethod,
    };

    #[tokio::test]
    async fn test_python_udf() {
        use crate::operators::PythonInlineUdf;
        use crate::ws_types::HabValue;
        use crate::ws_types::Tuple;

        crate::operators::start_python_with_modules(&["numpy", "pandas"]);

        let function_name = "test_func";
        let pyfunc = pyo3::Python::with_gil(|py| {
            let pymodule = PyModule::from_code(
                py,
                c"\
import numpy as np
def test_func(field1, field2):
    print(field1)
    print(field2)
    # sequence of random bytes
    arr = np.ones((field1, field2))
    float_arr = arr.astype(np.float32)
    print(float_arr)
    float_arr_bytes = float_arr.tobytes()
    return [field1 + field2, field1 * field2, float_arr_bytes]\
                ",
                c"test_file",
                c"test_mod",
            )
            .expect("failed to create module");
            pymodule
                .getattr(function_name)
                .expect("function not found")
                .into_py_any(py)
                .expect("failed to convert to PyAny")
        });

        let udf = PythonInlineUdf {
            id: 0,
            func: pyfunc,
            decoder: PythonDecodingMethod::PyAnyToHabValues {
                fields: vec!["field3".into(), "field4".into(), "field5".into()],
            },
            encoder: crate::PythonEncodingMethod::HabValueToPyAny {
                fields: vec!["field1".into(), "field2".into()],
            },
            parent: Some(1),
            input: 0,
            script_name: "".into(),
            scripts_dir_path: "".into(),
            function_name: "".into(),
        };

        let mut tuple = Tuple::default_internal();
        tuple.insert("field1".into(), HabValue::Integer(1));
        tuple.insert("field2".into(), HabValue::Integer(2));

        let expected3 = HabValue::Integer(1 + 2);
        let expected4 = HabValue::Integer(1 * 2);

        use std::sync::{atomic::AtomicUsize, Arc};
        let shared_counter = Arc::new(AtomicUsize::new(0));
        let callback_counter = Arc::clone(&shared_counter);
        let dummy_parent_op = crate::UdfBolt {
            id: 1usize,
            child: 0usize,
            parent: None,
            process: Arc::new(move |t| {
                callback_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let field3 = t.get("field3").unwrap_or_else(|| {
                    panic!("field3 not found in tuple");
                });
                let field4 = t.get("field4").unwrap_or_else(|| {
                    panic!("field4 not found in tuple");
                });
                assert_eq!(field3, &expected3);
                assert_eq!(field4, &expected4);
                // field5 is the bytes
                let field5 = t.get("field5").unwrap_or_else(|| {
                    panic!("field5 not found in tuple");
                });
                let _field5 = field5.as_byte_buffer().unwrap_or_else(|| {
                    panic!("field5 not a bytes");
                });
                vec![]
            }),
        };

        let ops = vec![
            PhysicalOperator::PythonInlineFunction(udf),
            PhysicalOperator::UserDefinedFunction(dummy_parent_op),
        ];
        let udf = &ops[0];

        let result = udf.process_tuples(vec![tuple], 0, &ops).await;
        assert_eq!(result, OperatorOutput::Something(Some(1)));
        let callback_value = shared_counter.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(callback_value, 1);
    }
}

impl Bolt for Union {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        trace!(
            "Union operator {} received {} tuples from source {}",
            self.id,
            tuples.len(),
            source
        );
        let Some(parent) = self.parent else {
            error!(
                "{}:{}:{} : Union operator missing parent",
                file!(),
                line!(),
                column!()
            );
            return OperatorOutput::Nothing;
        };
        let num_to_output = tuples.len();
        bolts[parent].process_tuples(tuples, self.id, bolts).await;
        if num_to_output > 0 {
            OperatorOutput::Something(Some(num_to_output))
        } else {
            OperatorOutput::Nothing
        }
    }
}

impl Bolt for GroupBy {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        // TODO: track "is_finished" state
        let mut outputs = Vec::new();
        trace!(
            "group by operator {} received inputs with {} fields",
            self.id,
            tuples
                .first()
                .iter()
                .flat_map(|t| t.keys())
                .collect::<BTreeSet<_>>()
                .len()
        );
        for tuple in tuples {
            let group_key: Vec<HabValue> = self
                .fields
                .iter()
                .map(|field| {
                    tuple
                        .get(field)
                        .unwrap_or_else(|| {
                            let op_id = self.id;
                            warn!("field {field} not found in GroupBy operator {op_id}");
                            panic!("field {field} not found in GroupBy operator {op_id}");
                        })
                        .clone()
                })
                .collect();
            let mut group = match self.state.entry(group_key) {
                dashmap::mapref::entry::Entry::Occupied(o) => o,
                dashmap::mapref::entry::Entry::Vacant(v) => v.insert_entry(Queue::new()),
            };
            // let mut group = self.state.entry(group_key).or_insert_with(Queue::new);
            group.get_mut().push_back(tuple);
            // now that we've gotten all of our keys, we can start processing the group
            let result = match &self.aggregate {
                AggregationExpression::Udf(f) => f(&mut *group.get_mut()),
                AggregationExpression::Componentized {
                    derive_decision_key: _,
                    should_emit: _,
                    derive_eviction_key: _,
                    should_evict: _,
                } => {
                    error!(
                        "{}:{}:{} : semantics of componentized aggregation not yet implemented",
                        file!(),
                        line!(),
                        column!()
                    );
                    unimplemented!("semantics of componentized aggregation not yet implemented");
                }
                AggregationExpression::Builtin { field, op } => match op {
                    BuiltinAggregator::Sum => {
                        let output_value =
                            group.get().iter().fold(HabValue::Integer(0), |acc, tuple| {
                                let value = tuple
                                    .get(field)
                                    .unwrap_or_else(|| {
                                        error!(
                                            "{}:{}:{} : aggregator field {field} not found in aggregation tuple in GroupBy operator {}",
                                            file!(),
                                            line!(),
                                            column!(),
                                            self.id
                                        );
                                        panic!("field {field} not found");
                                    });
                                match (acc, value) {
                                    (HabValue::Integer(a), HabValue::Integer(b)) => {
                                        HabValue::Integer(a + b)
                                    }
                                    (HabValue::Float(a), HabValue::Float(b)) => {
                                        HabValue::Float(a + b)
                                    }
                                    (HabValue::Integer(a), HabValue::Float(b)) => {
                                        HabValue::Float(b + a as f64)
                                    }
                                    (HabValue::Float(a), HabValue::Integer(b)) => {
                                        HabValue::Float(a + *b as f64)
                                    }
                                    (acc, value) => {
                                        error!(
                                            "{}:{}:{} : Type Error in Sum expression {:?} + {:?}",
                                            file!(),
                                            line!(),
                                            column!(),
                                            acc,
                                            value
                                        );
                                        unimplemented!(
                                        "Type Error in Sum expression {:?} + {:?}",
                                        acc,
                                        value
                                    );},
                                }
                            });
                        let output_field_name = format!("sum({field})");
                        let mut output_tuple = get_tuple();
                        output_tuple.insert(output_field_name.into(), output_value);
                        for (key, val) in self.fields.iter().zip(group.key().iter()) {
                            output_tuple.insert(key.clone(), val.clone());
                        }
                        trace!("group by {} outputting tuple {:#?}", self.id, output_tuple);
                        AggregationResult {
                            emit: Some(vec![output_tuple]),
                            is_finished: false,
                        }
                    }
                    BuiltinAggregator::Count => {
                        error!(
                            "{}:{}:{} : count not yet implemented",
                            file!(),
                            line!(),
                            column!()
                        );
                        todo!("count not yet implemented");
                    }
                    BuiltinAggregator::Min => {
                        error!(
                            "{}:{}:{} : min not yet implemented",
                            file!(),
                            line!(),
                            column!()
                        );
                        todo!("min not yet implemented");
                    }
                    BuiltinAggregator::Max => {
                        error!(
                            "{}:{}:{} : max not yet implemented",
                            file!(),
                            line!(),
                            column!()
                        );
                        todo!("max not yet implemented");
                    }
                    BuiltinAggregator::Avg => {
                        error!(
                            "{}:{}:{} : avg not yet implemented",
                            file!(),
                            line!(),
                            column!()
                        );
                        todo!("avg not yet implemented");
                    }
                },
            };
            if let Some(emit) = result.emit {
                outputs.extend(emit);
            }
            if result.is_finished {
                let (hab_values, mut window) = group.remove_entry();
                while let Some(mut t) = window.pop_front() {
                    t.clear();
                    crate::basic_pooling::return_tuple(t);
                }
                drop(hab_values);
            }
        }
        let num_tuples = outputs.len();
        let Some(parent) = self.parent else {
            warn!("GroupBy operator missing parent");
            return OperatorOutput::Nothing;
        };
        bolts[parent].process_tuples(outputs, self.id, bolts).await;
        if num_tuples > 0 {
            OperatorOutput::Something(Some(num_tuples))
        } else {
            OperatorOutput::Nothing
        }
    }
}

impl Bolt for ChromaJoin {
    async fn process_tuples<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        if source == self.index_stream {
            let ids: Vec<&str> = tuples
                .iter()
                .map(|t| {
                    &**t.get("chroma_id")
                        .unwrap_or_else(|| {
                            error!(
                                "{}:{}:{} : chroma_id not found",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("chroma_id not found")
                        })
                        .as_string()
                        .unwrap_or_else(|| {
                            error!(
                                "{}:{}:{} : chroma_id not a string",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("chroma_id not a string")
                        })
                })
                .collect();
            // TODO: what are we doing here? do we expect embeddings to already be there or not?
            // we ask for the embedding here
            // but later we make a new converter for our operator.
            // fix it.
            let embeddings: Vec<Vec<f32>> = tuples
                .iter()
                .map(|t| match t.get("embedding"){
                        Some(v) => v.as_list()
                        .unwrap_or_else(|| {
                            error!(
                                "{}:{}:{} : embedding not a list",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("embedding not a list")
                        })
                        .iter()
                        .map(|v| {
                            v.as_float()
                                .unwrap_or_else(|| {
                                    error!(
                                        "{}:{}:{} : embedding valuenot a float",
                                        file!(),
                                        line!(),
                                        column!()
                                    );
                                    panic!("embedding value was not a float")
                                })
                                .into_inner() as _
                        })
                        .collect(),
                    None => {
                        let Ok(embedding_method) = self.embedding_method.lock() else {
                            error!(
                                "{}:{}:{} : failed to aquire lock on embedding method",
                                file!(),
                                line!(),
                                column!()
                            );
                            panic!("failed to aquire lock on embedding method")
                        };
                        let text = t.get("text")
                            .unwrap_or_else(|| {
                                error!(
                                    "{}:{}:{} : text not found",
                                    file!(),
                                    line!(),
                                    column!()
                                );
                                panic!("questiotextn not found")
                            })
                            .as_string()
                            .unwrap_or_else(|| {
                                error!(
                                    "{}:{}:{} : text not a string",
                                    file!(),
                                    line!(),
                                    column!()
                                );
                                panic!("text not a string")
                            });
                        let Ok(mut embeddings) = embedding_method.encode(&[&**text])
                            else {
                                error!("Failed to encode text {text}");
                                panic!("Failed to encode text {text}");
                            };
                        let Some(e) = embeddings.pop() else {
                            error!("Failed to get embedding for text {text}. The list of embeddings was empty (this should be impossible)");
                            unreachable!("Failed to get embedding for text {text}. The list of embeddings was empty (this should be impossible)");
                        };
                        e
                    }
                })
                .collect();

            // TODO: keep metadata about a specific thing by getting a metadata field
            //  ( probably don't actually do anything regarding this. the data can be joined later using a relational lookup)
            // let metadatas : Option<Vec<_>> = tuples.iter.map()

            // let collection_entries = chromadb::collection::CollectionEntries {
            //     ids,
            //     embeddings: Some(embeddings),
            //     metadatas: None,
            //     documents: None,
            // };

            // let collection =
            //     match crate::chroma_utils::get_collection(&self.collection_name, &self.chroma_url)
            //         .await
            //     {
            //         Ok(c) => c,
            //         Err(e) => {
            //             error!("Failed to get collection during insertion: {e}");
            //             panic!("Failed to get collection during insertion: {e}");
            //         }
            //     };

            // if let Err(e) = collection.upsert(collection_entries, None).await {
            //     error!("Failed to upsert {e:?}");
            //     panic!("Failed to upsert {e:?}");
            // }

            if let Err(e) = crate::chroma_utils::upsert(ids, embeddings, None).await {
                error!("Failed to upsert {e:?}");
                panic!("Failed to upsert {e:?}");
            }
            OperatorOutput::Nothing
        } else if source == self.lookup_stream {
            debug!("chroma g1");
            debug!(
                "ChromaJoin {} received {} tuples from source {}",
                self.id,
                tuples.len(),
                source
            );
            let Some(p) = self.parent else {
                error!("{}:{}:{} : Chroma Join initiated without parent index. This is likely a mistake.",file!(),
                line!(),
                column!());
                return OperatorOutput::Nothing;
            };
            let Some(p) = bolts.get(p) else {
                error!(
                    "{}:{}:{} : parent {p} of join not found within bolts",
                    file!(),
                    line!(),
                    column!()
                );
                panic!("parent {p} of join not found within bolts");
            };

            debug!("chroma g2");
            // do lookup
            let (is_right_outer, query_n_matches, keep_n_matches) = match self.join_info {
                ChromaJoinKind::Inner {
                    query_n_matches,
                    keep_n_matches,
                } => (false, query_n_matches, keep_n_matches),
                ChromaJoinKind::RightOuter { .. } => (true, 1, 1),
            };
            if tuples.is_empty() {
                return OperatorOutput::Nothing;
            }

            debug!("chroma g3");

            let (query_embeddings, skipped_indices) = if tuples[0].get("embedding").is_some() {
                debug!("chroma g4: embedding was found");
                let mut query_embeddings = Vec::with_capacity(tuples.len());
                let mut skipped_indices = Vec::with_capacity(tuples.len());
                'valid_indices: for (i, t) in tuples.iter().enumerate() {
                    let Some(q) = t.get("embedding") else {
                        error!("embedding not found");
                        skipped_indices.push(i);
                        continue;
                    };
                    match q {
                        HabValue::List(l) => {
                            let mut emb = Vec::with_capacity(l.len());
                            for val in l.iter() {
                                let Some(val) = val.as_float() else {
                                    error!("embedding value was not a float");
                                    skipped_indices.push(i);
                                    continue 'valid_indices;
                                };
                                emb.push(val.into_inner() as _);
                            }
                            debug!("embedding before normalization: {emb:?}");
                            // normalize it
                            let denominator =
                                emb.iter().copied().map(|x| x * x).sum::<f32>().sqrt();
                            for x in emb.iter_mut() {
                                *x /= denominator
                            }
                            debug!("embedding after normalization: {emb:?}");
                            query_embeddings.push(emb);
                        }
                        HabValue::IntBuffer(buf) => {
                            let Some(buf) = bytemuck::try_cast_slice::<_, f32>(buf).ok() else {
                                error!("embedding was not a float buffer for tuple {} in chroma join operator {}", t.id(), self.id);
                                skipped_indices.push(i);
                                continue;
                            };
                            let mut emb = buf.iter().copied().collect::<Vec<f32>>();
                            debug!("embedding before normalization: {emb:?}");
                            // normalize it
                            let denominator =
                                emb.iter().copied().map(|x| x * x).sum::<f32>().sqrt();
                            for x in emb.iter_mut() {
                                *x /= denominator
                            }
                            debug!("embedding after normalization: {emb:?}");
                            query_embeddings.push(emb);
                        }
                        HabValue::Null => {
                            warn!(
                                "embedding was null for tuple {} in chroma join operator {}",
                                t.id(),
                                self.id
                            );
                            skipped_indices.push(i);
                        }
                        _ => {
                            error!("embedding for tuple {} in chroma join operator {} was not a list or int buffer", t.id(), self.id);
                            skipped_indices.push(i);
                        }
                    }
                }
                debug!(
                    "successfully mapped {} embeddings and skipped {} tuples out of {} from batch input to chroma operator",
                    query_embeddings.len(), skipped_indices.len(), tuples.len()
                );

                (query_embeddings, skipped_indices)
            } else {
                debug!("chroma g4: embedding not found");
                let mut query_texts = Vec::with_capacity(tuples.len());
                let mut skipped_indices = Vec::with_capacity(tuples.len());
                for (i, t) in tuples.iter().enumerate() {
                    let Some(q) = t.get("question") else {
                        error!("question not found");

                        skipped_indices.push(i);
                        continue;
                    };
                    let Some(q) = q.as_string() else {
                        error!("question not a string");
                        skipped_indices.push(i);
                        continue;
                    };
                    query_texts.push(q.as_ref());
                }

                let Ok(valid_query_embeddings) = self
                    .embedding_method
                    .lock()
                    .expect("Failed to acquire lock on embeddings")
                    .encode(&query_texts)
                else {
                    error!("Failed to encode embeddings");
                    panic!("Failed to encode embeddings");
                };
                (valid_query_embeddings, skipped_indices)
            };
            let tuple_id = tuples[0].id();
            debug!("chroma g5: querying db for tuple {tuple_id}");
            trace!("querying db");
            let query_start_time = std::time::Instant::now();
            let url_out = &mut String::with_capacity(64);
            let include = &["metadatas", "distances"];
            let query_result = match crate::chroma_utils::query_collection(
                &self.chroma_url,
                &self.collection_id,
                &*query_embeddings,
                query_n_matches,
                &mut self.client.clone(),
                url_out,
                include,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    error!(
                        "Failed to query for tuples {:?}: {e}",
                        tuples.iter().map(|t| t.id()).collect::<Vec<_>>()
                    );
                    for tuple in tuples.iter_mut() {
                        tuple.insert("match_ids".into(), HabValue::Null);
                        tuple.insert("match_distances".into(), HabValue::Null);
                        tuple.insert("passages".into(), HabValue::Null);
                    }
                    let tuples_to_process = tuples;

                    let num_tuples = tuples_to_process.len();
                    let emitted_any = num_tuples > 0;
                    p.process_tuples(tuples_to_process, self.id, bolts).await;
                    return if emitted_any {
                        OperatorOutput::Something(Some(num_tuples))
                    } else {
                        OperatorOutput::Nothing
                    };
                }
            };
            let query_end_time = std::time::Instant::now();
            debug!(
                "chroma g6: chroma query for tuple {tuple_id} took {:.5} ms",
                (query_end_time - query_start_time).as_secs_f32() * 1000.0
            );

            let mut match_data_iter = query_result
                .ids
                .into_iter()
                .zip(
                    query_result
                        .distances
                        .expect("No distances found in query result"),
                )
                .zip(if let Some(metas) = query_result.metadatas {
                    either::Either::Left(metas.into_iter().map(|meta| Some(meta)))
                } else {
                    warn!("No metadatas found in query result");
                    either::Either::Right(std::iter::repeat(None))
                });
            let mut skipped_index_iter = skipped_indices.iter().copied();
            let mut next_skipped_index = skipped_index_iter.next();
            let mut tuples_to_process = Vec::with_capacity(tuples.len());
            for (original_index, mut original_tuple) in tuples.into_iter().enumerate() {
                if Some(original_index) == next_skipped_index {
                    original_tuple.insert("match_ids".into(), HabValue::Null);
                    original_tuple.insert("match_distances".into(), HabValue::Null);
                    original_tuple.insert("passages".into(), HabValue::Null);
                    next_skipped_index = skipped_index_iter.next();
                    tuples_to_process.push(original_tuple);
                    continue;
                }
                let Some(((ids, distances), metadatas)) = match_data_iter.next() else {
                    error!("query result was not long enough");
                    original_tuple.insert("match_ids".into(), HabValue::Null);
                    original_tuple.insert("match_distances".into(), HabValue::Null);
                    original_tuple.insert("passages".into(), HabValue::Null);
                    tuples_to_process.push(original_tuple);
                    continue;
                };
                debug!("chroma distances = {:?}", distances);
                let mut id_vec = Vec::new();
                let mut distance_vec = Vec::new();
                let mut passage_vec = Vec::new();
                for ((match_id, match_distance), meta) in
                    ids.into_iter()
                        .zip(distances)
                        .zip(if let Some(metadatas) = metadatas {
                            either::Either::Left(metadatas.into_iter().map(|meta| Some(meta)))
                        } else {
                            either::Either::Right(std::iter::repeat(None))
                        })
                {
                    let passage = 'passage_data: {
                        let Some(meta) = meta else {
                            error!("metadata not found");
                            break 'passage_data HabValue::Null;
                        };
                        let Some(meta) = meta else {
                            error!("metadata not found");
                            break 'passage_data HabValue::Null;
                        };
                        let Some(passage) = meta.get("text") else {
                            error!("passage not found");
                            break 'passage_data HabValue::Null;
                        };
                        let Some(passage) = passage.as_str() else {
                            error!("passage (\"text\" metadata field) not a string");
                            break 'passage_data HabValue::Null;
                        };
                        HabValue::from(String::from(passage))
                    };
                    if match_distance < self.distance_threshold {
                        id_vec.push(HabValue::from(String::from(match_id)));
                        distance_vec.push(HabValue::from(match_distance as f64));
                        passage_vec.push(passage);
                        // we don't want to receive more than this amount after filtering
                        if id_vec.len() >= keep_n_matches {
                            break;
                        }
                    } else {
                        // short circuit because the matches should be in order of distance,
                        //  so no more should match
                        break;
                    }
                }
                debug!(
                    "after filtering in chroma {} there are {:?} matches: {:?}",
                    self.id,
                    id_vec.len(),
                    &id_vec
                );

                // TODO: handle provenance when UUID becomes a provenance tree
                let num_results = id_vec.len();
                match (num_results, is_right_outer) {
                    // outer join but found nothing
                    (0, true) => {
                        original_tuple.insert("match_ids".into(), HabValue::Null);
                        original_tuple.insert("match_distances".into(), HabValue::Null);
                        original_tuple.insert("passages".into(), HabValue::Null);
                        tuples_to_process.push(original_tuple);
                    }
                    // // outer join, only keep closest one
                    // (_at_least_one, true) => {
                    //     // let first_result_id = id_vec.remove(0);
                    //     // let first_result_distance = distance_vec.remove(0);
                    //     // let first_result_passage = passage_vec.remove(0);
                    //     // original_tuple.insert("match_ids".into(), first_result_id);
                    //     // original_tuple.insert("match_distances".into(), first_result_distance);
                    //     // original_tuple.insert("passages".into(), first_result_passage);
                    //     original_tuple.insert("match_ids".into(), HabValue::List(id_vec));
                    //     original_tuple
                    //         .insert("match_distances".into(), HabValue::List(distance_vec));
                    //     original_tuple.insert("passages".into(), HabValue::List(passage_vec));
                    // }
                    // // inner join returns all results
                    // (_any_amount, false) => {

                    // for the rest, we include everything we have
                    (_any_amount, _any_policy) => {
                        original_tuple.insert("match_ids".into(), HabValue::List(id_vec));
                        original_tuple
                            .insert("match_distances".into(), HabValue::List(distance_vec));
                        original_tuple.insert("passages".into(), HabValue::List(passage_vec));
                        tuples_to_process.push(original_tuple);
                    }
                }
            }
            // print what keys we have
            let keys = tuples_to_process[0]
                .keys()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ");
            debug!("chroma g7: keys: {keys}");
            // debug!("chroma g8: tuples_to_process: {tuples_to_process:?}");
            debug!(
                "chroma g8: tuples_to_process: {:?}",
                tuples_to_process.len()
            );
            let num_tuples = tuples_to_process.len();
            let emitted_any = num_tuples > 0;
            p.process_tuples(tuples_to_process, self.id, bolts).await;
            if emitted_any {
                OperatorOutput::Something(Some(num_tuples))
            } else {
                OperatorOutput::Nothing
            }
        } else {
            error!(
                "{}:{}:{} : Join operator {} received data from unknown source {source}. expected left was {}.expected right was {}.",
                file!(),
                line!(),
                column!(),
                self.id,
                self.index_stream,
                self.lookup_stream
            );
            panic!("join received data from unknown source {source}. expected left was {}.expected right was {}.", self.index_stream, self.lookup_stream);
        }
    }
}

pub enum InitializationState {
    NotInitialized {
        background_task:
            Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>>,
    },
    Initialized(UnboundedReceiver<Option<Vec<Tuple>>>),
}
pub struct UdfSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub init_state: InitializationState,
}

impl Operator for UdfSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id);
    }
}

impl AsyncSpout for UdfSpout {
    fn initialize<'this>(
        &'this mut self,
        ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()> {
        info!("UdfSpout {} initializing", self.id);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let state = std::mem::replace(&mut self.init_state, InitializationState::Initialized(rx));
        let InitializationState::NotInitialized { background_task } = state else {
            error!(
                "{}:{}:{} : UdfSpout operator {} was initialized twice",
                file!(),
                line!(),
                column!(),
                self.id
            );
            panic!("UdfSpout operator {} was initialized twice", self.id);
        };
        let bg_task = background_task();
        let op_id = self.id;
        async move {
            let input_stream = Box::into_pin(bg_task);
            tokio::spawn(async_spout_produce_loop(
                op_id,
                ready_to_start,
                input_stream,
                tx,
            ));
        }
    }

    fn produce<'this>(
        &'this mut self,
        bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput> {
        async move {
            let InitializationState::Initialized(rx) = &mut self.init_state else {
                error!(
                    "{}:{}:{} : UdfSpout {} was asked to produce when not initialized",
                    file!(),
                    line!(),
                    column!(),
                    self.id
                );
                return OperatorOutput::Nothing;
            };
            let Some(tuples) = rx.recv().await else {
                debug!("UdfSpout operator {} received None from background task, so we will say it is finished", self.id);
                return OperatorOutput::Finished;
            };
            let Some(outputs) = tuples else {
                debug!("UdfSpout operator {} received no items from background task, so we will say it is finished", self.id);
                return OperatorOutput::Finished;
            };
            if outputs.is_empty() {
                debug!(
                    "{}:{}:{} : UdfSpout operator {} received no items from background task, so we will say it had nothing", 
                    file!(),
                    line!(),
                    column!(),
                    self.id
                );
                return OperatorOutput::Nothing;
            }
            let output_len = outputs.len();
            let Some(parent_id) = self.parent else {
                error!(
                    "{}:{}:{} : UdfSpout operator {} has no parent",
                    file!(),
                    line!(),
                    column!(),
                    self.id
                );
                return OperatorOutput::Finished;
            };
            let Some(parent) = bolts.get(parent_id) else {
                return OperatorOutput::Nothing;
            };
            debug!(
                "UdfSpout operator {} sending {} tuples to parent {}",
                self.id, output_len, parent_id
            );
            let _ = parent.process_tuples(outputs, self.id, &bolts).await;
            OperatorOutput::Something(Some(output_len))

            // TODO: do we want to reflect the parent's output in our count from the returned value?
            // match parent.process_tuples(outputs, self.id, bolts) {
            //     OperatorOutput::Something(Some(_)) => OperatorOutput::Something(Some(output_len)),
            //     OperatorOutput::Something(None) => OperatorOutput::Nothing,
            //     OperatorOutput::Nothing => OperatorOutput::Nothing,
            //     OperatorOutput::Finished => OperatorOutput::Finished,
            // }
        }
    }
}

async fn async_spout_produce_loop(
    op_id: usize,
    mut ready_to_start: watch::Receiver<bool>,
    mut input_stream: impl Unpin + Send + Stream<Item = Vec<Tuple>>,
    tx: UnboundedSender<Option<Vec<Tuple>>>,
) {
    use futures::StreamExt;
    info!("UdfSpout operator {op_id} background task started. waiting for signal to start");
    if let Err(e) = ready_to_start.changed().await {
        error!(
            "{}:{}:{} : UdfSpout operator {op_id} failed when ready_to_start channel was closed unexpectedly: {e}",
            file!(),
            line!(),
            column!()
        );
        return;
    }
    info!("UdfSpout operator {op_id} received signal to start");
    loop {
        debug!("UdfSpout {op_id} waiting for next item");
        match input_stream.next().await {
            Some(outputs) => {
                debug!("UdfSpout operator {op_id} received {} items", outputs.len());
                for output in &outputs {
                    let tuple_id = output.id();
                    debug!("UdfSpout operator {op_id} received tuple {tuple_id:?}");
                }
                const MINIMUM_BATCH_SIZE: usize = 4;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("failed to get time since epoch")
                    .as_nanos();
                let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                    smallvec::SmallVec::new();
                for tuple in outputs.iter() {
                    let age = now - tuple.unix_time_created_ns();
                    ages.push((tuple.id(), age as f64));
                }
                debug!(
                    "UdfSpout received {} tuples, ids and ages: {:?}",
                    outputs.len(),
                    ages
                );

                if let Err(e) = tx.send(Some(outputs)) {
                    warn!("UdfSpout operator {op_id} encountered closed error when producing items: {e}");
                    break;
                }
            }
            None => {
                if let Err(e) = tx.send(None) {
                    warn!("UdfSpout operator {op_id} encountered closed error when trying to end the stream: {e}");
                }
                info!("UdfSpout operator {op_id} received None so it's wrapping up");
                break;
            }
        }
    }
    info!("UdfSpout {op_id} background task finished");
}

pub struct AsyncChannelSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub input: BoundedAsyncReceiver,
    pub timeouts: AtomicUsize,
}

impl Operator for AsyncChannelSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id);
    }
}

impl AsyncSpout for AsyncChannelSpout {
    fn initialize<'this>(
        &'this mut self,
        _ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()> {
        async {}
    }

    fn produce<'this>(
        &'this mut self,
        bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput> {
        async move {
            let Some(outputs) = self.input.recv().await else {
                return OperatorOutput::Finished;
            };

            if outputs.is_empty() {
                return OperatorOutput::Nothing;
            }
            let output_len = outputs.len();
            let Some(parent_id) = self.parent else {
                error!("Spout operator {} has no parent", self.id);
                return OperatorOutput::Finished;
            };
            const MINIMUM_BATCH_SIZE: usize = 4;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("failed to get time since epoch")
                .as_nanos();
            let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                smallvec::SmallVec::new();
            for tuple in outputs.iter() {
                let age = now - tuple.unix_time_created_ns();
                ages.push((tuple.id(), age as f64));
            }
            debug!(
                "AsynChannelSpout received {} tuples, ids and ages: {:?}",
                outputs.len(),
                ages
            );

            let Some(parent) = bolts.get(parent_id) else {
                return OperatorOutput::Nothing;
            };
            let _ = parent.process_tuples(outputs, self.id, &bolts).await;
            OperatorOutput::Something(Some(output_len))
        }
    }
}

pub struct Merge {
    pub id: usize,
    pub child: usize,
    pub parent_channel: AsyncPipe,
    pub on_merge_fn: Option<Box<dyn Send + Sync + Fn(&Tuple)>>,
}
impl Operator for Merge {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, _: usize) {
        error!("there's no reason to pass a parent to Merge, since it'll be sending that over the channel");
        unimplemented!("there's no reason to pass a parent to Merge, since it'll be sending that over the channel");
    }
}

impl Bolt for Merge {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        _source: usize,
        _bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        let len = tuples.len();
        if let Some(on_merge_fn) = &self.on_merge_fn {
            for t in &tuples {
                on_merge_fn(t);
            }
        }
        if let Err(e) = self.parent_channel.send(tuples) {
            error!(
                "Failed to send tuples over channel. Did the other side spuriously disconnect? or was it full?: {e:?}",
            );
            return OperatorOutput::Finished;
        }
        OperatorOutput::Something(Some(len))
    }
}

pub struct AsyncMergeSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub input: BoundedAsyncReceiver,
    pub timeouts: AtomicUsize,
    pub value_setter: Option<ValueSetterState>,
    // pub on_receipt: Box<dyn Send + Sync + for<'a> Fn(usize, &'a Tuple) -> BoxFuture<'a, ()>>,
}

impl Operator for AsyncMergeSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

impl AsyncSpout for AsyncMergeSpout {
    fn initialize<'this>(
        &'this mut self,
        _ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()> {
        async {}
    }

    fn produce<'this>(
        &'this mut self,
        bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput> {
        async move {
            let mut outputs = match tokio::time::timeout(
                Duration::from_micros(100),
                self.input.recv(),
            )
            .await
            {
                Ok(Some(out)) => out,
                Ok(None) => {
                    debug!("AsyncMergeSpout operator {} received None from input channel, so it is finished", self.id);
                    return OperatorOutput::Finished;
                }
                Err(_e) => {
                    return OperatorOutput::Nothing;
                }
            };
            match &self.value_setter {
                Some(ValueSetterState::UseDefaults(field_infos)) => {
                    for t in &mut outputs {
                        for field in field_infos.iter() {
                            let std::collections::btree_map::Entry::Vacant(entry) =
                                t.entry(field.key.clone())
                            else {
                                // we have already found this key to be assigned so we do not need to set the default
                                continue;
                            };
                            let value = if let Some(v) = &field.value {
                                v.clone().into()
                            } else {
                                HabValue::Null
                            };
                            entry.insert(value);
                        }
                    }
                }
                Some(ValueSetterState::SetDefaultsFn) => {
                    error!("Default setting function not currently supported for async spout operator {}",self.id);
                    return OperatorOutput::Finished;
                }
                None => (), // no-op
            }
            const MINIMUM_BATCH_SIZE: usize = 4;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("failed to get time since epoch")
                .as_nanos();
            let mut ages: smallvec::SmallVec<[(usize, f64); MINIMUM_BATCH_SIZE]> =
                smallvec::SmallVec::new();
            for tuple in outputs.iter() {
                let age = now - tuple.unix_time_created_ns();
                ages.push((tuple.id(), age as f64));
            }
            debug!(
                "AsyncMergeSpout received {} tuples, ids and ages: {:?}",
                outputs.len(),
                ages
            );

            if outputs.is_empty() {
                return OperatorOutput::Nothing;
            }
            // for output in outputs.iter() {
            //     let on_receipt = &self.on_receipt;
            //     on_receipt(source, output).await;
            // }

            let output_len = outputs.len();
            debug!(
                "AsyncMergeSpout operator {} sending {} tuples to parent",
                self.id, output_len
            );
            let Some(parent_id) = self.parent else {
                error!("Spout operator {} has no parent", self.id);
                return OperatorOutput::Finished;
            };
            let Some(parent) = bolts.get(parent_id) else {
                return OperatorOutput::Nothing;
            };
            let _ = parent.process_tuples(outputs, self.id, &bolts).await;
            OperatorOutput::Something(Some(output_len))
        }
    }
}

impl AsyncSpout for DummyBolt {
    fn initialize<'this>(
        &'this mut self,
        _ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()> {
        async {}
    }

    fn produce<'this>(
        &'this mut self,
        _bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput> {
        async { OperatorOutput::Finished }
    }
}

// #[derive(Debug)]
pub struct PythonRemoteUdf {
    pub id: usize,
    pub child: usize,
    pub initialized: bool,
    pub parent: Option<usize>,
    pub scripts_dir_path: HabString,
    pub script_name: HabString,
    // task_state: PythonRemoteTaskState,
    pub task_state: AsyncPythonRemoteTaskState,
    pub exit_channel: UnboundedSender<()>,
}

pub struct AsyncPythonRemoteTaskState {
    pub port: u16,
    pub pending_items: Arc<AtomicUsize>,
    pub input_to_background_thread: BoundedSender<Vec<Tuple>>,
    pub input_from_main_thread: Option<BoundedReceiver<Vec<Tuple>>>,
    pub output_to_main_thread: UnboundedSender<Vec<Tuple>>,
    pub output_from_background_thread: Option<UnboundedReceiver<Vec<Tuple>>>,
    pub encode: Option<EncoderFunction>,
    pub decode:
        Option<Box<dyn Sync + Send + Fn(zeromq::ZmqMessage, &DashMap<usize, Tuple>) -> Vec<Tuple>>>,
    pub shutdown: Option<Box<dyn Sync + Send + FnOnce() -> zeromq::ZmqMessage>>,
    pub should_stop: watch::Receiver<bool>,
    pub runtime_handle: tokio::runtime::Handle,
    pub script_name: HabString,
    pub scripts_dir_path: HabString,
}

impl Operator for PythonRemoteUdf {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id);
    }
}

impl PythonRemoteUdf {
    pub(crate) async fn start_background_task_async(
        &mut self,
        mut ready_to_start: watch::Receiver<bool>,
    ) {
        debug!(
            "starting background task for python remote udf operator {}",
            self.id
        );
        let exit_channel = self.exit_channel.clone();
        let state: &mut AsyncPythonRemoteTaskState = &mut self.task_state;
        let port: u16 = state.port;
        let pending_items = state.pending_items.clone();
        let mut input_from_main_thread = state
            .input_from_main_thread
            .take()
            .expect("input_from_main_thread not set (was start task method called twice?)");
        let output_to_main_thread = state.output_to_main_thread.clone();
        let encode_fn = state
            .encode
            .take()
            .expect("unable to get encode function (did we call start task twice?)");
        let decode_fn = state
            .decode
            .take()
            .expect("unable to get decode function (did we call start task twice?)");
        let shutdown_fn = state
            .shutdown
            .take()
            .expect("unable to get shutdown function (did we call start task twice?)");
        let mut should_stop = state.should_stop.clone();
        let op_id = self.id;

        let script_name = state.script_name.clone();
        let script_dir = state.scripts_dir_path.clone();
        debug!("python remote udf operator {op_id} at the beginning of background task");
        use zeromq::{Socket, SocketRecv, SocketSend};

        let tuple_map = DashMap::new();

        let mut pathbuf = std::path::PathBuf::from(&*script_dir);
        let background_task_script = "python_remote_worker.py";
        pathbuf.push(background_task_script);
        let Some(background_task_script) = pathbuf.to_str() else {
            error!(
                "{}:{}:{} : failed to convert path {} to string",
                file!(),
                line!(),
                column!(),
                pathbuf.display()
            );
            panic!("failed to convert path {} to string", pathbuf.display());
        };
        // use std::io::Write;
        // let mut addr = *b"tcp://127.0.0.1:____";
        // let addr_len = addr.len();
        // let start_index = addr_len - 4;
        // let end_index = addr_len;

        // let mut addr = *b"ipc:///tmp/zmq____.icp";
        // //  we need to account for the extension
        // let start_index = addr.len() - "ipc".len() - 4;
        // let end_index = addr.len() - 4;
        // // print port padded to 4 digits
        // write!(&mut addr[start_index..end_index], "{:04}", port).expect("failed to format port");
        let addr = format!("tcp://127.0.0.1:{}", port);
        let addr = addr.as_str();
        debug!("addr for python remote udf operator {op_id} is {:?}", addr);

        // start up python file client
        info!("Starting server for python remote udf operator {op_id}");
        // let mut command = tokio::process::Command::new("python");
        // let mut child = match command
        //     .arg(background_task_script)
        //     .arg(&*script_name)
        //     .arg(addr)
        //     .stdout(std::process::Stdio::piped())
        //     .spawn()
        let mut command = tokio::process::Command::new("stdbuf");
        let mut child = match command
            .arg("-oL") // Set stdout to line-buffered
            .arg("python")
            .arg(background_task_script)
            .arg(&*script_name)
            .arg(addr)
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "unable to start python process {} in operator {} with error {:?}",
                    script_name, self.id, e
                );
                panic!(
                    "unable to start python process {} in operator {} with error {:?}",
                    script_name, self.id, e
                );
            }
        };
        // make an output channel
        let (tx, mut rx) = tokio::sync::watch::channel(false);

        // Capture the stdout of the child process
        if let Some(stdout) = child.stdout.take() {
            let reader = tokio::io::BufReader::new(stdout);
            let script_name = script_name.clone();
            // TODO: do we need the should_stop?
            // the outside should be able to handle it by sending the kill signal if the process hasn't exited upon receiving the shutdown message
            // let mut should_stop = should_stop.clone();
            use tokio::io::AsyncBufReadExt;
            tokio::spawn(async move {
                let mut lines = reader.lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) => {
                            debug!("Child process #{op_id} ({script_name}) output: {line}");
                            if line.contains("socket connected") {
                                if let Err(e) = tx.send(true) {
                                    error!("Child process #{op_id} ({script_name}) error: failed to send server started signal to primary background thread in python remote udf operator {op_id}: {e:?} ");
                                    error!("Child process #{op_id} ({script_name}): exiting Child process observation loop now");
                                    break;
                                }
                            }
                        }
                        Ok(None) => {
                            debug!("Child process #{op_id} ({script_name}) output: EOF");
                            break;
                        }
                        Err(e) => {
                            error!("Child process #{op_id} ({script_name}) output: error: {e:?}",);
                            break;
                        }
                    }
                }
            });
        }

        debug!("waiting for server to start in python remote udf operator {op_id}");
        // give python vm time to start up and warm up
        match tokio::time::timeout(Duration::from_millis(30_000), rx.changed()).await {
            Ok(Ok(_)) => {
                info!("server started for python remote udf operator {op_id}");
            }
            Ok(Err(_)) => {
                error!("server start channel was dropped for python remote udf operator {op_id}");
            }
            Err(_) => {
                warn!("server start channel timed out for python remote udf operator {op_id}");
            }
        }
        debug!("after the wait point in python remote op {op_id}. does the other side say it's started yet?");

        info!("Binding socket for python remote udf operator {op_id}");
        let mut socket = zeromq::ReqSocket::new();

        // match socket.connect(addr).await {
        match socket.bind(addr).await {
            Ok(_endpoint) => {
                info!("successfully connected to python side");
            }
            Err(e) => {
                warn!("error connecting to start server for python remote udf operator {op_id}");
                match e {
                    zeromq::ZmqError::Endpoint(_endpoint_error) => error!("error was Endpoint"),
                    zeromq::ZmqError::Network(_error) => error!("error was Network"),
                    zeromq::ZmqError::NoSuchBind(_endpoint) => error!("error was NoSuchBind"),
                    zeromq::ZmqError::Codec(_codec_error) => error!("error was Codec(codec_error"),
                    zeromq::ZmqError::Socket(_err) => error!("error was Socket"),
                    zeromq::ZmqError::BufferFull(_err) => error!("error was BufferFull"),
                    zeromq::ZmqError::ReturnToSender {
                        reason: _,
                        message: _,
                    } => error!("error was ReturnToSender"),
                    zeromq::ZmqError::ReturnToSenderMultipart {
                        reason: _,
                        messages: _,
                    } => error!("error was ReturnToSenderMultipart"),
                    zeromq::ZmqError::Task(_task_error) => error!("error was Task(task_error"),
                    zeromq::ZmqError::Other(_err) => error!("error was Other(_"),
                    zeromq::ZmqError::NoMessage => error!("error was NoMessage"),
                    zeromq::ZmqError::PeerIdentity => error!("error was PeerIdentity"),
                    zeromq::ZmqError::UnsupportedVersion(_err) => {
                        error!("error was UnsupportedVersion")
                    }
                }
            }
        }

        let _background_task: tokio::task::JoinHandle<Result<(), Box<dyn Sync+Send+std::error::Error>>> = state.runtime_handle.spawn(async move {
            info!("waiting for signal to start loop");
            ready_to_start.changed().await.expect("ready_to_start channel was closed unexpectedly");
            info!("starting loop");

            // This loop waits for each input to receive its output
            // TODO: can we handle multiple requests asynchronously?
            //  when we are able to do that, the dashmap of ids should help us re-match the outputs to the inputs they were derived from
            'outer: loop {
                let receive_inputs = tokio::time::timeout(Duration::from_micros(1000), input_from_main_thread.recv());

                let wait_for_should_stop = should_stop.changed();
                // select on both
                let receive_inputs_output = tokio::select! {
                    stop_signal = wait_for_should_stop => {
                        if stop_signal.is_err() {
                            warn!("should stop channel was dropped for python remote udf operator {op_id}");
                        }else{
                            info!("should stop reached for python remote udf operator {op_id}");
                        }
                        break 'outer;
                    }
                    val = receive_inputs => {
                        val
                    }
                };

                // TODO: changel loop to push all updates in a batch
                match receive_inputs_output {
                    Ok(val) => {
                        let Some(inputs) = val else {
                            warn!("Failed to receive inputs from main thread of remote python udf {op_id}, exiting");
                            break 'outer;
                        };
                        debug!("background thread for python remote udf operator {op_id} got {} inputs from main", inputs.len());
                        let now_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_nanos();
                        for t in &inputs {
                            let time_created = t.unix_time_created_ns;
                            debug!("background thread input had time created {time_created}-unix-ns so it is {} ms old", (now_ns - time_created) / 1_000_000);
                        }
                        match &encode_fn {
                            EncoderFunction::Single(encode_fn) =>{
                                for input in inputs {
                                    let tuple_id = input.id();
                                    let msg = encode_fn(tuple_id as _, &input);
                                    // let time_created = input.get("time_created").expect("time_created field not found").as_unsigned_long_long().expect("time_created field is not an int");
                                    let time_created = input.unix_time_created_ns;
                                    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_nanos();
                                    let time_diff = now - time_created;
                                    debug!("time diff when sending from PRUDF#{} ({script_name}) is {}ns", op_id, time_diff);
                                    'log_python_send: {
                                        let mut location_buffer = *b"send_to_python_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                                        let write_start = "send_to_python_".len();
                                        let max_script_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                                        let write_end = write_start + max_script_len;
                                        use std::io::Write;
                                        // write the script name
                                        if let Err(e) = write!(location_buffer[write_start..write_end].as_mut(), "{}", script_name) {
                                            error!("Failed to write to location buffer script name for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                                            break 'log_python_send;
                                        }

                                        let write_start = location_buffer.len() - 8;
                                        if let Err(e) = write!(location_buffer[write_start..].as_mut(), "{op_id}") {
                                            error!("Failed to write to location buffer op id for sending tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                                            break 'log_python_send;
                                        }
                                        let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                                            error!("Failed to convert location buffer to string for sending tuple {tuple_id} in python remote udf operator {op_id}");
                                            break 'log_python_send;
                                        };

                                        if let Err(e) = crate::global_logger::log_data(tuple_id, location_buffer.to_raw_key(), NO_AUX_DATA){
                                            error!("Failed to log send to python in python remote udf operator {op_id} with error {:?}", e);
                                        }
                                    }
                                    tuple_map.insert(tuple_id as _, input);
                                    if let Err(e) = socket.send(msg).await{
                                        match e{
                                            zeromq::ZmqError::Endpoint(_) => error!("python remote udf operator {op_id} Failed to send message to Python with endpoint error"),
                                            zeromq::ZmqError::Network(_) => error!("python remote udf operator {op_id} Failed to send message to Python with network error"),
                                            zeromq::ZmqError::NoSuchBind(_) => error!("python remote udf operator {op_id} Failed to send message to Python with no such bind error"),
                                            zeromq::ZmqError::Codec(_) => error!("python remote udf operator {op_id} Failed to send message to Python with codec error"),
                                            zeromq::ZmqError::Socket(_) => error!("python remote udf operator {op_id} Failed to send message to Python with socket error"),
                                            zeromq::ZmqError::BufferFull(_) => error!("python remote udf operator {op_id} Failed to send message to Python with buffer full error"),
                                            zeromq::ZmqError::ReturnToSender { reason, message: _ } => error!("python remote udf operator {op_id} Failed to send message to Python with return to sender error. \n\treason: {:?}", reason),
                                            zeromq::ZmqError::ReturnToSenderMultipart { reason: _, messages: _ } => error!("python remote udf operator {op_id} Failed to send message to Python with return to sender multipart error"),
                                            zeromq::ZmqError::Task(_) => error!("python remote udf operator {op_id} Failed to send message to Python with task error"),
                                            zeromq::ZmqError::Other(_) => error!("python remote udf operator {op_id} Failed to send message to Python with Other error"),
                                            zeromq::ZmqError::NoMessage =>  error!("python remote udf operator {op_id} Failed to send message to Python with NoMessage error"),
                                            zeromq::ZmqError::PeerIdentity =>  error!("python remote udf operator {op_id} Failed to send message to Python with PeerIdentity error"),
                                            zeromq::ZmqError::UnsupportedVersion(_) =>  error!("python remote udf operator {op_id} Failed to send message to Python with UnsupportedVersion error"),
                                        }
                                        panic!("Failed to send message to Python worker {} (in python remote operator #{})", script_name, op_id);
                                    }
                                    debug!("python udf {op_id} sent one to python {}", script_name);
                                    pending_items.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                    let send_outputs =   tokio::time::timeout(Duration::from_millis(120_000), socket.recv());

                                    match send_outputs.await {
                                        Ok(outputs) => {
                                            let outputs = outputs.unwrap_or_else(|e| {
                                                error!("{}:{}:{} : Failed to receive outputs from Python in python remote udf operator #{}: {e}", file!(), line!(), column!(), op_id);
                                                panic!("Failed to receive outputs from Python in python remote udf operator #{op_id}: {e}");
                                            });
                                            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_nanos();
                                            let time_diff = now - time_created;
                                            debug!("time diff when data came back from {} is {} ns", &script_name,  time_diff);
                                            'log_return_from_python:{
                                                let mut location_buffer = *b"return_from_python_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                                                let write_start = "return_from_python_".len();
                                                let max_script_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                                                let write_end = write_start + max_script_len;
                                                use std::io::Write;
                                                // write the script name
                                                if let Err(e) = write!(location_buffer[write_start..write_end].as_mut(), "{}", script_name) {
                                                    error!("Failed to write to location buffer script name for returning tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                                                    break 'log_return_from_python;
                                                }

                                                let write_start = location_buffer.len() - 8;
                                                if let Err(e) = write!(location_buffer[write_start..].as_mut(), "{op_id}") {
                                                    error!("Failed to write to location buffer op id for returning tuple {tuple_id} in python remote udf operator {op_id} with error {:?}", e);
                                                    break 'log_return_from_python;
                                                }
                                                let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                                                    error!("Failed to convert location buffer to string for returning tuple {tuple_id} in python remote udf operator {op_id}");
                                                    break 'log_return_from_python;
                                                };

                                                if let Err(e) = crate::global_logger::log_data(tuple_id, location_buffer.to_raw_key(), NO_AUX_DATA){
                                                    error!("Failed to log return from python in python remote udf operator {op_id} with error {:?}", e);
                                                }
                                            }
                                            let outputs = decode_fn(outputs, &tuple_map);
                                            let num_outputs = outputs.len();
                                            debug!("bg thread message contained {num_outputs} inputs from python");
                                            pending_items.fetch_sub(outputs.len(), std::sync::atomic::Ordering::Relaxed);
                                            output_to_main_thread.send(outputs).expect("Failed to send outputs from Python back to main thread");
                                            debug!("sent {} outputs to main thread", num_outputs);
                                        }
                                        Err(_e) => {
                                            // no outputs received
                                            info!("no outputs received in operator {op_id} (timeout). exiting now");
                                        }
                                    }
                                    // TODO: should we check should_stop here in the middle of a non-batched loop?
                                }
                            }
                            EncoderFunction::Batch(encode_fn) => {
                                let mut ids = vec![];
                                let mut earliest_time = u128::MAX;
                                for input in &inputs {
                                    let tuple_id = input.id();
                                    let time_created = input.unix_time_created_ns;
                                    if time_created < earliest_time {
                                        earliest_time = time_created;
                                    }

                                    ids.push(tuple_id as usize);
                                    pending_items.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                let msg = encode_fn(&ids, &inputs);
                                for (tuple_id, tuple) in ids.into_iter().zip(inputs.into_iter()) {
                                    tuple_map.insert(tuple_id, tuple);
                                }
                                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_nanos();
                                let time_diff = now - earliest_time;
                                debug!("time diff when sending batch is {} ns", time_diff);
                                socket.send(msg).await?;
                                debug!("sent one to python");
                                let send_outputs =   tokio::time::timeout(Duration::from_millis(1000), socket.recv());
                                match send_outputs.await {
                                    Ok(outputs) => {
                                        let outputs = outputs.expect("Failed to receive outputs from Python");
                                        debug!("bg thread for remote python udf {op_id} got a message back from python");
                                        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_nanos();
                                        let time_diff = now - earliest_time;
                                        debug!("time diff when data came back is {}ns", time_diff);        
                                        let outputs = decode_fn(outputs, &tuple_map);
                                        let num_outputs = outputs.len();
                                        // TODO: log number of outputs
                                        debug!("python remote udf {op_id} bg thread message contained {num_outputs} inputs from python");
                                        pending_items.fetch_sub(outputs.len(), std::sync::atomic::Ordering::Relaxed);
                                        output_to_main_thread.send(outputs).expect("Failed to send outputs from Python back to main thread");
                                        debug!("python remote udf {op_id} sent {} outputs to main thread", num_outputs);
                                    }
                                    Err(_e) => {
                                        // no outputs received
                                        info!("no outputs received (timeout). exiting now");
                                    }
                                }
                                // TODO: I don't see where should_stop is set to true?
                                // if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                                //     info!("should stop reached according to remote python udf");
                                // }
                                // TODO: what should we do if we get a should_stop signal in the middle of a batch?
                            }
                        }
                    }
                    Err(_e) => {
                        // no inputs received
                        continue;
                    },
                }
            }

            // TODO: send shutdown notice to child when finished
            // (how do we get here? do we get a condvar or something from the main thread?)
            socket.send(shutdown_fn()).await?;
            // wait with timeout
            match tokio::time::timeout(Duration::from_millis(500), child.wait()).await {
                Ok(Ok(status)) => {
                    info!("child for remote python operator {op_id} exited with: {}", status);
                        }
                Ok(Err(e)) => {
                    error!("error checking child status: {}", e);
                    // kill with timeout
                    match tokio::time::timeout(Duration::from_millis(500), child.kill()).await{
                        Ok(Ok(())) => {},
                        Ok(Err(e)) => {
                            error!("child process for remote python operator {op_id} was still running and we were unable to kill the process in 500ms:\n{e:?}");
                        }
                        Err(e) => {
                            error!("child process for remote python operator {op_id} was still running and we were unable to kill the process in 500ms:\n{e:?}");
                        }
                    }
                }
                Err(e) => {
                    warn!("child process for remote python operator {op_id} was still running and we were unable to kill the process in 500ms:\n{e:?}");
                    match tokio::time::timeout(Duration::from_millis(500), child.kill()).await{
                        Ok(Ok(())) => {},
                        Ok(Err(e)) => {
                            error!("child process for remote python operator {op_id} was still running and we were unable to kill the process in 500ms:\n{e:?}");
                        }
                        Err(e) => {
                            error!("child process for remote python operator {op_id} was still running and we were unable to kill the process in 500ms:\n{e:?}");
                        }
                    }
                }
            }

            debug!("remote python udf operator {op_id} finished background task. signalling to main thread");

            if let Err(e) = exit_channel.send(()) {
                error!("failed to send exit message to main thread in remote python udf operator {op_id}: {e:?}");
            }

            Ok(())
        });
    }
}

impl AsyncSpout for PythonRemoteUdf {
    fn initialize<'this>(
        &'this mut self,
        ready_to_start: watch::Receiver<bool>,
    ) -> impl 'this + Send + Future<Output = ()> {
        async {
            if !self.initialized {
                // TODO: do we want to have a fallible initialize specifically for start background task?
                self.start_background_task_async(ready_to_start).await;
                self.initialized = true;
            } else {
                error!(
                    "Python remote udf operator {} was initialized twice",
                    self.id
                );
            }
        }
    }

    fn produce<'this>(
        &'this mut self,
        // TODO: should this be a reference?
        bolts: Arc<[PhysicalOperator]>,
    ) -> impl 'this + Send + Future<Output = OperatorOutput> {
        async move {
            let Some(output_channel) = &mut self.task_state.output_from_background_thread else {
                error!("Python remote udf operator {} was asked to produce items but the spout doesn't have the channel to receive input from the background task", self.id);
                return OperatorOutput::Nothing;
            };
            let outputs = match tokio::time::timeout(
                Duration::from_micros(100),
                output_channel.recv(),
            )
            .await
            {
                Ok(Some(outputs)) => outputs,
                Err(_elapsed) => {
                    trace!("Python remote udf operator {} timed out waiting for output from background task", self.id);
                    // nothing so far
                    return OperatorOutput::Nothing;
                }
                Ok(None) => {
                    warn!(
                        "Failed to receive output from background task because the channel was closed"
                    );
                    return OperatorOutput::Finished;
                }
            };
            if outputs.is_empty() {
                warn!(
                    "Python remote udf operator {} received no outputs from background task",
                    self.id
                );
                return OperatorOutput::Nothing;
            }
            debug!(
                "Python remote udf operator {} received {} outputs from background task",
                self.id,
                outputs.len()
            );
            let output_len = outputs.len();
            let Some(parent_id) = self.parent else {
                error!("Python remote udf operator {} has no parent", self.id);
                return OperatorOutput::Finished;
            };
            let Some(parent) = bolts.get(parent_id) else {
                return OperatorOutput::Nothing;
            };
            let _ = parent.process_tuples(outputs, self.id, &bolts).await;
            OperatorOutput::Something(Some(output_len))
        }
    }
}

impl Bolt for PythonRemoteUdf {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        _source: usize,
        _bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        let num_to_output = tuples.len();
        if tuples.is_empty() {
            return OperatorOutput::Something(None);
        }
        // send the tuples to the background thread
        let queue = &self.task_state.input_to_background_thread;
        debug!("inside process_tuples for python remote udf operator {}. queue has max capacity {} and current capacity {}", self.id, queue.max_capacity(), queue.capacity());
        if let Err(e) = self
            .task_state
            .input_to_background_thread
            .send(tuples)
            .await
        {
            warn!(
                "PythonRemoteUdf {} failed to send inputs to background thread: {:?}",
                self.id, e
            );
        }
        OperatorOutput::Something(Some(num_to_output))
    }
}

pub struct AsyncChannelRouter {
    pub id: usize,
    pub child: usize,
    pub parent_channels: Arc<Vec<AsyncPipe>>,
    pub route: Arc<Mutex<Box<dyn Send + Sync + FnMut(Vec<Tuple>, &[AsyncPipe]) -> Option<usize>>>>,
}

impl Operator for AsyncChannelRouter {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, _id: usize) {
        error!("we don't want parents on the channel router");
        unimplemented!("we don't want parents on the channel router");
        // self.parents.push(id);
        // TODO: we need a way to get parent channels to push
        // self.parent_channels.push(value)
    }
}

impl Bolt for AsyncChannelRouter {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        _source: usize,
        _bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        info!(
            "channel router {} received {} tuples",
            self.id,
            tuples.len()
        );
        if tuples.is_empty() {
            OperatorOutput::Nothing
        } else {
            // the routing operation may choose to drop some data, or to hide how much it output in total
            // all data is sent or dropped *immediately*, so there is no awaiting on the output.
            // The queues throughout the application will inform this step as to whether there is capacity,
            //  and the transmission will either succeed or be dropped at this point. This is effectively a terminal operator in that regard.
            // TODO: if this is blocking, then the system may fall behind. can this be made 'static so it can be used with spawn_blocking?

            let num_outputs = match tokio::task::spawn_blocking({
                let route = self.route.clone();
                let parent_channels = self.parent_channels.clone();
                move || (route.lock().unwrap())(tuples, &parent_channels)
            })
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    error!("channel router {} failed to route tuples: {:?}", self.id, e);
                    None
                }
            };
            OperatorOutput::Something(num_outputs)
        }
    }
}

// TODO: handle outputs
// we need to be able to say which ones we want because the detector has too many outputs
// TODO: extract those outputs
pub struct OnnxInferenceOperator {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub model_path: HabString,
    pub batch_size: usize,
    pub onnx_session: std::sync::Arc<parking_lot::Mutex<ort::session::Session>>,
    // TODO: have a Heap<std::cmp::Rev<(Instant, Shape)>> to clear out least-recently used tensors. for now we will just use a map
    // pub tensors: parking_lot::Mutex<Vec<ort::value::Tensor<f32>>>,
    pub args: Vec<OnnxValue>,
    pub outputs: Vec<OnnxValue>,
}

// returns the number of items written
fn concatenate_into_tensor<'a, 'b, 'c>(
    tensors: impl IntoIterator<Item = anyhow::Result<ndarray::ArrayViewD<'a, f32>>>,
    out: &'b mut ndarray::ArrayViewMutD<'c, f32>,
    start_at_index: Option<usize>,
) -> anyhow::Result<usize> {
    let tensors = tensors.into_iter();
    // make copy so we have a reference for when we need it, without conflicting with the references made later for mutation
    let out_shape: Batched<usize> = out.shape().into();
    let out_shape = &out_shape[..];
    let Some(expected_items) = out_shape.get(0) else {
        anyhow::bail!("output tensor has no shape/empty shape");
    };
    let Some(expected_item_shape) = out_shape.get(1..) else {
        anyhow::bail!("output tensor was 1D so you can't concatenate into it");
    };

    let mut current_index = start_at_index.unwrap_or(0);
    for tensor in tensors {
        let tensor = tensor?;
        let tensor_shape = tensor.shape();
        let &[batch_size, ref item_shape @ ..] = tensor_shape else {
            anyhow::bail!("input tensor was not at least 1D");
        };
        // we validate that the lengths are the same, as well as the values being uniform
        if item_shape != expected_item_shape {
            anyhow::bail!("input tensor shape {item_shape:?} does not match expected shape {expected_item_shape:?}");
        }
        if batch_size != 1 {
            warn!("input tensor had batch size {batch_size} which is not 1. this may lead to unexpected results as multiple items will be concatenated at once into the output tensor, which expects a batch size of 1 for each input tensor, totalling to {expected_items} items");
        }
        let end_index = current_index + batch_size;
        if end_index > *expected_items {
            anyhow::bail!("output tensor is not large enough to hold all input tensors. output has room for {expected_items} items but we need at least {} items", current_index + batch_size);
        }
        let mut view_mut = out.slice_axis_mut(
            ndarray::Axis(0),
            ndarray::Slice::from(current_index..end_index),
        );
        debug!("writing tensor with shape {:?} into output tensor at index range {}..{} (exclusive) of shape {:?}", tensor.shape(), current_index, end_index, view_mut.shape());
        view_mut.assign(&tensor);
        current_index = end_index;
    }
    // return how many items we wrote, so the caller knows what point has relevant data versus which point has junk data
    Ok(current_index)
}

// for a particular input to an onnx model, we will iterate through some tuples, then extract the relevant field from each tuple
// and prepare a single ndarray that can be passed to onnxruntime as input
// fn prepare_inputs(
//     tuples: impl Iterator<Item = &Tuple>,
//     tuple_to_input: impl Fn(&Tuple) -> ndarray::ArrayViewD<'_, f32>,
//     allocator: &ort::memory::Allocator,
//     expected_shape: &[usize],
//     tensor_lookup: &mut Vec<ort::value::Tensor<f32>>,
// ) -> anyhow::Result<ort::value::Tensor<f32>> {
// }

pub enum OnnxValue {
    SplitParts {
        shape_field: HabString,
        buffer_field: HabString,
        buffer: parking_lot::Mutex<Option<ort::value::Tensor<f32>>>,
    }, // NdArray{
       //     ndarray_field: HabString,
       // },
}

impl Operator for OnnxInferenceOperator {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id);
    }
}

#[cfg(test)]
#[test]
fn ndarray_stack_dimensions() {
    let arr2_a = ndarray::Array::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let arr2_b = ndarray::Array::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let arr3_stacked = ndarray::stack(ndarray::Axis(0), &[arr2_a.view(), arr2_b.view()]).unwrap();
    assert_eq!(arr3_stacked.shape(), &[2, 2, 3]);
}

#[cfg(test)]
#[test]
fn ndarray_concatenate_dimensions() {
    let arr2_a = ndarray::Array2::from_shape_vec((1, 3), vec![1., 2., 3.]).unwrap();
    let arr2_b = ndarray::Array2::from_shape_vec((1, 3), vec![4., 5., 6.]).unwrap();
    let arr2_stacked =
        ndarray::concatenate(ndarray::Axis(0), &[arr2_a.view(), arr2_b.view()]).unwrap();
    assert_eq!(arr2_stacked.shape(), &[2, 3]);
}

impl OnnxInferenceOperator {
    fn null_outputs(&self, tuple: &mut Tuple) {
        for output in &self.outputs {
            match output {
                OnnxValue::SplitParts {
                    shape_field,
                    buffer_field,
                    buffer: _,
                } => {
                    tuple.insert(shape_field.clone(), HabValue::Null);
                    tuple.insert(buffer_field.clone(), HabValue::Null);
                }
            }
        }
    }

    fn populate_batch_captures(
        &self,
        inference_future_captures: &mut Vec<(
            Vec<usize>,
            Vec<usize>,
            Vec<ort::session::SessionInputValue<'_>>,
        )>,
        current_batch: &mut smallvec::SmallVec<
            [(usize, usize, Vec<ndarray::ArrayViewD<'_, f32>>); 16],
        >,
        inputs_len: usize,
    ) -> Result<(), impl IntoIterator<Item = usize>> {
        let tuple_indices = current_batch
            .iter()
            .map(|(i, _, _)| *i)
            .collect::<Batched<_>>();
        // let tuple_ids = current_batch.iter().map(|(_, id, _)| *id).collect::<Batched<_>>();

        for (_, _, inputs) in &*current_batch {
            if inputs.len() != inputs_len {
                return Err(tuple_indices);
            }
        }
        // let's zip all of them together now
        let inputs_indices: Vec<usize> = current_batch.iter().map(|(i, _, _)| *i).collect();
        let input_tuple_ids: Vec<usize> = current_batch.iter().map(|(_, id, _)| *id).collect();
        let inputs = (0..inputs_len).map(|i| {
            let arrays: Batched<ndarray::ArrayViewD<f32>> = current_batch
                .iter()
                .map(|(_, _, inputs)| inputs[i].view())
                .collect();
            let all_ones = arrays.iter().all(|a| a.shape().get(0).map(|v|*v== 1).unwrap_or(false));
            let stacked = if all_ones {
                match ndarray::concatenate(
                    ndarray::Axis(0),
                    &arrays.iter().map(|a| a.view()).collect::<Batched<_>>()[..],
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("onnx inference operator {}: failed to concatenate input arrays for input index {} in batch: {:?}", self.id, i, e);
                        return None;
                    }
                }
            } else { match ndarray::stack(
                ndarray::Axis(0),
                &arrays.iter().map(|a| a.view()).collect::<Batched<_>>()[..],
            ) {
                Ok(v) => v,
                Err(e) => {
                    error!("onnx inference operator {}: failed to stack input arrays for input index {} in batch: {:?}", self.id, i, e);
                    return None;
                }
            }};
            let tensor = match ort::value::Tensor::from_array(stacked) {
                Ok(v) => v,
                Err(e) => {
                    error!("onnx inference operator {}: failed to create tensor from stacked array for input index {} in batch: {:?}", self.id, i, e);
                    return None;
                }
            };
            Some(
                ort::session::input::SessionInputValue::Owned(
                    tensor.into_dyn()
                )
            )
        }).collect::<Option<Vec<_>>>();

        let Some(inputs) = inputs else {
            return Err(tuple_indices);
        };

        inference_future_captures.push((inputs_indices, input_tuple_ids, inputs));
        Ok(())
    }
}

const MAX_INLINE_BATCH_SIZE: usize = 16;
pub(crate) type Batched<T> = smallvec::SmallVec<[T; MAX_INLINE_BATCH_SIZE]>;
pub(crate) type SmallBatch<T> = smallvec::SmallVec<[T; MAX_INLINE_BATCH_SIZE / 2]>;

impl Bolt for OnnxInferenceOperator {
    async fn process_tuples<'a>(
        &'a self,
        tuples: Vec<Tuple>,
        source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        static USE_OLD: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| match std::env::var("WATERSHED_ONNX_INFERENCE_OLD") {
                Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
                Err(_) => false,
            });

        if *USE_OLD {
            warn!("Using old onnx inference operator code path. This is not recommended and may be removed in future versions. Set the environment variable WATERSHED_ONNX_INFERENCE_OLD=0, or do not set it, in order to use the new code path.");
            return self.process_tuples_old(tuples, source, bolts).await;
        } else {
            self.process_tuples_new(tuples, source, bolts).await
        }
    }
}
impl OnnxInferenceOperator {
    async fn process_tuples_new<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        let self_id = self.id;
        if tuples.is_empty() {
            warn!(
                "onnx inference operator {} received no tuples to process",
                self_id
            );
            return_tuple_vec(tuples);
            return OperatorOutput::Nothing;
        }
        let Some(parent) = self.parent else {
            warn!("onnx inference operator {} has no parent", self_id);
            return OperatorOutput::Nothing;
        };
        let Some(parent) = bolts.get(parent) else {
            error!(
                "Index {} as parent for OnnxInferenceOperator {} is out of bounds",
                parent, self_id
            );
            return OperatorOutput::Nothing;
        };
        let mut tuple_index = 0;
        let start_at_index = Some(0);
        let encountered_failure = loop {
            if tuple_index >= tuples.len() {
                break false;
            }

            let batch_start_index = tuple_index;
            let batch_end_index = (tuple_index + self.batch_size).min(tuples.len());
            let this_batch_size = batch_end_index - batch_start_index;
            let mut input_args = Batched::new();
            let mut expected_rows = None;
            let mut encountered_failure = false;
            for arg in &self.args {
                match arg {
                    OnnxValue::SplitParts {
                        shape_field,
                        buffer_field,
                        buffer: buffer_lock,
                    } => {
                        let Some(mut buffer) = std::mem::take(&mut *buffer_lock.lock()) else {
                            encountered_failure = true;
                            error!("Onnx Inference Operator {}: no available output buffer for argument with shape field {:?} and buffer field {:?}. Quitting inference early.", self_id, shape_field, buffer_field);
                            break;
                        };
                        let tensors = tuples[batch_start_index..batch_end_index].iter().map(|t| {
                            use anyhow::bail;
                            let Some(shape) = t.get(shape_field) else {
                                // ("shape field not found");
                                let msg = format!("Onnx Inference Operator {}: shape field {:?} not found in tuple with id {}. Available fields are {:?}", self_id, shape_field, t.id(), t.keys().collect::<Vec<_>>());
                                error!("{msg}");
                                bail!("{msg}");
                            };
                            let Some(buffer) = t.get(buffer_field) else {
                                // ("shape field not found");
                                let msg = format!("Onnx Inference Operator {}: buffer field {:?} not found in tuple with id {}. Available fields are {:?}", self_id, shape_field, t.id(), t.keys().collect::<Vec<_>>());
                                error!("{msg}");
                                bail!("{msg}");
                            };
                            let Some(shape) = shape.as_shape_buffer() else {
                                let msg = format!("onnx inference operator {}: shape field {:?} is not a shape buffer for tuple with id {}", self_id, shape_field, t.id());
                                warn!("{msg}");
                                bail!("{msg}");
                            };
                            let Some(buffer) = buffer.as_int_buffer() else {
                                let msg = format!("onnx inference operator {}: buffer field {:?} is not a int buffer for tuple with id {}", self_id, buffer_field, t.id());
                                warn!("{msg}");
                                bail!("{msg}");
                            };
                            let buffer = bytemuck::cast_slice::<i32, f32>(buffer);
                            match ndarray::ArrayViewD::from_shape(shape, buffer) {
                                Ok(v) => Ok(v),
                                Err(e) => {
                                    // ("failed to create ndarray from shape and buffer");
                                    let msg = format!("Onnx Inference Operator {}: failed to create ndarray from shape and buffer: {:?} for tuple with id {}. Shape was {:?} and length of buffer was {}", self_id, e, t.id(), shape, buffer.len());
                                    error!("{msg}");
                                    bail!("{msg}");
                                }
                            }
                        });

                        let mut out = buffer.extract_array_mut();
                        let written_rows = match concatenate_into_tensor(
                            tensors,
                            &mut out,
                            start_at_index,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                error!("Onnx Inference Operator {}: failed to concatenate tensors into output tensor for argument with shape field {:?} and buffer field {:?}. Error: {e:?}", self_id, shape_field, buffer_field);
                                // put back the buffer we took
                                *buffer_lock.lock() = Some(buffer);
                                encountered_failure = true;
                                break;
                            }
                        };
                        if let Some(expected_rows) = expected_rows {
                            if expected_rows != written_rows {
                                // encountered_failure = true;
                                error!("Onnx Inference Operator {}: inconsistent number of rows written for argument with shape field {:?} and buffer field {:?}. expected {} but got {}", self_id, shape_field, buffer_field, expected_rows, written_rows);
                                // put back the buffer we took
                                *buffer_lock.lock() = Some(buffer);
                                encountered_failure = true;
                                break;
                            }
                        } else {
                            expected_rows = Some(written_rows);
                        }
                        input_args.push(buffer);
                    }
                }
            }

            if let Some(0) | None = expected_rows {
                error!(
                    "Onnx Inference Operator {}: no input arguments were processed successfully",
                    self_id
                );
                encountered_failure = true;
            }
            // put back any buffers we took
            let expected_rows = if encountered_failure {
                return_values(input_args, &self.args);
                break true;
            } else {
                expected_rows.expect("we always set expected_rows if we didn't encounter failure")
            };
            let tuple_ids = tuples[batch_start_index..batch_end_index]
                .iter()
                .map(|t| t.id())
                .collect::<Batched<_>>();

            let num_outputs = expected_rows;
            let session = self.onnx_session.clone();
            let run_result = tokio::task::spawn_blocking(move || {
                let ref_input_args = input_args.iter().map(|b| ort::session::SessionInputValue::View(b.view().into_dyn())).collect::<Batched<_>>();
                let mut session_lock = session.lock();
                let inference_start = std::time::Instant::now();
                let run_result = session_lock.run(ref_input_args.as_slice());
                drop(ref_input_args);
                let elapsed_micros = inference_start.elapsed().as_micros();
                debug!("onnx inference operator {}: inference for a batch of size {} with tuple ids {:?} took {} us", self_id, this_batch_size, &tuple_ids, elapsed_micros);
                    // log time in global logger
                    'log_onnx_inference_time: {
                        let mut location_buffer = *b"onnx_inference_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                        let write_start = "onnx_inference_".len();
                        let max_op_id_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                        let write_end = write_start + max_op_id_len;
                        use std::io::Write;
                        // write the op id
                        if let Err(e) = write!(location_buffer[write_start..write_end].as_mut(), "{}", self_id) {
                            error!("Failed to write to location buffer op id for onnx inference operator {} with error {:?}", self_id, e);
                            break 'log_onnx_inference_time;
                        }
                        let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                            error!("Failed to convert location buffer to string for onnx inference operator {}", self_id);
                            break 'log_onnx_inference_time;
                        };

                        for tid in tuple_ids.iter(){
                            if let Err(e) = crate::global_logger::log_data(*tid, location_buffer.to_raw_key(), Some(
                                std::collections::HashMap::from(
                                    [
                                        (
                                            "inference_time".to_raw_key(),
                                            LimitedHabValue::UnsignedInteger(elapsed_micros as _)
                                        ),
                                        (
                                            "batch_size".to_raw_key(),
                                            LimitedHabValue::UnsignedInteger(this_batch_size as _)
                                        )
                                    ]
                                )
                            )){
                                error!("Failed to log onnx inference time in onnx inference operator {} with error {:?}", self_id, e);
                            }
                        }
                    }

                let session_outputs = match run_result {
                    Ok(session_outputs) => {
                        // for each one and each tuple, we need to extract the relevant data
                        let mut outputs: SmallBatch<Batched<ndarray::ArrayD<f32>>> = SmallBatch::with_capacity(num_outputs);
                        for (output_name, output) in session_outputs {
                            let tensor = match output.try_extract_array() {
                                Ok(t) => t,
                                Err(e) => {
                                    let msg = format!("Onnx Inference Operator {}: failed to extract tensor from output named {}: {:?}", self_id, output_name, e);
                                    error!("{msg}");
                                    return (input_args, Err(anyhow::anyhow!(msg)));
                                }
                            };
                            if expected_rows== 1 {
                                // if we only have one row, then we can just return the output as is
                                outputs.push([tensor.to_owned()].into_iter().collect());
                                continue
                            }
                            outputs.push(
                                // for each output tensor, we need to split it into individual tensors for each tuple
                                (0..expected_rows).map(|i| {
                                    let slice = tensor.slice_axis(ndarray::Axis(0), ndarray::Slice::from(i..i+1));
                                    let owned = slice.to_owned();
                                    // ort::value::Tensor::from_array(owned).expect("we should be able to create a tensor from a slice of a tensor")
                                    owned
                                }).collect()
                            )
                        }

                        outputs
                    },
                    Err(e) => {
                        let msg = format!("Onnx Inference Operator {}: failed to run onnx session: {:?}", self_id, e);
                        error!("{msg}");
                        return (input_args, Err(anyhow::anyhow!(msg)));
                    }
                };
                (input_args, Ok(session_outputs))
            }).await;

            let (input_args, output_arrays) = match run_result {
                Ok((input_args, Ok(output_arrays))) => (input_args, output_arrays),
                Ok((input_args, Err(e))) => {
                    return_values(input_args, &self.args);
                    error!(
                        "Onnx Inference Operator {}: failed to run onnx session: {:?}",
                        self_id, e
                    );
                    break true;
                }
                Err(e) => {
                    // TODO: record shapes so that we can re-reconstitute the arrays and recover
                    error!("Onnx Inference Operator {}: !!!-- Unrecoverable Error --!!! failed to join when running onnx session in blocking task: {:?}", self_id, e);
                    break true;
                }
            };
            if output_arrays.len() != self.outputs.len() {
                return_values(input_args, &self.args);
                error!("Onnx Inference Operator {}: number of outputs from onnx session ({}) does not match number of expected outputs ({})", self_id, output_arrays.len(), self.outputs.len());
                break true;
            }

            let mut encountered_failure = false;

            'write_outputs: for (output, output_tensors) in
                self.outputs.iter().zip(output_arrays.into_iter())
            {
                match output {
                    OnnxValue::SplitParts {
                        shape_field,
                        buffer_field,
                        buffer: _buffer_lock,
                    } => {
                        // let mut buffer = buffer_lock.lock();
                        for (tuple, output_array) in tuples[batch_start_index..batch_end_index]
                            .iter_mut()
                            .zip(output_tensors)
                        {
                            // let output_array = output_array
                            let shape = output_array.shape().to_vec();
                            let backing = match output_array.into_raw_vec_and_offset() {
                                (backing, Some(0) | None) => backing,
                                (_backing, Some(offset)) => {
                                    warn!("Onnx Inference Operator {}: output array had non-zero offset ({}), which means we need to copy the data to a new buffer. this is inefficient and should be avoided", self_id, offset);
                                    // backing[offset..].to_vec()
                                    encountered_failure = true;
                                    break 'write_outputs;
                                }
                            };
                            tuple.insert(shape_field.clone(), HabValue::ShapeBuffer(shape));
                            let int_backing = bytemuck::cast_vec::<f32, i32>(backing);
                            tuple.insert(buffer_field.clone(), HabValue::IntBuffer(int_backing));
                        }
                    }
                }
            }

            // always return the input buffers
            return_values(input_args, &self.args);
            if encountered_failure {
                break true;
            }
            tuple_index += self.batch_size;

            fn return_values(
                input_args: Batched<ort::value::Tensor<f32>>,
                self_args: &[OnnxValue],
            ) {
                for (
                    OnnxValue::SplitParts {
                        buffer: buffer_lock,
                        ..
                    },
                    buffer,
                ) in self_args.iter().zip(input_args.into_iter())
                {
                    *buffer_lock.lock() = Some(buffer);
                }
            }
        };

        let num_to_output = tuples.len();
        if encountered_failure {
            // TODO: can this be more forgiving? we could try to output the ones that succeeded by marking where the first error was and nullifying everything after that
            // null out all outputs for all tuples
            error!("onnx inference operator {self_id} failed to process batch of tuples. nulling out outputs for all {num_to_output} tuples.");
            if let Some(t) = tuples.get(tuple_index) {
                error!("We received an error when we had processed up to input {tuple_index}, which has tuple_id {:?}", t.id());
            } else {
                error!("We received an error when we had processed up to input {tuple_index}, but there is no such input (we had {} inputs). Investigate how we got an error after we should have succeeded.", num_to_output);
            }
            for tuple in &mut tuples {
                self.null_outputs(tuple);
            }
        }
        let _ = parent.process_tuples(tuples, self_id, bolts).await;
        return OperatorOutput::Something(Some(num_to_output));
    }

    async fn process_tuples_old<'a>(
        &'a self,
        mut tuples: Vec<Tuple>,
        _source: usize,
        bolts: &'a [PhysicalOperator],
    ) -> OperatorOutput {
        if tuples.is_empty() {
            warn!(
                "onnx inference operator {} received no tuples to process",
                self.id
            );
            return_tuple_vec(tuples);
            return OperatorOutput::Nothing;
        }
        let Some(parent) = self.parent else {
            warn!("onnx inference operator {} has no parent", self.id);
            return OperatorOutput::Nothing;
        };
        let Some(parent) = bolts.get(parent) else {
            error!(
                "Index {} as parent for OnnxInferenceOperator {} is out of bounds",
                parent, self.id
            );
            return OperatorOutput::Nothing;
        };
        // let mut future_vec = Vec::with_capacity(tuples.len());
        let mut inference_future_captures: Vec<(_, _, _)> = Vec::with_capacity(tuples.len());
        let mut inference_futures = futures::stream::FuturesUnordered::new();
        let mut current_batch = Batched::with_capacity(self.batch_size);
        let mut tuples_in_error: crate::ArrayMap<usize, (), MAX_INLINE_BATCH_SIZE> =
            crate::ArrayMap::new();
        'next_tuple: for (input_index, tuple) in tuples.iter_mut().enumerate() {
            let mut this_tuple_inputs = vec![];
            for arg in &self.args {
                match arg {
                    OnnxValue::SplitParts {
                        shape_field,
                        buffer_field,
                        buffer: _,
                    } => {
                        let Some(shape) = tuple.get(shape_field) else {
                            // ("shape field not found");
                            error!("Onnx Inference Operator {}: shape field {:?} not found in tuple with id {}. Available fields are {:?}", self.id, shape_field, tuple.id(), tuple.keys().collect::<Vec<_>>());
                            tuples_in_error.insert(input_index, ());
                            continue 'next_tuple;
                        };
                        let Some(buffer) = tuple.get(buffer_field) else {
                            // ("shape field not found");
                            error!("Onnx Inference Operator {}: buffer field {:?} not found in tuple with id {}. Available fields are {:?}", self.id, shape_field, tuple.id(), tuple.keys().collect::<Vec<_>>());
                            tuples_in_error.insert(input_index, ());
                            continue 'next_tuple;
                        };
                        let Some(shape) = shape.as_shape_buffer() else {
                            warn!("onnx inference operator {}: shape field {:?} is not a shape buffer for tuple with id {}", self.id, shape_field, tuple.id());
                            tuples_in_error.insert(input_index, ());
                            continue 'next_tuple;
                        };
                        let Some(buffer) = buffer.as_int_buffer() else {
                            warn!("onnx inference operator {}: buffer field {:?} is not a int buffer for tuple with id {}", self.id, buffer_field, tuple.id());
                            tuples_in_error.insert(input_index, ());
                            continue 'next_tuple;
                        };
                        let buffer = bytemuck::cast_slice::<i32, f32>(buffer);
                        let arr = match ndarray::ArrayViewD::from_shape(shape, buffer) {
                            Ok(v) => v,
                            Err(e) => {
                                // ("failed to create ndarray from shape and buffer");
                                error!("Onnx Inference Operator {}: failed to create ndarray from shape and buffer: {:?} for tuple with id {}. Shape was {:?} and length of buffer was {}", self.id, e, tuple.id(), shape, buffer.len());
                                tuples_in_error.insert(input_index, ());
                                continue 'next_tuple;
                            }
                        };
                        this_tuple_inputs.push(arr);
                        // let array_val = match ort::value::Tensor::from_array(
                        //     arr.view().into_owned(),
                        // ) {
                        //     Ok(v) => v,
                        //     Err(e) => {
                        //         // ("failed to create tensor from ndarray");
                        //         error!("Onnx Inference Operator {}: failed to create tensor value from ndarray for tuple with id {}: {:?}", self.id, tuple.id(), e);
                        //         self.null_outputs(tuple);
                        //         continue 'next_tuple;
                        //     }
                        // };
                        // this_tuple_inputs.push(ort::session::input::SessionInputValue::Owned(
                        //     array_val.into_dyn(),
                        // ));
                    }
                }
            }
            if this_tuple_inputs.is_empty() {
                error!(
                    "onnx inference operator {}: no inputs found for tuple with id {}",
                    self.id,
                    tuple.id()
                );
                tuples_in_error.insert(tuple.id(), ());
                continue 'next_tuple;
            }
            let inputs_len = this_tuple_inputs.len();
            current_batch.push((input_index, tuple.id(), this_tuple_inputs));
            if current_batch.len() >= self.batch_size {
                // let mut dimension_aligned_map = crate::ArrayMap::<_
                // ort::value::Tensor::from;
                // ndarray::stack!
                // let mut this_argument_inputs = Batched::new();
                if let Err(err_indices) = self.populate_batch_captures(
                    &mut inference_future_captures,
                    &mut current_batch,
                    inputs_len,
                ) {
                    for i in err_indices {
                        tuples_in_error.insert(i, ());
                    }
                }
                current_batch.clear();
            }
        }
        if !current_batch.is_empty() {
            let (_, _, inputs) = &current_batch[0];
            let inputs_len = inputs.len();
            if let Err(err_indices) = self.populate_batch_captures(
                &mut inference_future_captures,
                &mut current_batch,
                inputs_len,
            ) {
                for i in err_indices {
                    tuples_in_error.insert(i, ());
                }
            }
        }
        // fininsh business with current batch
        drop(current_batch);
        // fill out null values for any tuples that had errors
        for (err_index, _) in tuples_in_error {
            let tuple = &mut tuples[err_index];
            let tuple_id = tuple.id();
            error!(
                "onnx inference operator {}: error for tuple with id {:?}. nulling outputs.",
                self.id, tuple_id
            );
            self.null_outputs(tuple);
        }

        let outputs_len = self.outputs.len();
        let self_id = self.id;
        for (tuple, tuple_ids, inputs) in inference_future_captures {
            let session = Arc::clone(&self.onnx_session);
            let this_batch_size = tuple_ids.len();
            let inference_future = async move {
                tokio::task::spawn_blocking(move ||{
                    let before = std::time::Instant::now();
                    let mut session_guard = session.lock();
                    let outputs = match session_guard.run(&*inputs){
                        Ok(outputs) => outputs,
                        Err(e) => {
                            error!("onnx inference operator {} failed to run inference for tuple ids {:?}: {:?}", self_id, &tuple_ids, e);
                            return None;
                        }
                    };

                    if outputs.len() != outputs_len {
                        error!("onnx inference operator {}: output values length {} does not match output fields length {} for inference involving tuple ids {:?}", self_id, outputs_len, outputs_len, tuple_ids);
                        return None;
                    }
                    let mut extracted_outputs = Vec::<Option<ndarray::ArrayD<f32>>>::with_capacity(outputs_len);
                    for (i, (ort_output_name, output) )in outputs.iter().enumerate() {
                        let output = match output.try_extract_tensor::<f32>()  {
                            // Ok(v) => v,
                            Ok((dims, v)) => match ndarray::ArrayViewD::from_shape(&smallvec::SmallVec::<[usize; 16]>::from_iter(dims.iter().map(|x| *x as usize))[..], v) {
                                Ok(v) => v,
                                Err(e) => {
                                    error!("onnx inference operator {}: failed to create ndarray from output value with index {} and ORT output name {:?} for tuple with id {:?}: {:?}", self_id, i, ort_output_name, tuple_ids, e);
                                    return None;
                                }
                            },
                            Err(e) => {
                                error!("onnx inference operator {}: failed to extract f32 tensor from output value with index {} and ORT output name {:?} for tuple with id {:?}: {:?}", self_id, i, ort_output_name, tuple_ids, e);
                                return None;
                            }
                        };
                        extracted_outputs.push(Some(output.to_owned()));
                    }
                    {
                        // drop the lock on the session as soon as possible
                        drop(outputs);
                        drop(session_guard);
                    }

                    let elapsed_micros = before.elapsed().as_micros();
                    debug!("onnx inference operator {}: inference for a batch of size {} with tuple ids {:?} took {} us", self_id, this_batch_size, &tuple_ids, elapsed_micros);
                    // log time in global logger
                    'log_onnx_inference_time: {
                        let mut location_buffer = *b"onnx_inference_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxx";
                        let write_start = "onnx_inference_".len();
                        let max_op_id_len = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".len();
                        let write_end = write_start + max_op_id_len;
                        use std::io::Write;
                        // write the op id
                        if let Err(e) = write!(location_buffer[write_start..write_end].as_mut(), "{}", self_id) {
                            error!("Failed to write to location buffer op id for onnx inference operator {} with error {:?}", self_id, e);
                            break 'log_onnx_inference_time;
                        }
                        let Ok(location_buffer) = std::str::from_utf8(&location_buffer) else {
                            error!("Failed to convert location buffer to string for onnx inference operator {}", self_id);
                            break 'log_onnx_inference_time;
                        };

                        for tid in tuple_ids.iter(){
                            if let Err(e) = crate::global_logger::log_data(*tid, location_buffer.to_raw_key(), Some(
                                std::collections::HashMap::from(
                                    [
                                        (
                                            "inference_time".to_raw_key(),
                                            LimitedHabValue::UnsignedInteger(elapsed_micros as _)
                                        ),
                                        (
                                            "batch_size".to_raw_key(),
                                            LimitedHabValue::UnsignedInteger(this_batch_size as _)
                                        )
                                    ]
                                )
                            )){
                                error!("Failed to log onnx inference time in onnx inference operator {} with error {:?}", self_id, e);
                            }
                        }
                    }

                    Some((tuple, extracted_outputs))
                }).await
            };

            inference_futures.push(inference_future)
        }

        use futures::stream::StreamExt;
        while let Some(execution) = inference_futures.next().await {
            let (input_indices, outputs) = match execution {
                Ok(Some(v)) => v,
                Ok(None) => {
                    warn!("onnx inference operator {} failed to run inference. See earlier error for details", self.id);
                    continue;
                }
                Err(e) => {
                    error!(
                        "onnx inference operator {} failed to execute inference future: {:?}",
                        self.id, e
                    );
                    continue;
                }
            };
            if input_indices.is_empty() {
                warn!(
                    "onnx inference operator {} got an inference result with no input indices",
                    self.id
                );
                continue;
            }
            if input_indices.len() == 1 {
                let input_index = input_indices[0];
                let tuple = &mut tuples[input_index];
                for (i, (output, output_fields)) in
                    outputs.into_iter().zip(self.outputs.iter()).enumerate()
                {
                    let Some(output_arr) = output else {
                        error!("onnx inference operator {}: failed to get output for tuple with id {} and input index {}", self.id, tuple.id(), input_index);
                        match output_fields {
                            OnnxValue::SplitParts {
                                shape_field,
                                buffer_field,
                                buffer: _,
                            } => {
                                tuple.insert(shape_field.clone(), HabValue::Null);
                                tuple.insert(buffer_field.clone(), HabValue::Null);
                            }
                        }
                        continue;
                    };
                    // do nothing, just wait for the futures to finish
                    let output_shape = output_arr.shape().to_vec();
                    let (output_buf, None | Some(0)) = output_arr.into_raw_vec_and_offset() else {
                        error!("onnx inference operator {}: failed to extract owned buffer from output value with index {} for tuple with id {}", self.id, i, tuple.id());
                        match output_fields {
                            OnnxValue::SplitParts {
                                shape_field,
                                buffer_field,
                                buffer: _,
                            } => {
                                tuple.insert(shape_field.clone(), HabValue::Null);
                                tuple.insert(buffer_field.clone(), HabValue::Null);
                            }
                        }
                        continue;
                    };

                    match output_fields {
                        OnnxValue::SplitParts {
                            shape_field,
                            buffer_field,
                            buffer: _,
                        } => {
                            tuple.insert(shape_field.clone(), HabValue::ShapeBuffer(output_shape));
                            tuple.insert(
                                buffer_field.clone(),
                                HabValue::IntBuffer(bytemuck::cast_vec(output_buf)),
                            );
                        }
                    }
                }
                continue;
            }

            // otherwise we have a batch of multiple inputs
            // we need to split them back out to the original tuples
            debug!(
                "onnx inference operator {}: processing batch output for input indices {:?}",
                self.id, &input_indices
            );
            for (i, (output, output_fields)) in
                outputs.into_iter().zip(self.outputs.iter()).enumerate()
            {
                debug!("onnx inference operator {}: processing output index {} for batch with input indices {:?}", self.id, i, &input_indices);
                let Some(output_arr) = output else {
                    let tuple_ids: Batched<usize> =
                        input_indices.iter().map(|&i| tuples[i].id()).collect();
                    error!("onnx inference operator {}: failed to get output for batch with tuple ids {:?} and input indices {:?}", self.id, tuple_ids, &input_indices);
                    match output_fields {
                        OnnxValue::SplitParts {
                            shape_field,
                            buffer_field,
                            buffer: _,
                        } => {
                            for &input_index in &input_indices {
                                let tuple = &mut tuples[input_index];
                                tuple.insert(shape_field.clone(), HabValue::Null);
                                tuple.insert(buffer_field.clone(), HabValue::Null);
                            }
                        }
                    }
                    continue;
                };
                debug!("onnx inference operator {}: got output array with shape {:?} for output index {} for batch with input indices {:?}", self.id, output_arr.shape(), i, &input_indices);
                for &input_index in &input_indices {
                    let tuple = &mut tuples[input_index];
                    let output_arr: ndarray::ArrayViewD<'_, f32> =
                        output_arr.index_axis(ndarray::Axis(0), input_index);
                    let output_arr: ndarray::ArrayD<f32> =
                        output_arr.into_owned().insert_axis(ndarray::Axis(0));

                    // do nothing, just wait for the futures to finish
                    let output_shape = output_arr.shape().to_vec();
                    let (output_buf, None | Some(0)) = output_arr.into_raw_vec_and_offset() else {
                        error!("onnx inference operator {}: failed to extract owned buffer from output value with index {} for tuple with id {}", self.id, i, tuple.id());
                        match output_fields {
                            OnnxValue::SplitParts {
                                shape_field,
                                buffer_field,
                                buffer: _,
                            } => {
                                tuple.insert(shape_field.clone(), HabValue::Null);
                                tuple.insert(buffer_field.clone(), HabValue::Null);
                            }
                        }
                        continue;
                    };

                    match output_fields {
                        OnnxValue::SplitParts {
                            shape_field,
                            buffer_field,
                            buffer: _,
                        } => {
                            tuple.insert(shape_field.clone(), HabValue::ShapeBuffer(output_shape));
                            tuple.insert(
                                buffer_field.clone(),
                                HabValue::IntBuffer(bytemuck::cast_vec(output_buf)),
                            );
                        }
                    }
                    debug!("onnx inference operator {}: populated output fields for tuple with id {} for output index {} for batch with input indices {:?}", self.id, tuple.id(), i, &input_indices);
                }
            }
        }
        drop(inference_futures);

        let num_outputs = tuples.len();
        parent.process_tuples(tuples, self.id, bolts).await;
        OperatorOutput::Something(Some(num_outputs))
    }
}

#[test]
fn onnx_test() -> Result<(), Box<dyn std::error::Error>> {
    use ort::session::{builder::GraphOptimizationLevel, Session};
    let _model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("yolov8m.onnx")?;

    Ok(())
}
