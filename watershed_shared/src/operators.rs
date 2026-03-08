#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyNone, PyString};

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dashmap::{DashMap, DashSet};

use crate::chroma_utils::DistanceMetric;
use crate::{HabString, HabValue, Tuple};

use crate::expression::ComputationExpression;
use crate::Operator;

use crate::Queue;

use pyo3::{prelude::*, IntoPyObjectExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as tokio_channel;

pub struct Select {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub pred: Box<dyn Send + Sync + Fn(&Tuple) -> bool>,
}

impl Operator for Select {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
pub struct Project {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub keep_list: Vec<HabString>,
}

impl Operator for Project {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

pub struct UdfBolt {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub process: Arc<dyn Send + Sync + Fn(Tuple) -> Vec<Tuple>>,
}

impl Operator for UdfBolt {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
pub struct UdfSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub produce: Box<dyn Send + Sync + Fn() -> Option<Vec<Tuple>>>,
}

impl Operator for UdfSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

#[derive(Debug)]
pub struct Join {
    pub id: usize,
    pub parent: Option<usize>,
    pub left: usize,
    pub right: usize,
    pub pred: fn(&Tuple, &Tuple) -> bool,
    pub join_info: JoinInner,
}

#[derive(Debug)]
pub enum JoinInner {
    InnerTable {
        build_data: DashMap<Vec<HabValue>, Vec<Tuple>>,
        fields: Vec<HabString>,
    },
    OuterTable {
        build_data: DashMap<Vec<HabValue>, Vec<Tuple>>,
        build_side_fields: DashSet<HabString>,
        fields: Vec<HabString>,
    },
    // We are hashing the tuples, which means that if a computation needs to be able to distinguish between them, it needs to make sure it provides a watermark field or something
    DoublePipeline {
        left_inputs: DashSet<Tuple>,
        right_inputs: DashSet<Tuple>,
        evict: fn(&DashSet<Tuple>, &DashSet<Tuple>) -> bool,
    },
}

impl Operator for Join {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

// TODO: build specific version of join that accepts boxes
// TODO: allow join to accept boxes
pub struct BuilderJoin {
    pub id: usize,
    pub left: usize,
    pub right: usize,
    pub parent: Option<usize>,
    pub left_inputs: DashSet<Tuple>,
    pub right_inputs: DashSet<Tuple>,
    // TODO: can we make this a function pointer? or just the expression inline? perhaps an enum of different dispatch types for evaluating function-like things?
    pub pred: Box<dyn Send + Sync + Fn(&Tuple) -> bool>,
    pub evict: Box<dyn Send + Sync + Fn(&DashSet<Tuple>, &DashSet<Tuple>)>,
}

pub struct ChannelRouter {
    pub id: usize,
    pub child: usize,
    pub parent_channels: Vec<crate::SyncPipe>,
    pub route: Mutex<Box<dyn Send + Sync + FnMut(Vec<Tuple>, &[crate::SyncPipe]) -> Option<usize>>>,
}

impl Operator for ChannelRouter {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, _id: usize) {
        unimplemented!("we don't want parents on the channel router")
        // self.parents.push(id);
        // TODO: we need a way to get parent channels to push
        // self.parent_channels.push(value)
    }
}
pub struct ChannelSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub input: crossbeam::channel::Receiver<Vec<Tuple>>,
    pub timeouts: AtomicUsize,
}

impl Operator for ChannelSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
pub struct Merge {
    pub id: usize,
    pub child: usize,
    pub parent_channel: crossbeam::channel::Sender<Vec<Tuple>>,
    pub on_merge_fn: Option<Box<dyn Send + Sync + Fn(&Tuple)>>,
}
impl Operator for Merge {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, _: usize) {
        unimplemented!("there's no reason to pass a parent to Merge, since it'll be sending that over the channel")
    }
}

pub struct MergeSpout {
    pub id: usize,
    pub parent: Option<usize>,
    pub input: crossbeam::channel::Receiver<Vec<Tuple>>,
    pub timeouts: AtomicUsize,
}

impl Operator for MergeSpout {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DummyBolt(pub usize);
impl Operator for DummyBolt {
    fn add_parent(&mut self, _: usize) {
        //
    }
    fn get_id(&self) -> usize {
        self.0
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
    pub exit_channel: crossbeam::channel::Sender<()>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct PythonRemoteDebug<'a> {
    id: usize,
    child: usize,
    parent: Option<usize>,
    script_name: &'a str,
}

impl std::fmt::Debug for PythonRemoteUdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let debug = PythonRemoteDebug {
            id: self.id,
            child: self.child,
            parent: self.parent,
            script_name: &self.script_name,
        };
        debug.fmt(f)
    }
}

// #[derive(Debug)]
pub struct AsyncPythonRemoteTaskState {
    pub port: u16,
    pub should_stop: Arc<AtomicBool>,
    pub runtime_handle: tokio::runtime::Handle,
    pub input_to_background_thread: tokio_channel::Sender<Vec<Tuple>>,
    pub input_from_main_thread: Option<tokio_channel::Receiver<Vec<Tuple>>>,
    pub output_to_main_thread: tokio_channel::Sender<Vec<Tuple>>,
    pub output_from_background_thread: Option<tokio_channel::Receiver<Vec<Tuple>>>,
    pub scripts_dir_path: HabString,
    pub script_name: HabString,
    pub pending_items: Arc<AtomicUsize>,
    // encode: Option<Box<dyn Sync + Send + Fn(usize, &Tuple) -> zeromq::ZmqMessage>>,
    pub encode: Option<EncoderFunction>,
    pub decode:
        Option<Box<dyn Sync + Send + Fn(zeromq::ZmqMessage, &DashMap<usize, Tuple>) -> Vec<Tuple>>>,
    pub shutdown: Option<Box<dyn Sync + Send + FnOnce() -> zeromq::ZmqMessage>>,
    // encode: Box<dyn Send + Sync + FnMut(usize, &Tuple) -> zeromq::ZmqMessage>,
    // decode: Box<dyn Send + Sync + FnMut(zeromq::ZmqMessage, DashMap<usize, Tuple>) -> Tuple>,
}

// TODO: change encode function to allow batching to cut down on the number of messages sent and time taken
pub enum EncoderFunction {
    Single(Box<dyn Sync + Send + Fn(usize, &Tuple) -> zeromq::ZmqMessage>),
    Batch(Box<dyn Sync + Send + Fn(&Vec<usize>, &Vec<Tuple>) -> zeromq::ZmqMessage>),
}

impl PythonRemoteUdf {
    pub(crate) fn start_background_task(&mut self) {
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
        let should_stop = Arc::clone(&state.should_stop);
        let op_id = self.id;

        let script_name = state.script_name.clone();
        let script_dir = state.scripts_dir_path.clone();

        let _background_task: tokio::task::JoinHandle<Result<(), Box<dyn Sync+Send+std::error::Error>>> = state.runtime_handle.spawn(async move {
            debug!("python remote udf operator {op_id} at the beginning of background task");
            use std::io::Write;
            use zeromq::{Socket, SocketSend, SocketRecv};
            let mut tuple_id_counter = 0;
            let tuple_map = DashMap::new();

            let mut pathbuf = std::path::PathBuf::from(&*script_dir);
            let background_task_script = "python_remote_worker.py";
            pathbuf.push(background_task_script);
            let background_task_script = pathbuf.to_str().expect("failed to convert path to string");
            let mut addr = *b"tcp://127.0.0.1:____";
            // print port padded to 4 digits
            let addr_len = addr.len();
            write!(&mut addr[addr_len-4..], "{:04}", port).expect("failed to format port");
            let addr = std::str::from_utf8(&addr[..]).expect("failed formatting of address");
            debug!("addr for python remote udf operator {op_id} is {:?}", addr);
            // start up python file client
            info!("Starting server for python remote udf operator {op_id}");
            let mut socket = zeromq::ReqSocket::new();
            // match socket.connect(addr).await {
            match socket.bind(addr).await {
                Ok(_endpoint) => {
                    info!("successfully connected to python side");
                },
                Err(e) => {
                    warn!("error connecting to start server for python remote udf operator {op_id}");
                    match e{
                    zeromq::ZmqError::Endpoint(_endpoint_error) => error!("error was Endpoint"),
                    zeromq::ZmqError::Network(_error) => error!("error was Network"),
                    zeromq::ZmqError::NoSuchBind(_endpoint) => error!("error was NoSuchBind"),
                    zeromq::ZmqError::Codec(_codec_error) => error!("error was Codec(codec_error"),
                    zeromq::ZmqError::Socket(_err) => error!("error was Socket"),
                    zeromq::ZmqError::BufferFull(_err) => error!("error was BufferFull"),
                    zeromq::ZmqError::ReturnToSender { reason: _, message: _ } => error!("error was ReturnToSender"),
                    zeromq::ZmqError::ReturnToSenderMultipart { reason: _, messages: _ } => error!("error was ReturnToSenderMultipart"),
                    zeromq::ZmqError::Task(_task_error) => error!("error was Task(task_error"),
                    zeromq::ZmqError::Other(_err) => error!("error was Other(_"),
                    zeromq::ZmqError::NoMessage => error!("error was NoMessage"),
                    zeromq::ZmqError::PeerIdentity => error!("error was PeerIdentity"),
                    zeromq::ZmqError::UnsupportedVersion(_err) => error!("error was UnsupportedVersion"),
                }
            }
            }
            let mut command = std::process::Command::new("python");
            let mut child = command
                .arg(background_task_script)
                .arg(&*script_name)
                .arg(addr)
                // .stdout(std::process::Stdio::piped())
                .spawn()
                .expect("unable to start python process");
            // let mut stdout = child.stdout.take().expect("unable to get stdout of child process");
            tokio::time::sleep(std::time::Duration::from_millis(1_000)).await;
            info!("starting loop");
            // tokio::time::sleep(Duration::from_millis(5)).await;

            // This loop waits for each input to receive its output
            // TODO: can we handle multiple requests asynchronously?
            //  when we are able to do that, the dashmap of ids should help us re-match the outputs to the inputs they were derived from
            'outer: loop {
                if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                    info!("should stop reached for python remote udf operator {op_id}");
                    break 'outer;
                }
                let receive_inputs = tokio::time::timeout(Duration::from_millis(1), input_from_main_thread.recv());
                // TODO: changel loop to push all updates in a batch
                match receive_inputs.await {
                    Ok(val) => {
                        let Some(inputs) = val else {
                            info!("Failed to receive inputs from main thread of remote python udf {op_id}, exiting");
                            break 'outer;
                        };
                        debug!("bg thread for python remote udf operator {op_id} got {} inputs from main", inputs.len());
                        match &encode_fn {
                            EncoderFunction::Single(encode_fn) =>{
                                for input in inputs {
                                    let msg = encode_fn(tuple_id_counter, &input);
                                    let time_created = input.get("time_created").expect("time_created field not found").as_unsigned_long_long().expect("time_created field is not an int");
                                    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_millis();
                                    let time_diff = now - time_created;
                                    debug!("time diff when sending is {}", time_diff);
                                    tuple_map.insert(tuple_id_counter, input);
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
                                        panic!("Failed to send message to Python worker {}", script_name);
                                    }
                                    debug!("python udf {op_id} sent one to python {}", script_name);
                                    pending_items.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                                    let send_outputs =   tokio::time::timeout(Duration::from_millis(1000), socket.recv());

                                    match send_outputs.await {
                                        Ok(outputs) => {
                                            let outputs = outputs.expect("Failed to receive outputs from Python");
                                            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_millis();
                                            let time_diff = now - time_created;
                                            debug!("time diff when data came back from {} is {}", &script_name,  time_diff);        
                                            let outputs = decode_fn(outputs, &tuple_map);
                                            let num_outputs = outputs.len();
                                            // TODO: log number of outputs
                                            debug!("bg thread message contained {num_outputs} inputs from python");
                                            pending_items.fetch_sub(outputs.len(), std::sync::atomic::Ordering::Relaxed);
                                            output_to_main_thread.send(outputs).await.expect("Failed to send outputs from Python back to main thread");
                                            debug!("sent {} outputs to main thread", num_outputs);
                                        }
                                        Err(_e) => {
                                            // no outputs received
                                            info!("no outputs received in operator {op_id} (timeout). exiting now");
                                        }
                                    }
                                    // pending_items.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                    // tuple_id_counter += 1;
                                    if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                                        info!("should stop reached");
                                        break 'outer;
                                    }
                                }
                            }
                            EncoderFunction::Batch(encode_fn) => {
                                let mut ids = vec![];
                                let mut earliest_time = u128::MAX;
                                for input in &inputs {

                                    let time_created = input.get("time_created").expect("time_created field not found").as_unsigned_long_long().expect("time_created field is not an int");
                                    if time_created < earliest_time {
                                        earliest_time = time_created;
                                    }

                                    ids.push(tuple_id_counter);
                                    tuple_id_counter += 1;
                                    pending_items.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                let msg = encode_fn(&ids, &inputs);
                                for (tuple_id, tuple) in ids.into_iter().zip(inputs.into_iter()) {
                                    tuple_map.insert(tuple_id, tuple);
                                }
                                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_millis();
                                let time_diff = now - earliest_time;
                                debug!("time diff when sending is {}", time_diff);
                                socket.send(msg).await?;
                                debug!("sent one to python");
                                let send_outputs =   tokio::time::timeout(Duration::from_millis(1000), socket.recv());
                                match send_outputs.await {
                                    Ok(outputs) => {
                                        let outputs = outputs.expect("Failed to receive outputs from Python");
                                        debug!("bg thread for remote python udf {op_id} got a message back from python");
                                        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("failed to get time since epoch").as_millis();
                                        let time_diff = now - earliest_time;
                                        debug!("time diff when data came back is {}", time_diff);        
                                        let outputs = decode_fn(outputs, &tuple_map);
                                        let num_outputs = outputs.len();
                                        // TODO: log number of outputs
                                        debug!("python remote udf {op_id} bg thread message contained {num_outputs} inputs from python");
                                        pending_items.fetch_sub(outputs.len(), std::sync::atomic::Ordering::Relaxed);
                                        output_to_main_thread.send(outputs).await.expect("Failed to send outputs from Python back to main thread");
                                        debug!("python remote udf {op_id} sent {} outputs to main thread", num_outputs);
                                    }
                                    Err(_e) => {
                                        // no outputs received
                                        info!("no outputs received (timeout). exiting now");
                                    }
                                }
                                // TODO: I don't see where should_stop is set to true?
                                if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                                    info!("should stop reached according to remote python udf");
                                }
                            }
                        }
                    }
                    Err(_e) => {
                        // no inputs received
                    },
                }
            }

            // TODO: send shutdown notice to child when finished
            // (how do we get here? do we get a condvar or something from the main thread?)
            socket.send(shutdown_fn()).await?;
            tokio::time::sleep(Duration::from_millis(500)).await;

            match child.try_wait() {
                Ok(Some(status)) => {
                    info!("child for remote python operator {op_id} exited with: {}", status);
                },
                Ok(None) => {
                    warn!("child process for remote python operator {op_id} was still running and we were unable to kill the process");
                    if let Err(e) = child.kill(){
                        error!("child process for remote python operator {op_id} was still running and we were unable to kill the process:\n{e:?}");
                        panic!("child process for remote python operator {op_id} was still running and we were unable to kill the process:\n{e:?}");
                    }
                },
                Err(e) => {
                    error!("error checking child status: {}", e);
                    child.kill().unwrap_or_else(|e2| panic!("error {e} while checking child process. task was unable to kill child process, with error {e2}"));
                }
            }

            debug!("remote python udf operator {op_id} finished background task. signalling to main thread");

            if let Err(e) = exit_channel.send(()){
                error!("failed to send exit message to main thread in remote python udf operator {op_id}: {e:?}");
            }

            Ok(())
        });
        // give python vm time to start up
        // std::thread::sleep(Duration::from_millis(1000));
        std::thread::sleep(Duration::from_millis(100));
    }
}

impl Operator for PythonRemoteUdf {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
    fn initialize(&mut self) {
        if !self.initialized {
            self.start_background_task();
            self.initialized = true;
            std::thread::sleep(Duration::from_secs(3));
        }
    }
}

pub struct PythonInlineUdf {
    pub id: usize,
    pub input: usize,
    pub parent: Option<usize>,
    pub script_name: HabString,
    pub scripts_dir_path: HabString,
    pub function_name: HabString,
    pub func: Py<PyAny>,
    pub encoder: PythonEncodingMethod,
    pub decoder: PythonDecodingMethod,
}

pub type InlineEncodeFn = Box<dyn Sync + Send + Fn(&Tuple) -> Option<Vec<Py<PyAny>>>>;
pub type InlineDecodeFn =
    Box<dyn Sync + Send + Fn(&Py<PyAny>, Tuple, &[HabString], &mut Vec<Tuple>) -> Option<usize>>;

pub enum PythonEncodingMethod {
    // no conversion
    PythonValues {
        fields: Vec<HabString>,
    },
    // default conversion
    HabValueToPyAny {
        fields: Vec<HabString>,
    },
    // TODO: native implementation for ndarrays to make it more efficient
    // NdArray {
    //     value_field: HabString,
    //     shape_field: HabString,
    // },
    // custom conversion
    CustomEncoder {
        func: InlineEncodeFn,
        fields: Vec<HabString>,
    },
}

fn habvalue_to_pyvalue<'py>(context: pyo3::Python<'py>, v: &HabValue) -> Option<Py<PyAny>> {
    match v {
        HabValue::Bool(b) => {
            let py_bool = PyBool::new(context, *b);
            Some(py_bool.into_py_any(context).ok()?)
        }
        HabValue::UnsignedLongLong(int) => {
            let py_int = PyInt::new(context, *int);
            Some(py_int.into())
        }
        HabValue::Integer(int) => Some(int.clone().into_py_any(context).ok()?),
        HabValue::String(hab_string_better) => {
            let py_string = PyString::new(context, hab_string_better.as_str());
            Some(py_string.into())
        }
        HabValue::Float(ordered_float) => {
            let py_float = PyFloat::new(context, ordered_float.0);
            Some(py_float.into())
        }
        HabValue::ByteBuffer(items) => {
            let py_bytes = PyBytes::new(context, items);
            Some(py_bytes.into())
        }
        HabValue::IntBuffer(items) => {
            let py_list = PyList::new(context, items).ok()?;
            Some(py_list.into())
        }
        HabValue::SharedArrayF32(items) => {
            let py_list =
                PyList::new(context, Vec::<f32>::from_iter(items.0.iter().map(|v| v.0))).ok()?;
            Some(py_list.into())
        }
        HabValue::SharedArrayU8(items) => {
            let py_list =
                PyList::new(context, Vec::<u8>::from_iter(items.0.iter().map(|v| *v))).ok()?;
            Some(py_list.into())
        }
        HabValue::ShapeBuffer(items) => {
            let py_list = PyList::new(context, items).ok()?;
            Some(py_list.into())
        }
        HabValue::List(hab_values) => {
            let py_list = PyList::new(
                context,
                hab_values
                    .iter()
                    .map(|item| habvalue_to_pyvalue(context, item)),
            )
            .ok()?;
            Some(py_list.into())
        }
        HabValue::PyObject(obj) => {
            // let py_object = obj.as_ref(py);
            let v = obj.0.as_ref().map(|p| p.clone_ref(context));
            if let Some(v) = v {
                Some(v)
            } else {
                warn!("Failed to clone PyObject in habvalue_to_pyvalue");
                PyNone::get(context).into_py_any(context).ok()
            }
        }
        HabValue::Null => None,
    }
}
impl PythonEncodingMethod {
    pub fn encode_tuple(&self, t: &Tuple) -> Option<Vec<Py<PyAny>>> {
        match self {
            PythonEncodingMethod::PythonValues { fields } => {
                encode_existing_python_values(t, fields)
            }
            PythonEncodingMethod::HabValueToPyAny { fields } => {
                encode_default_habvalue_to_pyany(t, fields)
            }
            PythonEncodingMethod::CustomEncoder { func, fields } => (func)(t),
        }
    }
}

fn encode_existing_python_values(t: &Tuple, fields: &[HabString]) -> Option<Vec<Py<PyAny>>> {
    let py_values = pyo3::Python::with_gil(|context| {
        let mut py_values = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(v) = t.get(field) else {
                error!("Inline python encoding: field {} not found in tuple", field);
                return None;
            };
            let Some(v) = v.as_pyobject() else {
                error!("Inline python encoding: field {} is not a PyObject", field);
                return None;
            };
            let v = match v {
                Some(v) => v,
                None => {
                    warn!("Inline python encoding: field {} is None", field);
                    // but we can still push a null
                    let Ok(none) = PyNone::get(context).into_py_any(context) else {
                        error!("Inline python encoding: failed to get PyNone");
                        return None;
                    };
                    py_values.push(none);
                    continue;
                }
            };
            let v = v.clone_ref(context);
            py_values.push(v);
        }
        Some(py_values)
    });
    py_values
}

fn encode_default_habvalue_to_pyany(t: &Tuple, fields: &[HabString]) -> Option<Vec<Py<PyAny>>> {
    let pyvalues = pyo3::Python::with_gil(|context| {
        let mut py_values = Vec::with_capacity(fields.len());
        for field in fields {
            let Some(v) = t.get(field) else {
                error!("Inline python encoding: field {} not found in tuple", field);
                return None;
            };
            let Some(v) = habvalue_to_pyvalue(context, v) else {
                error!(
                    "Inline python encoding: field {} failed to encode as a python object",
                    field
                );
                return None;
            };
            py_values.push(v);
        }
        Some(py_values)
    });
    pyvalues
}

// encoding an individual tensor using the provided fields
pub fn encode_f32_ndarray(t: &Tuple, buffer_field: &str, shape_field: &str) -> Option<Py<PyAny>> {
    let Some(buffer) = t.get(buffer_field) else {
        error!(
            "Inline python ndarray encoding: buffer field {} not found in tuple",
            buffer_field
        );
        return None;
    };
    let Some(shape) = t.get(shape_field) else {
        error!(
            "Inline python ndarray encoding: shape field {} not found in tuple",
            shape_field
        );
        return None;
    };
    let Some(buffer) = buffer.as_int_buffer() else {
        error!(
            "Inline python ndarray encoding: buffer field {} is not an IntBuffer",
            buffer_field
        );
        return None;
    };
    let reinterpreted_floats = buffer
        .iter()
        .map(|i| f32::from_bits(*i as u32))
        .collect::<Vec<f32>>();
    let Some(shape) = shape.as_shape_buffer() else {
        error!(
            "Inline python ndarray encoding: shape field {} is not an IntBuffer",
            shape_field
        );
        return None;
    };
    let dims = shape.to_vec();
    let py_array = pyo3::Python::with_gil(|context| {
        // get "numpy"
        let numpy = pyo3::types::PyModule::import(context, "numpy").ok()?;
        // call "numpy.array" with the reinterpreted_floats and the shape
        let py_array = match numpy.call_method1("array", (reinterpreted_floats,)) {
            Ok(arr) => arr,
            Err(e) => {
                error!(
                    "Inline python ndarray encoding: failed to call numpy.array: {:?}",
                    e
                );
                return None;
            }
        };
        // call "numpy.reshape" with the shape
        let py_array = match py_array.call_method1("reshape", (dims,)) {
            Ok(arr) => arr,
            Err(e) => {
                error!(
                    "Inline python ndarray encoding: failed to call numpy.reshape: {:?}",
                    e
                );
                return None;
            }
        };
        // convert to PyAny
        Some(py_array.unbind())
    });
    py_array
}

pub fn decode_f32_ndarray(py_value: &Py<PyAny>) -> Option<(Vec<f32>, Vec<usize>)> {
    pyo3::Python::with_gil(|context| {
        let numpy = pyo3::types::PyModule::import(context, "numpy").ok()?;
        let shape_array = match py_value.getattr(context, "shape") {
            Ok(arr) => arr,
            Err(e) => {
                error!(
                    "Inline python ndarray decoding: failed to call numpy.shape: {:?}",
                    e
                );
                return None;
            }
        };
        // extract vec of usizes
        let Ok(shape) = shape_array.extract::<Vec<usize>>(context) else {
            error!(
                "Inline python ndarray decoding: failed to extract shape as Vec<usize>: {:?}",
                shape_array
            );
            return None;
        };
        // extract vec of f32
        let Ok(buffer) = py_value
            // flatten and convert tolist
            .call_method1(context, "flatten", ())
            .and_then(|v| v.call_method1(context, "tolist", ()))
        else {
            error!(
                "Inline python ndarray decoding: failed to call flatten and tolist: {:?}",
                py_value
            );
            return None;
        };
        let Ok(buffer) = buffer.extract::<Vec<f32>>(context) else {
            error!(
                "Inline python ndarray decoding: failed to extract buffer as Vec<f32>: {:?}",
                py_value
            );
            return None;
        };
        Some((buffer, shape))
    })
}

#[cfg(test)]
// cfg for not debug
#[cfg(not(debug_assertions))]
#[test]
fn test_encode_decode_f32_batch_loop() {
    start_python_with_modules(&["numpy"]);
    let iters = 100u128;
    let mut times = vec![];
    pyo3::Python::with_gil(|_| {
        for _ in 0..iters {
            pyo3_speed_test_inner_loop(&mut times);
        }
    });
    let sum: u128 = times[1..].iter().sum();
    let avg = sum / (iters - 1);
    println!("Average elapsed time: {} micros", avg);
    assert!(
        avg < 500,
        "Average elapsed time is too long: {} micros",
        avg
    );
}

#[cfg(test)]
// cfg for not debug
#[cfg(not(debug_assertions))]
#[test]
fn test_encode_decode_f32_individual_loop() {
    start_python_with_modules(&["numpy"]);
    let iters = 100u128;
    let mut times = vec![];
    for _ in 0..iters {
        pyo3::Python::with_gil(|_| {
            pyo3_speed_test_inner_loop(&mut times);
        });
    }
    let sum: u128 = times[1..].iter().sum();
    let avg = sum / iters;
    println!("Average elapsed time: {} micros", avg);
    assert!(
        avg < 500,
        "Average elapsed time is too long: {} micros",
        avg
    );
}

// #[cfg(test)]
// // cfg for not debug
// #[cfg(not(debug_assertions))]
pub fn start_python_with_modules(module_names: &[impl AsRef<str>]) {
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        // some versions may need to add library path, but this should ideally be added by the user when running the program
        const SHOULD_TRY_SET_PATH: bool = false;
        if SHOULD_TRY_SET_PATH {
            'try_set_path: {
                // get $PYENV_ROOT environment variable
                let Ok(pyenv_root) = std::env::var("PYENV_ROOT") else {
                    error!("PYENV_ROOT environment variable not set. pyo3 will try to use the default location");
                    break 'try_set_path;
                };
                // add on /versions/3.10.15/lib/python3.10/site-packages/
                let mut pyenv_path = std::path::PathBuf::from(pyenv_root);
                pyenv_path.push("versions");
                pyenv_path.push("3.10.15");
                pyenv_path.push("lib");
                pyenv_path.push("python3.10");
                pyenv_path.push("site-packages");
                let Ok(sys) = py.import("sys") else {
                    error!("failed to import sys");
                    break 'try_set_path;
                };
                let Ok(exec_path) = sys.getattr("path") else {
                    error!("failed to get sys.path");
                    break 'try_set_path;
                };

                let Ok(_) = exec_path.call_method1("append", (pyenv_path.to_str().unwrap(),))
                else {
                    error!("failed to append to sys.path");
                    break 'try_set_path;
                };

                let Ok(_numpy) = py.import("numpy") else {
                    error!("failed to import numpy");
                    break 'try_set_path;
                };
            }
        }
        let _ = py
            .run(
                c"\
print('importing packages for start_python_with_modules')\
",
                None,
                None,
            )
            .expect("failed to run python");
        // import every module
        let mut error_count = 0;
        for module in module_names {
            let module = module.as_ref();
            if let Err(e) = py.import(module) {
                warn!("failed to import python module {module:?}: {e:?}");
                error_count += 1;
            }
        }
        if error_count > 0 {
            error!("failed to import {error_count} python modules");
        } else {
            debug!("successfully imported all python modules");
        }
    });
}

#[cfg(test)]
// cfg for not debug
#[cfg(not(debug_assertions))]
fn pyo3_speed_test_inner_loop(times: &mut Vec<u128>) {
    use std::time::Instant;
    let before = Instant::now();
    let buffer = vec![1.0, 2.0, 3.0, 4.0];
    let f32buf = buffer.clone();
    let buffer = bytemuck::cast_vec(buffer);
    let shape = vec![2, 2];
    let mut tuple = Tuple::default_internal();
    tuple.insert("buffer".into(), HabValue::IntBuffer(buffer));
    tuple.insert("shape".into(), HabValue::ShapeBuffer(shape.clone()));
    let py_value =
        encode_f32_ndarray(&tuple, "buffer", "shape").expect("failed to encode f32 ndarray");
    let (decoded_buffer, decoded_shape) =
        decode_f32_ndarray(&py_value).expect("failed to decode f32 ndarray");
    let elapsed = before.elapsed();
    assert_eq!(decoded_buffer, f32buf);
    assert_eq!(decoded_shape, shape);
    let micros = elapsed.as_micros();
    println!("Elapsed time: {} micros", micros);
    // assert!(micros < 500, "Elapsed time is too long: {} micros", micros);
    times.push(micros);
}

pub enum PythonDecodingMethod {
    // no conversion
    PythonValues {
        fields: Vec<HabString>,
    },
    // default conversion
    PyAnyToHabValues {
        fields: Vec<HabString>,
    },
    CustomDecoder {
        func: InlineDecodeFn,
        fields: Vec<HabString>,
    },
}
fn pyvalue_to_habvalue<'py>(context: pyo3::Python<'py>, py_value: &Py<PyAny>) -> Option<HabValue> {
    {
        let Ok(py_value) = py_value.downcast_bound(context) else {
            error!("Failed to downcast PyAny to PyObject");
            return None;
        };
        if py_value.is_instance_of::<pyo3::types::PyList>() {
            return Some(HabValue::List(
                py_value
                    .extract::<Vec<Py<PyAny>>>()
                    .ok()?
                    .into_iter()
                    .map(|item| pyvalue_to_habvalue(context, &item))
                    .collect::<Option<Vec<_>>>()?,
            ));
        }

        if py_value.is_instance_of::<pyo3::types::PyBool>() {
            return Some(HabValue::Bool(py_value.extract::<bool>().ok()?));
        }

        if py_value.is_instance_of::<pyo3::types::PyInt>() {
            if let Ok(i) = py_value.extract::<i32>() {
                return Some(HabValue::Integer(i));
            }
            if let Ok(i) = py_value.extract::<u128>() {
                return Some(HabValue::UnsignedLongLong(i));
            }
        }

        if py_value.is_instance_of::<pyo3::types::PyFloat>() {
            return Some(HabValue::Float(ordered_float::OrderedFloat(
                py_value.extract::<f64>().ok()?,
            )));
        }

        if py_value.is_instance_of::<pyo3::types::PyString>() {
            return Some(HabValue::String(py_value.extract::<String>().ok()?.into()));
        }

        if py_value.is_instance_of::<pyo3::types::PyBytes>() {
            return Some(HabValue::ByteBuffer(py_value.extract::<Vec<u8>>().ok()?));
        }
    }

    // If no other type matches, wrap it in a PyObject
    Some(HabValue::PyObject(crate::MyPyObject(Some(
        py_value.clone_ref(context),
    ))))
}

impl PythonDecodingMethod {
    pub fn decode_tuple(
        &self,
        py_value: &Py<PyAny>,
        mut original: Tuple,
        fields: &[HabString],
        out: &mut Vec<Tuple>,
    ) -> Option<usize> {
        let field_values = match self {
            PythonDecodingMethod::PythonValues { .. } => decode_existing_python_values(py_value),
            PythonDecodingMethod::PyAnyToHabValues { .. } => {
                decode_default_pyany_to_habvalues(py_value)
            }
            PythonDecodingMethod::CustomDecoder { func, .. } => {
                return (func)(py_value, original, fields, out)
            }
        }?;
        if fields.len() != field_values.len() {
            error!("Inline python decoding: field count mismatch");
            return None;
        }
        for (field, value) in fields.iter().zip(field_values) {
            original.insert(field.clone(), value);
        }
        out.push(original);
        Some(1)
    }
}

fn decode_existing_python_values(py_value: &Py<PyAny>) -> Option<Vec<HabValue>> {
    pyo3::Python::with_gil(|context| {
        if let Ok(py_list) = py_value.extract::<Vec<Py<PyAny>>>(context) {
            Some(
                py_list
                    .into_iter()
                    .map(|i| HabValue::PyObject(crate::MyPyObject(Some(i))))
                    .collect(),
            )
        } else {
            // just take whatever is already there
            Some(vec![HabValue::PyObject(crate::MyPyObject(Some(
                py_value.clone_ref(context),
            )))])
        }
    })
}

fn decode_default_pyany_to_habvalues(py_value: &Py<PyAny>) -> Option<Vec<HabValue>> {
    pyo3::Python::with_gil(|py| {
        let mut hab_values = vec![];
        let is_list = py_value
            .downcast_bound(py)
            .expect("unable to convert to bound?")
            .is_instance_of::<pyo3::types::PyList>();
        if is_list {
            let Ok(py_list) = py_value.extract::<Vec<Py<PyAny>>>(py) else {
                error!("failed to extract list after checking that it was a list");
                return None;
            };
            for item in py_list {
                let hab_value = pyvalue_to_habvalue(py, &item)?;
                hab_values.push(hab_value);
            }
        } else {
            let hab_value = pyvalue_to_habvalue(py, py_value)?;
            hab_values.push(hab_value);
        }
        Some(hab_values)
    })
}

#[cfg(test)]
#[test]
fn test_list_of_lists_decodes() {
    start_python_with_modules(&["numpy"]);
    let py_list = pyo3::Python::with_gil(|py| {
        let list_of_list_mod = pyo3::types::PyModule::from_code(
            py,
            c"
import numpy as np
a = np.array([[1, 2], [3, 4]])
a_list = a.tolist()
",
            c"file_name",
            c"test_list_of_lists",
        )
        .expect("failed to run python");

        let list_of_lists_val = list_of_list_mod
            .getattr("a_list")
            .expect("failed to get a_list from python module");
        let hab_values = decode_default_pyany_to_habvalues(
            &list_of_lists_val
                .into_py_any(py)
                .expect("could not make a reference counted copy of the list of lists"),
        )
        .expect("failed to decode list of lists");
        hab_values
    });
    let expected = vec![
        HabValue::List(vec![HabValue::Integer(1), HabValue::Integer(2)]),
        HabValue::List(vec![HabValue::Integer(3), HabValue::Integer(4)]),
    ];
    assert_eq!(py_list, expected);
}

#[cfg(test)]
#[test]
fn test_ndarray_decodes_as_pyany() {
    start_python_with_modules(&["numpy"]);
    let py_list = pyo3::Python::with_gil(|py| {
        let list_of_list_mod = pyo3::types::PyModule::from_code(
            py,
            c"
import numpy as np
a = np.array([[1, 2], [3, 4]])
a_list = a.tolist()
length = len(a_list)
together = (a_list, length)
",
            c"file_name",
            c"test_list_of_lists",
        )
        .expect("failed to run python");

        let array_val = list_of_list_mod
            .getattr("a")
            .expect("failed to get a_list from python module");

        let hab_values = decode_default_pyany_to_habvalues(
            &array_val
                .into_py_any(py)
                .expect("could not make a reference counted copy of the list of lists"),
        )
        .expect("failed to decode list of lists");
        hab_values
    });
    // let's print the types of all the stuff in the list
    for item in &py_list {
        println!("{:?}", item.get_type());
    }
    let expected_len = 1;
    let is_pyobject = match py_list[0] {
        HabValue::PyObject(_) => true,
        _ => false,
    };
    assert_eq!(expected_len, py_list.len());
    assert_eq!(is_pyobject, true);

    // let's check "together" to see if the second item is an int
    let together = pyo3::Python::with_gil(|py| {
        let list_of_list_mod = pyo3::types::PyModule::from_code(
            py,
            c"
import numpy as np
a = np.array([[1, 2], [3, 4]])
a_list = a.tolist()
length = len(a_list)
together = [a_list, length]
",
            c"file_name",
            c"test_list_of_lists",
        )
        .expect("failed to run python");

        let together_val = list_of_list_mod
            .getattr("together")
            .expect("failed to get a_list from python module");
        let hab_values = decode_default_pyany_to_habvalues(
            &together_val
                .into_py_any(py)
                .expect("could not make a reference counted copy of the list of lists"),
        )
        .expect("failed to decode list of lists");
        hab_values
    });
    let expected_len = 2;
    assert_eq!(expected_len, together.len());
    assert_eq!(together[1].get_type(), crate::HabValueType::Integer);
}

impl Operator for PythonInlineUdf {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
pub struct Union {
    pub id: usize,
    pub left: usize,
    pub right: usize,
    pub parent: Option<usize>,
}

impl Operator for Union {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

// document
/// GroupBy operator
/// This operator groups tuples by a set of fields and aggregates the state of the group
/// The state of the group is a set of fields that are updated by the aggregate function
/// The aggregate function is a function that takes the current state of the group and the next tuple
/// and returns the updated state of the group and a boolean indicating whether the group should be emitted
/// The group is emitted when the boolean is true
/// The group is evicted when the boolean is false
/// The group is evicted when the eviction policy is triggered
///
/// # Fields
/// * `id` - the unique identifier of the operator
/// * `child` - the child operator
/// * `parent` - the parent operator
/// * `fields` - the fields to group by
/// * `state` - the state of the group
pub struct GroupBy {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub fields: Vec<HabString>,
    pub state: DashMap<Vec<HabValue>, Queue<Tuple>>,
    pub aggregate: AggregationExpression,
}

pub struct AggregationResult {
    pub emit: Option<Vec<Tuple>>,
    pub is_finished: bool,
}

pub enum AggregationExpression {
    Udf(Box<dyn Send + Sync + Fn(&mut Queue<Tuple>) -> AggregationResult>),
    Componentized {
        derive_decision_key: ComputationExpression,
        should_emit: ComputationExpression,
        derive_eviction_key: ComputationExpression,
        should_evict: ComputationExpression,
    },
    Builtin {
        field: HabString,
        op: BuiltinAggregator,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinAggregator {
    Sum,
    Count,
    Min,
    Max,
    Avg,
}

impl Operator for GroupBy {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
pub struct DeriveValue {
    pub id: usize,
    pub child: usize,
    pub parent: Option<usize>,
    pub fields: Vec<HabString>,
    pub action: ComputationExpression,
    pub new_field_name: HabString,
}

impl Operator for DeriveValue {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}

#[cfg(feature = "bert")]
pub type SentenceEmbeddingsModel =
    rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

#[cfg(not(feature = "bert"))]
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DummySentenceEmbeddingsModel;
#[cfg(not(feature = "bert"))]
pub type SentenceEmbeddingsModel = DummySentenceEmbeddingsModel;
#[cfg(not(feature = "bert"))]
impl DummySentenceEmbeddingsModel {
    pub fn encode<S>(&self, inputs: &[S]) -> Result<Vec<Vec<f32>>, anyhow::Error>
    where
        S: AsRef<str> + Send + Sync,
    {
        error!("BERT feature not enabled, returning dummy embeddings");
        Ok(inputs.iter().map(|_| vec![0.0; 768]).collect())
    }
}

pub struct ChromaJoin {
    pub id: usize,
    pub parent: Option<usize>,
    pub index_stream: usize,
    pub lookup_stream: usize,
    pub metric: DistanceMetric,
    // pub embedding_method: Mutex<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>,
    pub embedding_method: Mutex<SentenceEmbeddingsModel>,
    pub client: Option<reqwest::Client>,
    pub collection_id: HabString,
    pub chroma_url: HabString,
    pub distance_threshold: f32,
    pub join_info: ChromaJoinKind,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "kind")]
pub enum ChromaJoinKind {
    #[serde(rename = "inner")]
    Inner {
        #[serde(rename = "topk")]
        query_n_matches: usize,
        keep_n_matches: usize,
    },
    #[serde(rename = "outer_right")]
    RightOuter {
        #[serde(rename = "topk")]
        query_n_matches: usize,
        keep_n_matches: usize,
    },
}

impl Operator for ChromaJoin {
    fn get_id(&self) -> usize {
        self.id
    }

    fn add_parent(&mut self, id: usize) {
        self.parent = Some(id)
    }
}
