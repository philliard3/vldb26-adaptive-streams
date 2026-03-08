use std::{
    collections::HashMap,
    sync::{
        atomic::{self, AtomicUsize},
        LazyLock,
    },
    time::{Instant, SystemTime},
};

use crate::{caching::RawKey, HabString, HabValue};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};

pub struct GlobalTimerStore {
    inner_map: dashmap::DashMap<RawKey, TimingDetails>,
}

// we can reduce collisions around startup time by pre-allocated enough space that there should be enough buckets available
static GLOBAL_TIMER_STORE: LazyLock<GlobalTimerStore> = LazyLock::new(|| GlobalTimerStore {
    inner_map: dashmap::DashMap::with_capacity(crate::caching::DEFAULT_TUPLES),
});

struct TimingDetails {
    unix_times_ns: Vec<u128>,
    relative_times_ns: Vec<u128>,
    tuple_ids: Vec<usize>,
    errors: Vec<usize>,
    auxilliary_data: Option<HashMap<RawKey, LimitedHabValues>>,
}

pub struct DifferentTypeError;

/// a very limited version of what can be contained in a habvalue
/// this is meant only to be used for very efficient logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitedHabValues {
    Integer(Vec<i64>),
    UnsignedInteger(Vec<u64>),
    UnsignedLongLong(Vec<u128>),
    Float(Vec<f64>),
    // use with caution! this can get large and cause lots of overhead with allocations
    String(Vec<HabString>),
}

impl LimitedHabValues {
    pub fn clear(&mut self) {
        match self {
            LimitedHabValues::Integer(v) => {
                v.clear();
            }
            LimitedHabValues::UnsignedInteger(v) => {
                v.clear();
            }
            LimitedHabValues::UnsignedLongLong(v) => {
                v.clear();
            }
            LimitedHabValues::Float(v) => {
                v.clear();
            }
            LimitedHabValues::String(v) => {
                v.clear();
            }
        }
    }

    pub fn push_default(&mut self) {
        match self {
            LimitedHabValues::Integer(v) => {
                v.push(0);
            }
            LimitedHabValues::UnsignedInteger(v) => {
                v.push(0);
            }
            LimitedHabValues::UnsignedLongLong(v) => {
                v.push(0);
            }
            LimitedHabValues::Float(v) => {
                v.push(0.0);
            }
            LimitedHabValues::String(v) => {
                v.push(HabString::Borrowed(""));
            }
        }
    }

    pub fn push_value(&mut self, val: LimitedHabValue) -> Result<usize, DifferentTypeError> {
        match (self, val) {
            (LimitedHabValues::Integer(v), LimitedHabValue::Integer(val)) => {
                v.push(val);
                Ok(v.len())
            }
            (LimitedHabValues::UnsignedInteger(v), LimitedHabValue::UnsignedInteger(val)) => {
                v.push(val);
                Ok(v.len())
            }
            (LimitedHabValues::UnsignedLongLong(v), LimitedHabValue::UnsignedLongLong(val)) => {
                v.push(val);
                Ok(v.len())
            }
            (LimitedHabValues::Float(v), LimitedHabValue::Float(val)) => {
                v.push(val);
                Ok(v.len())
            }
            (LimitedHabValues::String(v), LimitedHabValue::String(val)) => {
                v.push(val);
                Ok(v.len())
            }
            _ => Err(DifferentTypeError),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LimitedHabValue {
    Integer(i64),
    UnsignedInteger(u64),
    UnsignedLongLong(u128),
    Float(f64),
    String(HabString),
}
impl From<LimitedHabValue> for HabValue {
    fn from(value: LimitedHabValue) -> Self {
        match value {
            LimitedHabValue::Integer(i) => HabValue::Integer(i as _),
            LimitedHabValue::UnsignedInteger(i) => HabValue::UnsignedLongLong(i as _),
            LimitedHabValue::UnsignedLongLong(l) => HabValue::UnsignedLongLong(l),
            LimitedHabValue::Float(f) => HabValue::from(f),
            LimitedHabValue::String(s) => HabValue::from(s),
        }
    }
}
impl From<i64> for LimitedHabValue {
    fn from(value: i64) -> Self {
        LimitedHabValue::Integer(value)
    }
}
impl From<u64> for LimitedHabValue {
    fn from(value: u64) -> Self {
        LimitedHabValue::UnsignedInteger(value)
    }
}
impl From<u128> for LimitedHabValue {
    fn from(value: u128) -> Self {
        LimitedHabValue::UnsignedLongLong(value)
    }
}
impl From<f64> for LimitedHabValue {
    fn from(value: f64) -> Self {
        LimitedHabValue::Float(value)
    }
}
impl From<HabString> for LimitedHabValue {
    fn from(value: HabString) -> Self {
        LimitedHabValue::String(value)
    }
}

// lazily initialized start time for the system
pub static SYSTEM_START: LazyLock<Instant> = LazyLock::new(|| {
    LazyLock::force(&SYSTEM_START_TIME_NS);
    Instant::now()
});
pub static SYSTEM_START_TIME_NS: LazyLock<u128> = LazyLock::new(|| {
    let ns = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_else(|e| {
            error!("Error getting system time at startup, defaulting to 0: {e:?}");
            0
        });
    info!("System start time (ns since epoch): {ns}");
    ns
});
static FLUSH_FREQUENCY: AtomicUsize = AtomicUsize::new(1000);
static LOGS_SINCE_FLUSH: AtomicUsize = AtomicUsize::new(0);

pub struct FlushHandle(());

impl Drop for FlushHandle {
    fn drop(&mut self) {
        flush();
    }
}

pub fn flush_handle() -> FlushHandle {
    FlushHandle(())
}

pub fn set_flush_frequency(frequency: usize) -> FlushHandle {
    FLUSH_FREQUENCY.store(frequency, atomic::Ordering::Relaxed);
    FlushHandle(())
}

pub fn flush_frequency() -> usize {
    FLUSH_FREQUENCY.load(atomic::Ordering::Relaxed)
}

static LOGGING_DIRECTORY: std::sync::Mutex<HabString> =
    std::sync::Mutex::new(HabString::Borrowed("./"));

pub fn set_logging_directory(directory: HabString) -> HabString {
    std::mem::replace(
        &mut LOGGING_DIRECTORY
            .lock()
            .expect("we should never poison the lock"),
        directory,
    )
}

pub fn logging_directory() -> HabString {
    LOGGING_DIRECTORY
        .lock()
        .inspect_err(|_| error!("we should never poison the lock"))
        .unwrap()
        .clone()
}

static TEST_NAME: std::sync::Mutex<HabString> =
    std::sync::Mutex::new(HabString::Borrowed("watershed_test"));

pub fn set_test_name(name: HabString) -> HabString {
    std::mem::replace(
        &mut TEST_NAME
            .lock()
            .inspect_err(|_| error!("we should never poison the name lock"))
            .unwrap(),
        name,
    )
}

pub fn test_name() -> HabString {
    TEST_NAME
        .lock()
        .inspect_err(|_| error!("we should never poison the name lock"))
        .unwrap()
        .clone()
}

#[derive(Debug)]
pub enum LogError {
    TimeWentBackwards,
    FieldTypeMismatch,
    MissingAuxData,
    MissingField(RawKey),
    UnexpectedData,
    UnexpectedField(RawKey),
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::TimeWentBackwards => write!(f, "Time went backwards"),
            LogError::FieldTypeMismatch => write!(f, "Field type mismatch"),
            LogError::MissingAuxData => write!(f, "Missing auxilliary data"),
            LogError::MissingField(key) => write!(f, "Missing field: {:?}", key),
            LogError::UnexpectedData => write!(f, "Unexpected data"),
            LogError::UnexpectedField(key) => write!(f, "Unexpected field: {:?}", key),
        }
    }
}

// pub struct NoAuxData;
// impl IntoIterator for NoAuxData {
//     type Item = (RawKey, LimitedHabValue);
//     type IntoIter = std::iter::Empty<(RawKey, LimitedHabValue)>;

//     fn into_iter(self) -> Self::IntoIter {
//         std::iter::empty()
//     }
// }
pub const NO_AUX_DATA: Option<[(RawKey, LimitedHabValue); 0]> = None;

// TODO: change this to take an iterator of pairs instead of a HashMap
// so that we can avoid the allocation
pub fn log_data(
    tuple_id: usize,
    log_location: RawKey,
    // aux_data: Option<std::collections::HashMap<RawKey, LimitedHabValue>>,
    // TODO: rewrite this to take an iterator of pairs instead of a hashmap, to avoid the allocation
    aux_data: Option<impl IntoIterator<Item = (RawKey, LimitedHabValue)>>,
) -> Result<(), Vec<LogError>> {
    let remaining_logs =
        flush_frequency().saturating_sub(LOGS_SINCE_FLUSH.fetch_add(1, atomic::Ordering::Relaxed));
    if remaining_logs == 0 {
        flush();
        LOGS_SINCE_FLUSH.fetch_add(1, atomic::Ordering::Relaxed);
    }

    let time_since_start_ns = SYSTEM_START.elapsed().as_nanos();
    let Ok(time) = SystemTime::now().duration_since(std::time::UNIX_EPOCH) else {
        error!("Error getting time since epoch");
        return Err(vec![LogError::TimeWentBackwards]);
    };
    let current_unix_time_ns = time.as_nanos();
    use dashmap::mapref::entry::Entry;
    match GLOBAL_TIMER_STORE.inner_map.entry(log_location) {
        Entry::Occupied(mut o) => {
            let mut error_list = Vec::new();
            let details = o.get_mut();
            match (&mut details.auxilliary_data, aux_data) {
                (None, None) => {
                    // nothing to do
                    trace!("(As expected), found no auxilliary data for tuple {tuple_id} at log location {log_location:?}");
                }
                (None, Some(_)) => {
                    error!(
                    "Auxilliary data provided for tuple {tuple_id} at log location {log_location:?} that doesn't have auxilliary data for previous tuples"
                    );
                    error_list.push(LogError::UnexpectedData);
                }
                (Some(existing_aux), None) => {
                    error!("No auxilliary data provided for tuple {tuple_id} at log location {log_location:?} that has auxilliary data. recording default values instead");
                    // fill with empty data
                    for (_field, list) in existing_aux {
                        list.push_default()
                    }
                    error_list.push(LogError::MissingAuxData);
                }
                (Some(existing_aux), Some(aux_data)) => {
                    // fill with all values, logging an error if they exist in one but not the other
                    const INLINE_AUX_FIELDS: usize = 16;
                    let mut aux_data_map: crate::ws_types::ArrayMap<
                        RawKey,
                        LimitedHabValue,
                        INLINE_AUX_FIELDS,
                    > = Default::default();
                    let mut aux_keys: crate::async_operators::Batched<RawKey> = Default::default();
                    for (k, v) in aux_data {
                        if !existing_aux.contains_key(&k) {
                            error!("Auxilliary data provided for tuple with id {tuple_id} at log location {log_location:?} that doesn't have the field {k:?}");
                            error_list.push(LogError::UnexpectedField(k.clone()));
                            continue;
                        }
                        aux_keys.push(k.clone());
                        if aux_data_map.insert(k, v).is_some() {
                            // TODO: duplicate key error
                        }
                    }
                    trace!("Logging tuple {tuple_id} at log location {log_location:?} with aux data keys: {aux_keys:?}");
                    for (k, v) in existing_aux.iter_mut() {
                        if aux_data_map.get(k).is_none() {
                            // this is an error, we are missing a field
                            error!("Auxilliary data missing for tuple with id {tuple_id} at log location {log_location:?} that should have the field {k:?}");
                            error_list.push(LogError::MissingField(k.clone()));
                            v.push_default();
                        }
                    }
                    for (field, aux_value) in aux_data_map {
                        let Some(list) = existing_aux.get_mut(&field) else {
                            // this should never happen because we checked above
                            error!("Auxilliary data missing for tuple with id {tuple_id}. it should have the field {field:?} because we already checked above");
                            continue;
                        };
                        match list.push_value(aux_value) {
                            Ok(pos) => {
                                trace!("Auxilliary data added at position {pos} for tuple with id {tuple_id}. it should have the field {field:?}");
                            }
                            Err(_) => {
                                error!("Auxilliary data type mismatch for tuple with id {tuple_id}. it should have the field {field:?}");
                                list.push_default();
                            }
                        }
                    }
                }
            }

            details.unix_times_ns.push(current_unix_time_ns);
            details.relative_times_ns.push(time_since_start_ns);
            details.tuple_ids.push(tuple_id);
            if error_list.is_empty() {
                return Ok(());
            }
            details.errors.push(tuple_id);
            Err(error_list)
        }
        Entry::Vacant(vacant_entry) => {
            let mut aux_keys: crate::async_operators::Batched<RawKey> = Default::default();
            let inner_aux = aux_data.map(|aux_data| {
                aux_data
                    .into_iter()
                    .map(|(k, v)| {
                        aux_keys.push(k.clone());
                        (
                            k,
                            match v {
                                LimitedHabValue::Integer(val) => {
                                    LimitedHabValues::Integer(vec![val])
                                }
                                LimitedHabValue::UnsignedInteger(val) => {
                                    LimitedHabValues::UnsignedInteger(vec![val])
                                }
                                LimitedHabValue::UnsignedLongLong(val) => {
                                    LimitedHabValues::UnsignedLongLong(vec![val])
                                }
                                LimitedHabValue::Float(val) => LimitedHabValues::Float(vec![val]),
                                LimitedHabValue::String(val) => LimitedHabValues::String(vec![val]),
                            },
                        )
                    })
                    .collect()
            });
            trace!("Logging tuple {tuple_id} at new log location {log_location:?} with aux data keys: {aux_keys:?}");
            vacant_entry.insert(TimingDetails {
                unix_times_ns: vec![current_unix_time_ns],
                relative_times_ns: vec![time_since_start_ns],
                tuple_ids: vec![tuple_id],
                errors: vec![],
                auxilliary_data: inner_aux,
            });
            Ok(())
        }
    }
}

static FLUSH_LOCK: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
thread_local! {
    static IS_FLUSHING: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSketchData {
    error_recording: Vec<RecordingError>,
    tuple_count: usize,
    error_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordingError {
    SketchError,
    CreateFileError(HabString),
    SerializeTimeError,
    SerializeAuxilliaryError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationLogData<'a> {
    #[serde(with = "serde_bytes")]
    unix_times_ns: &'a [u8],
    #[serde(with = "serde_bytes")]
    relative_times_ns: &'a [u8],
    #[serde(with = "serde_bytes")]
    tuple_ids: &'a [u8],
    #[serde(with = "serde_bytes")]
    errors: &'a [u8],
}

#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct LocationLogAuxData<'a> {
    auxilliary_data: &'a HashMap<RawKey, LimitedHabValues>,
}

pub fn flush() {
    if IS_FLUSHING.with(|f| f.get()) {
        let bt = backtrace::Backtrace::new();
        error!("Thread {:?} reentrant call to flush detected. Aborting flush to prevent stack overflow.\nBacktrace:\n{bt:?}\n", std::thread::current().id());
        return;
    }

    IS_FLUSHING.with(|f| f.set(true));
    let flush_lock = FLUSH_LOCK.try_lock();
    let mut current_iteration = match flush_lock {
        Err(std::sync::TryLockError::WouldBlock) => {
            warn!("Flush already in progress, skipping flush");
            IS_FLUSHING.with(|f| f.set(false));
            return; // another thread is already flushing, so we don't need to do anything
        }
        Ok(flush_lock) => flush_lock,
        Err(e) => {
            error!("Error getting flush lock: {:?}", e);
            IS_FLUSHING.with(|f| f.set(false));
            return;
        }
    };

    let iteration_count = *current_iteration;
    LOGS_SINCE_FLUSH.store(0, atomic::Ordering::Relaxed);
    let log_dir = logging_directory();
    let test_name = test_name();
    let Some(current_time_nanos) = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())
    else {
        error!("Error getting current time");
        IS_FLUSHING.with(|f| f.set(false));
        return;
    };
    let current_time_elapsed = SYSTEM_START.elapsed().as_nanos();
    info!(
        "Flushing logs for iteration {iteration_count} at unix time {current_time_nanos}, elapsed time {current_time_elapsed}"
    );
    let mut location_sketch_map = std::collections::HashMap::new();

    for mut entry in GLOBAL_TIMER_STORE.inner_map.iter_mut() {
        let log_location = entry.key().0.as_str();
        let timing_details = entry.value_mut();
        let tuple_count = timing_details.tuple_ids.len();
        let error_count = timing_details.errors.len();
        let mut write_errors = Vec::new();
        // let mut pathbuf = std::path::PathBuf::from(&log_dir);
        // pathbuf.clear();

        'serialize_times: {
            let location_log_path = format!("{log_dir}/{test_name}__location_{log_location}__iter_{iteration_count}__unix_time_{current_time_nanos}__times.bin.log");
            let log_data = LocationLogData {
                unix_times_ns: bytemuck::cast_slice(timing_details.unix_times_ns.as_slice()),
                relative_times_ns: bytemuck::cast_slice(
                    timing_details.relative_times_ns.as_slice(),
                ),
                tuple_ids: bytemuck::cast_slice(timing_details.tuple_ids.as_slice()),
                errors: bytemuck::cast_slice(timing_details.errors.as_slice()),
            };
            // binary log with msgpack
            let mut file = match std::fs::File::create(&location_log_path) {
                Ok(f) => f,
                Err(e) => {
                    error!("Error creating time log file {location_log_path:?}: {e:?}");
                    write_errors.push(RecordingError::CreateFileError(location_log_path.into()));
                    break 'serialize_times;
                }
            };

            let v = match rmp_serde::to_vec(&log_data) {
                Ok(v) => v,
                Err(e) => {
                    error!("Error serializing time log data: {:?}", e);
                    write_errors.push(RecordingError::SerializeTimeError);
                    break 'serialize_times;
                }
            };

            //write to file
            if let Err(e) = std::io::Write::write_all(&mut file, &v) {
                error!("Error writing time log data: {:?}", e);
                write_errors.push(RecordingError::SerializeTimeError);
                break 'serialize_times;
            }
        }

        'serialize_aux: {
            if let Some(aux) = &timing_details.auxilliary_data {
                let aux_log_path = format!("{log_dir}/{test_name}__location_{log_location}__iter_{iteration_count}__unix_time_{current_time_nanos}__aux.json.log");
                let aux_log_data = LocationLogAuxData {
                    auxilliary_data: aux,
                };
                // json log
                let file = match std::fs::File::create(&aux_log_path) {
                    Ok(f) => f,
                    Err(e) => {
                        error!("Error creating aux log file: {:?}", e);
                        write_errors.push(RecordingError::CreateFileError(aux_log_path.into()));
                        break 'serialize_aux;
                    }
                };

                // TODO: consider using a binary format like msgpack or bincode for aux data as well
                if let Err(e) = serde_json::to_writer(file, &aux_log_data) {
                    error!("Error writing aux log data: {:?}", e);
                    write_errors.push(RecordingError::SerializeAuxilliaryError);
                    break 'serialize_aux;
                }
                // if let Err(e) = rmp_serde::encode::write(&mut std::io::BufWriter::new(file), &aux_log_data) {
                //     error!("Error writing aux log data: {:?}", e);
                //     write_errors.push(RecordingError::SerializeAuxilliaryError);
                //     break 'serialize_aux;
                // }
            }
        }
        location_sketch_map.insert(
            log_location,
            LogSketchData {
                error_recording: write_errors,
                tuple_count,
                error_count,
            },
        );
        timing_details.unix_times_ns.clear();
        timing_details.relative_times_ns.clear();
        timing_details.tuple_ids.clear();
        timing_details.errors.clear();
        if let Some(aux) = &mut timing_details.auxilliary_data {
            for (_field, list) in aux.iter_mut() {
                list.clear();
            }
        }
    }

    // serialize the sketch data
    let sketch_log_path = format!(
        "{log_dir}/{test_name}__iter_{iteration_count}__unix_time_{current_time_nanos}__sketch.json.log"
    );
    match std::fs::File::create(&sketch_log_path) {
        Ok(file) => {
            if let Err(e) = serde_json::to_writer(file, &location_sketch_map) {
                error!(
                    "Error writing sketch log data file {sketch_log_path}: {:?}",
                    e
                );
            }
        }
        Err(e) => {
            error!("Error creating sketch log file: {:?}", e);
        }
    };

    *current_iteration += 1;
    IS_FLUSHING.with(|f| f.set(false));
}
