#![allow(dead_code)]

use crate::basic_pooling::get_tuple;
use crate::HabString;
use dashmap::DashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::{atomic, atomic::AtomicUsize};
use std::time::Instant;

use crate::Tuple;

enum HabStringBacking<'a> {
    Owned(String),
    Borrowed(&'a str),
    Shared(Arc<str>),
    Reference(StringRef),
}

impl<'a> From<&'a str> for HabStringBacking<'a> {
    fn from(s: &'a str) -> Self {
        HabStringBacking::Borrowed(s)
    }
}
impl From<String> for HabStringBacking<'_> {
    fn from(s: String) -> Self {
        HabStringBacking::Owned(s)
    }
}
impl From<Arc<str>> for HabStringBacking<'_> {
    fn from(s: Arc<str>) -> Self {
        HabStringBacking::Shared(s)
    }
}
impl From<StringRef> for HabStringBacking<'_> {
    fn from(s: StringRef) -> Self {
        HabStringBacking::Reference(s)
    }
}

// impl std::ops::Deref for HabStringBacking<'_> {
//     type Target = str;

//     fn deref(&self) -> &str {
//         match self {
//             HabStringBacking::Owned(s) => s,
//             HabStringBacking::Borrowed(s) => s,
//             HabStringBacking::Shared(s) => s,
//             HabStringBacking::Reference(s) => s.runtime_pools.string_cache.get(&s.id).unwrap().inner.as_ref(),
//         }
//     }
// }

// impl AsRef<str> for HabStringBacking<'_> {
//     fn as_ref(&self) -> &str {
//         self
//     }
// }

struct RuntimePools {
    tuple_pool: crossbeam::queue::SegQueue<Tuple>,
    tuple_buffer_pool: crossbeam::queue::SegQueue<Vec<Tuple>>,

    string_lookup: DashMap<HabString, usize>,
    string_cache: DashMap<usize, TimedStringRefCount<HabString>>,
    global_string_id_counter: AtomicUsize,

    byte_buffer_cache: DashMap<usize, TimedStringRefCount<Vec<u8>>>,
    global_byte_buffer_id_counter: AtomicUsize,

    db_connections: DashMap<HabString, ConnectionKind>,
}

// report counts and sizes for all the pools instead of printing all values inside
impl Debug for RuntimePools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tuple_pool_size = self.tuple_pool.len();
        let tuple_buffer_pool_size = self.tuple_buffer_pool.len();
        let string_cache_size = self.string_cache.len();
        let byte_buffer_cache_size = self.byte_buffer_cache.len();
        let num_db_connections = self.db_connections.len();
        write!(f, "RuntimePools {{ tuple_pool: {}, tuple_buffer_pool: {}, string_cache: {}, byte_buffer_cache: {}, db_connnections: {} }}", tuple_pool_size, tuple_buffer_pool_size, string_cache_size, byte_buffer_cache_size, num_db_connections)
    }
}

struct StringRef {
    id: usize,
    runtime_pools: Arc<RuntimePools>,
}

struct StringRefDropGuard<'a>(&'a mut StringRef);

impl Drop for StringRef {
    fn drop(&mut self) {
        RuntimePools::drop_string_ref(StringRefDropGuard(self));
    }
}

struct ByteBufferRef {
    id: usize,
    runtime_pools: Arc<RuntimePools>,
}

struct ByteBufferRefDropGuard<'a>(&'a mut ByteBufferRef);

impl Drop for ByteBufferRef {
    fn drop(&mut self) {
        RuntimePools::drop_byte_buffer_ref(ByteBufferRefDropGuard(self));
    }
}

pub const DEFAULT_TUPLE_BUFFERS: usize = 1024;
pub const DEFAULT_TUPLES: usize = 1024;
pub const DEFAULT_STRINGS: usize = 1024;
pub const DEFAULT_BYTE_BUFFERS: usize = 1024;
impl RuntimePools {
    pub fn new() -> Self {
        let buf_queue = crossbeam::queue::SegQueue::new();
        for _ in 0..DEFAULT_TUPLE_BUFFERS {
            buf_queue.push(Vec::with_capacity(16));
        }

        let tuple_queue = crossbeam::queue::SegQueue::new();
        for _ in 0..DEFAULT_TUPLES {
            tuple_queue.push(get_tuple());
        }

        RuntimePools {
            tuple_pool: tuple_queue,
            tuple_buffer_pool: buf_queue,
            string_lookup: dashmap::DashMap::with_capacity(DEFAULT_STRINGS),
            string_cache: dashmap::DashMap::with_capacity(DEFAULT_STRINGS),
            global_string_id_counter: AtomicUsize::new(0),
            byte_buffer_cache: dashmap::DashMap::with_capacity(DEFAULT_BYTE_BUFFERS),
            global_byte_buffer_id_counter: AtomicUsize::new(0),
            db_connections: dashmap::DashMap::with_capacity(8),
        }
    }

    pub fn get_or_insert_string(self: Arc<Self>, string: HabString) -> StringRef {
        let id = self.string_lookup.get(&string).as_deref().cloned();
        if let Some(id) = id {
            self.string_cache.get_mut(&id).unwrap().ref_count += 1;
            // *self.string_cache.get_mut(&id).unwrap().ref_count += 1;
            return StringRef {
                id,
                runtime_pools: self,
            };
        }
        let id = self
            .global_string_id_counter
            .fetch_add(1, atomic::Ordering::SeqCst);
        self.string_lookup.insert(string.clone(), id);
        self.string_cache.insert(
            id,
            TimedStringRefCount {
                inner: string,
                ref_count: 1,
                last_access: Instant::now(),
            },
        );
        StringRef {
            id,
            runtime_pools: self,
        }
    }

    pub fn drop_string_ref(string_ref: StringRefDropGuard<'_>) {
        let string_ref = &string_ref.0;
        let this = string_ref.runtime_pools.clone();
        let id = string_ref.id;
        let mut entry = this
            .string_cache
            .get_mut(&id)
            .unwrap_or_else(|| unreachable!("string ref {} should be in cache", id));
        entry.ref_count -= 1;
        // possibly just decrease to zero and then cull later?
        if entry.ref_count == 0 {
            entry.last_access = Instant::now();
            this.string_cache.remove(&id);
        }
    }

    pub fn clean_string_cache(&self) {
        let mut to_remove = Vec::new();
        for val in self.string_cache.iter() {
            let id = val.key();
            let entry = val.value();
            if entry.ref_count == 0 && entry.last_access.elapsed().as_secs() > 60 {
                to_remove.push(*id);
            }
        }
        for id in to_remove {
            self.string_cache.remove(&id);
            self.string_lookup
                .remove(&self.string_cache.get(&id).unwrap().inner);
        }
    }

    pub fn create_byte_buffer_cached_data(self: Arc<Self>, data: Vec<u8>) -> ByteBufferRef {
        let id = self
            .global_byte_buffer_id_counter
            .fetch_add(1, atomic::Ordering::SeqCst);
        self.byte_buffer_cache.insert(
            id,
            TimedStringRefCount {
                inner: data,
                ref_count: 1,
                last_access: Instant::now(),
            },
        );
        ByteBufferRef {
            id,
            runtime_pools: self.clone(),
        }
    }

    pub fn drop_byte_buffer_ref(byte_buffer_ref: ByteBufferRefDropGuard<'_>) {
        let this = byte_buffer_ref.0.runtime_pools.clone();
        let id = byte_buffer_ref.0.id;
        let mut entry = this
            .byte_buffer_cache
            .get_mut(&id)
            .unwrap_or_else(|| unreachable!("byte buffer ref {} should be in cache", id));
        entry.ref_count -= 1;
        if entry.ref_count == 0 {
            entry.last_access = Instant::now();
            this.byte_buffer_cache.remove(&id);
        }
    }

    pub fn get_or_create_tuple(&self) -> Tuple {
        self.tuple_pool
            .pop()
            .unwrap_or_else(Tuple::default_internal)
    }

    pub fn get_or_create_tuple_buffer(&self) -> Vec<Tuple> {
        self.tuple_buffer_pool.pop().unwrap_or_default()
    }

    pub fn return_tuple(&self, mut tuple: Tuple) {
        tuple.clear();
        self.tuple_pool.push(tuple);
    }

    pub fn return_tuple_buffer(&self, mut buffer: Vec<Tuple>) {
        for t in buffer.drain(..) {
            self.return_tuple(t);
        }
        self.tuple_buffer_pool.push(buffer);
    }
}

enum ConnectionKind {
    Postgres,
    Mysql,
    Sqlite,
    Mongodb,
    Chroma(chromadb::ChromaClient),
}
impl Debug for ConnectionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionKind::Postgres => write!(f, "Postgres"),
            ConnectionKind::Mysql => write!(f, "Mysql"),
            ConnectionKind::Sqlite => write!(f, "Sqlite"),
            ConnectionKind::Mongodb => write!(f, "Mongodb"),
            ConnectionKind::Chroma(_) => write!(f, "Chroma"),
        }
    }
}
impl Hash for ConnectionKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ConnectionKind::Postgres => "Postgres".hash(state),
            ConnectionKind::Mysql => "Mysql".hash(state),
            ConnectionKind::Sqlite => "Sqlite".hash(state),
            ConnectionKind::Mongodb => "Mongodb".hash(state),
            ConnectionKind::Chroma(_) => "Chroma".hash(state),
        }
    }
}
impl PartialEq for ConnectionKind {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (ConnectionKind::Postgres, ConnectionKind::Postgres)
                | (ConnectionKind::Mysql, ConnectionKind::Mysql)
                | (ConnectionKind::Sqlite, ConnectionKind::Sqlite)
                | (ConnectionKind::Mongodb, ConnectionKind::Mongodb)
                | (ConnectionKind::Chroma(_), ConnectionKind::Chroma(_))
        )
    }
}
impl Eq for ConnectionKind {}

struct TimedStringRefCount<T> {
    inner: T,
    ref_count: usize,
    last_access: Instant,
}

#[derive(Clone)]
pub enum HabStringBetter<'a> {
    Owned(String),
    Borrowed(&'a str),
    InternedKey(symbol_table::GlobalSymbol),
}

impl<'a> From<&'a str> for HabStringBetter<'a> {
    fn from(s: &'a str) -> Self {
        HabStringBetter::Borrowed(s)
    }
}

impl std::ops::Deref for HabStringBetter<'_> {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            HabStringBetter::Owned(s) => s,
            HabStringBetter::Borrowed(s) => s,
            HabStringBetter::InternedKey(s) => s.as_str(),
        }
    }
}

impl AsRef<str> for HabStringBetter<'_> {
    fn as_ref(&self) -> &str {
        self
    }
}

impl PartialOrd for HabStringBetter<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        <str as PartialOrd<str>>::partial_cmp(self, other)
    }
}

impl PartialOrd<str> for HabStringBetter<'_> {
    fn partial_cmp(&self, other: &str) -> Option<std::cmp::Ordering> {
        <str as PartialOrd<str>>::partial_cmp(self, other)
    }
}

impl PartialOrd<HabStringBetter<'_>> for str {
    fn partial_cmp(&self, other: &HabStringBetter<'_>) -> Option<std::cmp::Ordering> {
        <str as PartialOrd<str>>::partial_cmp(&self, &other)
    }
}

impl Ord for HabStringBetter<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        <str as Ord>::cmp(self, other)
    }
}

impl PartialEq for HabStringBetter<'_> {
    fn eq(&self, other: &Self) -> bool {
        <str as PartialEq<str>>::eq(&self, &other)
    }
}

impl PartialEq<str> for HabStringBetter<'_> {
    fn eq(&self, other: &str) -> bool {
        <str as PartialEq<str>>::eq(&self, &other)
    }
}

impl PartialEq<HabStringBetter<'_>> for str {
    fn eq(&self, other: &HabStringBetter<'_>) -> bool {
        <str as PartialEq<str>>::eq(&self, &other)
    }
}

impl Eq for HabStringBetter<'_> {}

impl Hash for HabStringBetter<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl Debug for HabStringBetter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_ref())
    }
}

impl std::borrow::Borrow<str> for HabStringBetter<'_> {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for HabStringBetter<'_> {
    fn from(s: String) -> Self {
        HabStringBetter::Owned(s)
    }
}

impl From<symbol_table::GlobalSymbol> for HabStringBetter<'_> {
    fn from(s: symbol_table::GlobalSymbol) -> Self {
        HabStringBetter::InternedKey(s)
    }
}

impl From<HabStringBetter<'_>> for String {
    fn from(s: HabStringBetter) -> Self {
        match s {
            HabStringBetter::Owned(s) => s,
            HabStringBetter::Borrowed(s) => s.to_string(),
            HabStringBetter::InternedKey(s) => s.to_string(),
        }
    }
}

impl std::fmt::Display for HabStringBetter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl HabStringBetter<'_> {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl serde::Serialize for HabStringBetter<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for HabStringBetter<'static> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(HabStringBetter::Owned(s))
    }
}

pub trait StrToKey {
    fn to_key(self) -> HabStringBetter<'static>;
    fn to_raw_key(self) -> RawKey;
}
impl StrToKey for &str {
    fn to_key(self) -> HabStringBetter<'static> {
        HabStringBetter::InternedKey(symbol_table::GlobalSymbol::new(self))
    }

    fn to_raw_key(self) -> RawKey {
        RawKey::new(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct RawKey(pub(crate) symbol_table::GlobalSymbol);

impl RawKey {
    pub fn new(s: &str) -> Self {
        RawKey(symbol_table::GlobalSymbol::new(s))
    }
}

impl std::fmt::Display for RawKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl<'a> From<RawKey> for HabStringBetter<'a> {
    fn from(s: RawKey) -> Self {
        HabStringBetter::InternedKey(s.0)
    }
}
