#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

// filter on feature
// #[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{de, Deserialize, Deserializer, Serialize};

use std::borrow::Cow;
use std::ops::{Deref, DerefMut};
use std::{collections::BTreeMap, fmt::Debug};

use std::hash::Hash;

// TODO: is there something better we can do than use this canonical float version?
use ordered_float::OrderedFloat;

pub(crate) type Queue<T> = crate::devec::DeVec<T>;
use crate::basic_pooling::return_tuple;
pub use crate::scheduler::{AsyncPipe, SyncPipe};

// #[cfg(feature = "pyo3")]
#[derive(Debug)]
pub struct MyPyObject(pub Option<Py<PyAny>>);

impl Clone for MyPyObject {
    fn clone(&self) -> Self {
        // error!("Cloning PyObject is not supported. please manually clone iwth a python context to get a cloned reference counted object");
        // MyPyObject(None)
        match &self.0 {
            Some(v) => pyo3::Python::with_gil(|py| {
                let v = v.clone_ref(py);
                MyPyObject(Some(v))
            }),
            None => Self(None),
        }
    }
}

// #[cfg(feature = "pyo3")]
impl<'de> Deserialize<'de> for MyPyObject {
    // Required method
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        warn!("Deserializing PyObject is not supported. please convert to native habitat values or encode/decode it within python");
        Ok(Self(None))
    }
}

// #[cfg(feature = "pyo3")]
impl Hash for MyPyObject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if let Some(v) = &self.0 {
            v.as_ptr().hash(state)
        }
    }
}

// #[cfg(feature = "pyo3")]
impl PartialEq for MyPyObject {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(a), Some(b)) => std::ptr::eq(a.as_ptr(), b.as_ptr()),
            (None, None) => true,
            _ => false,
        }
    }
}

// #[cfg(feature = "pyo3")]
impl Eq for MyPyObject {}

// #[cfg(feature = "pyo3")]
impl Ord for MyPyObject {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (&self.0, &other.0) {
            (Some(a), Some(b)) => a.as_ptr().cmp(&b.as_ptr()),
            (None, None) => std::cmp::Ordering::Equal,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
        }
    }
}

// #[cfg(feature = "pyo3")]
impl PartialOrd for MyPyObject {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for SharedF32Array {}
impl PartialEq for SharedF32Array {
    fn eq(&self, other: &Self) -> bool {
        self.0.shape() == other.0.shape() && self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}
impl Hash for SharedF32Array {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.shape().hash(state);
        for v in self.0.iter() {
            v.hash(state);
        }
    }
}
impl Ord for SharedF32Array {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.shape().cmp(other.0.shape()) {
            std::cmp::Ordering::Equal => {
                for (a, b) in self.0.iter().zip(other.0.iter()) {
                    match a.cmp(b) {
                        std::cmp::Ordering::Equal => continue,
                        non_eq => return non_eq,
                    }
                }
                std::cmp::Ordering::Equal
            }
            non_eq => non_eq,
        }
    }
}

impl PartialOrd for SharedF32Array {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ArrayF32SerializationHelper {
    shape: smallvec::SmallVec<[usize; 4]>,
    strides: smallvec::SmallVec<[usize; 4]>,
}
impl Serialize for SharedF32Array {
    fn serialize<S: serde::Serializer>(
        self: &SharedF32Array,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shape = self.0.shape().iter().copied().collect();
        let strides = self.0.strides().iter().map(|&v| v as _).collect();
        Serialize::serialize(&ArrayF32SerializationHelper { shape, strides }, serializer)
    }
}

impl<'de> Deserialize<'de> for SharedF32Array {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use ndarray::{Dim, Shape, ShapeBuilder};
        let data: ArrayF32SerializationHelper = Deserialize::deserialize(deserializer)?;
        let shape: Dim<ndarray::IxDynImpl> = Dim(data.shape.as_slice());
        let shape_strides = shape.strides(Dim(data.strides.as_slice()));
        let arr = ndarray::ArcArray::<f32, ndarray::IxDyn>::zeros(shape_strides.raw_dim().clone());
        let array_ordered = unsafe { std::mem::transmute(arr) };
        Ok(array_ordered)
    }
}

impl Ord for SharedU8Array {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.shape().cmp(other.0.shape()) {
            std::cmp::Ordering::Equal => {
                for (a, b) in self.0.iter().zip(other.0.iter()) {
                    match a.cmp(b) {
                        std::cmp::Ordering::Equal => continue,
                        non_eq => return non_eq,
                    }
                }
                std::cmp::Ordering::Equal
            }
            non_eq => non_eq,
        }
    }
}
impl PartialOrd for SharedU8Array {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ArrayU8SerializationHelper {
    shape: smallvec::SmallVec<[usize; 4]>,
    strides: smallvec::SmallVec<[usize; 4]>,
}
impl Serialize for SharedU8Array {
    fn serialize<S: serde::Serializer>(
        self: &SharedU8Array,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shape = self.0.shape().iter().copied().collect();
        let strides = self.0.strides().iter().map(|&v| v as _).collect();
        Serialize::serialize(&ArrayU8SerializationHelper { shape, strides }, serializer)
    }
}
impl<'de> Deserialize<'de> for SharedU8Array {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use ndarray::{Dim, Shape, ShapeBuilder};
        let data: ArrayU8SerializationHelper = Deserialize::deserialize(deserializer)?;
        let shape: Dim<ndarray::IxDynImpl> = Dim(data.shape.as_slice());
        let shape_strides = shape.strides(Dim(data.strides.as_slice()));
        let arr = ndarray::ArcArray::<u8, ndarray::IxDyn>::zeros(shape_strides.raw_dim().clone());
        Ok(SharedU8Array(arr))
    }
}

fn serialize_ordered_float<S>(f: &OrderedFloat<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    f.0.serialize(serializer)
}

fn deserialize_ordered_float<'de, D>(deserializer: D) -> Result<OrderedFloat<f64>, D::Error>
where
    D: de::Deserializer<'de>,
{
    let f = f64::deserialize(deserializer)?;
    Ok(OrderedFloat(f))
}

// byte buffers should be serialized as debug string, and they should alwaays cause an error when deserialized this way
pub fn serialize_byte_buffer<S>(b: &[u8], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&format!("ByteBuffer({:?} items)", b.len()))
}
pub fn deserialize_byte_buffer<'de, D>(_deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Err(de::Error::custom(
        "Cannot deserialize byte buffer that was serialized in the debug format",
    ))
}
// same for int buffers
pub fn serialize_int_buffer<S>(b: &[i32], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&format!("IntBuffer({:?} items)", b.len()))
}
pub fn deserialize_int_buffer<'de, D>(_deserializer: D) -> Result<Vec<i32>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Err(de::Error::custom(
        "Cannot deserialize int buffer that was serialized in the debug format",
    ))
}

#[repr(transparent)]
#[derive(Clone)]
pub struct SharedF32Array(pub ArcArrayD<OrderedFloat<f32>>);

#[repr(transparent)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SharedU8Array(pub ArcArrayD<u8>);

pub type ArcArrayD<A> = ndarray::ArcArray<A, ndarray::IxDyn>;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum HabValue {
    Bool(bool),
    UnsignedLongLong(u128),
    Integer(i32),
    String(HabString),
    #[serde(serialize_with = "serialize_ordered_float")]
    #[serde(deserialize_with = "deserialize_ordered_float")]
    Float(OrderedFloat<f64>),
    // TODO: implement sharing for byte buffers so we can avoid copying data that points to the same thing
    #[serde(serialize_with = "serialize_byte_buffer")]
    #[serde(deserialize_with = "deserialize_byte_buffer")]
    ByteBuffer(Vec<u8>),
    #[serde(serialize_with = "serialize_int_buffer")]
    #[serde(deserialize_with = "deserialize_int_buffer")]
    IntBuffer(Vec<i32>),
    // shape buffer is fine though, since it's just a small list of usize
    ShapeBuffer(Vec<usize>),

    // #[serde(serialize_with = "serialize_shared_array_f32")]
    // #[serde(deserialize_with = "deserialize_shared_array_f32")]
    SharedArrayF32(SharedF32Array),
    SharedArrayU8(SharedU8Array),

    List(Vec<HabValue>),
    // #[cfg(feature = "pyo3")]
    #[serde(serialize_with = "serialize_pyobject")]
    PyObject(MyPyObject),

    Null,
}

fn serialize_pyobject<S>(_obj: &MyPyObject, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    error!("Cannot serialize PyObject");
    // TODO: should this just be an Err?
    serializer.serialize_str("PyObject")
}

impl HabValue {
    pub const HAB_NULL: HabValue = HabValue::Null;
    pub const HAB_NULL_REF: &'static HabValue = &HabValue::Null;
    pub fn is_null(&self) -> bool {
        matches!(self, HabValue::Null)
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HabValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn into_bool(&self) -> Option<bool> {
        match self {
            HabValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_unsigned_long_long(&self) -> Option<u128> {
        match self {
            HabValue::UnsignedLongLong(b) => Some(*b),
            _ => None,
        }
    }
    pub fn into_unsigned_long_long(&self) -> Option<u128> {
        match self {
            HabValue::UnsignedLongLong(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_integer(&self) -> Option<i32> {
        match self {
            HabValue::Integer(b) => Some(*b),
            _ => None,
        }
    }
    pub fn into_integer(&self) -> Option<i32> {
        match self {
            HabValue::Integer(b) => Some(*b),
            _ => None,
        }
    }

    pub fn into_byte_buffer(&self) -> Option<Vec<u8>> {
        match self {
            HabValue::ByteBuffer(b) => Some(b.clone()),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&HabString> {
        match self {
            HabValue::String(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<OrderedFloat<f64>> {
        match self {
            HabValue::Float(b) => Some(*b),
            _ => None,
        }
    }
    pub fn into_float(&self) -> Option<OrderedFloat<f64>> {
        match self {
            HabValue::Float(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_byte_buffer(&self) -> Option<impl '_ + AsRef<[u8]>> {
        match self {
            HabValue::ByteBuffer(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_int_buffer(&self) -> Option<&[i32]> {
        match self {
            HabValue::IntBuffer(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_shared_array_f32(&self) -> Option<&ArcArrayD<f32>> {
        match self {
            HabValue::SharedArrayF32(b) => Some({
                let b: &ArcArrayD<OrderedFloat<f32>> = &b.0;
                // SAFETY: we know that SharedF32Array is repr(transparent) over ArcArrayD<OrderedFloat<f32>>,
                unsafe { std::mem::transmute(b) }
            }),
            _ => None,
        }
    }

    pub fn as_array_view_f32(&self) -> Option<ndarray::ArrayViewD<'_, f32>> {
        let arr = self.as_shared_array_f32()?;
        Some(arr.view())
    }

    pub fn as_shared_array_u8(&self) -> Option<&ArcArrayD<u8>> {
        match self {
            HabValue::SharedArrayU8(b) => Some(&b.0),
            _ => None,
        }
    }

    pub fn as_array_view_u8(&self) -> Option<ndarray::ArrayViewD<'_, u8>> {
        let arr = self.as_shared_array_u8()?;
        Some(arr.view())
    }

    pub fn as_shape_buffer(&self) -> Option<&[usize]> {
        match self {
            HabValue::ShapeBuffer(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[HabValue]> {
        match self {
            HabValue::List(b) => Some(b),
            _ => None,
        }
    }

    // #[cfg(feature = "pyo3")]
    pub fn as_pyobject(&self) -> Option<Option<&PyObject>> {
        match self {
            HabValue::PyObject(b) => Some(b.0.as_ref()),
            _ => None,
        }
    }
}

impl std::fmt::Debug for HabValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HabValue::Bool(b) => write!(f, "Bool({})", b),
            HabValue::UnsignedLongLong(b) => write!(f, "UnsignedLongLong({})", b),
            HabValue::Integer(b) => write!(f, "Integer({})", b),
            HabValue::String(b) => write!(f, "String({:?})", b),
            HabValue::Float(b) => write!(f, "Float({})", b),
            HabValue::ByteBuffer(b) => write!(f, "ByteBuffer({} items)", b.len()),
            HabValue::IntBuffer(b) => write!(f, "IntBuffer({} items)", b.len()),
            // shape buffers will never be more than a few items long
            HabValue::ShapeBuffer(b) => write!(f, "ShapeBuffer({:?})", b),
            HabValue::SharedArrayF32(b) => {
                write!(
                    f,
                    "SharedArrayF32(shape={:?}, strides={:?}, items={})",
                    b.0.shape(),
                    b.0.strides(),
                    b.0.len()
                )
            }
            HabValue::SharedArrayU8(b) => {
                write!(
                    f,
                    "SharedArrayU8(shape={:?}, strides={:?}, items={})",
                    b.0.shape(),
                    b.0.strides(),
                    b.0.len()
                )
            }
            HabValue::List(b) => {
                if b.len() <= 10 {
                    write!(f, "List({:?})", b)
                } else {
                    write!(f, "List({} items)", b.len())
                }
            }
            // #[cfg(feature = "pyo3")]
            HabValue::PyObject(b) => {
                if let Some(v) = &b.0 {
                    write!(f, "PyObject({:p})", v.as_ptr())
                } else {
                    write!(f, "PyObject(None)")
                }
            }
            HabValue::Null => write!(f, "Null"),
        }
    }
}

impl From<bool> for HabValue {
    fn from(b: bool) -> Self {
        HabValue::Bool(b)
    }
}
impl From<u128> for HabValue {
    fn from(b: u128) -> Self {
        HabValue::UnsignedLongLong(b)
    }
}
impl From<i32> for HabValue {
    fn from(b: i32) -> Self {
        HabValue::Integer(b)
    }
}
impl From<&'static str> for HabValue {
    fn from(b: &'static str) -> Self {
        HabValue::String(b.into())
    }
}
impl From<String> for HabValue {
    fn from(b: String) -> Self {
        HabValue::String(b.into())
    }
}
impl From<HabString> for HabValue {
    fn from(b: HabString) -> Self {
        HabValue::String(b)
    }
}
impl From<OrderedFloat<f64>> for HabValue {
    fn from(b: OrderedFloat<f64>) -> Self {
        HabValue::Float(b)
    }
}
impl From<f64> for HabValue {
    fn from(b: f64) -> Self {
        HabValue::Float(b.into())
    }
}
impl From<Vec<u8>> for HabValue {
    fn from(b: Vec<u8>) -> Self {
        HabValue::ByteBuffer(b)
    }
}
impl From<Vec<i32>> for HabValue {
    fn from(b: Vec<i32>) -> Self {
        HabValue::IntBuffer(b)
    }
}
impl From<Vec<usize>> for HabValue {
    fn from(b: Vec<usize>) -> Self {
        HabValue::ShapeBuffer(b)
    }
}
impl From<Vec<HabValue>> for HabValue {
    fn from(b: Vec<HabValue>) -> Self {
        HabValue::List(b)
    }
}

impl From<MyPyObject> for HabValue {
    fn from(b: MyPyObject) -> Self {
        HabValue::PyObject(b)
    }
}

impl From<Py<PyAny>> for HabValue {
    fn from(b: Py<PyAny>) -> Self {
        HabValue::PyObject(MyPyObject(Some(b)))
    }
}

impl Drop for HabValue {
    fn drop(&mut self) {
        match self {
            Self::List(l) => {
                for _i in l.drain(..) {
                    // nothing extra to do
                    // drop(i);
                }
                // return the vec itself
                crate::basic_pooling::return_value_vec(std::mem::take(l));
            }
            _other => {
                // nothing extra to do at this time
                // later we could implement other things like
                // - reference counting
                // - pooling for vecs
                // - pooling for strings
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub enum HabValueType {
    Bool,
    UnsignedLongLong,
    Integer,
    String,
    Float,
    ByteBuffer,
    IntBuffer,
    ShapeBuffer,
    ArrayOfF32,
    ArrayOfU8,
    List,
    // #[cfg(feature = "pyo3")]
    PyObject,
    Null,
}

impl HabValue {
    pub fn get_type(&self) -> HabValueType {
        match self {
            HabValue::Bool(_) => HabValueType::Bool,
            HabValue::UnsignedLongLong(_) => HabValueType::UnsignedLongLong,
            HabValue::Integer(_) => HabValueType::Integer,
            HabValue::String(_) => HabValueType::String,
            HabValue::Float(_) => HabValueType::Float,
            HabValue::ByteBuffer(_) => HabValueType::ByteBuffer,
            HabValue::IntBuffer(_) => HabValueType::IntBuffer,
            HabValue::SharedArrayU8(_) => HabValueType::ArrayOfU8,
            HabValue::ShapeBuffer(_) => HabValueType::ShapeBuffer,
            HabValue::List(_) => HabValueType::List,
            HabValue::SharedArrayF32(_) => HabValueType::ArrayOfF32,
            // #[cfg(feature = "pyo3")]
            HabValue::PyObject(_) => HabValueType::PyObject,
            HabValue::Null => HabValueType::Null,
        }
    }
}

pub type HabString = crate::caching::HabStringBetter<'static>;

// pub type Tuple = BTreeMap<HabString, HabValue>;
pub type Tuple = BetterTuple;

pub trait Operator {
    fn get_id(&self) -> usize;
    fn add_parent(&mut self, id: usize);
    fn initialize(&mut self) {}
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd)]
pub enum OperatorOutput {
    Something(Option<usize>),
    Nothing,
    Finished,
}

// TODO: custom clone implementation. ids should be unique
#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub struct BetterTuple {
    pub(crate) tuple: BTreeMap<HabString, HabValue>,
    pub(crate) tuple_id: usize,
    pub(crate) unix_time_created_ns: u128,
}

// clones need to have their own unique ids but because they are derived from the original they should have the same timestamp since they are subject to the same time constraints
impl Clone for BetterTuple {
    fn clone(&self) -> Self {
        let mut new_tuple = crate::basic_pooling::get_tuple();
        new_tuple.unix_time_created_ns = self.unix_time_created_ns;
        for (k, v) in self.tuple.iter() {
            new_tuple.tuple.insert(k.clone(), v.clone());
        }
        new_tuple
    }
}

impl BetterTuple {
    pub fn id(&self) -> usize {
        self.tuple_id
    }
    pub fn unix_time_created_ns(&self) -> u128 {
        self.unix_time_created_ns
    }
    pub fn reset_time_created(&mut self) {
        self.unix_time_created_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|dur| dur.as_nanos())
            .unwrap_or(0);
    }
    pub fn mirror_time_created(&mut self, other: &Self) {
        self.unix_time_created_ns = other.unix_time_created_ns;
    }
    // default that is never valid. for use with swapping and the pooling system
    pub(crate) const fn default_internal() -> Self {
        Self {
            tuple_id: 0,
            unix_time_created_ns: 0,
            tuple: BTreeMap::new(),
        }
    }

    pub fn into_serializable(&self) -> FullySerializableTuple<'_> {
        let unix_time_created_ns = self.unix_time_created_ns;
        let relative_time_created_ns = unix_time_created_ns.checked_sub(*crate::global_logger::SYSTEM_START_TIME_NS).unwrap_or_else(|| {
            warn!("Tuple has a timestamp earlier than system start time. setting relative time to 0");
            0
        });
        FullySerializableTuple {
            tuple_id: self.tuple_id,
            relative_time_created_ns,
            // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
            //   so any type where it appears in the generics will also be represented the same way
            tuple: unsafe { std::mem::transmute(&self.tuple) },
        }
    }
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Vec<Self>> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Vec<Self>> {
        let deserialized: Vec<FullyDeserializableTuple> = rmp_serde::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(deserialized
            .into_iter()
            .map(|deserialized| Self {
                tuple_id: deserialized.tuple_id,
                unix_time_created_ns: deserialized.relative_time_created_ns
                    + *crate::global_logger::SYSTEM_START_TIME_NS,
                // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
                //  so any type where it appears in the generics will also be represented the same way
                tuple: unsafe { std::mem::transmute(deserialized.tuple) },
            })
            .collect())
    }
    pub fn from_bytes_lazy_timestamp(bytes: &[u8]) -> std::io::Result<impl Iterator<Item = Self>> {
        let start_time_ns = *crate::global_logger::SYSTEM_START_TIME_NS;
        let deserialized: Vec<FullyDeserializableTuple> = rmp_serde::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(deserialized.into_iter().map(move |deserialized| Self {
            tuple_id: deserialized.tuple_id,
            unix_time_created_ns: deserialized.relative_time_created_ns + start_time_ns,
            // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
            //  so any type where it appears in the generics will also be represented the same way
            tuple: unsafe { std::mem::transmute(deserialized.tuple) },
        }))
    }

    // makes a new tuple with a delay corresponding to the relative timestamp for each tuple in the file
    pub fn stream_from_file_lazy_timestamp<P: AsRef<std::path::Path>>(
        fname: P,
        runtime: tokio::runtime::Handle,
    ) -> std::io::Result<
        Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>>,
    > {
        let bytes = std::fs::read(fname)?;
        let deserialized: Vec<FullyDeserializableTuple> = rmp_serde::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self::stream_lazy_timestamp(
            deserialized,
            runtime,
            |t: FullyDeserializableTuple| -> Result<FullyDeserializableTuple, std::convert::Infallible> {
                Ok(t)
            },
        ))
    }

    // makes a new tuple with a delay corresponding to the relative timestamp for each tuple in the list of strings
    pub fn stream_from_strings_lazy_timestamp(
        base64_rmpe_strings: Vec<HabString>,
        runtime: tokio::runtime::Handle,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>> {
        Self::stream_lazy_timestamp(
            base64_rmpe_strings,
            runtime,
            |t: HabString| -> anyhow::Result<FullyDeserializableTuple> {
                use anyhow::Context;
                use base64::prelude::*;
                let decoded_bytes = BASE64_STANDARD
                    .decode(t.as_str())
                    .context("Failed to decode base64 string")?;
                let t: FullyDeserializableTuple =
                    rmp_serde::from_slice(&decoded_bytes).context("Failed to decode rmp tuple")?;
                Ok(t)
            },
        )
    }

    pub fn stream_lazy_timestamp<Input, Decoder, DecodeError>(
        inputs: Vec<Input>,
        runtime: tokio::runtime::Handle,
        decoder: Decoder,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>>
    where
        Decoder: 'static
            + Send
            + Sync
            + Clone
            + Fn(Input) -> Result<FullyDeserializableTuple, DecodeError>,
        DecodeError: std::fmt::Display,
        Input: 'static + Send,
    {
        let start_time_ns = *crate::global_logger::SYSTEM_START_TIME_NS;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let time_so_far = now_ns.checked_sub(start_time_ns).unwrap_or(0);
        let (channel_in, channel_out) = tokio::sync::mpsc::channel(100);
        for input in inputs {
            let my_sender = channel_in.clone();
            let decoder = decoder.clone();
            runtime.spawn_blocking(move || {
                let t: FullyDeserializableTuple = match decoder(input) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Failed to decode tuple: {}", e);
                        return;
                    }
                };
                std::thread::sleep(std::time::Duration::from_nanos(
                    // do not wait for longer if we are already on the way to or past the timestamp
                    (t.relative_time_created_ns - time_so_far) as _,
                ));
                let tuple = Self {
                    tuple_id: t.tuple_id,
                    unix_time_created_ns: t.relative_time_created_ns + start_time_ns,
                    // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
                    //  so any type where it appears in the generics will also be represented the same way
                    tuple: unsafe { std::mem::transmute(t.tuple) },
                };
                my_sender.blocking_send(vec![tuple]).unwrap();
            });
        }
        Box::new(move || Box::new(tokio_stream::wrappers::ReceiverStream::new(channel_out)))
    }

    pub fn stream_from_strings_manual_timestamp_adjustment(
        base64_rmpe_strings: Vec<HabString>,
        adjustments: impl 'static + Send + Iterator<Item = RehydratedTupleIteratorInfo>,
        runtime: tokio::runtime::Handle,
        behavior: SourceStreamBehavior,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>> {
        Self::stream_lazy_manual_timestamp_adjustment(
            base64_rmpe_strings,
            adjustments,
            runtime,
            |t: HabString| -> anyhow::Result<FullyDeserializableTuple> {
                use anyhow::Context;
                use base64::prelude::*;
                let decoded_bytes = BASE64_STANDARD
                    .decode(t.as_str())
                    .context("Failed to decode base64 string")?;
                let t: FullyDeserializableTuple =
                    rmp_serde::from_slice(&decoded_bytes).context("Failed to decode rmp tuple")?;
                Ok(t)
            },
            behavior,
        )
    }

    pub fn stream_lazy_manual_timestamp_adjustment<Input, AdjustmentInfo, Decoder, DecodeError>(
        inputs: Vec<Input>,
        adjustments: AdjustmentInfo,
        runtime: tokio::runtime::Handle,
        decoder: Decoder,
        behavior: SourceStreamBehavior,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn Send + futures::Stream<Item = Vec<Tuple>>>>
    where
        Decoder: 'static
            + Send
            + Sync
            + Clone
            + Fn(Input) -> Result<FullyDeserializableTuple, DecodeError>,
        DecodeError: std::fmt::Display,
        Input: 'static + Send,
        AdjustmentInfo: 'static + Send + Iterator<Item = RehydratedTupleIteratorInfo>,
    {
        let start_time_ns = *crate::global_logger::SYSTEM_START_TIME_NS;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let time_so_far = now_ns.checked_sub(start_time_ns).unwrap_or(0);
        let (channel_in, channel_out) = tokio::sync::mpsc::channel(100);
        match behavior {
            SourceStreamBehavior::NonRepeating => Self::handle_non_repeating(
                inputs,
                adjustments,
                runtime,
                decoder,
                start_time_ns,
                time_so_far,
                channel_in,
                channel_out,
            ),
            SourceStreamBehavior::Take { num_items } => Self::handle_finite_take(
                inputs,
                adjustments,
                runtime,
                decoder,
                start_time_ns,
                time_so_far,
                channel_in,
                channel_out,
                num_items,
                1, // cycles = 1 (just take the first num_items and then stop)
            ),
            SourceStreamBehavior::Infinite { num_items: _ } => todo!("not supported yet"),
            SourceStreamBehavior::Finite { num_items, cycles } => Self::handle_finite_take(
                inputs,
                adjustments,
                runtime,
                decoder,
                start_time_ns,
                time_so_far,
                channel_in,
                channel_out,
                num_items,
                cycles,
            ),
        }
    }

    fn handle_non_repeating<Input, AdjustmentInfo, Decoder, DecodeError>(
        inputs: Vec<Input>,
        adjustments: AdjustmentInfo,
        runtime: tokio::runtime::Handle,
        decoder: Decoder,
        start_time_ns: u128,
        time_so_far: u128,
        channel_in: tokio::sync::mpsc::Sender<Vec<BetterTuple>>,
        channel_out: tokio::sync::mpsc::Receiver<Vec<BetterTuple>>,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn futures::Stream<Item = Vec<BetterTuple>> + Send>>
    where
        Decoder: 'static
            + Send
            + Sync
            + Clone
            + Fn(Input) -> Result<FullyDeserializableTuple, DecodeError>,
        AdjustmentInfo: Iterator<Item = RehydratedTupleIteratorInfo>,
        DecodeError: std::fmt::Display,
        Input: 'static + Send,
    {
        for (input, adjustment) in inputs.into_iter().zip(adjustments) {
            let my_sender = channel_in.clone();
            let decoder = decoder.clone();
            runtime.spawn_blocking(move || {
                let t: FullyDeserializableTuple = match decoder(input) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Failed to decode tuple: {}", e);
                        return;
                    }
                };
                std::thread::sleep(std::time::Duration::from_nanos(
                    // do not wait for longer if we are already on the way to or past the timestamp
                    (adjustment.relative_emit_time_ns as i128 - time_so_far as i128).max(0) as _,
                ));
                let tuple = Self {
                    tuple_id: t.tuple_id,
                    unix_time_created_ns: adjustment.relative_emit_time_ns
                        + start_time_ns
                        + adjustment.creation_time_adjustment_ns.max(0) as u128,
                    // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
                    //  so any type where it appears in the generics will also be represented the same way
                    tuple: unsafe { std::mem::transmute(t.tuple) },
                };
                my_sender.blocking_send(vec![tuple]).unwrap();
            });
        }
        Box::new(move || Box::new(tokio_stream::wrappers::ReceiverStream::new(channel_out)) as _)
    }

    fn handle_finite_take<Input, AdjustmentInfo, Decoder, DecodeError>(
        inputs: Vec<Input>,
        mut adjustments: AdjustmentInfo,
        runtime: tokio::runtime::Handle,
        decoder: Decoder,
        start_time_ns: u128,
        time_so_far: u128,
        channel_in: tokio::sync::mpsc::Sender<Vec<BetterTuple>>,
        channel_out: tokio::sync::mpsc::Receiver<Vec<BetterTuple>>,
        num_items: usize,
        cycles: usize,
    ) -> Box<dyn Send + Sync + FnOnce() -> Box<dyn futures::Stream<Item = Vec<BetterTuple>> + Send>>
    where
        Decoder: 'static
            + Send
            + Sync
            + Clone
            + Fn(Input) -> Result<FullyDeserializableTuple, DecodeError>,
        AdjustmentInfo: 'static + Send + Iterator<Item = RehydratedTupleIteratorInfo>,
        DecodeError: std::fmt::Display,
        Input: 'static + Send,
    {
        if cycles == 0 {
            error!("Cycles cannot be zero in Finite behavior. Defaulting to 1");
        }
        if num_items == 0 {
            warn!("num_items was set to zero, which means no items will be emitted");
        }
        let clone_needed = cycles > 1;
        let mut copied_inputs = Vec::new();

        for (input, adjustment) in inputs.into_iter().zip(&mut adjustments).take(num_items) {
            let my_sender = channel_in.clone();
            let t: FullyDeserializableTuple = match decoder(input) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to decode tuple: {}", e);
                    continue;
                }
            };
            if clone_needed {
                copied_inputs.push(t.clone());
            }
            runtime.spawn_blocking(move || {
                debug!(
                    "tuple {} has an adjusted emit time of {} ns",
                    t.tuple_id, adjustment.relative_emit_time_ns
                );
                std::thread::sleep(std::time::Duration::from_nanos(
                    // do not wait for longer if we are already on the way to or past the timestamp
                    (adjustment.relative_emit_time_ns as i128 - time_so_far as i128).max(0) as _,
                ));

                let now = std::time::Instant::now();
                let system_start = *crate::global_logger::SYSTEM_START;
                let elapsed_ns = now.duration_since(system_start).as_nanos();
                let adjusted_unix_time = (
                    (adjustment.relative_emit_time_ns + start_time_ns) as i128 + adjustment.creation_time_adjustment_ns
                ).max(0) as u128;
                debug!(
                    "Emitting tuple {} after {} , at {} ns since system start, for an adjusted unix time of {})",
                    t.tuple_id, adjustment.relative_emit_time_ns, elapsed_ns, adjusted_unix_time,
                );
                let tuple = Self {
                    tuple_id: t.tuple_id,
                    unix_time_created_ns: adjustment.relative_emit_time_ns
                        + start_time_ns
                        + adjustment.creation_time_adjustment_ns.max(0) as u128,
                    // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
                    //  so any type where it appears in the generics will also be represented the same way
                    tuple: unsafe { std::mem::transmute(t.tuple) },
                };
                my_sender.blocking_send(vec![tuple]).unwrap();
            });
        }
        let remaining_cycles = cycles.saturating_sub(1);
        let remaining_items = copied_inputs.len() * remaining_cycles;
        debug!(
            "Cached results spout will emit a total of {} additional items in {} additional cycles. Our cache has {} items",
            remaining_items,
            remaining_cycles,
            copied_inputs.len()
        );
        runtime.spawn_blocking(move || {
            const LOG_FREQ: usize = 100;
            let mut cycled_items = 0;
            // let my_sender = channel_in.clone();
            let my_sender = channel_in;
            for (tuple, adjustment) in copied_inputs
                .into_iter()
                .cycle()
                .take(remaining_items)
                .zip(adjustments)
            {

                let now = std::time::Instant::now();
                let system_start = *crate::global_logger::SYSTEM_START;
                let elapsed_ns = now.duration_since(system_start).as_nanos();
                let time_to_sleep = (
                    adjustment.relative_emit_time_ns as i128
                        - elapsed_ns as i128
                        + adjustment.creation_time_adjustment_ns
                ).max(0) as u64;
                cycled_items += 1;
                if cycled_items % LOG_FREQ == 0{
                    debug!(
                        "tuple {} will use adjustments {:?} to sleep for {time_to_sleep}",
                        tuple.tuple_id, adjustment
                    );
                }

                std::thread::sleep(std::time::Duration::from_nanos(
                    // do not wait for longer if we are already on the way to or past the timestamp
                    time_to_sleep,
                ));
                let adjusted_unix_time = (
                    (adjustment.relative_emit_time_ns + start_time_ns) as i128 + adjustment.creation_time_adjustment_ns
                ).max(0) as u128;
                if cycled_items % LOG_FREQ == 0{
                    let now = std::time::Instant::now();
                    let system_start = *crate::global_logger::SYSTEM_START;
                    let elapsed_ns = now.duration_since(system_start).as_nanos();
                    debug!(
                        "Emitting tuple {} after {} , at {} ns since system start, for an adjusted unix time of {})",
                        tuple.tuple_id, adjustment.relative_emit_time_ns, elapsed_ns, adjusted_unix_time,
                    );
                }
                let tuple = Self {
                    tuple_id: tuple.tuple_id,
                    unix_time_created_ns: adjusted_unix_time,
                    // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue,
                    //  so any type where it appears in the generics will also be represented the same way
                    tuple: unsafe { std::mem::transmute(tuple.tuple) },
                };
                my_sender.blocking_send(vec![tuple]).unwrap();
            }
            debug!("finished cycling through a total of {cycled_items} items");
        });
        Box::new(move || Box::new(tokio_stream::wrappers::ReceiverStream::new(channel_out)) as _)
    }
}
// pub struct SourceStreamControlConfig {
// }
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub enum SourceStreamBehavior {
    #[default]
    NonRepeating,
    Take {
        num_items: usize,
    },
    Infinite {
        num_items: usize,
    },
    Finite {
        num_items: usize,
        cycles: usize,
    },
}

#[derive(Debug)]
pub struct RehydratedTupleIteratorInfo {
    pub relative_emit_time_ns: u128,
    // After we have rehydrated the tuple at the relative emit time,
    //  we may want to adjust the creation time by some amount to simulate processing delays
    //  that would have happened by that point in the replay.
    pub creation_time_adjustment_ns: i128,
}

#[repr(transparent)]
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HabValueWrapper(HabValue);
impl Serialize for HabValueWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // self.0.serialize(serializer)
        let v = hab_to_serializable(&self.0);
        v.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for HabValueWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = OwnedSerializeableHabValueVariants::deserialize(deserializer)?;
        let v = match v {
            OwnedSerializeableHabValueVariants::Bool(b) => HabValue::Bool(b),
            OwnedSerializeableHabValueVariants::UnsignedLongLong(b) => {
                HabValue::UnsignedLongLong(b)
            }
            OwnedSerializeableHabValueVariants::Integer(b) => HabValue::Integer(b),
            OwnedSerializeableHabValueVariants::String(b) => {
                HabValue::String(b.into_owned().into())
            }
            OwnedSerializeableHabValueVariants::Float(b) => HabValue::Float(b.into()),
            OwnedSerializeableHabValueVariants::ByteBuffer(b) => HabValue::ByteBuffer(b),
            OwnedSerializeableHabValueVariants::IntBuffer(b) => {
                let ints: Vec<i32> = b
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().expect("slice with incorrect length");
                        i32::from_le_bytes(arr)
                    })
                    .collect();
                HabValue::IntBuffer(ints.to_vec())
            }
            OwnedSerializeableHabValueVariants::ShapeBuffer(b) => {
                let usizes: Vec<usize> = b
                    .chunks_exact(std::mem::size_of::<usize>())
                    .map(|chunk| {
                        let arr: [u8; std::mem::size_of::<usize>()] =
                            chunk.try_into().expect("slice with incorrect length");
                        usize::from_le_bytes(arr)
                    })
                    .collect();
                HabValue::ShapeBuffer(usizes.to_vec())
            }
            OwnedSerializeableHabValueVariants::ArrayF32(b) => {
                unimplemented!("Deserialization of SharedArrayF32 is not implemented yet")
                // let floats: Vec<f32> = b
                //     .chunks_exact(4)
                //     .map(|chunk| {
                //         let arr: [u8; 4] = chunk.try_into().expect("slice with incorrect length");
                //         f32::from_le_bytes(arr)
                //     })
                //     .collect();
                // // HabValue::ArrayF32(floats.to_vec())
                // HabValue::SharedArrayF32(std::sync::Arc::new(ndarray::ArrayD::from_shape_vec(
                //     ndarray::IxDyn(&[floats.len()]),
                //     floats,
                // ).expect("Failed to create ArrayD from Vec<f32>")))
            }
            OwnedSerializeableHabValueVariants::ArrayU8(b) => {
                unimplemented!("Deserialization of SharedArrayU8 is not implemented yet")
                // HabValue::SharedArrayU8(std::sync::Arc::new(ndarray::ArrayD::from_shape_vec(
                //     ndarray::IxDyn(&[b.len()]),
                //     b,
                // ).expect("Failed to create ArrayD from Vec<u8>")))
            }
            OwnedSerializeableHabValueVariants::List(b) => {
                let mut vec = crate::basic_pooling::get_value_vec();
                for item in b.iter() {
                    vec.push(item.0.clone());
                }
                HabValue::List(vec)
            }
            OwnedSerializeableHabValueVariants::Null => HabValue::Null,
        };
        Ok(HabValueWrapper(v))
    }
}

#[derive(Serialize, Debug)]
enum SerializeableHabValueVariants<'a> {
    Bool(bool),
    UnsignedLongLong(u128),
    Integer(i32),
    String(Cow<'a, str>),
    Float(f64),
    #[serde(with = "serde_bytes")]
    ByteBuffer(Cow<'a, [u8]>),
    #[serde(with = "serde_bytes")]
    // IntBuffer(&'a [i32]),
    IntBuffer(Cow<'a, [u8]>),
    #[serde(with = "serde_bytes")]
    // ShapeBuffer(&'a [usize]),
    ShapeBuffer(Cow<'a, [u8]>),
    #[serde(with = "serde_bytes")]
    // ShapeBuffer(&'a [usize]),
    ArrayF32(Cow<'a, [u8]>),
    List(Cow<'a, [HabValueWrapper]>),
    Null,
}

#[derive(Deserialize, Serialize, Debug)]
enum OwnedSerializeableHabValueVariants {
    Bool(bool),
    UnsignedLongLong(u128),
    Integer(i32),
    String(Cow<'static, str>),
    Float(f64),
    ByteBuffer(Vec<u8>),
    IntBuffer(Vec<u8>),
    ShapeBuffer(Vec<u8>),
    ArrayF32(Vec<u8>),
    ArrayU8(Vec<u8>),
    List(Vec<HabValueWrapper>),
    Null,
}

fn hab_to_serializable<'a>(v: &'a HabValue) -> SerializeableHabValueVariants<'a> {
    use bytemuck::cast_slice;
    match v {
        HabValue::Bool(b) => SerializeableHabValueVariants::Bool(*b),
        HabValue::UnsignedLongLong(b) => SerializeableHabValueVariants::UnsignedLongLong(*b),
        HabValue::Integer(b) => SerializeableHabValueVariants::Integer(*b),
        HabValue::String(b) => SerializeableHabValueVariants::String(Cow::Borrowed(b.as_str())),
        HabValue::Float(b) => SerializeableHabValueVariants::Float(b.0),
        HabValue::ByteBuffer(b) => {
            SerializeableHabValueVariants::ByteBuffer(Cow::Borrowed(b.as_slice()))
        }
        HabValue::IntBuffer(b) => {
            let bytes: &[u8] = cast_slice(b.as_slice());
            SerializeableHabValueVariants::IntBuffer(Cow::Borrowed(bytes))
        }
        HabValue::ShapeBuffer(b) => {
            let bytes: &[u8] = cast_slice(b.as_slice());
            SerializeableHabValueVariants::ShapeBuffer(Cow::Borrowed(bytes))
        }
        HabValue::SharedArrayF32(b) => {
            // let bytes: &[u8] = cast_slice(b.as_slice());
            unimplemented!("SharedArrayF32 serialization is not implemented yet")
            // SerializeableHabValueVariants::ArrayF32(Cow::Borrowed(bytes))
        }
        HabValue::SharedArrayU8(b) => {
            // SerializeableHabValueVariants::ArrayF32(Cow::Borrowed(b.as_slice()))
            unimplemented!("SharedArrayU8 serialization is not implemented yet")
        }
        HabValue::List(b) => {
            // SAFETY: we know that HabValueWrapper is repr(transparent) over HabValue
            let wrapped: &[HabValueWrapper] = unsafe { std::mem::transmute(b.as_slice()) };
            SerializeableHabValueVariants::List(Cow::Borrowed(wrapped))
        }
        HabValue::Null => SerializeableHabValueVariants::Null,
        HabValue::PyObject(_p) => {
            warn!("Cannot serialize PyObject. please convert to native habitat values or encode/decode it within python");
            SerializeableHabValueVariants::Null
        }
    }
}

#[derive(Serialize)]
pub struct FullySerializableTuple<'a> {
    pub tuple_id: usize,
    // pub unix_time_created_ns: u128,
    pub relative_time_created_ns: u128,
    pub tuple: &'a BTreeMap<HabString, HabValueWrapper>,
}

#[derive(Debug, Clone)]
pub struct FullyDeserializableTuple {
    pub tuple_id: usize,
    pub relative_time_created_ns: u128,
    pub tuple: BTreeMap<HabString, HabValueWrapper>,
}
// we need a custom deserializer to act on additional behavior,
// like ensuring that the global counter is updated when we load tuples from disk
impl<'de> Deserialize<'de> for FullyDeserializableTuple {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            tuple_id: usize,
            relative_time_created_ns: u128,
            tuple: BTreeMap<HabString, HabValueWrapper>,
        }
        let h = Helper::deserialize(deserializer)?;
        // ensure the global counter does not use the ids that have been loaded
        let _old = crate::basic_pooling::UUID_COUNTER
            .fetch_max(h.tuple_id + 1, std::sync::atomic::Ordering::SeqCst);
        Ok(FullyDeserializableTuple {
            tuple_id: h.tuple_id,
            relative_time_created_ns: h.relative_time_created_ns,
            tuple: h.tuple,
        })
    }
}
#[cfg(test)]
// use msgpack to serialize and deserialize tuples when we need the full data
mod binary_tuple_serialization_tests {
    use super::*;
    #[test]
    fn test_tuple_serialization() {
        let start_time = *crate::global_logger::SYSTEM_START_TIME_NS;
        let mut t = crate::basic_pooling::get_tuple();
        t.tuple.insert("a".into(), HabValue::Integer(42));
        t.tuple.insert("b".into(), HabValue::String("hello".into()));
        t.tuple.insert("c".into(), HabValue::Float(3.14.into()));
        t.tuple.insert("d".into(), HabValue::Bool(true));
        t.tuple
            .insert("e".into(), HabValue::ByteBuffer(vec![1, 2, 3, 4]));
        t.tuple
            .insert("f".into(), HabValue::IntBuffer(vec![10, 20, 30]));
        t.tuple
            .insert("g".into(), HabValue::ShapeBuffer(vec![2, 3, 4]));
        t.tuple.insert(
            "h".into(),
            HabValue::List(vec![
                HabValue::Integer(1),
                HabValue::String("two".into()),
                HabValue::Float(3.0.into()),
            ]),
        );
        let serializable = t.into_serializable();
        let bytes = rmp_serde::to_vec(&serializable).unwrap();
        let deserialized: FullyDeserializableTuple = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(deserialized.tuple_id, serializable.tuple_id);
        assert_eq!(
            deserialized.relative_time_created_ns,
            serializable.relative_time_created_ns
        );
        assert_eq!(deserialized.tuple.len(), serializable.tuple.len());
        for (k, v) in deserialized.tuple.iter() {
            let original_v = serializable.tuple.get(k).unwrap();
            assert_eq!(v, original_v);
        }
        // serialize a vec of tuples to bytes
        let bytes = rmp_serde::to_vec(&vec![&serializable]).unwrap();

        // also test that we can convert back to BetterTuple using the byte
        let t: Vec<BetterTuple> =
            BetterTuple::from_bytes(&bytes).expect("Failed to deserialize tuples");
        assert_eq!(t.len(), 1);
        let t = &t[0];
        assert_eq!(t.tuple_id, serializable.tuple_id);
        assert_eq!(
            t.unix_time_created_ns,
            serializable.relative_time_created_ns + start_time
        );
        assert_eq!(t.tuple.len(), serializable.tuple.len());
        for (k, v) in t.tuple.iter() {
            let original_v = serializable.tuple.get(k).unwrap();
            assert_eq!(v, &original_v.0);
        }
    }
}

impl std::ops::Deref for BetterTuple {
    type Target = BTreeMap<HabString, HabValue>;

    fn deref(&self) -> &Self::Target {
        &self.tuple
    }
}
impl std::ops::DerefMut for BetterTuple {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tuple
    }
}

impl IntoIterator for BetterTuple {
    type Item = (HabString, HabValue);
    type IntoIter = std::collections::btree_map::IntoIter<HabString, HabValue>;

    fn into_iter(mut self) -> Self::IntoIter {
        std::mem::take(&mut self.tuple).into_iter()
    }
}

impl Drop for BetterTuple {
    fn drop(&mut self) {
        // return to the pool
        // this `take` is fine because the new empty tuple doesn't allocate anything
        let mut tmp = BetterTuple::default_internal();
        let id = self.tuple_id;
        debug!("Tuple::drop called => Returning tuple with id {id} to pool");
        tmp.tuple_id = self.tuple_id;
        tmp.unix_time_created_ns = self.unix_time_created_ns;
        std::mem::swap(&mut tmp.tuple, &mut self.tuple);
        crate::basic_pooling::return_tuple(tmp);
        // and then this tuple, which is now empty, finishes its drop process
    }
}

// TODO: use the new tuple vec pooling everywhere, and change all references to Vec<Tuple> to this new wrapper (name pending)

#[repr(transparent)]
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TupleVecWrapper {
    vec: Vec<Tuple>,
}

impl TupleVecWrapper {
    pub fn drain<R>(&mut self, range: R) -> TupleVecDrain<'_>
    where
        R: std::ops::RangeBounds<usize>,
    {
        TupleVecDrain {
            inner: self.vec.drain(range),
        }
    }
}

impl Debug for TupleVecWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.vec, f)
    }
}

impl Deref for TupleVecWrapper {
    type Target = Vec<Tuple>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl DerefMut for TupleVecWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl Drop for TupleVecWrapper {
    fn drop(&mut self) {
        // return to the pool
        crate::basic_pooling::return_tuple_vec(std::mem::take(&mut self.vec));
    }
}

impl FromIterator<Tuple> for TupleVecWrapper {
    fn from_iter<T: IntoIterator<Item = Tuple>>(iter: T) -> Self {
        Self {
            vec: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<Tuple>> for TupleVecWrapper {
    fn from(vec: Vec<Tuple>) -> Self {
        Self { vec }
    }
}

impl IntoIterator for TupleVecWrapper {
    type Item = Tuple;
    type IntoIter = TupleVecIntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        TupleVecIntoIter {
            inner: std::mem::take(&mut self.vec).into_iter(),
        }
    }
}

pub struct TupleVecIntoIter {
    inner: std::vec::IntoIter<Tuple>,
}

impl Iterator for TupleVecIntoIter {
    type Item = Tuple;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl Drop for TupleVecIntoIter {
    fn drop(&mut self) {
        // return to the pool
        for t in &mut self.inner {
            return_tuple(t);
        }
    }
}

pub struct TupleVecDrain<'a> {
    inner: std::vec::Drain<'a, Tuple>,
}

impl Iterator for TupleVecDrain<'_> {
    type Item = Tuple;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
impl Drop for TupleVecDrain<'_> {
    fn drop(&mut self) {
        // return to the pool
        for t in &mut self.inner {
            return_tuple(t);
        }
    }
}

// TODO: track provenance system

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
// pub enum ProvenanceOperatorType {
//     Project,
//     Select,
//     Join,
//     GroupBy,
//     ChromaJoin,
//     ChannelRouter,
//     ChannelSpout,
//     Merge,
//     MergeSpout,
//     DeriveValue,
//     UserDefinedFunction,
//     UserDefinedSource,
//     PythonInlineFunction,
//     PythonRemoteFunction,
//     Union,
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub enum UnaryOpType {
//     Project,
//     Select,
//     PythonInlineFunction,
//     PythonRemoteFunction,
// }

// // struct ProvenanceCellBox {
// //     kind: ProvenanceCellKind,
// //     children: Vec<usize>,
// // }
// // TODO: should we use a Trie instead of a vec for some cases?
// //  what does a trie over this data structure look like?
// //  any solution would have to remember to extract the timestamps from each part
// //   since most timestamps should not match.
// //  The ones that are most likely to match are the ones that will merge anyway
// //   such as the ones that are the same in the first place.

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub struct ProvenanceCellBox{
//     unix_time_nanos: u128,
//     kind: ProvenanceCellBoxKind,
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub enum ProvenanceCellBoxKind {
//     SourceTuple(u128),
//     ProjectResult{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     SelectResult{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     JoinResult{
//         op_id: usize,
//         left_source: Box<ProvenanceCellBox>,
//         right_source: Box<ProvenanceCellBox>
//     },
//     GroupByResult{
//         op_id: usize,
//         group_provenances: Vec<ProvenanceCellBox>
//     },
//     ChromaJoinResult {
//         op_id: usize,
//         build_matches: Vec<ProvenanceCellBox>,
//         lookup_source: Box<ProvenanceCellBox>,
//     },
//     ChannelRouterResult {
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     ChannelSpout{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     Merge{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     MergeSpout{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     DeriveValue {
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     UserDefinedFunction{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     UserDefinedSource{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     PythonInlineFunction{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     PythonRemoteFunction{
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     },
//     Union {
//         op_id: usize,
//         derived_from: Box<ProvenanceCellBox>
//     }
// }

// // restructure the above box-based tree to use a Vec of indices to back it

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub struct ProvenanceCellIndex{
//     unix_time_nanos: u128,
//     kind: ProvenanceCellIndexKind,
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub enum ProvenanceCellIndexKind {
//     SourceTuple(u128),
//     ProjectResult{
//         op_id: usize,
//         derived_from: usize
//     },
//     SelectResult{
//         op_id: usize,
//         derived_from: usize
//     },
//     JoinResult{
//         op_id: usize,
//         left_source: usize,
//         right_source: usize
//     },
//     GroupByResult{
//         op_id: usize,
//         group_provenances: Vec<usize>
//     },
//     ChromaJoinResult {
//         op_id: usize,
//         build_matches: Vec<usize>,
//         lookup_source: usize,
//     },
//     ChannelRouterResult {
//         op_id: usize,
//         derived_from: usize
//     },
//     ChannelSpout{
//         op_id: usize,
//         derived_from: usize
//     },
//     Merge{
//         op_id: usize,
//         derived_from: usize
//     },
//     MergeSpout{
//         op_id: usize,
//         derived_from: usize
//     },
//     DeriveValue {
//         op_id: usize,
//         derived_from: usize
//     },
//     UserDefinedFunction{
//         op_id: usize,
//         derived_from: usize
//     },
//     UserDefinedSource{
//         op_id: usize,
//         derived_from: usize
//     },
//     PythonInlineFunction{
//         op_id: usize,
//         derived_from: usize
//     },
//     PythonRemoteFunction{
//         op_id: usize,
//         derived_from: usize
//     },
//     Union {
//         op_id: usize,
//         derived_from: usize
//     }
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
// pub struct ProvenanceRecordIndex {
//     root: Option<usize>,
//     backing_vec: Vec<ProvenanceCellIndexKind>,
// }

// impl ProvenanceRecordIndex {
//     pub fn new() -> Self {
//         Self {
//             backing_vec: Vec::new(),
//             root: None,
//         }
//     }

//     pub fn add(&mut self, cell: ProvenanceCellIndexKind) -> usize {
//         match cell {
//             cell @ ProvenanceCellIndexKind::SourceTuple(_) => {
//                 let idx = self.backing_vec.len();
//                 self.backing_vec.push(cell);
//                 self.root = Some(idx);
//                 idx
//             },
//             cell @ ProvenanceCellIndexKind::ProjectResult { op_id, derived_from } => {
//                 let Some(root) = self.root else {
//                     error!("Cannot add a derived cell without a root cell");
//                     panic!("Cannot add a derived cell without a root cell");
//                 };
//                 let idx = self.backing_vec.len();
//                 ce
//             },
//             ProvenanceCellIndexKind::SelectResult { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::JoinResult { op_id, left_source, right_source } => todo!(),
//             ProvenanceCellIndexKind::GroupByResult { op_id, group_provenances } => todo!(),
//             ProvenanceCellIndexKind::ChromaJoinResult { op_id, build_matches, lookup_source } => todo!(),
//             ProvenanceCellIndexKind::ChannelRouterResult { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::ChannelSpout { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::Merge { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::MergeSpout { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::DeriveValue { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::UserDefinedFunction { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::UserDefinedSource { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::PythonInlineFunction { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::PythonRemoteFunction { op_id, derived_from } => todo!(),
//             ProvenanceCellIndexKind::Union { op_id, derived_from } => todo!(),
//         }
//     }
// }

pub struct ListMap<K, V> {
    map: Vec<(K, V)>,
}

impl<K, V> ListMap<K, V> {
    pub fn new() -> Self {
        Self { map: Vec::new() }
    }
}
impl<K: PartialEq, V> ListMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some((_, v)) = self.map.iter_mut().find(|(k, _)| k == &key) {
            Some(std::mem::replace(v, value))
        } else {
            self.map.push((key, value));
            None
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(pos) = self.map.iter().position(|(k, _)| k == key) {
            Some(self.map.remove(pos).1)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter().map(|(k, v)| (k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.map.iter_mut().map(|(k, v)| (&*k, v))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn drain(&mut self) -> impl '_ + Iterator<Item = (K, V)> {
        self.map.drain(..)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.map.iter_mut().map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.iter().any(|(k, _)| k == key)
    }

    pub fn entry(&mut self, key: K) -> list_map_entry::ListMapEntry<'_, K, V> {
        if let Some((i, (_k, _v))) = self
            .map
            .iter_mut()
            .enumerate()
            .find(|(_i, (k, _v))| k == &key)
        {
            list_map_entry::ListMapEntry::Occupied(list_map_entry::OccupiedEntry {
                key,
                index: i,
                map: self,
            })
        } else {
            list_map_entry::ListMapEntry::Vacant(list_map_entry::VacantEntry { key, map: self })
        }
    }
}

pub mod list_map_entry {
    pub enum ListMapEntry<'a, K, V> {
        Occupied(OccupiedEntry<'a, K, V>),
        Vacant(VacantEntry<'a, K, V>),
    }

    impl<'a, K: PartialEq, V> ListMapEntry<'a, K, V> {
        pub fn or_insert(self, default: V) -> &'a mut V {
            match self {
                ListMapEntry::Occupied(e) => e.into_mut(),
                ListMapEntry::Vacant(e) => e.insert(default),
            }
        }

        pub fn or_insert_with<F: FnOnce() -> V>(self, default_fn: F) -> &'a mut V {
            match self {
                ListMapEntry::Occupied(e) => e.into_mut(),
                ListMapEntry::Vacant(e) => e.insert(default_fn()),
            }
        }

        pub fn or_default(self) -> &'a mut V
        where
            V: Default,
        {
            match self {
                ListMapEntry::Occupied(e) => e.into_mut(),
                ListMapEntry::Vacant(e) => e.insert(V::default()),
            }
        }
    }

    pub struct OccupiedEntry<'a, K, V> {
        pub(crate) key: K,
        pub(crate) index: usize,
        pub(crate) map: &'a mut super::ListMap<K, V>,
    }
    impl<'a, K: PartialEq, V> OccupiedEntry<'a, K, V> {
        pub fn get(&self) -> &V {
            &self.map.map[self.index].1
        }
        pub fn into_mut(self) -> &'a mut V {
            &mut self.map.map[self.index].1
        }
        pub fn get_mut(&mut self) -> &mut V {
            &mut self.map.map[self.index].1
        }
        pub fn insert(&mut self, value: V) -> V {
            std::mem::replace(&mut self.map.map[self.index].1, value)
        }
        pub fn remove(self) -> V {
            self.map.map.swap_remove(self.index).1
        }
    }

    pub struct VacantEntry<'a, K, V> {
        pub(crate) key: K,
        pub(crate) map: &'a mut super::ListMap<K, V>,
    }
    impl<'a, K: PartialEq, V> VacantEntry<'a, K, V> {
        pub fn insert(self, value: V) -> &'a mut V {
            self.map.map.push((self.key, value));
            &mut self.map.map.last_mut().unwrap().1
        }
    }
}

pub struct ListSet<T> {
    set: ListMap<T, ()>,
}
impl<T> ListSet<T> {
    pub fn new() -> Self {
        Self {
            set: ListMap::new(),
        }
    }
}
impl<T: PartialEq> ListSet<T> {
    pub fn insert(&mut self, value: T) -> bool {
        self.set.insert(value, ()).is_none()
    }

    pub fn contains(&self, value: &T) -> bool {
        self.set.contains_key(value)
    }

    pub fn remove(&mut self, value: &T) -> bool {
        self.set.remove(value).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.set.keys()
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    pub fn clear(&mut self) {
        self.set.clear();
    }
}

#[derive(Debug, Clone)]
pub struct Multiset<K> {
    inner: rustc_hash::FxHashMap<K, usize>,
}

impl<K: std::hash::Hash + Eq> PartialEq for Multiset<K> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<K: std::hash::Hash + Eq> Eq for Multiset<K> {}

impl<K: std::hash::Hash + Eq> Multiset<K> {
    pub fn new() -> Self {
        Self {
            inner: Default::default(),
        }
    }

    pub fn insert(&mut self, key: K) {
        *self.inner.entry(key).or_insert(0) += 1;
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<usize>
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        if let Some(count) = self.inner.get_mut(key) {
            if *count > 1 {
                *count -= 1;
                Some(*count)
            } else {
                self.inner.remove(key);
                Some(0)
            }
        } else {
            None
        }
    }
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: std::hash::Hash + Eq,
    {
        self.inner.contains_key(key)
    }

    pub fn count(&self, key: &K) -> usize {
        self.inner.get(key).copied().unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct ArrayMap<K, V, const N: usize> {
    map: smallvec::SmallVec<[(K, V); N]>,
}
impl<K, V, const N: usize> Default for ArrayMap<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K, V, const N: usize> ArrayMap<K, V, N> {
    pub fn new() -> Self {
        Self {
            map: smallvec::SmallVec::new(),
        }
    }
}

impl<K: Eq + PartialEq, V, const N: usize> ArrayMap<K, V, N> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some((_, v)) = self.map.iter_mut().find(|(k, _)| k == &key) {
            Some(std::mem::replace(v, value))
        } else {
            self.map.push((key, value));
            None
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + PartialEq,
    {
        self.map
            .iter()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + PartialEq,
    {
        self.map
            .iter_mut()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + PartialEq,
    {
        if let Some(pos) = self.map.iter().position(|(k, _)| k.borrow() == key) {
            Some(self.map.remove(pos).1)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter().map(|(k, v)| (k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.map.iter_mut().map(|(k, v)| (&*k, v))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn drain(&mut self) -> impl '_ + Iterator<Item = (K, V)> {
        self.map.drain(..)
    }

    pub fn entry(&mut self, key: K) -> array_map_entry::ArrayMapEntry<'_, K, V, N> {
        if let Some((i, (_k, _v))) = self
            .map
            .iter_mut()
            .enumerate()
            .find(|(_i, (k, _v))| k == &key)
        {
            array_map_entry::ArrayMapEntry::Occupied(array_map_entry::OccupiedEntry {
                key,
                index: i,
                map: self,
            })
        } else {
            array_map_entry::ArrayMapEntry::Vacant(array_map_entry::VacantEntry { key, map: self })
        }
    }
}

impl<K, V, const N: usize> IntoIterator for ArrayMap<K, V, N> {
    type Item = (K, V);
    type IntoIter = smallvec::IntoIter<[(K, V); N]>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

pub mod array_map_entry {
    pub enum ArrayMapEntry<'a, K, V, const N: usize> {
        Occupied(OccupiedEntry<'a, K, V, N>),
        Vacant(VacantEntry<'a, K, V, N>),
    }

    impl<'a, K: PartialEq, V, const N: usize> ArrayMapEntry<'a, K, V, N> {
        pub fn or_insert(self, default: V) -> &'a mut V {
            match self {
                ArrayMapEntry::Occupied(e) => e.into_mut(),
                ArrayMapEntry::Vacant(e) => e.insert(default),
            }
        }

        pub fn or_insert_with<F: FnOnce() -> V>(self, default_fn: F) -> &'a mut V {
            match self {
                ArrayMapEntry::Occupied(e) => e.into_mut(),
                ArrayMapEntry::Vacant(e) => e.insert(default_fn()),
            }
        }

        pub fn or_default(self) -> &'a mut V
        where
            V: Default,
        {
            match self {
                ArrayMapEntry::Occupied(e) => e.into_mut(),
                ArrayMapEntry::Vacant(e) => e.insert(V::default()),
            }
        }
    }

    pub struct OccupiedEntry<'a, K, V, const N: usize> {
        pub(crate) key: K,
        pub(crate) index: usize,
        pub(crate) map: &'a mut super::ArrayMap<K, V, N>,
    }
    impl<'a, K: PartialEq, V, const N: usize> OccupiedEntry<'a, K, V, N> {
        pub fn get(&self) -> &V {
            &self.map.map[self.index].1
        }
        pub fn into_mut(self) -> &'a mut V {
            &mut self.map.map[self.index].1
        }
        pub fn get_mut(&mut self) -> &mut V {
            &mut self.map.map[self.index].1
        }
        pub fn insert(&mut self, value: V) -> V {
            std::mem::replace(&mut self.map.map[self.index].1, value)
        }
        pub fn remove(self) -> V {
            self.map.map.swap_remove(self.index).1
        }
    }

    pub struct VacantEntry<'a, K, V, const N: usize> {
        pub(crate) key: K,
        pub(crate) map: &'a mut super::ArrayMap<K, V, N>,
    }
    impl<'a, K: PartialEq, V, const N: usize> VacantEntry<'a, K, V, N> {
        pub fn insert(self, value: V) -> &'a mut V {
            self.map.map.push((self.key, value));
            &mut self.map.map.last_mut().unwrap().1
        }
    }
}
