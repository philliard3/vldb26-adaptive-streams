#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

use std::sync::{
    atomic::{self, AtomicUsize},
    OnceLock,
};

use crate::{HabString, HabValue, Tuple};

use crate::caching::StrToKey;

// TODO: change UUID into a provenance tree of UUIDs
pub static UUID_FIELD: &str = "__tuple_uuid__";
pub(crate) static UUID_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub fn new_uuid() -> usize {
    UUID_COUNTER.fetch_add(1, atomic::Ordering::Relaxed)
}

pub fn init_pools() {
    TUPLE_POOL.get_or_init(init_tuple_pool);
    TUPLE_VEC_POOL.get_or_init(init_tuple_vec_pool);
    VALUE_VEC_POOL.get_or_init(init_value_vec_pool);
}

pub(crate) trait CollectTuple {
    fn collect_tuple(self) -> Tuple;
}

impl<I> CollectTuple for I
where
    I: IntoIterator<Item = (HabString, HabValue)>,
{
    fn collect_tuple(self) -> Tuple {
        let mut t = get_tuple();
        t.extend(self);
        t
    }
}

pub(crate) trait CollectTupleVec {
    fn collect_tuple_vec(self) -> Vec<Tuple>;
}

impl<I> CollectTupleVec for I
where
    I: IntoIterator<Item = Tuple>,
{
    fn collect_tuple_vec(self) -> Vec<Tuple> {
        let mut vec = get_tuple_vec();
        vec.extend(self);
        vec
    }
}

pub(crate) trait CollectValueVec {
    fn collect_value_vec(self) -> Vec<HabValue>;
}

impl<I> CollectValueVec for I
where
    I: IntoIterator<Item = HabValue>,
{
    fn collect_value_vec(self) -> Vec<HabValue> {
        let mut vec = get_value_vec();
        vec.extend(self);
        vec
    }
}

pub fn make_tuple() -> Tuple {
    let mut tuple = Tuple::default_internal();
    let uuid = new_uuid();
    tuple.tuple_id = uuid;
    let unix_time_created_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    tuple.unix_time_created_ns = unix_time_created_ns;
    tuple.insert(UUID_FIELD.to_key(), HabValue::Integer(uuid as _));
    tuple.insert(
        "time_created".to_key(),
        HabValue::UnsignedLongLong(unix_time_created_ns),
    );
    tuple
}

use crossbeam::channel::{Receiver, Sender};
pub struct ChannelBundle<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

static TUPLE_POOL: OnceLock<ChannelBundle<Tuple>> = OnceLock::new();
const MINIMUM_TUPLE_COUNT: usize = 64;
const MAXIMUM_TUPLE_COUNT: usize = 1024;

pub fn init_tuple_pool() -> ChannelBundle<Tuple> {
    let (tx, rx) = crossbeam::channel::bounded(MAXIMUM_TUPLE_COUNT);
    for _ in 0..MINIMUM_TUPLE_COUNT {
        let t: Tuple = make_tuple();
        if let Err(e) = tx.send(t) {
            error!("Error sending tuple to pool: {:?}", e);
            panic!("Error sending tuple to pool: {:?}", e);
        }
    }
    ChannelBundle { tx, rx }
}

pub fn get_tuple() -> Tuple {
    let pool = TUPLE_POOL.get_or_init(init_tuple_pool);
    let mut t = match pool.rx.try_recv() {
        Ok(mut t) => {
            let tuple_id = new_uuid();
            t.tuple_id = tuple_id;
            let unix_time_created_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            t.unix_time_created_ns = unix_time_created_ns;
            t
        }
        Err(crossbeam::channel::TryRecvError::Empty) => make_tuple(),
        Err(e @ crossbeam::channel::TryRecvError::Disconnected) => {
            error!("Error receiving tuple from pool: {:?}", e);
            // panic!("Error receiving tuple from pool: {:?}", e);
            make_tuple()
        }
    };
    let tuple_id = t.tuple_id;
    t.insert(UUID_FIELD.to_key(), HabValue::Integer(tuple_id as _));
    let time = t.unix_time_created_ns;
    t.insert("time_created".to_key(), HabValue::UnsignedLongLong(time));
    t
}

pub fn return_tuple(mut t: Tuple) {
    let pool = TUPLE_POOL.get_or_init(|| {
        let (tx, rx) = crossbeam::channel::unbounded();
        for _ in 0..(MINIMUM_TUPLE_COUNT - 1) {
            let t: Tuple = make_tuple();
            if let Err(e) = tx.send(t) {
                error!("Error sending tuple to pool while initializing: {:?}", e);
            }
        }
        ChannelBundle { tx, rx }
    });
    debug!(
        "basic_pooling::return_tuple => returning tuple with id {} to pool",
        t.id()
    );

    if pool.tx.len() >= MAXIMUM_TUPLE_COUNT {
        // let Tuple { tuple_id, unix_time_created_ns, tuple } = t;
        let mut t = std::mem::ManuallyDrop::new(t);
        // ensuring that we have just a btreemap to drop,
        // since we do not want to drop the whole Tuple with its custom drop impl that would infinitely recurse
        let extra_btree: std::collections::BTreeMap<_, _> = std::mem::take(&mut t.tuple);
        drop(extra_btree);
        // we do not want to recursively call the drop implementation of Tuple, which would try to return it to this funciton
        std::mem::forget(t);
        return;
    }
    t.clear();
    if let Err(e) = pool.tx.send(t) {
        error!("Error sending tuple to pool: {:?}", e);
    }
}

// TODO: perhaps Tuple should be a wrapper and impl Drop so it automatically returns to the pool

static TUPLE_VEC_POOL: OnceLock<ChannelBundle<Vec<Tuple>>> = OnceLock::new();
const MINIMUM_VEC_COUNT: usize = 64;
const MINIMUM_VEC_CAPACITY: usize = 8;
const MAXIMUM_VEC_COUNT: usize = 1024;

pub fn init_tuple_vec_pool() -> ChannelBundle<Vec<Tuple>> {
    let (tx, rx) = crossbeam::channel::bounded(MAXIMUM_VEC_COUNT);
    for _ in 0..MINIMUM_VEC_COUNT {
        let vec = Vec::<Tuple>::with_capacity(MINIMUM_VEC_CAPACITY);
        tx.send(vec).unwrap();
    }
    ChannelBundle { tx, rx }
}

pub fn get_tuple_vec() -> Vec<Tuple> {
    let pool = TUPLE_VEC_POOL.get_or_init(init_tuple_vec_pool);
    if let Ok(vec) = pool.rx.try_recv() {
        vec
    } else {
        Vec::with_capacity(MINIMUM_VEC_CAPACITY)
    }
}

pub fn return_tuple_vec(mut vec: Vec<Tuple>) {
    let pool = TUPLE_VEC_POOL.get_or_init(|| {
        let (tx, rx) = crossbeam::channel::unbounded();
        for _ in 0..(MINIMUM_VEC_COUNT - 1) {
            let vec = Vec::<Tuple>::with_capacity(MINIMUM_VEC_CAPACITY);
            tx.send(vec).unwrap();
        }
        ChannelBundle { tx, rx }
    });
    if vec.capacity() < MINIMUM_VEC_CAPACITY {
        vec.reserve(MINIMUM_VEC_CAPACITY - vec.capacity());
    }

    for t in vec.drain(..) {
        return_tuple(t);
    }

    if pool.tx.len() >= MAXIMUM_VEC_COUNT {
        return;
    }

    if let Err(e) = pool.tx.send(vec) {
        error!("Error sending tuple vec to pool: {:?}", e);
        // panic!("Error sending tuple vec to pool: {:?}", e);
    }
}

static VALUE_VEC_POOL: OnceLock<ChannelBundle<Vec<HabValue>>> = OnceLock::new();
const MINIMUM_VAL_VEC_COUNT: usize = 8;
const MINIMUM_VAL_VEC_CAPACITY: usize = 8;
const MAXIMUM_VAL_VEC_COUNT: usize = 1024;

pub fn init_value_vec_pool() -> ChannelBundle<Vec<HabValue>> {
    let (tx, rx) = crossbeam::channel::bounded(MAXIMUM_VAL_VEC_COUNT);
    for _ in 0..MINIMUM_VAL_VEC_COUNT {
        let vec = Vec::<HabValue>::with_capacity(MINIMUM_VAL_VEC_CAPACITY);
        tx.send(vec).unwrap();
    }
    ChannelBundle { tx, rx }
}

pub fn get_value_vec() -> Vec<HabValue> {
    let pool = VALUE_VEC_POOL.get_or_init(init_value_vec_pool);
    if let Ok(vec) = pool.rx.recv() {
        vec
    } else {
        Vec::with_capacity(MINIMUM_VAL_VEC_CAPACITY)
    }
}

pub fn return_value_vec(mut vec: Vec<HabValue>) {
    let pool = VALUE_VEC_POOL.get_or_init(|| {
        let (tx, rx) = crossbeam::channel::unbounded();
        for _ in 0..(MINIMUM_VAL_VEC_COUNT - 1) {
            let vec = Vec::<HabValue>::with_capacity(MINIMUM_VAL_VEC_CAPACITY);
            tx.send(vec).unwrap();
        }
        ChannelBundle { tx, rx }
    });
    if vec.capacity() < MINIMUM_VAL_VEC_CAPACITY {
        vec.reserve(MINIMUM_VAL_VEC_CAPACITY - vec.capacity());
    }

    if pool.tx.len() >= MAXIMUM_VAL_VEC_COUNT {
        return;
    }

    // TODO: are there any variants that would like to have a chance to be recycled themselves?
    vec.clear();

    if let Err(e) = pool.tx.send(vec) {
        error!("Error sending value vec to pool: {:?}", e);
        panic!("Error sending value vec to pool: {:?}", e);
    }
}

pub trait GenPool: Sized {
    fn get_from_pool() -> Self;
}

impl GenPool for Tuple {
    fn get_from_pool() -> Self {
        get_tuple()
    }
}

impl GenPool for Vec<Tuple> {
    fn get_from_pool() -> Self {
        get_tuple_vec()
    }
}

impl GenPool for Vec<HabValue> {
    fn get_from_pool() -> Self {
        get_value_vec()
    }
}

pub trait RecyclePool: Sized {
    fn recycle(self);
    fn return_to_pool(self) {
        self.recycle();
    }
}

impl RecyclePool for Tuple {
    fn recycle(self) {
        return_tuple(self);
    }
}

impl RecyclePool for Vec<Tuple> {
    fn recycle(self) {
        return_tuple_vec(self);
    }
}

impl RecyclePool for Vec<HabValue> {
    fn recycle(self) {
        return_value_vec(self);
    }
}
