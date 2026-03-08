use dashmap::DashMap;
use std::any::TypeId;
use std::sync::OnceLock;

use crossbeam::channel::{unbounded, Receiver, Sender};

struct AbstractVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

// SAFETY: we do not allow touching of the fields of AbstractVec concurrently
unsafe impl Sync for AbstractVec {}
// SAFETY: we do not allow an AbstractVec to be created if the type it points to is not Send
unsafe impl Send for AbstractVec {}

struct AbstractVecChannel {
    tx: Sender<AbstractVec>,
    rx: Receiver<AbstractVec>,
}

static VEC_STORE: OnceLock<DashMap<TypeId, AbstractVecChannel>> = OnceLock::new();

fn get_vec_store() -> &'static DashMap<TypeId, AbstractVecChannel> {
    VEC_STORE.get_or_init(DashMap::new)
}

const MINIMUM_VEC_COUNT: usize = 64;
const MINIMUM_VEC_CAPACITY: usize = 16;

pub fn get_vec<T: Send + Sync + 'static>() -> Vec<T> {
    let type_id = TypeId::of::<T>();
    let vec_store = get_vec_store();
    let entry = vec_store.entry(type_id);
    match entry {
        dashmap::mapref::entry::Entry::Occupied(occupied_entry) => {
            let channel = occupied_entry.get();
            if let Ok(vec) = channel.rx.recv() {
                unsafe { Vec::from_raw_parts(vec.ptr as *mut T, vec.len, vec.cap) }
            } else {
                Vec::new()
            }
        }
        dashmap::mapref::entry::Entry::Vacant(entry) => {
            let (tx, rx) = unbounded();
            for _ in 0..MINIMUM_VEC_COUNT {
                let mut vec = Vec::<T>::with_capacity(MINIMUM_VEC_CAPACITY);
                let ptr = vec.as_mut_ptr() as _;
                let len = vec.len();
                let cap = vec.capacity();
                std::mem::forget(vec);
                let abstract_vec = AbstractVec { ptr, len, cap };
                tx.send(abstract_vec).unwrap();
            }
            entry.insert(AbstractVecChannel { tx, rx });
            Vec::new()
        }
    }
}

pub fn return_vec<T: Send + Sync + 'static>(mut vec: Vec<T>) {
    vec.clear();
    let type_id = TypeId::of::<T>();
    let vec_store = get_vec_store();
    let entry = vec_store.entry(type_id);
    match entry {
        dashmap::mapref::entry::Entry::Occupied(occupied_entry) => {
            let channel = occupied_entry.get();
            let vec = vec.into_boxed_slice();
            let ptr = vec.as_ptr() as *mut u8;
            let len = vec.len();
            let cap = vec.len();
            let abstract_vec = AbstractVec { ptr, len, cap };
            channel.tx.send(abstract_vec).unwrap();
        }
        dashmap::mapref::entry::Entry::Vacant(vacant_entry) => {
            // panic!("Tried to return a vec of a type that was not allocated");
            // make a new channel and return the vec
            let (tx, rx) = unbounded();
            let ptr = vec.as_ptr() as *mut u8;
            let len = vec.len();
            let cap = vec.len();
            let abstract_vec = AbstractVec { ptr, len, cap };
            std::mem::forget(vec);
            tx.send(abstract_vec).unwrap();
            // do that for the rest of the MINIMUM_VEC_COUNT times
            for _ in 1..MINIMUM_VEC_COUNT {
                let vec = Vec::<T>::with_capacity(MINIMUM_VEC_CAPACITY);
                let ptr = vec.as_ptr() as _;
                let len = vec.len();
                let cap = vec.capacity();
                std::mem::forget(vec);
                let abstract_vec = AbstractVec { ptr, len, cap };
                tx.send(abstract_vec).unwrap();
            }
            vacant_entry.insert(AbstractVecChannel { tx, rx });
        }
    }
}

pub trait VecPoolCollectExt
where
    Self: IntoIterator,
    Self::Item: Send + Sync + 'static,
{
    fn collect_into_vec(self) -> Vec<Self::Item>;
}
impl<I> VecPoolCollectExt for I
where
    I: IntoIterator,
    I::Item: Send + Sync + 'static,
{
    fn collect_into_vec(self) -> Vec<Self::Item> {
        let mut vec = get_vec();
        for item in self {
            vec.push(item);
        }
        vec
    }
}

pub trait VecPoolReturnExt {
    fn return_to_pool(self);
    fn return_vec(self)
    where
        Self: Sized,
    {
        self.return_to_pool();
    }
}

impl<T> VecPoolReturnExt for Vec<T>
where
    T: Send + Sync + 'static,
{
    fn return_to_pool(self) {
        return_vec(self);
    }
}

// TODO: make a global object pool for btrees as well
