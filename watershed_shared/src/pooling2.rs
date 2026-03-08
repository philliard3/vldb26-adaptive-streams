use std::{
    collections::VecDeque,
    mem::{ManuallyDrop, MaybeUninit},
};

trait AutoTraitConstraints<Prop> {}

struct SendSync;
struct Constraint<Prop>(Prop);
impl<T> AutoTraitConstraints<SendSync> for T where T: Send + Sync {}

pub struct CacheWrapper<T, C = ThreadLocalCache> {
    value: Box<T>,
    cache: C,
}

pub trait CacheControl<'a> {
    fn withdraw<T: 'a>(&self) -> Option<Box<MaybeUninit<T>>>;
    fn withdraw_initialized<T: 'a>(&self, init_value: T) -> Option<Box<T>> {
        self.withdraw().map(|b| Box::write(b, init_value))
    }
    fn deposit<T: 'a>(&self, value: Box<MaybeUninit<T>>);
    fn deposit_and_drop<T: 'a>(&self, value: Box<T>) {
        self.deposit_and_drop_with_zero(value, false);
    }
    fn deposit_and_drop_with_zero<T: 'a>(&self, value: Box<T>, should_zero: bool) {
        // let mut value = ManuallyDrop::new(value);
        let value = Box::into_raw(value);
        unsafe {
            // SAFETY: we are manually dropping the value, so we can safely call drop_in_place
            std::ptr::drop_in_place(value);
            if should_zero {
                // SAFETY: we are zeroing the memory of the value, which is safe as long as we don't use it again
                std::ptr::write(value as _, MaybeUninit::<T>::zeroed());
            }
            // SAFETY: MaybeUninit<T> is repr(transparent), so it isvalid to cast a former Box<T> to a Box<MaybeUninit<T>>
            self.deposit::<T>(Box::from_raw(value as _))
        }
    }
}

// thread local type id to queue of boxes map
thread_local! {
    static THREAD_LOCAL_CACHE: std::cell::RefCell<std::collections::HashMap<std::any::TypeId, VecDeque<*mut ()>>> = {
        std::cell::RefCell::new(std::collections::HashMap::new())
    };
}

struct ThreadLocalCache;
impl CacheControl<'static> for ThreadLocalCache {
    fn withdraw<T: 'static>(&self) -> Option<Box<MaybeUninit<T>>> {
        THREAD_LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let type_id = std::any::TypeId::of::<T>();
            cache.get_mut(&type_id).and_then(|queue| {
                queue.pop_front().map(|ptr| unsafe {
                    // SAFETY: we are casting a raw pointer to a MaybeUninit<T>
                    Box::from_raw(ptr as *mut MaybeUninit<T>)
                })
            })
        })
    }

    fn deposit<T: 'static>(&self, value: Box<MaybeUninit<T>>) {
        THREAD_LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let type_id = std::any::TypeId::of::<T>();
            cache
                .entry(type_id)
                .or_default()
                .push_back(Box::into_raw(value) as *mut ());
        });
    }
}

struct GlobalDashMapCache;
struct GlobalDashMapCacheWrapper {
    cache: dashmap::DashMap<std::any::TypeId, (Sender<*mut ()>, Receiver<*mut ()>)>,
}
// SAFETY: we never allow mutable iteration over it, and we never use the pointers in a way where they can't be sent between threads
unsafe impl Send for GlobalDashMapCacheWrapper {}
unsafe impl Sync for GlobalDashMapCacheWrapper {}

use crossbeam::channel::{Receiver, Sender};
// static GLOBAL_DASH_MAP_CACHE: GlobalDashMapCacheWrapper = GlobalDashMapCacheWrapper{ cache: dashmap::DashMap::new() } ;
// make it lazy locked
use std::sync::LazyLock;
static GLOBAL_DASH_MAP_CACHE: LazyLock<GlobalDashMapCacheWrapper> = LazyLock::new(|| {
    let cache = dashmap::DashMap::new();
    GlobalDashMapCacheWrapper { cache }
});

impl CacheControl<'static> for GlobalDashMapCache {
    fn withdraw<T: 'static>(&self) -> Option<Box<MaybeUninit<T>>> {
        let cache = LazyLock::force(&GLOBAL_DASH_MAP_CACHE);
        let e = cache
            .cache
            .entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| {
                let (sender, receiver) = crossbeam::channel::unbounded();
                (sender, receiver)
            });
        e.value().1.recv().ok().and_then(|ptr| {
            // SAFETY: we only ever store pointers with the appropriate type, so this cast is valid
            unsafe { Some(Box::from_raw(ptr as *mut MaybeUninit<T>)) }
        })
    }

    fn deposit<T: 'static>(&self, value: Box<MaybeUninit<T>>) {
        let type_id = std::any::TypeId::of::<T>();
        // GLOBAL_DASH_MAP_CACHE.entry(type_id).or_default().lock().push_back(Box::into_raw(value) as *mut ());
        let cache = LazyLock::force(&GLOBAL_DASH_MAP_CACHE);
        let e = cache.cache.entry(type_id).or_insert_with(|| {
            let (sender, receiver) = crossbeam::channel::unbounded();
            (sender, receiver)
        });
        let sender = &e.value().0;
        // SAFETY: we are sending a pointer to a MaybeUninit<T>, which is valid as long as we don't use it again
        let ptr = Box::into_raw(value) as *mut ();
        if sender.send(ptr).is_err() {
            // If the channel is closed, we can leak to make sure we don't double drop
            // unsafe { std::ptr::drop_in_place(ptr) };
        }
    }
}
