use std::mem::ManuallyDrop;

type PhantomInvariantLifetime<'a> = std::marker::PhantomData<fn(&'a mut ()) -> &'a mut ()>;

trait MapApi<K, QueryKey, V> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;

    fn clear(&mut self);

    fn get(&self, key: &QueryKey) -> Option<&V>;
    fn get_mut(&mut self, key: &QueryKey) -> Option<&mut V>;
    fn insert(&mut self, key: K, value: V) -> Option<V>;
    fn remove(&mut self, key: &QueryKey) -> Option<V>;

    type Iter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;
    type IterMut<'a>: Iterator<Item = (&'a K, &'a mut V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;
    type IntoIter: Iterator<Item = (K, V)>;
    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    fn into_iter(self) -> Self::IntoIter;

    type Keys<'a>: Iterator<Item = &'a K>
    where
        Self: 'a,
        K: 'a;
    type Values<'a>: Iterator<Item = &'a V>
    where
        Self: 'a,
        V: 'a;
    type ValuesMut<'a>: Iterator<Item = &'a mut V>
    where
        Self: 'a,
        V: 'a;
    fn keys(&self) -> Self::Keys<'_>;
    fn values(&self) -> Self::Values<'_>;
    fn values_mut(&mut self) -> Self::ValuesMut<'_>;

    type Entry<'a>: EntryApi<'a, K, V>
    where
        Self: 'a,
        K: 'a,
        V: 'a;
    fn entry(&mut self, key: K) -> Self::Entry<'_>;
}

trait EntryApi<'a, K, V> {
    fn or_insert(&mut self, default: V) -> &'a mut V;
    fn or_insert_with<F: FnOnce() -> V>(&mut self, default: F) -> &'a mut V;
    fn remove_entry(self) -> Option<(K, V)>;

    fn key(&self) -> &K;

    type VacantEntry<'b>: VacantEntryApi<'b, K, V>
    where
        Self: 'b,
        K: 'b,
        V: 'b;
    type OccupiedEntry<'b>: OccupiedEntryApi<'b, K, V>
    where
        Self: 'b,
        K: 'b,
        V: 'b;
    fn into_vacant(self) -> Option<Self::VacantEntry<'a>>;
    fn into_occupied(self) -> Option<Self::OccupiedEntry<'a>>;
}

trait OccupiedEntryApi<'a, K, V> {
    fn get(&self) -> &V;
    fn get_mut(&mut self) -> &mut V;
    fn remove(self) -> V;
    fn remove_entry(self) -> (K, V);
}

trait VacantEntryApi<'a, K, V> {
    fn insert(self, value: V) -> &'a mut V;
    fn key(&self) -> &K;
}

#[repr(C)]
struct VacantEntryApiVtable<'a, K, V> {
    insert_fn: unsafe extern "C" fn(*mut (), value: V) -> *mut V,
    key_fn: unsafe extern "C" fn(*const ()) -> *const K,
    drop_fn: unsafe extern "C" fn(*mut ()),
    _marker: PhantomInvariantLifetime<'a>,
}

fn make_vacant_entry_api_vtable<'a, K, V, T: VacantEntryApi<'a, K, V>>(
) -> VacantEntryApiVtable<'a, K, V>
where
    K: 'a,
    V: 'a,
    T: 'a,
{
    unsafe extern "C" fn insert_impl<'b, K: 'b, V: 'b, T: 'b + VacantEntryApi<'b, K, V>>(
        this: *mut (),
        value: V,
    ) -> *mut V {
        let this = std::ptr::read(this as *mut T);
        this.insert(value) as *mut V
    }
    unsafe extern "C" fn key_impl<'b, K, V, T: VacantEntryApi<'b, K, V>>(
        this: *const (),
    ) -> *const K {
        let this = &*(this as *const T);
        this.key() as *const K
    }
    unsafe extern "C" fn drop_impl<'b, K, V, T: VacantEntryApi<'b, K, V>>(this: *mut ()) {
        std::ptr::drop_in_place(this as *mut T)
    }
    VacantEntryApiVtable {
        insert_fn: insert_impl::<'a, K, V, T>,
        key_fn: key_impl::<'a, K, V, T>,
        drop_fn: drop_impl::<'a, K, V, T>,
        _marker: PhantomInvariantLifetime::<'a>::default(),
    }
}
struct FfiVacantEntryApi<'a, K, V> {
    entry_ptr: *mut (),
    vtable: &'a VacantEntryApiVtable<'a, K, V>,
    _marker: PhantomInvariantLifetime<'a>,
}

#[repr(C)]
pub enum FfiEntry<'a, K, V> {
    Vacant(Box<dyn VacantEntryApi<'a, K, V>>),
    Occupied(Box<dyn OccupiedEntryApi<'a, K, V>>),
}
