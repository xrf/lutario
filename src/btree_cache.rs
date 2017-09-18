use std::mem;
use std::cell::RefCell;
use std::collections::BTreeMap;
use stable_deref_trait::StableDeref;

/// Works like a `BTreeMap`, but elements can never be removed.  Therefore,
/// stable references (i.e. `StableDeref`) to its elements remain valid as
/// long as the cache is not destroyed.
#[derive(Debug)]
pub struct BTreeCache<K, V>(RefCell<BTreeMap<K, V>>);

impl<K: Ord, V> Default for BTreeCache<K, V> {
    fn default() -> Self {
        BTreeCache(Default::default())
    }
}

impl<K: Ord, V: StableDeref> BTreeCache<K, V> {
    pub fn get_or_insert_with<F>(&self, key: K, f: F) -> &V::Target
        where F: FnOnce(&K) -> V
    {
        // we don't use the entry API here
        // because we want to unlock the RefCell while
        // f is being executed
        if let Some(v) = self.0.borrow_mut().get(&key) {
            return unsafe { mem::transmute(v.deref()) };
        }
        let v = f(&key);
        let r = unsafe { mem::transmute(v.deref()) };
        self.0.borrow_mut().insert(key, v);
        r
    }
}
