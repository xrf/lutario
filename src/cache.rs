//! Insert-only caching of arbitrary data
//!
//! Experimental module, not yet used for anything.

use any_key::AnyOrd;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::{cmp, mem};

pub trait CacheKey: Ord + 'static {
    type Value: 'static;
    fn get(&self) -> Self::Value;
}

/// Implementation detail for `Cache`.
#[derive(Clone, Copy, Debug)]
pub struct CacheEntry<K, V> {
    pub key: K,
    pub value: Option<V>,
}

impl<K: PartialEq, V> PartialEq for CacheEntry<K, V> {
    fn eq(&self, other: &CacheEntry<K, V>) -> bool {
        self.key.eq(&other.key)
    }
}

impl<K: Eq, V> Eq for CacheEntry<K, V> {}

impl<K: Hash, V> Hash for CacheEntry<K, V> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.key.hash(hasher)
    }
}

impl<K: PartialOrd, V> PartialOrd for CacheEntry<K, V> {
    fn partial_cmp(&self, other: &CacheEntry<K, V>) -> Option<cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl<K: Ord, V> Ord for CacheEntry<K, V> {
    fn cmp(&self, other: &CacheEntry<K, V>) -> cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// An associative container where entries can never be removed.  Therefore,
/// references to its entries remain valid as long as the cache is not
/// destroyed.  Arbitrary types are allowed.
#[derive(Debug)]
pub struct Cache(RefCell<BTreeSet<Box<dyn AnyOrd>>>);

impl Default for Cache {
    fn default() -> Self {
        Cache(Default::default())
    }
}

impl Cache {
    pub fn get<K: CacheKey>(&self, key: K) -> &K::Value {
        // unsafety:
        //
        //   - extending the lifetimes: okay because we never delete from cache
        //   - downcast_ref_unchecked: the cache is implicitly keyed by TypeId
        //     and we trust AnyOrd to be correct (don't need to trust Ord for K)
        //
        unsafe {
            let query: CacheEntry<_, K::Value> = CacheEntry { key, value: None };
            // we don't use the entry API here
            // because we want to unlock the RefCell while
            // CacheKey::get is being executed
            if let Some(v) = self.0.borrow_mut().get(&query as &dyn AnyOrd) {
                let v: &dyn AnyOrd = &**v;
                let r: &dyn AnyOrd = mem::transmute(v);
                let r: &CacheEntry<K, _> = r.downcast_ref_unchecked();
                return r
                    .value
                    .as_ref()
                    .expect("stored CacheEntrys should never have None");
            }
            let key = query.key;
            let value = Some(key.get());
            let entry = Box::new(CacheEntry { key, value });
            let r = mem::transmute(
                entry
                    .value
                    .as_ref()
                    .expect("stored CacheEntrys should never have None"),
            );
            self.0.borrow_mut().replace(entry);
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct Foo(&'static str);

    impl CacheKey for Foo {
        type Value = String;
        fn get(&self) -> Self::Value {
            ":".to_owned() + self.0
        }
    }

    #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct Bar(i64);

    impl CacheKey for Bar {
        type Value = f64;
        fn get(&self) -> Self::Value {
            (self.0 as f64).sqrt()
        }
    }

    #[test]
    fn test() {
        let cache = Cache::default();

        assert_eq!(cache.get(Foo("3")), ":3");
        assert_eq!(cache.get(Foo("p")), ":p");
        assert_eq!(cache.get(Bar(16)), &4.0);
        assert_eq!(cache.get(Bar(64)), &8.0);

        assert_eq!(cache.get(Foo("3")), ":3");
        assert_eq!(cache.get(Foo("p")), ":p");
        assert_eq!(cache.get(Bar(16)), &4.0);
        assert_eq!(cache.get(Bar(64)), &8.0);
    }
}
