//! Reference-counted caching of arbitrary data
//!
//! Experimental module, not yet used for anything.

use std::{cmp, fmt};
use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, Weak};
use any_key::AnyHash;

pub trait Key: Hash + Eq + Send + Sync + 'static {
    type Value: Send + Sync + 'static;
    fn get(&self) -> Self::Value;
}

#[derive(Clone)]
struct Dropper {
    map: Weak<Mutex<CacheInner>>,
    key: Weak<AnyHash + Send + Sync>,
}

impl Drop for Dropper {
    fn drop(&mut self) {
        // this can may happen after the Arc has already been dropped, in
        // which case someone else might be waiting on the Condvar so we must
        // awaken them
        if let Some(map) = self.map.upgrade() {
            if let Some(key) = self.key.upgrade() {
                if let Some(CacheValue { condvar, .. })
                    = map.lock().unwrap().remove(&key)
                {
                    condvar.notify_all();
                }
            }
        }
    }
}

/// A cached value.
pub struct Cached<V>(Arc<CachedInner<V>>);
// we need a wrapper type over Arc<CachedInner<V>> to prevent the user from
// try_unwrapping and then holding onto the CachedInner object, which can
// cause Cache::get to block forever

impl<V> Clone for Cached<V> {
    fn clone(&self) -> Self {
        Cached(self.0.clone())
    }
}

impl<V: fmt::Debug> fmt::Debug for Cached<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Cached")
            .field(&self.0.value)
            .finish()
    }
}

impl<V: PartialEq> PartialEq for Cached<V> {
    fn eq(&self, other: &Cached<V>) -> bool {
        (**self).eq(&**other)
    }
}

impl<V: Eq> Eq for Cached<V> {}

impl<V: Hash> Hash for Cached<V> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (**self).hash(hasher)
    }
}

impl<V: PartialOrd> PartialOrd for Cached<V> {
    fn partial_cmp(&self, other: &Cached<V>) -> Option<cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<V: Ord> Ord for Cached<V> {
    fn cmp(&self, other: &Cached<V>) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<V> Deref for Cached<V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.0.value
    }
}

impl<V> Cached<V> {
    pub fn try_unwrap(this: Self) -> Result<V, Self> {
        match Arc::try_unwrap(this.0) {
            Ok(x) => Ok(x.value),
            Err(x) => Err(Cached(x)),
        }
    }
}

struct CachedInner<V> {
    value: V,
    _dropper: Dropper,
}

struct CacheValue {
    value: Box<Any + Send + Sync>,      // ~ Box<Weak<CachedInner<V>>>
    condvar: Arc<Condvar>,
}

type CacheInner = HashMap<Arc<AnyHash + Send + Sync>, CacheValue>;

/// An associative array that weakly owns its elements and can store entries
/// of arbitrary type.
#[derive(Clone)]
pub struct Cache(Arc<Mutex<CacheInner>>);

impl fmt::Debug for Cache {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Cache").finish()
    }
}

impl Default for Cache {
    fn default() -> Self {
        Cache(Default::default())
    }
}

impl Cache {
    /// Returns the number of active elements.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// Retrieve the value with the given key or, if it's not yet cached,
    /// compute the value.
    pub fn get<K: Key + Clone>(&self, key: &K) -> Cached<K::Value> {
        let qkey = key as &(AnyHash + Send + Sync);
        let wkey = {
            let mut map = self.0.lock().unwrap();
            loop {
                let condvar = match map.get(qkey) {
                    Some(&CacheValue { ref value, ref condvar }) => {
                        let value: &(Any + Send) = &**value;
                        let value = value.downcast_ref()
                            .expect("value has wrong type");
                        if let Some(value) = Weak::upgrade(value) {
                            return Cached(value);
                        }
                        condvar.clone()
                    }
                    None => break,
                };
                map = condvar.wait(map).unwrap();
            }
            map.remove(qkey);          // make sure the key object is replaced
            let key = Arc::new(key.clone());
            let wkey = Arc::downgrade(&key);
            let condvar = Arc::new(Condvar::new());
            map.insert(key, CacheValue {
                value: Box::new(Weak::<CachedInner<K::Value>>::new()),
                condvar,
            });
            wkey
        };
        let value = Key::get(key);
        let value = Arc::new(CachedInner {
            value,
            _dropper: Dropper {
                map: Arc::downgrade(&self.0),
                key: wkey,
            },
        });
        let mut map = self.0.lock().unwrap();
        let &mut CacheValue { value: ref mut rval, ref condvar } =
            map.get_mut(qkey).expect("entry vanished");
        *rval = Box::new(Arc::downgrade(&value));
        condvar.notify_all();
        Cached(value)
    }
}

#[cfg(test)]
mod tests {
    use std::{mem, thread};
    use std::time::Duration;
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Foo(&'static str);

    impl Key for Foo {
        type Value = String;
        fn get(&self) -> Self::Value {
            thread::sleep(Duration::new(0, 1000));
            ":".to_owned() + self.0
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Bar(i64);

    impl Key for Bar {
        type Value = f64;
        fn get(&self) -> Self::Value {
            (self.0 as f64).sqrt()
        }
    }

    #[test]
    fn test() {
        let cache = Cache::default();
        {
            let foo3 = cache.get(&Foo("3"));
            assert_eq!(*foo3, ":3");
            mem::forget(foo3);
            assert_eq!(cache.len(), 1);
            assert_eq!(*cache.get(&Foo("p")), ":p");
            let food = cache.get(&Foo("D"));
            assert_eq!(*food, ":D");
            assert_eq!(*cache.get(&Foo("]")), ":]");
            assert_eq!(cache.len(), 2);
            let bar16 = cache.get(&Bar(16));
            assert_eq!(*bar16, 4.0);
            assert_eq!(*cache.get(&Bar(64)), 8.0);
            assert_eq!(*cache.get(&Foo("3")), ":3");
            assert_eq!(*cache.get(&Foo("p")), ":p");
            assert_eq!(*cache.get(&Foo("D")), ":D");
            assert_eq!(*cache.get(&Foo("]")), ":]");
            assert_eq!(*cache.get(&Bar(16)), 4.0);
            assert_eq!(*cache.get(&Bar(64)), 8.0);
            assert_eq!(cache.len(), 3);
        }
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_threaded() {
        let cache = Cache::default();

        let foo3 = cache.get(&Foo("3"));
        assert_eq!(*foo3, ":3");
        mem::forget(foo3);

        let mut children = Vec::default();
        for _ in 0 .. 16 {
            let cache = cache.clone();
            children.push(thread::spawn(move || {
                for _ in 0 .. 16 {
                    assert_eq!(*cache.get(&Foo("p")), ":p");
                    let food = cache.get(&Foo("D"));
                    assert_eq!(*food, ":D");
                    assert_eq!(*cache.get(&Foo("]")), ":]");

                    assert_eq!(*cache.get(&Foo("3")), ":3");
                    assert_eq!(*cache.get(&Foo("p")), ":p");
                    assert_eq!(*cache.get(&Foo("D")), ":D");
                    assert_eq!(*cache.get(&Foo("]")), ":]");
                }
            }));
        }
        for child in children {
            child.join().unwrap();
        }
    }
}
