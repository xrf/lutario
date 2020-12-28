//! Generalizes the idea of vector operations to enable transparent
//! parallelization.
//!
//! A vector driver is a device that understands how to perform
//! map-reduce-like operations on its own family of vector-like objects.  It
//! is the interface used by the sg-ode solver and is quite powerful, albeit
//! awkward to use.  The `vec_apply!` macro helps a bit.
//!
//! The documentation here is very incomplete.  There is a bit more
//! information in the [docs of the sg-ode vector driver interface in
//! C](https://xrf.github.io/sg-ode/vector_8h.html), after which the Rust
//! interface is modelled.
//!
//! The MPI vector driver will be implemented some day, but for now you can
//! only use the [`basic`](basic/index.html) vector driver.  There is nothing
//! difficult about the MPI implementation – proof of concepts have already
//! been made.  The main concern is that it adds a really heavy dependency
//! (MPI) to Lutario, so it probably should be done in a separate crate.

pub mod basic;
pub mod c;

use std::{fmt, iter};
use std::ops::Deref;
use serde::{Deserialize, Serialize};

pub fn assert_all_eq<I: IntoIterator>(xs: I, msg: &str) -> Option<I::Item>
    where I::Item: fmt::Debug + Eq
{
    let mut xs = xs.into_iter();
    let x0 = match xs.next() {
        None => return None,
        Some(x0) => x0
    };
    for x in xs {
        assert_eq!(x, x0, "{}", msg);
    }
    Some(x0)
}

pub trait VectorDriver {
    type Item;
    type Vector;

    fn len(&self) -> usize;

    fn create_vector_with<F>(&self, f: F) -> Option<Self::Vector>
        where F: Fn() -> Self::Item + Sync;

    /// Performs an applicative operation on multiple vectors.
    ///
    /// This is the fundamental operation of vector drviers: it allows an
    /// arbitrary operation to be applied element-wise to multiple vectors
    /// with low overhead.
    ///
    /// Conceptually, you can think of this as a map-reduce-like operation,
    /// but with an ugly interface for efficiency reasons.
    fn operate<F>(
        &self,
        accum: &mut [u8],
        offset: usize,
        vectors: &[&Self::Vector],
        mut_vectors: &mut [&mut Self::Vector],
        f: F,
    ) where
        F: Fn(&mut [u8],
              &[u8],
              usize,
              &[&[Self::Item]],
              &mut [&mut [Self::Item]]) + Sync;

    fn create_vector(&self, value: Self::Item) -> Option<Self::Vector>
        where Self::Item: Clone + Sync
    {
        self.create_vector_with(|| value.clone())
    }

    /// Note: The accumulator must serialize to a fixed length.
    fn operate_on<F, S>(
        &self,
        accum: &mut S,
        offset: usize,
        vectors: &[&Self::Vector],
        mut_vectors: &mut [&mut Self::Vector],
        f: F,
    ) where
        F: Fn(&mut S,
              S,
              usize,
              &[&[Self::Item]],
              &mut [&mut [Self::Item]]) + Sync,
        S: Serialize + for<'a> Deserialize<'a>,
    {
        use bincode::{deserialize, serialize, serialize_into};
        let mut accum_buf = serialize(accum).unwrap();
        self.operate(
            &mut accum_buf,
            offset,
            vectors,
            mut_vectors,
            |mut accum_buf, val_buf, offset, slices, mut_slices| {
                let mut accum = deserialize(accum_buf).unwrap();
                let val = deserialize(val_buf).unwrap();
                f(&mut accum, val, offset, slices, mut_slices);
                serialize_into(&mut accum_buf, &accum).unwrap();
            },
        );
        *accum = deserialize(&accum_buf).unwrap();
    }

    /// Sum all elements of the vector.
    ///
    /// This is a demonstration of the flexibility of vector driver.
    fn sum(&self, v: &Self::Vector) -> Self::Item where
        Self::Item: for<'a> iter::Sum<&'a Self::Item>
                  + Serialize + for<'a> Deserialize<'a> + Clone
    {
        let zero = || -> Self::Item { [].iter().sum() };
        let mut sum = zero();
        self.operate_on(
            &mut sum,
            0,
            &[&v],
            &mut [],
            |accum: &mut Self::Item, val, _, a: &[&_], _: &mut [&mut _]| {
                *accum = iter::once(&*accum)
                    .chain(iter::once(&val))
                    .chain(a[0]).sum();
            });
        sum
    }
}

impl<T> VectorDriver for T where
    T: Deref,
    T::Target: VectorDriver,
{
    type Item = <T::Target as VectorDriver>::Item;
    type Vector = <T::Target as VectorDriver>::Vector;

    fn len(&self) -> usize {
        (**self).len()
    }

    fn create_vector_with<F>(&self, f: F) -> Option<Self::Vector>
        where F: Fn() -> Self::Item + Sync
    {
        (**self).create_vector_with(f)
    }

    fn operate<F>(&self,
                  accum: &mut [u8],
                  offset: usize,
                  vectors: &[&Self::Vector],
                  mut_vectors: &mut [&mut Self::Vector],
                  f: F)
        where F: Fn(&mut [u8],
                    &[u8],
                    usize,
                    &[&[Self::Item]],
                    &mut [&mut [Self::Item]]) + Sync
    {
        (**self).operate(accum, offset, vectors, mut_vectors, f)
    }
}
