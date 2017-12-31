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
        use bincode::{Infinite, deserialize, serialize, serialize_into};
        let mut accum_buf = serialize(accum, Infinite).unwrap();
        self.operate(
            &mut accum_buf,
            offset,
            vectors,
            mut_vectors,
            |mut accum_buf, val_buf, offset, slices, mut_slices| {
                let mut accum = deserialize(accum_buf).unwrap();
                let val = deserialize(val_buf).unwrap();
                f(&mut accum, val, offset, slices, mut_slices);
                serialize_into(&mut accum_buf, &accum, Infinite).unwrap();
            },
        );
        *accum = deserialize(&accum_buf).unwrap();
    }

    fn sum(&self, v: &Self::Vector) -> Self::Item where
        Self::Item: for<'a> iter::Sum<&'a Self::Item>
                  + Serialize + for<'a> Deserialize<'a> + Clone
    {
        let zero = || -> Self::Item { [].into_iter().sum() };
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
