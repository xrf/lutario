use std::marker::PhantomData;
use super::{VectorDriver, assert_all_eq};

#[derive(Clone, Copy, Debug)]
pub struct BasicVectorDriver<T> {
    len: usize,
    phantom: PhantomData<(T, fn(T))>,
}

impl<T> BasicVectorDriver<T> {
    pub fn new(len: usize) -> Self {
        Self { len, phantom: PhantomData }
    }

    /// Shortcut for creating a vector from a slice.
    pub fn create_vector_from(&self, xs: &[T]) -> Vec<T>
        where T: Clone
    {
        assert_eq!(self.len(), xs.len(),
                   "slice must have same length as BasicVectorDriver");
        xs.to_owned()
    }
}

impl<T> VectorDriver for BasicVectorDriver<T> {
    type Item = T;
    type Vector = Vec<Self::Item>;

    fn create_vector_with<F>(&self, f: F) -> Option<Self::Vector>
        where F: Fn() -> Self::Item + Sync
    {
        let n = self.len();
        let mut v = Vec::with_capacity(n);
        // testing shows unsafe is needed for it to be optimized to memset
        unsafe {
            v.set_len(n);
            for i in 0 .. n {
                *v.get_unchecked_mut(i) = f();
            }
        }
        Some(v)
    }

    fn len(&self) -> usize {
        self.len
    }

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
              &mut [&mut [Self::Item]]) + Sync
    {
        assert_all_eq([self.len()].iter().cloned()
                      .chain(vectors.iter().map(|v| v.len()))
                      .chain(mut_vectors.iter().map(|v| v.len())),
                      "every vector must have length equal to \
                       that of BasicVectorDriver");
        let slices: Vec<_> = vectors.into_iter()
            .map(|v| v.as_ref()).collect();
        let mut mut_slices: Vec<_> = mut_vectors.into_iter()
            .map(|v| v.as_mut()).collect();
        let identity = accum.to_owned();
        f(accum,
          &identity,
          offset,
          &slices,
          &mut mut_slices);
    }
}

#[test]
fn test_basic_vector_driver() {
    let d = BasicVectorDriver::new(42);

    let ref mut v = d.create_vector(0.0).unwrap();
    let ref mut u = d.create_vector(0.0).unwrap();

    vec_apply! { for (mut v) in d { *v = 1.0; } };

    assert_eq!(d.sum(&v), d.len() as f64);

    d.operate_on(
        &mut (),
        0,
        &[],
        &mut [v],
        |_: &mut _, _, offset, _: &[&[f64]], r: &mut [&mut [f64]]| {
            for (i, vi) in r[0].iter_mut().enumerate() {
                *vi = (offset + i) as _;
            }
        });
    let sum = (d.len() * (d.len() - 1) / 2) as f64;
    assert_eq!(d.sum(&v), sum);

    vec_apply! { for (v, mut u) in d { *u = *v; } };
    assert_eq!(d.sum(&u), sum);
    assert_eq!(d.sum(&v), sum);

    vec_apply! { for (v, mut u) in d { *u = *v; } };
}
