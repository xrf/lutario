use std::ptr;
use blas;
use blas::c::{Layout, Transpose};
use num::{Complex, Zero};
use utils::cast;

/// A simple matrix wrapper that stores data in `D`.
///
/// The indexing convention here is column-major, but to avoid ambiguity we
/// simply call them “fast” (`dim0`) and “slow” (`dim1`) indices rather than
/// “row” and “column” indices.
#[derive(Clone, Copy, Debug)]
pub struct Mat<D> {
    pub data: D,
    /// The separation between the segments (“columns”).
    pub stride: usize,
    /// The fast index.
    pub dim0: usize,
    /// The slow index.
    pub dim1: usize,
}

pub type MatRef<'a, T> = Mat<&'a [T]>;

pub type MatMut<'a, T> = Mat<&'a mut [T]>;

pub type MatOwned<T> = Mat<Box<[T]>>;

impl<T: PartialEq, D: Deref<Target=[T]>> PartialEq for Mat<D> {
    fn eq(&self, other: &Self) -> bool {
        if self.dim0 != other.dim0 {
            return false;
        }
        if self.dim1 != other.dim1 {
            return false;
        }
        for j in 0 .. self.dim1 {
            for i in 0 .. self.dim0 {
                if !self.get(i, j).eq(other.get(i, j)) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Eq, D: Deref<Target=[T]>> Eq for Mat<D> {}

impl<T> From<Vec<Vec<T>>> for MatOwned<T> {
    fn from(rows: Vec<Vec<T>>) -> Self {
        let dim0 = rows.len();
        let dim1 = if dim0 == 0 { 0 } else { rows[0].len() };
        let mut data = Vec::with_capacity(dim0 * dim1);
        for mut row in rows {
            data.extend(row.drain(..));
        }
        Self::from_vec(dim0, dim1, data).into_transpose()
    }
}

impl<T> MatOwned<T> {
    /// Note: this expects a vector in column-major format!
    pub fn from_vec(dim0: usize, dim1: usize, mut data: Vec<T>) -> Self {
        data.truncate(dim0 * dim1);
        Mat {
            data: data.into_boxed_slice(),
            stride: dim0,
            dim0,
            dim1,
        }
    }

    pub fn into_transpose(self) -> MatOwned<T> {
        let n = self.dim0 * self.dim1;
        let mut data = Vec::with_capacity(n);
        unsafe {
            data.set_len(n);
            let mut m = MatOwned::from_vec(self.dim1, self.dim0, data);
            for j in 0 .. self.dim1 {
                for i in 0 .. self.dim0 {
                    ptr::write(m.get_mut(j, i), ptr::read(self.get(i, j)));
                }
            }
            // forget the data
            self.data.into_vec().set_len(0);
            m
        }
    }
}

impl<T: Clone> MatOwned<T> {
    pub fn repeat(dim0: usize, dim1: usize, value: T) -> Self {
        Self::from_vec(dim0, dim1, vec![value; dim0 * dim1])
    }
}

impl<T: Clone + Zero> MatOwned<T> {
    pub fn zero(dim0: usize, dim1: usize) -> Self {
        Self::repeat(dim0, dim1, Zero::zero())
    }
}

impl<D> Mat<D> {
    pub fn raw_index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.dim0);
        debug_assert!(j < self.dim1);
        i + self.stride * j
    }

    pub fn transpose_dims(&self, trans: Transpose) -> (usize, usize) {
        if trans == Transpose::None {
            (self.dim0, self.dim1)
        } else {
            (self.dim1, self.dim0)
        }
    }
}

impl<T, D: Deref<Target=[T]>> Mat<D> {
    pub fn as_mat_ref(&self) -> MatRef<T> {
        Mat {
            data: &*self.data,
            stride: self.stride,
            dim0: self.dim0,
            dim1: self.dim1,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> &T {
        let k = self.raw_index(i, j);
        &self.data[k]
    }

    pub fn validate(&self) {
        assert!(self.data.len() >= self.stride * self.dim1);
    }
}

impl<T, D: DerefMut<Target=[T]>> Mat<D> {
    pub fn as_mat_mut(&mut self) -> MatMut<T> {
        Mat {
            data: &mut *self.data,
            stride: self.stride,
            dim0: self.dim0,
            dim1: self.dim1,
        }
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        let k = self.raw_index(i, j);
        &mut self.data[k]
    }
}

use std::ops::{Deref, DerefMut};

pub trait Blas: Copy {
    unsafe fn unsafe_gemm(layout: Layout,
                          transa: Transpose,
                          transb: Transpose,
                          m: i32,
                          n: i32,
                          k: i32,
                          alpha: Self,
                          a: &[Self],
                          lda: i32,
                          b: &[Self],
                          ldb: i32,
                          beta: Self,
                          c: &mut [Self],
                          ldc: i32);

    /// A thin wrapper over `unsafe_gemm` that panics if the buffers are too
    /// small or if the sizes don't match.  We omit `Layout` because it can be
    /// trivially emulated by exchanging `a` and `b`.
    fn gemm(transa: Transpose,
            transb: Transpose,
            alpha: Self,
            a: Mat<&[Self]>,
            b: Mat<&[Self]>,
            beta: Self,
            c: Mat<&mut [Self]>) {
        a.validate();
        b.validate();
        c.validate();
        let (ma, ka) = a.transpose_dims(transa);
        let (kb, nb) = b.transpose_dims(transb);
        let mc = c.dim0;
        let nc = c.dim1;
        assert_eq!(ma, mc);
        assert_eq!(nb, nc);
        assert_eq!(ka, kb);
        // FIXME: handle integer overflow for very large matrices
        unsafe {
            Self::unsafe_gemm(Layout::ColumnMajor, transa, transb,
                              cast(ma), cast(nb), cast(ka),
                              alpha, a.data, cast(a.stride),
                              b.data, cast(b.stride),
                              beta, c.data, cast(c.stride));
        }
    }
}

impl Blas for f32 {
    unsafe fn unsafe_gemm(layout: Layout,
                          transa: Transpose,
                          transb: Transpose,
                          m: i32,
                          n: i32,
                          k: i32,
                          alpha: Self,
                          a: &[Self],
                          lda: i32,
                          b: &[Self],
                          ldb: i32,
                          beta: Self,
                          c: &mut [Self],
                          ldc: i32) {
        blas::c::sgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl Blas for f64 {
    unsafe fn unsafe_gemm(layout: Layout,
                          transa: Transpose,
                          transb: Transpose,
                          m: i32,
                          n: i32,
                          k: i32,
                          alpha: Self,
                          a: &[Self],
                          lda: i32,
                          b: &[Self],
                          ldb: i32,
                          beta: Self,
                          c: &mut [Self],
                          ldc: i32) {
        blas::c::dgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl Blas for Complex<f32> {
    unsafe fn unsafe_gemm(layout: Layout,
                          transa: Transpose,
                          transb: Transpose,
                          m: i32,
                          n: i32,
                          k: i32,
                          alpha: Self,
                          a: &[Self],
                          lda: i32,
                          b: &[Self],
                          ldb: i32,
                          beta: Self,
                          c: &mut [Self],
                          ldc: i32) {
        blas::c::cgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl Blas for Complex<f64> {
    unsafe fn unsafe_gemm(layout: Layout,
                          transa: Transpose,
                          transb: Transpose,
                          m: i32,
                          n: i32,
                          k: i32,
                          alpha: Self,
                          a: &[Self],
                          lda: i32,
                          b: &[Self],
                          ldb: i32,
                          beta: Self,
                          c: &mut [Self],
                          ldc: i32) {
        blas::c::zgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let a = Mat::from(vec![vec![1.0, 2.0],
                               vec![3.0, 4.0]]);
        let b = Mat::from(vec![vec![5.0, 6.0],
                               vec![7.0, 8.0]]);
        let c0 = Mat::from(vec![vec![-1.0, -2.0],
                                vec![-3.0, -4.0]]);

        let mut c = c0.clone();
        f64::gemm(Transpose::None, Transpose::None,
                  2.0, a.as_mat_ref(), b.as_mat_ref(), 3.0, c.as_mat_mut());
        assert_eq!(c, Mat::from(vec![vec![35.0, 38.0],
                                     vec![77.0, 88.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::Ordinary, Transpose::None,
                  2.0, a.as_mat_ref(), b.as_mat_ref(), 3.0, c.as_mat_mut());
        assert_eq!(c, Mat::from(vec![vec![49.0, 54.0],
                                     vec![67.0, 76.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::None, Transpose::Ordinary,
                  2.0, a.as_mat_ref(), b.as_mat_ref(), 3.0, c.as_mat_mut());
        assert_eq!(c, Mat::from(vec![vec![31.0, 40.0],
                                     vec![69.0, 94.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::Ordinary, Transpose::Ordinary,
                  2.0, a.as_mat_ref(), b.as_mat_ref(), 3.0, c.as_mat_mut());
        assert_eq!(c, Mat::from(vec![vec![43.0, 56.0],
                                     vec![59.0, 80.0]]));
    }
}
