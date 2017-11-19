//! Linear algebra.
use std::mem;
use std::cmp::max;
use std::ops::{Add, Mul, Neg, Range};
use cblas;
use lapacke;
use num::{Complex, Num};
use super::matrix::{Mat, MatMut};
use super::utils::{self, RangeInclusive, cast};

pub use cblas::{Part, Transpose};

pub mod lamch {
    //! Defines and re-exports some floating-point constants following the
    //! LAPACK convention.  Basically, this module contains anything you would
    //! otherwise obtain using `lamch`.

    pub mod f32 {
        pub use std::f32::RADIX as BASE;
        pub use std::f32::MANTISSA_DIGITS as T;
        pub use std::f32::EPSILON as PREC;
        pub use std::f32::MIN_POSITIVE as RMIN;
        pub use std::f32::MAX as RMAX;
        pub use std::f32::MIN_EXP as EMIN;
        pub use std::f32::MAX_EXP as EMAX;

        /// Relative machine epsilon according to the LAPACK convention.
        /// Equal to half of `std::*::EPSILON`.
        pub const EPS: f32 = PREC / (BASE as f32);

        /// Whether proper rounding (`true`) or chopping (`false`) occurs in
        /// addition.
        pub const RND: bool = true;

        /// Safe minimum such that `1.0 / SFMIN` does not overflow.
        pub const SFMIN: f32 =
        // workaround because Rust doesn't support if-else in const-expressions
            (1.0 / RMAX >= RMIN) as i32 as f32 * (1.0 / RMAX * (1.0 + EPS)) +
            (1.0 / RMAX < RMIN) as i32 as f32 * RMIN;
    }

    pub mod f64 {
        pub use std::f64::RADIX as BASE;
        pub use std::f64::MANTISSA_DIGITS as T;
        pub use std::f64::EPSILON as PREC;
        pub use std::f64::MIN_POSITIVE as RMIN;
        pub use std::f64::MAX as RMAX;
        pub use std::f64::MIN_EXP as EMIN;
        pub use std::f64::MAX_EXP as EMAX;

        /// Relative machine epsilon according to the LAPACK convention.
        /// Equal to half of `std::*::EPSILON`.
        pub const EPS: f64 = PREC / (BASE as f64);

        /// Whether proper rounding (`true`) or chopping (`false`) occurs in
        /// addition.
        pub const RND: bool = true;

        /// Safe minimum such that `1.0 / SFMIN` does not overflow.
        pub const SFMIN: f64 =
        // workaround because Rust doesn't support if-else in const-expressions
            (1.0 / RMAX >= RMIN) as i32 as f64 * (1.0 / RMAX * (1.0 + EPS)) +
            (1.0 / RMAX < RMIN) as i32 as f64 * RMIN;
    }
}

pub trait Conj {
    fn conj(&self) -> Self;
}

impl Conj for f32 {
    fn conj(&self) -> Self {
        *self
    }
}

impl Conj for f64 {
    fn conj(&self) -> Self {
        *self
    }
}

impl<T: Clone + Num + Neg<Output = T>> Conj for Complex<T> {
    fn conj(&self) -> Self {
        self.conj()
    }
}

pub trait NormSqr {
    type Real: PartialOrd;

    fn norm_sqr(&self) -> Self::Real;
}

impl NormSqr for f32 {
    type Real = Self;

    fn norm_sqr(&self) -> Self::Real {
        f32::abs(*self)
    }
}

impl NormSqr for f64 {
    type Real = Self;

    fn norm_sqr(&self) -> Self::Real {
        f64::abs(*self)
    }
}

impl<T: Num + PartialOrd + Clone> NormSqr for Complex<T> {
    type Real = T;

    fn norm_sqr(&self) -> Self::Real {
        Complex::norm_sqr(self)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdjSym {
    conjugate: bool,
    negate: bool,
}

impl AdjSym {
    pub fn apply<T: Conj + Neg<Output = T>>(self, mut x: T) -> T {
        x = if self.conjugate { x.conj() } else { x };
        x = if self.negate { -x } else { x };
        x
    }
}

pub const HERMITIAN: AdjSym = AdjSym { conjugate: true, negate: false };
pub const ANTIHERMITIAN: AdjSym = AdjSym { conjugate: true, negate: true };
pub const SYMMETRIC: AdjSym = AdjSym { conjugate: false, negate: false };
pub const ANTISYMMETRIC: AdjSym = AdjSym { conjugate: false, negate: true };

/// Desired range of eigenvalues.
#[derive(Clone, Debug)]
pub enum EigenvalueRange<T> {
    All,
    /// Half-open range of desired eigenvalues.
    Values(Range<T>),
    /// 1-indexed indices of the desired eigenvalues in ascending order.
    Indices(RangeInclusive<i32>),
}

impl<T> Default for EigenvalueRange<T> {
    fn default() -> Self {
        EigenvalueRange::All
    }
}

impl<T: PartialOrd> EigenvalueRange<T> {
    pub fn to_raw(
        self,
        n: i32,
        vl: &mut T,
        vu: &mut T,
        il: &mut i32,
        iu: &mut i32,
    ) -> (u8, bool, i32) {
        match self {
            EigenvalueRange::All => (b'A', true, n),
            EigenvalueRange::Values(Range { start, end }) => {
                assert!(start < end);
                *vl = start;
                *vu = end;
                (b'V', false, n)
            }
            EigenvalueRange::Indices(RangeInclusive { start, end }) => {
                assert!(1 <= start);
                assert!(start <= end);
                assert!(end <= n);
                *il = start;
                *iu = end;
                let max_m = end - start + 1;
                (b'I', max_m == n, max_m)
            }
        }
    }
}

pub fn part_to_u8(part: Part) -> u8 {
    match part {
        Part::Upper => b'U',
        Part::Lower => b'L',
    }
}

pub trait Gemm: Copy {
    unsafe fn gemm(
        layout: cblas::Layout,
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
        ldc: i32,
    );
}

impl Gemm for f32 {
    unsafe fn gemm(
        layout: cblas::Layout,
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
        ldc: i32,
    ) {
        cblas::sgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

impl Gemm for f64 {
    unsafe fn gemm(
        layout: cblas::Layout,
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
        ldc: i32,
    ) {
        cblas::dgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

impl Gemm for Complex<f32> {
    unsafe fn gemm(
        layout: cblas::Layout,
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
        ldc: i32,
    ) {
        cblas::cgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

impl Gemm for Complex<f64> {
    unsafe fn gemm(
        layout: cblas::Layout,
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
        ldc: i32,
    ) {
        cblas::zgemm(layout, transa, transb, m, n, k,
                       alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

/// A thin wrapper over `Gemm::gemm` that panics if the buffers are too
/// small or if the sizes don't match.  We omit `Layout` because it can be
/// trivially emulated by exchanging `a` and `b`.
pub fn gemm<T: Gemm>(
    transa: Transpose,
    transb: Transpose,
    alpha: T,
    a: Mat<T>,
    b: Mat<T>,
    beta: T,
    c: MatMut<T>,
) {
    let (ma, ka) = utils::swap_if(transa != Transpose::None, a.dims());
    let (kb, nb) = utils::swap_if(transb != Transpose::None, b.dims());
    let (mc, nc) = c.dims();
    assert_eq!(ma, mc);
    assert_eq!(nb, nc);
    assert_eq!(ka, kb);
    let lda = cast(a.stride());
    let ldb = cast(b.stride());
    let ldc = cast(c.stride());
    // FIXME: handle integer overflow for very large matrices
    unsafe {
        T::gemm(
            cblas::Layout::RowMajor,
            transa,
            transb,
            cast(ma),
            cast(nb),
            cast(ka),
            alpha,
            a.to_slice(),
            lda,
            b.to_slice(),
            ldb,
            beta,
            c.to_slice(),
            ldc,
        );
    }
}

pub trait Heevr: NormSqr + Copy {
    unsafe fn heevr(
        layout: lapacke::Layout,
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
    ) -> i32;
}

impl Heevr for f32 {
    unsafe fn heevr(
        layout: lapacke::Layout,
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
    ) -> i32 {
        lapacke::ssyevr(
            layout,
            jobz,
            range,
            uplo,
            n,
            a,
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz,
            isuppz,
        )
    }
}

impl Heevr for f64 {
    unsafe fn heevr(
        layout: lapacke::Layout,
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
    ) -> i32 {
        lapacke::dsyevr(
            layout,
            jobz,
            range,
            uplo,
            n,
            a,
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz,
            isuppz,
        )
    }
}

impl Heevr for Complex<f32> {
    unsafe fn heevr(
        layout: lapacke::Layout,
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
    ) -> i32 {
        lapacke::cheevr(
            layout,
            jobz,
            range,
            uplo,
            n,
            a,
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz,
            isuppz,
        )
    }
}

impl Heevr for Complex<f64> {
    unsafe fn heevr(
        layout: lapacke::Layout,
        jobz: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self::Real,
        vu: Self::Real,
        il: i32,
        iu: i32,
        abstol: Self::Real,
        m: &mut i32,
        w: &mut [Self::Real],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
    ) -> i32 {
        lapacke::zheevr(
            layout,
            jobz,
            range,
            uplo,
            n,
            a,
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            m,
            w,
            z,
            ldz,
            isuppz,
        )
    }
}

/// If `left` is true, calculate left eigenvectors stored as rows;
/// otherwise, calculate right eigenvectors stored as columns.
pub fn heevr<T: Heevr>(
    left: bool,
    range: EigenvalueRange<T::Real>,
    uplo: Part,
    a: MatMut<T>,
    abstol: T::Real,
    w: &mut [T::Real],
    z: MatMut<T>,
    mut isuppz: Result<&mut [i32], &mut Vec<i32>>,
) -> Result<usize, i32> {
    unsafe {
        let n = a.num_rows();
        assert_eq!(n, a.num_cols());
        let lda = cast(a.stride());
        let ldz = cast(z.stride());
        let mut vl = mem::uninitialized();
        let mut vu = mem::uninitialized();
        let mut il = mem::uninitialized();
        let mut iu = mem::uninitialized();
        let (range, all, max_m) = range.to_raw(
            cast(n),
            &mut vl,
            &mut vu,
            &mut il,
            &mut iu,
        );
        let max_m = max_m as usize;
        assert!(w.len() >= max(1, n));
        let (nz, mz) = utils::swap_if(left, z.dims());
        assert!(nz >= n);
        assert!(mz >= max_m);
        if all {
            let isuppz_len = 2 * max(1, n);
            match isuppz {
                Ok(ref isuppz) => {
                    assert!(isuppz.len() >= isuppz_len);
                }
                Err(ref mut isuppz) => {
                    isuppz.resize(isuppz_len, 0);
                }
            }
        }
        let mut m = mem::uninitialized();
        let e = T::heevr(
            if left {
                lapacke::Layout::ColumnMajor
            } else {
                lapacke::Layout::RowMajor
            },
            b'V',
            range,
            part_to_u8(uplo),
            cast(n),
            a.to_slice(),
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w,
            z.to_slice(),
            ldz,
            match isuppz {
                Ok(isuppz) => isuppz,
                Err(isuppz) => isuppz,
            },
        );
        if e == 0 {
            Ok(cast(m))
        } else {
            Err(e)
        }
    }
}

/// If `left` is true, calculate left eigenvalues;
/// otherwise, calculate right eigenvalues
pub fn heevr_n<T: Heevr>(
    left: bool,
    range: EigenvalueRange<T::Real>,
    uplo: Part,
    a: MatMut<T>,
    abstol: T::Real,
    w: &mut [T::Real],
) -> Result<usize, i32> {
    unsafe {
        let n = a.num_rows();
        assert_eq!(n, a.num_cols());
        let lda = cast(a.stride());
        let mut vl = mem::uninitialized();
        let mut vu = mem::uninitialized();
        let mut il = mem::uninitialized();
        let mut iu = mem::uninitialized();
        let (range, _, _) = range.to_raw(
            cast(n),
            &mut vl,
            &mut vu,
            &mut il,
            &mut iu,
        );
        assert!(w.len() >= max(1, n));
        let mut m = mem::uninitialized();
        let e = T::heevr(
            if left {
                lapacke::Layout::ColumnMajor
            } else {
                lapacke::Layout::RowMajor
            },
            b'N',
            range,
            part_to_u8(uplo),
            cast(n),
            a.to_slice(),
            lda,
            vl,
            vu,
            il,
            iu,
            abstol,
            &mut m,
            w,
            &mut [],
            1,
            &mut [],
        );
        if e == 0 {
            Ok(cast(m))
        } else {
            Err(e)
        }
    }
}

/// `y ← α × x + β × y`
pub fn mat_axpby<T>(alpha: T, x: Mat<T>, beta: T, mut y: MatMut<T>)
    where T: Add<Output = T> + Mul<Output = T> + Clone,
{
    assert_eq!(x.shape().num_rows, y.shape().num_rows);
    assert_eq!(x.shape().num_cols, y.shape().num_cols);
    for i in 0 .. x.shape().num_rows {
        for j in 0 .. x.shape().num_cols {
            let xij = x.get(i, j).unwrap().clone();
            let yij = y.as_mut().get(i, j).unwrap();
            *yij = alpha.clone() * xij + beta.clone() * yij.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::matrix::Matrix;

    #[test]
    fn it_works() {
        let a = Matrix::from(vec![vec![1.0, 2.0],
                                  vec![3.0, 4.0]]);
        let b = Matrix::from(vec![vec![5.0, 6.0],
                                  vec![7.0, 8.0]]);
        let c0 = Matrix::from(vec![vec![-1.0, -2.0],
                                   vec![-3.0, -4.0]]);

        let mut c = c0.clone();
        gemm(Transpose::None, Transpose::None,
             2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());
        assert_eq!(c, Matrix::from(vec![vec![35.0, 38.0],
                                        vec![77.0, 88.0]]));

        let mut c = c0.clone();
        gemm(Transpose::Ordinary, Transpose::None,
             2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());
        assert_eq!(c, Matrix::from(vec![vec![49.0, 54.0],
                                        vec![67.0, 76.0]]));

        let mut c = c0.clone();
        gemm(Transpose::None, Transpose::Ordinary,
             2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());
        assert_eq!(c, Matrix::from(vec![vec![31.0, 40.0],
                                        vec![69.0, 94.0]]));

        let mut c = c0.clone();
        gemm(Transpose::Ordinary, Transpose::Ordinary,
             2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());
        assert_eq!(c, Matrix::from(vec![vec![43.0, 56.0],
                                        vec![59.0, 80.0]]));
    }
}
