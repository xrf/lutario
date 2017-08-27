use std::{cmp, fmt, mem, ptr, slice};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use blas;
use blas::c::{Layout, Transpose};
use num::{Complex, Zero};
use utils::{Offset, cast, try_cast};

pub trait AsMat {
    type Elem;
    fn as_mat(&self) -> Mat<Self::Elem>;

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_mat().ptr
    }

    fn shape(&self) -> Shape {
        self.as_mat().shape
    }

    fn nrows(&self) -> usize {
        self.shape().nrows()
    }

    fn ncols(&self) -> usize {
        self.shape().ncols()
    }

    fn dims(&self) -> (usize, usize) {
        (self.shape().nrows(), self.shape().ncols())
    }

    fn transpose_dims(&self, trans: Transpose) -> (usize, usize) {
        if trans == Transpose::None {
            self.dims()
        } else {
            (self.ncols(), self.nrows())
        }
    }

    fn stride(&self) -> usize {
        self.shape().stride()
    }

    fn raw_len(&self) -> usize {
        self.shape().raw_len()
    }
}

impl<M: Deref> AsMat for M
    where M::Target: AsMat
{
    type Elem = <M::Target as AsMat>::Elem;
    fn as_mat(&self) -> Mat<Self::Elem> {
        (**self).as_mat()
    }
}

pub trait AsMatMut: AsMat {
    fn as_mat_mut(&mut self) -> MatMut<Self::Elem>;

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mat_mut().ptr
    }
}

impl<M: DerefMut> AsMatMut for M
    where M::Target: AsMatMut
{
    fn as_mat_mut(&mut self) -> MatMut<Self::Elem> {
        (**self).as_mat_mut()
    }
}

pub trait IntoMat<'a>: Sized + AsMat
    where Self::Elem: 'a
{
    fn into_mat(self) -> Mat<'a, Self::Elem>;

    fn get(self, i: usize, j: usize) -> Option<&'a Self::Elem> {
        unsafe {
            if self.shape().contains(i, j) {
                Some(self.get_unchecked(i, j))
            } else {
                None
            }
        }
    }

    unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a Self::Elem {
        Mat::get_unchecked(self.into_mat(), i, j)
    }

    fn index(self, i: usize, j: usize) -> &'a Self::Elem {
        self.get(i, j).expect(&format!("out of range: ({}, {})", i, j))
    }

    fn row(self, i: usize) -> Option<&'a [Self::Elem]> {
        unsafe {
            if i < self.nrows() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    unsafe fn row_unchecked(self, i: usize) -> &'a [Self::Elem] {
        Mat::row_unchecked(self.into_mat(), i)
    }

    fn rows(self) -> Rows<'a, Self::Elem> {
        let nrows = self.nrows();
        Rows { mat: self.into_mat(), range: 0 .. nrows }
    }

    fn slice(self, is: Range<usize>, js: Range<usize>)
             -> Mat<'a, Self::Elem> {
        Mat::slice(self.into_mat(), is, js)
    }

    fn split_at_row(self, i: usize) -> (Mat<'a, Self::Elem>,
                                            Mat<'a, Self::Elem>) {
        Mat::split_at_row(self.into_mat(), i)
    }

    fn split_at_col(self, j: usize) -> (Mat<'a, Self::Elem>,
                                        Mat<'a, Self::Elem>) {
        Mat::split_at_col(self.into_mat(), j)
    }
}

impl<'a, T: AsMat> IntoMat<'a> for &'a T {
    fn into_mat(self) -> Mat<'a, Self::Elem> {
        self.as_mat()
    }
}

impl<'a, T: AsMat> IntoMat<'a> for &'a mut T {
    fn into_mat(self) -> Mat<'a, Self::Elem> {
        (&*self).as_mat()
    }
}

pub trait IntoMatMut<'a>: IntoMat<'a>
    where <Self as AsMat>::Elem: 'a
{
    fn into_mat_mut(self) -> MatMut<'a, Self::Elem>;

    fn get_mut(self, i: usize, j: usize) -> Option<&'a mut Self::Elem> {
        unsafe {
            if self.shape().contains(i, j) {
                Some(self.get_unchecked_mut(i, j))
            } else {
                None
            }
        }
    }

    unsafe fn get_unchecked_mut(self, i: usize, j: usize)
                                -> &'a mut Self::Elem {
        MatMut::get_unchecked_mut(self.into_mat_mut(), i, j)
    }

    fn index_mut(self, i: usize, j: usize) -> &'a mut Self::Elem {
        self.get_mut(i, j).expect(&format!("out of range: ({}, {})", i, j))
    }

    fn row_mut(self, i: usize) -> Option<&'a mut [Self::Elem]> {
        MatMut::row_mut(self.into_mat_mut(), i)
    }

    fn rows_mut(self) -> RowsMut<'a, Self::Elem> {
        let nrows = self.nrows();
        RowsMut { mat: self.into_mat_mut(), range: 0 .. nrows }
    }

    fn slice_mut(self, is: Range<usize>, js: Range<usize>)
                 -> MatMut<'a, Self::Elem> {
        MatMut::slice_mut(self.into_mat_mut(), is, js)
    }

    fn split_at_row_mut(self, i: usize) -> (MatMut<'a, Self::Elem>,
                                            MatMut<'a, Self::Elem>) {
        MatMut::split_at_row_mut(self.into_mat_mut(), i)
    }

    fn split_at_col_mut(self, j: usize) -> (MatMut<'a, Self::Elem>,
                                            MatMut<'a, Self::Elem>) {
        MatMut::split_at_col_mut(self.into_mat_mut(), j)
    }
}

impl<'a, T: AsMatMut> IntoMatMut<'a> for &'a mut T {
    fn into_mat_mut(self) -> MatMut<'a, Self::Elem> {
        self.as_mat_mut()
    }
}

/// The indexing convention is row-major.
///
///   - `stride`: The separation between rows.  Must be at least `ncols`.
///   - `nrows`: The slow index.
///   - `ncols`: The fast index.
///
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    stride: usize,
    nrows: usize,
    ncols: usize,
}

impl Shape {
    fn is_valid(self) -> bool {
        let stride = self.stride();
        match self.nrows().checked_mul(stride) {
            Some(n) => {
                let r: Option<isize> = try_cast(n);
                r.is_some() && stride >= self.ncols()
            }
            _ => false,
        }
    }

    pub fn new(stride: usize, nrows: usize, ncols: usize) -> Option<Self> {
        let shape = Self { stride, nrows, ncols };
        if shape.is_valid() {
            Some(shape)
        } else {
            None
        }
    }

    pub unsafe fn from_raw(stride: usize, nrows: usize, ncols: usize) -> Self {
        let shape = Self { stride, nrows, ncols };
        debug_assert!(shape.is_valid());
        shape
    }

    pub fn nrows(self) -> usize {
        self.nrows
    }

    pub fn ncols(self) -> usize {
        self.ncols
    }

    pub fn stride(self) -> usize {
        self.stride
    }

    pub unsafe fn modify_ncols_unchecked(self, ncols: usize) -> Self {
        let shape = Self { ncols, .. self };
        debug_assert!(shape.is_valid());
        shape
    }

    pub unsafe fn modify_nrows_unchecked(self, nrows: usize) -> Self {
        let shape = Self { nrows, .. self };
        debug_assert!(shape.is_valid());
        shape
    }

    pub fn contains(self, i: usize, j: usize) -> bool {
        i < self.nrows() && j < self.ncols()
    }

    pub fn raw_index(self, i: usize, j: usize) -> usize {
        i * self.stride() + j
    }

    pub fn raw_len(self) -> usize {
        let nrows = self.nrows();
        if nrows == 0 {
            0
        } else {
            (nrows - 1) * self.stride() + self.ncols()
        }
    }

    pub unsafe fn offset_unchecked<P: Offset>(self, ptr: P,
                                              i: usize, j: usize) -> P {
        ptr.offset(self.raw_index(i, j) as _)
    }
}

pub struct Mat<'a, T: 'a> {
    phantom: PhantomData<&'a [T]>,
    ptr: *const T,
    shape: Shape,
}

unsafe impl<'a, T: Sync> Send for Mat<'a, T> {}
unsafe impl<'a, T: Sync> Sync for Mat<'a, T> {}

impl<'a, T> Clone for Mat<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for Mat<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for Mat<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("mat!")?;
        let mut rows = f.debug_list();
        for row in self.rows() {
            rows.entry(&row);
        }
        rows.finish()
    }
}

impl<'a, T: PartialEq<Rhs::Elem>, Rhs: AsMat> PartialEq<Rhs> for Mat<'a, T> {
    fn eq(&self, rhs: &Rhs) -> bool {
        if self.nrows() != rhs.nrows() {
            return false;
        }
        if self.ncols() != rhs.ncols() {
            return false;
        }
        for (self_row, rhs_row) in self.rows().zip(rhs.rows()) {
            if self_row != rhs_row {
                return false;
            }
        }
        true
    }
}

impl<'a, T: Eq> Eq for Mat<'a, T> {}

impl<'a, T> Mat<'a, T> {
    pub fn new(slice: &'a [T], shape: Shape) -> Option<Self> {
        unsafe {
            if slice.len() >= shape.raw_len() {
                Some(Self::from_raw(slice.as_ptr(), shape))
            } else {
                None
            }
        }
    }

    pub unsafe fn from_raw(ptr: *const T, shape: Shape) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    unsafe fn offset_unchecked(&self, i: usize, j: usize) -> *const T {
        self.shape().offset_unchecked(self.as_ptr(), i, j)
    }

    unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a T {
        &*self.offset_unchecked(i, j)
    }

    unsafe fn row_unchecked(self, i: usize) -> &'a [T] {
        let ptr = self.offset_unchecked(i, 0);
        slice::from_raw_parts(ptr, self.nrows())
    }

    fn slice(self, is: Range<usize>, js: Range<usize>) -> Self {
        unsafe {
            let shape = self.shape()
                .modify_nrows_unchecked(cmp::min(is.end, self.nrows())
                                      - cmp::min(is.start, self.nrows()))
                .modify_ncols_unchecked(cmp::min(js.end, self.ncols())
                                      - cmp::min(js.start, self.ncols()));
            let ptr = self.offset_unchecked(is.start, js.start);
            Self::from_raw(ptr, shape)
        }
    }

    fn split_at_row(self, i: usize) -> (Self, Self) {
        let i = cmp::min(i, self.nrows());
        unsafe {
            let ptr = self.offset_unchecked(i, 0);
            (
                Self::from_raw(
                    self.ptr,
                    self.shape().modify_nrows_unchecked(i),
                ),
                Self::from_raw(
                    ptr as _,
                    self.shape().modify_nrows_unchecked(self.nrows() - i),
                ),
            )
        }
    }

    fn split_at_col(self, j: usize) -> (Self, Self) {
        let j = cmp::min(j, self.ncols());
        unsafe {
            let ptr = self.offset_unchecked(0, j);
            (
                Self::from_raw(
                    self.ptr,
                    self.shape().modify_ncols_unchecked(j),
                ),
                Self::from_raw(
                    ptr as _,
                    self.shape().modify_ncols_unchecked(self.ncols() - j),
                ),
            )
        }
    }
}

impl<'a, T> AsMat for Mat<'a, T> {
    type Elem = T;
    fn as_mat(&self) -> Mat<Self::Elem> {
        *self
    }
}

impl<'a, T> IntoMat<'a> for Mat<'a, T> {
    fn into_mat(self) -> Mat<'a, Self::Elem> {
        self
    }
}

pub struct MatMut<'a, T: 'a> {
    phantom: PhantomData<&'a mut [T]>,
    ptr: *mut T,
    shape: Shape,
}

impl<'a, T: fmt::Debug> fmt::Debug for MatMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_mat().fmt(f)
    }
}

unsafe impl<'a, T: Send> Send for MatMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatMut<'a, T> {}

impl<'a, T: PartialEq<Rhs::Elem>, Rhs: AsMat> PartialEq<Rhs> for MatMut<'a, T> {
    fn eq(&self, rhs: &Rhs) -> bool {
        self.as_mat().eq(rhs)
    }
}

impl<'a, T: Eq> Eq for MatMut<'a, T> {}

impl<'a, T> MatMut<'a, T> {
    pub fn new(slice: &'a mut [T], shape: Shape) -> Option<Self> {
        unsafe {
            if slice.len() >= shape.raw_len() {
                Some(Self::from_raw(slice.as_mut_ptr(), shape))
            } else {
                None
            }
        }
    }

    pub unsafe fn from_raw(ptr: *mut T, shape: Shape) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    unsafe fn offset_unchecked_mut(&mut self, i: usize, j: usize) -> *mut T {
        self.shape().offset_unchecked(self.as_mut_ptr(), i, j)
    }

    unsafe fn get_unchecked_mut(mut self, i: usize, j: usize) -> &'a mut T {
        &mut *self.offset_unchecked_mut(i, j)
    }

    unsafe fn row_unchecked_mut(mut self, i: usize) -> &'a mut [T] {
        let ptr = self.offset_unchecked_mut(i, 0) as _;
        slice::from_raw_parts_mut(ptr, self.nrows())
    }

    fn slice_mut(mut self, is: Range<usize>, js: Range<usize>) -> Self {
        unsafe {
            let shape = self.shape()
                .modify_nrows_unchecked(cmp::min(is.end, self.nrows())
                                      - cmp::min(is.start, self.nrows()))
                .modify_ncols_unchecked(cmp::min(js.end, self.ncols())
                                      - cmp::min(js.start, self.ncols()));
            let ptr = self.offset_unchecked_mut(is.start, js.start);
            Self::from_raw(ptr, shape)
        }
    }

    fn split_at_row_mut(mut self, i: usize) -> (Self, Self) {
        let i = cmp::min(i, self.nrows());
        unsafe {
            let ptr = self.offset_unchecked_mut(i, 0);
            (
                Self::from_raw(
                    self.ptr,
                    self.shape().modify_nrows_unchecked(i),
                ),
                Self::from_raw(
                    ptr as _,
                    self.shape().modify_nrows_unchecked(self.nrows() - i),
                ),
            )
        }
    }

    fn split_at_col_mut(mut self, j: usize) -> (Self, Self) {
        let j = cmp::min(j, self.ncols());
        unsafe {
            let ptr = self.offset_unchecked_mut(0, j);
            (
                Self::from_raw(
                    self.ptr,
                    self.shape().modify_ncols_unchecked(j),
                ),
                Self::from_raw(
                    ptr as _,
                    self.shape().modify_ncols_unchecked(self.ncols() - j),
                ),
            )
        }
    }
}

impl<'a, T> AsMat for MatMut<'a, T> {
    type Elem = T;
    fn as_mat(&self) -> Mat<Self::Elem> {
        unsafe {
            Mat::from_raw(self.ptr, self.shape)
        }
    }
}

impl<'a, T> AsMatMut for MatMut<'a, T> {
    fn as_mat_mut(&mut self) -> MatMut<Self::Elem> {
        unsafe {
            MatMut::from_raw(self.ptr, self.shape)
        }
    }
}

impl<'a, T> IntoMat<'a> for MatMut<'a, T> {
    fn into_mat(self) -> Mat<'a, Self::Elem> {
        unsafe {
            Mat::from_raw(self.ptr, self.shape)
        }
    }
}

impl<'a, T> IntoMatMut<'a> for MatMut<'a, T> {
    fn into_mat_mut(self) -> MatMut<'a, Self::Elem> {
        self
    }
}

#[derive(Clone, Debug)]
pub struct Rows<'a, T: 'a> {
    mat: Mat<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let mat = self.mat;
        self.range.next().map(|i| unsafe {
            mat.row_unchecked(i)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Rows<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = self.mat;
        self.range.next_back().map(|i| unsafe {
            mat.row_unchecked(i)
        })
    }
}

impl<'a, T> ExactSizeIterator for Rows<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

#[derive(Debug)]
pub struct RowsMut<'a, T: 'a> {
    mat: MatMut<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        let mat = &mut self.mat;
        self.range.next().map(|i| unsafe {
            mem::transmute(mat.as_mat_mut().row_unchecked_mut(i))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for RowsMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = &mut self.mat;
        self.range.next_back().map(|i| unsafe {
            mem::transmute(mat.as_mat_mut().row_unchecked_mut(i))
        })
    }
}

impl<'a, T> ExactSizeIterator for RowsMut<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

pub struct Matrix<T> {
    phantom: PhantomData<Box<[T]>>,
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_mat().fmt(f)
    }
}

impl<T: PartialEq<Rhs::Elem>, Rhs: AsMat> PartialEq<Rhs> for Matrix<T> {
    fn eq(&self, rhs: &Rhs) -> bool {
        self.as_mat().eq(rhs)
    }
}

impl<T: Eq> Eq for Matrix<T> {}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            mem::replace(self, mem::uninitialized()).into_boxed_slice();
        }
    }
}

impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        unsafe {
            Self::from_vec_unchecked(self.as_slice().to_vec(),
                                     self.nrows, self.ncols)
        }
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
    fn from(rows: Vec<Vec<T>>) -> Self {
        let ni = rows.len();
        let nj = if ni == 0 { 0 } else { rows[0].len() };
        let mut data = Vec::with_capacity(ni * nj);
        for mut row in rows {
            data.extend(row.drain(..));
        }
        data.shrink_to_fit();
        unsafe {
            Self::from_vec_unchecked(data, ni, nj)
        }
    }
}

impl<T: Clone + Default> Matrix<T> {
    pub fn from_vec(mut vec: Vec<T>, nrows: usize, ncols: usize) -> Self {
        let n = nrows * ncols;
        vec.resize(n, Default::default());
        vec.shrink_to_fit();
        unsafe {
            Self::from_vec_unchecked(vec, nrows, ncols)
        }
    }
}

impl<T: Clone> Matrix<T> {
    pub fn replicate(nrows: usize, ncols: usize, value: T) -> Self {
        unsafe {
            Self::from_vec_unchecked(vec![value; nrows * ncols], nrows, ncols)
        }
    }
}

impl<T: Clone + Zero> Matrix<T> {
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        Self::replicate(nrows, ncols, Zero::zero())
    }
}

impl<T> Matrix<T> {
    pub unsafe fn from_raw(ptr: *mut T, nrows: usize, ncols: usize) -> Self {
        Self {
            phantom: PhantomData,
            ptr: ptr,
            nrows: nrows,
            ncols: ncols,
        }
    }

    pub unsafe fn from_vec_unchecked(mut vec: Vec<T>,
                                     nrows: usize, ncols: usize) -> Self {
        debug_assert_eq!(vec.capacity(), nrows * ncols);
        debug_assert_eq!(vec.len(), nrows * ncols);
        let ptr = vec.as_mut_ptr();
        mem::forget(vec);
        Self::from_raw(ptr, nrows, ncols)
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.nrows * self.ncols)
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr, self.nrows * self.ncols)
        }
    }

    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        unsafe {
            let b = Box::from_raw(self.as_slice_mut());
            mem::forget(self);
            b
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.into_boxed_slice().into_vec()
    }

    pub fn into_transpose(self) -> Self {
        let (ni, nj) = (&self).dims();
        let n = ni * nj;
        unsafe {
            let mut data = Vec::with_capacity(n);
            data.set_len(n);
            let mut m = Self::from_vec_unchecked(data, nj, ni);
            for i in 0 .. ni {
                for j in 0 .. nj {
                    ptr::write(m.get_unchecked_mut(j, i),
                               ptr::read(self.get_unchecked(i, j)));
                }
            }
            // forget the data
            self.into_boxed_slice().into_vec().set_len(0);
            m
        }
    }
}

impl<T> AsMat for Matrix<T> {
    type Elem = T;
    fn as_mat(&self) -> Mat<Self::Elem> {
        unsafe {
            Mat::from_raw(self.ptr, Shape::from_raw(
                self.ncols,
                self.nrows,
                self.ncols,
            ))
        }
    }
}

impl<T> AsMatMut for Matrix<T> {
    fn as_mat_mut(&mut self) -> MatMut<Self::Elem> {
        unsafe {
            MatMut::from_raw(self.ptr, (&self).shape())
        }
    }
}

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
            a: Mat<Self>,
            b: Mat<Self>,
            beta: Self,
            mut c: MatMut<Self>) {
        let (ma, ka) = a.transpose_dims(transa);
        let (kb, nb) = b.transpose_dims(transb);
        let (mc, nc) = c.dims();
        assert_eq!(ma, mc);
        assert_eq!(nb, nc);
        assert_eq!(ka, kb);
        let lda = cast(a.stride());
        let ldb = cast(b.stride());
        let ldc = cast(c.stride());
        // FIXME: handle integer overflow for very large matrices
        unsafe {
            Self::unsafe_gemm(
                Layout::RowMajor,
                transa,
                transb,
                cast(ma),
                cast(nb),
                cast(ka),
                alpha,
                slice::from_raw_parts(a.as_ptr(), a.raw_len()),
                lda,
                slice::from_raw_parts(b.as_ptr(), b.raw_len()),
                ldb,
                beta,
                slice::from_raw_parts_mut(c.as_mut_ptr(), c.raw_len()),
                ldc,
            );
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
        let a = Matrix::from(vec![vec![1.0, 2.0],
                                  vec![3.0, 4.0]]);
        let b = Matrix::from(vec![vec![5.0, 6.0],
                                  vec![7.0, 8.0]]);
        let c0 = Matrix::from(vec![vec![-1.0, -2.0],
                                   vec![-3.0, -4.0]]);

        let mut c = c0.clone();
        f64::gemm(Transpose::None, Transpose::None,
                  2.0, a.as_mat(), b.as_mat(), 3.0, c.as_mat_mut());
        assert_eq!(c, Matrix::from(vec![vec![35.0, 38.0],
                                        vec![77.0, 88.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::Ordinary, Transpose::None,
                  2.0, a.as_mat(), b.as_mat(), 3.0, c.as_mat_mut());
        assert_eq!(c, Matrix::from(vec![vec![49.0, 54.0],
                                        vec![67.0, 76.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::None, Transpose::Ordinary,
                  2.0, a.as_mat(), b.as_mat(), 3.0, c.as_mat_mut());
        assert_eq!(c, Matrix::from(vec![vec![31.0, 40.0],
                                        vec![69.0, 94.0]]));

        let mut c = c0.clone();
        f64::gemm(Transpose::Ordinary, Transpose::Ordinary,
                  2.0, a.as_mat(), b.as_mat(), 3.0, c.as_mat_mut());
        assert_eq!(c, Matrix::from(vec![vec![43.0, 56.0],
                                        vec![59.0, 80.0]]));
    }
}
