//! BLAS-compatible matrix data types.

use std::{fmt, mem, ptr, slice};
use std::cmp::min;
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut, Range};
use num::Zero;
use super::op::{Vector, VectorMut};
use super::tri_mat::{Trs, TrsMat};
use super::utils::{self, Offset, try_cast};

/// The indexing convention is row-major.
///
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MatShape {
    /// The separation between rows.  Must be at least `num_cols`.
    pub stride: usize,
    /// The slow index.
    pub num_rows: usize,
    /// The fast index.
    pub num_cols: usize,
}

impl MatShape {
    pub fn is_valid(&self) -> bool {
        match self.num_rows.checked_mul(self.stride) {
            Some(n) => {
                let r: Option<isize> = try_cast(n);
                r.is_some() && self.stride >= self.num_cols
            }
            _ => false,
        }
    }

    pub fn packed(num_rows: usize, num_cols: usize) -> Self {
        Self {
            stride: num_cols,
            num_rows,
            num_cols
        }
    }

    pub fn validate(self) -> Result<ValidMatShape, Self> {
        if self.is_valid() {
            Ok(unsafe { self.assert_valid() })
        } else {
            Err(self)
        }
    }

    pub unsafe fn assert_valid(self) -> ValidMatShape {
        debug_assert!(self.is_valid());
        ValidMatShape(self)
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        i < self.num_rows && j < self.num_cols
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ValidMatShape(MatShape);

impl Deref for ValidMatShape {
    type Target = MatShape;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ValidMatShape {
    pub fn extent(&self) -> usize {
        let num_rows = self.num_rows;
        if num_rows == 0 {
            0
        } else {
            (num_rows - 1) * self.stride + self.num_cols
        }
    }

    pub fn raw_index(&self, i: usize, j: usize) -> usize {
        i * self.stride + j
    }

    pub unsafe fn offset_unchecked<P: Offset>(&self, ptr: P,
                                              i: usize, j: usize) -> P {
        ptr.offset(self.raw_index(i, j) as _)
    }

    pub fn row_width(&self) -> usize {
        self.num_cols
    }
}

pub struct MatRef<'a, T: 'a> {
    phantom: PhantomData<&'a [T]>,
    ptr: *const T,
    shape: ValidMatShape,
}

unsafe impl<'a, T: Sync> Send for MatRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatRef<'a, T> {}

impl<'a, T> Clone for MatRef<'a, T> { fn clone(&self) -> Self { *self } }
impl<'a, T> Copy for MatRef<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for MatRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("mat!")?;
        let mut rows = f.debug_list();
        for row in self.rows() {
            rows.entry(&row);
        }
        rows.finish()
    }
}

impl<'a, T> Default for MatRef<'a, T> {
    fn default() -> Self {
        Self::new(&mut (&[] as &[_]), Default::default()).expect("!?")
    }
}

impl<'a, 'b, T: PartialEq<U>, U> PartialEq<MatRef<'b, U>> for MatRef<'a, T> {
    fn eq(&self, rhs: &MatRef<'b, U>) -> bool {
        if self.num_rows() != rhs.num_rows() {
            return false;
        }
        if self.num_cols() != rhs.num_cols() {
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

impl<'a, T: Eq> Eq for MatRef<'a, T> {}

impl<'a, T> Index<(usize, usize)> for MatRef<'a, T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        (*self).index(i, j)
    }
}

impl<'a, T> MatRef<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a [T], shape: ValidMatShape) -> Option<Self> {
        unsafe {
            utils::chop_slice(slice, shape.extent()).map(|slice| {
                Self::from_raw(slice.as_ptr(), shape)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *const T, shape: ValidMatShape) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    pub fn shape(self) -> ValidMatShape {
        self.shape
    }

    pub fn num_rows(self) -> usize {
        self.shape().num_rows
    }

    pub fn num_cols(self) -> usize {
        self.shape().num_cols
    }

    pub fn dims(self) -> (usize, usize) {
        (self.shape().num_rows, self.shape().num_cols)
    }

    pub fn stride(self) -> usize {
        self.shape().stride
    }

    pub fn extent(self) -> usize {
        self.shape().extent()
    }

    pub fn as_ptr(self) -> *const T {
        self.ptr
    }

    pub fn index(self, i: usize, j: usize) -> &'a T {
        self.get(i, j).expect(&format!("out of range: ({}, {})", i, j))
    }

    pub fn get(self, i: usize, j: usize) -> Option<&'a T> {
        unsafe {
            if self.shape().contains(i, j) {
                Some(self.get_unchecked(i, j))
            } else {
                None
            }
        }
    }

    pub unsafe fn get_unchecked(self, i: usize, j: usize) -> &'a T {
        &*self.offset_unchecked(i, j)
    }

    pub unsafe fn offset_unchecked(self, i: usize, j: usize) -> *const T {
        self.shape().offset_unchecked(self.as_ptr(), i, j)
    }

    pub fn rows(self) -> MatRows<'a, T> {
        let num_rows = self.num_rows();
        MatRows { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a [T]> {
        unsafe {
            if i < self.num_rows() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(self, i: usize) -> &'a [T] {
        let ptr = self.offset_unchecked(i, 0);
        slice::from_raw_parts(ptr, self.shape().row_width())
    }

    /// Unsafe because the slice includes padding elements as well.
    pub unsafe fn to_slice(self) -> &'a [T] {
        slice::from_raw_parts(self.as_ptr(), self.extent())
    }

    pub fn slice(self, is: Range<usize>, js: Range<usize>) -> Self {
        unsafe {
            let shape = MatShape {
                num_rows: min(is.end, self.num_rows())
                        - min(is.start, self.num_rows()),
                num_cols: min(js.end, self.num_cols())
                        - min(js.start, self.num_cols()),
                .. *self.shape()
            }.assert_valid();
            let ptr = self.offset_unchecked(is.start, js.start);
            Self::from_raw(ptr, shape)
        }
    }

    pub fn split_at_row(self, i: usize) -> (Self, Self) {
        let i = min(i, self.num_rows());
        unsafe {
            let ptr = self.offset_unchecked(i, 0);
            (
                Self::from_raw(
                    self.ptr,
                    MatShape {
                        num_rows: i,
                        .. *self.shape()
                    }.assert_valid(),
                ),
                Self::from_raw(
                    ptr as _,
                    MatShape {
                        num_rows: self.num_rows() - i,
                        .. *self.shape()
                    }.assert_valid(),
                ),
            )
        }
    }

    pub fn split_at_col(self, j: usize) -> (Self, Self) {
        let j = min(j, self.num_cols());
        unsafe {
            let ptr = self.offset_unchecked(0, j);
            (
                Self::from_raw(
                    self.ptr,
                    MatShape {
                        num_cols: j,
                        .. *self.shape()
                    }.assert_valid(),
                ),
                Self::from_raw(
                    ptr as _,
                    MatShape {
                        num_cols: self.num_cols() - j,
                        .. *self.shape()
                    }.assert_valid(),
                ),
            )
        }
    }
}

impl<'a, T> Vector for MatRef<'a, T> {
    type Elem = T;
}

pub struct MatMut<'a, T: 'a> {
    phantom: PhantomData<&'a mut [T]>,
    ptr: *mut T,
    shape: ValidMatShape,
}

unsafe impl<'a, T: Send> Send for MatMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatMut<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for MatMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<'a, T> Default for MatMut<'a, T> {
    fn default() -> Self {
        Self::new(&mut (&mut [] as &mut [_]), Default::default()).expect("!?")
    }
}

impl<'a, 'b, T: PartialEq<U>, U> PartialEq<MatMut<'b, U>> for MatMut<'a, T> {
    fn eq(&self, rhs: &MatMut<'b, U>) -> bool {
        self.as_ref().eq(&rhs.as_ref())
    }
}

impl<'a, T: Eq> Eq for MatMut<'a, T> {}

impl<'a, T> Index<(usize, usize)> for MatMut<'a, T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.as_ref().index(i, j)
    }
}

impl<'a, T> IndexMut<(usize, usize)> for MatMut<'a, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.as_mut().index(i, j)
    }
}

impl<'a, T> MatMut<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a mut [T], shape: ValidMatShape) -> Option<Self> {
        unsafe {
            utils::chop_slice_mut(slice, shape.extent()).map(|slice| {
                Self::from_raw(slice.as_mut_ptr(), shape)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *mut T, shape: ValidMatShape) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    pub fn shape(&self) -> ValidMatShape {
        self.as_ref().shape
    }

    pub fn num_rows(&self) -> usize {
        self.as_ref().num_rows()
    }

    pub fn num_cols(&self) -> usize {
        self.as_ref().num_cols()
    }

    pub fn dims(&self) -> (usize, usize) {
        self.as_ref().dims()
    }

    pub fn stride(&self) -> usize {
        self.as_ref().stride()
    }

    pub fn into_ref(self) -> MatRef<'a, T> {
        unsafe {
            MatRef::from_raw(self.ptr, self.shape)
        }
    }

    pub fn as_ref(&self) -> MatRef<T> {
        unsafe {
            MatRef::from_raw(self.ptr, self.shape)
        }
    }

    pub fn as_mut(&mut self) -> MatMut<T> {
        unsafe {
            MatMut::from_raw(self.ptr, self.shape)
        }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn index(self, i: usize, j: usize) -> &'a mut T {
        self.get(i, j).expect(&format!("out of range: ({}, {})", i, j))
    }

    pub fn get(self, i: usize, j: usize) -> Option<&'a mut T> {
        unsafe {
            if self.shape().contains(i, j) {
                Some(self.get_unchecked(i, j))
            } else {
                None
            }
        }
    }

    pub unsafe fn get_unchecked(mut self, i: usize, j: usize) -> &'a mut T {
        &mut *self.offset_unchecked(i, j)
    }

    pub unsafe fn offset_unchecked(&mut self, i: usize, j: usize) -> *mut T {
        self.shape().offset_unchecked(self.as_ptr(), i, j)
    }

    pub fn rows(self) -> MatRowsMut<'a, T> {
        let num_rows = self.num_rows();
        MatRowsMut { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a mut [T]> {
        unsafe {
            if i < self.num_rows() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(mut self, i: usize) -> &'a mut [T] {
        let ptr = self.offset_unchecked(i, 0) as _;
        slice::from_raw_parts_mut(ptr, self.shape().row_width())
    }

    /// Unsafe because the slice includes padding elements as well.
    pub unsafe fn to_slice(self) -> &'a mut [T] {
        slice::from_raw_parts_mut(self.as_ptr(), self.as_ref().extent())
    }

    pub fn slice(mut self, is: Range<usize>, js: Range<usize>) -> Self {
        unsafe {
            let shape = MatShape {
                num_rows: min(is.end, self.num_rows())
                        - min(is.start, self.num_rows()),
                num_cols: min(js.end, self.num_cols())
                        - min(js.start, self.num_cols()),
                .. *self.shape()
            }.assert_valid();
            let ptr = self.offset_unchecked(is.start, js.start);
            Self::from_raw(ptr, shape)
        }
    }

    pub fn split_at_row(mut self, i: usize) -> (Self, Self) {
        let i = min(i, self.num_rows());
        unsafe {
            let ptr = self.offset_unchecked(i, 0);
            (
                Self::from_raw(
                    self.ptr,
                    MatShape {
                        num_rows: i,
                        .. *self.shape()
                    }.assert_valid(),
                ),
                Self::from_raw(
                    ptr as _,
                    MatShape {
                        num_rows: self.num_rows() - i,
                        .. *self.shape()
                    }.assert_valid(),
                ),
            )
        }
    }

    pub fn split_at_col(mut self, j: usize) -> (Self, Self) {
        let j = min(j, self.num_cols());
        unsafe {
            let ptr = self.offset_unchecked(0, j);
            (
                Self::from_raw(
                    self.ptr,
                    MatShape {
                        num_cols: j,
                        .. *self.shape()
                    }.assert_valid(),
                ),
                Self::from_raw(
                    ptr as _,
                    MatShape {
                        num_cols: self.num_cols() - j,
                        .. *self.shape()
                    }.assert_valid(),
                ),
            )
        }
    }
}

impl<'a, T: Clone> MatMut<'a, T> {
    pub fn fill(&mut self, value: &T) {
        for i in 0 .. self.num_rows() {
            for j in 0 .. self.num_cols() {
                self[(i, j)] = value.clone();
            }
        }
    }

    pub fn clone_from_ref(&mut self, source: MatRef<T>) {
        assert_eq!(self.shape().num_rows, source.shape().num_rows);
        assert_eq!(self.shape().num_cols, source.shape().num_cols);
        for (self_row, source_row) in self.as_mut().rows().zip(source.rows()) {
            self_row.clone_from_slice(source_row);
        }
    }
}

impl<'a, T: Clone> MatMut<'a, T> {
    pub fn clone_from_trs_mat<S>(&mut self, source: &TrsMat<S, T>) where
        S: Trs<T>,
    {
        let n = *source.mat.as_ref().dim();
        assert_eq!(n, self.shape().num_rows);
        assert_eq!(n, self.shape().num_cols);
        for i in 0 .. self.shape().num_rows {
            for j in 0 .. self.shape().num_cols {
                *self.as_mut().get(i, j).unwrap() = source.get(i, j).unwrap();
            }
        }
    }
}

impl<'a, T> Vector for MatMut<'a, T> {
    type Elem = T;
}

impl<'a, T: Zero + Clone> VectorMut for MatMut<'a, T> {
    fn set_zero(&mut self) {
        self.fill(&Zero::zero());
    }
}

#[derive(Clone, Debug)]
pub struct MatRows<'a, T: 'a> {
    mat: MatRef<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for MatRows<'a, T> {
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

impl<'a, T> DoubleEndedIterator for MatRows<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = self.mat;
        self.range.next_back().map(|i| unsafe {
            mat.row_unchecked(i)
        })
    }
}

impl<'a, T> ExactSizeIterator for MatRows<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

#[derive(Debug)]
pub struct MatRowsMut<'a, T: 'a> {
    mat: MatMut<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for MatRowsMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        let mat = &mut self.mat;
        self.range.next().map(|i| unsafe {
            mem::transmute(mat.as_mut().row_unchecked(i))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for MatRowsMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = &mut self.mat;
        self.range.next_back().map(|i| unsafe {
            mem::transmute(mat.as_mut().row_unchecked(i))
        })
    }
}

impl<'a, T> ExactSizeIterator for MatRowsMut<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

pub struct Mat<T> {
    phantom: PhantomData<Box<[T]>>,
    ptr: *mut T,
    num_rows: usize,
    num_cols: usize,
}

impl<T: fmt::Debug> fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T: Clone> Clone for Mat<T> {
    fn clone(&self) -> Self {
        Self::from_vec(self.as_slice().to_vec(), self.num_rows, self.num_cols)
    }
    fn clone_from(&mut self, source: &Self) {
        // simplest solution that satisfies panic safety
        if self.extent() != source.extent() {
            *self = source.clone();
        } else {
            // might panic halfway, in which case the data is corrupt but
            // there still shouldn't be any memory unsafety
            self.as_slice_mut().clone_from_slice(source.as_slice());
            self.num_rows = source.num_rows;
            self.num_cols = source.num_cols;
        }
    }
}

impl<T> Drop for Mat<T> {
    fn drop(&mut self) {
        unsafe {
            mem::replace(self, mem::uninitialized()).into_boxed_slice();
        }
    }
}

impl<T> Default for Mat<T> {
    fn default() -> Self {
        Self::from_vec(vec![], 0, 0)
    }
}

impl<T: PartialEq<U>, U> PartialEq<Mat<U>> for Mat<T> {
    fn eq(&self, rhs: &Mat<U>) -> bool {
        self.as_ref().eq(&rhs.as_ref())
    }
}

impl<T: Eq> Eq for Mat<T> {}

/// Convenience function for creating matrices directly in code.
impl<T> From<Vec<Vec<T>>> for Mat<T> {
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

impl<T: Clone> Mat<T> {
    pub fn replicate(num_rows: usize, num_cols: usize, value: T) -> Self {
        MatShape::packed(num_rows, num_cols).validate().unwrap();
        unsafe {
            Self::from_vec_unchecked(
                vec![value; num_rows * num_cols],
                num_rows,
                num_cols,
            )
        }
    }
}

impl<T: Clone + Zero> Mat<T> {
    pub fn zero(num_rows: usize, num_cols: usize) -> Self {
        Self::replicate(num_rows, num_cols, Zero::zero())
    }
}

impl<T> Mat<T> {
    /// Panics if the vector is too short.
    pub fn from_vec(mut vec: Vec<T>, num_rows: usize, num_cols: usize) -> Self {
        MatShape::packed(num_rows, num_cols).validate().unwrap();
        let n = num_rows * num_cols;
        assert!(vec.len() >= n);
        vec.truncate(n);
        vec.shrink_to_fit();
        unsafe {
            Self::from_vec_unchecked(vec, num_rows, num_cols)
        }
    }

    pub unsafe fn from_vec_unchecked(mut vec: Vec<T>,
                                     num_rows: usize, num_cols: usize) -> Self {
        // in order to reconstruct the vector correctly during Drop,
        // we need to ensure capacity == num_rows * num_cols
        debug_assert_eq!(vec.capacity(), num_rows * num_cols);
        debug_assert_eq!(vec.len(), num_rows * num_cols);
        let ptr = vec.as_mut_ptr();
        mem::forget(vec);
        Self::from_raw(ptr, num_rows, num_cols)
    }

    pub unsafe fn from_raw(ptr: *mut T, num_rows: usize, num_cols: usize) -> Self {
        Self {
            phantom: PhantomData,
            ptr,
            num_rows,
            num_cols,
        }
    }

    pub fn shape(&self) -> ValidMatShape {
        MatShape::packed(self.num_rows, self.num_cols)
            .validate().expect("!?")
    }

    pub fn extent(&self) -> usize {
        self.num_rows * self.num_cols
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn as_ref(&self) -> MatRef<T> {
        unsafe { MatRef::from_raw(self.as_ptr(), self.shape()) }
    }

    pub fn as_mut(&mut self) -> MatMut<T> {
        unsafe { MatMut::from_raw(self.as_ptr(), self.shape()) }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.extent()) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_ptr(), self.extent()) }
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
        let (ni, nj) = (&self).as_ref().dims();
        let n = ni * nj;
        unsafe {
            let mut data = Vec::with_capacity(n);
            data.set_len(n);
            let mut m = Self::from_vec_unchecked(data, nj, ni);
            for i in 0 .. ni {
                for j in 0 .. nj {
                    ptr::write(m.as_mut().get_unchecked(j, i),
                               ptr::read(self.as_ref().get_unchecked(i, j)));
                }
            }
            // forget the data
            self.into_boxed_slice().into_vec().set_len(0);
            m
        }
    }
}

impl<T> Vector for Mat<T> {
    type Elem = T;
}

impl<T: Zero + Clone> VectorMut for Mat<T> {
    fn set_zero(&mut self) {
        self.as_mut().set_zero();
    }
}
