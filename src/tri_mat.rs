//! Row-major, lower-triangular matrices (`i ≥ j`).

use std::{fmt, mem, slice};
use std::marker::PhantomData;
use std::ops::{AddAssign, Deref, Index, IndexMut, Mul, MulAssign, Range};
use num::{FromPrimitive, Zero};
use super::mat::MatRef;
use super::op::{Vector, VectorMut};
use super::utils::{self, Offset, try_cast};

/// Dimensions of a non-strict lower triangular matrix, equal to the number of
/// rows or columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TriMatDim(usize);

impl Deref for TriMatDim {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TriMatDim {
    pub fn is_valid(dim: usize) -> bool {
        // make sure (dim + 1) * dim / 2 doesn't overflow
        || -> Result<isize, ()> {
            try_cast(
                dim
                    .checked_add(1).ok_or(())?
                    .checked_mul(dim).ok_or(())?
                    / 2
            ).ok_or(())
        }().is_ok()
    }

    pub unsafe fn from_raw(dim: usize) -> Self {
        TriMatDim(dim)
    }

    pub fn new(dim: usize) -> Result<Self, usize> {
        if Self::is_valid(dim) {
            Ok(unsafe { Self::from_raw(dim) })
        } else {
            Err(dim)
        }
    }

    pub fn extent(self) -> usize {
        Self::raw_index(self.0, 0)
    }

    pub fn contains(self, i: usize, j: usize) -> bool {
        i >= j && j < self.0
    }

    pub fn raw_index(i: usize, j: usize) -> usize {
        i * (i + 1) / 2 + j
    }

    pub unsafe fn offset_unchecked<P: Offset>(self, ptr: P,
                                              i: usize, j: usize) -> P {
        ptr.offset(Self::raw_index(i, j) as _)
    }

    pub fn row_width(self, i: usize) -> usize {
        i + 1
    }

    pub fn upcast_slice(slice: &[Self]) -> &[usize] {
        unsafe { mem::transmute(slice) }
    }

    pub unsafe fn downcast_slice(slice: &[usize]) -> &[Self] {
        for &dim in slice {
            debug_assert!(TriMatDim::is_valid(dim));
        }
        mem::transmute(slice)
    }
}

/// Dimensions of a strict lower triangular matrix, equal to the number of
/// rows or columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct StrictTriMatDim(usize);

impl Deref for StrictTriMatDim {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StrictTriMatDim {
    pub fn is_valid(dim: usize) -> bool {
        // make sure (dim - 1) * dim / 2 doesn't overflow
        || -> Result<isize, ()> {
            try_cast(
                dim
                    .checked_sub(1).ok_or(())?
                    .checked_mul(dim).ok_or(())?
                    / 2
            ).ok_or(())
        }().is_ok()
    }

    pub unsafe fn from_raw(dim: usize) -> Self {
        StrictTriMatDim(dim)
    }

    pub fn new(dim: usize) -> Result<Self, usize> {
        if Self::is_valid(dim) {
            Ok(unsafe { Self::from_raw(dim) })
        } else {
            Err(dim)
        }
    }

    pub fn extent(self) -> usize {
        Self::raw_index(self.0, 0)
    }

    pub fn contains(self, i: usize, j: usize) -> bool {
        i > j && j < self.0
    }

    pub fn raw_index(i: usize, j: usize) -> usize {
        // wrapping is harmless here
        i * i.wrapping_sub(1) / 2 + j
    }

    pub fn row_width(self, i: usize) -> usize {
        i
    }
}

pub struct TriMatRef<'a, T: 'a> {
    phantom: PhantomData<&'a [T]>,
    ptr: *const T,
    shape: TriMatDim,
}

unsafe impl<'a, T: Sync> Send for TriMatRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for TriMatRef<'a, T> {}

impl<'a, T> Clone for TriMatRef<'a, T> { fn clone(&self) -> Self { *self } }
impl<'a, T> Copy for TriMatRef<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for TriMatRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("tri_mat!")?;
        let mut rows = f.debug_list();
        for row in self.rows() {
            rows.entry(&row);
        }
        rows.finish()
    }
}

impl<'a, T> Default for TriMatRef<'a, T> {
    fn default() -> Self {
        Self::new(&mut (&[] as &[_]), Default::default()).expect("!?")
    }
}

impl<'a, 'b, T: PartialEq<U>, U> PartialEq<TriMatRef<'b, U>> for TriMatRef<'a, T> {
    fn eq(&self, rhs: &TriMatRef<'b, U>) -> bool {
        if self.shape() != rhs.shape() {
            return false;
        }
        for i in 0 .. *self.shape() {
            for j in 0 .. i + 1 {
                if (*self).index(i, j) != (*rhs).index(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T: Eq> Eq for TriMatRef<'a, T> {}

impl<'a, T> Index<(usize, usize)> for TriMatRef<'a, T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        (*self).index(i, j)
    }
}

impl<'a, T> Vector for TriMatRef<'a, T> {
    type Elem = T;
    fn len(&self) -> usize {
        self.shape().extent()
    }
}

impl<'a, T> TriMatRef<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a [T], shape: TriMatDim) -> Option<Self> {
        unsafe {
            utils::chop_slice(slice, shape.extent()).map(|slice| {
                Self::from_raw(slice.as_ptr(), shape)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *const T, shape: TriMatDim) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    pub fn shape(self) -> TriMatDim {
        self.shape
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

    pub fn rows(self) -> TriMatRows<'a, T> {
        let num_rows = *self.shape();
        TriMatRows { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a [T]> {
        unsafe {
            if i < *self.shape() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(self, i: usize) -> &'a [T] {
        let ptr = self.offset_unchecked(i, 0);
        slice::from_raw_parts(ptr, self.shape().row_width(i))
    }

    pub fn to_slice(self) -> &'a [T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.shape().extent())
        }
    }
}

pub struct TriMatMut<'a, T: 'a> {
    phantom: PhantomData<&'a mut [T]>,
    ptr: *mut T,
    shape: TriMatDim,
}

unsafe impl<'a, T: Send> Send for TriMatMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for TriMatMut<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for TriMatMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<'a, T> Default for TriMatMut<'a, T> {
    fn default() -> Self {
        Self::new(&mut (&mut [] as &mut [_]), Default::default()).expect("!?")
    }
}

impl<'a, 'b, T, U> PartialEq<TriMatMut<'b, U>> for TriMatMut<'a, T>
    where T: PartialEq<U>
{
    fn eq(&self, rhs: &TriMatMut<'b, U>) -> bool {
        self.as_ref().eq(&rhs.as_ref())
    }
}

impl<'a, T: Eq> Eq for TriMatMut<'a, T> {}

impl<'a, T> Index<(usize, usize)> for TriMatMut<'a, T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        self.as_ref().index(i, j)
    }
}

impl<'a, T> IndexMut<(usize, usize)> for TriMatMut<'a, T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.as_mut().index(i, j)
    }
}

impl<'a, T> Vector for TriMatMut<'a, T> {
    type Elem = T;
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<'a, T: MulAssign + Zero + Clone> VectorMut for TriMatMut<'a, T> {
    fn set_zero(&mut self) {
        self.as_mut().fill(&Zero::zero());
    }

    fn scale(&mut self, factor: &Self::Elem) {
        for x in self.as_mut().to_slice() {
            *x *= factor.clone();
        }
    }
}

impl<'a, T> TriMatMut<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a mut [T], shape: TriMatDim) -> Option<Self> {
        unsafe {
            utils::chop_slice_mut(slice, shape.extent()).map(|slice| {
                Self::from_raw(slice.as_mut_ptr(), shape)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *mut T, shape: TriMatDim) -> Self {
        Self { phantom: PhantomData, ptr, shape }
    }

    pub fn shape(&self) -> TriMatDim {
        self.as_ref().shape()
    }

    pub fn into_ref(self) -> TriMatRef<'a, T> {
        unsafe {
            TriMatRef::from_raw(self.ptr, self.shape)
        }
    }

    pub fn as_ref(&self) -> TriMatRef<T> {
        unsafe {
            TriMatRef::from_raw(self.ptr, self.shape)
        }
    }

    pub fn as_mut(&mut self) -> TriMatMut<T> {
        unsafe {
            TriMatMut::from_raw(self.ptr, self.shape)
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

    pub fn rows(self) -> TriMatRowsMut<'a, T> {
        let num_rows = *self.shape();
        TriMatRowsMut { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a mut [T]> {
        unsafe {
            if i < *self.shape() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(mut self, i: usize) -> &'a mut [T] {
        let ptr = self.offset_unchecked(i, 0) as _;
        slice::from_raw_parts_mut(ptr, self.shape().row_width(i))
    }

    pub fn to_slice(self) -> &'a mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr, self.shape().extent())
        }
    }
}

impl<'a, T: Clone> TriMatMut<'a, T> {
    pub fn fill(&mut self, value: &T) {
        for r in self.as_mut().to_slice() {
            *r = value.clone();
        }
    }

    pub fn clone_from_ref(&mut self, source: TriMatRef<T>) {
        assert_eq!(self.shape(), source.shape());
        self.as_mut().to_slice().clone_from_slice(source.to_slice());
    }
}

#[derive(Clone, Debug)]
pub struct TriMatRows<'a, T: 'a> {
    mat: TriMatRef<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for TriMatRows<'a, T> {
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

impl<'a, T> DoubleEndedIterator for TriMatRows<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = self.mat;
        self.range.next_back().map(|i| unsafe {
            mat.row_unchecked(i)
        })
    }
}

impl<'a, T> ExactSizeIterator for TriMatRows<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

#[derive(Debug)]
pub struct TriMatRowsMut<'a, T: 'a> {
    mat: TriMatMut<'a, T>,
    range: Range<usize>,
}

impl<'a, T> Iterator for TriMatRowsMut<'a, T> {
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

impl<'a, T> DoubleEndedIterator for TriMatRowsMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let mat = &mut self.mat;
        self.range.next_back().map(|i| unsafe {
            mem::transmute(mat.as_mut().row_unchecked(i))
        })
    }
}

impl<'a, T> ExactSizeIterator for TriMatRowsMut<'a, T> {
    fn len(&self) -> usize {
        self.range.len()
    }
}

pub struct TriMat<T> {
    phantom: PhantomData<Box<[T]>>,
    ptr: *mut T,
    shape: TriMatDim,
}

impl<T: fmt::Debug> fmt::Debug for TriMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T: Clone> Clone for TriMat<T> {
    fn clone(&self) -> Self {
        Self::from_vec(self.as_slice().to_vec(), *self.shape())
    }
    fn clone_from(&mut self, source: &Self) {
        // simplest solution that satisfies panic safety
        if self.extent() != source.extent() {
            *self = source.clone();
        } else {
            // might panic halfway, in which case the data is corrupt but
            // there still shouldn't be any memory unsafety
            self.as_slice_mut().clone_from_slice(source.as_slice());
            self.shape = source.shape;
        }
    }
}

impl<T> Drop for TriMat<T> {
    fn drop(&mut self) {
        unsafe {
            mem::replace(self, mem::uninitialized()).into_boxed_slice();
        }
    }
}

impl<T> Default for TriMat<T> {
    fn default() -> Self {
        Self::from_vec(vec![], 0)
    }
}

impl<T: PartialEq<U>, U> PartialEq<TriMat<U>> for TriMat<T> {
    fn eq(&self, rhs: &TriMat<U>) -> bool {
        self.as_ref().eq(&rhs.as_ref())
    }
}

impl<T: Eq> Eq for TriMat<T> {}

impl<T: Clone> TriMat<T> {
    pub fn replicate(dim: usize, value: T) -> Self {
        let dim = TriMatDim::new(dim).unwrap();
        unsafe {
            Self::from_vec_unchecked(vec![value; dim.extent()], dim)
        }
    }
}

impl<T: Clone + Zero> TriMat<T> {
    pub fn zero(dim: usize) -> Self {
        Self::replicate(dim, Zero::zero())
    }
}

impl<T> TriMat<T> {
    /// Panics if the vector is too short.
    pub fn from_vec(mut vec: Vec<T>, dim: usize) -> Self {
        let shape = TriMatDim::new(dim).unwrap();
        let n = shape.extent();
        assert!(vec.len() >= n);
        vec.truncate(n);
        vec.shrink_to_fit();
        unsafe {
            Self::from_vec_unchecked(vec, shape)
        }
    }

    pub unsafe fn from_vec_unchecked(mut vec: Vec<T>, shape: TriMatDim) -> Self {
        // in order to reconstruct the vector correctly during Drop,
        // we need to ensure capacity == num_rows * num_cols
        debug_assert_eq!(vec.capacity(), shape.extent());
        debug_assert_eq!(vec.len(), shape.extent());
        let ptr = vec.as_mut_ptr();
        mem::forget(vec);
        Self::from_raw(ptr, shape)
    }

    pub unsafe fn from_raw(ptr: *mut T, shape: TriMatDim) -> Self {
        Self {
            phantom: PhantomData,
            ptr,
            shape
        }
    }

    pub fn shape(&self) -> TriMatDim {
        self.shape
    }

    pub fn extent(&self) -> usize {
        self.shape().extent()
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn as_ref(&self) -> TriMatRef<T> {
        unsafe { TriMatRef::from_raw(self.as_ptr(), self.shape()) }
    }

    pub fn as_mut(&mut self) -> TriMatMut<T> {
        unsafe { TriMatMut::from_raw(self.as_ptr(), self.shape()) }
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
}

impl<T> Vector for TriMat<T> {
    type Elem = T;
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<T: MulAssign + Zero + Clone> VectorMut for TriMat<T> {
    fn set_zero(&mut self) {
        self.as_mut().set_zero();
    }

    fn scale(&mut self, factor: &Self::Elem) {
        self.as_mut().scale(factor);
    }
}

/// Transpose symmetry.
pub trait Trs<T> {
    fn trs(&self, value: T) -> T;
}

pub mod trs {
    use std::ops::Neg;
    use super::super::linalg::Conj;
    use super::Trs;

    /// Marks an Hermitian matrix.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct He;

    impl<T: Conj> Trs<T> for He {
        fn trs(&self, value: T) -> T {
            value.conj()
        }
    }

    /// Marks an antihermitian matrix.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Ah;

    impl<T: Conj + Neg<Output = T>> Trs<T> for Ah {
        fn trs(&self, value: T) -> T {
            -value.conj()
        }
    }

    /// Marks a symmetric matrix.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Sy;

    impl<T> Trs<T> for Sy {
        fn trs(&self, value: T) -> T {
            value
        }
    }

    /// Marks an antisymmetric matrix.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct As;

    impl<T: Neg<Output = T>> Trs<T> for As {
        fn trs(&self, value: T) -> T {
            -value
        }
    }
}

/// Matrices that possess symmetry under transposition, represented as
/// triangular matrices.
pub struct TrsMat<S, T> {
    pub mat: TriMat<T>,
    pub trs: S,
}

impl<S, T> Vector for TrsMat<S, T> {
    type Elem = T;
    fn len(&self) -> usize {
        Vector::len(&self.mat)
    }
}

impl<S, T: MulAssign + Zero + Clone> VectorMut for TrsMat<S, T> {
    fn set_zero(&mut self) {
        self.mat.as_mut().set_zero();
    }

    fn scale(&mut self, factor: &Self::Elem) {
        self.mat.as_mut().scale(factor);
    }
}

impl<S: Trs<T>, T: Clone> TrsMat<S, T> {
    pub fn get(&self, i: usize, j: usize) -> Result<T, (usize, usize)> {
        let (i2, j2) = if i >= j {
            (i, j)
        } else {
            (j, i)
        };
        let value = self.mat.as_ref().get(i2, j2).ok_or((i, j))?.clone();
        Ok(if i >= j {
            value
        } else {
            self.trs.trs(value)
        })
    }
}

impl<S: Trs<T>, T> TrsMat<S, T> {
    pub fn set(
        &mut self,
        i: usize,
        j: usize,
        mut value: T,
    ) -> Result<(), (usize, usize)>
    {
        let (i2, j2) = if i >= j {
            (i, j)
        } else {
            value = self.trs.trs(value);
            (j, i)
        };
        *self.mat.as_mut().get(i2, j2).ok_or((i, j))? = value;
        Ok(())
    }
}

impl<S: Trs<T>, T: AddAssign + Mul<Output = T> + FromPrimitive> TrsMat<S, T> {
    pub fn add(
        &mut self,
        i: usize,
        j: usize,
        mut value: T,
    ) -> Result<(), (usize, usize)>
    {
        let (i2, j2) = if i >= j {
            (i, j)
        } else {
            value = self.trs.trs(value);
            (j, i)
        };
        let factor = T::from_f64(if i == j { 1.0 } else { 0.5 }).unwrap();
        *self.mat.as_mut().get(i2, j2).ok_or((i, j))? += factor * value;
        Ok(())
    }
}

impl<S, T> TrsMat<S, T> where
    S: Trs<T>,
    T: AddAssign + Mul<Output = T> + MulAssign + FromPrimitive + Zero + Clone,
{
    pub fn clone_from_mat(&mut self, source: MatRef<T>) {
        let n = *self.mat.as_ref().shape();
        assert_eq!(n, source.shape().num_rows);
        assert_eq!(n, source.shape().num_cols);
        self.set_zero();
        for i in 0 .. source.shape().num_rows {
            for j in 0 .. source.shape().num_cols {
                self.add(i, j, source.get(i, j).unwrap().clone()).unwrap();
            }
        }
    }
}
