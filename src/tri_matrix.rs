//! Row-major, lower-triangular matrices (`i > j`).

use std::{fmt, mem, slice};
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut, Range};
use super::utils::{self, Offset, try_cast};

/// Number of rows (or, equivalently, columns).
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

    pub fn extent(&self) -> usize {
        Self::raw_index(self.0, 0)
    }

    pub fn contains(&self, i: usize, j: usize) -> bool {
        i < self.0 && j < self.0
    }

    pub fn raw_index(i: usize, j: usize) -> usize {
        i * (i + 1) / 2 + j
    }

    pub unsafe fn offset_unchecked<P: Offset>(&self, ptr: P,
                                              i: usize, j: usize) -> P {
        ptr.offset(Self::raw_index(i, j) as _)
    }

    pub fn row_width(&self, i: usize) -> usize {
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

pub struct TriMat<'a, T: 'a> {
    phantom: PhantomData<&'a [T]>,
    ptr: *const T,
    dim: TriMatDim,
}

unsafe impl<'a, T: Sync> Send for TriMat<'a, T> {}
unsafe impl<'a, T: Sync> Sync for TriMat<'a, T> {}

impl<'a, T> Clone for TriMat<'a, T> { fn clone(&self) -> Self { *self } }
impl<'a, T> Copy for TriMat<'a, T> {}

impl<'a, T: fmt::Debug> fmt::Debug for TriMat<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("tri_mat!")?;
        let mut rows = f.debug_list();
        for row in self.rows() {
            rows.entry(&row);
        }
        rows.finish()
    }
}

impl<'a, T> Default for TriMat<'a, T> {
    fn default() -> Self {
        Self::new(&mut (&[] as &[_]), Default::default()).expect("!?")
    }
}

impl<'a, 'b, T: PartialEq<U>, U> PartialEq<TriMat<'b, U>> for TriMat<'a, T> {
    fn eq(&self, rhs: &TriMat<'b, U>) -> bool {
        if self.dim() != rhs.dim() {
            return false;
        }
        for i in 0 .. *self.dim() {
            for j in i .. *self.dim() {
                if (*self).index(i, j) != (*rhs).index(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T: Eq> Eq for TriMat<'a, T> {}

impl<'a, T> Index<(usize, usize)> for TriMat<'a, T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        (*self).index(i, j)
    }
}

impl<'a, T> TriMat<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a [T], dim: TriMatDim) -> Option<Self> {
        unsafe {
            utils::chop_slice(slice, dim.extent()).map(|slice| {
                Self::from_raw(slice.as_ptr(), dim)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *const T, dim: TriMatDim) -> Self {
        Self { phantom: PhantomData, ptr, dim }
    }

    pub fn dim(self) -> TriMatDim {
        self.dim
    }

    pub fn as_ptr(self) -> *const T {
        self.ptr
    }

    pub fn index(self, i: usize, j: usize) -> &'a T {
        self.get(i, j).expect(&format!("out of range: ({}, {})", i, j))
    }

    pub fn get(self, i: usize, j: usize) -> Option<&'a T> {
        unsafe {
            if self.dim().contains(i, j) {
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
        self.dim().offset_unchecked(self.as_ptr(), i, j)
    }

    pub fn rows(self) -> TriMatRows<'a, T> {
        let num_rows = *self.dim();
        TriMatRows { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a [T]> {
        unsafe {
            if i < *self.dim() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(self, i: usize) -> &'a [T] {
        let ptr = self.offset_unchecked(i, 0);
        slice::from_raw_parts(ptr, self.dim().row_width(i))
    }
}

pub struct TriMatMut<'a, T: 'a> {
    phantom: PhantomData<&'a mut [T]>,
    ptr: *mut T,
    dim: TriMatDim,
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

impl<'a, T> TriMatMut<'a, T> {
    /// Split `*slice` into two parts: the first part becomes the matrix,
    /// while the second part is stored back in `slice`.  If the slice is too
    /// short, `None` is returned.
    pub fn new(slice: &mut &'a mut [T], dim: TriMatDim) -> Option<Self> {
        unsafe {
            utils::chop_slice_mut(slice, dim.extent()).map(|slice| {
                Self::from_raw(slice.as_mut_ptr(), dim)
            })
        }
    }

    pub unsafe fn from_raw(ptr: *mut T, dim: TriMatDim) -> Self {
        Self { phantom: PhantomData, ptr, dim }
    }

    pub fn dim(&self) -> TriMatDim {
        self.as_ref().dim()
    }

    pub fn into_ref(self) -> TriMat<'a, T> {
        unsafe {
            TriMat::from_raw(self.ptr, self.dim)
        }
    }

    pub fn as_ref(&self) -> TriMat<T> {
        unsafe {
            TriMat::from_raw(self.ptr, self.dim)
        }
    }

    pub fn as_mut(&mut self) -> TriMatMut<T> {
        unsafe {
            TriMatMut::from_raw(self.ptr, self.dim)
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
            if self.dim().contains(i, j) {
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
        self.dim().offset_unchecked(self.as_ptr(), i, j)
    }

    pub fn rows(self) -> TriMatRowsMut<'a, T> {
        let num_rows = *self.dim();
        TriMatRowsMut { mat: self, range: 0 .. num_rows }
    }

    pub fn row(self, i: usize) -> Option<&'a mut [T]> {
        unsafe {
            if i < *self.dim() {
                Some(self.row_unchecked(i))
            } else {
                None
            }
        }
    }

    pub unsafe fn row_unchecked(mut self, i: usize) -> &'a mut [T] {
        let ptr = self.offset_unchecked(i, 0) as _;
        slice::from_raw_parts_mut(ptr, self.dim().row_width(i))
    }
}

#[derive(Clone, Debug)]
pub struct TriMatRows<'a, T: 'a> {
    mat: TriMat<'a, T>,
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
