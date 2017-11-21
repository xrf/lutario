//! Packed block-diagonal lower-triangular matrices.

use std::{mem, ptr, slice};
use std::marker::PhantomData;
use super::tri_matrix::{TriMatRef, TriMatMut, TriMatDim};

/// ugly: there's a lot of hidden invariants here that we aren't
/// being explicit about
#[derive(Clone, Copy, Debug)]
pub struct BlockTriMatShape<'a> {
    num_blocks: usize,
    block_offsets: *const usize,
    block_dims: *const TriMatDim,
    phantom: PhantomData<&'a ()>,
}

unsafe impl<'a> Send for BlockTriMatShape<'a> {}
unsafe impl<'a> Sync for BlockTriMatShape<'a> {}

impl<'a> BlockTriMatShape<'a> {
    /// invariants:
    ///
    /// - block_offsets must not overlap
    /// - num_blocks must be correct
    ///
    /// TODO: we need a better interface.
    /// this is not how one would normally construct the shape object;
    /// instead, one is more likely to create something that's correct
    /// by construction
    pub unsafe fn new(
        num_blocks: usize,
        block_offsets: &'a [usize],
        block_dims: &'a [TriMatDim],
    ) -> Self {
        assert_eq!(block_offsets.len(), num_blocks.checked_add(1).unwrap());
        assert_eq!(block_dims.len(), num_blocks);
        Self {
            num_blocks,
            block_offsets: block_offsets.as_ptr(),
            block_dims: block_dims.as_ptr(),
            phantom: PhantomData,
        }
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub fn num_blocks(self) -> usize {
        self.num_blocks
    }

    pub fn block_offsets(self) -> &'a [usize] {
        unsafe {
            slice::from_raw_parts(self.block_offsets, self.num_blocks)
        }
    }

    pub fn block_dims(self) -> &'a [TriMatDim] {
        unsafe {
            slice::from_raw_parts(self.block_dims as _, self.num_blocks)
        }
    }

    pub unsafe fn offset_unchecked(self, l: usize) -> usize {
        *self.block_offsets().get_unchecked(l)
    }

    pub unsafe fn dim_unchecked(self, l: usize) -> TriMatDim {
        *self.block_dims().get_unchecked(l)
    }
}

/// A block-diagonal matrix.
pub struct BlockTriMatRef<'a, T: 'a> {
    ptr: *const T,
    shape: BlockTriMatShape<'a>,
    phantom: PhantomData<&'a T>,
}

unsafe impl<'a, T: Sync> Send for BlockTriMatRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for BlockTriMatRef<'a, T> {}
impl<'a, T> Clone for BlockTriMatRef<'a, T> { fn clone(&self) -> Self { *self } }
impl<'a, T> Copy for BlockTriMatRef<'a, T> {}

impl<'a, T> BlockTriMatRef<'a, T> {
    pub unsafe fn from_raw(ptr: *const T, shape: BlockTriMatShape<'a>) -> Self {
        Self { ptr, shape, phantom: PhantomData }
    }

    pub fn as_ptr(self) -> *const T {
        self.ptr
    }

    pub fn shape(self) -> BlockTriMatShape<'a> {
        self.shape
    }

    pub fn get(self, l: usize) -> Option<TriMatRef<'a, T>> {
        if l < self.shape().num_blocks() {
            Some(unsafe { self.get_unchecked(l) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(self, l: usize) -> TriMatRef<'a, T> {
        TriMatRef::from_raw(
            self.as_ptr().offset(self.shape().offset_unchecked(l) as _),
            self.shape().dim_unchecked(l),
        )
    }
}

/// A mutable block-diagonal matrix.
pub struct BlockTriMatMut<'a, T: 'a> {
    ptr: *mut T,
    shape: BlockTriMatShape<'a>,
    phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for BlockTriMatMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for BlockTriMatMut<'a, T> {}

impl<'a, T> BlockTriMatMut<'a, T> {
    pub unsafe fn from_raw(ptr: *mut T, shape: BlockTriMatShape<'a>) -> Self {
        Self { ptr, shape, phantom: PhantomData }
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn shape(&self) -> BlockTriMatShape<'a> {
        self.shape
    }

    pub fn into_ref(self) -> BlockTriMatRef<'a, T> {
        unsafe { mem::transmute(self.as_ref()) }
    }

    pub fn as_ref<'b>(&'b self) -> BlockTriMatRef<'b, T> {
        unsafe { BlockTriMatRef::from_raw(self.as_ptr(), self.shape()) }
    }

    pub fn as_mut<'b>(&'b mut self) -> BlockTriMatMut<'b, T> {
        unsafe { ptr::read(self) }
    }

    pub fn get(self, l: usize) -> Option<TriMatMut<'a, T>> {
        if l < self.shape().num_blocks() {
            Some(unsafe { self.get_unchecked(l) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(self, l: usize) -> TriMatMut<'a, T> {
        TriMatMut::from_raw(
            self.as_ptr().offset(self.shape().offset_unchecked(l) as _),
            self.shape().dim_unchecked(l),
        )
    }
}
