//! Packed block-diagonal matrices.

use super::linalg::{self, EigenvalueRange, Gemm, Heevr, Part, Transpose};
use super::mat::{MatMut, MatRef, MatShape, ValidMatShape};
use super::op::{IndexBlockMatMut, IndexBlockMatRef, Vector};
use super::utils;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::{mem, ptr, slice};

/// A block-diagonal matrix with unspecified lifetime and unspecified type.
#[derive(Clone, Copy, Debug)]
pub struct RawBlockMatRef {
    ptr: *mut (),
    num_blocks: usize,
    block_offsets: *const usize,
    block_strides: *const usize,
    block_num_rows: *const usize,
    block_num_cols: *const usize,
}

impl RawBlockMatRef {
    /// Unsafe preconditions:
    ///
    ///   - `ptr` must point to an array with sufficient length.
    ///   - All the indexing arrays must be at least as long as `num_blocks`.
    ///
    pub unsafe fn new(
        ptr: *mut (),
        num_blocks: usize,
        block_offsets: *const usize,
        block_strides: *const usize,
        block_num_rows: *const usize,
        block_num_cols: *const usize,
    ) -> Self {
        Self {
            ptr,
            num_blocks,
            block_offsets,
            block_strides,
            block_num_rows,
            block_num_cols,
        }
    }

    /// Unsafe preconditions: `T` must be the correct type.
    pub unsafe fn as_ptr<T>(self) -> *const T {
        self.ptr as _
    }

    /// Unsafe preconditions: `T` must be the correct type.
    pub unsafe fn mut_ptr<T>(self) -> *mut T {
        self.ptr as _
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub fn num_blocks(self) -> usize {
        self.num_blocks
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub unsafe fn block_offsets<'a>(self) -> &'a [usize] {
        slice::from_raw_parts(self.block_offsets, self.num_blocks)
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub unsafe fn block_strides<'a>(self) -> &'a [usize] {
        slice::from_raw_parts(self.block_strides, self.num_blocks)
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub unsafe fn block_num_rows<'a>(self) -> &'a [usize] {
        slice::from_raw_parts(self.block_num_rows, self.num_blocks)
    }

    /// Unsafety: No attempt is made to constrain the lifetime.
    pub unsafe fn block_num_cols<'a>(self) -> &'a [usize] {
        slice::from_raw_parts(self.block_num_cols, self.num_blocks)
    }

    pub unsafe fn shape_at(self, l: usize) -> ValidMatShape {
        MatShape {
            stride: *self.block_strides().get_unchecked(l),
            num_rows: *self.block_num_rows().get_unchecked(l),
            num_cols: *self.block_num_cols().get_unchecked(l),
        }
        .assert_valid()
    }

    pub unsafe fn offset_at(self, l: usize) -> usize {
        *self.block_offsets().get_unchecked(l)
    }
}

/// A block-diagonal matrix.
pub struct BlockMatRef<'a, T: 'a> {
    raw: RawBlockMatRef,
    phantom: PhantomData<&'a T>,
}

unsafe impl<'a, T: Sync> Send for BlockMatRef<'a, T> {}
unsafe impl<'a, T: Sync> Sync for BlockMatRef<'a, T> {}
impl<'a, T> Clone for BlockMatRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for BlockMatRef<'a, T> {}

pub fn block_mat_extent(
    num_blocks: usize,
    block_offsets: &[usize],
    block_strides: &[usize],
    block_num_rows: &[usize],
    block_num_cols: &[usize],
) -> usize {
    assert!(block_strides.len() >= num_blocks);
    assert!(block_num_rows.len() >= num_blocks);
    assert!(block_num_cols.len() >= num_blocks);

    // check for overlaps
    let mut i = 0;
    for l in 0..num_blocks {
        let shape = MatShape {
            stride: block_strides[l],
            num_rows: block_num_rows[l],
            num_cols: block_num_cols[l],
        }
        .validate()
        .unwrap();
        let new_i = block_offsets[l]
            .checked_add(shape.extent())
            .expect("arithmetic overflow");
        assert!(new_i >= i);
        i = new_i;
    }
    i
}

impl<'a, T> BlockMatRef<'a, T> {
    pub fn new(
        slice: &mut &'a [T],
        num_blocks: usize,
        block_offsets: &'a [usize],
        block_strides: &'a [usize],
        block_num_rows: &'a [usize],
        block_num_cols: &'a [usize],
    ) -> Self {
        let mine = utils::chop_slice(
            slice,
            block_mat_extent(
                num_blocks,
                block_offsets,
                block_strides,
                block_num_rows,
                block_num_cols,
            ),
        )
        .expect("slice too short");
        unsafe {
            Self::from_raw(RawBlockMatRef::new(
                mine.as_ptr() as _,
                num_blocks,
                block_offsets.as_ptr(),
                block_strides.as_ptr(),
                block_num_rows.as_ptr(),
                block_num_cols.as_ptr(),
            ))
        }
    }

    pub unsafe fn from_raw(raw: RawBlockMatRef) -> Self {
        Self {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn as_ptr(self) -> *const T {
        unsafe { self.raw.as_ptr() }
    }

    pub fn num_blocks(self) -> usize {
        self.raw.num_blocks()
    }

    pub fn block_offsets(self) -> &'a [usize] {
        unsafe { self.raw.block_offsets() }
    }

    pub fn block_strides(self) -> &'a [usize] {
        unsafe { self.raw.block_strides() }
    }

    pub fn block_num_rows(self) -> &'a [usize] {
        unsafe { self.raw.block_num_rows() }
    }

    pub fn block_num_cols(self) -> &'a [usize] {
        unsafe { self.raw.block_num_cols() }
    }

    pub fn get(self, l: usize) -> Option<MatRef<'a, T>> {
        if l < self.num_blocks() {
            Some(unsafe { self.get_unchecked(l) })
        } else {
            None
        }
    }

    pub fn index(self, l: usize) -> MatRef<'a, T> {
        self.get(l).unwrap()
    }

    pub unsafe fn get_unchecked(self, l: usize) -> MatRef<'a, T> {
        MatRef::from_raw(
            self.as_ptr().offset(self.raw.offset_at(l) as _),
            self.raw.shape_at(l),
        )
    }

    pub unsafe fn slice(
        self,
        block_offsets: &'a [usize],
        block_num_rows: &'a [usize],
        block_num_cols: &'a [usize],
    ) -> Self {
        Self::from_raw(RawBlockMatRef::new(
            self.raw.mut_ptr(),
            self.raw.num_blocks(),
            block_offsets.as_ptr(),
            self.raw.block_strides().as_ptr(),
            block_num_rows.as_ptr(),
            block_num_cols.as_ptr(),
        ))
    }
}

impl<'a, T> Vector for BlockMatRef<'a, T> {
    type Elem = T;
    fn len(&self) -> usize {
        (0..self.num_blocks())
            .map(|l| Vector::len(&self.get(l).unwrap()))
            .sum()
    }
}

impl<'a, T: Clone> IndexBlockMatRef for BlockMatRef<'a, T> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.get(l).unwrap().get(u1, u2).unwrap().clone()
    }
}

/// A mutable block-diagonal matrix.
pub struct BlockMatMut<'a, T: 'a> {
    raw: RawBlockMatRef,
    phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for BlockMatMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for BlockMatMut<'a, T> {}

impl<'a, T> BlockMatMut<'a, T> {
    pub unsafe fn from_raw(raw: RawBlockMatRef) -> Self {
        Self {
            raw,
            phantom: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *mut T {
        unsafe { self.raw.mut_ptr() }
    }

    pub fn into_ref(self) -> BlockMatRef<'a, T> {
        unsafe { mem::transmute(self.as_ref()) }
    }

    pub fn as_ref<'b>(&'b self) -> BlockMatRef<'b, T> {
        unsafe { BlockMatRef::from_raw(self.raw) }
    }

    pub fn as_mut<'b>(&'b mut self) -> BlockMatMut<'b, T> {
        unsafe { ptr::read(self) }
    }

    pub fn get(self, l: usize) -> Option<MatMut<'a, T>> {
        if l < self.as_ref().num_blocks() {
            Some(unsafe { self.get_unchecked(l) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(self, l: usize) -> MatMut<'a, T> {
        MatMut::from_raw(
            self.as_ptr().offset(self.raw.offset_at(l) as _),
            self.raw.shape_at(l),
        )
    }
}

impl<'a, T> Vector for BlockMatMut<'a, T> {
    type Elem = T;
    fn len(&self) -> usize {
        (0..self.as_ref().num_blocks())
            .map(|l| Vector::len(&self.as_ref().get(l).unwrap()))
            .sum()
    }
}

impl<'a, T: Clone> IndexBlockMatRef for BlockMatMut<'a, T> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.as_ref().get(l).unwrap().get(u1, u2).unwrap().clone()
    }
}

impl<'a, T: AddAssign> IndexBlockMatMut for BlockMatMut<'a, T> {
    fn set_block_mat(&mut self, l: usize, u1: usize, u2: usize, value: Self::Elem) {
        *self.as_mut().get(l).unwrap().get(u1, u2).unwrap() = value;
    }

    fn add_block_mat(&mut self, l: usize, u1: usize, u2: usize, value: Self::Elem) {
        *self.as_mut().get(l).unwrap().get(u1, u2).unwrap() += value;
    }
}

pub fn block_gemm<T: Gemm>(
    transa: Transpose,
    transb: Transpose,
    alpha: T,
    a: BlockMatRef<T>,
    b: BlockMatRef<T>,
    beta: T,
    mut c: BlockMatMut<T>,
) {
    let num_blocks = a.num_blocks();
    assert_eq!(num_blocks, b.num_blocks());
    assert_eq!(num_blocks, c.as_ref().num_blocks());
    for l in 0..num_blocks {
        unsafe {
            linalg::gemm(
                transa,
                transb,
                alpha,
                a.get_unchecked(l),
                b.get_unchecked(l),
                beta,
                c.as_mut().get_unchecked(l),
            )
        }
    }
}

pub fn block_heevr<T: Heevr>(
    left: bool,
    uplo: Part,
    mut a: BlockMatMut<T>,
    abstol: T::Real,
    mut w: BlockMatMut<T::Real>,
    mut z: BlockMatMut<T>,
    mut isuppz: Option<BlockMatMut<i32>>,
    m: &mut [usize],
) -> Result<(), i32>
where
    T::Real: Clone,
{
    // not implemented: range: EigenvalueRange<T::Real>,
    // because selecting eigenvalues by index makes no sense here!
    let mut isuppz_buf = Vec::new();
    for l in 0..a.as_ref().num_blocks() {
        m[l] = linalg::heevr(
            left,
            EigenvalueRange::All,
            uplo,
            a.as_mut().get(l).unwrap(),
            abstol.clone(),
            w.as_mut().get(l).unwrap().row(0).unwrap(),
            z.as_mut().get(l).unwrap(),
            match isuppz {
                None => Err(&mut isuppz_buf),
                Some(ref mut isuppz) => Ok(isuppz.as_mut().get(l).unwrap().row(0).unwrap()),
            },
        )?;
    }
    Ok(())
}
