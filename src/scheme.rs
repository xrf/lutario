use std::mem;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use super::btree_cache::BTreeCache;
use super::block_matrix::{BlockMat, BlockMatMut};
use super::matrix::MatShape;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Basis {
    /// x P - y P
    PP(u8, u8),
    /// x I - y A
    IA(u8, u8),
    /// x A - y I
    AI(u8, u8),
}

#[derive(Debug)]
pub struct Scheme {
    // block offsets depends not just on the basis but also subbasis
    block_offsets: BTreeCache<(Basis, Basis, Basis, Basis), Box<[usize]>>,
    block_dims: BTreeMap<Basis, Box<[usize]>>,
    states: BTreeMap<Basis, Box<[usize]>>,
    slice_offsets: BTreeMap<(Basis, Basis), Box<[usize]>>,
}

impl Scheme {
    pub fn block_offsets(
        &self,
        row_basis: Basis,
        col_basis: Basis,
        row_subbasis: Basis,
        col_subbasis: Basis,
    ) -> &[usize] {
        let key = (row_basis, col_basis, row_subbasis, col_subbasis);
        self.block_offsets.get_or_insert_with(key, |_| {
            if row_basis == row_subbasis && col_basis == col_subbasis {
                let row_block_dims =
                    self.block_dims.get(&row_basis).unwrap();
                let col_block_dims =
                    self.block_dims.get(&col_basis).unwrap();
                let mut offsets = Vec::with_capacity(row_block_dims.len() + 1);
                let mut n = 0;
                for l in 0 .. row_block_dims.len() {
                    offsets.push(n);
                    let num_rows = row_block_dims[l];
                    let num_cols = col_block_dims[l];
                    let shape = MatShape::packed(
                        num_rows,
                        num_cols,
                    ).validate().unwrap();
                    n += shape.extent();
                }
                offsets.push(n);
                offsets.into_boxed_slice()
            } else {
                let row_block_dims =
                    self.block_dims.get(&row_basis).unwrap();
                let row_subblock_dims =
                    self.block_dims.get(&row_subbasis).unwrap();
                let col_block_dims =
                    self.block_dims.get(&col_basis).unwrap();
                let col_subblock_dims =
                    self.block_dims.get(&col_subbasis).unwrap();
                let base_offsets =
                    self.block_offsets(row_basis, row_basis,
                                       row_basis, col_basis);
                let row_slice_offsets =
                    self.slice_offsets.get(&(row_basis, row_subbasis)).unwrap();
                let col_slice_offsets =
                    self.slice_offsets.get(&(col_basis, col_subbasis)).unwrap();
                let mut offsets = Vec::with_capacity(base_offsets.len());
                let mut n = 0;
                for l in 0 .. row_block_dims.len() {
                    let i = row_slice_offsets[l];
                    let j = col_slice_offsets[l];
                    let sub_num_rows = row_subblock_dims[l];
                    let sub_num_cols = col_subblock_dims[l];
                    let num_rows = row_block_dims[l];
                    let num_cols = col_block_dims[l];
                    let shape = MatShape::packed(
                        num_rows,
                        num_cols,
                    ).validate().unwrap();
                    assert!(shape.contains(i, j));
                    assert!(MatShape::packed(num_rows - i, num_cols - j)
                            .validate().unwrap()
                            .contains(sub_num_rows, sub_num_cols));
                    let off = base_offsets[l] + shape.raw_index(i, j);
                    offsets.push(off);
                    n = off + shape.extent();
                }
                offsets.push(n);
                offsets.into_boxed_slice()
            }
        })
    }

    pub fn block_dims(&self, basis: Basis) -> &[usize] {
        self.block_dims.get(&basis).unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpInfo<'a> {
    // needed for slicing and also diagnostics when things go wrong
    scheme: &'a Scheme,
    row_basis: Basis,
    col_basis: Basis,
    row_subbasis: Basis,
    col_subbasis: Basis,
}

impl<'a> OpInfo<'a> {
    pub fn block_offsets(self) -> &'a [usize] {
        self.scheme.block_offsets(
            self.row_basis,
            self.col_basis,
            self.row_subbasis,
            self.col_subbasis,
        )
    }
}

pub struct Op<'a, T: 'a> {
    block_mat: BlockMat<'a, T>,
    info: OpInfo<'a>,
}

impl<'a, T> Clone for Op<'a, T> { fn clone(&self) -> Self { *self } }
impl<'a, T> Copy for Op<'a, T> {}

impl<'a, T> Op<'a, T> {
    pub fn slice(self, row_subbasis: Basis, col_subbasis: Basis) -> Op<'a, T> {
        let info = OpInfo { row_subbasis, col_subbasis, .. self.info };
        Op {
            block_mat: unsafe { self.block_mat.slice(
                info.block_offsets(),
                self.info.scheme.block_dims(row_subbasis),
                self.info.scheme.block_dims(col_subbasis),
            ) },
            info,
        }
    }
}

pub struct OpMut<'a, T: 'a> {
    block_mat: BlockMatMut<'a, T>,
    info: OpInfo<'a>,
}
