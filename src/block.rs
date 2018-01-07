//! Block-diagonal matrices and similar things.
use std::ops::AddAssign;
use num::Zero;
use super::mat::Mat;
use super::op::{IndexBlockMatRef, IndexBlockMatMut, Vector, VectorMut};
use super::tri_mat::{TriMat, Trs};
use super::utils::RefAdd;

/// A block of a block-diagonal matrix
pub struct Block<M> {
    /// Channel index of this block
    pub chan: usize,
    pub data: M,
}

impl<M: RefAdd> RefAdd for Block<M> {
    fn ref_add(&self, rhs: &Self) -> Self {
        assert_eq!(self.chan, rhs.chan);
        Block {
            chan: self.chan,
            data: self.data.ref_add(&rhs.data),
        }
    }
}

impl<M: Vector> Vector for Block<M> {
    type Elem = M::Elem;
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<M: VectorMut> VectorMut for Block<M> {
    fn set_zero(&mut self) {
        self.data.set_zero();
    }

    fn scale(&mut self, factor: &Self::Elem) {
        self.data.scale(factor);
    }
}

impl<T: Clone + Zero> IndexBlockMatRef for Block<Vec<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        debug_assert_eq!(self.chan, l);
        debug_assert_eq!(u1, u2);
        self.data[u1].clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Block<Mat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        debug_assert_eq!(self.chan, l);
        self.data.as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Block<TriMat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        debug_assert_eq!(self.chan, l);
        self.data.as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: AddAssign + Zero + Clone> IndexBlockMatMut for Block<Vec<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        debug_assert_eq!(u1, u2);
        self.data[u1] = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        debug_assert_eq!(u1, u2);
        self.data[u1] += value;
    }
}

impl<T: AddAssign> IndexBlockMatMut for Block<Mat<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        *self.data.as_mut().get(u1, u2).unwrap() = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        *self.data.as_mut().get(u1, u2).unwrap() += value;
    }
}

impl<T: AddAssign> IndexBlockMatMut for Block<TriMat<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        *self.data.as_mut().get(u1, u2).unwrap() = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(self.chan, l);
        *self.data.as_mut().get(u1, u2).unwrap() += value;
    }
}

/// A block-diagonal matrix is a vector of matrix blocks.
///
/// This wrapper is necessary to avoid conflicting implementations of the
/// vector traits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bd<M>(pub Vec<M>);

impl<T> Bd<Mat<T>> {
    pub fn extent_mat_as_tri(&self) -> usize {
        self.0.iter().map(|block| block.as_ref().extent_as_tri()).sum()
    }
}

impl<T: Clone> Bd<Mat<T>> {
    /// Pack the lower triangles of the matrix blocks into the given array and
    /// then return the remaining part of the array.
    pub fn clone_mat_to_tri_slice<'a>(
        &self,
        mut a: &'a mut [T],
    ) -> &'a mut [T]
    {
        for block in &self.0 {
            a = block.as_ref().clone_to_tri_slice(move_ref!(a));
        }
        a
    }

    pub fn clone_mat_from_tri_slice<'a, S: Trs<T>>(
        &mut self,
        trs: &S,
        mut a: &'a [T],
    ) -> &'a [T]
    {
        for block in &mut self.0 {
            a = block.as_mut().clone_from_tri_slice(trs, a);
        }
        a
    }
}

impl<M: RefAdd> RefAdd for Bd<M> {
    fn ref_add(&self, rhs: &Self) -> Self {
        Bd(self.0.iter().zip(&rhs.0).map(|(x, y)| x.ref_add(y)).collect())
    }
}

impl<M: Vector> Vector for Bd<M> {
    type Elem = M::Elem;
    fn len(&self) -> usize {
        self.0.iter().map(Vector::len).sum()
    }
}

impl<M: VectorMut> VectorMut for Bd<M> {
    fn set_zero(&mut self) {
        for block in &mut self.0 {
            block.set_zero();
        }
    }

    fn scale(&mut self, factor: &Self::Elem) {
        for block in &mut self.0 {
            block.scale(factor);
        }
    }
}

impl<T: Zero + Clone> IndexBlockMatRef for Bd<Vec<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        debug_assert_eq!(u1, u2);
        self.0[l][u1].clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Bd<Mat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.0[l].as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Bd<TriMat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.0[l].as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: AddAssign + Zero + Clone> IndexBlockMatMut for Bd<Vec<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(u1, u2);
        self.0[l][u1] = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        debug_assert_eq!(u1, u2);
        self.0[l][u1] += value;
    }
}

impl<T: AddAssign> IndexBlockMatMut for Bd<Mat<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        *self.0[l].as_mut().get(u1, u2).unwrap() = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        *self.0[l].as_mut().get(u1, u2).unwrap() += value;
    }
}

impl<T: AddAssign> IndexBlockMatMut for Bd<TriMat<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        *self.0[l].as_mut().get(u1, u2).unwrap() = value;
    }

    fn add_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        *self.0[l].as_mut().get(u1, u2).unwrap() += value;
    }
}
