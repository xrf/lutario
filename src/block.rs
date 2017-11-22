//! Block-diagonal matrices and similar things.
use std::ops::AddAssign;
use super::mat::Mat;
use super::op::{IndexBlockMatRef, IndexBlockMatMut, Vector, VectorMut};
use super::tri_mat::TriMat;

/// A block matrix is a vector of matrix blocks.
///
/// This wrapper is necessary to avoid conflicting implementations of the
/// vector traits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block<M>(pub Vec<M>);

impl<M: Vector> Vector for Block<M> {
    type Elem = M::Elem;
}

impl<T: VectorMut> VectorMut for Block<T> {
    fn set_zero(&mut self) {
        for block in &mut self.0 {
            block.set_zero();
        }
    }
}

impl<T: Clone> IndexBlockMatRef for Block<Vec<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        assert_eq!(u1, u2);
        self.0[l][u1].clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Block<Mat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.0[l].as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: Clone> IndexBlockMatRef for Block<TriMat<T>> {
    fn at_block_mat(&self, l: usize, u1: usize, u2: usize) -> Self::Elem {
        self.0[l].as_ref().get(u1, u2).unwrap().clone()
    }
}

impl<T: AddAssign> IndexBlockMatMut for Block<Vec<T>> {
    fn set_block_mat(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
        value: Self::Elem,
    )
    {
        assert_eq!(u1, u2);
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
        assert_eq!(u1, u2);
        self.0[l][u1] += value;
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

impl<T: AddAssign> IndexBlockMatMut for Block<TriMat<T>> {
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
