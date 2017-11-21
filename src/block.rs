//! Block-diagonal matrices and similar things.
use super::matrix::Mat;
use super::op::{IndexBlockMatRef, IndexBlockMatMut, Vector, VectorMut};

/// A block matrix is a vector of matrix blocks.
///
/// This wrapper is necessary to avoid conflicting implementations of the
/// vector traits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block<M>(pub Vec<M>);

impl<M: Vector + Clone> Vector for Block<M> {
    type Elem = M::Elem;
}

impl<T: VectorMut + Clone> VectorMut for Block<T> {
    fn fill(&mut self, value: &Self::Elem) {
        for block in &mut self.0 {
            block.fill(value);
        }
    }
}

impl<T: Clone> IndexBlockMatRef for Block<Vec<T>> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        assert_eq!(u1, u2);
        &self.0[l][u1]
    }
}

impl<T: Clone> IndexBlockMatRef for Block<Mat<T>> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.0[l].as_ref().get(u1, u2).unwrap()
    }
}
impl<T: Clone> IndexBlockMatMut for Block<Vec<T>> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        assert_eq!(u1, u2);
        &mut self.0[l][u1]
    }
}

impl<T: Clone> IndexBlockMatMut for Block<Mat<T>> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        self.0[l].as_mut().get(u1, u2).unwrap()
    }
}
