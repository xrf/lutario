use std::{f64, iter};
use std::ops::{AddAssign, Mul};
use num::{FromPrimitive, Zero};
use super::basis::BasisLayout;
use super::block_matrix::{BlockMat, BlockMatMut};
use super::block_vector::BlockVector;
use super::matrix::Matrix;

pub trait ChartedBasis {
    type State;
    fn layout(&self) -> &BasisLayout;
    fn reify_state(&self, state: Self::State) -> ReifiedState;
}

#[derive(Clone, Copy, Debug)]
pub struct ReifiedState {
    pub chan: usize,
    pub aux: usize,
    pub get_factor: f64,
    pub set_factor: f64,
    pub add_factor: f64,
}

pub trait Vector {
    type Elem: Clone;
}

impl<T: Clone> Vector for BlockVector<T> {
    type Elem = T;
}

impl<'a, T: Clone> Vector for BlockMat<'a, T> {
    type Elem = T;
}

impl<'a, T: Clone> Vector for BlockMatMut<'a, T> {
    type Elem = T;
}

impl<T: Clone> Vector for Vec<Matrix<T>> {
    type Elem = T;
}

pub trait VectorMut: Vector {
    fn fill(&mut self, value: &Self::Elem);
}

impl<T: Clone> VectorMut for BlockVector<T> {
    fn fill(&mut self, value: &Self::Elem) {
        for block in &mut self.0 {
            let n = block.len();
            block.clear();
            block.resize(n, value.clone());
        }
    }
}

impl<T: Clone> VectorMut for Vec<Matrix<T>> {
    fn fill(&mut self, value: &Self::Elem) {
        for block in self {
            block.as_mut().fill(value);
        }
    }
}

pub trait IndexBlockVec: Vector {
    fn index_block_vec(&self, l: usize, u: usize) -> &Self::Elem;
}

impl<T: Clone> IndexBlockVec for BlockVector<T> {
    fn index_block_vec(&self, l: usize, u: usize) -> &Self::Elem {
        &self.0[l][u]
    }
}

pub trait IndexBlockVecMut: IndexBlockVec + VectorMut {
    fn index_block_vec_mut(&mut self, l: usize, u: usize) -> &mut Self::Elem;
}

impl<T: Clone> IndexBlockVecMut for BlockVector<T> {
    fn index_block_vec_mut(&mut self, l: usize, u: usize) -> &mut Self::Elem {
        &mut self.0[l][u]
    }
}

pub trait IndexBlockMat: Vector {
    fn index_block_mat(
        &self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &Self::Elem;
}

impl<'a, T: Clone> IndexBlockMat for BlockMat<'a, T> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<'a, T: Clone> IndexBlockMat for BlockMatMut<'a, T> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.as_ref().get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<T: Clone> IndexBlockMat for Vec<Matrix<T>> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self[l].as_ref().get(u1, u2).unwrap()
    }
}

pub trait IndexBlockMatMut: IndexBlockMat {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem;
}

impl<'a, T: Clone> IndexBlockMatMut for BlockMatMut<'a, T> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        self.as_mut().get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<T: Clone> IndexBlockMatMut for Vec<Matrix<T>> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        self[l].as_mut().get(u1, u2).unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagOp<B, D> {
    pub basis: B,
    pub data: D,
}

impl<B, T> DiagOp<B, BlockVector<T>> where
    B: ChartedBasis,
    T: Default + Clone,
{
    pub fn new(basis: B) -> Self {
        let data = {
            let layout = basis.layout();
            BlockVector((0 .. layout.num_chans()).map(|l| {
                vec![Default::default(); layout.num_auxs(l) as _]
            }).collect())
        };
        Self { basis, data }
    }
}

impl<B, T> DiagOp<B, BlockVector<T>> where
    T: Clone + iter::Sum<T>
{
    pub fn sum(&self) -> T {
        self.data.0.iter().map(|x| x.iter().cloned().sum()).sum()
    }
}

impl<B, D> DiagOp<B, D> where
    B: ChartedBasis,
    D: IndexBlockVec,
    D::Elem: FromPrimitive + Mul<Output = D::Elem> + Zero + Clone,
{
    #[inline]
    pub fn at(&self, i: B::State) -> D::Elem {
        let ri = self.basis.reify_state(i);
        self.data.index_block_vec(ri.chan, ri.aux).clone()
            * D::Elem::from_f64(ri.get_factor).unwrap()
    }
}

impl<B, D> DiagOp<B, D> where
    B: ChartedBasis,
    D: IndexBlockVecMut,
    D::Elem: FromPrimitive + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn set(&mut self, i: B::State, value: D::Elem) {
        let ri = self.basis.reify_state(i);
        *self.data.index_block_vec_mut(ri.chan, ri.aux) =
            value
            * D::Elem::from_f64(ri.set_factor).unwrap()
    }
}

impl<B, D> DiagOp<B, D> where
    B: ChartedBasis,
    D: IndexBlockVecMut,
    D::Elem: FromPrimitive + AddAssign + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn add(&mut self, i: B::State, value: D::Elem) {
        let ri = self.basis.reify_state(i);
        *self.data.index_block_vec_mut(ri.chan, ri.aux) +=
            value
            * D::Elem::from_f64(ri.add_factor).unwrap();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Op<L, R, D> {
    pub left_basis: L,
    pub right_basis: R,
    pub data: D,
}

impl<L, R, D: Vector> Vector for Op<L, R, D> {
    type Elem = D::Elem;
}

impl<L, R, D: VectorMut> VectorMut for Op<L, R, D> {
    fn fill(&mut self, value: &Self::Elem) {
        self.data.fill(value);
    }
}

impl<L, R, T> Op<L, R, Vec<Matrix<T>>> where
    L: ChartedBasis,
    R: ChartedBasis,
    T: Default + Clone,
{
    pub fn new(left_basis: L, right_basis: R) -> Self {
        let data = {
            let left_layout = left_basis.layout();
            let right_layout = right_basis.layout();
            assert_eq!(left_layout.num_chans(), right_layout.num_chans());
            (0 .. left_layout.num_chans()).map(|l| {
                Matrix::replicate(
                    left_layout.num_auxs(l) as _,
                    right_layout.num_auxs(l) as _,
                    Default::default(),
                )
            }).collect()
        };
        Self { left_basis, right_basis, data }
    }
}

impl<L, R, D> Op<L, R, D> where
    L: ChartedBasis,
    R: ChartedBasis,
    D: IndexBlockMat,
    D::Elem: FromPrimitive + Mul<Output = D::Elem> + Zero,
{
    #[inline]
    pub fn at(&self, i: L::State, j: R::State) -> D::Elem {
        let ri = self.left_basis.reify_state(i);
        let rj = self.right_basis.reify_state(j);
        if ri.chan != rj.chan {
            return D::Elem::zero();
        }
        self.data.index_block_mat(ri.chan, ri.aux, rj.aux).clone()
            * D::Elem::from_f64(ri.get_factor).unwrap()
            * D::Elem::from_f64(rj.get_factor).unwrap()
    }
}

impl<L, R, D> Op<L, R, D> where
    L: ChartedBasis,
    R: ChartedBasis,
    D: IndexBlockMatMut,
    D::Elem: FromPrimitive + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn set(&mut self, i: L::State, j: R::State, value: D::Elem) {
        let ri = self.left_basis.reify_state(i);
        let rj = self.right_basis.reify_state(j);
        assert_eq!(ri.chan, rj.chan, "channels do not match");
        *self.data.index_block_mat_mut(ri.chan, ri.aux, rj.aux) =
            value
            * D::Elem::from_f64(ri.set_factor).unwrap()
            * D::Elem::from_f64(rj.set_factor).unwrap();
    }
}

impl<L, R, D> Op<L, R, D> where
    L: ChartedBasis,
    R: ChartedBasis,
    D: IndexBlockMatMut,
    D::Elem: FromPrimitive + AddAssign + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn add(&mut self, i: L::State, j: R::State, value: D::Elem) {
        let ri = self.left_basis.reify_state(i);
        let rj = self.right_basis.reify_state(j);
        assert_eq!(ri.chan, rj.chan, "channels do not match");
        *self.data.index_block_mat_mut(ri.chan, ri.aux, rj.aux) +=
            value
            * D::Elem::from_f64(ri.add_factor).unwrap()
            * D::Elem::from_f64(rj.add_factor).unwrap();
    }
}
