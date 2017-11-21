use std::{f64, iter};
use std::ops::{AddAssign, Mul};
use num::{FromPrimitive, Zero};
use super::basis::BasisLayout;
use super::block::Block;
use super::block_mat::{BlockMatRef, BlockMatMut};
use super::mat::Mat;

pub trait ChartedBasis {
    type Scheme;
    fn layout<'a>(&self, scheme: &'a Self::Scheme) -> &'a BasisLayout;
}

pub trait ReifyState {
    type Basis;
    type Scheme;
    fn reify_state(
        self,
        scheme: &Self::Scheme,
        basis: &Self::Basis,
    ) -> ReifiedState;
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

impl<T: Clone> Vector for Vec<T> {
    type Elem = T;
}

impl<T: Clone> Vector for Mat<T> {
    type Elem = T;
}

impl<'a, T: Clone> Vector for BlockMatRef<'a, T> {
    type Elem = T;
}

impl<'a, T: Clone> Vector for BlockMatMut<'a, T> {
    type Elem = T;
}

pub trait VectorMut: Vector {
    fn fill(&mut self, value: &Self::Elem);
}

impl<T: Clone> VectorMut for Vec<T> {
    fn fill(&mut self, value: &Self::Elem) {
        let n = self.len();
        self.clear();
        self.resize(n, value.clone());
    }
}

impl<T: Clone> VectorMut for Mat<T> {
    fn fill(&mut self, value: &Self::Elem) {
        self.as_mut().fill(value);
    }
}

pub trait IndexBlockMatRef: Vector {
    fn index_block_mat(
        &self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &Self::Elem;
}

impl<'a, T: Clone> IndexBlockMatRef for BlockMatRef<'a, T> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<'a, T: Clone> IndexBlockMatRef for BlockMatMut<'a, T> {
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.as_ref().get(l).unwrap().get(u1, u2).unwrap()
    }
}

pub trait IndexBlockMatMut: IndexBlockMatRef {
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

#[derive(Clone, Copy, Debug)]
pub struct Op<S, L, R, D> {
    pub scheme: S,
    pub left_basis: L,
    pub right_basis: R,
    pub data: D,
}

impl<S, L, R, D> Op<S, L, R, D> {
    pub fn scheme(&self) -> &S {
        &self.scheme
    }
}

impl<S, L, R, T> Op<S, L, R, Block<Mat<T>>> where
    L: ChartedBasis<Scheme = S> + Default,
    R: ChartedBasis<Scheme = S> + Default,
    T: Default + Clone,
{
    pub fn new(scheme: S) -> Self {
        let left_basis = L::default();
        let right_basis = R::default();
        let data = {
            let left_layout = left_basis.layout(&scheme);
            let right_layout = right_basis.layout(&scheme);
            assert_eq!(left_layout.num_chans(), right_layout.num_chans());
            Block((0 .. left_layout.num_chans()).map(|l| {
                Mat::replicate(
                    left_layout.num_auxs(l) as _,
                    right_layout.num_auxs(l) as _,
                    Default::default(),
                )
            }).collect())
        };
        Self { scheme, left_basis, right_basis, data }
    }
}

impl<S, B, T> Op<S, B, B, Block<Vec<T>>> where
    B: ChartedBasis<Scheme = S> + Default,
    T: Default + Clone,
{
    /// Create a diagonal operator.
    pub fn new_vec(scheme: S) -> Self {
        let left_basis = B::default();
        let right_basis = B::default();
        let data = {
            let layout = left_basis.layout(&scheme);
            Block((0 .. layout.num_chans()).map(|l| {
                vec![Default::default(); layout.num_auxs(l) as _]
            }).collect())
        };
        Self { scheme, left_basis, right_basis, data }
    }
}

impl<S, B, T> Op<S, B, B, Block<Vec<T>>> where
    T: Clone + iter::Sum<T>
{
    pub fn sum_vec(&self) -> T {
        self.data.0.iter().map(|x| x.iter().cloned().sum()).sum()
    }
}

impl<S, L, R, D> Op<S, L, R, D> where
    D: IndexBlockMatRef,
    D::Elem: FromPrimitive + Mul<Output = D::Elem> + Zero,
{
    #[inline]
    pub fn at<I, J>(&self, i: I, j: J) -> D::Elem where
        I: ReifyState<Scheme = S, Basis = L>,
        J: ReifyState<Scheme = S, Basis = R>,
    {
        let ri = i.reify_state(&self.scheme, &self.left_basis);
        let rj = j.reify_state(&self.scheme, &self.right_basis);
        if ri.chan != rj.chan {
            return D::Elem::zero();
        }
        self.data.index_block_mat(ri.chan, ri.aux, rj.aux).clone()
            * D::Elem::from_f64(ri.get_factor).unwrap()
            * D::Elem::from_f64(rj.get_factor).unwrap()
    }
}

impl<S, L, R, D> Op<S, L, R, D> where
    D: IndexBlockMatMut,
    D::Elem: FromPrimitive + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn set<I, J>(&mut self, i: I, j: J, value: D::Elem) where
        I: ReifyState<Scheme = S, Basis = L>,
        J: ReifyState<Scheme = S, Basis = R>,
    {
        let ri = i.reify_state(&self.scheme, &self.left_basis);
        let rj = j.reify_state(&self.scheme, &self.right_basis);
        assert_eq!(ri.chan, rj.chan, "channels do not match");
        *self.data.index_block_mat_mut(ri.chan, ri.aux, rj.aux) =
            value
            * D::Elem::from_f64(ri.set_factor).unwrap()
            * D::Elem::from_f64(rj.set_factor).unwrap();
    }
}

impl<S, L, R, D> Op<S, L, R, D> where
    D: IndexBlockMatMut,
    D::Elem: FromPrimitive + AddAssign + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn add<I, J>(&mut self, i: I, j: J, value: D::Elem) where
        I: ReifyState<Scheme = S, Basis = L>,
        J: ReifyState<Scheme = S, Basis = R>,
    {
        let ri = i.reify_state(&self.scheme, &self.left_basis);
        let rj = j.reify_state(&self.scheme, &self.right_basis);
        assert_eq!(ri.chan, rj.chan, "channels do not match");
        *self.data.index_block_mat_mut(ri.chan, ri.aux, rj.aux) +=
            value
            * D::Elem::from_f64(ri.add_factor).unwrap()
            * D::Elem::from_f64(rj.add_factor).unwrap();
    }
}

impl<S, L, R, D: Vector> Vector for Op<S, L, R, D> {
    type Elem = D::Elem;
}

impl<S, L, R, D: VectorMut> VectorMut for Op<S, L, R, D> {
    fn fill(&mut self, value: &Self::Elem) {
        self.data.fill(value);
    }
}
