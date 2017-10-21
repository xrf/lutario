use std::ops::{AddAssign, Mul, Range};
use num::{FromPrimitive, Zero};
use super::basis::{occ, ChanState, IndexBlockVec, IndexBlockVecMut,
                   IndexBlockMat, IndexBlockMatMut, Occ, Occ20, Orb};
use super::half::Half;
use super::tri_matrix::TriMatrix;

#[derive(Clone, Copy, Debug)]
pub struct JChan {
    /// Angular momentum magnitude
    pub j: Half<i32>,
    /// System-dependent integer
    pub k: u32,
}

#[derive(Clone, Debug)]
pub struct BasisJ10 {
    pub chans: Box<[JChan]>,
    // l -> s_offset
    pub offsets: Box<[u32]>,
    // s -> (l, u)
    pub encoder: Box<[ChanState]>,
    // l -> u
    pub num_occ: Box<[u32]>,
}

impl BasisJ10 {
    // pub fn new(states: &mut Iterator<Item = (JChan, >) {

    // }

    #[inline]
    pub fn occ(&self, lu: ChanState) -> Occ {
        if lu.u >= self.num_occ[lu.l as usize] as _ {
            Occ::A
        } else {
            Occ::I
        }
    }

    #[inline]
    pub fn j_chan(&self, l: u32) -> JChan {
        self.chans[l as usize]
    }

    #[inline]
    pub fn decode(&self, lu: ChanState) -> Orb {
        Orb(self.offsets[lu.l as usize] + lu.u)
    }

    #[inline]
    pub fn encode(&self, s: Orb) -> ChanState {
        self.encoder[s.0 as usize]
    }
}

#[derive(Clone, Debug)]
pub struct BasisJ20 {
    pub chans: Vec<JChan>,
    // l12 -> x12 -> u12
    pub part_offsets: Vec<Vec<u32>>,
    // l12 -> u12 -> (s1, s2)
    pub decoder: Vec<Vec<(Orb, Orb)>>,
    // (s1, s2) -> (l12, u12)
    pub encoder: TriMatrix<ChanState>,
}

impl BasisJ20 {
    #[inline]
    pub fn num_chans(&self) -> u32 {
        self.chans.len() as _
    }

    #[inline]
    pub fn auxs(&self, l: u32, x1: Occ20, x2: Occ20) -> Range<u32> {
        let l = l as usize;
        Range {
            start: self.part_offsets[l][usize::from(x1)],
            end: self.part_offsets[l][usize::from(x2) + 1],
        }
    }

    #[inline]
    pub fn aux_range(&self, l: u32, x: Occ20) -> Range<u32> {
        let l = l as usize;
        self.part_offsets[l][usize::from(x)] as _ ..
        self.part_offsets[l][usize::from(x) + 1] as _
    }

    #[inline]
    pub fn j_chan(&self, l: u32) -> JChan {
        self.chans[l as usize]
    }

    #[inline]
    pub fn decode(&self, lu: ChanState) -> (Orb, Orb) {
        self.decoder[lu.l as usize][lu.u as usize]
    }

    #[inline]
    pub fn encode(&self, (s1, s2): (Orb, Orb)) -> ChanState {
        *self.encoder.as_ref().get(s1.0 as _, s2.0 as _).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct StatesJ20<'a> {
    pub l_range: Range<u32>,
    pub states: RelatedStatesJ20<'a>,
}

impl<'a> StatesJ20<'a> {
    pub fn new(scheme: &'a JScheme, xs: &[[Occ; 2]]) -> Self {
        Self {
            l_range: 1 .. scheme.basis_j20.num_chans(),
            states: RelatedStatesJ20::new(scheme, 0, xs),
        }
    }
}

impl<'a> Iterator for StatesJ20<'a> {
    type Item = StateJ20<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let r@Some(_) = self.states.next() {
                return r;
            }
            if let Some(l) = self.l_range.next() {
                self.states = RelatedStatesJ20::with_x_permuts(
                    self.states.scheme,
                    l,
                    self.states.x_permuts,
                );
                continue;
            }
            return None;
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelatedStatesJ20<'a> {
    pub scheme: &'a JScheme,
    pub l: u32,
    pub x: Occ20,
    pub u_range: Range<u32>,
    pub u: u32,
    pub permut: u8,
    /// `x_permuts[occ20 * 2 + permut]`
    pub x_permuts: [bool; 6],
}

impl<'a> RelatedStatesJ20<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, l: u32, xs: &[[Occ; 2]]) -> Self {
        // de-antisymmetrization within specific occupation parts is tricky
        let mut x_permuts = [false; 6];
        if xs.contains(&occ::II) {
            x_permuts[usize::from(Occ20::II) * 2 + 0] = true;
            x_permuts[usize::from(Occ20::II) * 2 + 1] = true;
        }
        if xs.contains(&occ::AI) {
            x_permuts[usize::from(Occ20::AI) * 2 + 0] = true;
        }
        if xs.contains(&occ::IA) {
            x_permuts[usize::from(Occ20::AI) * 2 + 1] = true;
        }
        if xs.contains(&occ::AA) {
            x_permuts[usize::from(Occ20::AA) * 2 + 0] = true;
            x_permuts[usize::from(Occ20::AA) * 2 + 1] = true;
        }
        Self::with_x_permuts(scheme, l, x_permuts)
    }

    #[inline]
    pub fn with_x_permuts(
        scheme: &'a JScheme,
        l: u32,
        x_permuts: [bool; 6],
    ) -> Self {
        let x = Occ20::from_usize(0).unwrap();
        Self {
            scheme,
            l,
            x,
            u_range: scheme.basis_j20.aux_range(0, x),
            u: 0,
            permut: 2, // empty at start
            x_permuts,
        }
    }
}

impl<'a> Iterator for RelatedStatesJ20<'a> {
    type Item = StateJ20<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // writing loops in inverted form is extremely annoying
        loop {
            if self.permut < 2 {
                let permut = self.permut;
                self.permut += 1;
                if !self.x_permuts[usize::from(self.x) * 2 + permut as usize] {
                    continue;
                }
                // FIXME: the ordering here assumes I < A
                // which is no longer true because we adopted (l, u) ordering
                match StateJ20::new(
                    self.scheme,
                    self.l,
                    self.u,
                    permut,
                ) {
                    Some(s) => return Some(s),
                    None => continue, // skip this permut
                }
            }
            loop {
                if let Some(u) = self.u_range.next() {
                    self.u = u;
                    self.permut = 0;
                    break;
                }
                if let Some(x) = self.x.step() {
                    self.x = x;
                    self.u_range = self.scheme.basis_j20.aux_range(self.l, x);
                    continue;
                }
                return None;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct JScheme {
    pub basis_j10: BasisJ10,
    pub basis_j20: BasisJ20,
}

impl JScheme {
    #[inline]
    pub fn states_j20(&self, xs: &[[Occ; 2]]) -> StatesJ20 {
        StatesJ20::new(self, xs)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ10<'a> {
    pub scheme: &'a JScheme,
    pub lu1: ChanState,
    pub j1: Half<i32>,
}

impl<'a> StateJ10<'a> {
    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j1.weight(exponent)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ20<'a> {
    pub scheme: &'a JScheme,
    pub lu12: ChanState,
    pub j12: Half<i32>,
    pub lu1: ChanState,
    pub j1: Half<i32>,
    pub lu2: ChanState,
    pub j2: Half<i32>,
    pub get_factor: f64,
    /// The set factor contains the inverse phase, but also inverse the number
    /// of antisymmetry-related entries.
    pub set_factor: f64,
}

impl<'a> StateJ20<'a> {
    #[inline]
    pub fn new(
        scheme: &'a JScheme,
        l12: u32,
        u12: u32,
        permut: u8,
    ) -> Option<Self>
    {
        let basis_j10 = &scheme.basis_j10;
        let basis_j20 = &scheme.basis_j20;
        let j12 = basis_j20.j_chan(l12).j;
        let lu12 = ChanState { l: l12, u: u12 };
        let (s1, s2) = basis_j20.decode(lu12);
        let num_permut = if s1 == s2 {
            1
        } else {
            2
        };
        if permut >= num_permut {
            return None;
        }
        let lu1 = basis_j10.encode(s1);
        let j1 = basis_j10.j_chan(lu1.l).j;
        let lu2 = basis_j10.encode(s2);
        let j2 = basis_j10.j_chan(lu2.l).j;
        let sign = if permut == 0 {
            1.0
        } else {
            -(j1 + j2 - j12).phase()
        };
        Some(StateJ20 {
            scheme,
            lu12,
            j12,
            lu1,
            j1,
            lu2,
            j2,
            get_factor: sign,
            set_factor: sign / num_permut as f64,
        })
    }

    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j12.weight(exponent)
    }

    #[inline]
    pub fn related_states(self, xs: &[[Occ; 2]]) -> RelatedStatesJ20<'a> {
        RelatedStatesJ20::new(self.scheme, self.lu12.l, xs)
    }

    #[inline]
    pub fn split_to_j10_j10(self) -> (StateJ10<'a>, StateJ10<'a>) {
        (
            StateJ10 { scheme: self.scheme, lu1: self.lu1, j1: self.j1 },
            StateJ10 { scheme: self.scheme, lu1: self.lu2, j1: self.j2 },
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagOpJ10<'a, D> {
    pub scheme: &'a JScheme,
    pub data: D,
}

impl<'a, D: IndexBlockVec> DiagOpJ10<'a, D> where
    D::Elem: Clone,
{
    #[inline]
    pub fn at(&self, s1: StateJ10) -> D::Elem {
        let l = s1.lu1.l as _;
        let u = s1.lu1.u as _;
        self.data.index_block_vec(l, u).clone()
    }
}

impl<'a, D: IndexBlockVecMut> DiagOpJ10<'a, D> where
    D::Elem: AddAssign,
{
    #[inline]
    pub fn add(&mut self, s1: StateJ10, value: D::Elem) {
        let l = s1.lu1.l as _;
        let u = s1.lu1.u as _;
        *self.data.index_block_vec_mut(l, u) += value;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpJ100<'a, D> {
    pub scheme: &'a JScheme,
    pub data: D,
}

impl<'a, D: IndexBlockMat> OpJ100<'a, D> where
    D::Elem: Zero + Clone,
{
    #[inline]
    pub fn at(&self, s1: StateJ10, s2: StateJ10) -> D::Elem {
        let ll = s1.lu1.l as _;
        let lr = s2.lu1.l as _;
        let ul = s1.lu1.u as _;
        let ur = s2.lu1.u as _;
        if ll != lr {
            return D::Elem::zero();
        }
        self.data.index_block_mat(ll, ul, ur).clone()
    }
}

impl<'a, D: IndexBlockMatMut> OpJ100<'a, D> where
    D::Elem: AddAssign,
{
    #[inline]
    pub fn add(&mut self, s1: StateJ10, s2: StateJ10, value: D::Elem) {
        let ll = s1.lu1.l as _;
        let lr = s2.lu1.l as _;
        let ul = s1.lu1.u as _;
        let ur = s2.lu1.u as _;
        if ll != lr {
            return;
        }
        *self.data.index_block_mat_mut(ll, ul, ur) += value;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpJ200<'a, D> {
    pub scheme: &'a JScheme,
    pub data: D,
}

impl<'a, D: IndexBlockMat> OpJ200<'a, D> where
    D::Elem: FromPrimitive + Mul<Output = D::Elem> + Zero + Clone,
{
    #[inline]
    pub fn at(&self, s12: StateJ20, s34: StateJ20) -> D::Elem {
        let ll = s12.lu12.l as _;
        let lr = s34.lu12.l as _;
        let ul = s12.lu12.u as _;
        let ur = s34.lu12.u as _;
        if ll != lr {
            return D::Elem::zero();
        }
        self.data.index_block_mat(ll, ul, ur).clone()
            * D::Elem::from_f64(s12.get_factor).unwrap()
            * D::Elem::from_f64(s34.get_factor).unwrap()
    }
}

impl<'a, D: IndexBlockMatMut> OpJ200<'a, D> where
    D::Elem: FromPrimitive + AddAssign + Mul<Output = D::Elem>,
{
    #[inline]
    pub fn add(&mut self, s12: StateJ20, s34: StateJ20, value: D::Elem) {
        let ll = s12.lu12.l as _;
        let lr = s34.lu12.l as _;
        let ul = s12.lu12.u as _;
        let ur = s34.lu12.u as _;
        if ll != lr {
            return;
        }
        *self.data.index_block_mat_mut(ll, ul, ur) += value
            * D::Elem::from_f64(s12.set_factor).unwrap()
            * D::Elem::from_f64(s34.set_factor).unwrap();
    }
}
