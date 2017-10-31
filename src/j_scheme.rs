use std::{f64, mem};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Deref, Mul, Range};
use fnv::FnvHashMap;
use num::{FromPrimitive, Zero};
use super::basis::{occ, BasisChart, BasisLayout, ChanState, HashChart,
                   IndexBlockVec, IndexBlockVecMut,
                   IndexBlockMat, IndexBlockMatMut,
                   Occ, Occ20, Orb, State};
use super::half::Half;

#[derive(Clone, Copy, Debug)]
pub struct JOrb<K, U> {
    /// Angular momentum magnitude.
    pub j: Half<i32>,
    /// Linear part of the channel.
    pub k: K,
    /// Auxiliary quantum numbers.
    pub u: U,
    /// Occupancy.
    pub x: Occ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JChan {
    /// Angular momentum magnitude
    pub j: Half<i32>,
    /// System-dependent integer
    pub k: u32,
}

#[derive(Clone, Debug)]
pub struct ChartedJScheme<K, U: Hash + Eq> {
    /// `K1 ↔ κ1`
    pub linchan1_chart: HashChart<K, u32>,
    /// `K2 ↔ κ2`
    pub linchan2_chart: HashChart<K, u32>,
    /// `P1 → υ1`
    pub aux_decoder: Box<[U]>,
    /// `(L1, υ1) → U1`
    pub aux_encoder: FnvHashMap<(u32, U), u32>,
    pub scheme: JScheme,
}

impl<K, U> ChartedJScheme<K, U>
    where K: Add<Output = K> + Hash + Eq + Clone,
          U: Hash + Eq + Clone,
{
    pub fn new<I: Iterator<Item = JOrb<K, U>>>(orbs: I) -> Self {
        // one-particle states
        let mut linchan1_chart = HashChart::default();
        let chart1 = BasisChart::new_with(orbs.map(|JOrb { j, k, u, x }| {
            State {
                chan: JChan {
                    j,
                    k: linchan1_chart.insert(k).index,
                },
                part: x,
                aux: u,
            }
        }), Default::default(), Occ::chart());

        let basis_j10 = BasisJ10 {
            chan_chart: chart1.chan_chart,
            layout: chart1.layout,
        };

        // two-particle states
        let mut linchan2_chart = HashChart::default();
        let mut states2 = Vec::default();
        for l1 in 0 .. basis_j10.num_chans() {
            for l2 in 0 .. basis_j10.num_chans() {
                for u1 in basis_j10.auxs(l1, Occ::I, Occ::A) {
                    for u2 in basis_j10.auxs(l2, Occ::I, Occ::A) {
                        let lu1 = ChanState { l: l1, u: u1 };
                        let lu2 = ChanState { l: l2, u: u2 };
                        let p1 = basis_j10.decode(lu1);
                        let p2 = basis_j10.decode(lu2);
                        if p1 < p2 {
                            continue;
                        }
                        let jk1 = basis_j10.j_chan(l1);
                        let jk2 = basis_j10.j_chan(l2);
                        let k12 =
                            linchan1_chart.decode(jk1.k as _).unwrap().clone()
                            + linchan1_chart.decode(jk2.k as _).unwrap().clone();
                        let k12 = linchan2_chart.insert(k12).index as _;
                        let x12 = Occ20::from_usize(
                            usize::from(basis_j10.occ(lu1))
                                + usize::from(basis_j10.occ(lu2))).unwrap();
                        for j12 in Half::tri_range(jk1.j, jk2.j) {
                            states2.push(State {
                                chan: JChan { j: j12, k: k12 },
                                part: x12,
                                aux: (p1, p2),
                            });
                        }
                    }
                }
            }
        }
        let chart2 = BasisChart::new_with(
            states2.into_iter(),
            Default::default(),
            Occ20::chart(),
        );
        let aux_encoder2 = chart2.aux_encoder.iter()
            .map(|(&(l, (p1, p2)), &u)| {
                let j = chart2.decode_chan(l).unwrap().j;
                ((j, p1, p2), ChanState { l, u })
            })
            .collect();
        let basis_j20 = BasisJ20 {
            chan_chart: chart2.chan_chart,
            aux_decoder: chart2.aux_decoder,
            aux_encoder: aux_encoder2,
            layout: chart2.layout,
        };

        Self {
            linchan1_chart,
            linchan2_chart,
            aux_decoder: chart1.aux_decoder,
            aux_encoder: chart1.aux_encoder,
            scheme: JScheme {
                basis_j10,
                basis_j20,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasisJ10 {
    pub chan_chart: HashChart<JChan, u32>,
    pub layout: BasisLayout,
}

impl BasisJ10 {
    #[inline]
    pub fn num_chans(&self) -> u32 {
        self.chan_chart.len() as _
    }

    #[inline]
    pub fn auxs(&self, l: u32, x1: Occ, x2: Occ) -> Range<u32> {
        Range {
            start: self.layout.part_offset(l, u32::from(x1)),
            end: self.layout.part_offset(l, u32::from(x2) + 1),
        }
    }

    #[inline]
    pub fn part_offset(&self, l: u32, x: u32) -> u32 {
        self.layout.part_offset(l, x)
    }

    #[inline]
    pub fn occ(&self, lu: ChanState) -> Occ {
        if lu.u >= self.layout.part_offset(lu.l, 1) {
            Occ::A
        } else {
            Occ::I
        }
    }

    #[inline]
    pub fn j_chan(&self, l: u32) -> JChan {
        *self.chan_chart.decode(l).unwrap()
    }

    #[inline]
    pub fn decode(&self, lu: ChanState) -> Orb {
        Orb(self.layout.dechannelize(lu))
    }

    #[inline]
    pub fn encode(&self, p: Orb) -> ChanState {
        self.layout.channelize(p.0)
    }
}

#[derive(Clone, Debug)]
pub struct BasisJ20 {
    pub chan_chart: HashChart<JChan, u32>,
    pub layout: BasisLayout,
    /// `(j12, p1, p2) → (l12, u12)`
    pub aux_encoder: FnvHashMap<(Half<i32>, Orb, Orb), ChanState>,
    /// `p12 → (p1, p2)`
    pub aux_decoder: Box<[(Orb, Orb)]>,
}

impl BasisJ20 {
    #[inline]
    pub fn num_chans(&self) -> u32 {
        self.layout.num_chans()
    }

    #[inline]
    pub fn auxs(&self, l: u32, x1: Occ20, x2: Occ20) -> Range<u32> {
        Range {
            start: self.layout.part_offset(l, u32::from(x1)),
            end: self.layout.part_offset(l, u32::from(x2) + 1),
        }
    }

    #[inline]
    pub fn aux_range(&self, l: u32, x: Occ20) -> Range<u32> {
        self.auxs(l, x, x)
    }

    #[inline]
    pub fn j_chan(&self, l: u32) -> JChan {
        *self.chan_chart.decode(l).unwrap()
    }

    #[inline]
    pub fn decode(&self, lu: ChanState) -> (Orb, Orb) {
        self.aux_decoder[self.layout.dechannelize(lu) as usize]
    }

    #[inline]
    pub fn encode(&self, j12: Half<i32>, s1: Orb, s2: Orb) -> ChanState {
        *self.aux_encoder.get(&(j12, s1, s2)).unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateMask20 {
    /// `occ12 -> needed`
    pub x_mask: [bool; 3],
    /// `(occ1 + occ2 * 2) -> needed`
    pub xx_mask: [bool; 4],
}

impl StateMask20 {
    #[inline]
    pub fn new(xs: &[[Occ; 2]]) -> Self {
        let ii = xs.contains(&occ::II);
        let ai = xs.contains(&occ::AI);
        let ia = xs.contains(&occ::IA);
        let aa = xs.contains(&occ::AA);
        StateMask20 {
            x_mask: [ii, ai || ia, aa],
            xx_mask: [ii, ai, ia, aa],
        }
    }

    #[inline]
    pub fn test_occ20(self, occ: Occ20) -> bool {
        self.x_mask[usize::from(occ)]
    }

    #[inline]
    pub fn test_occ_occ(self, occ1: Occ, occ2: Occ) -> bool {
        self.xx_mask[usize::from(occ1) + usize::from(occ2) * 2]
    }
}

#[derive(Clone, Debug)]
pub struct StatesJ20<'a> {
    pub l_range: Range<u32>,
    pub states: RelatedStatesJ20<'a>,
}

impl<'a> StatesJ20<'a> {
    pub fn new(scheme: &'a JScheme, mask: StateMask20) -> Self {
        Self {
            l_range: 1 .. scheme.basis_j20.num_chans(),
            states: RelatedStatesJ20::new(scheme, 0, mask),
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
                self.states = RelatedStatesJ20::new(
                    self.states.scheme(),
                    l,
                    self.states.mask(),
                );
                continue;
            }
            return None;
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelatedStatesJ20<'a> {
    pub state: StateJ20<'a>,
    pub x: Occ20,
    pub u_range: Range<u32>,
    pub mask: StateMask20,
}

impl<'a> RelatedStatesJ20<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, l: u32, mask: StateMask20) -> Self {
        let x = Occ20::from_usize(0).unwrap();
        Self {
            state: StateJ20 {
                scheme,
                lu12: ChanState { l, u: 0 },
                // we modify self.state.scheme and self.state.lu12 directly;
                // everything else will be overwritten later on
                j12: Half(0),
                s1: JChanOrb {
                    lu: ChanState { l: 0, u: 0 },
                    j: Half(0),
                    x: Occ::I,
                },
                s2: JChanOrb {
                    lu: ChanState { l: 0, u: 0 },
                    j: Half(0),
                    x: Occ::I,
                },
                permut: 0,
                num_permut: 0, // initially empty
                get_factor: f64::NAN,
                set_factor: f64::NAN,
            },
            x,
            u_range: scheme.basis_j20.aux_range(0, x),
            mask,
        }
    }

    #[inline]
    pub fn scheme(&self) -> &'a JScheme {
        self.state.scheme
    }

    #[inline]
    pub fn mask(&self) -> StateMask20 {
        self.mask
    }
}

impl<'a> Iterator for RelatedStatesJ20<'a> {
    type Item = StateJ20<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // next permut
            while let Some(state) = self.state.next_permut() {
                self.state = state;
                if self.mask.test_occ_occ(state.s1.x, state.s2.x) {
                    return Some(state);
                }
            }

            loop {
                // next u
                if let Some(u) = self.u_range.next() {
                    self.state.lu12.u = u;
                    self.state = StateJ20::new(
                        self.state.scheme,
                        self.state.lu12,
                    );
                    break;
                }

                // next x
                let mut found_x = false;
                while let Some(x) = self.x.step() {
                    self.x = x;
                    if self.mask.test_occ20(x) {
                        found_x = true;
                        break;
                    }
                }
                if found_x {
                    self.u_range = self.scheme().basis_j20.aux_range(
                        self.state.lu12.l,
                        self.x,
                    );
                    continue;
                }
                return None;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JChanOrb {
    pub lu: ChanState,
    pub j: Half<i32>,
    pub x: Occ,
}

impl JChanOrb {
    #[inline]
    pub fn new(scheme: &JScheme, lu: ChanState) -> Self {
        JChanOrb {
            lu,
            j: scheme.basis_j10.j_chan(lu.l).j,
            x: scheme.basis_j10.occ(lu),
        }
    }

    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j.weight(exponent)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ10<'a> {
    pub scheme: &'a JScheme,
    pub s1: JChanOrb,
}

impl<'a> Deref for StateJ10<'a> {
    type Target = JChanOrb;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.s1
    }
}

impl<'a> StateJ10<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, lu1: ChanState) -> Self {
        Self { scheme, s1: JChanOrb::new(scheme, lu1) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ20<'a> {
    pub scheme: &'a JScheme,
    pub lu12: ChanState,
    pub j12: Half<i32>,
    pub s1: JChanOrb,
    pub s2: JChanOrb,
    pub permut: u8,
    /// Number of states related by antisymmetry
    pub num_permut: u8,
    pub get_factor: f64,
    /// `set_factor == 1.0 / (get_factor * num_permut)`
    pub set_factor: f64,
}

impl<'a> StateJ20<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, lu12: ChanState) -> Self {
        let (s1, s2) = scheme.basis_j20.decode(lu12);
        let num_permut = if s1 == s2 { 1 } else { 2 };
        Self {
            scheme,
            lu12,
            j12: scheme.basis_j20.j_chan(lu12.l).j,
            s1: JChanOrb::new(scheme, scheme.basis_j10.encode(s1)),
            s2: JChanOrb::new(scheme, scheme.basis_j10.encode(s2)),
            permut: 0,
            num_permut,
            get_factor: 1.0,
            set_factor: 1.0 / num_permut as f64,
        }
    }

    #[inline]
    pub fn next_permut(mut self) -> Option<Self> {
        self.permut += 1;
        if self.permut >= self.num_permut {
            return None;
        }
        mem::swap(&mut self.s1, &mut self.s2);
        let sign = -(self.s1.j + self.s2.j - self.j12).phase();
        self.get_factor = sign;
        self.set_factor = sign / self.num_permut as f64;
        Some(self)
    }

    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j12.weight(exponent)
    }

    #[inline]
    pub fn related_states(self, xs: &[[Occ; 2]]) -> RelatedStatesJ20<'a> {
        RelatedStatesJ20::new(self.scheme, self.lu12.l, StateMask20::new(xs))
    }

    #[inline]
    pub fn split_to_j10_j10(self) -> (StateJ10<'a>, StateJ10<'a>) {
        (
            StateJ10 { scheme: self.scheme, s1: self.s1 },
            StateJ10 { scheme: self.scheme, s1: self.s2 },
        )
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
        StatesJ20::new(self, StateMask20::new(xs))
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
        let l = s1.lu.l as _;
        let u = s1.lu.u as _;
        self.data.index_block_vec(l, u).clone()
    }
}

impl<'a, D: IndexBlockVecMut> DiagOpJ10<'a, D> where
    D::Elem: AddAssign,
{
    #[inline]
    pub fn add(&mut self, s1: StateJ10, value: D::Elem) {
        let l = s1.lu.l as _;
        let u = s1.lu.u as _;
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
        let ll = s1.lu.l as _;
        let lr = s2.lu.l as _;
        let ul = s1.lu.u as _;
        let ur = s2.lu.u as _;
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
        let ll = s1.lu.l as _;
        let lr = s2.lu.l as _;
        let ul = s1.lu.u as _;
        let ur = s2.lu.u as _;
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
