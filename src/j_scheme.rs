use std::{f64, fmt, mem};
use std::hash::Hash;
use std::ops::{Add, Deref, Range};
use fnv::FnvHashMap;
use super::basis::{occ, BasisChart, BasisLayout, ChanState, Fence, HashChart,
                   Occ, Occ20, Orb, PartState};
use super::block::Block;
use super::mat::Mat;
use super::half::Half;
use super::op::{ChartedBasis, Op, ReifiedState};
use super::utils;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JChan<K = u32> {
    /// Angular momentum magnitude
    pub j: Half<i32>,
    /// Linear part of the channel.  If `K = u32` then this is usually some
    /// system-dependent integer of unknown interpretation.
    pub k: K,
}

/// Construct a trivial `JChan` where `j` is zero.
impl<K> From<K> for JChan<K> {
    fn from(k: K) -> Self {
        Self { j: Half(0), k }
    }
}

#[derive(Clone, Debug)]
pub struct JAtlas<K, U: Hash + Eq> { // FIXME: spurious constraints
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

impl<K, U> JAtlas<K, U>
    where K: Add<Output = K> + Hash + Eq + Clone,
          U: Hash + Eq + Clone,
{
    pub fn new<I>(orbs: I) -> Self where
        I: Iterator<Item = PartState<Occ, ChanState<JChan<K>, U>>>,
    {
        // one-particle states
        let mut linchan1_chart = HashChart::default();
        let chart1 = BasisChart::new_with(orbs.map(|state| {
            PartState {
                x: state.x,
                p: ChanState {
                    l: JChan {
                        j: state.p.l.j,
                        k: linchan1_chart.insert(state.p.l.k).index,
                    },
                    u: state.p.u,
                },
            }
        }), Default::default(), Occ::chart());

        let basis_10 = BasisSchemeJ10 {
            layout: chart1.layout,
            chan_chart: chart1.chan_chart,
        };

        // two-particle states
        let mut linchan2_chart = HashChart::default();
        let mut states2 = Vec::default();
        for l1 in 0 .. basis_10.num_chans() {
            for l2 in 0 .. basis_10.num_chans() {
                for u1 in basis_10.auxs(l1, Occ::I, Occ::A) {
                    for u2 in basis_10.auxs(l2, Occ::I, Occ::A) {
                        let lu1 = ChanState { l: l1, u: u1 };
                        let lu2 = ChanState { l: l2, u: u2 };
                        let p1 = basis_10.decode(lu1);
                        let p2 = basis_10.decode(lu2);
                        if p1 < p2 {
                            continue;
                        }
                        let jk1 = basis_10.j_chan(l1);
                        let jk2 = basis_10.j_chan(l2);
                        let k12 =
                            linchan1_chart.decode(jk1.k).unwrap().clone()
                            + linchan1_chart.decode(jk2.k).unwrap().clone();
                        let k12 = linchan2_chart.insert(k12).index as _;
                        let x12 = Occ20::from_usize(
                            usize::from(basis_10.occ(lu1))
                                + usize::from(basis_10.occ(lu2))).unwrap();
                        let j1_j2 = jk1.j + jk2.j;
                        for j12 in Half::tri_range(jk1.j, jk2.j) {
                            if p1 == p2
                                && j1_j2.abs_diff(j12).unwrap() % 2 == 0
                            {
                                // forbidden by antisymmetry
                                continue;
                            }
                            states2.push(PartState {
                                x: x12,
                                p: ChanState {
                                    l: JChan { j: j12, k: k12 },
                                    u: (p1, p2),
                                },
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
        let basis_20 = BasisSchemeJ20 {
            layout: chart2.layout,
            chan_chart: chart2.chan_chart,
            aux_encoder: aux_encoder2,
            aux_decoder: chart2.aux_decoder,
        };

        Self {
            linchan1_chart,
            linchan2_chart,
            aux_decoder: chart1.aux_decoder,
            aux_encoder: chart1.aux_encoder,
            scheme: JScheme {
                basis_10,
                basis_20,
            },
        }
    }
}

impl<K, U> JAtlas<K, U> where
    U: Hash + Eq,                       // FIXME: spurious constraints
    K: Clone,
    U: Clone,
{
    pub fn decode(&self, s: StateJ10) -> Option<ChanState<JChan<K>, U>> {
        (|| -> Result<_, ()> {
            let Orb(p) = self.scheme.basis_10.decode(s.s1.lu);
            let JChan { j, k } = self.scheme.basis_10.j_chan(s.s1.lu.l);
            Ok(ChanState {
                l: JChan {
                    j,
                    k: self.linchan1_chart.decode(k).ok_or(())?.clone(),
                },
                u: self.aux_decoder.get(p as usize).ok_or(())?.clone(),
            })
        })().ok()
    }
}

impl<K, U> JAtlas<K, U> where
    K: Hash + Eq,
    U: Hash + Eq,
{
    pub fn encode(&self, lu: &ChanState<JChan<K>, U>) -> Option<StateJ10> {
        (|| -> Result<_, ()> {
            let l = *self.scheme.basis_10.chan_chart.encode(&JChan {
                j: lu.l.j,
                k: *self.linchan1_chart.encode(&lu.l.k).ok_or(())?,
            }).ok_or(())?;
            let u = *utils::with_tuple2_ref(&l, &lu.u, |lu| {
                self.aux_encoder.get(lu)
            }).ok_or(())?;
            Ok(self.scheme.state_10(ChanState { l, u }))
        })().ok()
    }
}

#[derive(Clone, Debug)]
pub struct BasisSchemeJ10 {
    pub layout: BasisLayout,
    pub chan_chart: HashChart<JChan, u32>,
}

impl BasisSchemeJ10 {
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
    pub fn aux_range(&self, l: u32, x: Occ) -> Range<u32> {
        self.auxs(l, x, x)
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
pub struct BasisSchemeJ20 {
    pub layout: BasisLayout,
    pub chan_chart: HashChart<JChan, u32>,
    /// `(j12, p1, p2) → (l12, u12)`
    pub aux_encoder: FnvHashMap<(Half<i32>, Orb, Orb), ChanState>,
    /// `p12 → (p1, p2)`
    pub aux_decoder: Box<[(Orb, Orb)]>,
}

impl BasisSchemeJ20 {
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
    pub fn encode(&self, j12: Half<i32>, s1: Orb, s2: Orb) -> Option<ChanState> {
        self.aux_encoder.get(&(j12, s1, s2)).cloned()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateMask10 {
    /// `occ1 -> needed`
    pub x_mask: [bool; 2],
}

impl StateMask10 {
    #[inline]
    pub fn new(xs: &[Occ]) -> Self {
        StateMask10 { x_mask: [xs.contains(&occ::I), xs.contains(&occ::A)] }
    }

    #[inline]
    pub fn test_occ(self, occ: Occ) -> bool {
        self.x_mask[usize::from(occ)]
    }

    pub fn next_occ(self, occ: &mut Fence<Option<Occ>>) -> Option<Occ> {
        while let Some(x) = occ.next() {
            if self.test_occ(x) {
                return Some(x);
            }
        }
        None
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

    pub fn next_occ20(self, occ: &mut Fence<Option<Occ20>>) -> Option<Occ20> {
        while let Some(x) = occ.next() {
            if self.test_occ20(x) {
                return Some(x);
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct StatesJ10<'a> {
    pub l_range: Range<u32>,
    pub states: CostatesJ10<'a>,
}

impl<'a> StatesJ10<'a> {
    pub fn new(scheme: &'a JScheme, mask: StateMask10) -> Self {
        Self {
            l_range: 1 .. scheme.basis_10.num_chans(),
            states: CostatesJ10::new(scheme, 0, mask),
        }
    }
}

impl<'a> Iterator for StatesJ10<'a> {
    type Item = StateJ10<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // for l in l_range {
        //     for s in costates_10(l) {
        //         yield s;
        //     }
        // }
        loop {
            if let r@Some(_) = self.states.next() {
                return r;
            }
            if let Some(l) = self.l_range.next() {
                self.states = CostatesJ10::new(
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
pub struct CostatesJ10<'a> {
    pub scheme: &'a JScheme,
    pub u_range: Range<u32>,
    pub x: Fence<Option<Occ>>,
    pub l: u32,
    pub mask: StateMask10,
}

impl<'a> CostatesJ10<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, l: u32, mask: StateMask10) -> Self {
        Self {
            scheme,
            u_range: 0 .. 0,
            x: Fence(Some(Occ::I)),
            l,
            mask,
        }
    }

    #[inline]
    pub fn scheme(&self) -> &'a JScheme {
        self.scheme
    }

    #[inline]
    pub fn mask(&self) -> StateMask10 {
        self.mask
    }
}

impl<'a> Iterator for CostatesJ10<'a> {
    type Item = StateJ10<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // for x in unmasked_occs(mask) {
        //     for u in aux_range(l, x) {
        //         yield state_10(l, u);
        //     }
        // }
        loop {
            // next u
            if let Some(u) = self.u_range.next() {
                return Some(self.scheme.state_10(ChanState { l: self.l, u }));
            }

            // next x
            if let Some(x) = self.mask.next_occ(&mut self.x) {
                self.u_range = self.scheme().basis_10.aux_range(
                    self.l,
                    x,
                );
                continue;
            }

            return None;
        }
    }
}

#[derive(Clone, Debug)]
pub struct StatesJ20<'a> {
    pub l_range: Range<u32>,
    pub states: CostatesJ20<'a>,
}

impl<'a> StatesJ20<'a> {
    pub fn new(scheme: &'a JScheme, mask: StateMask20) -> Self {
        Self {
            l_range: 1 .. scheme.basis_20.num_chans(),
            states: CostatesJ20::new(scheme, 0, mask),
        }
    }
}

impl<'a> Iterator for StatesJ20<'a> {
    type Item = StateJ20<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // for l in l_range {
        //     for s in costates_j20(l) {
        //         yield s;
        //     }
        // }
        loop {
            if let r@Some(_) = self.states.next() {
                return r;
            }
            if let Some(l) = self.l_range.next() {
                self.states = CostatesJ20::new(
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
pub struct CostatesJ20<'a> {
    pub scheme: &'a JScheme,
    pub state: Option<StateJ20<'a>>,
    pub u_range: Range<u32>,
    pub x: Fence<Option<Occ20>>,
    pub l: u32,
    pub mask: StateMask20,
}

impl<'a> CostatesJ20<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, l: u32, mask: StateMask20) -> Self {
        Self {
            scheme,
            state: None,
            u_range: 0 .. 0,
            x: Fence(Some(Occ20::II)),
            l,
            mask,
        }
    }

    #[inline]
    pub fn scheme(&self) -> &'a JScheme {
        self.scheme
    }

    #[inline]
    pub fn mask(&self) -> StateMask20 {
        self.mask
    }
}

impl<'a> Iterator for CostatesJ20<'a> {
    type Item = StateJ20<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // for x in unmasked_occs(mask) {
        //     for u in aux_range(l, x) {
        //         for state in state_20_permuts(l, x) {
        //             yield state;
        //         }
        //     }
        // }
        loop {
            // next permut
            while let Some(state) = StateJ20::next(&mut self.state) {
                if self.mask.test_occ_occ(state.s1.x, state.s2.x) {
                    return Some(state);
                }
            }

            loop {
                // next u
                if let Some(u) = self.u_range.next() {
                    self.state = Some(self.scheme.state_20(
                        ChanState { l: self.l, u },
                    ));
                    break;
                }

                // next x
                if let Some(x) = self.mask.next_occ20(&mut self.x) {
                    self.u_range = self.scheme().basis_20.aux_range(
                        self.l,
                        x,
                    );
                    continue;
                }

                return None;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JPartedOrb {
    pub lu: ChanState,
    pub j: Half<i32>,
    pub x: Occ,
}

impl fmt::Display for JPartedOrb {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{lu:{}}}", self.lu)
    }
}

impl JPartedOrb {
    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j.weight(exponent)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ10<'a> {
    pub scheme: &'a JScheme,
    pub s1: JPartedOrb,
}

impl<'a> fmt::Display for StateJ10<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{s1:{}}}", self.s1)
    }
}

impl<'a> Deref for StateJ10<'a> {
    type Target = JPartedOrb;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.s1
    }
}

impl<'a> StateJ10<'a> {
    #[inline]
    pub fn costates_10(self, xs: &[Occ]) -> CostatesJ10<'a> {
        CostatesJ10::new(self.scheme, self.s1.lu.l, StateMask10::new(xs))
    }

    #[inline]
    pub fn combine_with_10(
        &self,
        s2: Self,
        j12: Half<i32>,
    ) -> Option<StateJ20<'a>> {
        let mut p1 = self.scheme.basis_10.decode(self.s1.lu);
        let mut p2 = self.scheme.basis_10.decode(s2.s1.lu);
        let permut = if p1 < p2 {
            mem::swap(&mut p1, &mut p2);
            true
        } else {
            false
        };
        let s12 = self.scheme.state_20(
            match self.scheme.basis_20.encode(j12, p1, p2) {
                None => return None,
                Some(lu12) => lu12,
            },
        );
        Some(if permut {
            s12.next_permut().unwrap()
        } else {
            s12
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateJ20<'a> {
    pub scheme: &'a JScheme,
    pub lu12: ChanState,
    pub j12: Half<i32>,
    pub s1: JPartedOrb,
    pub s2: JPartedOrb,
    pub permut: u8,
    /// Number of states related by antisymmetry
    pub num_permut: u8,
    pub get_factor: f64,
    /// `set_factor == 1.0 / get_factor`
    pub set_factor: f64,
    /// `add_factor == 1.0 / (get_factor * num_permut)`
    pub add_factor: f64,
}

impl<'a> fmt::Display for StateJ20<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{lu12:{},permut:{}}}", self.lu12, self.permut)
    }
}

impl<'a> StateJ20<'a> {
    #[inline]
    pub fn next_permut(mut self) -> Option<Self> {
        self.permut += 1;
        if self.permut >= self.num_permut {
            return None;
        }
        mem::swap(&mut self.s1, &mut self.s2);
        let sign = -(self.s1.j + self.s2.j - self.j12).phase();
        self.get_factor = sign;
        self.set_factor = sign;
        self.add_factor = sign / self.num_permut as f64;
        Some(self)
    }

    #[inline]
    pub fn next(this: &mut Option<Self>) -> Option<Self> {
        if let &mut Some(x) = this {
            mem::replace(this, x.next_permut())
        } else {
            None
        }
    }

    #[inline]
    pub fn j(self) -> Half<i32> {
        self.j12
    }

    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j12.weight(exponent)
    }

    #[inline]
    pub fn costates_20(self, xs: &[[Occ; 2]]) -> CostatesJ20<'a> {
        CostatesJ20::new(self.scheme, self.lu12.l, StateMask20::new(xs))
    }

    #[inline]
    pub fn split_to_10_10(self) -> (StateJ10<'a>, StateJ10<'a>) {
        (
            StateJ10 { scheme: self.scheme, s1: self.s1 },
            StateJ10 { scheme: self.scheme, s1: self.s2 },
        )
    }
}

#[derive(Clone, Debug)]
pub struct JScheme {
    pub basis_10: BasisSchemeJ10,
    pub basis_20: BasisSchemeJ20,
}

impl JScheme {
    #[inline]
    pub fn parted_orb(&self, lu: ChanState) -> JPartedOrb {
        JPartedOrb {
            lu,
            j: self.basis_10.j_chan(lu.l).j,
            x: self.basis_10.occ(lu),
        }
    }

    #[inline]
    pub fn state_10(&self, lu1: ChanState) -> StateJ10 {
        StateJ10 {
            scheme: self,
            s1: self.parted_orb(lu1),
        }
    }

    #[inline]
    pub fn state_20(&self, lu12: ChanState) -> StateJ20 {
        let (s1, s2) = self.basis_20.decode(lu12);
        let num_permut = if s1 == s2 { 1 } else { 2 };
        StateJ20 {
            scheme: self,
            lu12,
            j12: self.basis_20.j_chan(lu12.l).j,
            s1: self.parted_orb(self.basis_10.encode(s1)),
            s2: self.parted_orb(self.basis_10.encode(s2)),
            permut: 0,
            num_permut,
            get_factor: 1.0,
            set_factor: 1.0,
            add_factor: 1.0 / num_permut as f64,
        }
    }

    #[inline]
    pub fn states_10(&self, xs: &[Occ]) -> StatesJ10 {
        StatesJ10::new(self, StateMask10::new(xs))
    }

    #[inline]
    pub fn states_20(&self, xs: &[[Occ; 2]]) -> StatesJ20 {
        StatesJ20::new(self, StateMask20::new(xs))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BasisJ10<'a>(pub &'a JScheme);

#[derive(Clone, Copy, Debug)]
pub struct BasisJ20<'a>(pub &'a JScheme);

impl<'a> ChartedBasis for BasisJ10<'a> {
    type State = StateJ10<'a>;
    fn layout(&self) -> &BasisLayout {
        &self.0.basis_10.layout
    }
    fn reify_state(&self, state: Self::State) -> ReifiedState {
        ReifiedState {
            chan: state.s1.lu.l as _,
            aux: state.s1.lu.u as _,
            get_factor: 1.0,
            set_factor: 1.0,
            add_factor: 1.0,
        }
    }
}

impl<'a> ChartedBasis for BasisJ20<'a> {
    type State = StateJ20<'a>;
    fn layout(&self) -> &BasisLayout {
        &self.0.basis_20.layout
    }
    fn reify_state(&self, state: Self::State) -> ReifiedState {
        ReifiedState {
            chan: state.lu12.l as _,
            aux: state.lu12.u as _,
            get_factor: state.get_factor,
            set_factor: state.set_factor,
            add_factor: state.add_factor,
        }
    }
}

pub type DiagOpJ10<'a, T> = Op<BasisJ10<'a>, BasisJ10<'a>, Block<Vec<T>>>;

pub type OpJ100<'a, T> = Op<BasisJ10<'a>, BasisJ10<'a>, Block<Mat<T>>>;

pub type OpJ200<'a, T> = Op<BasisJ20<'a>, BasisJ20<'a>, Block<Mat<T>>>;
