//! Implementation of the J-scheme abstraction.
//!
//! In our J-scheme, we classify states according to channels.  Unlike
//! M-scheme, however, we also track the angular momentum magnitude (J) of
//! each channel.
//!
//! J-scheme is strictly more general than M-scheme and can be used to emulate
//! M-scheme simply by treating all J's as zero.

use std::{f64, fmt, io, mem};
use std::hash::Hash;
use std::ops::{Add, Deref, Range, Sub};
use std::sync::Arc;
use fnv::FnvHashMap;
use num::Zero;
use rand;
use wigner_symbols::Wigner6j;
use super::ang_mom::Wigner6jCtx;
use super::basis::{occ, BasisChart, BasisLayout, ChanState, Fence, HashChart,
                   Occ, Occ20, Orb, OrbIx, PartState};
use super::block::Block;
use super::half::Half;
use super::mat::Mat;
use super::op::{ChartedBasis, Op, ReifiedState, ReifyState, VectorMut};
use super::tri_mat::Trs;
use super::utils::{self, Toler};

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

pub type JOrbBasis<K, U> = Vec<PartState<Occ, ChanState<JChan<K>, U>>>;

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

#[derive(Debug)]
pub struct JAtlas<K, U: Hash + Eq> { // FIXME: spurious constraints
    /// `k1 ↔ κ1`
    pub linchan1_chart: HashChart<K, u32>,
    /// `k2 ↔ κ2`
    pub linchan2_chart: HashChart<K, u32>,
    /// `(l1, μ1) → u1`
    pub aux_encoder: FnvHashMap<(u32, U), u32>,
    /// `i1 → μ1`
    pub aux_decoder: Box<[U]>,
    pub scheme: Arc<JScheme>,
}

impl<K, U: Hash + Eq> JAtlas<K, U> {
    pub fn scheme(&self) -> &Arc<JScheme> {
        &self.scheme
    }
}

impl<K, U: Hash + Eq> JAtlas<K, U>
    where K: Add<Output = K> + Sub<Output = K> + Hash + Eq + Clone,
          U: Hash + Eq + Clone,
{
    pub fn new(orbs: &JOrbBasis<K, U>) -> Self {
        let mut linchan1_chart = HashChart::default();
        let mut linchan2_chart = HashChart::default();
        let (basis_10, aux_encoder, aux_decoder) = BasisSchemeJ10::new(
            orbs,
            &mut linchan1_chart,
        );
        let basis_20 = BasisSchemeJ20::new(
            &basis_10,
            &linchan1_chart,
            &mut linchan2_chart,
        );
        let basis_21 = BasisSchemeJ21::new(
            &basis_10,
            &linchan1_chart,
            &mut linchan2_chart,
        );
        Self {
            linchan1_chart,
            linchan2_chart,
            aux_encoder,
            aux_decoder,
            scheme: Arc::new(JScheme {
                basis_10,
                basis_20,
                basis_21,
            }),
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
    /// `p → i`
    pub orb_from_ix: Box<[Orb]>,
    /// `i → p`
    pub orb_to_ix: Box<[OrbIx]>,
}

impl BasisSchemeJ10 {
    pub fn new<K: Add<Output = K> + Hash + Eq + Clone, U: Hash + Eq + Clone>(
        orbs: &JOrbBasis<K, U>,
        linchan1_chart: &mut HashChart<K, u32>,
    ) -> (Self, FnvHashMap<(u32, U), u32>, Box<[U]>)
    {
        let chart1 = BasisChart::new_with(orbs.iter().map(|state| {
            PartState {
                x: state.x,
                p: ChanState {
                    l: JChan {
                        j: state.p.l.j,
                        k: linchan1_chart.insert(state.p.l.k.clone()).index,
                    },
                    u: state.p.u.clone(),
                },
            }
        }), Default::default(), Occ::chart());
        (
            Self {
                layout: chart1.layout,
                chan_chart: chart1.chan_chart,
                orb_from_ix: chart1.orb_from_ix,
                orb_to_ix: chart1.orb_to_ix,
            },
            chart1.aux_encoder,
            chart1.aux_decoder,
        )
    }

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
    pub fn encode(&self, i: Orb) -> ChanState {
        self.layout.channelize(i.0)
    }

    /// Panics if the argument is invalid.
    #[inline]
    pub fn orb_from_ix(&self, p: OrbIx) -> Orb {
        self.orb_from_ix[p.0 as usize]
    }

    /// Panics if the argument is invalid.
    #[inline]
    pub fn orb_to_ix(&self, i: Orb) -> OrbIx {
        self.orb_to_ix[i.0 as usize]
    }
}

#[derive(Clone, Debug)]
pub struct BasisSchemeJ20 {
    pub layout: BasisLayout,
    pub chan_chart: HashChart<JChan, u32>,
    /// `(j12, i1, i2) → (l12, u12)`
    pub aux_encoder: FnvHashMap<(Half<i32>, Orb, Orb), ChanState>,
    /// `i12 → (i1, i2)`
    pub aux_decoder: Box<[(Orb, Orb)]>,
}

impl BasisSchemeJ20 {
    pub fn new<K: Add<Output = K> + Hash + Eq + Clone>(
        basis_10: &BasisSchemeJ10,
        linchan1_chart: &HashChart<K, u32>,
        linchan2_chart: &mut HashChart<K, u32>,
    ) -> Self
    {
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
                        let k12 = linchan2_chart.insert(k12).index;
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
        Self {
            layout: chart2.layout,
            chan_chart: chart2.chan_chart,
            aux_encoder: aux_encoder2,
            aux_decoder: chart2.aux_decoder,
        }
    }

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

#[derive(Clone, Debug)]
pub struct BasisSchemeJ21 {
    pub layout: BasisLayout,
    pub chan_chart: HashChart<JChan, u32>,
    /// `(j12, p1, p2) → (l12, u12)`
    pub aux_encoder: FnvHashMap<(Half<i32>, Orb, Orb), ChanState>,
    /// `p12 → (p1, p2)`
    pub aux_decoder: Box<[(Orb, Orb)]>,
}

impl BasisSchemeJ21 {
    pub fn new<K: Sub<Output = K> + Hash + Eq + Clone>(
        basis_10: &BasisSchemeJ10,
        linchan1_chart: &HashChart<K, u32>,
        linchan2_chart: &mut HashChart<K, u32>,
    ) -> Self
    {
        let mut states2 = Vec::default();
        for l1 in 0 .. basis_10.num_chans() {
            for l2 in 0 .. basis_10.num_chans() {
                for u1 in basis_10.auxs(l1, Occ::I, Occ::A) {
                    for u2 in basis_10.auxs(l2, Occ::I, Occ::A) {
                        let lu1 = ChanState { l: l1, u: u1 };
                        let lu2 = ChanState { l: l2, u: u2 };
                        let p1 = basis_10.decode(lu1);
                        let p2 = basis_10.decode(lu2);
                        let jk1 = basis_10.j_chan(l1);
                        let jk2 = basis_10.j_chan(l2);
                        let k12 =
                            linchan1_chart.decode(jk1.k).unwrap().clone()
                            - linchan1_chart.decode(jk2.k).unwrap().clone();
                        let k12 = linchan2_chart.insert(k12).index;
                        let x12 = [Occ::from(basis_10.occ(lu1)),
                                   Occ::from(basis_10.occ(lu2))];
                        for j12 in Half::tri_range(jk1.j, jk2.j) {
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
            Occ::chart2(),
        );
        let aux_encoder2 = chart2.aux_encoder.iter()
            .map(|(&(l, (p1, p2)), &u)| {
                let j = chart2.decode_chan(l).unwrap().j;
                ((j, p1, p2), ChanState { l, u })
            })
            .collect();
        Self {
            layout: chart2.layout,
            chan_chart: chart2.chan_chart,
            aux_encoder: aux_encoder2,
            aux_decoder: chart2.aux_decoder,
        }
    }

    #[inline]
    pub fn num_chans(&self) -> u32 {
        self.layout.num_chans()
    }

    #[inline]
    pub fn auxs(&self, l: u32, x1: [Occ; 2], x2: [Occ; 2]) -> Range<u32> {
        Range {
            start: self.layout.part_offset(l, Occ::occ2_to_usize(x1) as u32),
            end: self.layout.part_offset(l, Occ::occ2_to_usize(x2) as u32 + 1),
        }
    }

    #[inline]
    pub fn aux_range(&self, l: u32, x: [Occ; 2]) -> Range<u32> {
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
    pub fn encode(
        &self,
        j12: Half<i32>,
        i1: Orb,
        i2: Orb,
    ) -> Option<ChanState>
    {
        self.aux_encoder.get(&(j12, i1, i2)).cloned()
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
        Self {
            x_mask: [ii, ai || ia, aa],
            xx_mask: [ii, ai, ia, aa],
        }
    }

    #[inline]
    pub fn test_occ20(self, occ: Occ20) -> bool {
        self.x_mask[usize::from(occ)]
    }

    #[inline]
    pub fn test_occ(self, occ: [Occ; 2]) -> bool {
        self.xx_mask[Occ::occ2_to_usize(occ)]
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

#[derive(Clone, Copy, Debug)]
pub struct StateMask21 {
    /// `(occ1 + occ2 * 2) -> needed`
    pub x_mask: [bool; 4],
}

impl StateMask21 {
    #[inline]
    pub fn new(xs: &[[Occ; 2]]) -> Self {
        Self {
            x_mask: [
                xs.contains(&occ::II),
                xs.contains(&occ::AI),
                xs.contains(&occ::IA),
                xs.contains(&occ::AA),
            ],
        }
    }

    #[inline]
    pub fn test_occ(self, occ: [Occ; 2]) -> bool {
        self.x_mask[Occ::occ2_to_usize(occ)]
    }

    pub fn next_occ(self, occ: &mut Option<[Occ; 2]>) -> Option<[Occ; 2]> {
        while let Some(x) = Occ::next_occ2(occ) {
            if self.test_occ(x) {
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
                if self.mask.test_occ([state.s1.x, state.s2.x]) {
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

#[derive(Clone, Debug)]
pub struct CostatesJ21<'a> {
    pub scheme: &'a JScheme,
    pub u_range: Range<u32>,
    pub x: Option<[Occ; 2]>,
    pub l: u32,
    pub mask: StateMask21,
}

impl<'a> CostatesJ21<'a> {
    #[inline]
    pub fn new(scheme: &'a JScheme, l: u32, mask: StateMask21) -> Self {
        Self {
            scheme,
            u_range: 0 .. 0,
            x: Some(occ::II),
            l,
            mask,
        }
    }

    #[inline]
    pub fn scheme(&self) -> &'a JScheme {
        self.scheme
    }

    #[inline]
    pub fn mask(&self) -> StateMask21 {
        self.mask
    }
}

impl<'a> Iterator for CostatesJ21<'a> {
    type Item = StateJ21<'a>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // for x in unmasked_occs(mask) {
        //     for u in aux_range(l, x) {
        //         yield state_21(l, u);
        //     }
        // }
        loop {
            // next u
            if let Some(u) = self.u_range.next() {
                return Some(self.scheme.state_21(ChanState { l: self.l, u }));
            }

            // next x
            if let Some(x) = self.mask.next_occ(&mut self.x) {
                self.u_range = self.scheme().basis_21.aux_range(
                    self.l,
                    x,
                );
                continue;
            }

            return None;
        }
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
    pub fn j(self) -> Half<i32> {
        self.s1.j
    }

    #[inline]
    pub fn lu(self) -> ChanState {
        self.s1.lu
    }

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

    #[inline]
    pub fn combine_with_10_to_21(
        &self,
        s2: Self,
        j12: Half<i32>,
    ) -> Option<StateJ21<'a>> {
        let p1 = self.scheme.basis_10.decode(self.s1.lu);
        let p2 = self.scheme.basis_10.decode(s2.s1.lu);
        let s12 = self.scheme.state_21(
            match self.scheme.basis_21.encode(j12, p1, p2) {
                None => return None,
                Some(lu12) => lu12,
            },
        );
        Some(s12)
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
    pub fn lu(self) -> ChanState {
        self.lu12
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

#[derive(Clone, Copy, Debug)]
pub struct StateJ21<'a> {
    pub scheme: &'a JScheme,
    pub lu12: ChanState,
    pub j12: Half<i32>,
    pub s1: JPartedOrb,
    pub s2: JPartedOrb,
}

impl<'a> fmt::Display for StateJ21<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{lu12:{}}}", self.lu12)
    }
}

impl<'a> StateJ21<'a> {
    #[inline]
    pub fn j(self) -> Half<i32> {
        self.j12
    }

    #[inline]
    pub fn lu(self) -> ChanState {
        self.lu12
    }

    #[inline]
    pub fn jweight(self, exponent: i32) -> f64 {
        self.j12.weight(exponent)
    }

    #[inline]
    pub fn costates_21(self, xs: &[[Occ; 2]]) -> CostatesJ21<'a> {
        CostatesJ21::new(self.scheme, self.lu12.l, StateMask21::new(xs))
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
    pub basis_21: BasisSchemeJ21,
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
    pub fn state_21(&self, lu12: ChanState) -> StateJ21 {
        let (s1, s2) = self.basis_21.decode(lu12);
        StateJ21 {
            scheme: self,
            lu12,
            j12: self.basis_21.j_chan(lu12.l).j,
            s1: self.parted_orb(self.basis_10.encode(s1)),
            s2: self.parted_orb(self.basis_10.encode(s2)),
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

#[derive(Clone, Copy, Debug, Default)]
pub struct BasisJ10;

impl ChartedBasis for BasisJ10 {
    type Scheme = Arc<JScheme>;
    fn layout<'a>(&self, scheme: &'a Self::Scheme) -> &'a BasisLayout {
        &scheme.basis_10.layout
    }
}

impl<'a> ReifyState for StateJ10<'a> {
    type Scheme = Arc<JScheme>;
    type Basis = BasisJ10;
    fn reify_state(
        self,
        _scheme: &Self::Scheme,
        _basis: &Self::Basis,
    ) -> ReifiedState {
        let lu = self.lu();
        ReifiedState {
            chan: lu.l as _,
            aux: lu.u as _,
            get_factor: 1.0,
            set_factor: 1.0,
            add_factor: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasisJ20;

impl ChartedBasis for BasisJ20 {
    type Scheme = Arc<JScheme>;
    fn layout<'a>(&self, scheme: &'a Self::Scheme) -> &'a BasisLayout {
        &scheme.basis_20.layout
    }
}

impl<'a> ReifyState for StateJ20<'a> {
    type Scheme = Arc<JScheme>;
    type Basis = BasisJ20;
    fn reify_state(
        self,
        _scheme: &Self::Scheme,
        _basis: &Self::Basis,
    ) -> ReifiedState {
        let lu = self.lu();
        ReifiedState {
            chan: lu.l as _,
            aux: lu.u as _,
            get_factor: self.get_factor,
            set_factor: self.set_factor,
            add_factor: self.add_factor,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BasisJ21;

impl ChartedBasis for BasisJ21 {
    type Scheme = Arc<JScheme>;
    fn layout<'a>(&self, scheme: &'a Self::Scheme) -> &'a BasisLayout {
        &scheme.basis_21.layout
    }
}

impl<'a> ReifyState for StateJ21<'a> {
    type Scheme = Arc<JScheme>;
    type Basis = BasisJ21;
    fn reify_state(
        self,
        _scheme: &Self::Scheme,
        _basis: &Self::Basis,
    ) -> ReifiedState {
        let lu = self.lu();
        ReifiedState {
            chan: lu.l as _,
            aux: lu.u as _,
            get_factor: 1.0,
            set_factor: 1.0,
            add_factor: 1.0,
        }
    }
}

/// A diagonal standard-coupled one-body matrix.
pub type DiagOpJ10<T = f64> =
    Op<Arc<JScheme>, BasisJ10, BasisJ10, Block<Vec<T>>>;

/// A standard-coupled one-body matrix.
pub type OpJ100<T = f64> = Op<Arc<JScheme>, BasisJ10, BasisJ10, Block<Mat<T>>>;

/// A standard-coupled two-body matrix.
pub type OpJ200<T = f64> = Op<Arc<JScheme>, BasisJ20, BasisJ20, Block<Mat<T>>>;

/// A Pandya-coupled two-body matrix.
pub type OpJ211<T = f64> = Op<Arc<JScheme>, BasisJ21, BasisJ21, Block<Mat<T>>>;

/// A standard-coupled (0, 1, 2)-body matrix.
pub type MopJ012<T> = (T, OpJ100<T>, OpJ200<T>);

pub fn new_mop_j012<T: Default + Clone>(scheme: &Arc<JScheme>) -> MopJ012<T> {
    (
        Default::default(),
        Op::new(scheme.clone()),
        Op::new(scheme.clone()),
    )
}

pub fn set_zero_mop_j012<T: Zero + Clone>(m: &mut MopJ012<T>) {
    m.0 = Zero::zero();
    m.1.set_zero();
    m.2.set_zero();
}

pub fn rand_mop_j012(
    scheme: &Arc<JScheme>,
    mut rng: &mut rand::Rng,
) -> MopJ012<f64> {
    use rand::Rand;
    use rand::distributions::normal::StandardNormal;
    let mut a = new_mop_j012(scheme);
    a.0 = StandardNormal::rand(&mut rng).0;
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            a.1.set(p, q, StandardNormal::rand(&mut rng).0);
        }
    }
    for pq in scheme.states_20(&occ::ALL2) {
        for rs in pq.costates_20(&occ::ALL2) {
            a.2.set(pq, rs, StandardNormal::rand(&mut rng).0);
        }
    }
    a
}

/// Read M-scheme matrix elements of a (0, 1, 2)-body operator from a text
/// file, where states are encoded as orbital indices.
pub fn read_mop_j012_txt(
    scheme: &Arc<JScheme>,
    reader: &mut io::BufRead,
) -> io::Result<MopJ012<f64>>
{
    use std::io::BufRead;
    let convert = |p: &str| -> StateJ10 {
        let p = OrbIx(p.parse().expect("cannot parse index of matrix element"));
        scheme.state_10(scheme.basis_10.encode(scheme.basis_10.orb_from_ix(p)))
    };
    let mut a = new_mop_j012(scheme);
    for line in reader.lines() {
        let line = line?;
        let words: Vec<_> = line.split_whitespace().collect();
        let first = match words.first() {
            Some(first) => first,
            None => continue,
        };
        if first.starts_with('#') {
            continue;
        }
        let value: f64 = words.last().unwrap().parse()
            .expect("cannot parse value of matrix element");
        match words.len() {
            1 => {
                a.0 = value;
            }
            3 => {
                let p = convert(words[0]);
                let q = convert(words[1]);
                a.1.set(p, q, value);
            }
            5 => {
                let p = convert(words[0]);
                let q = convert(words[1]);
                let r = convert(words[2]);
                let s = convert(words[3]);
                if p.lu == q.lu || r.lu == s.lu {
                    // if we don't catch it here, combine_with_10 will fail
                    panic!("Pauli violation");
                }
                let pq = p.combine_with_10(q, Half(0))
                    .expect("basis is not M-scheme");
                let rs = r.combine_with_10(s, Half(0))
                    .expect("basis is not M-scheme");
                a.2.set(pq, rs, value);
            }
            _ => panic!("line has incorrect number of fields"),
        }
    }
    Ok(a)
}

// FIXME: I think it's high time we created a dedicated MopJ012 type

pub fn extent_mop_j012_as_tri<'a, T: Clone>(m: &MopJ012<T>) -> usize {
    1
        + m.1.data.extent_mat_as_tri()
        + m.2.data.extent_mat_as_tri()
}

pub fn clone_mop_j012_to_tri_slice<'a, T: Clone>(
    m: &MopJ012<T>,
    mut a: &'a mut [T],
) -> &'a mut [T]
{
    a[0] = m.0.clone();
    a = &mut move_ref!(a)[1 ..];
    a = m.1.data.clone_mat_to_tri_slice(move_ref!(a));
    a = m.2.data.clone_mat_to_tri_slice(move_ref!(a));
    a
}

pub fn clone_mop_j012_from_tri_slice<'a, S: Trs<T>, T: Clone>(
    m: &mut MopJ012<T>,
    trs: &S,
    mut a: &'a [T],
) -> &'a [T]
{
    m.0 = a[0].clone();
    a = &a[1 ..];
    a = m.1.data.clone_mat_from_tri_slice(trs, a);
    a = m.2.data.clone_mat_from_tri_slice(trs, a);
    a
}

pub fn check_eq_op_j100(
    toler: Toler,
    a1: &OpJ100,
    b1: &OpJ100,
) -> Result<(), String> {
    let scheme = a1.scheme();
    let mut status = Ok(());
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            let left = a1.at(p, q);
            let right = b1.at(p, q);
            if !toler.is_eq(left, right) {
                let err = format!("{},{}: {} != {} within {:?}",
                                  p.s1.lu, q.s1.lu, left, right, toler);
                eprintln!("{}", err);
                status = status.and(Err(err));
            }
        }
    }
    status
}

pub fn check_eq_op_j200(
    toler: Toler,
    a2: &OpJ200,
    b2: &OpJ200,
) -> Result<(), String> {
    let scheme = a2.scheme();
    let mut status = Ok(());
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            let left = a2.at(pq, rs);
            let right = b2.at(pq, rs);
            if !toler.is_eq(left, right) {
                let err = format!("{},{},{},{}: {} != {} within {:?}",
                                  p.s1.lu, q.s1.lu, r.s1.lu, s.s1.lu,
                                  left, right, toler);
                eprintln!("{}", err);
                status = status.and(Err(err));
            }
        }
    }
    status
}

pub fn check_eq_mop_j012(
    toler: Toler,
    a: &MopJ012<f64>,
    b: &MopJ012<f64>,
) -> Result<(), String> {
    if toler.is_eq(a.0, b.0) {
        Ok(())
    } else {
        let err = format!("{} != {} within {:?}", a.0, b.0, toler);
        eprintln!("{}", err);
        Err(err)
    }   .and(check_eq_op_j100(toler, &a.1, &b.1))
        .and(check_eq_op_j200(toler, &a.2, &b.2))
}

/// Pandya transformation.
///
/// ```text
/// B[p s r q] ←+ −α ∑[Jpq] { Jp Jq Jpq; Jr Js Jps } (−)^(2 Jpq) Jpq^2 A[p q r s]
/// ```
pub fn op200_to_op211(
    w6j_ctx: &mut Wigner6jCtx,
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &mut OpJ211<f64>,
)
{
    let scheme = a2.scheme();
    for pq in scheme.states_20(&occ::ALL2) {
        for rs in pq.costates_20(&occ::ALL2) {
            let w = -alpha
                * pq.j().double().phase()
                * pq.jweight(2)
                * a2.at(pq, rs);
            let (p, q) = pq.split_to_10_10();
            let (r, s) = rs.split_to_10_10();
            for jps in Half::tri_range_2(
                (p.j(), s.j()),
                (r.j(), q.j()),
            ) {
                let ps = p.combine_with_10_to_21(s, jps).unwrap();
                let rq = r.combine_with_10_to_21(q, jps).unwrap();
                b2.add(ps, rq, w * w6j_ctx.get(Wigner6j {
                    tj1: p.j().twice(),
                    tj2: q.j().twice(),
                    tj3: pq.j().twice(),
                    tj4: r.j().twice(),
                    tj5: s.j().twice(),
                    tj6: ps.j().twice(),
                }));
            }
        }
    }
}

/// Inverse Pandya transformation.
///
/// ```text
/// B[p q r s] ←+ −α ∑[Jpq] { Jp Jq Jpq; Jr Js Jps } (−)^(2 Jpq) Jps^2 A[p s r q]
/// ```
pub fn op211_to_op200(
    w6j_ctx: &mut Wigner6jCtx,
    alpha: f64,
    a2: &OpJ211<f64>,
    b2: &mut OpJ200<f64>,
)
{
    let scheme = a2.scheme();
    for pq in scheme.states_20(&occ::ALL2) {
        for rs in pq.costates_20(&occ::ALL2) {
            let (p, q) = pq.split_to_10_10();
            let (r, s) = rs.split_to_10_10();
            let mut x = 0.0;
            for jps in Half::tri_range_2(
                (p.j(), s.j()),
                (r.j(), q.j()),
            ) {
                let ps = p.combine_with_10_to_21(s, jps).unwrap();
                let rq = r.combine_with_10_to_21(q, jps).unwrap();
                x += w6j_ctx.get(Wigner6j {
                    tj1: p.j().twice(),
                    tj2: q.j().twice(),
                    tj3: pq.j().twice(),
                    tj4: r.j().twice(),
                    tj5: s.j().twice(),
                    tj6: ps.j().twice(),
                }) * ps.jweight(2) * a2.at(ps, rq);
            }
            b2.add(pq, rs, -alpha * pq.j().double().phase() * x);
        }
    }
}
