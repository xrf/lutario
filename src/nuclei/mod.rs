//! Nuclei systems.
//!
//! Here we use the particle physics convention of proton = +½, neutron = −½.

pub mod darmstadt;
pub mod vrenorm;

use std::{fmt, iter, str};
use std::error::Error;
use std::ops::{Add, Sub};
use fnv::{FnvHashMap, FnvHashSet};
use num::Zero;
use regex::Regex;
use wigner_symbols::ClebschGordan;
use super::ang_mom::Wigner3jmCtx;
use super::basis::{occ, ChanState, HashChart, Occ, PartState};
use super::half::Half;
use super::j_scheme::{JAtlas, JChan, JOrbBasis, OpJ100, OpJ200};
use super::op::Op;
use super::parity::{self, Parity};
use super::utils;

lazy_static! {
    pub static ref ORB_ANG_CHART: HashChart<char> =
        "spdfghiklmnoqrtuvwxyz".chars().collect();
}

/// Orbital angular momentum magnitude (l)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrbAng(pub i32);

/// Display the spectroscopic label using the `ORB_ANG_CHART` alphabet.
impl fmt::Display for OrbAng {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s: String = utils::encode_number(self.0 as _, &ORB_ANG_CHART)
            .into_iter().collect();
        write!(f, "{}", s)
    }
}

/// Parse a spectroscopic label using the `ORB_ANG_CHART` alphabet.
impl str::FromStr for OrbAng {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(OrbAng(utils::decode_number(&mut s.chars(), &ORB_ANG_CHART)? as _))
    }
}

/// Principal quantum number and total angular momentum magnitude
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nj {
    pub n: i32,
    pub j: Half<i32>,
}

impl From<Npjmw> for Nj {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, j: s.j }
    }
}

/// Principal quantum number, orbital angular momentum magnitude, and total
/// angular momentum magnitude
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nlj {
    /// Principal quantum number (n)
    pub n: i32,
    /// Orbital angular momentum magnitude (l)
    pub l: OrbAng,
    /// Total angular momentum magnitude (j)
    pub j: Half<i32>,
}

/// Display using spectroscopic notation.
impl fmt::Display for Nlj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}", self.n, self.l, self.j)
    }
}

/// Parse from spectroscopic notation.
impl str::FromStr for Nlj {
    type Err = Box<Error + Send + Sync>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let m = re!(r"(\d+)(\w+)(\d+)/2").captures(s).ok_or("invalid format")?;
        let n = m.get(1).unwrap().as_str().parse()?;
        let OrbAng(l) = m.get(2).unwrap().as_str().parse()?;
        let j = Half(m.get(3).unwrap().as_str().parse()?);
        if Half::from(l).abs_diff(j) != Half(1) {
            return Err("l must be within 1/2 of j".into());
        }
        Ok(Self { n, l: OrbAng(l), j })
    }
}

impl From<Npj> for Nlj {
    fn from(this: Npj) -> Self {
        debug_assert!(this.j.try_get().is_err());
        let a = this.j.twice() / 2;     // intentional integer division
        Self {
            n: this.n,
            l: OrbAng(a + (a + i32::from(this.p)) % 2),
            j: this.j,
        }
    }
}

impl From<Npjw> for Nlj {
    fn from(s: Npjw) -> Self {
        Npj::from(s).into()
    }
}

impl Nlj {
    /// Shell index (e)
    pub fn shell(self) -> i32 {
        2 * self.n + self.l.0
    }

    /// Returns harmonic oscillator energy in natural units.
    pub fn osc_energy(self) -> f64 {
        1.5 + self.shell() as f64
    }
}

/// Principal quantum number, parity, and total angular momentum magnitude
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Npj {
    /// Principal quantum number (n)
    pub n: i32,
    /// Parity (π)
    pub p: Parity,
    /// Total angular momentum magnitude (j)
    pub j: Half<i32>,
}

/// Display using spectroscopic notation.
impl fmt::Display for Npj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Nlj::from(*self).fmt(f)
    }
}

/// Parse from spectroscopic notation.
impl str::FromStr for Npj {
    type Err = Box<Error + Send + Sync>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Nlj::from_str(s).map(Npj::from)
    }
}

impl From<Nlj> for Npj {
    fn from(this: Nlj) -> Self {
        Self { n: this.n, p: Parity::of(this.l.0), j: this.j }
    }
}

impl From<Npjw> for Npj {
    fn from(s: Npjw) -> Self {
        Self { n: s.n, p: s.p, j: s.j }
    }
}

impl From<Npjmw> for Npj {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, p: s.p, j: s.j }
    }
}

impl Npj {
    /// Shell index (e)
    pub fn shell(self) -> i32 {
        Nlj::from(self).shell()
    }

    /// Returns harmonic oscillator energy in natural units.
    pub fn osc_energy(self) -> f64 {
        Nlj::from(self).osc_energy()
    }

    pub fn and_w(self, w: Half<i32>) -> Npjw {
        Npjw {
            n: self.n,
            p: self.p,
            j: self.j,
            w,
        }
    }

    pub fn to_j_chan_state(self) -> ChanState<JChan<Parity>, i32> {
        ChanState {
            l: JChan { j: self.j, k: self.p },
            u: self.n,
        }
    }
}

/// Principal quantum number, parity, total angular momentum magnitude, and
/// isospin projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
         Serialize, Deserialize)]
pub struct Npjw {
    /// Principal quantum number (n)
    pub n: i32,
    /// Parity (π)
    pub p: Parity,
    /// Total angular momentum magnitude (j)
    pub j: Half<i32>,
    /// Isospin projection (w)
    pub w: Half<i32>,
}

/// Display using spectroscopic notation.
impl fmt::Display for Npjw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Npj::from(*self).fmt(f)?;
        match self.w {
            Half(-1) => write!(f, "n"),
            Half(1) => write!(f, "p"),
            _ => write!(f, "(w={})", self.w),
        }
    }
}

/// Parse from spectroscopic notation.
impl str::FromStr for Npjw {
    type Err = Box<Error + Send + Sync>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let m = re!(r"(.+)[pn]").captures(s).ok_or("expected p or n suffix")?;
        let npj: Npj = m.get(1).unwrap().as_str().parse()?;
        let w = match m.get(2).unwrap().as_str() {
            "p" => Half(-1),
            "n" => Half(1),
            _ => panic!("huh?"),
        };
        Ok(npj.and_w(w))
    }
}

impl From<ChanState<JChan<Pw>, i32>> for Npjw {
    fn from(s: ChanState<JChan<Pw>, i32>) -> Self {
        Self { n: s.u, p: s.l.k.p, j: s.l.j, w: s.l.k.w }
    }
}

impl From<Npjmw> for Npjw {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, p: s.p, j: s.j, w: s.w }
    }
}

impl Npjw {
    /// Shell index.
    pub fn shell(self) -> i32 {
        Npj::from(self).shell()
    }

    /// Returns harmonic oscillator energy in natural units.
    pub fn osc_energy(self) -> f64 {
        Npj::from(self).osc_energy()
    }

    pub fn and_m(self, m: Half<i32>) -> Npjmw {
        Npjmw { n: self.n, p: self.p, j: self.j, m, w: self.w }
    }
}

/// Principal quantum number, parity, total angular momentum magnitude, total
/// angular momentum projection, and isospin projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Npjmw {
    /// Principal quantum number (n)
    pub n: i32,
    /// Parity (π)
    pub p: Parity,
    /// Total angular momentum magnitude (j)
    pub j: Half<i32>,
    /// Total angular momentum projection (m)
    pub m: Half<i32>,
    /// Isospin projection (w)
    pub w: Half<i32>,
}

impl From<ChanState<JChan<Pmw>, Nj>> for Npjmw {
    fn from(s: ChanState<JChan<Pmw>, Nj>) -> Self {
        debug_assert_eq!(s.l.j, Zero::zero());
        Self { n: s.u.n, p: s.l.k.p, j: s.u.j, m: s.l.k.m, w: s.l.k.w }
    }
}

impl Npjmw {
    /// Shell index.
    pub fn shell(self) -> i32 {
        Npj::from(self).shell()
    }
}

/// Display using spectroscopic notation.
impl fmt::Display for Npjmw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Npj::from(*self).fmt(f)?;
        write!(f, "[{}]", self.m)?;
        match self.w {
            Half(-1) => write!(f, "n"),
            Half(1) => write!(f, "p"),
            _ => write!(f, "(w={})", self.w),
        }
    }
}

/// Parity and total angular momentum magnitude
#[derive(Clone, Copy, Debug)]
pub struct Pj {
    pub p: Parity,
    pub j: Half<i32>,
}

/// Parity, total angular momentum magnitude, isospin magnitude, and isospin
/// projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pjtw {
    pub p: Parity,
    pub j: Half<i32>,
    pub t: Half<i32>,
    pub w: Half<i32>,
}

impl fmt::Display for Pjtw {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}[{}{}{}]",
            self.j,
            self.p.sign_char(),
            self.t,
            if self.w == Zero::zero() {
                ","
            } else if self.w > Zero::zero() {
                "+"
            } else {
                ""
            },
            self.w,
        )
    }
}

/// Parity, total angular momentum projection, and isospin projection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pmw {
    pub p: Parity,
    pub m: Half<i32>,
    pub w: Half<i32>,
}

impl From<Npjmw> for Pmw {
    fn from(s: Npjmw) -> Self {
        Self { p: s.p, m: s.m, w: s.w }
    }
}

impl From<Npjmw> for ChanState<JChan<Pmw>, Nj> {
    fn from(s: Npjmw) -> Self {
        Self { l: Pmw::from(s).into(), u: s.into() }
    }
}

impl Add for Pmw {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self { p: self.p + other.p, m: self.m + other.m, w: self.w + other.w }
    }
}

impl Sub for Pmw {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self { p: self.p - other.p, m: self.m - other.m, w: self.w - other.w }
    }
}

impl Zero for Pmw {
    fn zero() -> Self {
        Self { p: Zero::zero(), m: Zero::zero(), w: Zero::zero() }
    }
    fn is_zero(&self) -> bool {
        self.p.is_zero() && self.m.is_zero() && self.w.is_zero()
    }
}

/// Parity and isospin projection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pw {
    pub p: Parity,
    pub w: Half<i32>,
}

impl From<Npjw> for Pw {
    fn from(s: Npjw) -> Self {
        Self { p: s.p, w: s.w }
    }
}

impl From<Npjw> for JChan<Pw> {
    fn from(s: Npjw) -> Self {
        Self { j: s.j, k: s.into() }
    }
}

impl From<Npjw> for ChanState<JChan<Pw>, i32> {
    fn from(s: Npjw) -> Self {
        Self { l: s.into(), u: s.n }
    }
}

impl Add for Pw {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self { p: self.p + other.p, w: self.w + other.w }
    }
}

impl Sub for Pw {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self { p: self.p - other.p, w: self.w - other.w }
    }
}

impl Zero for Pw {
    fn zero() -> Self {
        Self { p: Zero::zero(), w: Zero::zero() }
    }
    fn is_zero(&self) -> bool {
        self.p.is_zero() && self.w.is_zero()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JtwNpjKey {
    pub j12: Half<i32>,
    pub t12: Half<i32>,
    pub w12: Half<i32>,
    pub npj1: Npj,
    pub npj2: Npj,
    pub npj3: Npj,
    pub npj4: Npj,
}

impl JtwNpjKey {
    pub fn canonicalize(self) -> (f64, bool, Self) {
        let (f12, npj1, npj2) = parity::sort2(self.npj1, self.npj2);
        let f12 = if f12 == Parity::Odd {
            (self.npj1.j + self.npj2.j - self.j12 - self.t12).phase()
        } else {
            1.0
        };
        let (f34, npj3, npj4) = parity::sort2(self.npj3, self.npj4);
        let f34 = if f34 == Parity::Odd {
            (self.npj3.j + self.npj4.j - self.j12 - self.t12).phase()
        } else {
            1.0
        };
        let (adj, (npj1, npj2), (npj3, npj4)) =
            parity::sort2((npj1, npj2), (npj3, npj4));
        (
            f12 * f34,
            adj == Parity::Odd,
            Self {
                j12: self.j12,
                t12: self.t12,
                w12: self.w12,
                npj1,
                npj2,
                npj3,
                npj4,
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JNpjwKey {
    pub j12: Half<i32>,
    pub npjw1: Npjw,
    pub npjw2: Npjw,
    pub npjw3: Npjw,
    pub npjw4: Npjw,
}

impl fmt::Display for JNpjwKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "J={} {} {} {} {}",
               self.j12, self.npjw1, self.npjw2, self.npjw3, self.npjw4)
    }
}

impl JNpjwKey {
    pub fn canonicalize(self) -> (f64, bool, Self) {
        let (f12, npjw1, npjw2) = parity::sort2(self.npjw1, self.npjw2);
        let f12 = if f12 == Parity::Odd {
            -(self.npjw1.j + self.npjw2.j - self.j12).phase()
        } else {
            1.0
        };
        let (f34, npjw3, npjw4) = parity::sort2(self.npjw3, self.npjw4);
        let f34 = if f34 == Parity::Odd {
            -(self.npjw3.j + self.npjw4.j - self.j12).phase()
        } else {
            1.0
        };
        let (adj, (npjw1, npjw2), (npjw3, npjw4)) =
            parity::sort2((npjw1, npjw2), (npjw3, npjw4));
        (
            f12 * f34,
            adj == Parity::Odd,
            Self { j12: self.j12, npjw1, npjw2, npjw3, npjw4 },
        )
    }
}

/// Iterator for 3D harmonic oscillator states in (e, l, j)-order.
#[derive(Clone, Debug)]
pub struct Ho3dIter(pub Nlj);

impl Default for Ho3dIter {
    fn default() -> Self {
        Ho3dIter(Nlj { n: 0, l: OrbAng(0), j: Half(1)})
    }
}

impl Iterator for Ho3dIter {
    type Item = Nlj;
    fn next(&mut self) -> Option<Self::Item> {
        let nlj = self.0;
        let OrbAng(l) = nlj.l;
        let greater_j = Half::from(l) + Half(1);
        self.0 = if nlj.j != greater_j {
            Nlj { j: greater_j, .. nlj }
        } else if nlj.n > 0 {
            Nlj { n: nlj.n - 1, l: OrbAng(l + 2), j: nlj.j + Half(2) }
        } else {
            let e = nlj.shell() + 1;
            Nlj { n: e / 2, l: OrbAng(e % 2), j: Half(1) }
        };
        Some(nlj)
    }
}

/// Truncation scheme for a 3D harmonic oscillator basis
#[derive(Clone, Copy, Debug)]
pub struct Ho3dTrunc {
    /// Maximum shell index
    pub e_max: i32,
    /// Maximum principal quantum number
    pub n_max: i32,
    /// Maximum orbital angular momentum magnitude
    pub l_max: i32,
}

impl fmt::Display for Ho3dTrunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{e_max: {}", self.e_max)?;
        if self.n_max != i32::max_value() {
            write!(f, ", n_max: {}", self.n_max)?;
        }
        if self.l_max != i32::max_value() {
            write!(f, ", l_max: {}", self.l_max)?;
        }
        write!(f, "}}")
    }
}

/// The default truncation sets `e_max` to `-1` and everything else to
/// `i32::max_value()`.
impl Default for Ho3dTrunc {
    fn default() -> Self {
        Self {
            e_max: -1,
            n_max: i32::max_value(),
            l_max: i32::max_value(),
        }
    }
}

impl Ho3dTrunc {
    pub fn is_empty(self) -> bool {
        self.e_max < 0 || self.n_max < 0 || self.l_max < 0
    }

    pub fn contains(self, nlj: Nlj) -> bool {
        nlj.shell() <= self.e_max
            && nlj.n <= self.n_max
            && nlj.l <= OrbAng(self.l_max)
    }

    /// Obtain a sequence of (n, π, j) quantum numbers in (e, l, j)-order.
    pub fn states(self) -> Vec<Npj> {
        Ho3dIter::default()
            .take_while(|nlj| nlj.shell() <= self.e_max)
            .filter(|&nlj| self.contains(nlj))
            .map(Npj::from)
            .collect()
    }
}

/// Modified 3D harmonic oscillator basis truncation
#[derive(Clone, Debug, Default)]
pub struct Ho3dModTrunc {
    pub trunc: Ho3dTrunc,
    pub incl: FnvHashSet<Nlj>,
    pub excl: FnvHashSet<Nlj>,
}

impl From<Ho3dTrunc> for Ho3dModTrunc {
    fn from(trunc: Ho3dTrunc) -> Self {
        Self { trunc, .. Default::default() }
    }
}

impl Ho3dModTrunc {
    pub fn e_max(&self) -> i32 {
        self.incl.iter()
            .map(|p| p.shell())
            .chain(iter::once(self.trunc.e_max))
            .max()
            .unwrap_or(-1)
    }

    pub fn contains(&self, nlj: Nlj) -> bool {
        self.trunc.contains(nlj)
            && !self.excl.contains(&nlj)
            || self.incl.contains(&nlj)
    }

    /// Obtain a sequence of (n, π, j) quantum numbers in (e, l, j)-order.
    pub fn states(&self) -> Vec<Npj> {
        Ho3dIter::default()
            .take_while(|nlj| nlj.shell() <= self.e_max())
            .filter(|&nlj| self.contains(nlj))
            .map(Npj::from)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Nucleons {
    pub all: Ho3dModTrunc,
    pub occ: Ho3dModTrunc,
}

impl Nucleons {
    pub fn e_max(&self) -> i32 {
        self.all.e_max()
    }

    pub fn states(&self) -> Vec<Npj> {
        self.all.states()
    }

    pub fn part_states(&self) -> Vec<PartState<Occ, Npj>> {
        let mut finder = FnvHashMap::default();
        let mut states: Vec<_> = self.states()
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                finder.insert(p, i);
                PartState { x: Occ::A, p }
            })
            .collect();
        for p in self.occ.states() {
            let &i = finder.get(&p).expect("occ state not in all");
            states[i].x = Occ::I;
        }
        states
    }

    pub fn orbs(&self) -> JOrbBasis<Parity, i32> {
        self.part_states()
            .into_iter()
            .map(|xp| xp.map_p(|p| p.to_j_chan_state()))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimpleNucleus<'a> {
    /// Maximum index of all available shells.
    pub e_max: i32,
    /// Maximum index of filled neutron shell.
    pub e_fermi_n: i32,
    /// Maximum index of filled proton shell.
    pub e_fermi_p: i32,
    /// Additional occupied orbitals included or excluded, specified using
    /// spectroscopic notation (see `FromStr` implementation for `Npjw`) with
    /// a prefix `+` indicating inclusion and `-` indicating exclusion.
    ///
    /// Example: `"-0s1/2 -0p1/2 +1s1/2"`.
    pub orbs: &'a str,
}

impl<'a> SimpleNucleus<'a> {
    pub fn to_nucleus(self) -> Result<Nucleus, Box<Error + Send + Sync>> {
        let mut neutrons_occ = Ho3dModTrunc::from(Ho3dTrunc {
            e_max: self.e_fermi_n,
            .. Default::default()
        });
        let mut protons_occ = Ho3dModTrunc::from(Ho3dTrunc {
            e_max: self.e_fermi_p,
            .. Default::default()
        });
        for orb in self.orbs.split_whitespace() {
            if orb.starts_with("+") {
                let npjw: Npjw = orb.split_at(1).1.parse()?;
                let occ = if npjw.w < Half(0) {
                    &mut neutrons_occ
                } else {
                    &mut protons_occ
                };
                let nlj = npjw.into();
                if occ.contains(nlj) {
                    Err(format!("already included: {}", npjw))?;
                }
                occ.incl.insert(nlj);
            } else if orb.starts_with("-") {
                let npjw: Npjw = orb.split_at(1).1.parse()?;
                let occ = if npjw.w < Half(0) {
                    &mut neutrons_occ
                } else {
                    &mut protons_occ
                };
                let nlj = npjw.into();
                if !occ.contains(nlj) {
                    Err(format!("already excluded: {}", npjw))?;
                }
                occ.excl.insert(nlj);
            } else {
                Err(format!("must start with '+' or '-': {}", orb))?;
            }
        }
        let all = Ho3dTrunc { e_max: self.e_max, .. Default::default() };
        Ok(Nucleus {
            neutrons: Nucleons { all: all.into(), occ: neutrons_occ },
            protons: Nucleons { all: all.into(), occ: protons_occ },
        })
    }
}

#[derive(Clone, Debug)]
pub struct Nucleus {
    pub neutrons: Nucleons,
    pub protons: Nucleons,
}

impl Nucleus {
    pub fn e_max(&self) -> i32 {
        self.neutrons.e_max().max(self.protons.e_max())
    }

    pub fn states(&self) -> Vec<Npjw> {
        self.neutrons.states().into_iter().map(|npj| {
            npj.and_w(Half(-1))
        }).chain(self.neutrons.states().into_iter().map(|npj| {
            npj.and_w(Half(1))
        })).collect()
    }

    pub fn part_states(&self) -> Vec<PartState<Occ, Npjw>> {
        self.neutrons.part_states().into_iter().map(|xp| {
            xp.map_p(|npj| npj.and_w(Half(-1)))
        }).chain(self.neutrons.part_states().into_iter().map(|xp| {
            xp.map_p(|npj| npj.and_w(Half(1)))
        })).collect()
    }

    pub fn basis(&self) -> JOrbBasis<Pw, i32> {
        self.part_states().into_iter().map(|xp| {
            xp.map_p(From::from)
        }).collect()
    }

    pub fn m_basis(&self) -> JOrbBasis<Pmw, Nj> {
        self.part_states().into_iter().flat_map(|xp| {
            xp.p.j.multiplet().map(move |m| xp.map_p(|p| p.and_m(m).into()))
        }).collect()
    }
}

/// Calculate the kinetic energy matrix element in a 3D harmonic oscillator
/// basis:
///
/// ```text
/// ⟨a| p² / (2 m) |b⟩ / ħ ω
/// ```
///
/// where `m` is the mass used to generate the basis.
pub fn kinetic_ho3d_mat_elem(a: Npj, b: Npj) -> f64 {
    let (_, na, nb) = parity::sort2(a.n, b.n);
    let dn = nb - na;
    if (a.p, a.j) != (b.p, b.j) {
        0.0
    } else if dn == 0 {
        let energy = a.osc_energy();
        0.5 * energy
    } else if dn == 1 {
        let nb = f64::from(nb);
        let l = f64::from(Nlj::from(a).l.0);
        0.5 * (nb * (nb + l + 0.5)).sqrt()
    } else {
        0.0
    }
}

pub fn make_ke_op_j(
    atlas: &JAtlas<Pw, i32>,
    omega: f64,
) -> OpJ100<f64>
{
    let scheme = atlas.scheme();
    let mut h1 = Op::new(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            let npjw1 = Npjw::from(atlas.decode(p).unwrap());
            let npjw2 = Npjw::from(atlas.decode(q).unwrap());
            if npjw1.w == npjw2.w {
                let ke = kinetic_ho3d_mat_elem(npjw1.into(), npjw2.into());
                h1.set(p, q, omega * ke);
            }
        }
    }
    h1
}

pub fn make_ke_op_m(
    atlas: &JAtlas<Pmw, Nj>,
    omega: f64,
) -> OpJ100<f64>
{
    let scheme = atlas.scheme();
    let mut h1 = Op::new(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            let npjmw1 = Npjmw::from(atlas.decode(p).unwrap());
            let npjmw2 = Npjmw::from(atlas.decode(q).unwrap());
            if (npjmw1.m, npjmw1.w) == (npjmw2.m, npjmw2.w) {
                let ke = kinetic_ho3d_mat_elem(npjmw1.into(), npjmw2.into());
                h1.set(p, q, omega * ke);
            }
        }
    }
    h1
}

pub fn make_ho3d_op_j(
    atlas: &JAtlas<Pw, i32>,
    omega: f64,
) -> OpJ100<f64>
{
    let scheme = atlas.scheme();
    let mut h1 = Op::new(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        let npjw = Npjw::from(atlas.decode(p).unwrap());
        let e = omega * Npj::from(npjw).osc_energy();
        h1.set(p, p, e);
    }
    h1
}

pub fn make_ho3d_op_m(
    atlas: &JAtlas<Pmw, Nj>,
    omega: f64,
) -> OpJ100<f64>
{
    let scheme = atlas.scheme();
    let mut h1 = Op::new(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        let npjmw = Npjmw::from(atlas.decode(p).unwrap());
        let e = omega * Npj::from(npjmw).osc_energy();
        h1.set(p, p, e);
    }
    h1
}

/// Load two-body matrix elements from the given hash table.  Note that this
/// function only uses canonicalized keys.
pub fn make_v_op_j(
    atlas: &JAtlas<Pw, i32>,
    two_body_mat_elems: &FnvHashMap<JNpjwKey, f64>,
) -> OpJ200<f64>
{
    let scheme = atlas.scheme();
    let mut h2 = Op::new(scheme.clone());
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        let p = Npjw::from(atlas.decode(p).unwrap());
        let q = Npjw::from(atlas.decode(q).unwrap());
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            let r = Npjw::from(atlas.decode(r).unwrap());
            let s = Npjw::from(atlas.decode(s).unwrap());
            let (sign, _, key) = JNpjwKey {
                j12: pq.j(),
                npjw1: p,
                npjw2: q,
                npjw3: r,
                npjw4: s,
            }.canonicalize();
            h2.add(pq, rs,
                   *two_body_mat_elems.get(&key)
                   .unwrap_or_else(|| {
                       panic!("matrix element not found: {}", key)
                   })
                   * sign);
        }
    }
    h2
}

// (this could be made more general / not specific to nuclei)
pub fn op1_j_to_m(
    j_atlas: &JAtlas<Pw, i32>,
    m_atlas: &JAtlas<Pmw, Nj>,
    a1: &OpJ100<f64>,
) -> OpJ100<f64>
{
    let m_scheme = m_atlas.scheme();
    let unveil = |pm| {
        let npjmw = Npjmw::from(m_atlas.decode(pm).unwrap());
        let pj = j_atlas.encode(&Npjw::from(npjmw).into()).unwrap();
        (npjmw.m, pj)
    };
    let mut b1 = Op::new(m_scheme.clone());
    for pm in m_scheme.states_10(&occ::ALL1) {
        let (_, pj) = unveil(pm);
        for qm in pm.costates_10(&occ::ALL1) {
            let (_, qj) = unveil(qm);
            b1.add(pm, qm, a1.at(pj, qj));
        }
    }
    b1
}

// (this could be made more general / not specific to nuclei)
pub fn op2_j_to_m(
    j_atlas: &JAtlas<Pw, i32>,
    m_atlas: &JAtlas<Pmw, Nj>,
    a2: &OpJ200<f64>,
) -> OpJ200<f64>
{
    let m_scheme = m_atlas.scheme();
    let unveil = |pm| {
        let npjmw = Npjmw::from(m_atlas.decode(pm).unwrap());
        let pj = j_atlas.encode(&Npjw::from(npjmw).into()).unwrap();
        (npjmw.m, pj)
    };
    let mut b2 = Op::new(m_scheme.clone());
    let mut w3jm_ctx = Wigner3jmCtx::default();
    for pqm in m_scheme.states_20(&occ::ALL2) {
        let (pm, qm) = pqm.split_to_10_10();
        let (mp, pj) = unveil(pm);
        let (mq, qj) = unveil(qm);
        let mpq = mp + mq;
        let jp = pj.j();
        let jq = qj.j();
        for rsm in pqm.costates_20(&occ::ALL2) {
            let (rm, sm) = rsm.split_to_10_10();
            let (mr, rj) = unveil(rm);
            let (ms, sj) = unveil(sm);
            let jr = rj.j();
            let js = sj.j();
            debug_assert_eq!(mpq, mr + ms);
            b2.add(pqm, rsm, Half::tri_range_2((jp, jq), (jr, js)).map(|jpq| {
                // handle forbidden states
                let pqj = match pj.combine_with_10(qj, jpq) {
                    None => return 0.0,
                    Some(x) => x,
                };
                let rsj = match rj.combine_with_10(sj, jpq) {
                    None => return 0.0,
                    Some(x) => x,
                };
                a2.at(pqj, rsj)
                    * w3jm_ctx.cg(ClebschGordan {
                        tj1: jp.twice(),
                        tj2: jq.twice(),
                        tj12: jpq.twice(),
                        tm1: mp.twice(),
                        tm2: mq.twice(),
                        tm12: mpq.twice(),
                    })
                    * w3jm_ctx.cg(ClebschGordan {
                        tj1: jr.twice(),
                        tj2: js.twice(),
                        tj12: jpq.twice(),
                        tm1: mr.twice(),
                        tm2: ms.twice(),
                        tm12: mpq.twice(),
                    })
            }).sum());
        }
    }
    b2
}

#[cfg(test)]
mod tests {
    use num::range_step_inclusive;
    use super::super::half::Half;
    use super::{Ho3dIter, Nlj, OrbAng};

    #[test]
    fn test_ho3d_iter() {
        let e_max = 100;
        let nljs1: Vec<_> =
            Ho3dIter::default().take_while(|x| x.shell() <= e_max).collect();
        let mut nljs2 = Vec::default();
        for e in 0 .. e_max + 1 {
            for l in range_step_inclusive(e % 2, e, 2) {
                let n = (e - l) / 2;
                for j in Half::tri_range(l.into(), Half(1)) {
                    nljs2.push(Nlj { n, l: OrbAng(l), j });
                }
            }
        }
        assert_eq!(nljs1, nljs2);
    }
}
