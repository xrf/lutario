//! Nuclei systems.
//!
//! Here we use the particle physics convention of proton = +½, neutron = −½.
use std::{fmt, io, str};
use std::cmp::min;
use std::ops::{Add, Sub};
use std::path::Path;
use fnv::FnvHashMap;
use num::{Zero, range_inclusive, range_step_inclusive};
use wigner_symbols::ClebschGordan;
use super::basis::{occ, ChanState, Occ, PartState};
use super::half::Half;
use super::j_scheme::{BasisJ10, BasisJ20, JAtlas, JChan, OpJ100, OpJ200};
use super::matrix::Mat;
use super::op::Op;
use super::parity::{self, Parity};
use super::utils;

/// Orbital angular momentum (l)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbAng(pub i32);

/// Display the spectroscopic label using the base-20 alphabet
/// "spdfghiklmnoqrtuvwxyz".
impl fmt::Display for OrbAng {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const ALPHABET: &[u8] = b"spdfghiklmnoqrtuvwxyz";
        let mut l = self.0 as usize;
        let mut s = Vec::default();
        loop {
            let r = l % ALPHABET.len();
            if r == 0 {
                break;
            }
            l /= ALPHABET.len();
            s.push(ALPHABET[r]);
        }
        s.reverse();
        if s.is_empty() {
            s.push(ALPHABET[0]);
        }
        write!(f, "{}", str::from_utf8(&s).unwrap())
    }
}

/// Parity and isospin
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pw {
    pub p: Parity,
    pub w: Half<i32>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pmw {
    pub p: Parity,
    pub m: Half<i32>,
    pub w: Half<i32>,
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

/// Principal quantum number, parity, and total angular momentum magnitude.
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
        write!(f, "{}{}{}", self.n, self.orb_ang(), self.j)
    }
}

impl Npj {
    /// Shell index.
    pub fn shell(self) -> i32 {
        2 * self.n + self.orb_ang().0
    }

    /// Returns harmonic oscillator energy in natural units.
    pub fn osc_energy(self) -> f64 {
        1.5 + self.shell() as f64
    }

    /// Orbital angular momentum.
    pub fn orb_ang(self) -> OrbAng {
        debug_assert!(self.j.try_get().is_err());
        let a = self.j.twice() / 2;     // intentional integer division
        OrbAng(a + (a + i32::from(self.p)) % 2)
    }
}

/// Principal quantum number, parity, total angular momentum magnitude, and
/// isospin projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl From<ChanState<JChan<Pw>, i32>> for Npjw {
    fn from(s: ChanState<JChan<Pw>, i32>) -> Self {
        Self { n: s.u, p: s.l.k.p, j: s.l.j, w: s.l.k.w }
    }
}

impl From<Npjw> for ChanState<JChan<Pw>, i32> {
    fn from(s: Npjw) -> Self {
        Self { l: s.into(), u: s.n }
    }
}

impl From<Npjw> for JChan<Pw> {
    fn from(s: Npjw) -> Self {
        Self { j: s.j, k: s.into() }
    }
}

impl From<Npjw> for Npj {
    fn from(s: Npjw) -> Self {
        Self { n: s.n, p: s.p, j: s.j }
    }
}

impl From<Npjw> for Pw {
    fn from(s: Npjw) -> Self {
        Self { p: s.p, w: s.w }
    }
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

impl From<Npjmw> for ChanState<JChan<Pmw>, Nj> {
    fn from(s: Npjmw) -> Self {
        Self { l: Pmw::from(s).into(), u: s.into() }
    }
}

impl From<Npjmw> for Nj {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, j: s.j }
    }
}

impl From<Npjmw> for Npjw {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, p: s.p, j: s.j, w: s.w }
    }
}

impl From<Npjmw> for Pmw {
    fn from(s: Npjmw) -> Self {
        Self { p: s.p, m: s.m, w: s.w }
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

impl From<Npjmw> for Npj {
    fn from(s: Npjmw) -> Self {
        Self { n: s.n, p: s.p, j: s.j }
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nj {
    pub n: i32,
    pub j: Half<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Pj {
    pub p: Parity,
    pub j: Half<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct NucleonBasisSpec {
    /// Maximum shell index
    pub e_max: i32,
    /// Maximum principal quantum number
    pub n_max: i32,
    /// Maximum orbital angular momentum magnitude
    pub l_max: i32,
}

impl NucleonBasisSpec {
    pub fn with_e_max(e_max: i32) -> Self {
        Self {
            e_max,
            n_max: i32::max_value(),
            l_max: i32::max_value(),
        }
    }

    /// Obtain a sequence of (n, π, j) quantum numbers in (e, l, j)-order.
    pub fn npj_states(self) -> Vec<Npj> {
        let mut orbitals = Vec::default();
        for e in 0 .. self.e_max + 1 {
            for l in range_step_inclusive(e % 2, min(e, self.l_max), 2) {
                let n = (e - l) / 2;
                if n > self.n_max {
                    continue;
                }
                let p = Parity::of(l);
                for j in Half::tri_range(l.into(), Half(1)) {
                    orbitals.push(Npj { n, p, j });
                }
            }
        }
        orbitals
    }

    pub fn npjw_states(self, w: Half<i32>) -> Vec<Npjw> {
        let mut orbitals = Vec::default();
        for Npj { n, p, j } in self.npj_states() {
            orbitals.push(Npjw { n, p, j, w });
        }
        orbitals
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Nucleus {
    pub neutron_basis_spec: NucleonBasisSpec,
    pub proton_basis_spec: NucleonBasisSpec,
    pub e_fermi_neutron: i32,
    pub e_fermi_proton: i32,
}

impl Nucleus {
    pub fn npjw_orbs(self) -> Vec<Npjw> {
        let mut states = Vec::default();
        let mut ns = self.neutron_basis_spec.npjw_states(Half(-1)).into_iter();
        let mut ps = self.proton_basis_spec.npjw_states(Half(1)).into_iter();
        // interleave the neutron and proton states
        loop {
            let mut found = false;
            if let Some(n) = ns.next() {
                states.push(n);
                found = true;
            }
            if let Some(p) = ps.next() {
                states.push(p);
                found = true;
            }
            if !found {
                break;
            }
        }
        states
    }

    pub fn jpwn_orbs(self) -> Vec<PartState<Occ, ChanState<JChan<Pw>, i32>>> {
        self.npjw_orbs().into_iter().map(|npjw| {
            PartState {
                x: (Npj::from(npjw).shell() >= self.e_fermi(npjw.w)).into(),
                p: npjw.into(),
            }
        }).collect()
    }

    pub fn pmwnj_orbs(self) -> Vec<PartState<Occ, ChanState<JChan<Pmw>, Nj>>> {
        self.npjw_orbs().into_iter().flat_map(|npjw| {
            npjw.j.multiplet().map(move |m| {
                PartState {
                    x: (Npj::from(npjw).shell() >= self.e_fermi(npjw.w)).into(),
                    p: npjw.and_m(m).into(),
                }
            })
        }).collect()
    }

    pub fn e_fermi(self, w: Half<i32>) -> i32 {
        if w < Half(0) {
            self.e_fermi_neutron
        } else {
            self.e_fermi_proton
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DarmstadtMe2j<'a> {
    pub npjs: &'a [Npj],
    pub e12_max: i32,
}

impl<'a> DarmstadtMe2j<'a> {
    pub fn foreach_isospin_block<F, E>(self, mut f: F) -> Result<(), E>
        where F: FnMut(Pj, Npj, Npj, Npj, Npj) -> Result<(), E>,
    {
        let e12_max = self.e12_max;
        let npjs = self.npjs;
        for i1 in 0 .. npjs.len() {
            let npj1 = self.npjs[i1];
            for i2 in range_inclusive(0, i1) {
                let npj2 = self.npjs[i2];
                if npj1.shell() + npj2.shell() > e12_max {
                    break;
                }
                for i3 in range_inclusive(0, i1) {
                    let npj3 = npjs[i3];
                    let i4_max = if i3 == i1 { i2 } else { i3 };
                    for i4 in range_inclusive(0, i4_max) {
                        let npj4 = npjs[i4];
                        if npj3.shell() + npj4.shell() > e12_max {
                            break;
                        }
                        let p12 = npj1.p + npj2.p;
                        if p12 != npj3.p + npj4.p {
                            continue;
                        }
                        for j12 in utils::intersect_range_inclusive(
                            Half::tri_range(npj1.j, npj2.j),
                            Half::tri_range(npj3.j, npj4.j),
                        ) {
                            f(Pj { p: p12, j: j12 }, npj1, npj2, npj3, npj4)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn foreach_elem<F, E>(self, mut f: F) -> Result<(), E>
        where F: FnMut(Pjtw, Npj, Npj, Npj, Npj) -> Result<(), E>,
    {
        self.foreach_isospin_block(|pj12, npj1, npj2, npj3, npj4| {
            for t12 in Half::tri_range(Half(1), Half(1)) {
                for w12 in t12.multiplet() {
                    f(Pjtw {
                        p: pj12.p,
                        j: pj12.j,
                        t: t12,
                        w: w12,
                    }, npj1, npj2, npj3, npj4)?;
                }
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JtwNpj2Pair {
    pub j12: Half<i32>,
    pub t12: Half<i32>,
    pub w12: Half<i32>,
    pub npj1: Npj,
    pub npj2: Npj,
    pub npj3: Npj,
    pub npj4: Npj,
}

impl JtwNpj2Pair {
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
pub struct JNpjw2Pair {
    pub j12: Half<i32>,
    pub npjw1: Npjw,
    pub npjw2: Npjw,
    pub npjw3: Npjw,
    pub npjw4: Npjw,
}

impl fmt::Display for JNpjw2Pair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "J={} {} {} {} {}",
               self.j12, self.npjw1, self.npjw2, self.npjw3, self.npjw4)
    }
}

impl JNpjw2Pair {
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

pub fn load_me2j<'a>(
    elems: &mut Iterator<Item = f64>,
    table: &[Npj],
    e12_max: i32,
    e_max: i32,
) -> io::Result<FnvHashMap<JtwNpj2Pair, f64>> {
    if let Some(npj) = table.last() {
        assert!(npj.shell() >= e_max, "table is too inadequate for this e_max");
    }
    let mut map = FnvHashMap::default();
    DarmstadtMe2j {
        npjs: &table,
        e12_max,
    }.foreach_elem(|pjtw12, npj1, npj2, npj3, npj4| {
        let x = elems.next().ok_or(Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF",
        )))?;
        let ft12 = pjtw12.t.abs_diff(Half(1) + Half(1));
        let f12 = pjtw12.j.abs_diff(npj1.j + npj2.j) + ft12;
        let f34 = pjtw12.j.abs_diff(npj3.j + npj4.j) + ft12;
        if
            (npj1 == npj2 && f12.unwrap() % 2 == 0)
            || (npj3 == npj4 && f34.unwrap() % 2 == 0)
        {
            if x != 0.0 {
                return Err(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "matrix elements of forbidden states must be zero",
                )));
            }
        } else {
            if npj1.shell() > e_max {
                // npj1 is the slowest index, so if it exceeds the
                // shell index we are completely done here
                return Err(Ok(()));
            } else if
                npj2.shell() > e_max
                || npj3.shell() > e_max
                || npj4.shell() > e_max
            {
                return Ok(());
            }
            map.insert(JtwNpj2Pair {
                j12: pjtw12.j,
                t12: pjtw12.t,
                w12: pjtw12.w,
                npj1,
                npj2,
                npj3,
                npj4,
            }, x);
        }
        Ok(())
    }).or_else(|x| x)?;
    Ok(map)
}

#[derive(Clone, Copy, Debug)]
pub struct DoLoadMe2jFile<'a> {
    pub path: &'a Path,
    pub table_basis_spec: NucleonBasisSpec,
    pub table_e12_max: i32,
    pub e_max: i32,
}

impl<'a> DoLoadMe2jFile<'a> {
    pub fn call(self) -> io::Result<FnvHashMap<JtwNpj2Pair, f64>> {
        use byteorder::LittleEndian;
        use super::io as lio;

        let (subpath, file) = lio::open_compressed(self.path)?;
        let ext = subpath.extension().unwrap_or("".as_ref());
        let mut elems: Box<Iterator<Item = _>> = match ext.to_str() {
            Some("dat") => Box::new(
                lio::BinArrayParser::<f32, LittleEndian, _>::new(file)
                    .map(|x| {
                        // don't invert this or you'll change the numbers!
                        let precision = 1e7;
                        (x as f64 * precision).round() / precision
                    })),
            Some("me2j") => Box::new(lio::MapleTableParser::new(
                io::BufReader::new(file),
            )),
            _ => return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown format: .{}", ext.to_string_lossy()),
            )),
        };
        let table = self.table_basis_spec.npj_states();
        load_me2j(
            &mut elems,
            &table,
            self.table_e12_max,
            self.e_max,
        )
    }
}

pub fn clebsch_gordan(cache: &mut FnvHashMap<ClebschGordan, f64>,
                  cg: ClebschGordan) -> f64 {
    *cache.entry(cg).or_insert_with(|| f64::from(cg.value()))
}

pub fn make_ke_op_j<'a>(
    atlas: &'a JAtlas<Pw, i32>,
    omega: f64,
) -> OpJ100<'a, Vec<Mat<f64>>>
{
    let scheme = &atlas.scheme;
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            let npjw1 = Npjw::from(atlas.decode(p).unwrap());
            let npjw2 = Npjw::from(atlas.decode(q).unwrap());
            let (_, na, nb) = parity::sort2(npjw1.n, npjw2.n);
            let dn = nb - na;
            assert_eq!(npjw1.p, npjw2.p);
            assert_eq!(npjw1.j, npjw2.j);
            assert_eq!(npjw1.w, npjw2.w);
            if dn == 0 {
                let e = Npj::from(npjw1).osc_energy();
                h1.set(p, q, 0.5 * omega * (e + 1.5));
            } else if dn == 1 {
                let nb = nb as f64;
                let l = Npj::from(npjw1).orb_ang().0 as f64;
                h1.set(p, q, 0.5 * omega * (nb * (nb + l + 0.5)).sqrt());
            }
        }
    }
    h1
}

pub fn make_ho3d_op_j<'a>(
    atlas: &'a JAtlas<Pw, i32>,
    omega: f64,
) -> OpJ100<'a, Vec<Mat<f64>>>
{
    let scheme = &atlas.scheme;
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    for p in scheme.states_10(&occ::ALL1) {
        let npjw = Npjw::from(atlas.decode(p).unwrap());
        let e = omega * Npj::from(npjw).osc_energy();
        h1.set(p, p, e);
    }
    h1
}

pub fn make_ho3d_op_m<'a>(
    atlas: &'a JAtlas<Pmw, Nj>,
    omega: f64,
) -> OpJ100<'a, Vec<Mat<f64>>>
{
    let scheme = &atlas.scheme;
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    for p in scheme.states_10(&occ::ALL1) {
        let npjmw = Npjmw::from(atlas.decode(p).unwrap());
        let e = omega * Npj::from(npjmw).osc_energy();
        h1.set(p, p, e);
    }
    h1
}

pub fn make_v_op_j<'a>(
    atlas: &'a JAtlas<Pw, i32>,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> OpJ200<'a, Vec<Mat<f64>>>
{
    let scheme = &atlas.scheme;
    let mut h2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        let p = Npjw::from(atlas.decode(p).unwrap());
        let q = Npjw::from(atlas.decode(q).unwrap());
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            let r = Npjw::from(atlas.decode(r).unwrap());
            let s = Npjw::from(atlas.decode(s).unwrap());
            let (sign, _, key) = JNpjw2Pair {
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

pub fn make_v_op_m<'a>(
    atlas: &'a JAtlas<Pmw, Nj>,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> OpJ200<'a, Vec<Mat<f64>>>
{
    let scheme = &atlas.scheme;
    let mut h2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    let mut cg_cache = FnvHashMap::default();
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        let p = Npjmw::from(atlas.decode(p).unwrap());
        let q = Npjmw::from(atlas.decode(q).unwrap());
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            let r = Npjmw::from(atlas.decode(r).unwrap());
            let s = Npjmw::from(atlas.decode(s).unwrap());
            h2.add(pq, rs, utils::intersect_range_inclusive(
                Half::tri_range(p.j, q.j),
                Half::tri_range(r.j, s.j),
            ).map(|j12| {
                let (sign, _, key) = JNpjw2Pair {
                    j12,
                    npjw1: p.into(),
                    npjw2: q.into(),
                    npjw3: r.into(),
                    npjw4: s.into(),
                }.canonicalize();
                if j12.unwrap() % 2 == 1
                    && (key.npjw1 == key.npjw2
                        || key.npjw3 == key.npjw4)
                {
                    return 0.0;
                }
                *two_body_mat_elems.get(&key)
                    .unwrap_or_else(|| {
                        panic!("matrix element not found: {}", key)
                    })
                    * sign
                    * clebsch_gordan(&mut cg_cache, ClebschGordan {
                        tj1: p.j.twice(),
                        tj2: q.j.twice(),
                        tj12: j12.twice(),
                        tm1: p.m.twice(),
                        tm2: q.m.twice(),
                        tm12: (p.m + q.m).twice(),
                    })
                    * clebsch_gordan(&mut cg_cache, ClebschGordan {
                        tj1: r.j.twice(),
                        tj2: s.j.twice(),
                        tj12: j12.twice(),
                        tm1: r.m.twice(),
                        tm2: s.m.twice(),
                        tm12: (r.m + s.m).twice(),
                    })
            }).sum());
        }
    }
    h2
}

pub mod morten_vint {
    use std::io;
    use std::fs::File;
    use std::path::Path;
    use byteorder::{LittleEndian, ReadBytesExt};
    use fnv::FnvHashMap;
    use regex::Regex;
    use super::super::half::Half;
    use super::super::io::{Parser, invalid_data};
    use super::super::parity::Parity;
    use super::{JNpjw2Pair, Npjw};

    pub fn load_sp_table(reader: &mut io::Read) -> io::Result<Vec<Npjw>> {
        let mut p = Parser::new(reader);
        let mut line_num = 1;

        // make sure the file looks sane (might be binary / have long lines)
        p.munch_space()?;
        if !p.match_bytes(b"----> Oscillator parameters, Model space and single-particle data")? {
            return Err(invalid_data("line 1 is invalid"));
        }

        // locate the legend line
        let mut line = String::default();
        loop {
            p.next_line(&mut line, &mut line_num)?;
            if line.is_empty() {
                return Err(invalid_data("can't find legend"));
            }
            if re!(r"^\s*Legend:").is_match(&line) {
                break;
            }
        }
        if !re!(r"^\s*Legend:\s+n\s+l\s+2j\s+tz\s+2n\+l\s+HO-energy\s+evalence\s+particle/hole  inside/outside").is_match(&line) {
            return Err(invalid_data("unsupported legend"));
        }

        // read the table
        let mut table = Vec::default();
        loop {
            p.next_line(&mut line, &mut line_num)?;
            if line.is_empty() {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let captures = match re!(r"^\s*Number:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$").captures(&line) {
                None => return Err(invalid_data(
                    format!("invalid line {}", line_num))),
                Some(x) => x,
            };
            let n = captures[2].parse().map_err(invalid_data)?;
            let l: i32 = captures[3].parse().map_err(invalid_data)?;
            let tj = captures[4].parse().map_err(invalid_data)?;
            let w_nucl: i32 = captures[5].parse().map_err(invalid_data)?;
            let p = Parity::of(l);
            let j = Half(tj);
            let w = Half(-w_nucl);      // convert nuclear → HEP convention
            table.push(Npjw { n, p, j, w });
        }
        Ok(table)
    }

    pub fn load_vint_table(
        reader: &mut io::Read,
        sp_table: &[Npjw],
    ) -> io::Result<FnvHashMap<JNpjw2Pair, f64>>
    {
        let mut p = Parser::new(reader);
        let mut line_num = 1;

        // make sure the file looks sane (might be binary / have long lines)
        p.munch_space()?;
        if !p.match_bytes(b"----> Interaction part")? {
            return Err(invalid_data("line 1 is invalid"));
        }

        // locate the legend line
        let mut line = String::default();
        loop {
            p.next_line(&mut line, &mut line_num)?;
            if line.is_empty() {
                return Err(invalid_data("can't find legend"));
            }
            if re!(r"^\s*Tz\s+").is_match(&line) {
                break;
            }
        }
        if !re!(r"^\s*Tz\s+Par\s+2J\s+a\s+b\s+c\s+d\s+<ab|V|cd>\s+<ab\|Hcom\|cd>\s+<ab\|r_ir_j\|cd>\s+<ab\|p_ip_j\|cd>").is_match(&line) {
            return Err(invalid_data("unsupported legend"));
        }

        // read the table
        let mut table = FnvHashMap::default();
        loop {
            p.next_line(&mut line, &mut line_num)?;
            if line.is_empty() {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let captures = match re!(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$").captures(&line) {
                None => return Err(invalid_data(
                    format!("invalid line {}", line_num))),
                Some(x) => x,
            };
            let tj12 = captures[3].parse().map_err(invalid_data)?;
            let i1: usize = captures[4].parse().map_err(invalid_data)?;
            let i2: usize = captures[5].parse().map_err(invalid_data)?;
            let i3: usize = captures[6].parse().map_err(invalid_data)?;
            let i4: usize = captures[7].parse().map_err(invalid_data)?;
            let value: f64 = captures[8].parse().map_err(invalid_data)?;
            let j12 = Half(tj12);
            let (sign, _, key) = JNpjw2Pair { // hermitian → symmetric
                j12,
                // convert 1-based to 0-based indices
                npjw1: sp_table[i1 - 1],
                npjw2: sp_table[i2 - 1],
                npjw3: sp_table[i3 - 1],
                npjw4: sp_table[i4 - 1],
            }.canonicalize();
            table.insert(key, sign * value);
        }
        Ok(table)
    }

    pub fn load_sp_table_bin(
        reader: &mut io::Read,
    ) -> io::Result<Box<[Npjw]>>
    {
        let mut table = Vec::default();
        loop {
            let n = match reader.read_i32::<LittleEndian>() {
                Err(e) => {
                    if e.kind() != io::ErrorKind::UnexpectedEof {
                        return Err(e);
                    }
                    break;
                }
                Ok(x) => x,
            };
            let p = Parity::of(reader.read_i32::<LittleEndian>()?);
            let j = Half(reader.read_i32::<LittleEndian>()?);
            // convert nuclear -> HEP convention
            let w = Half(-reader.read_i32::<LittleEndian>()?);
            table.push(Npjw { n, p, j, w });
        }
        Ok(table.into_boxed_slice())
    }

    pub fn load_vint_table_bin(
        reader: &mut io::Read,
        sp_table: &[Npjw],
    ) -> io::Result<FnvHashMap<JNpjw2Pair, f64>>
    {
        let mut table = FnvHashMap::default();
        loop {
            let j12 = Half(match reader.read_i32::<LittleEndian>() {
                Err(e) => {
                    if e.kind() != io::ErrorKind::UnexpectedEof {
                        return Err(e);
                    }
                    break;
                }
                Ok(x) => x,
            });
            let i1 = reader.read_u32::<LittleEndian>()? as usize;
            let i2 = reader.read_u32::<LittleEndian>()? as usize;
            let i3 = reader.read_u32::<LittleEndian>()? as usize;
            let i4 = reader.read_u32::<LittleEndian>()? as usize;
            let value = reader.read_f64::<LittleEndian>()?;
            let (sign, _, key) = JNpjw2Pair { // hermitian → symmetric
                j12,
                // convert 1-based to 0-based indices
                npjw1: sp_table[i1 - 1],
                npjw2: sp_table[i2 - 1],
                npjw3: sp_table[i3 - 1],
                npjw4: sp_table[i4 - 1],
            }.canonicalize();
            table.insert(key, sign * value);
        }
        Ok(table)
    }

    #[derive(Clone, Copy, Debug)]
    pub struct LoadTwoBodyMatElems<'a> {
        pub sp_table_path: &'a Path,
        pub vint_table_path: &'a Path,
    }

    impl<'a> LoadTwoBodyMatElems<'a> {
        pub fn call(self) -> io::Result<FnvHashMap<JNpjw2Pair, f64>> {
            let sp_table = load_sp_table(&mut File::open(self.sp_table_path)?)?;
            load_vint_table(&mut File::open(self.vint_table_path)?, &sp_table)
        }
    }
}
