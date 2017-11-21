//! Quantum dots.
use std::{fmt, io};
use std::ops::{Add, Sub};
use fnv::FnvHashMap;
use num::{Zero, range_step_inclusive};
use super::basis::{occ, ChanState, Occ, PartState};
use super::half::Half;
use super::j_scheme::{BasisJ10, BasisJ20, JAtlas, JChan, OpJ100, OpJ200};
use super::op::Op;
use super::parity::{self, Parity};
use super::utils::cast;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ls {
    pub l: i32,
    pub s: Half<i32>,
}

impl Add for Ls {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self { l: self.l + other.l, s: self.s + other.s }
    }
}

impl Sub for Ls {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self { l: self.l - other.l, s: self.s - other.s }
    }
}

impl Zero for Ls {
    fn zero() -> Self {
        Self { l: Zero::zero(), s: Zero::zero() }
    }
    fn is_zero(&self) -> bool {
        self.l.is_zero() && self.s.is_zero()
    }
}

/// Principal quantum number, orbital angular momentum projection, and spin
/// projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nls {
    /// Principal quantum number (n)
    pub n: i32,
    /// Orbital angular momentum projection (m<sub>l</sub>)
    pub l: i32,
    /// Spin projection (m<sub>s</sub>)
    pub s: Half<i32>,
}

impl fmt::Display for Nls {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{},{}", self.n, self.l)?;
        match self.s {
            Half(-1) => write!(f, "↓"),
            Half(1) => write!(f, "↑"),
            x => write!(f, "{}", x),
        }
    }
}

impl From<ChanState<JChan<Ls>, i32>> for Nls {
    fn from(this: ChanState<JChan<Ls>, i32>) -> Self {
        Self { n: this.u, l: this.l.k.l, s: this.l.k.s }
    }
}

impl From<Nls> for ChanState<JChan<Ls>, i32> {
    fn from(this: Nls) -> Self {
        let Nls { n, l, s } = this;
        Self {
            l: Ls { l, s }.into(),
            u: n,
        }
    }
}

impl Nls {
    /// Shell index.
    pub fn shell(self) -> i32 {
        2 * self.n + self.l.abs()
    }

    /// Returns harmonic oscillator energy in natural units.
    pub fn osc_energy(self) -> f64 {
        1.0 + self.shell() as f64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Qdot {
    pub num_shells: i32,
    pub num_filled: i32,
}

impl Qdot {
    pub fn orbs(self) -> Vec<Nls> {
        let mut orbitals = Vec::default();
        for k in 0 .. self.num_shells {
            for l in range_step_inclusive(-k, k, 2) {
                let n = (k - l.abs()) / 2;
                for s in Half(1).multiplet() {
                    orbitals.push(Nls { n, l, s });
                }
            }
        }
        orbitals
    }

    pub fn parted_orbs(self) -> Vec<PartState<Occ, ChanState<JChan<Ls>, i32>>> {
        self.orbs().into_iter().map(|nls| {
            PartState {
                x: (nls.shell() >= self.num_filled).into(),
                p: nls.into(),
            }
        }).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nl2Pair {
    pub n1: u8,
    pub l1: i8,
    pub n2: u8,
    pub l2: i8,
    pub n3: u8,
    pub l3: i8,
    pub n4: u8,
    pub l4: i8,
}

impl Nl2Pair {
    pub fn canonicalize(self) -> (bool, Self) {
        let Self { n1, l1, n2, l2, n3, l3, n4, l4 } = self;
        let (f12, (n1, l1), (n2, l2)) = parity::sort2((n1, l1), (n2, l2));
        let (f34, (n3, l3), (n4, l4)) = parity::sort2((n3, l3), (n4, l4));
        let (adj, (n1, l1, n2, l2), (n3, l3, n4, l4)) =
            parity::sort2((n1, l1, n2, l2), (n3, l3, n4, l4));
        let (n3, l3, n4, l4) = if
            (n1, l1) != (n2, l2)
            && f12 + f34 == Parity::Odd
        {
            (n4, l4, n3, l3)
        } else {
            (n3, l3, n4, l4)
        };
        (
            adj == Parity::Odd,
            Self { n1, l1, n2, l2, n3, l3, n4, l4 },
        )
    }
}

pub fn make_ho2d_op<'a>(
    atlas: &'a JAtlas<Ls, i32>,
    omega: f64,
) -> OpJ100<'a, f64>
{
    let scheme = &atlas.scheme;
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    for p in scheme.states_10(&occ::ALL1) {
        let sp = Nls::from(atlas.decode(p).unwrap());
        h1.add(p, p, omega * sp.osc_energy());
    }
    h1
}

pub fn make_v_op<'a>(
    atlas: &'a JAtlas<Ls, i32>,
    table: &FnvHashMap<Nl2Pair, f64>,
    omega: f64,
) -> OpJ200<'a, f64>
{
    let sqrt_omega = omega.sqrt();
    let scheme = &atlas.scheme;
    let mut h2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        let p = Nls::from(atlas.decode(p).unwrap());
        let q = Nls::from(atlas.decode(q).unwrap());
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            let r = Nls::from(atlas.decode(r).unwrap());
            let s = Nls::from(atlas.decode(s).unwrap());
            let mut z = 0.0;
            if p.s == r.s && q.s == s.s {
                let (_, key) = Nl2Pair {
                    n1: cast(p.n),
                    l1: cast(p.l),
                    n2: cast(q.n),
                    l2: cast(q.l),
                    n3: cast(r.n),
                    l3: cast(r.l),
                    n4: cast(s.n),
                    l4: cast(s.l),
                }.canonicalize();
                z += *table.get(&key)
                    .unwrap_or_else(|| {
                        panic!("matrix element not found: {:?}", key)
                    });
            }
            if p.s == s.s && q.s == r.s {
                let (_, key) = Nl2Pair {
                    n1: cast(p.n),
                    l1: cast(p.l),
                    n2: cast(q.n),
                    l2: cast(q.l),
                    n3: cast(s.n),
                    l3: cast(s.l),
                    n4: cast(r.n),
                    l4: cast(r.l),
                }.canonicalize();
                z -= *table.get(&key)
                    .unwrap_or_else(|| {
                        panic!("matrix element not found: {:?}", key)
                    });
            }
            h2.add(pq, rs, sqrt_omega * z);
        }
    }
    h2
}

pub fn read_clh2_bin(reader: &mut io::Read) -> io::Result<FnvHashMap<Nl2Pair, f64>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut map = FnvHashMap::default();
    loop {
        let n1 = match reader.read_u8() {
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
            Ok(x) => x,
        };
        let l1 = reader.read_i8()?;
        let n2 = reader.read_u8()?;
        let l2 = reader.read_i8()?;
        let n3 = reader.read_u8()?;
        let l3 = reader.read_i8()?;
        let n4 = reader.read_u8()?;
        let l4 = reader.read_i8()?;
        let x = reader.read_f64::<LittleEndian>()?;
        let (_, key) =
            Nl2Pair { n1, l1, n2, l2, n3, l3, n4, l4 }.canonicalize();
        map.insert(key, x);
    }
    Ok(map)
}
