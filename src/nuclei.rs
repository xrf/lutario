//! Nuclei systems.
use std::{fmt, str};
use std::cmp::min;
use std::ops::{Add, Sub};
use std::path::Path;
use num::{Zero, range_inclusive, range_step_inclusive};
use super::{Error, ResultExt};
use super::basis::{Abelian, MatChart, State, State2};
use super::half::Half;
use super::linalg::AdjSym;
use super::parity::Parity;
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
        let mut s = Vec::new();
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
        Pw { p: self.p + other.p, w: self.w + other.w }
    }
}

impl Sub for Pw {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Pw { p: self.p - other.p, w: self.w - other.w }
    }
}

impl Zero for Pw {
    fn zero() -> Self {
        Pw { p: Zero::zero(), w: Zero::zero() }
    }
    fn is_zero(&self) -> bool {
        self.p.is_zero() && self.w.is_zero()
    }
}

impl Abelian for Pw {}

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
        write!(f, "{}{}{}", self.n, self.l(), self.j)
    }
}

impl Npj {
    /// Shell index
    pub fn e(self) -> i32 {
        2 * self.n + self.l().0
    }

    /// Orbital angular momentum
    pub fn l(self) -> OrbAng {
        debug_assert!(self.j.try_get().is_err());
        let a = self.j.twice() / 2; // caution: integer division!
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
    /// Shell index
    pub fn e(self) -> i32 {
        self.npj().e()
    }

    pub fn npj(self) -> Npj {
        Npj { n: self.n, p: self.p, j: self.j }
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

impl Npjmw {
    /// Shell index
    pub fn e(self) -> i32 {
        self.npj().e()
    }

    pub fn npj(self) -> Npj {
        Npj { n: self.n, p: self.p, j: self.j }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NuclBasisSpec {
    /// Maximum shell index
    pub e_max: i32,
    /// Maximum principal quantum number
    pub n_max: i32,
    /// Maximum orbital angular momentum magnitude
    pub l_max: i32,
}

impl NuclBasisSpec {
    pub fn with_e_max(e_max: i32) -> Self {
        Self {
            e_max,
            n_max: i32::max_value(),
            l_max: i32::max_value(),
        }
    }

    /// Obtain a sequence of (n, π, j) quantum numbers in (e, l, j)-order.
    pub fn npj_states(self) -> Vec<Npj> {
        let mut orbitals = Vec::new();
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

    pub fn npjw_states(self) -> Vec<Npjw> {
        let mut orbitals = Vec::new();
        for Npj { n, p, j } in self.npj_states() {
            for w in Half(1).multiplet() {
                orbitals.push(Npjw { n, p, j, w });
            }
        }
        orbitals
    }

    pub fn npjmw_states(self) -> Vec<Npjmw> {
        let mut orbitals = Vec::new();
        for Npjw { n, p, j, w } in self.npjw_states() {
            for m in j.multiplet() {
                orbitals.push(Npjmw { n, p, j, m, w });
            }
        }
        orbitals
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
            if self.w == Half(0) {
                ","
            } else if self.w > Half(0) {
                "+"
            } else {
                ""
            },
            self.w,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pmw {
    pub p: Parity,
    pub m: Half<i32>,
    pub w: Half<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nj {
    pub n: i32,
    pub j: Half<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Nucleus {
    pub basis_spec: NuclBasisSpec,
    pub num_filled: i32,
}

impl Nucleus {
    pub fn channelize_j1(&self, npjw: Npjw) -> State<Pjtw, i32, i32> {
        let Npjw { n, p, j, w } = npjw;
        State {
            chan: Pjtw { p, j, t: Half(1), w },
            part: (npjw.e() >= self.num_filled) as i32,
            aux: n,
        }
    }

    pub fn channelize_m1(&self, npjmw: Npjmw) -> State<Pmw, i32, Nj> {
        let Npjmw { n, p, j, m, w } = npjmw;
        State {
            chan: Pmw { p, m, w },
            part: (npjmw.e() >= self.num_filled) as _,
            aux: Nj { n, j },
        }
    }

    pub fn channelize_j2(&self, (pjtw12, npj1, npj2): (Pjtw, Npj, Npj))
                         -> State<Pjtw, i32, State2<Npj>> {
        let x1 = (npj1.e() >= self.num_filled) as i32;
        let x2 = (npj2.e() >= self.num_filled) as i32;
        State {
            chan: pjtw12,
            part: x1 + x2,
            aux: (npj1, npj2),
        }
    }

    pub fn channelize_m2(&self, (npjmw1, npjmw2): (Npjmw, Npjmw))
                         -> State<Pmw, i32, State2<Npjmw>> {
        let x1 = (npjmw1.e() >= self.num_filled) as i32;
        let x2 = (npjmw2.e() >= self.num_filled) as i32;
        State {
            chan: Pmw {
                p: npjmw1.p + npjmw2.p,
                m: npjmw1.m + npjmw2.m,
                w: npjmw1.w + npjmw2.w,
            },
            part: x1 + x2,
            aux: (npjmw1, npjmw2),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pj {
    pub p: Parity,
    pub j: Half<i32>,
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
                if npj1.e() + npj2.e() > e12_max {
                    break;
                }
                for i3 in range_inclusive(0, i1) {
                    let npj3 = npjs[i3];
                    let i4_max = if i3 == i1 { i2 } else { i3 };
                    for i4 in range_inclusive(0, i4_max) {
                        let npj4 = npjs[i4];
                        if npj3.e() + npj4.e() > e12_max {
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

/// Construct a table of symmetric j-scheme states with isospin.
///
/// Assumes j of orbitals are half-integers.
pub fn jt2_states(orbitals: &[Npj]) -> Vec<(Pjtw, Npj, Npj)> {
    let mut states = Vec::default();
    // loop: ip1 >= ip2
    for (i1, &npj1) in orbitals.iter().enumerate() {
        for (i2, &npj2) in orbitals.iter().enumerate() {
            // Q: um should we use ip or (l, u) or (l, x, u) or something else?
            if i1 < i2 {
                continue;
            }
            let j1 = npj1.j;
            let j2 = npj2.j;
            let p12 = npj1.p + npj2.p;
            for j12 in Half::tri_range(j1, j2) {
                for t12 in Half::tri_range(Half(1), Half(1)) {
                    if i1 == i2 && ((j1 + j2).abs_diff(j12)
                                    + t12).unwrap() % 2 != 0 {
                        // forbidden state
                        continue;
                    }
                    for w12 in t12.multiplet() {
                        states.push((
                            Pjtw { p: p12, j: j12, t: t12, w: w12 },
                            npj1,
                            npj2,
                        ));
                    }
                }
            }
        }
    }
    states
}

quick_error! {
    #[derive(Debug)]
    pub enum LoadError {
        UnexpectedEof {
            description("file ended prematurely")
        }
        NonzeroForbidden {
            description("matrix elements of forbidden states must be zero")
        }
    }
}

pub fn load_me2j<'a>(
    elems: &mut Iterator<Item = f64>,
    table: &[Npj],
    e_max: i32,
    e12_max: i32,
    mc2: &MatChart<Pjtw, i32, i32, State2<Npj>, State2<Npj>>,
    progress: &mut FnMut(usize),
    dest_adj: AdjSym,
    dest: &mut [f64],
) -> Result<(), LoadError> {
    if let Some(npj) = table.last() {
        assert!(npj.e() >= e_max, "table is too inadequate for this e_max");
    }
    let mut n = 0;
    let mut progress_timer = 0;
    progress(n);
    let r = DarmstadtMe2j {
        npjs: &table,
        e12_max,
    }.foreach_elem(|pjtw12, npj1, npj2, npj3, npj4| {
        let x = elems.next().ok_or(Err(LoadError::UnexpectedEof))?;
        let t_phase = pjtw12.t.abs_diff(Half(1) + Half(1));
        let atsy_phase12 = pjtw12.j.abs_diff(npj1.j + npj2.j) + t_phase;
        let atsy_phase34 = pjtw12.j.abs_diff(npj3.j + npj4.j) + t_phase;
        if (npj1 == npj2 && atsy_phase12.unwrap() % 2 == 0)
        || (npj3 == npj4 && atsy_phase34.unwrap() % 2 == 0) {
            if x != 0.0 {
                return Err(Err(LoadError::NonzeroForbidden));
            }
        } else {
            if npj1.e() > e_max {
                // npj1 is the slowest index, so if it exceeds the
                // shell index we are completely done here
                return Err(Ok(()));
            } else if npj2.e() > e_max
                   || npj3.e() > e_max
                   || npj4.e() > e_max {
                return Ok(());
            }
            let l = pjtw12;
            let u1 = (npj1, npj2);
            let u2 = (npj3, npj4);
            let il = mc2.left.encode_chan(&l).expect("invalid channel") as u32;
            // note: left and right ought to be the same
            let iu1 = mc2.left.encode_aux(il, &u1).expect("invalid left state") as u32;
            let iu2 = mc2.left.encode_aux(il, &u2).expect("invalid right state") as u32;
            dest[mc2.layout.offset(il, iu1, iu2)] = x;
            n += 1;
            if iu1 != iu2 {
                dest[mc2.layout.offset(il, iu2, iu1)] = dest_adj.apply(x);
                n += 1;
            }
        }
        if progress_timer == 10000 {
            progress(n);
            progress_timer = 0;
        }
        progress_timer += 1;
        Ok(())
    }).or_else(|x| x);
    progress(n);
    r
}

pub fn do_load_me2j_file(
    path: &Path,
    e_max: i32,
    e12_max: i32,
    mc2: &MatChart<Pjtw, i32, i32, State2<Npj>, State2<Npj>>,
    dest_adj: AdjSym,
    dest: &mut [f64],
) -> Result<(), Error> {
    use std::io::{self, Write};
    use byteorder::LittleEndian;
    use super::io as lio;

    let (subpath, file) = lio::open_compressed(path.as_ref())
        .chain_err(|| "cannot open file")?;
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
        _ => return Err(format!("unknown format: .{}",
                                ext.to_string_lossy()).into()),
    };
    let table = NuclBasisSpec::with_e_max(e_max).npj_states();
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\r");
    let _ = stdout.flush();
    let num_elems = dest.len();
    load_me2j(
        &mut elems,
        &table,
        e_max,
        e12_max,
        mc2,
        &mut |n| {
            let _ = write!(stdout, "\r{:3.0}% ({} / {})",
                           (n * 100) as f64 / num_elems as f64,
                           n, num_elems);
            let _ = stdout.flush();
        },
        dest_adj,
        dest,
    ).chain_err(|| "load error")?;
    let _ = writeln!(stdout, "");
    let _ = stdout.flush();
    Ok(())
}
