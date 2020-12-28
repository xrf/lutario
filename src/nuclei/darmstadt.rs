//! Darmstadt ME2J matrix element format.

use super::super::ang_mom::{Coupled2HalfSpinsBlock, Uncoupled2HalfSpinsBlock};
use super::super::half::Half;
use super::super::io as lio;
use super::{Ho3dTrunc, JNpjwKey, JtwNpjKey, Npj, Pj, Pjtw};
use fnv::FnvHashMap;
use num::range_inclusive;
use regex::Regex;
use std::fs::File;
use std::path::Path;
use std::{fmt, io};

/// This factor is used to adjust ME2J matrix elements for the pairwise dot
/// product of momentum (`tpp` elements).
///
/// In the files, they are stored in this strange convention:
///
/// ```text
/// −(a / fm)² ⟨…| p² / (2 m MeV) |…⟩
/// ```
///
/// Therefore, given `ω / (MeV ħ⁻¹)`, one must compute `(a / fm)⁻²`, where
/// `a` is the characteristic length of the harmonic oscillator:
///
/// ```text
/// a = ħ / m ω
/// ```
///
/// This can be achieved simply by multiplying `ω / (MeV ħ⁻¹)` by this
/// constant.
//
// this constant is from Heiko's IM-SRG code (lib/GLO_Base.h: INVM);
// we use their exact value for reproducibility reasons.
// INVM = ħ / (2 m MeV fm²) ≈ 20.7355
pub const TPP_FACTOR: f64 = 1.0 / (2.0 * 20.7355285386);

#[derive(Clone, Copy, Debug)]
pub struct Me2j<'a> {
    pub npjs: &'a [Npj],
    pub e12_max: i32,
}

impl<'a> Me2j<'a> {
    pub fn foreach_isospin_block<F, E>(self, mut f: F) -> Result<(), E>
    where
        F: FnMut(Pj, Npj, Npj, Npj, Npj) -> Result<(), E>,
    {
        let e12_max = self.e12_max;
        let npjs = self.npjs;
        for i1 in 0..npjs.len() {
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
                        for j12 in Half::tri_range_2((npj1.j, npj2.j), (npj3.j, npj4.j)) {
                            f(Pj { p: p12, j: j12 }, npj1, npj2, npj3, npj4)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn foreach_elem<F, E>(self, mut f: F) -> Result<(), E>
    where
        F: FnMut(Pjtw, Npj, Npj, Npj, Npj) -> Result<(), E>,
    {
        self.foreach_isospin_block(|pj12, npj1, npj2, npj3, npj4| {
            for t12 in Half::tri_range(Half(1), Half(1)) {
                for w12 in t12.multiplet() {
                    f(
                        Pjtw {
                            p: pj12.p,
                            j: pj12.j,
                            t: t12,
                            w: w12,
                        },
                        npj1,
                        npj2,
                        npj3,
                        npj4,
                    )?;
                }
            }
            Ok(())
        })
    }
}

pub fn load_me2j_j(
    elems: &mut dyn Iterator<Item = f64>,
    table: &[Npj],
    e12_max: i32,
    e_max: i32,
) -> io::Result<FnvHashMap<JNpjwKey, f64>> {
    fn unexpected_eof() -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF",
        ))
    }

    if let Some(npj) = table.last() {
        assert!(npj.shell() >= e_max, "table is inadequate for this e_max");
    }
    let mut map = FnvHashMap::default();
    Me2j {
        npjs: &table,
        e12_max,
    }
    .foreach_isospin_block(|pj12, npj1, npj2, npj3, npj4| {
        let z00 = elems.next().ok_or_else(unexpected_eof)?;
        let m11 = elems.next().ok_or_else(unexpected_eof)?;
        let z10 = elems.next().ok_or_else(unexpected_eof)?;
        let p11 = elems.next().ok_or_else(unexpected_eof)?;
        if npj1.shell() > e_max {
            // npj1 is the slowest index, so if it exceeds the
            // shell index we are completely done here
            return Err(Ok(()));
        }
        if npj2.shell() > e_max || npj3.shell() > e_max || npj4.shell() > e_max {
            return Ok(());
        }
        let coupled = Coupled2HalfSpinsBlock { z00, z10 };
        let uncoupled = Uncoupled2HalfSpinsBlock::from(coupled);
        for &(w12, w1, w3, value) in &[
            (Half(0), Half(-1), Half(-1), uncoupled.mpmp),
            (Half(0), Half(-1), Half(1), uncoupled.mppm),
            (Half(0), Half(1), Half(-1), uncoupled.pmmp),
            (Half(0), Half(1), Half(1), uncoupled.pmpm),
            (Half(-2), Half(-1), Half(-1), m11),
            (Half(2), Half(1), Half(1), p11),
        ] {
            let npjw1 = npj1.and_w(w1);
            let npjw2 = npj2.and_w(w12 - w1);
            let npjw3 = npj3.and_w(w3);
            let npjw4 = npj4.and_w(w12 - w3);
            let f12 = pj12.j.abs_diff(npjw1.j + npjw2.j);
            let f34 = pj12.j.abs_diff(npjw3.j + npjw4.j);
            if (npjw1 == npjw2 && f12.unwrap() % 2 == 0)
                || (npjw3 == npjw4 && f34.unwrap() % 2 == 0)
            {
                if value != 0.0 {
                    return Err(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "matrix elements of forbidden states must be zero",
                    )));
                }
                continue;
            }
            let (sign, _, key) = JNpjwKey {
                j12: pj12.j,
                npjw1,
                npjw2,
                npjw3,
                npjw4,
            }
            .canonicalize();
            map.insert(key, sign * value);
        }
        Ok(())
    })
    .or_else(|x| x)?;
    Ok(map)
}

pub fn load_me2j_jt(
    elems: &mut dyn Iterator<Item = f64>,
    table: &[Npj],
    e12_max: i32,
    e_max: i32,
) -> io::Result<FnvHashMap<JtwNpjKey, f64>> {
    if let Some(npj) = table.last() {
        assert!(npj.shell() >= e_max, "table is inadequate for this e_max");
    }
    let mut map = FnvHashMap::default();
    Me2j {
        npjs: &table,
        e12_max,
    }
    .foreach_elem(|pjtw12, npj1, npj2, npj3, npj4| {
        let value = elems.next().ok_or(Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF",
        )))?;
        let ft12 = pjtw12.t.abs_diff(Half(1) + Half(1));
        let f12 = pjtw12.j.abs_diff(npj1.j + npj2.j) + ft12;
        let f34 = pjtw12.j.abs_diff(npj3.j + npj4.j) + ft12;
        if (npj1 == npj2 && f12.unwrap() % 2 == 0) || (npj3 == npj4 && f34.unwrap() % 2 == 0) {
            if value != 0.0 {
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
            } else if npj2.shell() > e_max || npj3.shell() > e_max || npj4.shell() > e_max {
                return Ok(());
            }
            let (sign, _, key) = JtwNpjKey {
                j12: pjtw12.j,
                t12: pjtw12.t,
                w12: pjtw12.w,
                npj1,
                npj2,
                npj3,
                npj4,
            }
            .canonicalize();
            map.insert(key, sign * value);
        }
        Ok(())
    })
    .or_else(|x| x)?;
    Ok(map)
}

/// ME2J loader that tries to guess parameters based on filename unless
/// explicitly overridden.
#[derive(Clone, Copy, Debug)]
pub struct Me2jGuessLoader<'a> {
    pub path: &'a Path,
    pub compression: Option<lio::Compression>,
    pub binary: Option<bool>,
    pub trunc: Option<Ho3dTrunc>,
    pub e12_max: Option<i32>,
    pub omega: Option<f64>,
}

/// By default, `path` is set to `"/dev/null"` and everything else is set to
/// their defaults.
impl<'a> Default for Me2jGuessLoader<'a> {
    fn default() -> Self {
        Self {
            path: "/dev/null".as_ref(),
            compression: Default::default(),
            binary: Default::default(),
            trunc: Default::default(),
            e12_max: Default::default(),
            omega: Default::default(),
        }
    }
}

impl<'a> Me2jGuessLoader<'a> {
    pub fn guess(self) -> io::Result<Me2jLoader<'a>> {
        // guess compression (note: even if we don't guess, we need to chop
        // off the compression file extension)
        let (path, guessed_compression) = lio::guess_compression(self.path)?;
        let compression = self
            .compression
            .or(guessed_compression)
            .ok_or_else(|| lio::invalid_data("unknown compression extension"))?;

        // guess binary or text
        let binary = match self.binary {
            Some(x) => x,
            None => {
                let (path, ext2) = lio::split_extension(path)?;
                let (_, ext1) = lio::split_extension(path.as_ref())?;
                match (ext1, ext2) {
                    (".me2j", ".dat") => true,
                    (_, ".me2j") => false,
                    (_, "") => Err(lio::invalid_data(format!(
                        "missing format extension: {}",
                        path
                    )))?,
                    _ => Err(lio::invalid_data(format!(
                        "unknown format extension: {}",
                        ext2
                    )))?,
                }
            }
        };

        // guess parameters
        let name = path
            .to_str()
            .ok_or_else(|| lio::invalid_data("path is not UTF-8"))?;
        let name = name.replace('_', " ");
        let trunc = match self.trunc {
            Some(x) => x,
            None => {
                let e_max = re!(r"\beMax(\d+)\b")
                    .captures(&name)
                    .ok_or_else(|| lio::invalid_data("can't find eMax"))?
                    .get(1)
                    .unwrap()
                    .as_str()
                    .parse()
                    .map_err(|_| lio::invalid_data("can't parse eMax"))?;
                let n_max = match re!(r"\bnMax(\d+)\b").captures(&name) {
                    None => i32::max_value(),
                    Some(m) => m
                        .get(1)
                        .unwrap()
                        .as_str()
                        .parse()
                        .map_err(|_| lio::invalid_data("can't parse nMax"))?,
                };
                let l_max = match re!(r"\blMax(\d+)\b").captures(&name) {
                    None => i32::max_value(),
                    Some(m) => m
                        .get(1)
                        .unwrap()
                        .as_str()
                        .parse()
                        .map_err(|_| lio::invalid_data("can't parse lMax"))?,
                };
                Ho3dTrunc {
                    e_max,
                    n_max,
                    l_max,
                }
            }
        };
        let e12_max = match self.e12_max {
            Some(x) => x,
            None => match re!(r"\bEMax(\d+)\b").captures(&name) {
                None => i32::max_value(),
                Some(m) => m
                    .get(1)
                    .unwrap()
                    .as_str()
                    .parse()
                    .map_err(|_| lio::invalid_data("can't parse EMax"))?,
            },
        };
        let omega = match self.omega {
            Some(x) => x,
            None => re!(r"\bhwHO(\d+)\b")
                .captures(&name)
                .ok_or_else(|| lio::invalid_data("can't find hwHO"))?
                .get(1)
                .unwrap()
                .as_str()
                .parse()
                .map_err(|_| lio::invalid_data("can't parse hwHO"))?,
        };

        Ok(Me2jLoader {
            path: self.path,
            compression,
            binary,
            trunc,
            e12_max,
            omega,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Me2jLoader<'a> {
    pub path: &'a Path,
    pub compression: lio::Compression,
    pub binary: bool,
    pub trunc: Ho3dTrunc,
    pub e12_max: i32,
    pub omega: f64,
}

impl<'a> fmt::Display for Me2jLoader<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{path: {}", self.path.display())?;
        write!(f, ", compression: {:?}", self.compression)?;
        write!(f, ", binary: {}", self.binary)?;
        write!(f, ", trunc: {}", self.trunc)?;
        if self.e12_max != i32::max_value() {
            writeln!(f, ", e12_max: {}", self.e12_max)?;
        }
        write!(f, ", omega: {}}}", self.omega)
    }
}

impl<'a> Me2jLoader<'a> {
    pub fn load(self, target_e_max: i32) -> io::Result<(f64, FnvHashMap<JNpjwKey, f64>)> {
        use byteorder::LittleEndian;
        let file = File::open(self.path)?;
        let file = lio::decode_compressed(file, self.compression);
        let mut elems: Box<dyn Iterator<Item = _>> = if self.binary {
            Box::new(
                lio::BinArrayParser::<f32, LittleEndian, _>::new(file).map(|x| {
                    // don't invert this or you'll change the numbers!
                    let precision = 1e7;
                    (x as f64 * precision).round() / precision
                }),
            )
        } else {
            Box::new(lio::MapleTableParser::new(io::BufReader::new(file)))
        };
        let table = self.trunc.states();
        let me = load_me2j_j(&mut elems, &table, self.e12_max, target_e_max)?;
        Ok((self.omega, me))
    }
}
