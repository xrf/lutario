//! Darmstadt ME2J matrix element format.

use std::io;
use std::path::Path;
use fnv::FnvHashMap;
use num::range_inclusive;
use super::super::ang_mom::{Coupled2HalfSpinsBlock, Uncoupled2HalfSpinsBlock};
use super::super::half::Half;
use super::{Ho3dTrunc, JNpjwKey, JtwNpjKey, Npj, Pj, Pjtw};

#[derive(Clone, Copy, Debug)]
pub struct Me2j<'a> {
    pub npjs: &'a [Npj],
    pub e12_max: i32,
}

impl<'a> Me2j<'a> {
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
                        for j12 in Half::tri_range_2(
                            (npj1.j, npj2.j),
                            (npj3.j, npj4.j),
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

pub fn load_me2j_j(
    elems: &mut Iterator<Item = f64>,
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
    }.foreach_isospin_block(|pj12, npj1, npj2, npj3, npj4| {
        let z00 = elems.next().ok_or_else(unexpected_eof)?;
        let m11 = elems.next().ok_or_else(unexpected_eof)?;
        let z10 = elems.next().ok_or_else(unexpected_eof)?;
        let p11 = elems.next().ok_or_else(unexpected_eof)?;
        if npj1.shell() > e_max {
            // npj1 is the slowest index, so if it exceeds the
            // shell index we are completely done here
            return Err(Ok(()));
        }
        if
            npj2.shell() > e_max
            || npj3.shell() > e_max
            || npj4.shell() > e_max
        {
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
            if
                (npjw1 == npjw2 && f12.unwrap() % 2 == 0)
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
            }.canonicalize();
            map.insert(key, sign * value);
        }
        Ok(())
    }).or_else(|x| x)?;
    Ok(map)
}

pub fn load_me2j_jt(
    elems: &mut Iterator<Item = f64>,
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
    }.foreach_elem(|pjtw12, npj1, npj2, npj3, npj4| {
        let value = elems.next().ok_or(Err(io::Error::new(
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
            } else if
                npj2.shell() > e_max
                || npj3.shell() > e_max
                || npj4.shell() > e_max
            {
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
            }.canonicalize();
            map.insert(key, sign * value);
        }
        Ok(())
    }).or_else(|x| x)?;
    Ok(map)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Me2jLoader<'a> {
    /// The file extension of the path is important: it determines the type of
    /// compression (if any) and whether the binary or text format is used.
    pub path: &'a Path,
    pub e_max: i32,
    pub n_max: i32,
    pub l_max: i32,
    pub e12_max: i32,
}

/// By default, `path` is set to `"/dev/null"`, `e_max` is set to `-1`, and
/// everything else is set to the maximum integer value.
impl<'a> Default for Me2jLoader<'a> {
    fn default() -> Self {
        Self {
            path: "/dev/null".as_ref(),
            e_max: -1,
            n_max: i32::max_value(),
            l_max: i32::max_value(),
            e12_max: i32::max_value(),
        }
    }
}

impl<'a> Me2jLoader<'a> {
    pub fn load(
        self,
        target_e_max: i32,
    ) -> io::Result<FnvHashMap<JNpjwKey, f64>>
    {
        use byteorder::LittleEndian;
        use super::super::io as lio;

        let trunc = Ho3dTrunc {
            e_max: self.e_max,
            n_max: self.n_max,
            l_max: self.l_max,
        };
        if trunc.is_empty() {
            return Ok(Default::default())
        }
        let (subpath, file) = lio::open_compressed(self.path)?;
        let ext = subpath.extension().unwrap_or("".as_ref());
        let mut elems: Box<Iterator<Item = _>> = match ext.to_str() {
            Some("dat") => {
                Box::new(lio::BinArrayParser::<f32, LittleEndian, _>::new(file)
                         .map(|x| {
                             // don't invert this or you'll change the numbers!
                             let precision = 1e7;
                             (x as f64 * precision).round() / precision
                         }))
            }
            Some("me2j") => {
                Box::new(lio::MapleTableParser::new(io::BufReader::new(file)))
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown format: .{}", ext.to_string_lossy()),
                ))
            },
        };
        let table = trunc.states();
        load_me2j_j(&mut elems, &table, self.e12_max, target_e_max)
    }
}
