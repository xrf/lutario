//! VRenormalize matrix element format used in CENS MBPT.

use std::io;
use std::fs::File;
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};
use fnv::FnvHashMap;
use regex::Regex;
use super::super::half::Half;
use super::super::io::{Parser, invalid_data};
use super::super::parity::Parity;
use super::{JNpjwKey, Npjw};

pub fn load_sp_table(reader: &mut io::Read) -> io::Result<(f64, Vec<Npjw>)> {
    let mut p = Parser::new(reader);
    let mut line_num = 1;

    // make sure the file looks sane (might be binary / have long lines)
    p.munch_space()?;
    if !p.match_bytes(b"----> Oscillator parameters, Model space and single-particle data")? {
        return Err(invalid_data("line 1 is invalid"));
    }

    let mut line = String::default();

    // locate and parse the oscillator energy line
    loop {
        p.next_line(&mut line, &mut line_num)?;
        if line.is_empty() {
            return Err(invalid_data("can't find oscillator energy"));
        }
        if re!(r"^\s*Oscillator length and energy:").is_match(&line) {
            break;
        }
    }
    let omega = re!(r"^\s*Oscillator length and energy:\s*\S+\s*(\S+)")
        .captures(&line)
        .ok_or(invalid_data("cannot parse oscillator energy line"))?
        .get(1).unwrap().as_str().parse()
        .map_err(|_| invalid_data("cannot parse oscillator energy"))?;

    // locate the legend line
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
    Ok((omega, table))
}

pub fn load_vint_table(
    reader: &mut io::Read,
    sp_table: &[Npjw],
) -> io::Result<FnvHashMap<JNpjwKey, f64>>
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
        // unnormalize matrix elements
        let value = value
            * (1.0 + (i1 == i2) as i32 as f64).sqrt()
            * (1.0 + (i3 == i4) as i32 as f64).sqrt();
        let j12 = Half(tj12);
        let (sign, _, key) = JNpjwKey { // hermitian → symmetric
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
) -> io::Result<Vec<Npjw>>
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
    Ok(table)
}

pub fn load_vint_table_bin(
    reader: &mut io::Read,
    sp_table: &[Npjw],
) -> io::Result<FnvHashMap<JNpjwKey, f64>>
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
        // unnormalize matrix elements
        let value = value
            * (1.0 + (i1 == i2) as i32 as f64).sqrt()
            * (1.0 + (i3 == i4) as i32 as f64).sqrt();
        let (sign, _, key) = JNpjwKey { // hermitian → symmetric
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
pub struct VintLoader<'a> {
    pub path: &'a Path,
    pub sp: &'a Path,
}

impl<'a> VintLoader<'a> {
    pub fn load(self) -> io::Result<(f64, FnvHashMap<JNpjwKey, f64>)> {
        let (omega, sp_table) = load_sp_table(&mut File::open(self.sp)?)?;
        let me = load_vint_table(&mut File::open(self.path)?, &sp_table)?;
        Ok((omega, me))
    }
}
