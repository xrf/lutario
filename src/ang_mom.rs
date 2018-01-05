//! Angular momentum coupling.
use std::ops::{Add, Mul, Sub};
use fnv::FnvHashMap;
use num::FromPrimitive;
use wigner_symbols::{ClebschGordan, Wigner3jm, Wigner6j};
use wigner_symbols::regge::{CanonicalRegge3jm, CanonicalRegge6j, Regge3jm};
use super::half::Half;

/// Reflection about the 22.5° axis.
///
/// ```text
/// ⎡√½  √½⎤
/// ⎣√½ −√½⎦
/// ```
#[inline]
pub fn reflect_16th<T>(x: T, y: T) -> (T, T) where
    T: FromPrimitive + Add<Output = T> + Sub<Output = T>
     + Mul<Output = T> + Clone,
{
    let sqrt_2 = T::from_f64(0.5f64.sqrt()).unwrap();
    (
        sqrt_2.clone() * (x.clone() + y.clone()),
        sqrt_2 * (x - y),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Coupled2HalfSpinsBlock {
    /// `j=0 m=0`
    pub z00: f64,
    /// `j=1 m=0`
    pub z10: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Uncoupled2HalfSpinsBlock {
    /// ↓↑ | ↓↑
    pub mpmp: f64,
    /// ↓↑ | ↑↓
    pub mppm: f64,
    /// ↑↓ | ↓↑
    pub pmmp: f64,
    /// ↑↓ | ↑↓
    pub pmpm: f64,
}

impl From<Coupled2HalfSpinsBlock> for Uncoupled2HalfSpinsBlock {
    #[inline]
    fn from(v: Coupled2HalfSpinsBlock) -> Self {
        let same = 0.5 * (v.z10 + v.z00);
        let diff = 0.5 * (v.z10 - v.z00);
        Self {
            mpmp: same,
            mppm: diff,
            pmmp: diff,
            pmpm: same,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Coupled2HalfSpins<T> {
    /// `j=0 m=0`
    pub z00: T,
    /// `j=1 m=−1`
    pub m11: T,
    /// `j=1 m=0`
    pub z10: T,
    /// `j=1 m=+1`
    pub p11: T,
}

impl<T> From<Uncoupled2HalfSpins<T>> for Coupled2HalfSpins<T> where
    T: FromPrimitive + Add<Output = T> + Sub<Output = T>
     + Mul<Output = T> + Clone,
{
    #[inline]
    fn from(x: Uncoupled2HalfSpins<T>) -> Self {
        let (z10, z00) = reflect_16th(x.pm, x.mp);
        Self { z00, m11: x.mm, z10, p11: x.pp }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Uncoupled2HalfSpins<T> {
    /// ↓↓
    pub mm: T,
    /// ↓↑
    pub mp: T,
    /// ↑↓
    pub pm: T,
    /// ↑↑
    pub pp: T,
}

impl<T> From<Coupled2HalfSpins<T>> for Uncoupled2HalfSpins<T> where
    T: FromPrimitive + Add<Output = T> + Sub<Output = T>
     + Mul<Output = T> + Clone,
{
    #[inline]
    fn from(x: Coupled2HalfSpins<T>) -> Self {
        let (pm, mp) = reflect_16th(x.z10, x.z00);
        Self { mm: x.m11, mp, pm, pp: x.p11 }
    }
}

#[derive(Default)]
pub struct Wigner3jmCtx(pub FnvHashMap<CanonicalRegge3jm, f64>);

impl Wigner3jmCtx {
    pub fn cg(&mut self, cg: ClebschGordan) -> f64 {
        let d = cg.tj1 - cg.tj2 + cg.tm12;
        let phase = if d % 2 == 0 { Half(d).phase() } else { 0.0 };
        phase * Half(cg.tj12).weight(1) * self.get(cg.into())
    }

    pub fn get(&mut self, w3jm: Wigner3jm) -> f64 {
        let (regge, phase) = Regge3jm::from(w3jm).canonicalize();
        phase as f64 * *self.0.entry(regge).or_insert_with(|| {
            f64::from(phase * w3jm.value())
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Wigner6jCtx(pub FnvHashMap<CanonicalRegge6j, f64>);

impl Wigner6jCtx {
    pub fn get(&mut self, w6j: Wigner6j) -> f64 {
        let regge = CanonicalRegge6j::from(w6j);
        *self.0.entry(regge).or_insert_with(|| f64::from(w6j.value()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::utils::Toler;
    use super::*;

    #[test]
    fn test_clebsch_gordan() {
        const TOLER: Toler = Toler { relerr: 1e-15, abserr: 1e-15 };
        let cg = ClebschGordan {
            tj1: 3,
            tj2: 1,
            tj12: 2,
            tm1: -1,
            tm2: -1,
            tm12: -2,
        };
        let expected = f64::from(cg.value());

        // avoid trivial case
        assert_ne!(expected, 0.0);

        // test twice to verify caching
        let ctx = &mut Wigner3jmCtx::default();
        toler_assert_eq!(TOLER, expected, ctx.cg(cg));
        toler_assert_eq!(TOLER, expected, ctx.cg(cg));
    }
}
