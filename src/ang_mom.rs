//! Angular momentum coupling.
use std::ops::{Add, Mul, Sub};
use num::FromPrimitive;

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
