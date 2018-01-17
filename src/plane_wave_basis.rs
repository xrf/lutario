//! 3D plane-wave basis.
//!
//! The “harmonic number” `n` of a 3D plane wave is defined as:
//!
//! ```text
//! n = k L / (2 π)
//! ```
//!
//! where `L` is the box length and `k` is the angular wave number.
use std::ops::{Add, Sub};
use super::basis::{ChanState, Occ, PartState};
use super::half::Half;
use super::isqrt::isqrt_i64 as isqrt;
use super::j_scheme::JChan;
use super::utils::cast;
use super::vecn::Vec3I8;

/// Test if this integer can be expressed as a sum of three squares using
/// Legendre’s three-square theorem.
pub fn is_sum_of_3_squares(mut i: i64) -> bool {
    if i < 0 {
        false
    } else if i == 0 {
        true
    } else {
        while i % 4 == 0 {
            i /= 4;
        }
        i % 8 != 7
    }
}

#[derive(Clone, Debug)]
pub struct HarmTable {
    pub table: Vec<Vec<Vec3I8>>,
}

impl HarmTable {
    /// Calculate the smallest `nsqmax` required to have at least `num_shells`.
    pub fn num_shells_to_nsqmax(num_shells: i64) -> i64 {
        let mut nsq = -1;
        let mut shells = 0;
        while shells < num_shells {
            nsq += 1;
            shells += is_sum_of_3_squares(nsq) as i64;
        }
        nsq
    }

    /// Construct a table with all harmonics satisfying `|n|² ≤ nsqmax`.
    ///
    /// If `nsqmax` is negative, no shells are created.
    pub fn with_nsqmax(nsqmax: i64) -> Self {
        // beware of integer overflows in Vec3I8!
        assert!(nsqmax < 10000);
        let nmax = if nsqmax < 0 {
            -1
        } else if nsqmax == 0 {
            0
        } else {
            isqrt(nsqmax - 1) + 1       // ⌈√nsqmax⌉
        };
        // it's not very sparse, so just store it as a Vec
        // (density is 5/6 due to Legendre's three-square theorem)
        let mut nsq_table = Vec::new(); // nsq -> Vec<n>
        for nx in -nmax .. nmax + 1 {
            for ny in -nmax .. nmax + 1 {
                for nz in -nmax .. nmax + 1 {
                    let n = Vec3I8::new(nx, ny, nz);
                    let nsq = n.norm_sq();
                    if nsq <= nsqmax {
                        let nsq: usize = cast(nsq);
                        if nsq >= nsq_table.len() {
                            nsq_table.resize(nsq + 1, Vec::new());
                        }
                        nsq_table[nsq].push(n);
                    }
                }
            }
        }
        Self {
            table: nsq_table.into_iter().filter(|v| v.len() > 0).collect(),
        }
    }

    pub fn with_num_shells(num_shells: i64) -> Self {
        Self::with_nsqmax(Self::num_shells_to_nsqmax(num_shells))
    }

    pub fn nsqmax(&self) -> i64 {
        match self.table.last() {
            None => -1,
            Some(ns) => ns[0].norm_sq(),
        }
    }

    pub fn num_shells(&self) -> i64 {
        cast(self.table.len())
    }

    pub fn num_states_to(&self, num_filled: u32) -> u32 {
        let n: usize = self.table[.. num_filled as usize].iter()
            .map(|shell| shell.len()).sum();
        n as _
    }

    pub fn parted_ns_orbs(
        self,
        num_filled: u32,
    ) -> Vec<PartState<Occ, ChanState<JChan<HarmSpin>, ()>>>
    {
        self.table.into_iter().enumerate().flat_map(|(i, shell)| {
            let x = (i >= num_filled as usize).into();
            shell.into_iter().flat_map(move |n| vec![
                PartState { x, p: HarmSpin { n, s: Half(-1) }.into() },
                PartState { x, p: HarmSpin { n, s: Half(1) }.into() },
            ])
        }).collect()
    }
}

impl From<ChanState<JChan<HarmSpin>, ()>> for HarmSpin {
    fn from(this: ChanState<JChan<HarmSpin>, ()>) -> Self {
        this.l.k
    }
}

impl From<HarmSpin> for ChanState<JChan<HarmSpin>, ()> {
    fn from(this: HarmSpin) -> Self {
        ChanState {
            l: JChan { j: Half(0), k: this },
            u: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HarmSpinIso {
    pub n: Vec3I8,
    pub s: Half<i8>,
    pub t: Half<i8>,
}

impl HarmSpinIso {
    pub fn to_harm_spin(self) -> HarmSpin {
        HarmSpin {
            n: self.n,
            s: self.s,
        }
    }
}

impl Add for HarmSpinIso {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            n: self.n + other.n,
            s: self.s + other.s,
            t: self.t + other.t,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HarmSpin {
    pub n: Vec3I8,
    pub s: Half<i8>,
}

impl HarmSpin {
    pub fn and_iso(self, t: Half<i8>) -> HarmSpinIso {
        HarmSpinIso {
            n: self.n,
            s: self.s,
            t,
        }
    }
}

impl Add for HarmSpin {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            n: self.n + other.n,
            s: self.s + other.s,
        }
    }
}

impl Sub for HarmSpin {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            n: self.n - other.n,
            s: self.s - other.s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_nsqmax() {
        let table = HarmTable::with_nsqmax(4);
        assert_eq!(table.num_shells(), 5);
    }

    #[test]
    fn test_num_shells_to_nsqmax() {
        assert_eq!(HarmTable::num_shells_to_nsqmax(0), -1);
        assert_eq!(HarmTable::num_shells_to_nsqmax(1), 0);
        assert_eq!(HarmTable::num_shells_to_nsqmax(2), 1);
        assert_eq!(HarmTable::num_shells_to_nsqmax(3), 2);
        assert_eq!(HarmTable::num_shells_to_nsqmax(4), 3);
        assert_eq!(HarmTable::num_shells_to_nsqmax(5), 4);
        assert_eq!(HarmTable::num_shells_to_nsqmax(6), 5);
        assert_eq!(HarmTable::num_shells_to_nsqmax(7), 6);
        assert_eq!(HarmTable::num_shells_to_nsqmax(8), 8);
        assert_eq!(HarmTable::num_shells_to_nsqmax(9), 9);
        assert_eq!(HarmTable::num_shells_to_nsqmax(10), 10);
        assert_eq!(HarmTable::num_shells_to_nsqmax(11), 11);
        assert_eq!(HarmTable::num_shells_to_nsqmax(12), 12);
        assert_eq!(HarmTable::num_shells_to_nsqmax(13), 13);
        assert_eq!(HarmTable::num_shells_to_nsqmax(14), 14);
        assert_eq!(HarmTable::num_shells_to_nsqmax(15), 16);
    }
}
