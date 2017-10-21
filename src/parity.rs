//! Parity data type.
use std::ops::{Add, Rem, Sub};
use num::{One, Zero};
use super::basis::Abelian;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    pub fn of<T: Rem<Output = T> + Zero + One>(i: T) -> Self {
        if (i % (T::one() + T::one())).is_zero() {
            Parity::Even
        } else {
            Parity::Odd
        }
    }

    pub fn sign_char(self) -> char {
        match self {
            Parity::Even => '+',
            Parity::Odd => '-',
        }
    }
}

impl From<Parity> for i64 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i32 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i16 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i8 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u64 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u32 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u16 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u8 {
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl Add for Parity {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let p1: i64 = self.into();
        let p2: i64 = other.into();
        Parity::of(p1 ^ p2)
    }
}

impl Sub for Parity {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + other
    }
}

impl Zero for Parity {
    fn zero() -> Self {
        Parity::Even
    }
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl Abelian for Parity {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(Parity::Even + Parity::Even, Parity::Even);
        assert_eq!(Parity::Even + Parity::Odd, Parity::Odd);
        assert_eq!(Parity::Odd + Parity::Odd, Parity::Even);
        assert_eq!(Parity::Odd + Parity::Even, Parity::Odd);
    }
}
