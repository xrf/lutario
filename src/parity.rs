//! Parity data type.
use num::{One, Zero};
use std::ops::{Add, Rem, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    #[inline]
    pub fn of<T: Rem<Output = T> + Zero + One>(i: T) -> Self {
        if (i % (T::one() + T::one())).is_zero() {
            Parity::Even
        } else {
            Parity::Odd
        }
    }

    #[inline]
    pub fn sign_i32(self) -> i32 {
        match self {
            Parity::Even => 1,
            Parity::Odd => -1,
        }
    }

    #[inline]
    pub fn sign_f64(self) -> f64 {
        match self {
            Parity::Even => 1.0,
            Parity::Odd => -1.0,
        }
    }

    #[inline]
    pub fn sign_char(self) -> char {
        match self {
            Parity::Even => '+',
            Parity::Odd => '-',
        }
    }
}

impl From<Parity> for i64 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i32 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i16 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for i8 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u64 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u32 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u16 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl From<Parity> for u8 {
    #[inline]
    fn from(p: Parity) -> Self {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl Add for Parity {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let p1: i64 = self.into();
        let p2: i64 = other.into();
        Parity::of(p1 ^ p2)
    }
}

impl Sub for Parity {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self + other
    }
}

impl Zero for Parity {
    #[inline]
    fn zero() -> Self {
        Parity::Even
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

#[inline]
pub fn sort2<T: Ord>(a: T, b: T) -> (Parity, T, T) {
    if b < a {
        (Parity::Odd, b, a)
    } else {
        (Parity::Even, a, b)
    }
}

#[inline]
pub fn sort3<T: Ord>(a: T, b: T, c: T) -> (Parity, T, T, T) {
    let (p, a, b) = sort2(a, b);
    if c < a {
        (p, c, a, b)
    } else if c < b {
        (p + Parity::Odd, a, c, b)
    } else {
        (p, a, b, c)
    }
}

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

    #[test]
    fn test_sort3() {
        assert_eq!(sort3(1, 2, 3), (Parity::Even, 1, 2, 3));
        assert_eq!(sort3(2, 1, 3), (Parity::Odd, 1, 2, 3));
        assert_eq!(sort3(2, 3, 1), (Parity::Even, 1, 2, 3));
        assert_eq!(sort3(3, 2, 1), (Parity::Odd, 1, 2, 3));
        assert_eq!(sort3(3, 1, 2), (Parity::Even, 1, 2, 3));
        assert_eq!(sort3(1, 3, 2), (Parity::Odd, 1, 2, 3));
    }
}
