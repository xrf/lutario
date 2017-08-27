use std::ops::Add;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Parity {
    Even,
    Odd,
}

impl From<i64> for Parity {
    fn from(i: i64) -> Self {
        if i % 2 == 0 {
            Parity::Even
        } else {
            Parity::Odd
        }
    }
}

impl From<Parity> for i64 {
    fn from(p: Parity) -> i64 {
        match p {
            Parity::Even => 0,
            Parity::Odd => 1,
        }
    }
}

impl Add<Parity> for Parity {
    type Output = Parity;
    fn add(self, other: Parity) -> Self::Output {
        let p1: i64 = self.into();
        let p2: i64 = other.into();
        Parity::from(p1 ^ p2)
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
}
