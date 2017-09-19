use std::ops::{Add, Div, Mul, Rem, Sub};
use debugit::DebugIt;
use num::{One, ToPrimitive, Zero};
use super::utils::RangeInclusive;

/// Type that logically behaves like half-integers, but what is actually
/// stored is twice its logical value.
///
/// For example, `Half(3)` represents the fraction `3/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Half<T>(pub T);

impl<T: Add<Output = T> + Clone> From<T> for Half<T> {
    fn from(t: T) -> Self {
        Half(t.clone() + t)
    }
}

impl<T: Clone + Div<Output = T> + Rem<Output = T> + Zero + One> Half<T> {
    /// Get the value if it's half-even.  Otherwise, returns `Err(self)`.
    pub fn try_get(self) -> Result<T, Half<T>> {
        let two = T::one() + T::one();
        if (self.0.clone() % two.clone()).is_zero() {
            Ok(self.0 / two)
        } else {
            Err(self)
        }
    }
}

impl<T: Ord + Sub> Half<T> {
    pub fn abs_diff(self, other: Self) -> Half<T::Output> {
        if self.0 >= other.0 {
            self - other
        } else {
            other - self
        }
    }
}

impl<T: Clone + Ord + Add<Output = T> + Sub<Output = T> + One> Half<T> {
    /// Obtain the range of values that satisfy the triangular condition, i.e.
    /// the range from `|self − other|` to `self + other` (inclusive).
    pub fn tri_range(self, other: Half<T>) -> RangeInclusive<Half<T>> {
        RangeInclusive {
            start: Half::abs_diff(self.clone(), other.clone()),
            end: self + other + Half(One::one()),
        }
    }
}

impl<T: Add<U>, U> Add<Half<U>> for Half<T> {
    type Output = Half<T::Output>;
    fn add(self, other: Half<U>) -> Self::Output {
        Half(self.0 + other.0)
    }
}

impl<T: Sub<U>, U> Sub<Half<U>> for Half<T> {
    type Output = Half<T::Output>;
    fn sub(self, other: Half<U>) -> Self::Output {
        Half(self.0 - other.0)
    }
}

impl<T: Zero> Zero for Half<T> {
    fn zero() -> Self {
        Half(Zero::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T> Mul for Half<T>
    where T: Clone + Div<Output = T> + Rem<Output = T> + Zero + One
{
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Half(Half(self.0 * other.0).try_get()
             .map_err(DebugIt)
             .expect("cannot multiply two half-odd integers"))
    }
}

impl<T> One for Half<T>
    where T: Clone + Div<Output = T> + Rem<Output = T> + Zero + One
{
    fn one() -> Self {
        Half(T::one() + T::one())
    }
}

impl<T: ToPrimitive> ToPrimitive for Half<T> {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64().and_then(|x| Half(x).try_get().ok())
    }
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64().and_then(|x| Half(x).try_get().ok())
    }
}
