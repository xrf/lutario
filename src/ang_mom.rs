//! Angular momentum coupling.
use super::half::Half;

/// Returns `(-1)^φ`
#[inline]
pub fn phase(phi: i32) -> f64 {
    if phi % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Returns `(2 * j + 1)^(e / 2)`.
#[inline]
pub fn jweight(j: Half<i32>, e: i32) -> f64 {
    ((j.twice() + 1) as f64).powf(e as f64 / 2.0)
}
