//! Quasidegenerate perturbation theory.
use std::ops::MulAssign;
use super::basis::occ;
use super::block_vector::BlockVector;
use super::j_scheme::{DiagOpJ10, OpJ100, OpJ200};
use super::matrix::Mat;

pub fn block_vec_set<T: Clone>(value: T, out: &mut Vec<Vec<T>>) {
    for out_l in out.iter_mut() {
        for out_l_u in out_l {
            *out_l_u = value.clone();
        }
    }
}

pub fn block_vec_mul_assign<T>(factor: T, out: &mut Vec<Vec<T>>)
    where T: Clone + MulAssign,
{
    for out_l in out.iter_mut() {
        for out_l_u in out_l {
            *out_l_u *= factor.clone();
        }
    }
}

/// Second-order Møller–Plesset perturbation theory.
///
/// ```text
/// R[p] = 1/2 ∑[i a b] J[p i]^2 abs(H[p i a b])^2 / Δ[p i a b]
/// ```
pub fn mp2<'a>(
    h1: &OpJ100<'a, Vec<Mat<f64>>>,
    h2: &OpJ200<'a, Vec<Mat<f64>>>,
) -> f64
{
    let mut r = 0.0;
    let scheme = h1.left_basis.0;
    for ij in scheme.states_20(&[occ::II]) {
        let (i, j) = ij.split_to_10_10();
        for ab in ij.costates_20(&[occ::AA]) {
            let (a, b) = ab.split_to_10_10();
            r +=
                ij.jweight(2)
                * h2.at(ij, ab).abs().powi(2)
                / (h1.at(i, i) + h1.at(j, j) - h1.at(a, a) - h1.at(b, b));
        }
    }
    r / 4.0
}

/// Diagonal QDPT2 term #3.
///
/// ```text
/// R[p] = 1/2 ∑[i a b] (J[p i]^2 / J[p]^2) abs(H[p i a b])^2 / Δ[p i a b]
/// ```
pub fn dqdpt2_term3<'a>(
    h1: &OpJ100<'a, Vec<Mat<f64>>>,
    h2: &OpJ200<'a, Vec<Mat<f64>>>,
    r: &mut DiagOpJ10<'a, BlockVector<f64>>,
)
{
    let scheme = h1.left_basis.0;
    for pi in scheme.states_20(&[occ::II, occ::AI]) {
        let (p, i) = pi.split_to_10_10();
        for ab in pi.costates_20(&[occ::AA]) {
            let (a, b) = ab.split_to_10_10();
            r.add(p, (
                pi.jweight(2)
                    / p.jweight(2)
                    * h2.at(pi, ab).abs().powi(2)
                    / (h1.at(p, p) + h1.at(i, i) - h1.at(a, a) - h1.at(b, b))
            ) / 2.0);
        }
    }
}

/// Diagonal QDPT2 term #4.
///
/// ```text
/// R[p] = −1/2 ∑[i j a] (J[i j]^2 / J[p]^2) abs(H[i j p a])^2 / Δ[i j p a]
/// ```
pub fn dqdpt2_term4<'a>(
    h1: &OpJ100<'a, Vec<Mat<f64>>>,
    h2: &OpJ200<'a, Vec<Mat<f64>>>,
    r: &mut DiagOpJ10<'a, BlockVector<f64>>,
)
{
    let scheme = h1.left_basis.0;
    for pa in scheme.states_20(&[occ::IA, occ::AA]) {
        let (p, a) = pa.split_to_10_10();
        for ij in pa.costates_20(&[occ::II]) {
            let (i, j) = ij.split_to_10_10();
            r.add(p, (
                ij.jweight(2)
                    / p.jweight(2)
                    * h2.at(ij, pa).abs().powi(2)
                    / (h1.at(i, i) + h1.at(j, j) - h1.at(p, p) - h1.at(a, a))
            ) / -2.0);
        }
    }
}
