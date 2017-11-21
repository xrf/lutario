//! Quasidegenerate perturbation theory.
use super::basis::occ;
use super::j_scheme::{DiagOpJ10, OpJ100, OpJ200};

/// Second-order Møller–Plesset perturbation theory.
///
/// ```text
/// R[p] = 1/2 ∑[i a b] J[p i]^2 abs(H[p i a b])^2 / Δ[p i a b]
/// ```
pub fn mp2(h1: &OpJ100<f64>, h2: &OpJ200<f64>) -> f64
{
    let mut r = 0.0;
    let scheme = h1.scheme();
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
pub fn dqdpt2_term3(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    r: &mut DiagOpJ10<f64>,
)
{
    let scheme = h1.scheme();
    for pi in scheme.states_20(&[occ::II, occ::AI]) {
        let (p, i) = pi.split_to_10_10();
        for ab in pi.costates_20(&[occ::AA]) {
            let (a, b) = ab.split_to_10_10();
            r.add(p, p, (
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
pub fn dqdpt2_term4(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    r: &mut DiagOpJ10<f64>,
)
{
    let scheme = h1.scheme();
    for pa in scheme.states_20(&[occ::IA, occ::AA]) {
        let (p, a) = pa.split_to_10_10();
        for ij in pa.costates_20(&[occ::II]) {
            let (i, j) = ij.split_to_10_10();
            r.add(p, p, (
                ij.jweight(2)
                    / p.jweight(2)
                    * h2.at(ij, pa).abs().powi(2)
                    / (h1.at(i, i) + h1.at(j, j) - h1.at(p, p) - h1.at(a, a))
            ) / -2.0);
        }
    }
}
