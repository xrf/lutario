//! Quasidegenerate perturbation theory.
use super::basis::{occ, Occ};
use super::half::Half;
use super::j_scheme::{OpJ100, OpJ200, OpJ211, StateJ10};

/// Second-order Møller–Plesset (nondegenerate) perturbation theory.
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

/// QDPT type A term (2nd order)
///
/// ```text
/// R(p, q) = 1/2 ∑[r s t] (J[r p]^2 / J[p]^2) H[r p s t] H[s t r q] / Δ
/// ```
pub fn qdpt_term_a<F>(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    p: StateJ10,
    q: StateJ10,
    r_occ: Occ,
    st_occ: [Occ; 2],
    denom: F,
) -> f64 where
    F: Fn(StateJ10, StateJ10, StateJ10) -> f64,
{
    debug_assert_eq!(p.lu().l, q.lu().l);
    let scheme = h1.scheme();
    let mut result = 0.0;
    for r in scheme.states_10(&[r_occ]) {
        for jrp in Half::tri_range(r.j(), p.j()) {
            let rp = match r.combine_with_10(p, jrp) {
                None => continue,
                Some(x) => x,
            };
            let rq = match r.combine_with_10(q, jrp) {
                None => continue,
                Some(x) => x,
            };
            for st in rp.costates_20(&[st_occ]) {
                let (s, t) = st.split_to_10_10();
                result += 1.0 / 2.0
                    * rp.jweight(2)
                    / p.jweight(2)
                    * h2.at(rp, st) * h2.at(st, rq)
                    / denom(r, s, t);
            }
        }
    }
    result
}

/// QDPT type B term (3rd order)
///
/// ```text
/// R(p, q) = 1/4 ∑[r s t u v] (J[r p]^2 / J[p]^2)
///     H[r p s t] H[s t u v] H[u v r q] / Δ
/// ```
pub fn qdpt_term_b<F>(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    p: StateJ10,
    q: StateJ10,
    r_occ: Occ,
    st_occ: [Occ; 2],
    uv_occ: [Occ; 2],
    denom: F,
) -> f64 where
    F: Fn(StateJ10, StateJ10, StateJ10, StateJ10, StateJ10) -> f64,
{
    debug_assert_eq!(p.lu().l, q.lu().l);
    let scheme = h1.scheme();
    let mut result = 0.0;
    for r in scheme.states_10(&[r_occ]) {
        for jrp in Half::tri_range(r.j(), p.j()) {
            let rp = match r.combine_with_10(p, jrp) {
                None => continue,
                Some(x) => x,
            };
            let rq = match r.combine_with_10(q, jrp) {
                None => continue,
                Some(x) => x,
            };
            for st in rp.costates_20(&[st_occ]) {
                let (s, t) = st.split_to_10_10();
                for uv in rp.costates_20(&[uv_occ]) {
                    let (u, v) = uv.split_to_10_10();
                    result += 1.0 / 4.0
                        * rp.jweight(2)
                        / p.jweight(2)
                        * h2.at(rp, st)
                        * h2.at(st, uv)
                        * h2.at(uv, rq)
                        / denom(r, s, t, u, v);
                }
            }
        }
    }
    result
}

/// QDPT type C term (3rd order)
///
/// ```text
/// R(p, q) = 1/2 ∑[r s t u v] (J[r p]^2 J[r t]^2 / (J[p]^2 J[r]^2))
///     H[r p s q] H[s t u v] H[u v r t] / Δ
/// ```
pub fn qdpt_term_c<F>(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    p: StateJ10,
    q: StateJ10,
    r_occ: Occ,
    s_occ: Occ,
    t_occ: Occ,
    uv_occ: [Occ; 2],
    denom: F,
) -> f64 where
    F: Fn(StateJ10, StateJ10, StateJ10, StateJ10, StateJ10) -> f64,
{
    debug_assert_eq!(p.lu().l, q.lu().l);
    let scheme = h1.scheme();
    let mut result = 0.0;
    for r in scheme.states_10(&[r_occ]) {
        for s in r.costates_10(&[s_occ]) {
            for jrp in Half::tri_range(r.j(), p.j()) {
                let rp = match r.combine_with_10(p, jrp) {
                    None => continue,
                    Some(x) => x,
                };
                let sq = match s.combine_with_10(q, jrp) {
                    None => continue,
                    Some(x) => x,
                };
                for t in scheme.states_10(&[t_occ]) {
                    for jrt in Half::tri_range(r.j(), t.j()) {
                        let rt = match r.combine_with_10(t, jrt) {
                            None => continue,
                            Some(x) => x,
                        };
                        let st = match s.combine_with_10(t, jrt) {
                            None => continue,
                            Some(x) => x,
                        };
                        for uv in rt.costates_20(&[uv_occ]) {
                            let (u, v) = uv.split_to_10_10();
                            result += 1.0 / 2.0
                                * rp.jweight(2)
                                * rt.jweight(2)
                                / p.jweight(2)
                                / r.jweight(2)
                                * h2.at(rp, sq)
                                * h2.at(st, uv)
                                * h2.at(uv, rt)
                                / denom(r, s, t, u, v);
                        }
                    }
                }
            }
        }
    }
    result
}

/// QDPT type D term (3rd order)
///
/// ```text
/// R(p, q) = 1/2 ∑[r s t u v] (J[r p]^2 / (J[p]^2))
///     H[s t r p] H[u v s t] H[r q u v] / Δ
/// ```
pub fn qdpt_term_d<F>(
    h1: &OpJ100<f64>,
    h2: &OpJ211<f64>,
    p: StateJ10,
    q: StateJ10,
    r_occ: Occ,
    st_occ: [Occ; 2],
    uv_occ: [Occ; 2],
    denom: F,
) -> f64 where
    F: Fn(StateJ10, StateJ10, StateJ10, StateJ10, StateJ10) -> f64,
{
    debug_assert_eq!(p.lu().l, q.lu().l);
    let scheme = h1.scheme();
    let mut result = 0.0;
    for r in scheme.states_10(&[r_occ]) {
        for jrp in Half::tri_range(r.j(), p.j()) {
            let rp = match r.combine_with_10_to_21(p, jrp) {
                None => continue,
                Some(x) => x,
            };
            let rq = match r.combine_with_10_to_21(q, jrp) {
                None => continue,
                Some(x) => x,
            };
            for st in rp.costates_21(&[st_occ]) {
                let (s, t) = st.split_to_10_10();
                for uv in rp.costates_21(&[uv_occ]) {
                    let (u, v) = uv.split_to_10_10();
                    result +=
                        rp.jweight(2)
                        / p.jweight(2)
                        * h2.at(st, rp)
                        * h2.at(uv, st)
                        * h2.at(rq, uv)
                        / denom(r, s, t, u, v);
                }
            }
        }
    }
    result
}

/// Obtain a specific QDPT term.
///
/// `term` is an integer starting from one.  Terms 3 and 4 are second order,
/// and terms 5 through 22 are third order.
///
/// Note that some third-order terms require a Pandya transformed matrix.
pub fn qdpt_term(
    term: u32,
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    h2p: Option<&OpJ211<f64>>,
    p: StateJ10,
    q: StateJ10,
) -> f64
{
    debug_assert_eq!(p.lu().l, q.lu().l);
    let hd = |p: StateJ10| h1.at(p, p);
    let get_h2p = || h2p.expect("Pandya transformed matrix needed for this term");
    match term {
        1 => {
            let scheme = h1.scheme();
            let mut result = 0.0;
            for i in scheme.states_10(&[occ::I]) {
                for jip in Half::tri_range(i.j(), p.j()) {
                    let ip = match i.combine_with_10(p, jip) {
                        None => continue,
                        Some(x) => x,
                    };
                    let iq = match i.combine_with_10(q, jip) {
                        None => continue,
                        Some(x) => x,
                    };
                    result += ip.jweight(2) / p.jweight(2) * h2.at(ip, iq);
                }
            }
            result
        },
        2 => h1.at(p, q),
        3 => qdpt_term_a(
            h1, h2, p, q,
            occ::I, occ::AA,
            |i, a, b| {
                hd(i) + hd(q) - hd(a) - hd(b)
            },
        ),
        4 => -qdpt_term_a(
            h1, h2, p, q,
            occ::A, occ::II,
            |a, i, j| {
                hd(i) + hd(j) - hd(a) - hd(p)
            },
        ),
        5 => qdpt_term_b(
            h1, h2, p, q,
            occ::I, occ::AA, occ::AA,
            |i, a, b, c, d| {
                (hd(i) + hd(q) - hd(a) - hd(b))
                    * (hd(i) + hd(q) - hd(c) - hd(d))
            },
        ),
        6 => -qdpt_term_b(
            h1, h2, p, q,
            occ::A, occ::AA, occ::II,
            |a, b, c, i, j| {
                (hd(i) + hd(j) - hd(a) - hd(p))
                    * (hd(i) + hd(j) - hd(b) - hd(c))
            },
        ),
        7 => -qdpt_term_b(
            h1, h2, p, q,
            occ::A, occ::II, occ::AA,
            |c, i, j, a, b| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(i) + hd(j) - hd(c) - hd(p))
            },
        ),
        8 => -qdpt_term_b(
            h1, h2, p, q,
            occ::A, occ::II, occ::II,
            |a, k, l, i, j| {
                (hd(i) + hd(j) - hd(a) - hd(p))
                    * (hd(k) + hd(l) - hd(a) - hd(p))
            },
        ),
        9 => qdpt_term_b(
            h1, h2, p, q,
            occ::I, occ::AA, occ::II,
            |i, a, b, j, k| {
                (hd(i) + hd(q) - hd(a) - hd(b))
                    * (hd(j) + hd(k) - hd(a) - hd(b))
            },
        ),
        10 => qdpt_term_b(
            h1, h2, p, q,
            occ::I, occ::II, occ::AA,
            |k, i, j, a, b| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(k) + hd(q) - hd(a) - hd(b))
            },
        ),
        11 => -qdpt_term_c(
            h1, h2, p, q,
            occ::I, occ::I, occ::I, occ::AA,
            |k, j, i, b, a| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(i) + hd(k) - hd(a) - hd(b))
            },
        ),
        12 => qdpt_term_c(
            h1, h2, p, q,
            occ::A, occ::A, occ::A, occ::II,
            |a, c, b, i, j| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(i) + hd(j) - hd(a) - hd(c))
            },
        ),
        13 => qdpt_term_c(
            h1, h2, p, q,
            occ::I, occ::A, occ::I, occ::AA,
            |i, a, j, b, c| {
                (hd(i) - hd(a))
                    * (hd(i) + hd(j) - hd(b) - hd(c))
            },
        ),
        14 => qdpt_term_c(
            h1, h2, p, q,
            occ::A, occ::I, occ::I, occ::AA,
            |c, j, i, b, a| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(j) + hd(q) - hd(c) - hd(p))
            },
        ),
        15 => -qdpt_term_c(
            h1, h2, p, q,
            occ::I, occ::A, occ::A, occ::II,
            |i, a, b, j, k| {
                (hd(i) - hd(a))
                    * (hd(j) + hd(k) - hd(a) - hd(b))
            },
        ),
        16 => -qdpt_term_c(
            h1, h2, p, q,
            occ::A, occ::I, occ::A, occ::II,
            |b, k, a, j, i| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(k) + hd(q) - hd(b) - hd(p))
            },
        ),
        17 => qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::A, occ::IA, occ::AI,
            |c, i, a, b, j| {
                (hd(i) + hd(q) - hd(a) - hd(c))
                    * (hd(i) + hd(j) - hd(a) - hd(b))
            },
        ),
        18 => qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::A, occ::AI, occ::IA,
            |c, b, j, i, a| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(i) + hd(q) - hd(a) - hd(c))
            },
        ),
        19 => qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::A, occ::IA, occ::IA,
            |c, i, a, j, b| {
                (hd(i) + hd(q) - hd(a) - hd(c))
                    * (hd(j) + hd(q) - hd(b) - hd(c))
            },
        ),
        20 => -qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::I, occ::AI, occ::AI,
            |k, b, j, a, i| {
                (hd(i) + hd(k) - hd(a) - hd(p))
                    * (hd(j) + hd(k) - hd(b) - hd(p))
            },
        ),
        21 => -qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::I, occ::IA, occ::AI,
            |k, j, b, a, i| {
                (hd(i) + hd(k) - hd(a) - hd(p))
                    * (hd(i) + hd(j) - hd(a) - hd(b))
            },
        ),
        22 => -qdpt_term_d(
            h1, get_h2p(), p, q,
            occ::I, occ::AI, occ::IA,
            |k, a, i, j, b| {
                (hd(i) + hd(j) + hd(q) - hd(a) - hd(b) - hd(p))
                    * (hd(i) + hd(k) - hd(a) - hd(p))
            },
        ),
        _ => panic!("term = {} is invalid", term),
    }
}

/// Sum of all terms at second order.
pub fn qdpt2_terms(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    p: StateJ10,
    q: StateJ10,
) -> f64
{
    (3 .. 5).map(|term| qdpt_term(term, h1, h2, None, p, q)).sum()
}

/// Sum of all terms at third order.
pub fn qdpt3_terms(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
    h2p: &OpJ211<f64>,
    p: StateJ10,
    q: StateJ10,
) -> f64
{
    (5 .. 23).map(|term| qdpt_term(term, h1, h2, Some(h2p), p, q)).sum()
}
