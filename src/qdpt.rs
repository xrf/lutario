use std::ops::MulAssign;
use super::ang_mom::{jweight, phase};
use super::basis::{Excit10, Excit20, JScheme};
use super::block_matrix::BlockMat;

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

/// Diagonal QDPT2 term #3.
///
/// ```text
/// 1/2 ∑[jpi i a b] (Jpi / Jp)² |V[p i a b]|² / (h1d[p] + h1d[i] − h1d[a] − h1[b])
/// ```
pub fn dqdpt2_term3(
    scheme: &JScheme,
    h1: BlockMat<f64>,
    h2: BlockMat<f64>,
    out: &mut Vec<Vec<f64>>,
)
{
    block_vec_set(0.0, out);
    for lpi in scheme.basis_j20.chans() {
        let lpi = lpi.index as usize;
        let jpi = scheme.basis_j20.chans[lpi].j;
        for upi in scheme.basis_j20.auxs(lpi, Excit20::II, Excit20::AI) {
            let (sp, si) = scheme.basis_j20.decode(lpi, upi);
            let (lp, up) = scheme.basis_j10.encode(sp);
            let (li, ui) = scheme.basis_j10.encode(si);
            let jp = scheme.basis_j10.chans[lp].j;
            let ji = scheme.basis_j10.chans[li].j;
            for uab in scheme.basis_j20.auxs(lpi, Excit20::AA, Excit20::AA) {
                let (sa, sb) = scheme.basis_j20.decode(lpi, uab);
                let (la, ua) = scheme.basis_j10.encode(sa);
                let (lb, ub) = scheme.basis_j10.encode(sb);
                let ja = scheme.basis_j10.chans[la].j;
                let jb = scheme.basis_j10.chans[lb].j;
                let x =
                    h2.index(lpi)[(upi, uab)].abs().powi(2)
                    / (h1.index(lp)[(up, up)]
                       + h1.index(li)[(ui, ui)]
                       - h1.index(la)[(ua, ua)]
                       - h1.index(lb)[(ub, ub)]);
                out[lp][up] +=
                    jweight(jpi, 2) / jweight(jp, 2)
                    * (1.0 - if sa != sb {
                        phase((ja + jb - jpi).unwrap())
                    } else {
                        0.0
                    })
                    * x;
                if scheme.basis_j10.excit(lp, up) == Excit10::I && sp != si {
                    out[li][ui] +=
                        jweight(jpi, 2) / jweight(ji, 2)
                        * phase((jp + ji - jpi).unwrap())
                        * (1.0 - if sa != sb {
                            phase((ja + jb - jpi).unwrap())
                        } else {
                            0.0
                        })
                        * x;
                }
            }
        }
    }
    block_vec_mul_assign(1.0 / 2.0, out);
}

/// Diagonal QDPT2 term #4.
///
/// ```text
/// -1/2 ∑[jij i j a] (Jij / Jp)² |V[i j p a]|² / (h1d[i] + h1d[j] - h1d[p] - h1d[a])
/// ```
#[allow(unused_variables)]
pub fn dqdpt2_term4(
    scheme: &JScheme,
    h1: BlockMat<f64>,
    h2: BlockMat<f64>,
    out: &mut Vec<Vec<f64>>,
)
{
    block_vec_set(0.0, out);
    for lap in scheme.basis_j20.chans() {
        let lap = lap.index as usize;
        let jap = scheme.basis_j20.chans[lap].j;
        for uap in scheme.basis_j20.auxs(lap, Excit20::II, Excit20::AI) {
            let (sa, sp) = scheme.basis_j20.decode(lap, uap);
            let (la, ua) = scheme.basis_j10.encode(sa);
            let (lp, up) = scheme.basis_j10.encode(sp);
            let ja = scheme.basis_j10.chans[la].j;
            let jp = scheme.basis_j10.chans[lp].j;
            for uij in scheme.basis_j20.auxs(lap, Excit20::AA, Excit20::AA) {
                let (si, sj) = scheme.basis_j20.decode(lap, uij);
                let (li, ui) = scheme.basis_j10.encode(si);
                let (lj, uj) = scheme.basis_j10.encode(sj);
                let ji = scheme.basis_j10.chans[li].j;
                let jj = scheme.basis_j10.chans[lj].j;
                let x =
                    if si != sj {
                        phase((ji + jj - jap).unwrap())
                    } else {
                        0.0
                    } * h2.index(lap)[(uij, uap)].abs().powi(2)
                    / (h1.index(li)[(ui, ui)]
                    + h1.index(lj)[(uj, uj)]
                    - h1.index(la)[(ua, ua)]
                    - h1.index(lp)[(up, up)]);
                out[lp][up] +=
                    jweight(jap, 2) / jweight(jp, 2)
                    * x;
                if scheme.basis_j10.excit(lp, up) == Excit10::A && sa != sp {
                    out[la][ua] +=
                        jweight(jap, 2) / jweight(ja, 2)
                        * phase((ja + jp - jap).unwrap())
                        * x;
                }
            }
        }
    }
    block_vec_mul_assign(-1.0 / 2.0, out);
}
