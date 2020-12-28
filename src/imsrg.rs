//! Implementation of IM-SRG(2)
//!
//! This module contains all the terms of IM-SRG(2) commutator or linked
//! product, implemented for J-scheme.  It also contains a set of helper
//! routines for driving the IM-SRG evolution ([`Run`](struct.Run.html)).

use std::f64;
use std::sync::Arc;
use super::ang_mom::Wigner6jCtx;
use super::basis::{occ, ChanState, Occ, Occ20};
use super::half::Half;
use super::j_scheme::{JScheme, MopJ012, OpJ100, OpJ200, StateJ10,
                      clone_mop_j012_from_tri_slice,
                      clone_mop_j012_to_tri_slice,
                      extent_mop_j012_as_tri, new_mop_j012,
                      op200_to_op211, op211_to_op200};
use super::linalg::{self, Transpose};
use super::mat::Mat;
use super::op::Op;
use super::sg_ode;
use super::tri_mat::trs;
use super::utils::{self, Toler};
use super::vector_driver::basic::BasicVectorDriver;

/// Term 011
///
/// ```text
/// C[] ←+ α ∑[i a] Ji^2 A[i a] B[a i]
/// ```
pub fn c011(
    alpha: f64,
    a1: &OpJ100<f64>,
    b1: &OpJ100<f64>,
    c0: &mut f64,
)
{
    let scheme = a1.scheme();
    for i in scheme.states_10(&[occ::I]) {
        for a in i.costates_10(&[occ::A]) {
            *c0 += alpha * i.jweight(2) * a1.at(i, a) * b1.at(a, i);
        }
    }
}

/// Term 022
///
/// ```text
/// C[] ←+ α/4 ∑[i j a b] Jij^2 A[i j a b] B[a b i j]
/// ```
pub fn c022(
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c0: &mut f64,
)
{
    let scheme = a2.scheme();
    for ij in scheme.states_20(&[occ::II]) {
        for ab in ij.costates_20(&[occ::AA]) {
            *c0 += alpha / 4.0 * ij.jweight(2) * a2.at(ij, ab) * b2.at(ab, ij);
        }
    }
}

/// Term 111
///
/// ```text
/// C[p q] ←+ −α (∑[i] A[i q] B[p i] + ∑[a] A[p a] B[a q])
/// ```
pub fn c111(
    alpha: f64,
    a1: &OpJ100<f64>,
    b1: &OpJ100<f64>,
    c1: &mut OpJ100<f64>,
)
{
    let scheme = a1.scheme();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for i in p.costates_10(&[occ::I]) {
                c1.add(p, q, -alpha * a1.at(i, q) * b1.at(p, i));
            }
            for a in p.costates_10(&[occ::A]) {
                c1.add(p, q, alpha * a1.at(p, a) * b1.at(a, q));
            }
        }
    }
}

/// Term 112
///
/// ```text
/// C[p q] ←+ α ∑[i a] (Jap / Jp)² A[i a] B[a p i q]
/// ```
pub fn c112(
    alpha: f64,
    a1: &OpJ100<f64>,
    b2: &OpJ200<f64>,
    c1: &mut OpJ100<f64>,
)
{
    let scheme = a1.scheme();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for i in scheme.states_10(&[occ::I]) {
                for a in i.costates_10(&[occ::A]) {
                    for jap in Half::tri_range_2(
                        (a.j(), p.j()),
                        (i.j(), q.j()),
                    ) {
                        for ap in a.combine_with_10(p, jap) {
                            for iq in i.combine_with_10(q, jap) {
                                c1.add(p, q, alpha
                                        * ap.jweight(2)
                                        / p.jweight(2)
                                        * a1.at(i, a)
                                        * b2.at(ap, iq));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Term 121
///
/// ```text
/// C[p q] ←+ α ∑[i a] (Jip / Jp)² A[i p a q] B[a i]
/// ```
pub fn c121(
    alpha: f64,
    a2: &OpJ200<f64>,
    b1: &OpJ100<f64>,
    c1: &mut OpJ100<f64>,
)
{
    let scheme = a2.scheme();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for a in scheme.states_10(&[occ::A]) {
                for i in a.costates_10(&[occ::I]) {
                    for jip in Half::tri_range_2(
                        (i.j(), p.j()),
                        (a.j(), q.j()),
                    ) {
                        for ip in i.combine_with_10(p, jip) {
                            for aq in a.combine_with_10(q, jip) {
                                c1.add(p, q, alpha
                                        * ip.jweight(2)
                                        / p.jweight(2)
                                        * a2.at(ip, aq)
                                        * b1.at(a, i));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Term 1220
///
/// ```text
/// C[p q] ←+ −α/2 ∑[i j a] (Jij / Jp)² A[i j a q] B[a p i j]
/// ```
pub fn c1220(
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c1: &mut OpJ100<f64>,
)
{
    let scheme = a2.scheme();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for a in scheme.states_10(&[occ::A]) {
                for jij in Half::tri_range(a.j(), p.j()) {
                    for ap in a.combine_with_10(p, jij) {
                        for aq in a.combine_with_10(q, jij) {
                            for ij in ap.costates_20(&[occ::II]) {
                                c1.add(p, q, -alpha / 2.0
                                        * ij.jweight(2)
                                        / p.jweight(2)
                                        * a2.at(ij, aq)
                                        * b2.at(ap, ij));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Term 1221
///
/// ```text
/// C[p q] ←+ α/2 ∑[i a b] (Jip / Jp)² A[i p a b] B[a b i q]
/// ```
pub fn c1221(
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c1: &mut OpJ100<f64>,
)
{
    let scheme = a2.scheme();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for i in scheme.states_10(&[occ::I]) {
                for jip in Half::tri_range(i.j(), p.j()) {
                    for ip in i.combine_with_10(p, jip) {
                        for iq in i.combine_with_10(q, jip) {
                            for ab in ip.costates_20(&[occ::AA]) {
                                c1.add(p, q, alpha / 2.0
                                        * ip.jweight(2)
                                        / p.jweight(2)
                                        * a2.at(ip, ab)
                                        * b2.at(ab, iq));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Term 212
///
/// ```text
/// C[p q r s] ←+ α (−2 𝒜[r s] ∑[i] A[i r] B[p q i s]
///                 + 2 𝒜[p q] ∑[a] A[p a] B[a q r s])
/// ```
pub fn c212(
    alpha: f64,
    a1: &OpJ100<f64>,
    b2: &OpJ200<f64>,
    c2: &mut OpJ200<f64>,
)
{
    let scheme = a1.scheme();
    for pq in scheme.states_20(&occ::ALL2) {
        for is in pq.costates_20(&[occ::II, occ::IA]) {
            let (i, s) = is.split_to_10_10();
            for r in i.costates_10(&occ::ALL1) {
                for rs in r.combine_with_10(s, is.j()) {
                    c2.add(pq, rs, -2.0 * alpha * a1.at(i, r) * b2.at(pq, is));
                }
            }
        }
    }
    for rs in scheme.states_20(&occ::ALL2) {
        for aq in rs.costates_20(&[occ::AI, occ::AA]) {
            let (a, q) = aq.split_to_10_10();
            for p in a.costates_10(&occ::ALL1) {
                for pq in p.combine_with_10(q, aq.j()) {
                    c2.add(pq, rs, 2.0 * alpha * a1.at(p, a) * b2.at(aq, rs));
                }
            }
        }
    }
}

/// Term 221
///
/// ```text
/// C[p q r s] ←+ α (−2 𝒜[p q] ∑[i] A[i q r s] B[p i]
///                 + 2 𝒜[r s] ∑[a] A[p q a s] B[a r])
/// ```
pub fn c221(
    alpha: f64,
    a2: &OpJ200<f64>,
    b1: &OpJ100<f64>,
    c2: &mut OpJ200<f64>,
)
{
    let scheme = a2.scheme();
    for rs in scheme.states_20(&occ::ALL2) {
        for iq in rs.costates_20(&[occ::II, occ::IA]) {
            let (i, q) = iq.split_to_10_10();
            for p in i.costates_10(&occ::ALL1) {
                for pq in p.combine_with_10(q, iq.j()) {
                    c2.add(pq, rs, -2.0 * alpha * a2.at(iq, rs) * b1.at(p, i));
                }
            }
        }
    }
    for pq in scheme.states_20(&occ::ALL2) {
        for as_ in pq.costates_20(&[occ::AI, occ::AA]) {
            let (a, s) = as_.split_to_10_10();
            for r in a.costates_10(&occ::ALL1) {
                for rs in r.combine_with_10(s, as_.j()) {
                    c2.add(pq, rs, 2.0 * alpha * a2.at(pq, as_) * b1.at(a, r));
                }
            }
        }
    }
}

/// Generic 2220/2222-type term
///
/// ```text
/// C[p q r s] ←+ α/2 ∑[t u] A[p q t u] B[t u r s]
/// ```
pub fn c2222_base(
    t_or_u_occ: Occ,
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c2: &mut OpJ200<f64>,
)
{
    let scheme = a2.scheme();
    let tu_occ = match t_or_u_occ {
        Occ::I => Occ20::II,
        Occ::A => Occ20::AA,
    };
    for l in 0 .. scheme.basis_20.num_chans() {
        let b2_l = b2.data.0[l as usize].as_ref();
        let mut wb2_l = Vec::default();
        for (utu, b2_ltu) in b2_l.rows().enumerate() {
            let (t, u) = scheme.basis_20.decode(ChanState { l, u: utu as _ });
            let weight = if t == u { 0.5 } else { 1.0 };
            for &b2_lturs in b2_ltu {
                wb2_l.push(weight * b2_lturs);
            }
        }
        let wb2_l = Mat::from_vec(wb2_l, b2_l.num_rows(), b2_l.num_cols());
        linalg::gemm(
            Transpose::None,
            Transpose::None,
            alpha,
            a2.data.0[l as usize].as_ref().slice(
                utils::max_range(),
                utils::cast_range(scheme.basis_20.aux_range(l, tu_occ)),
            ),
            wb2_l.as_ref().slice(
                utils::cast_range(scheme.basis_20.aux_range(l, tu_occ)),
                utils::max_range(),
            ),
            1.0,
            c2.data.0[l as usize].as_mut(),
        );
    }
}

/// Term 2220
///
/// ```text
/// C[p q r s] ←+ α/2 ∑[i j] A[i j r s] B[p q i j]
/// ```
pub fn c2220(
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c2: &mut OpJ200<f64>,
)
{
    c2222_base(Occ::I, alpha, b2, a2, c2);
}

/// Term 2222
///
/// ```text
/// C[p q r s] ←+ α/2 ∑[a b] A[p q a b] B[a b r s]
/// ```
pub fn c2222(
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c2: &mut OpJ200<f64>,
)
{
    c2222_base(Occ::A, alpha, a2, b2, c2);
}

/// Term 2221 (particle-hole term)
///
/// ```text
/// C[p q r s] ←+ 4 α 𝒜[p q] 𝒜[r s] ∑[i a] A[i p a r] B[a q i s]
/// ```
pub fn c2221(
    w6j_ctx: &mut Wigner6jCtx,
    alpha: f64,
    a2: &OpJ200<f64>,
    b2: &OpJ200<f64>,
    c2: &mut OpJ200<f64>,
)
{
    let scheme = a2.scheme();
    let mut ac2 = Op::new(scheme.clone());
    let mut bc2 = Op::new(scheme.clone());
    let mut cc2 = Op::new(scheme.clone());
    op200_to_op211(w6j_ctx, 1.0, a2, &mut ac2);
    op200_to_op211(w6j_ctx, 1.0, b2, &mut bc2);
    for l in 0 .. scheme.basis_21.num_chans() {
        linalg::gemm(
            Transpose::None,
            Transpose::None,
            4.0 * alpha,
            bc2.data.0[l as usize].as_ref().slice(
                utils::max_range(),
                utils::cast_range(scheme.basis_21.aux_range(l, occ::IA)),
            ),
            ac2.data.0[l as usize].as_ref().slice(
                utils::cast_range(scheme.basis_21.aux_range(l, occ::IA)),
                utils::max_range(),
            ),
            1.0,
            cc2.data.0[l as usize].as_mut(),
        );
    }
    op211_to_op200(w6j_ctx, 1.0, &cc2, c2);
}

/// Linked product of two many-body operators
pub fn linked(
    w6j_ctx: &mut Wigner6jCtx,
    alpha: f64,
    a: &MopJ012<f64>,
    b: &MopJ012<f64>,
    c: &mut MopJ012<f64>,
)
{
    c011(alpha, &a.1, &b.1, &mut c.0);
    c022(alpha, &a.2, &b.2, &mut c.0);
    c111(alpha, &a.1, &b.1, &mut c.1);
    c112(alpha, &a.1, &b.2, &mut c.1);
    c121(alpha, &a.2, &b.1, &mut c.1);
    c1220(alpha, &a.2, &b.2, &mut c.1);
    c1221(alpha, &a.2, &b.2, &mut c.1);
    c212(alpha, &a.1, &b.2, &mut c.2);
    c221(alpha, &a.2, &b.1, &mut c.2);
    c2220(alpha, &a.2, &b.2, &mut c.2);
    c2221(w6j_ctx, alpha, &a.2, &b.2, &mut c.2);
    c2222(alpha, &a.2, &b.2, &mut c.2);
}

/// Commutator of two many-body operators
pub fn commut(
    w6j_ctx: &mut Wigner6jCtx,
    alpha: f64,
    a: &MopJ012<f64>,
    b: &MopJ012<f64>,
    c: &mut MopJ012<f64>,
)
{
    linked(w6j_ctx, alpha, a, b, c);
    linked(w6j_ctx, -alpha, b, a, c);
}

/// Compute the monopole matrix element.
pub fn monopole_elem(h2: &OpJ200<f64>, p: StateJ10, q: StateJ10) -> f64 {
    let jpqs = Half::tri_range(p.j(), q.j());
    let x: f64 = jpqs.clone().flat_map(|jpq| {
        p.combine_with_10(q, jpq).map(|pq| {
            jpq.weight(2) * h2.at(pq, pq)
        })
    }).sum();
    let y: f64 = jpqs.map(|jpq| jpq.weight(2)).sum();
    x / y
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DenomType {
    MoellerPlesset,
    EpsteinNesbet,
}

/// Calculate the White generator, multiply by `alpha`, and then add it to
/// `eta`.
pub fn white_gen(
    denom_type: DenomType,
    alpha: f64,
    h: &MopJ012<f64>,
    eta: &mut MopJ012<f64>,
)
{
    let scheme = h.1.scheme();
    let monopole = |p, q| match denom_type {
        DenomType::MoellerPlesset => 0.0,
        DenomType::EpsteinNesbet => monopole_elem(&h.2, p, q),
    };
    for a in scheme.states_10(&[occ::A]) {
        for i in a.costates_10(&[occ::I]) {
            let x = alpha * h.1.at(a, i) / (
                h.1.at(a, a)
                    - h.1.at(i, i)
                    - monopole(a, i)
            );
            eta.1.add(a, i, x);
            eta.1.add(i, a, -x);
        }
    }
    for ab in scheme.states_20(&[occ::AA]) {
        let (a, b) = ab.split_to_10_10();
        for ij in ab.costates_20(&[occ::II]) {
            let (i, j) = ij.split_to_10_10();
            let x = alpha * h.2.at(ab, ij) / (
                h.1.at(a, a)
                    + h.1.at(b, b)
                    - h.1.at(i, i)
                    - h.1.at(j, j)
                    + monopole(a, b)
                    - monopole(a, i)
                    - monopole(b, i)
                    + monopole(i, j)
                    - monopole(a, j)
                    - monopole(b, j)
            );
            eta.2.add(ab, ij, x);
            eta.2.add(ij, ab, -x);
        }
    }
}

quick_error! {
    /// Error type for `Run`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Error {
        MaxFlowReached {}
        Solver(err: sg_ode::Error) {}
    }
}

/// Configuration for an IM-SRG(2) run
#[derive(Clone, Copy, Debug)]
pub struct Conf {
    pub toler: Toler,
    pub flow_step: f64,
    pub max_flow: f64,
    pub solver_conf: sg_ode::Conf,
}

/// `{ flow_step: 1.0, max_flow: 65535.0, .. }`
impl Default for Conf {
    fn default() -> Self {
        Self {
            toler: Default::default(),
            flow_step: 1.0,
            max_flow: 65535.0,
            solver_conf: Default::default(),
        }
    }
}

impl Conf {
    pub fn make_run(self, h: &MopJ012<f64>) -> Run {
        let mut h_buf = vec![0.0; extent_mop_j012_as_tri(&h)];
        clone_mop_j012_to_tri_slice(h, &mut h_buf);
        let driver = Arc::new(BasicVectorDriver::new(h_buf.len()));
        Run {
            conf: self,
            scheme: h.1.scheme().clone(),
            w6j_ctx: Default::default(),
            solver: self.solver_conf.make_solver(driver).unwrap(),
            hamil: h_buf,
            flow: 0.0,
        }
    }
}

/// The state of an IM-SRG(2) run
#[derive(Debug)]
pub struct Run {
    /// You can modify `conf` while the run is ongoing, but changing
    /// `conf.solver_conf` has no effect.
    pub conf: Conf,
    pub scheme: Arc<JScheme>,
    pub w6j_ctx: Wigner6jCtx,
    pub solver: sg_ode::Solver<Arc<BasicVectorDriver<f64>>>,
    pub hamil: Vec<f64>,
    pub flow: f64,
}

impl Run {
    pub fn conf(&self) -> &Conf {
        &self.conf
    }

    pub fn conf_mut(&mut self) -> &mut Conf {
        &mut self.conf
    }

    pub fn energy(&self) -> f64 {
        self.hamil[0]
    }

    /// Retrieve the current flow parameter
    pub fn flow(&self) -> f64 {
        self.flow
    }

    pub fn hamil(&self) -> MopJ012<f64> {
        let mut h = new_mop_j012(&self.scheme);
        clone_mop_j012_from_tri_slice(&mut h, &trs::He, &self.hamil);
        h
    }

    pub fn step(&mut self) -> Result<(), sg_ode::Error> {
        let scheme = &self.scheme;
        let w6j_ctx = &mut self.w6j_ctx;
        self.solver.step(|_, h_buf, mut dhds_buf| {
            let mut h = new_mop_j012(scheme);
            let mut dhds = new_mop_j012(scheme);
            let mut eta = new_mop_j012(scheme);
            clone_mop_j012_from_tri_slice(&mut h, &trs::He, &h_buf);
            white_gen(DenomType::EpsteinNesbet, 1.0, &h, &mut eta);
            commut(w6j_ctx, 1.0, &eta, &h, &mut dhds);
            clone_mop_j012_to_tri_slice(&dhds, &mut dhds_buf);
        }, self.flow + self.conf.flow_step, &mut self.flow, &mut self.hamil)
    }

    pub fn do_run(&mut self) -> Result<(), Error> {
        println!("imsrg:");
        println!("- {{flow: {}, energy: {}}}", self.flow(), self.energy());
        while self.flow() < self.conf().max_flow {
            let e_prev = self.energy();
            self.step().map_err(Error::Solver)?;
            println!("- {{flow: {}, energy: {}}}", self.flow(), self.energy());
            if self.conf().toler.is_eq(e_prev, self.energy()) {
                return Ok(());
            }
        }
        Err(Error::MaxFlowReached)
    }
}
