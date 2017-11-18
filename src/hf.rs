use std::mem;
use std::ops::{Add, Mul};
use super::basis::occ;
use super::block_vector::BlockVector;
use super::j_scheme::{BasisJ10, BasisJ20, DiagOpJ10, OpJ100, OpJ200};
use super::linalg::{self, Conj, EigenvalueRange, Part};
use super::matrix::Matrix;
use super::op::{DiagOp, Op, VectorMut};
use super::utils::Toler;

/// Hartree–Fock with ad hoc linear mixing.
#[derive(Clone, Copy, Debug)]
pub struct HfConf {
    pub init_mix_factor: f64,
    pub mix_modifier: f64,
    pub max_iterations: u64,
    pub toler: Toler,
    pub heevr_abstol: f64,
}

impl Default for HfConf {
    fn default() -> Self {
        Self {
            init_mix_factor: 0.5,
            mix_modifier: 2.0,
            max_iterations: 1024,
            toler: Default::default(),
            heevr_abstol: Default::default(),
        }
    }
}

impl HfConf {
    pub fn new_run<'a>(
        self,
        h1: OpJ100<'a, Vec<Matrix<f64>>>,
        h2: OpJ200<'a, Vec<Matrix<f64>>>,
    ) -> HfRun<'a> {
        let scheme = h1.left_basis.0;
        let mut dcoeff = Op::new(BasisJ10(scheme), BasisJ10(scheme));
        for p in scheme.states_10(&occ::ALL1) {
            dcoeff.set(p, p, 1.0);
        }
        HfRun {
            conf: self,
            h1,
            h2,
            energies: DiagOp::new(BasisJ10(scheme)),
            dcoeff,
            qcoeff: Op::new(BasisJ10(scheme), BasisJ10(scheme)),
            fock: Op::new(BasisJ10(scheme), BasisJ10(scheme)),
            fock_old: Op::new(BasisJ10(scheme), BasisJ10(scheme)),
            iter: Default::default(),
            mix_factor: self.init_mix_factor,
        }
    }

    pub fn adjust_mix_factor(&self, de: f64, de_new: f64, mix_factor: &mut f64) {
        if de == 0.0 {
            return;
        }
        // adjust mixing depending on whether oscillation is present
        if de_new * de < 0.0 {
            // oscillating (don't make it too high though!)
            *mix_factor = f64::min(0.5, *mix_factor * self.mix_modifier);
        } else {
            // not oscillating
            *mix_factor *= 1.0 / self.mix_modifier;
        }
    }
}

pub struct HfRun<'a> {
    pub conf: HfConf,
    pub h1: OpJ100<'a, Vec<Matrix<f64>>>,
    pub h2: OpJ200<'a, Vec<Matrix<f64>>>,
    pub energies: DiagOpJ10<'a, BlockVector<f64>>,
    pub dcoeff: OpJ100<'a, Vec<Matrix<f64>>>,
    pub qcoeff: OpJ100<'a, Vec<Matrix<f64>>>,
    pub fock: OpJ100<'a, Vec<Matrix<f64>>>,
    pub fock_old: OpJ100<'a, Vec<Matrix<f64>>>,
    pub iter: usize,
    pub mix_factor: f64,
}

impl<'a> HfRun<'a> {
    pub fn step(&mut self) {
        // Q[λ; v x] = ∑[i] D[λ; x i] D†[λ; i v]
        qcoeff(&self.dcoeff, &mut self.qcoeff);

        mem::swap(&mut self.fock_old, &mut self.fock);

        // compute Fock matrix
        // TODO: use "copy_from" here
        self.fock = self.h1.clone();
        fock2(&self.h2, &self.qcoeff, &mut self.fock);

        // mix in a little bit of the previous matrix to dampen oscillations
        // that can sometimes occur
        if self.iter > 0 {
            block_mat_axpby(
                self.mix_factor,
                &self.fock_old.data,
                1.0 - self.mix_factor,
                &mut self.fock.data,
            );
        }

        // solve the HF eigenvalue problem.
        // ∑[w] f[u w] D[w g] = ε[g] D[u g]
        block_heevr(
            false,
            Part::Lower,
            &self.fock.data,
            self.conf.heevr_abstol,
            &mut self.energies.data.0,
            &mut self.dcoeff.data,
        ).unwrap();
    }

    /// Iterates until the convergence criterion has been met.
    pub fn do_run(&mut self) {
        self.mix_factor = self.conf.init_mix_factor;
        self.step();
        let mut e = self.energies.sum();
        let mut de = 0.0;
        let mut i = 0;
        println!("hf:");
        while i < self.conf.max_iterations {
            self.step();
            let e_new = self.energies.sum();
            let de_new = e_new - e;
            self.conf.adjust_mix_factor(de, de_new, &mut self.mix_factor);
            println!("- {{iter: {}, orbital_energy_sum: {}, mix_factor: {}}}",
                     i, e_new, self.mix_factor);
            if self.conf.toler.check(e - e_new, e) {
                break;
            }
            e = e_new;
            de = de_new;
            i += 1;
        }
        println!("hf_converged: {}", i < self.conf.max_iterations);
    }
}

/// `y ← α × x + β × y`
pub fn block_mat_axpby<T>(
    alpha: T,
    x: &[Matrix<T>],
    beta: T,
    y: &mut [Matrix<T>],
) where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    assert_eq!(x.len(), y.len());
    for l in 0 .. x.len() {
        linalg::mat_axpby(
            alpha.clone(),
            x[l].as_ref(),
            beta.clone(),
            y[l].as_mut(),
        );
    }
}

pub fn block_heevr<T: linalg::Heevr>(
    left: bool,
    part: Part,
    a: &[Matrix<T>],
    abstol: T::Real,
    w: &mut [Vec<T::Real>],
    z: &mut [Matrix<T>],
) -> Result<(), i32> where
    T::Real: Clone,
{
    assert_eq!(a.len(), w.len());
    assert_eq!(w.len(), z.len());
    let mut isuppz = Vec::new();
    for l in 0 .. a.len() {
        // make sure matrix `a` is not modified!
        let mut a_l = a[l].clone();
        linalg::heevr(
            left,
            EigenvalueRange::All,
            part,
            a_l.as_mut(),
            abstol.clone(),
            &mut w[l],
            z[l].as_mut(),
            Err(&mut isuppz),
        )?;
    }
    Ok(())
}

/// ```text
/// Q[v x] = ∑[i] D[x i] D†[i v]
/// ```
pub fn qcoeff<'a>(
    d1: &OpJ100<'a, Vec<Matrix<f64>>>,
    q1: &mut OpJ100<'a, Vec<Matrix<f64>>>,
)
{
    let scheme = d1.left_basis.0;
    q1.fill(&0.0);
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for i in p.costates_10(&[occ::I]) {
                q1.add(p, q, d1.at(q, i) * d1.at(p, i).conj());
            }
        }
    }
}

pub fn fock2<'a>(
    h2: &OpJ200<'a, Vec<Matrix<f64>>>,
    q1: &OpJ100<'a, Vec<Matrix<f64>>>,
    f1: &mut OpJ100<'a, Vec<Matrix<f64>>>,
)
{
    let scheme = h2.left_basis.0;
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        for r in p.costates_10(&occ::ALL1) {
            for s in q.costates_10(&occ::ALL1) {
                for rs in r.combine_with_10(s, pq.j()) {
                    f1.add(p, r, (
                        pq.jweight(2) / p.jweight(2)
                            * h2.at(pq, rs)
                            * q1.at(q, s)
                    ));
                }
            }
        }
    }
}

pub fn transform_h1<'a>(
    h1: &OpJ100<'a, Vec<Matrix<f64>>>,
    d1: &OpJ100<'a, Vec<Matrix<f64>>>,
    r1: &mut OpJ100<'a, Vec<Matrix<f64>>>,
)
{
    let scheme = h1.left_basis.0;
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for r in p.costates_10(&occ::ALL1) {
                for s in p.costates_10(&occ::ALL1) {
                    r1.add(p, s, (
                        h1.at(q, r)
                            * d1.at(q, p).conj()
                            * d1.at(r, s)
                    ));
                }
            }
        }
    }
}

pub fn transform_h2<'a>(
    h2: &OpJ200<'a, Vec<Matrix<f64>>>,
    d1: &OpJ100<'a, Vec<Matrix<f64>>>,
    r2: &mut OpJ200<'a, Vec<Matrix<f64>>>,
)
{
    let scheme = h2.left_basis.0;
    let mut t2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            for tu in rs.costates_20(&occ::ALL2) {
                t2.add(pq, tu, (
                    h2.at(rs, tu)
                        * d1.at(r, p).conj()
                        * d1.at(s, q).conj()
                ));
            }
        }
    }
    for pq in scheme.states_20(&occ::ALL2) {
        let (p, q) = pq.split_to_10_10();
        for rs in pq.costates_20(&occ::ALL2) {
            let (r, s) = rs.split_to_10_10();
            for tu in rs.costates_20(&occ::ALL2) {
                r2.add(tu, pq, (
                    t2.at(tu, rs)
                        * d1.at(p, r)
                        * d1.at(q, s)
                ));
            }
        }
    }
}

pub fn hf_energy<'a>(
    h1: &OpJ100<'a, Vec<Matrix<f64>>>,
    h2: &OpJ200<'a, Vec<Matrix<f64>>>,
) -> f64
{
    // ZN[p q] =
    //     ∑[i] Ji^2 U[i i]
    //     + 1/2 ∑[i j] Jij^2 V[i j i j]
    //     + 1/6 ∑[i j k] Jijk^2 W[i j k i j k]    (NYI)
    let scheme = h1.left_basis.0;
    let mut r = 0.0;
    for i in scheme.states_10(&[occ::I]) {
        r += i.jweight(2) * h1.at(i, i);
    }
    for ij in scheme.states_20(&[occ::II]) {
        r += ij.jweight(2) * h2.at(ij, ij) / 2.0;
    }
    r
}

pub fn normord<'a>(
    h1: &OpJ100<'a, Vec<Matrix<f64>>>,
    h2: &OpJ200<'a, Vec<Matrix<f64>>>,
    r0: &mut f64,
    r1: &mut OpJ100<'a, Vec<Matrix<f64>>>,
    r2: &mut OpJ200<'a, Vec<Matrix<f64>>>,
)
{
    // UN[p q] =
    //     U[p q]
    //     + ∑[i] (Jpi/Jp)^2 V[p i q i]
    //     + 1/2 ∑[i j] (Jpij/Jp)^2 W[p i j q i j]    (NYI)
    //
    // VN[p q r s] =
    //     V[p q r s]
    //     + ∑[i] (Jpqi/Jpq)^2 W[p q i r s i]    (NYI)
    let scheme = h1.left_basis.0;
    *r0 += hf_energy(h1, h2);
    // FIXME: use copy_from
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            r1.add(p, q, h1.at(p, q));
        }
    }
    for pi in scheme.states_20(&[occ::II, occ::AI]) {
        let (p, i) = pi.split_to_10_10();
        for q in p.costates_10(&occ::ALL1) {
            for qi in q.combine_with_10(i, pi.j()) {
                r1.add(p, q, pi.jweight(2) / p.jweight(2) * h2.at(pi, qi));
            }
        }
    }
    // FIXME: use copy_from
    for pq in scheme.states_20(&occ::ALL2) {
        for rs in pq.costates_20(&occ::ALL2) {
            r2.add(pq, rs, h2.at(pq, rs));
        }
    }
}
