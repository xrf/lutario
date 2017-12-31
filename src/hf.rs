use std::{f64, mem};
use std::ops::{Add, Mul};
use super::basis::occ;
use super::j_scheme::{DiagOpJ10, MopJ012, OpJ100, OpJ200};
use super::linalg::{self, Conj, EigenvalueRange, Part};
use super::mat::Mat;
use super::op::{Op, VectorMut};
use super::utils::Toler;

/// Hartree–Fock with ad hoc linear mixing.
#[derive(Clone, Copy, Debug)]
pub struct Conf {
    pub init_mix_factor: f64,
    pub mix_modifier: f64,
    pub toler: Toler,
    pub heevr_abstol: f64,
}

impl Default for Conf {
    fn default() -> Self {
        Self {
            init_mix_factor: 0.5,
            mix_modifier: 2.0,
            toler: Default::default(),
            heevr_abstol: Default::default(),
        }
    }
}

impl Conf {
    pub fn make_run<'a>(
        self,
        h1: &'a OpJ100<f64>,
        h2: &'a OpJ200<f64>,
    ) -> Run<'a> {
        let scheme = h1.scheme();
        let mut dcoeff = Op::new(scheme.clone());
        for p in scheme.states_10(&occ::ALL1) {
            dcoeff.set(p, p, 1.0);
        }
        Run {
            conf: self,
            h1,
            h2,
            energies: Op::new_vec(scheme.clone()),
            dcoeff,
            qcoeff: Op::new(scheme.clone()),
            fock: Op::new(scheme.clone()),
            fock_old: Op::new(scheme.clone()),
            energy_sum: f64::NAN,
            energy_change: 0.0,
            mix_factor: self.init_mix_factor,
            first: true,
        }
    }

    pub fn adjust_mix_factor(
        &self,
        de: f64,
        de_new: f64,
        mix_factor: &mut f64,
    ) {
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

pub struct Run<'a> {
    pub conf: Conf,
    pub h1: &'a OpJ100<f64>,
    pub h2: &'a OpJ200<f64>,
    pub energies: DiagOpJ10<f64>,
    pub dcoeff: OpJ100<f64>,
    pub qcoeff: OpJ100<f64>,
    pub fock: OpJ100<f64>,
    pub fock_old: OpJ100<f64>,
    pub energy_sum: f64,
    pub energy_change: f64,
    pub mix_factor: f64,
    pub first: bool,
}

impl<'a> Run<'a> {
    pub fn step(&mut self) -> Result<(), ()> {
        // Q[λ; v x] = ∑[i] D[λ; x i] D†[λ; i v]
        qcoeff(&self.dcoeff, &mut self.qcoeff);

        mem::swap(&mut self.fock_old, &mut self.fock);

        // compute Fock matrix
        self.fock.clone_from(&self.h1);
        fock2(&self.h2, &self.qcoeff, &mut self.fock);

        // mix in a little bit of the previous matrix to dampen oscillations
        // that can sometimes occur
        if self.first {
            self.first = false;
            block_mat_axpby(
                self.mix_factor,
                &self.fock_old.data.0,
                1.0 - self.mix_factor,
                &mut self.fock.data.0,
            );
        }

        // solve the HF eigenvalue problem.
        // ∑[w] f[u w] D[w g] = ε[g] D[u g]
        block_heevr(
            false,
            Part::Lower,
            &self.fock.data.0,
            self.conf.heevr_abstol,
            &mut self.energies.data.0,
            &mut self.dcoeff.data.0,
        ).unwrap();

        // test convergence using energy sum and adjust mixing
        let old_energy_sum = mem::replace(
            &mut self.energy_sum,
            weighted_sum(&self.energies),
        );
        let old_energy_change = mem::replace(
            &mut self.energy_change,
            self.energy_sum - old_energy_sum,
        );
        self.conf.adjust_mix_factor(
            old_energy_change,
            self.energy_change,
            &mut self.mix_factor,
        );
        if self.conf.toler.is_eq(self.energy_sum, old_energy_sum) {
            Ok(())
        } else {
            Err(())
        }
    }

    /// Iterates until the convergence criterion has been met.
    pub fn do_run(&mut self) -> Result<(), ()> {
        println!("hf:");
        for i in 0 .. 1024 {
            if self.step().is_ok() {
                println!("hf_converged: true");
                return Ok(());
            }
            println!("- {{iter: {}, energy_sum: {}, mix_factor: {}}}",
                     i, self.energy_sum, self.mix_factor);
        }
        println!("hf_converged: false");
        Err(())
    }
}

/// `y ← α × x + β × y`
pub fn block_mat_axpby<T>(
    alpha: T,
    x: &[Mat<T>],
    beta: T,
    y: &mut [Mat<T>],
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
    a: &[Mat<T>],
    abstol: T::Real,
    w: &mut [Vec<T::Real>],
    z: &mut [Mat<T>],
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
/// R = ∑[p] Jp E[p]
/// ```
pub fn weighted_sum(e1: &DiagOpJ10<f64>) -> f64 {
    let scheme = e1.scheme();
    let mut r = 0.0;
    for p in scheme.states_10(&occ::ALL1) {
        r += p.jweight(2) * e1.at(p, p);
    }
    r
}

/// ```text
/// Q[v x] = ∑[i] D[x i] D†[i v]
/// ```
pub fn qcoeff(
    d1: &OpJ100<f64>,
    q1: &mut OpJ100<f64>,
)
{
    let scheme = d1.scheme();
    q1.set_zero();
    for p in scheme.states_10(&occ::ALL1) {
        for q in p.costates_10(&occ::ALL1) {
            for i in p.costates_10(&[occ::I]) {
                q1.add(p, q, d1.at(q, i) * d1.at(p, i).conj());
            }
        }
    }
}

pub fn fock2(
    h2: &OpJ200<f64>,
    q1: &OpJ100<f64>,
    f1: &mut OpJ100<f64>,
)
{
    let scheme = h2.scheme();
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

pub fn transform_h1(
    h1: &OpJ100<f64>,
    d1: &OpJ100<f64>,
    r1: &mut OpJ100<f64>,
)
{
    let scheme = h1.scheme();
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

pub fn transform_h2(
    h2: &OpJ200<f64>,
    d1: &OpJ100<f64>,
    r2: &mut OpJ200<f64>,
)
{
    let scheme = h2.scheme();
    let mut t2 = Op::new(scheme.clone());
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
                        * d1.at(r, p)
                        * d1.at(s, q)
                ));
            }
        }
    }
}

pub fn hf_energy(
    h1: &OpJ100<f64>,
    h2: &OpJ200<f64>,
) -> f64
{
    // ZN[p q] =
    //     ∑[i] Ji² U[i i]
    //     + 1/2 ∑[i j] Jij^2 V[i j i j]
    //     + 1/6 ∑[i j k] Jijk^2 W[i j k i j k]    (NYI)
    let scheme = h1.scheme();
    let mut r = 0.0;
    for i in scheme.states_10(&[occ::I]) {
        r += i.jweight(2) * h1.at(i, i);
    }
    for ij in scheme.states_20(&[occ::II]) {
        r += ij.jweight(2) * h2.at(ij, ij) / 2.0;
    }
    r
}

pub fn normord(
    h: &MopJ012<f64>,
    r: &mut MopJ012<f64>,
)
{
    // UN[p q] =
    //     U[p q]
    //     + ∑[i] (Jpi / Jp)² V[p i q i]
    //     + 1/2 ∑[i j] (Jpij/Jp)² W[p i j q i j]    (NYI)
    //
    // VN[p q r s] =
    //     V[p q r s]
    //     + ∑[i] (Jpqi / Jpq)² W[p q i r s i]    (NYI)
    let scheme = h.1.scheme();
    r.0 += h.0 + hf_energy(&h.1, &h.2);
    r.1.clone_from(&h.1);
    for pi in scheme.states_20(&[occ::II, occ::AI]) {
        let (p, i) = pi.split_to_10_10();
        for q in p.costates_10(&occ::ALL1) {
            for qi in q.combine_with_10(i, pi.j()) {
                r.1.add(p, q, pi.jweight(2) / p.jweight(2) * h.2.at(pi, qi));
            }
        }
    }
    r.2.clone_from(&h.2);
}
