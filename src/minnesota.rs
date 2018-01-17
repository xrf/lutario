//! Minnesota interaction of nucleons

use std::f64::consts::PI;
use super::basis::occ;
use super::half::Half;
use super::j_scheme::{JAtlas, OpJ200};
use super::op::Op;
use super::plane_wave_basis::{HarmSpin, HarmSpinIso};

/// A Gaussian potential:
///
/// ```text
/// V(r) = V₀ exp(−κ r²)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Gaussian {
    pub v0: f64,
    pub kappa: f64,
}

impl Gaussian {
    /// Calculate the essential part of the matrix element with momentum
    /// transfer `q` in the 3D plane-wave basis.  Specifically, it calculates:
    ///
    /// ```text
    /// (L / √π)³ ⟨(k₃ + q) (k₄ - q)| V |k₃ k₄⟩
    ///     = (V0 / κ^(3/2)) exp((−q² / 4) / κ)
    /// ```
    pub fn elem(&self, minus_q_sq_over_4: f64) -> f64 {
        self.v0 * self.kappa.powf(-1.5) * (minus_q_sq_over_4 / self.kappa).exp()
    }
}

/// Parameters for the Minnesota interaction.
///
/// <https://doi.org/10.1016/0375-9474(77)90007-0>
#[derive(Clone, Copy, Debug)]
pub struct Minnesota {
    pub r: Gaussian,
    pub t: Gaussian,
    pub s: Gaussian,
}

/// Default Minnesota interaction parameters
/// (V0 in MeV, kappa in fm⁻²)
impl Default for Minnesota {
    fn default() -> Self {
        Self {
            r: Gaussian { v0: 200.0, kappa: 1.487 },
            t: Gaussian { v0: 178.0, kappa: 0.639 },
            s: Gaussian { v0: 91.85, kappa: 0.465 },
        }
    }
}

/// Minnesota interaction in the 3D plane-wave basis.
#[derive(Clone, Copy, Debug)]
pub struct MinnesotaBox {
    pub minnesota: Minnesota,
    pub box_len: f64,
}

impl MinnesotaBox {
    pub fn new(box_len: f64) -> Self {
        Self {
            minnesota: Default::default(),
            box_len,
        }
    }

    pub fn nucleon_prod_elem(
        &self,
        l1: HarmSpinIso,
        l2: HarmSpinIso,
        l3: HarmSpinIso,
        l4: HarmSpinIso,
    ) -> f64
    {
        assert_eq!(l1 + l2, l3 + l4);

        // m² = (ns1 - ns3)²
        let m_sq = (l1.n - l3.n).norm_sq() as f64;

        // ±V₀ √(π / κ)³ / L³ × exp(−(π / L)² / κ × m²)

        // a = (√π / L)³
        let a = PI.powf(1.5) / self.box_len.powi(3);

        // b = −(π / L)²
        let b = -(PI / self.box_len).powi(2);

        // Since the conservation law is assumed to be satisfied, spin and
        // isospin can only be straight, cross, or both
        let straight_s = l1.s == l3.s && l2.s == l4.s;
        let straight_t = l1.t == l3.t && l2.t == l4.t;
        let crossed_s = l1.s == l4.s && l2.s == l3.s;
        let crossed_t = l1.t == l4.t && l2.t == l3.t;

        let vr = self.minnesota.r.elem(b * m_sq);
        let vt = -self.minnesota.t.elem(b * m_sq);
        let vs = -self.minnesota.s.elem(b * m_sq);

        let mut v = 0.0;
        if straight_s && straight_t {
            v += 0.5 * (vr + 0.5 * vt + 0.5 * vs);
        }
        if crossed_s && crossed_t {
            v -= 0.5 * (vr + 0.5 * vt + 0.5 * vs);
        }
        if straight_s && crossed_t {
            v -= 0.25 * (vt - vs);
        }
        if crossed_s && straight_t {
            v += 0.25 * (vt - vs);
        }
        a * v
    }

    pub fn neutron_elem(
        &self,
        l1: HarmSpin,
        l2: HarmSpin,
        l3: HarmSpin,
        l4: HarmSpin,
    ) -> f64
    {
        self.nucleon_prod_elem(
            l1.and_iso(Half(-1)),
            l2.and_iso(Half(-1)),
            l3.and_iso(Half(-1)),
            l4.and_iso(Half(-1)),
        ) - self.nucleon_prod_elem(
            l1.and_iso(Half(-1)),
            l2.and_iso(Half(-1)),
            l4.and_iso(Half(-1)),
            l3.and_iso(Half(-1)),
        )
    }

    pub fn make_op_neutron(
        &self,
        atlas: &JAtlas<HarmSpin, ()>,
    ) -> OpJ200<f64>
    {
        let scheme = atlas.scheme();
        let mut h2 = Op::new(scheme.clone());
        for pq in scheme.states_20(&occ::ALL2) {
            let (p, q) = pq.split_to_10_10();
            let p = HarmSpin::from(atlas.decode(p).unwrap());
            let q = HarmSpin::from(atlas.decode(q).unwrap());
            for rs in pq.costates_20(&occ::ALL2) {
                let (r, s) = rs.split_to_10_10();
                let r = HarmSpin::from(atlas.decode(r).unwrap());
                let s = HarmSpin::from(atlas.decode(s).unwrap());
                h2.add(pq, rs, self.neutron_elem(p, q, r, s));
            }
        }
        h2
    }
}
