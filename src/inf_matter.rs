//! Infinite matter, both electronic and nuclear

use std::f64::consts::PI;
use super::basis::occ;
use super::half::Half;
use super::j_scheme::{JAtlas, OpJ100};
use super::op::Op;
use super::phys_consts::{HBAR_C_MEVFM, M_NEUTRON_MEVPC2, M_PROTON_MEVPC2};
use super::plane_wave_basis::{HarmSpin, HarmSpinIso};

/// Kinetic energy of two species of particles, distinguished by isospin.
#[derive(Clone, Copy, Debug)]
pub struct IsoKinetic {
    pub up: Kinetic,
    pub down: Kinetic,
}

/// Units of result: MeV
impl IsoKinetic {
    /// Kinetic energy of nucleons in MeV.  Box length must be in fm.
    pub fn nucleon_mev(box_len: f64) -> Self {
        Self {
            up: Kinetic::proton_mev(box_len),
            down: Kinetic::neutron_mev(box_len),
        }
    }

    pub fn diag_op1_elem(&self, l: HarmSpinIso) -> f64 {
        if l.t > Half(0) {
            self.up.diag_op1_elem(l.to_harm_spin())
        } else {
            self.down.diag_op1_elem(l.to_harm_spin())
        }
    }
}

/// Kinetic energy of a single species of particles.
#[derive(Clone, Copy, Debug)]
pub struct Kinetic {
    /// Units: [energy]
    pub coeff: f64,
}

/// Returns the smallest positive wavenumber in a box.
///
/// Units: `rad / [box_len]`.
pub fn unit_wavenumber(box_len: f64) -> f64 {
    2.0 * PI / box_len
}

impl Kinetic {
    /// Kinetic energy of neutrons in MeV.  Box length must be in fm.
    pub fn neutron_mev(box_len: f64) -> Self {
        Self::natural(M_NEUTRON_MEVPC2 / HBAR_C_MEVFM.powi(2), box_len)
    }

    /// Kinetic energy of protons in MeV.  Box length must be in fm.
    pub fn proton_mev(box_len: f64) -> Self {
        Self::natural(M_PROTON_MEVPC2 / HBAR_C_MEVFM.powi(2), box_len)
    }

    /// Kinetic energy of particles in natural units.
    ///
    /// For electrons in atomic units, set mass to 1, in which case `box_len`
    /// would be in Bohr radii and energy would be in hartrees.
    pub fn natural(mass: f64, box_len: f64) -> Self {
        Self { coeff: unit_wavenumber(box_len).powi(2) / (2.0 * mass) }
    }

    pub fn diag_op1_elem(&self, l: HarmSpin) -> f64 {
        self.coeff * (l.n.norm_sq() as f64)
    }

    pub fn make_op(
        &self,
        atlas: &JAtlas<HarmSpin, ()>,
    ) -> OpJ100<f64>
    {
        let scheme = atlas.scheme();
        let mut h1 = Op::new(scheme.clone());
        for p in scheme.states_10(&occ::ALL1) {
            let sp = HarmSpin::from(atlas.decode(p).unwrap());
            h1.add(p, p, self.diag_op1_elem(sp));
        }
        h1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Coulomb {
    /// Units: [energy]
    pub coeff: f64,
}

impl Coulomb {
    /// Coulomb interaction in natural units.
    ///
    /// For electrons in atomic units, set mass to 1, in which case `box_len`
    /// would be in Bohr radii and energy would be in hartrees.
    pub fn natural(charge: f64, box_len: f64) -> Self {
        Self { coeff: 4.0 * PI * charge.powi(2) /
                      (box_len.powi(3) * unit_wavenumber(box_len).powi(2)) }
    }

    pub fn op2_prod_elem(&self, l: [HarmSpin; 4]) -> f64 {
        assert_eq!(l[0] + l[1], l[2] + l[3]);
        if !(l[0].s == l[2].s && l[1].s == l[3].s) {
            return 0.0;
        }
        let n_sq = (l[0] - l[2]).n.norm_sq();
        if n_sq == 0 {
            return 0.0;                 // Ewald interaction
        }
        self.coeff / (n_sq as f64)
    }

    pub fn op2_elem(&self, l: [HarmSpin; 4]) -> f64 {
        self.op2_prod_elem([l[0], l[1], l[2], l[3]])
            - self.op2_prod_elem([l[0], l[1], l[3], l[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::super::utils::Toler;
    use super::*;
    use vecn::Vec3I8;

    #[test]
    fn test_kinetic() {
        let nucleon_kinetic = IsoKinetic::nucleon_mev(1.0);
        let proton_orbital = HarmSpinIso {
            n: Vec3I8::new(1, 0, 0),
            s: Half(1),
            t: Half(1),
        };
        let neutron_orbital = HarmSpinIso {
            n: Vec3I8::new(0, -1, 0),
            s: Half(1),
            t: Half(-1),
        };
        toler_assert_eq!(Toler { relerr: 1e-8, abserr: 0.0 },
                         nucleon_kinetic.diag_op1_elem(proton_orbital),
                         819.16970);
        toler_assert_eq!(Toler { relerr: 1e-8, abserr: 0.0 },
                         nucleon_kinetic.diag_op1_elem(neutron_orbital),
                         818.04210);
    }

    #[test]
    fn test_coulomb() {
        let coulomb = Coulomb::natural(1.0, 1.0);
        let l1 = HarmSpin {
            n: Vec3I8::new(1, 0, 0),
            s: Half(1),
        };
        let l2 = HarmSpin {
            n: Vec3I8::new(0, 1, 0),
            s: Half(-1),
        };
        let l3 = HarmSpin {
            n: Vec3I8::new(2, -1, 0),
            s: Half(-1),
        };
        let l4 = HarmSpin {
            n: Vec3I8::new(-1, 2, 0),
            s: Half(1),
        };
        toler_assert_eq!(Toler { relerr: 1e-16, abserr: 0.0 },
                         coulomb.op2_elem([l1, l2, l3, l4]),
                         -coulomb.op2_elem([l2, l1, l3, l4]));
    }
}
