use std::cmp::min;
use num::range_step_inclusive;
use super::half::Half;

#[derive(Clone, Copy, Debug)]
pub struct Nlj {
    pub n: u32,
    pub l: u32,
    pub j: Half<u32>,
}

impl Nlj {
    pub fn shell_index(self) -> u32 {
        2 * self.n + self.l
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NljBasisSpec {
    pub k_max: u32,
    pub n_max: u32,
    pub l_max: u32,
}

impl NljBasisSpec {
    pub fn with_k_max(k_max: u32) -> Self {
        Self {
            k_max,
            n_max: u32::max_value(),
            l_max: u32::max_value(),
        }
    }

    pub fn foreach<F: FnMut(Nlj)>(self, mut f: F) {
        for k in 0 .. self.k_max + 1 {
            for l in range_step_inclusive(k % 2, min(k, self.l_max), 2) {
                let n = (k - l) / 2;
                if n > self.n_max {
                    continue;
                }
                for j in Half::tri_range(l.into(), Half(1)) {
                    f(Nlj { n, l, j });
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitalBasisConf {
    pub num_shells: u32,
    pub num_filled: u32,
}

pub struct OrbitalBasis {
    pub orbitals: Box<[(Nlj, Half<i8>)]>,
    pub num_occupied: usize,
}

impl OrbitalBasisConf {
    pub fn get_orbital_basis(self) -> OrbitalBasis {
        let mut orbitals = Vec::new();
        NljBasisSpec::with_k_max(self.num_shells).foreach(|nlj| {
            orbitals.push((nlj, Half(-1)));
            orbitals.push((nlj, Half(1)));
        });
        let num_occupied = orbitals.iter()
            .position(|&(nlj, _)| nlj.shell_index() >= self.num_filled)
            .unwrap_or(orbitals.len());
        OrbitalBasis {
            orbitals: orbitals.into_boxed_slice(),
            num_occupied,
        }
    }
}
