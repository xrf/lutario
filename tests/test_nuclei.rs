//! Integration tests for nuclei.

extern crate fnv;
#[macro_use]
extern crate lutario;

use fnv::FnvHashMap;
use lutario::*;
use lutario::basis::*;
use lutario::j_scheme::*;
use lutario::nuclei::*;
use lutario::op::DiagOp;
use lutario::utils::Toler;

const TOLER: Toler = Toler { relerr: 1e-13, abserr: 1e-13 };

fn calc_j(
    nucleus: Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> FnvHashMap<Npjw, f64> {
    let atlas = JAtlas::new(nucleus.jpwn_orbs().into_iter());
    let h1 = make_ho3d_op_j(&atlas, omega);
    let h2 = make_v_op_j(&atlas, two_body_mat_elems);
    let mut r = DiagOp::new(BasisJ10(&atlas.scheme));
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut results = FnvHashMap::default();
    for p in atlas.scheme.states_10(&occ::ALL1) {
        let npjw = Npjw::from(atlas.decode(p).unwrap());
        let value = r.at(p);
        if !value.is_finite() {
            continue;
        }
        assert!(results.insert(npjw, value).is_none());
    }
    results
}

fn calc_m(
    nucleus: Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> FnvHashMap<Npjw, f64> {
    let atlas = JAtlas::new(nucleus.pmwnj_orbs().into_iter());
    let h1 = make_ho3d_op_m(&atlas, omega);
    let h2 = make_v_op_m(&atlas, two_body_mat_elems);
    let mut r = DiagOp::new(BasisJ10(&atlas.scheme));
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut results = FnvHashMap::default();
    for p in atlas.scheme.states_10(&occ::ALL1) {
        let npjmw = Npjmw::from(atlas.decode(p).unwrap());
        let npjw = Npjw::from(npjmw);
        let value = r.at(p);
        if !value.is_finite() {
            continue;
        }
        toler_assert_eq!(TOLER, *results.entry(npjw).or_insert(value), value);
    }
    results
}

#[test]
fn test_nuclei() {
    let omega = 24.0;
    let basis_spec = NucleonBasisSpec::with_e_max(3);
    let nucleus = Nucleus {
        neutron_basis_spec: basis_spec,
        proton_basis_spec: basis_spec,
        e_fermi_neutron: 2,
        e_fermi_proton: 2,
    };
    let two_body_mat_elems = morten_vint::LoadTwoBodyMatElems {
        sp_table_path: "data/cens-mbpt/spox16.dat".as_ref(),
        vint_table_path: "data/cens-mbpt/vintnn3lohw24.dat".as_ref(),
    }.call().unwrap();
    let j_results = calc_j(nucleus, omega, &two_body_mat_elems);
    let m_results = calc_m(nucleus, omega, &two_body_mat_elems);
    for npjw in nucleus.npjw_orbs() {
        if !j_results.contains_key(&npjw) && !m_results.contains_key(&npjw) {
            continue;
        }
        let xj = *j_results.get(&npjw).unwrap();
        let xm = *m_results.get(&npjw).unwrap();
        toler_assert_eq!(TOLER, xj, xm);
        println!("- {} {}", npjw, xj);
    }
}
