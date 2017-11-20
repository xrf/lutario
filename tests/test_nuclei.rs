//! Integration tests for nuclei.

extern crate fnv;
#[macro_use]
extern crate lutario;
extern crate libblas;

use fnv::FnvHashMap;
use lutario::*;
use lutario::basis::*;
use lutario::j_scheme::*;
use lutario::nuclei::*;
use lutario::op::{DiagOp, Op};
use lutario::utils::Toler;

const TOLER: Toler = Toler { relerr: 1e-13, abserr: 1e-13 };

#[derive(Clone, Debug)]
pub struct Results {
    pub de_dqdpt2: FnvHashMap<Npjw, f64>,
    pub e_hf: f64,
    pub de_mp2: f64,
}

fn calc_j(
    nucleus: Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> Results {
    let atlas = JAtlas::new(nucleus.jpwn_orbs().into_iter());
    let scheme = &atlas.scheme;
    let h1 = make_ho3d_op_j(&atlas, omega);
    let h2 = make_v_op_j(&atlas, two_body_mat_elems);

    let mut r = DiagOp::new(BasisJ10(scheme));
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut de_dqdpt2 = FnvHashMap::default();
    for p in scheme.states_10(&occ::ALL1) {
        let npjw = Npjw::from(atlas.decode(p).unwrap());
        let value = r.at(p);
        if !value.is_finite() {
            continue;
        }
        assert!(de_dqdpt2.insert(npjw, value).is_none());
    }

    let mut hf = hf::HfConf {
        toler: TOLER,
        .. Default::default()
    }.new_run(h1, h2);
    hf.do_run().unwrap();
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    let mut h2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    hf::transform_h1(&hf.h1, &hf.dcoeff, &mut h1);
    hf::transform_h2(&hf.h2, &hf.dcoeff, &mut h2);
    let mut hn0 = 0.0;
    let mut hn1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    let mut hn2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    hf::normord(&h1, &h2, &mut hn0, &mut hn1, &mut hn2);
    let de_mp2 = qdpt::mp2(&hn1, &hn2);
    Results {
        de_dqdpt2,
        e_hf: hn0,
        de_mp2,
    }
}

fn calc_m(
    nucleus: Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<JNpjw2Pair, f64>,
) -> Results {
    let atlas = JAtlas::new(nucleus.pmwnj_orbs().into_iter());
    let scheme = &atlas.scheme;
    let h1 = make_ho3d_op_m(&atlas, omega);
    let h2 = make_v_op_m(&atlas, two_body_mat_elems);

    let mut r = DiagOp::new(BasisJ10(scheme));
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut de_dqdpt2 = FnvHashMap::default();
    for p in scheme.states_10(&occ::ALL1) {
        let npjmw = Npjmw::from(atlas.decode(p).unwrap());
        let npjw = Npjw::from(npjmw);
        let value = r.at(p);
        if !value.is_finite() {
            continue;
        }
        toler_assert_eq!(TOLER, *de_dqdpt2.entry(npjw).or_insert(value), value);
    }

    let mut hf = hf::HfConf {
        toler: TOLER,
        .. Default::default()
    }.new_run(h1, h2);
    hf.do_run().unwrap();
    let mut h1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    let mut h2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    hf::transform_h1(&hf.h1, &hf.dcoeff, &mut h1);
    hf::transform_h2(&hf.h2, &hf.dcoeff, &mut h2);
    let mut hn0 = 0.0;
    let mut hn1 = Op::new(BasisJ10(scheme), BasisJ10(scheme));
    let mut hn2 = Op::new(BasisJ20(scheme), BasisJ20(scheme));
    hf::normord(&h1, &h2, &mut hn0, &mut hn1, &mut hn2);
    let de_mp2 = qdpt::mp2(&hn1, &hn2);

    Results {
        de_dqdpt2,
        e_hf: hn0,
        de_mp2,
    }
}

#[test]
fn test_nuclei() {
    let omega = 24.0;
    let basis_spec = NucleonBasisSpec::with_e_max(2);
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
        if !j_results.de_dqdpt2.contains_key(&npjw)
            && !m_results.de_dqdpt2.contains_key(&npjw)
        {
            continue;
        }
        let xj = *j_results.de_dqdpt2.get(&npjw).unwrap();
        let xm = *m_results.de_dqdpt2.get(&npjw).unwrap();
        toler_assert_eq!(TOLER, xj, xm);
        println!("- {} {}", npjw, xj);
    }
    toler_assert_eq!(TOLER, j_results.e_hf, m_results.e_hf);
    toler_assert_eq!(TOLER, j_results.de_mp2, m_results.de_mp2);
}
