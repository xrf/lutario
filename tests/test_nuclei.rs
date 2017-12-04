#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;
extern crate fnv;
#[macro_use]
extern crate lutario;
extern crate netlib_src;

use std::fs::{self, File};
use std::io::Write;
use fnv::FnvHashMap;
use lutario::{hf, nuclei, qdpt};
use lutario::basis::occ;
use lutario::j_scheme::JAtlas;
use lutario::op::Op;
use lutario::utils::Toler;

const TOLER: Toler = Toler { relerr: 1e-13, abserr: 1e-13 };

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Results {
    pub de_dqdpt2: FnvHashMap<nuclei::Npjw, f64>,
    pub e_hf: f64,
    pub de_mp2: f64,
}

fn calc_j(
    nucleus: nuclei::Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<nuclei::JNpjw2Pair, f64>,
) -> Results {
    let atlas = JAtlas::new(&mut nucleus.jpwn_orbs().into_iter());
    let scheme = atlas.scheme();
    let h1 = nuclei::make_ho3d_op_j(&atlas, omega);
    let h2 = nuclei::make_v_op_j(&atlas, two_body_mat_elems);

    let mut r = Op::new_vec(scheme.clone());
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut de_dqdpt2 = FnvHashMap::default();
    for p in scheme.states_10(&occ::ALL1) {
        let npjw = nuclei::Npjw::from(atlas.decode(p).unwrap());
        let value = r.at(p, p);
        if !value.is_finite() {
            continue;
        }
        assert!(de_dqdpt2.insert(npjw, value).is_none());
    }

    let mut hf = hf::HfConf {
        toler: TOLER,
        .. Default::default()
    }.new_run(&h1, &h2);
    hf.do_run().unwrap();
    let mut hh1 = Op::new(scheme.clone());
    let mut hh2 = Op::new(scheme.clone());
    hf::transform_h1(&h1, &hf.dcoeff, &mut hh1);
    hf::transform_h2(&h2, &hf.dcoeff, &mut hh2);
    let mut hn0 = 0.0;
    let mut hn1 = Op::new(scheme.clone());
    let mut hn2 = Op::new(scheme.clone());
    hf::normord(&hh1, &hh2, &mut hn0, &mut hn1, &mut hn2);
    let de_mp2 = qdpt::mp2(&hn1, &hn2);
    Results {
        de_dqdpt2,
        e_hf: hn0,
        de_mp2,
    }
}

fn calc_m(
    nucleus: nuclei::Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<nuclei::JNpjw2Pair, f64>,
) -> Results {
    let atlas = JAtlas::new(&mut nucleus.pmwnj_orbs().into_iter());
    let scheme = atlas.scheme();
    let h1 = nuclei::make_ho3d_op_m(&atlas, omega);
    let h2 = nuclei::make_v_op_m(&atlas, two_body_mat_elems);

    let mut r = Op::new_vec(scheme.clone());
    qdpt::dqdpt2_term3(&h1, &h2, &mut r);
    qdpt::dqdpt2_term4(&h1, &h2, &mut r);
    let mut de_dqdpt2 = FnvHashMap::default();
    for p in scheme.states_10(&occ::ALL1) {
        let npjmw = nuclei::Npjmw::from(atlas.decode(p).unwrap());
        let npjw = nuclei::Npjw::from(npjmw);
        let value = r.at(p, p);
        if !value.is_finite() {
            continue;
        }
        toler_assert_eq!(TOLER, *de_dqdpt2.entry(npjw).or_insert(value), value);
    }

    let mut hf = hf::HfConf {
        toler: TOLER,
        .. Default::default()
    }.new_run(&h1, &h2);
    hf.do_run().unwrap();
    let mut hh1 = Op::new(scheme.clone());
    let mut hh2 = Op::new(scheme.clone());
    hf::transform_h1(&h1, &hf.dcoeff, &mut hh1);
    hf::transform_h2(&h2, &hf.dcoeff, &mut hh2);
    let mut hn0 = 0.0;
    let mut hn1 = Op::new(scheme.clone());
    let mut hn2 = Op::new(scheme.clone());
    hf::normord(&hh1, &hh2, &mut hn0, &mut hn1, &mut hn2);
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
    let e_max = 2;
    let basis_spec = nuclei::NucleonBasisSpec::with_e_max(e_max);
    let nucleus = nuclei::Nucleus {
        neutron_basis_spec: basis_spec,
        proton_basis_spec: basis_spec,
        e_fermi_neutron: 2,
        e_fermi_proton: 2,
    };
    let two_body_mat_elems = nuclei::morten_vint::LoadTwoBodyMatElems {
        sp_table_path: "data/cens-mbpt/spox16.dat".as_ref(),
        vint_table_path: "data/cens-mbpt/vintnn3lohw24.dat".as_ref(),
    }.call().unwrap();
    let suffix = format!("ho-n3lo_omega={}_emax={}_en={}_ep={}.txt",
                         omega,
                         e_max,
                         nucleus.e_fermi_neutron,
                         nucleus.e_fermi_proton);
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
    }
    toler_assert_eq!(TOLER, j_results.e_hf, m_results.e_hf);
    toler_assert_eq!(TOLER, j_results.de_mp2, m_results.de_mp2);
    fs::create_dir_all("out").unwrap();
    let mut f = File::create(&format!("out/test_nuclei_{}", suffix)).unwrap();
    serde_yaml::to_writer(&mut f, &j_results).unwrap();
    writeln!(f, "").unwrap();
}
