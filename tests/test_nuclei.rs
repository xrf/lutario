extern crate fnv;
#[macro_use]
extern crate lutario;
extern crate netlib_src;
extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde_yaml;

use std::fs::{self, File};
use std::io::Write;
use fnv::FnvHashMap;
use lutario::{hf, imsrg, nuclei, qdpt};
use lutario::basis::occ;
use lutario::j_scheme::{JAtlas, MopJ012, check_eq_mop_j012, new_mop_j012,
                        op200_to_op211, rand_mop_j012};
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
    let atlas = JAtlas::new(&nucleus.jpwn_orbs());
    let scheme = atlas.scheme();
    let h1 = nuclei::make_ho3d_op_j(&atlas, omega);
    let h2 = nuclei::make_v_op_j(&atlas, two_body_mat_elems);
    let mut w6j_ctx = Default::default();
    let mut h2p = Op::new(scheme.clone());
    op200_to_op211(&mut w6j_ctx, 1.0, &h2, &mut h2p);

    let mut r = Op::new_vec(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        r.add(p, p, qdpt::qdpt2_terms(&h1, &h2, p, p));
        r.add(p, p, qdpt::qdpt3_terms(&h1, &h2, &h2p, p, p));
    }
    let mut de_dqdpt2 = FnvHashMap::default();
    for p in scheme.states_10(&occ::ALL1) {
        let npjw = nuclei::Npjw::from(atlas.decode(p).unwrap());
        let value = r.at(p, p);
        if !value.is_finite() {
            continue;
        }
        assert!(de_dqdpt2.insert(npjw, value).is_none());
    }

    let mut hf = hf::Conf {
        toler: TOLER,
        .. Default::default()
    }.make_run(&h1, &h2);
    hf.do_run().unwrap();
    let mut hh = new_mop_j012(scheme);
    hf::transform_h1(&h1, &hf.dcoeff, &mut hh.1);
    hf::transform_h2(&h2, &hf.dcoeff, &mut hh.2);
    let mut hn = new_mop_j012(scheme);
    hf::normord(&hh, &mut hn);
    let de_mp2 = qdpt::mp2(&hn.1, &hn.2);
    Results {
        de_dqdpt2,
        e_hf: hn.0,
        de_mp2,
    }
}

fn calc_m(
    nucleus: nuclei::Nucleus,
    omega: f64,
    two_body_mat_elems: &FnvHashMap<nuclei::JNpjw2Pair, f64>,
) -> Results {
    let j_atlas = JAtlas::new(&nucleus.jpwn_orbs());
    let atlas = JAtlas::new(&nucleus.pmwnj_orbs());
    let scheme = atlas.scheme();
    let h1j = nuclei::make_ho3d_op_j(&j_atlas, omega);
    let h2j = nuclei::make_v_op_j(&j_atlas, two_body_mat_elems);
    let h1 = nuclei::op1_j_to_m(&j_atlas, &atlas, &h1j);
    let h2 = nuclei::op2_j_to_m(&j_atlas, &atlas, &h2j);
    let mut w6j_ctx = Default::default();
    let mut h2p = Op::new(scheme.clone());
    op200_to_op211(&mut w6j_ctx, 1.0, &h2, &mut h2p);

    let mut r = Op::new_vec(scheme.clone());
    for p in scheme.states_10(&occ::ALL1) {
        r.add(p, p, qdpt::qdpt2_terms(&h1, &h2, p, p));
        r.add(p, p, qdpt::qdpt3_terms(&h1, &h2, &h2p, p, p));
    }
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

    let mut hf = hf::Conf {
        toler: TOLER,
        .. Default::default()
    }.make_run(&h1, &h2);
    hf.do_run().unwrap();
    let mut hh = new_mop_j012(scheme);
    hf::transform_h1(&h1, &hf.dcoeff, &mut hh.1);
    hf::transform_h2(&h2, &hf.dcoeff, &mut hh.2);
    let mut hn = new_mop_j012(scheme);
    hf::normord(&hh, &mut hn);
    let de_mp2 = qdpt::mp2(&hn.1, &hn.2);

    Results {
        de_dqdpt2,
        e_hf: hn.0,
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

// cross-check commutator in J-scheme with M-scheme
#[test]
fn test_commut_nuclei() {
    use rand::SeedableRng;
    let mut rng = rand::XorShiftRng::from_seed([
        0x193a6754,
        0xa8a7d469,
        0x97830e05,
        0x113ba7bb,
    ]);
    let toler = Toler { relerr: 1e-12, abserr: 1e-12 };
    let e_max = 2;
    let basis_spec = nuclei::NucleonBasisSpec::with_e_max(e_max);
    let nucleus = nuclei::Nucleus {
        neutron_basis_spec: basis_spec,
        proton_basis_spec: basis_spec,
        e_fermi_neutron: 2,
        e_fermi_proton: 2,
    };
    let mut w6j_ctx = Default::default();
    let j_atlas = JAtlas::new(&nucleus.jpwn_orbs());
    let m_atlas = JAtlas::new(&nucleus.pmwnj_orbs());
    let j_scheme = j_atlas.scheme();
    let m_scheme = m_atlas.scheme();
    let mop_j_to_m = |cj: &MopJ012<f64>| (
        cj.0,
        nuclei::op1_j_to_m(&j_atlas, &m_atlas, &cj.1),
        nuclei::op2_j_to_m(&j_atlas, &m_atlas, &cj.2),
    );
    let aj = rand_mop_j012(j_scheme, &mut rng);
    let bj = rand_mop_j012(j_scheme, &mut rng);

    // commutator
    let mut cj = new_mop_j012(j_scheme);
    imsrg::commut(&mut w6j_ctx, 1.0, &aj, &bj, &mut cj);
    let am = mop_j_to_m(&aj);
    let bm = mop_j_to_m(&bj);
    let mut cm = new_mop_j012(m_scheme);
    imsrg::commut(&mut w6j_ctx, 1.0, &am, &bm, &mut cm);
    let cjm = mop_j_to_m(&cj);
    check_eq_mop_j012(toler, &cjm, &cm).unwrap();

    // White generator (can only compare Møller–Plesset denominators)
    let mut whj = new_mop_j012(j_scheme);
    imsrg::white_gen(imsrg::DenomType::MoellerPlesset, 1.0, &bj, &mut whj);
    let mut whm = new_mop_j012(m_scheme);
    imsrg::white_gen(imsrg::DenomType::MoellerPlesset, 1.0, &bm, &mut whm);
    let whjm = mop_j_to_m(&whj);
    check_eq_mop_j012(toler, &whjm, &whm).unwrap();
}
