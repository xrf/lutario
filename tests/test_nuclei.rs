extern crate fnv;
#[macro_use]
extern crate lutario;
extern crate netlib_src;
extern crate rand;
extern crate rand_xorshift;

use fnv::FnvHashMap;
use lutario::basis::occ;
use lutario::j_scheme::{
    check_eq_mop_j012, new_mop_j012, op200_to_op211, rand_mop_j012, JAtlas, MopJ012,
};
use lutario::op::Op;
use lutario::utils::Toler;
use lutario::{hf, imsrg, nuclei, qdpt};

const TOLER: Toler = Toler {
    relerr: 1e-13,
    abserr: 1e-13,
};

#[derive(Clone, Debug)]
pub struct Results {
    pub de_dqdpt2: FnvHashMap<nuclei::Npjw, f64>,
    pub e_hf: f64,
    pub de_mp2: f64,
}

fn calc_j(
    nucleus: &nuclei::Nucleus,
    omega: f64,
    me2: &FnvHashMap<nuclei::JNpjwKey, f64>,
) -> Results {
    let atlas = JAtlas::new(&nucleus.basis());
    let scheme = atlas.scheme();
    let h1 = nuclei::make_ho3d_op_j(&atlas, omega);
    let h2 = nuclei::make_v_op_j(&atlas, me2);
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
        ..Default::default()
    }
    .make_run(&h1, &h2);
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
    nucleus: &nuclei::Nucleus,
    omega: f64,
    me2: &FnvHashMap<nuclei::JNpjwKey, f64>,
) -> Results {
    let j_atlas = JAtlas::new(&nucleus.basis());
    let atlas = JAtlas::new(&nucleus.m_basis());
    let scheme = atlas.scheme();
    let h1j = nuclei::make_ho3d_op_j(&j_atlas, omega);
    let h2j = nuclei::make_v_op_j(&j_atlas, me2);
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
        ..Default::default()
    }
    .make_run(&h1, &h2);
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
    let nucleus = nuclei::SimpleNucleus {
        e_max: 2,
        e_fermi_n: 1,
        e_fermi_p: 1,
        orbs: "",
    }
    .to_nucleus()
    .unwrap();
    let (_, me2) = nuclei::vrenorm::VintLoader {
        path: "data/cens-mbpt/vintnn3lohw24.dat".as_ref(),
        sp: "data/cens-mbpt/spox16.dat".as_ref(),
    }
    .load()
    .unwrap();
    let j_results = calc_j(&nucleus, omega, &me2);
    let m_results = calc_m(&nucleus, omega, &me2);
    for npjw in nucleus.states() {
        if !j_results.de_dqdpt2.contains_key(&npjw) && !m_results.de_dqdpt2.contains_key(&npjw) {
            continue;
        }
        let xj = *j_results.de_dqdpt2.get(&npjw).unwrap();
        let xm = *m_results.de_dqdpt2.get(&npjw).unwrap();
        toler_assert_eq!(TOLER, xj, xm);
    }
    toler_assert_eq!(TOLER, j_results.e_hf, m_results.e_hf);
    toler_assert_eq!(TOLER, j_results.de_mp2, m_results.de_mp2);
}

// cross-check commutator in J-scheme with M-scheme
#[test]
fn test_commut_nuclei() {
    use rand::SeedableRng;
    let mut rng = rand_xorshift::XorShiftRng::from_seed([
        0x54, 0x67, 0x3a, 0x19, 0x69, 0xd4, 0xa7, 0xa8, 0x05, 0x0e, 0x83, 0x97, 0xbb, 0xa7, 0x3b,
        0x11,
    ]);
    let toler = Toler {
        relerr: 1e-12,
        abserr: 1e-12,
    };
    let nucleus = nuclei::SimpleNucleus {
        e_max: 2,
        e_fermi_n: 1,
        e_fermi_p: 1,
        orbs: "",
    }
    .to_nucleus()
    .unwrap();
    let mut w6j_ctx = Default::default();
    let j_atlas = JAtlas::new(&nucleus.basis());
    let m_atlas = JAtlas::new(&nucleus.m_basis());
    let j_scheme = j_atlas.scheme();
    let m_scheme = m_atlas.scheme();
    let mop_j_to_m = |cj: &MopJ012<f64>| {
        (
            cj.0,
            nuclei::op1_j_to_m(&j_atlas, &m_atlas, &cj.1),
            nuclei::op2_j_to_m(&j_atlas, &m_atlas, &cj.2),
        )
    };
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
