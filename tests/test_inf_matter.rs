#[macro_use]
extern crate lutario;
extern crate netlib_src;

use lutario::{hf, imsrg, inf_matter, minnesota, qdpt, plane_wave_basis, sg_ode};
use lutario::j_scheme::{JAtlas, new_mop_j012};
use lutario::utils::Toler;

#[test]
fn test_inf_matter() {
    let num_filled = 2;
    let system = plane_wave_basis::HarmTable::with_num_shells(4);
    let density = 0.08;
    let num_particles = system.num_states_to(num_filled) * 2;
    let box_len = (num_particles as f64 / density).powf(1.0 / 3.0);
    let atlas = JAtlas::new(&system.parted_ns_orbs(num_filled));
    let scheme = atlas.scheme();
    let h1 = inf_matter::Kinetic::neutron_mev(box_len).make_op(&atlas);
    let h2 = minnesota::MinnesotaBox::new(box_len).make_op_neutron(&atlas);

    let toler = Toler { relerr: 1e-8, abserr: 1e-8 };

    // test HF
    let mut hf = hf::Conf {
        toler: toler,
        .. Default::default()
    }.make_run(&h1, &h2);
    hf.do_run().unwrap();
    let mut hh = new_mop_j012(scheme);
    hf::transform_h1(&h1, &hf.dcoeff, &mut hh.1);
    hf::transform_h2(&h2, &hf.dcoeff, &mut hh.2);
    let mut hn = new_mop_j012(scheme);
    hf::normord(&hh, &mut hn);
    toler_assert_eq!(Toler { relerr: 1e-7, abserr: 1e-7 },
                     hn.0, 144.67192879294362);

    // test MP2
    let de_mp2 = qdpt::mp2(&hn.1, &hn.2);
    toler_assert_eq!(toler, de_mp2, -2.459104949285347);

    // test IMSRG
    let imsrg_toler = Toler { relerr: 1e-6, abserr: 1e-5 };
    let mut irun = imsrg::Conf {
        toler: imsrg_toler,
        solver_conf: sg_ode::Conf {
            toler: imsrg_toler,
            .. Default::default()
        },
        .. Default::default()
    }.make_run(&hn);
    irun.do_run().unwrap();
    println!("- imsrg: {}", irun.energy());
    toler_assert_eq!(imsrg_toler, irun.energy(), 142.14556953083158);
}
