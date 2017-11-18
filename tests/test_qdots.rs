extern crate fnv;
#[macro_use]
extern crate lutario;

use std::fs::File;
use lutario::{hf, qdots, qdpt};
use lutario::j_scheme::{BasisJ10, BasisJ20, JAtlas};
use lutario::op::Op;
use lutario::utils::Toler;

const TOLER: Toler = Toler { relerr: 1e-10, abserr: 1e-10 };

#[derive(Clone, Copy, Debug)]
pub struct QdotTest {
    pub system: qdots::Qdot,
    pub omega: f64,
    pub e_hf: f64,
    pub de_mp2: f64,
}

impl QdotTest {
    fn run(self) {
        let v_elems = qdots::read_clh2_bin(
            &mut File::open("data/clh2-openfci/shells6.dat").unwrap(),
        ).unwrap();
        let atlas = JAtlas::new(self.system.parted_orbs().into_iter());
        let scheme = &atlas.scheme;
        let h1 = qdots::make_ho2d_op(&atlas, self.omega);
        let h2 = qdots::make_v_op(&atlas, &v_elems, self.omega);
        let mut hf = hf::HfConf {
            toler: TOLER,
            .. Default::default()
        }.new_run(h1, h2);
        hf.do_run();
        let mut h1 = Op::new(BasisJ10(&scheme), BasisJ10(&scheme));
        let mut h2 = Op::new(BasisJ20(&scheme), BasisJ20(&scheme));
        hf::transform_h1(&hf.h1, &hf.dcoeff, &mut h1);
        hf::transform_h2(&hf.h2, &hf.dcoeff, &mut h2);
        let mut hn0 = 0.0;
        let mut hn1 = Op::new(BasisJ10(&scheme), BasisJ10(&scheme));
        let mut hn2 = Op::new(BasisJ20(&scheme), BasisJ20(&scheme));
        hf::normord(&h1, &h2, &mut hn0, &mut hn1, &mut hn2);
        toler_assert_eq!(TOLER, hn0, self.e_hf);
        let de_mp2 = qdpt::mp2(&hn1, &hn2);
        toler_assert_eq!(TOLER, de_mp2, self.de_mp2);
    }
}

#[test]
fn test_qdots_3_2_1() {
    QdotTest {
        system: qdots::Qdot {
            num_shells: 3,
            num_filled: 2,
        },
        omega: 1.0,
        e_hf: 21.593198476284833,
        de_mp2: -0.15975791887897517,
    }.run()
}

#[test]
fn test_qdots_4_2_d28() {
    QdotTest {
        system: qdots::Qdot {
            num_shells: 4,
            num_filled: 2,
        },
        omega: 0.28,
        e_hf: 8.1397185533436094,
        de_mp2: -0.23476770967641344,
    }.run()
}
