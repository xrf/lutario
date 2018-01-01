extern crate fnv;
#[macro_use]
extern crate lutario;
extern crate netlib_src;

use std::io::{self, BufReader};
use std::fs::File;
use lutario::{hf, imsrg, qdots, qdpt, sg_ode};
use lutario::basis::OrbIx;
use lutario::j_scheme::{JAtlas, new_mop_j012, read_mop_j012_txt,
                        check_eq_op_j100, check_eq_op_j200, check_eq_mop_j012,
                        op200_to_op211};
use lutario::op::Op;
use lutario::utils::Toler;

fn open(path: &str) -> io::Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path)?))
}

#[derive(Clone, Copy, Debug)]
pub struct QdotTest<'a> {
    pub system: qdots::Qdot,
    pub omega: f64,
    pub e_hf: f64,
    pub de_mp2: f64,
    pub e_imsrg: Option<f64>,
    pub qdpt_file: Option<&'a str>,
}

impl<'a> QdotTest<'a> {
    fn run(self) {
        use std::io::BufRead;

        let toler = Toler { relerr: 1e-8, abserr: 1e-8 };
        let v_elems = qdots::read_clh2_bin(
            &mut File::open("data/clh2-openfci/shells6.dat").unwrap(),
        ).unwrap();
        let atlas = JAtlas::new(&self.system.parted_orbs());
        let scheme = atlas.scheme();
        let h1 = qdots::make_ho2d_op(&atlas, self.omega);
        let h2 = qdots::make_v_op(&atlas, &v_elems, self.omega);

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
        toler_assert_eq!(toler, hn.0, self.e_hf);

        // test MP2
        let de_mp2 = qdpt::mp2(&hn.1, &hn.2);
        toler_assert_eq!(toler, de_mp2, self.de_mp2);

        // test QDPT
        if let Some(qdpt_file) = self.qdpt_file {
            let mut w6j_ctx = Default::default();
            let mut hn2p = Op::new(scheme.clone());
            op200_to_op211(&mut w6j_ctx, 1.0, &hn.2, &mut hn2p);
            for line in open(&qdpt_file).unwrap().lines() {
                let line = line.unwrap();
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let fields: Vec<_> = line.split_whitespace().collect();
                let orbital = OrbIx(fields[0].parse().unwrap());
                let term = fields[1].parse().unwrap();
                let energy = fields[2].parse().unwrap();
                let p = scheme.state_10(
                    scheme.basis_10.encode(
                        scheme.basis_10.orb_from_ix(orbital),
                    ),
                );
                let de = qdpt::qdpt_term(term, &hn.1, &hn.2, Some(&hn2p), p, p);
                toler_assert_eq!(toler, energy, de);
            }
        }

        // test IMSRG
        if let Some(e_imsrg) = self.e_imsrg {
            let imsrg_toler = Toler { relerr: 1e-6, abserr: 1e-5 };
            let mut irun = imsrg::Conf {
                toler: imsrg_toler,
                solver_conf: sg_ode::Conf {
                    toler: imsrg_toler,
                    .. Default::default()
                },
                .. Default::default()
            }.make_run(hn);
            irun.do_run().unwrap();
            toler_assert_eq!(imsrg_toler, irun.energy(), e_imsrg);
        }
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
        e_imsrg: Some(21.412110),
        qdpt_file: Some("data/clh2-openfci/qdpt_3_2_1.txt"),
    }.run()
}

#[test]
fn test_qdots_4_2_d28() {
    // check a case where ω ≠ 1
    QdotTest {
        system: qdots::Qdot {
            num_shells: 4,
            num_filled: 2,
        },
        omega: 0.28,
        e_hf: 8.1397185533436094,
        de_mp2: -0.23476770967641344,
        e_imsrg: Some(7.839145241377399),
        qdpt_file: Some("data/clh2-openfci/qdpt_4_2_d28.txt"),
    }.run()
}

// this test is sensitive to the asymmetric nature of HF coefficient matrix;
// i.e. so I don't mix up left and right indices during HF transform
const QDOT_TEST_5_3_1: QdotTest = QdotTest {
    system: qdots::Qdot {
        num_shells: 5,
        num_filled: 3,
    },
    omega: 1.0,
    e_hf: 67.569930237865776,
    de_mp2: -0.5483183431301903,
    e_imsrg: Some(67.005660),
    qdpt_file: Some("data/clh2-openfci/qdpt_5_3_1.txt"),
};

#[test]
fn test_qdots_5_3_1_hf_only() {
    QdotTest {
        e_imsrg: None,
        .. QDOT_TEST_5_3_1
    }.run()
}

#[test]
fn slowtest_qdots_5_3_1_full() {
    QdotTest {
        qdpt_file: None,
        .. QDOT_TEST_5_3_1
    }.run()
}

// Test the commutator in the quantum dots basis using random matrix elements.
#[test]
fn test_commut_qdots() {
    let toler = Toler { relerr: 1e-14, abserr: 1e-14 };
    let system = qdots::Qdot {
        num_shells: 3,
        num_filled: 2,
    };
    let atlas = JAtlas::new(&system.parted_orbs());
    let scheme = atlas.scheme();
    let load = |s| read_mop_j012_txt(scheme, &mut open(s).unwrap()).unwrap();
    let mut w6j_ctx = Default::default();
    let a = load("data/lutra/a.txt");
    let b = load("data/lutra/b.txt");

    // commutator terms
    {
        let c0 = load("data/lutra/c011.txt").0;
        let mut r0 = 0.0;
        imsrg::c011(1.0, &a.1, &b.1, &mut r0);
        imsrg::c011(-1.0, &b.1, &a.1, &mut r0);
        toler_assert_eq!(toler, c0, r0);
    }
    {
        let c0 = load("data/lutra/c022.txt").0;
        let mut r0 = 0.0;
        imsrg::c022(1.0, &a.2, &b.2, &mut r0);
        imsrg::c022(-1.0, &b.2, &a.2, &mut r0);
        toler_assert_eq!(toler, c0, r0);
    }
    {
        let c1 = load("data/lutra/c111.txt").1;
        let mut r1 = Op::new(scheme.clone());
        imsrg::c111(1.0, &a.1, &b.1, &mut r1);
        imsrg::c111(-1.0, &b.1, &a.1, &mut r1);
        check_eq_op_j100(toler, &c1, &r1).unwrap();
    }
    {
        let c1 = load("data/lutra/c112_c121.txt").1;
        let mut r1 = Op::new(scheme.clone());
        imsrg::c112(1.0, &a.1, &b.2, &mut r1);
        imsrg::c121(1.0, &a.2, &b.1, &mut r1);
        imsrg::c112(-1.0, &b.1, &a.2, &mut r1);
        imsrg::c121(-1.0, &b.2, &a.1, &mut r1);
        check_eq_op_j100(toler, &c1, &r1).unwrap();
    }
    {
        let c1 = load("data/lutra/c1220.txt").1;
        let mut r1 = Op::new(scheme.clone());
        imsrg::c1220(1.0, &a.2, &b.2, &mut r1);
        imsrg::c1220(-1.0, &b.2, &a.2, &mut r1);
        check_eq_op_j100(toler, &c1, &r1).unwrap();
    }
    {
        let c1 = load("data/lutra/c1221.txt").1;
        let mut r1 = Op::new(scheme.clone());
        imsrg::c1221(1.0, &a.2, &b.2, &mut r1);
        imsrg::c1221(-1.0, &b.2, &a.2, &mut r1);
        check_eq_op_j100(toler, &c1, &r1).unwrap();
    }
    {
        let c2 = load("data/lutra/c212_c221.txt").2;
        let mut r2 = Op::new(scheme.clone());
        imsrg::c212(1.0, &a.1, &b.2, &mut r2);
        imsrg::c221(1.0, &a.2, &b.1, &mut r2);
        imsrg::c212(-1.0, &b.1, &a.2, &mut r2);
        imsrg::c221(-1.0, &b.2, &a.1, &mut r2);
        check_eq_op_j200(toler, &c2, &r2).unwrap();
    }
    {
        let c2 = load("data/lutra/c2220.txt").2;
        let mut r2 = Op::new(scheme.clone());
        imsrg::c2220(1.0, &a.2, &b.2, &mut r2);
        imsrg::c2220(-1.0, &b.2, &a.2, &mut r2);
        check_eq_op_j200(toler, &c2, &r2).unwrap();
    }
    {
        let c2 = load("data/lutra/c2221.txt").2;
        let mut r2 = Op::new(scheme.clone());
        imsrg::c2221(&mut w6j_ctx, 1.0, &a.2, &b.2, &mut r2);
        imsrg::c2221(&mut w6j_ctx, -1.0, &b.2, &a.2, &mut r2);
        check_eq_op_j200(toler, &c2, &r2).unwrap();
    }
    {
        let c2 = load("data/lutra/c2222.txt").2;
        let mut r2 = Op::new(scheme.clone());
        imsrg::c2222(1.0, &a.2, &b.2, &mut r2);
        imsrg::c2222(-1.0, &b.2, &a.2, &mut r2);
        check_eq_op_j200(toler, &c2, &r2).unwrap();
    }
    {
        let c = load("data/lutra/c.txt");
        let mut r = new_mop_j012(scheme);
        imsrg::commut(&mut w6j_ctx, 1.0, &a, &b, &mut r);
        check_eq_mop_j012(toler, &c, &r).unwrap();
    }

    // White generator
    {
        let wh = load("data/lutra/wh.txt");
        let mut r = new_mop_j012(scheme);
        imsrg::white_gen(imsrg::DenomType::EpsteinNesbet, 1.0, &b, &mut r);
        check_eq_mop_j012(toler, &wh, &r).unwrap();
    }

    // normal ordering
    {
        let no = load("data/lutra/no.txt");
        let mut r = new_mop_j012(scheme);
        hf::normord(&b, &mut r);
        check_eq_mop_j012(toler, &no, &r).unwrap();
    }
}
