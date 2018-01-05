#![feature(test)]

extern crate lutario;
extern crate netlib_src;
extern crate rand;
extern crate test;

use std::env;
use std::sync::Arc;
use lutario::{imsrg, nuclei};
use lutario::j_scheme::{JAtlas, JScheme, new_mop_j012, rand_mop_j012};
use lutario::op::VectorMut;
use rand::SeedableRng;

const RNG_SEED: [u32; 4] = [
    0x193a6754,
    0xa8a7d469,
    0x97830e05,
    0x113ba7bb,
];

fn scheme_o16() -> Arc<JScheme> {
    let trunc = nuclei::Ho3dTrunc {
        e_max: env::var("LT_EMAX")
            .map(|s| s.parse().unwrap())
            .unwrap_or(4),
        .. Default::default()
    };
    JAtlas::new(&nuclei::Nucleus {
        neutron_trunc: trunc,
        proton_trunc: trunc,
        e_fermi_neutron: 2,
        e_fermi_proton: 2,
    }.jpwn_orbs()).scheme().clone()
}

#[bench]
fn bench_commut_o16_011_022(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.0 = 0.0;
        imsrg::c011(alpha, &a.1, &b.1, &mut c.0);
        imsrg::c022(alpha, &a.2, &b.2, &mut c.0);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_111_112_121(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.1.set_zero();
        imsrg::c111(alpha, &a.1, &b.1, &mut c.1);
        imsrg::c112(alpha, &a.1, &b.2, &mut c.1);
        imsrg::c121(alpha, &a.2, &b.1, &mut c.1);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_122(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.1.set_zero();
        imsrg::c1220(alpha, &a.2, &b.2, &mut c.1);
        imsrg::c1221(alpha, &a.2, &b.2, &mut c.1);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_212_221(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.2.set_zero();
        imsrg::c212(alpha, &a.1, &b.2, &mut c.2);
        imsrg::c221(alpha, &a.2, &b.1, &mut c.2);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_2220_2222(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.2.set_zero();
        imsrg::c2220(alpha, &a.2, &b.2, &mut c.2);
        imsrg::c2222(alpha, &a.2, &b.2, &mut c.2);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_2221(bencher: &mut test::Bencher) {
    let mut rng = rand::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let alpha = test::black_box(-1.0);
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    let mut w6j_ctx = Default::default();
    // warm up the w6j_ctx
    imsrg::c2221(&mut w6j_ctx, alpha, &a.2, &b.2, &mut c.2);
    bencher.iter(|| {
        c.2.set_zero();
        imsrg::c2221(&mut w6j_ctx, alpha, &a.2, &b.2, &mut c.2);
        test::black_box(&mut c);
    });
}
