#![feature(test)]

extern crate lutario;
extern crate netlib_src;
extern crate rand;
extern crate rand_xorshift;
extern crate test;

use lutario::j_scheme::{new_mop_j012, rand_mop_j012, JAtlas, JScheme};
use lutario::op::VectorMut;
use lutario::{hf, imsrg, nuclei};
use rand::SeedableRng;
use std::env;
use std::sync::Arc;

const RNG_SEED: [u8; 16] = [
    0x54, 0x67, 0x3a, 0x19, 0x69, 0xd4, 0xa7, 0xa8, 0x05, 0x0e, 0x83, 0x97, 0xbb, 0xa7, 0x3b, 0x11,
];

fn scheme_o16() -> Arc<JScheme> {
    JAtlas::new(
        &nuclei::SimpleNucleus {
            e_max: env::var("LT_EMAX").map(|s| s.parse().unwrap()).unwrap_or(4),
            e_fermi_n: 1,
            e_fermi_p: 1,
            orbs: "",
        }
        .to_nucleus()
        .unwrap()
        .basis(),
    )
    .scheme()
    .clone()
}

#[bench]
fn bench_hf_transform_h2(bencher: &mut test::Bencher) {
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
    let scheme = &scheme_o16();
    let a = rand_mop_j012(scheme, &mut rng);
    let b = rand_mop_j012(scheme, &mut rng);
    let mut c = new_mop_j012(scheme);
    bencher.iter(|| {
        c.2.set_zero();
        hf::transform_h2(&b.2, &a.1, &mut c.2);
        test::black_box(&mut c);
    });
}

#[bench]
fn bench_commut_o16_011_022(bencher: &mut test::Bencher) {
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
    let mut rng = rand_xorshift::XorShiftRng::from_seed(RNG_SEED);
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
