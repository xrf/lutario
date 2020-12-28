extern crate clap;
extern crate lutario;
extern crate netlib_src;

use lutario::j_scheme::{new_mop_j012, op200_to_op211, JAtlas};
use lutario::op::{Op, VectorMut};
use lutario::utils::{RefAdd, Toler};
use lutario::{hf, imsrg, nuclei, qdpt, sg_ode};
use std::collections::HashMap;
use std::mem;

fn main() {
    let matches = clap::App::new(env!("CARGO_PKG_NAME"))
        .args_from_usage("--emax=<emax> 'Maximum shell index for calculation'")
        .args_from_usage("--efn=<emaxn> 'Maximum shell index of occupied neutrons'")
        .args_from_usage("--efp=<efp> 'Maximum shell index of occupied protons'")
        .args_from_usage("--orbs=[orbs] 'Include (-) or exclude (+) additional orbitals'")
        .args_from_usage("--input=<input> 'File containing input matrix elements (note: parameters will be guessed based on filename)'")
        .args_from_usage("--input-sp=[input-sp] 'File containing single-particle state table (CENS format only)'")
        .args_from_usage("--a=[a] 'Number of particles (for subtracting center-of-mass kinetic energy)'")
        .args_from_usage("--input-tpp=[input-tpp] 'File containing matrix elements of the two-body momentum product: −(a / fm)² ⟨…| p₁ ⋅ p₂ / (2 m MeV) |…⟩ where ‘a’ is the characteristic length'")
        .get_matches();

    let e_max = matches.value_of("emax").unwrap().parse().unwrap();
    println!("e_max: {}", e_max);
    let e_fermi_n: i32 = matches.value_of("efn").unwrap().parse().unwrap();
    println!("e_fermi_n: {}", e_fermi_n);
    let e_fermi_p: i32 = matches.value_of("efp").unwrap().parse().unwrap();
    println!("e_fermi_p: {}", e_fermi_p);
    let orbs = matches.value_of("orbs").unwrap_or("");
    println!("orbs: {}", orbs);
    let input = matches.value_of("input").unwrap();

    let nucleus = nuclei::SimpleNucleus {
        e_max,
        e_fermi_n,
        e_fermi_p,
        orbs,
    }
    .to_nucleus()
    .unwrap();

    let (omega, me2) = if let Some(sp) = matches.value_of_os("input-sp") {
        let (omega, me2) = nuclei::vrenorm::VintLoader {
            path: input.as_ref(),
            sp: sp.as_ref(),
        }
        .load()
        .unwrap();
        println!("input: {{path: {}, omega: {}}}", input, omega);
        (omega, me2)
    } else {
        let loader = nuclei::darmstadt::Me2jGuessLoader {
            path: input.as_ref(),
            ..Default::default()
        }
        .guess()
        .unwrap();
        println!("input: {}", loader);
        loader.load(e_max).unwrap()
    };

    let atlas = JAtlas::new(&nucleus.basis());
    let scheme = atlas.scheme();
    let mut h1 = nuclei::make_ke_op_j(&atlas, omega);
    let mut h2 = nuclei::make_v_op_j(&atlas, &me2);

    // subtract center-of-mass kinetic energy
    if let Some(mass_num) = matches.value_of("a") {
        let mass_num: i32 = mass_num.parse().unwrap();
        h1.scale(&(1.0 - 1.0 / mass_num as f64));
        if let Some(input_tpp) = matches.value_of("input-tpp") {
            let loader = nuclei::darmstadt::Me2jGuessLoader {
                path: input_tpp.as_ref(),
                omega: Some(1.0),
                ..Default::default()
            }
            .guess()
            .unwrap();
            println!("input_tpp: {}", loader);
            let (_, me_tpp) = loader.load(e_max).unwrap();
            let mut tpp = nuclei::make_v_op_j(&atlas, &me_tpp);
            tpp.scale(&(2.0 * nuclei::darmstadt::TPP_FACTOR * omega / mass_num as f64));
            h2 = h2.ref_add(&tpp);
        } else {
            panic!("--a was provided but not --input-tpp?");
        }
    } else if let Some(_) = matches.value_of("input-tpp") {
        panic!("--input-tpp was provided but not --a?");
    }

    let mut hh;
    {
        let mut hrun = hf::Conf {
            toler: Toler {
                relerr: 1e-13,
                abserr: 1e-13,
            },
            ..Default::default()
        }
        .make_run(&h1, &h2);
        hrun.do_run().unwrap();
        println!("# transforming matrices into HF basis...");
        hh = new_mop_j012(scheme);
        hf::transform_h1(&h1, &hrun.dcoeff, &mut hh.1);
        hf::transform_h2(&h2, &hrun.dcoeff, &mut hh.2);
    }
    mem::drop(h1);
    mem::drop(h2);
    println!("# normal ordering...");
    let mut hn = new_mop_j012(scheme);
    hf::normord(&hh, &mut hn);
    mem::drop(hh);
    println!("hf_energy: {}", hn.0);

    let de_mp2 = qdpt::mp2(&hn.1, &hn.2);
    println!("mp2_correction: {}", de_mp2);

    let imsrg_toler = Toler {
        relerr: 1e-7,
        abserr: 1e-7,
    };
    let mut irun = imsrg::Conf {
        toler: imsrg_toler,
        solver_conf: sg_ode::Conf {
            toler: imsrg_toler,
            ..Default::default()
        },
        ..Default::default()
    }
    .make_run(&hn);
    mem::drop(hn);
    irun.do_run().unwrap();
    let hi = irun.hamil();
    mem::drop(irun);
    println!("imsrg_energy: {}", hi.0);

    println!("orbital_energies:");
    let mut de_dqdpt2 = HashMap::new();
    let mut de_dqdpt3 = HashMap::new();
    let mut hi2p = Op::new(scheme.clone());
    let mut w6j_ctx = Default::default();
    op200_to_op211(&mut w6j_ctx, 1.0, &hi.2, &mut hi2p);
    for sp in nucleus.states() {
        let npjw = nuclei::Npjw::from(sp);
        let p = atlas.encode(&sp.into()).unwrap();
        let ep = hi.1.at(p, p);
        let dep2 = qdpt::qdpt2_terms(&hi.1, &hi.2, p, p);
        let dep3 = qdpt::qdpt3_terms(&hi.1, &hi.2, &hi2p, p, p);
        de_dqdpt2.insert(npjw, dep2);
        de_dqdpt3.insert(npjw, dep3);
        println!("  '{}':", npjw);
        println!("    imsrg:       {}", ep);
        println!("    imsrg+qdpt2: {}", ep + dep2);
        println!("    imsrg+qdpt3: {}", ep + dep2 + dep3);
    }
}
