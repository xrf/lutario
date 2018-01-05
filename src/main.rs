extern crate clap;
extern crate lutario;
extern crate netlib_src;

use std::mem;
use std::collections::HashMap;
use lutario::{hf, imsrg, nuclei, qdpt, sg_ode};
use lutario::j_scheme::{JAtlas, new_mop_j012, op200_to_op211};
use lutario::op::Op;
use lutario::utils::Toler;

fn main() {
    let matches = clap::App::new(env!("CARGO_PKG_NAME"))
        .args_from_usage("--emax=<emax> 'Maximum shell index for calculation'")
        .args_from_usage("--efn=<efn> 'Number of neutron shells'")
        .args_from_usage("--efp=<efp> 'Number of proton shells'")
        .args_from_usage("--omega=<omega> 'Frequency of HO2D basis in energy units'")
        .args_from_usage("--input=<input> 'File containing input matrix elements'")
        .args_from_usage("--input-sp=<input-sp> 'File containing single-particle state table (CENS format only)'")
        .group(clap::ArgGroup::with_name("input-params")
               .args(&["input-sp", "input-emax"])
               .required(true))
        .args_from_usage("--input-emax=<input-emax> 'Maximum shell index in input file (ME2J format only)'")
        .get_matches();
    let input = matches.value_of_os("input").unwrap().as_ref();
    let omega = matches.value_of("omega").unwrap().parse().unwrap();
    let trunc = nuclei::Ho3dTrunc {
        e_max: matches.value_of("emax").unwrap().parse().unwrap(),
        .. Default::default()
    };
    let nucleus = nuclei::Nucleus {
        neutron_trunc: trunc,
        proton_trunc: trunc,
        e_fermi_neutron: matches.value_of("efn").unwrap().parse().unwrap(),
        e_fermi_proton: matches.value_of("efp").unwrap().parse().unwrap(),
    };
    let me2 = if let Some(e_max) = matches.value_of("input-emax") {
        let e_max = e_max.parse().unwrap();
        nuclei::darmstadt::Me2jLoader {
            path: input,
            e_max,
            .. Default::default()
        }.load(trunc.e_max).unwrap()
    } else {
        let sp = matches.value_of_os("input-sp").unwrap().as_ref();
        nuclei::vrenorm::VintLoader { path: input, sp }.load().unwrap()
    };

    let atlas = JAtlas::new(&nucleus.jpwn_orbs());
    let scheme = atlas.scheme();
    let h1 = nuclei::make_ke_op_j(&atlas, omega);
    let h2 = nuclei::make_v_op_j(&atlas, &me2);
    let mut w6j_ctx = Default::default();
    let mut h2p = Op::new(scheme.clone());
    op200_to_op211(&mut w6j_ctx, 1.0, &h2, &mut h2p);

    let mut hh;
    {
        let mut hrun = hf::Conf {
            toler: Toler { relerr: 1e-13, abserr: 1e-13 },
            .. Default::default()
        }.make_run(&h1, &h2);
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

    let imsrg_toler = Toler { relerr: 1e-7, abserr: 1e-7 };
    let mut irun = imsrg::Conf {
        toler: imsrg_toler,
        solver_conf: sg_ode::Conf {
            toler: imsrg_toler,
            .. Default::default()
        },
        .. Default::default()
    }.make_run(&hn);
    mem::drop(hn);
    irun.do_run().unwrap();
    let hi = irun.hamil();
    mem::drop(irun);
    println!("imsrg_energy: {}", hi.0);

    println!("orbital_energies:");
    let mut de_dqdpt2 = HashMap::new();
    let mut de_dqdpt3 = HashMap::new();
    let mut hi2p = Op::new(scheme.clone());
    op200_to_op211(&mut w6j_ctx, 1.0, &hi.2, &mut hi2p);
    for ip in nucleus.jpwn_orbs() {
        let npjw = nuclei::Npjw::from(ip.p);
        let p = atlas.encode(&ip.p).unwrap();
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
