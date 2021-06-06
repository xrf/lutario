![icon](tools/icon.svg)

# `lutario`

[![Build status](https://github.com/xrf/lutario/actions/workflows/build.yml/badge.svg)](https://github.com/xrf/lutario/actions/workflows/build.yml)

**Quick links**: [Documentation for `master` branch](https://xrf.github.io/lutario)

`lutario` is an implementation of the [in-medium similarity renormalization group (IM-SRG) method](https://arxiv.org/abs/1512.06956) and quasidegenerate perturbation theory (QDPT) for calculation of addition and removal energies of various quantums systems, including circular quantum dots, atomic nuclei, infinite nuclear matter, and homogeneous electron gas.

## Building

### Prerequisites

- [Rust](https://rust-lang.org) compiler with its Cargo package manager
  - For ordinary uses, any recent version from the stable channel should work.
  - To run benchmarks, you will need a version from the nightly channel.
  - Known to work with:
    - rustc 1.48.0 (7eac88abb 2020-11-16)
    - rustc 1.51.0-nightly (257becbfe 2020-12-27)
- CBLAS and LAPACKE libraries: Refer to
  [External dependencies](#external-dependencies) for details.
  - Known to work with OpenBLAS 0.3.13.
- [sg-ode](https://github.com/xrf/sg-ode) library: Refer to its README for details.

### Steps

 1. Download a copy of the source code.

 2. Run `cargo build --release` and wait a few minutes.  The `--release` flag
    enables optimizations.

 3. Try out the binary.  Use `target/release/lutario --help` to list all the
    command-line flags.

You can also use `cargo test --release` to run the tests, or `cargo bench`
(requires a nightly build of rust) to run the benchmarks.

## External dependencies

A CBLAS and LAPACKE [implementation package](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki#sources) is required to build and run any program that depends on `lutario`.

For the tests and example programs bundled with `lutario`, `netlib-src` is used as the implementation package.  This is configured to link to an *external* CBLAS and LAPACKE implementation, supplied by libraries named `cblas` and `lapacke` respectively.  If your external libraries are not named `cblas` and/or `lapacke`, you will need to set up symbolic links (or, alternatively, linker scripts) to redirect `cblas` and/or `lapacke` to the library(s) of your choice.  Alternatively, you could modify this package directly and either tweak the `netlib-src` feature flags in `Cargo.toml` or switch to a different implementation package entirely.

On the other hand, if you are building a program yourself that depends on `lutario`, then you are free to choose whatever implementation package you want.  You can even write your own implementation package to link to an arbitrary external library of your choice:

~~~toml
# my_program/my_blas_impl/Cargo.toml
[package]
name = "my_blas_impl"
version = "0.1.0"
build = "build.rs"
links = "blas"
~~~

~~~rust
// my_program/my_blas_impl/build.rs
fn main() {
    println!("cargo:rustc-link-lib=dylib=my_external_cblas_lib");
    println!("cargo:rustc-link-lib=dylib=my_external_lapacke_lib");
}
~~~

~~~rust
// my_program/my_blas_impl/src/lib.rs
// (empty file)
~~~

~~~toml
# append this to my_program/Cargo.toml
[dependencies.my_blas_impl]
path = "my_blas_impl"
~~~

For more information, see:

  - [blas-lapack-rs documentation](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki)
  - [Choosing features of libraries](http://doc.crates.io/specifying-dependencies.html#choosing-features)
  - [Choosing features of end products](http://doc.crates.io/manifest.html#usage-in-end-products)
