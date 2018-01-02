# `lutario` [![Build status](https://travis-ci.org/xrf/lutario.svg)](https://travis-ci.org/xrf/lutario)

## External dependencies

A CBLAS and LAPACKE [implementation package](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki/Usage-of-the-BLAS%E2%80%93LAPACK-family-of-packages#implementations) is required to build and run any program that depends on `lutario`.

For the tests and example programs bundled with `lutario`, `netlib-src` is used as the implementation package.  This is configured to link to an *external* CBLAS and LAPACKE implementation, supplied by libraries named `cblas` and `lapacke` respectively.  If your external libraries are not named `cblas` and/or `lapacke`, you will need to set up symbolic links (or, alternatively, linker scripts) to redirect `cblas` and/or `lapacke` to the library(s) of your choice.  Alternatively, you could modify this package directly and either tweak the `netlib-src` feature flags in `Cargo.toml` or switch to a different implementation package entirely.

On the other hand, if you are building a program yourself that depends on `lutario`, then you are free to choose whatever implementation package you want.  (Don't forget to add the `extern crate` statement.)  You can even write your own implementation package to link to an arbitrary external library of your choice:

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

  - [Usage of the BLAS–LAPACK family of packages](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki/Usage-of-the-BLAS%E2%80%93LAPACK-family-of-packages)
  - [Choosing features of libraries](http://doc.crates.io/specifying-dependencies.html#choosing-features)
  - [Choosing features of end products](http://doc.crates.io/manifest.html#usage-in-end-products)
