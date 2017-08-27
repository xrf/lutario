extern crate blas;
extern crate conv;
extern crate num;
extern crate lapack;
#[cfg(feature="netlib-src")]
extern crate netlib_src;

pub mod matrix;
pub mod parity;
pub mod utils;
