// needed for error_chain
#![recursion_limit = "1024"]

extern crate any_key;
extern crate blas;
extern crate byteorder;
extern crate conv;
extern crate debugit;
extern crate flate2;
extern crate fnv;
#[macro_use]
extern crate error_chain;
extern crate num;
extern crate lapack;
#[allow(unused_extern_crates)]
#[cfg(feature="netlib-src")]
extern crate netlib_src;
#[macro_use]
extern crate quick_error;
extern crate stable_deref_trait;
extern crate take_mut;
extern crate xz2;

pub mod basis;
pub mod block_matrix;
pub mod block_tri_matrix;
pub mod btree_cache;
pub mod cache;
pub mod half;
pub mod io;
pub mod linalg;
pub mod matrix;
pub mod nuclei;
pub mod parity;
pub mod scheme;
pub mod tri_matrix;
pub mod utils;

error_chain! {
    types { Error, ErrorKind, ResultExt; }
}
