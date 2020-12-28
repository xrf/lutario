//! Lutario is an implementation of IM-SRG(2) and QDPT3.
//!
//! The best way to get started is to look at the tests and examples.  The
//! code is always in flux, but the test and examples within the codebase are
//! guaranteed to stay up to date thanks to automated testing.
//!
//! The code is currently organized in a very monolithic manner to enable
//! rapid development and prototyping.  When the various submodules stabilize
//! and become sufficiently generic, they may eventually graduate to become
//! their own separate crates, which would help reduce bloat as well as
//! compilation times.

extern crate any_key;
extern crate bincode;
extern crate byteorder;
extern crate cblas;
extern crate conv;
extern crate flate2;
extern crate fnv;
extern crate lapacke;
extern crate num;
#[macro_use]
extern crate lazy_static;
extern crate libc;
#[cfg(test)]
extern crate netlib_src;
#[macro_use]
extern crate quick_error;
extern crate rand;
extern crate rand_distr;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate siphasher;
extern crate stable_deref_trait;
extern crate take_mut;
extern crate wigner_symbols;
extern crate xz2;

#[macro_use]
mod macros;

pub mod ang_mom;
pub mod basis;
pub mod block;
pub mod block_mat;
pub mod block_tri_mat;
pub mod btree_cache;
pub mod cache;
pub mod cache2;
pub mod half;
pub mod hf;
pub mod imsrg;
pub mod inf_matter;
pub mod io;
pub mod isqrt;
pub mod j_scheme;
pub mod linalg;
pub mod mat;
pub mod minnesota;
pub mod nuclei;
pub mod op;
pub mod parity;
pub mod phys_consts;
pub mod plane_wave_basis;
pub mod qdots;
pub mod qdpt;
pub mod sg_ode;
pub mod tri_mat;
pub mod utils;
pub mod vecn;
pub mod vector_driver;
