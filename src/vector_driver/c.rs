//! Vector driver interface with C.

pub mod ffi {
    use libc;
    use std::os::raw;

    pub type Vector = raw::c_void;

    pub type VectorAccum = raw::c_void;

    pub type VectorAccumType = raw::c_int;

    pub type VectorOperation = unsafe extern "C" fn(
        *mut raw::c_void,
        *mut VectorAccum,
        *const VectorAccum,
        libc::size_t,
        *mut *mut raw::c_double,
        libc::size_t,
    );

    pub type VectorDriverBase = raw::c_void;

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct VectorDriverVt {
        pub len: unsafe extern "C" fn(*const VectorDriverBase) -> libc::size_t,
        pub try_new: unsafe extern "C" fn(*const VectorDriverBase) -> *mut Vector,
        pub del: unsafe extern "C" fn(*const VectorDriverBase, *mut Vector),
        pub operate: unsafe extern "C" fn(
            *const VectorDriverBase,
            *mut VectorAccum,
            VectorAccumType,
            VectorOperation,
            *mut raw::c_void,
            libc::size_t,
            *mut *mut Vector,
            libc::size_t,
        ),
    }

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct VectorDriver {
        pub data: *const VectorDriverBase,
        pub vtable: *const VectorDriverVt,
    }
}

use super::super::utils::{abort_on_unwind, UnsafeSync};
use super::{assert_all_eq, VectorDriver};
use conv::ValueInto;
use libc;
use std::os::raw;
use std::{ptr, slice};

/// This is a reified vector driver that is compatible with C vector driver
/// interface.
#[derive(Clone, Copy, Debug)]
pub struct CVectorDriver<D> {
    vtable: ffi::VectorDriverVt,
    driver: D,
}

impl<D: VectorDriver<Item = f64> + Sized> CVectorDriver<D> {
    pub fn new(driver: D) -> Self {
        Self {
            vtable: ffi::VectorDriverVt {
                len: Self::len,
                try_new: Self::try_new,
                del: Self::del,
                operate: Self::operate,
            },
            driver,
        }
    }

    /// Warning: the returned driver is invalidated if `self` is moved!
    pub fn as_raw(&self) -> ffi::VectorDriver {
        ffi::VectorDriver {
            data: &self.driver as *const _ as _,
            vtable: &self.vtable,
        }
    }

    unsafe extern "C" fn len(d: *const ffi::VectorDriverBase) -> libc::size_t {
        abort_on_unwind(|| {
            let d = &*(d as *const D);
            d.len()
        })
    }

    unsafe extern "C" fn try_new(d: *const ffi::VectorDriverBase) -> *mut ffi::Vector {
        abort_on_unwind(|| {
            let d = &*(d as *const D);
            d.create_vector(0.0)
                .map(|p| Box::into_raw(Box::new(p)) as *mut _)
                .unwrap_or(ptr::null_mut())
        })
    }

    unsafe extern "C" fn del(_: *const ffi::VectorDriverBase, v: *mut ffi::Vector) {
        abort_on_unwind(|| {
            if !v.is_null() {
                Box::from_raw(v as *mut D::Vector);
            }
        })
    }

    unsafe extern "C" fn operate(
        d: *const ffi::VectorDriverBase,
        mut accum: *mut ffi::VectorAccum,
        accum_type: ffi::VectorAccumType,
        f: ffi::VectorOperation,
        f_ctx: *mut raw::c_void,
        offset: libc::size_t,
        mut vectors: *mut *mut ffi::Vector,
        num_vectors: libc::size_t,
    ) {
        abort_on_unwind(|| {
            let d = &*(d as *const D);
            let f_ctx = UnsafeSync::new(f_ctx);
            // avoid creating slices with null pointers (which would be bad)
            if num_vectors == 0 {
                vectors = [].as_mut_ptr();
            }
            if accum_type == 0 {
                accum = [].as_mut_ptr();
            }
            // note: this is a bit sketchy depending how Rust defines aliasing;
            // in the LLVM sense of aliasing, defined through memory
            // dependencies, this is valid
            let mut mut_vectors: Vec<_> = slice::from_raw_parts_mut(vectors, num_vectors)
                .into_iter()
                .map(|p| &mut *(*p as *mut D::Vector))
                .collect();
            if accum_type >= 0 {
                let accum_len = accum_type.value_into().expect("accum_len overflows usize");
                let accum = slice::from_raw_parts_mut(accum as *mut u8, accum_len);
                let mut accum_vec = accum.to_owned();
                d.operate_on(
                    &mut accum_vec,
                    offset,
                    &[],
                    &mut mut_vectors,
                    |accum, val, offset, slices, mut_slices| {
                        wrapper(
                            f,
                            f_ctx,
                            accum_len,
                            num_vectors,
                            accum,
                            val,
                            offset,
                            slices,
                            mut_slices,
                        )
                    },
                );
                accum.copy_from_slice(&accum_vec)
            } else {
                let accum_len = accum_type
                    .checked_neg()
                    .expect("negation of accum_type overflows c_int")
                    .value_into()
                    .expect("accum_len overflows usize");
                let accum = slice::from_raw_parts_mut(accum as *mut f64, accum_len);
                let mut accum_vec = accum.to_owned();
                d.operate_on(
                    &mut accum_vec,
                    offset,
                    &[],
                    &mut mut_vectors,
                    |accum, val, offset, slices, mut_slices| {
                        wrapper(
                            f,
                            f_ctx,
                            accum_len,
                            num_vectors,
                            accum,
                            val,
                            offset,
                            slices,
                            mut_slices,
                        )
                    },
                );
                accum.copy_from_slice(&accum_vec)
            }
        })
    }
}

fn wrapper<S>(
    f: ffi::VectorOperation,
    f_ctx: UnsafeSync<*mut raw::c_void>,
    accum_len: usize,
    num_vectors: usize,
    accum: &mut Vec<S>,
    val: Vec<S>,
    offset: usize,
    slices: &[&[f64]],
    mut_slices: &mut [&mut [f64]],
) {
    abort_on_unwind(|| {
        // needed for safety reasons
        assert!(accum.len() >= accum_len);
        assert!(val.len() >= accum_len);
        assert_eq!(mut_slices.len(), num_vectors);
        debug_assert_eq!(slices.len(), 0);
        let num_elems = assert_all_eq(
            mut_slices.iter().map(|s| s.len()),
            "every slice in mut_slices must be of equal length",
        )
        .unwrap_or(0);
        let mut mut_slices: Vec<_> = mut_slices.into_iter().map(|s| s.as_mut_ptr()).collect();
        unsafe {
            f(
                f_ctx.into_inner(),
                accum.as_mut_ptr() as *mut _,
                val.as_ptr() as *const _,
                offset,
                mut_slices.as_mut_ptr(),
                num_elems,
            );
        }
    })
}
