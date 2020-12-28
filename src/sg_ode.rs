//! Shampine-Gordon ODE solver.
//!
//! This module provides bindings to the [Shampine-Gordon ODE solver][sg-ode].
//!
//! [sg-ode]: https://github.com/xrf/sg-ode
//!
//! ## Example
//!
//! ```
//! use lutario::sg_ode;
//! use lutario::utils::Toler;
//! use lutario::vector_driver::basic::BasicVectorDriver;
//!
//! let driver = BasicVectorDriver::new(1);
//!
//! // create solver with modified tolerances
//! let mut solver = sg_ode::Conf {
//!     toler: Toler { abserr: 1e-9, relerr: 1e-9 },
//!     .. Default::default()
//! }.make_solver(&driver).unwrap();
//!
//! // initialize parameter (x) and vector (y)
//! let mut x = 0.0;
//! let mut y = vec![1.0];
//!
//! // solve for y at x = 1.0
//! let x_target = 1.0;
//! solver.step(|x, y, dydx| {
//!     dydx[0] = x.cos() * y[0];
//! }, x_target, &mut x, &mut y).unwrap();
//!
//! // check if the results are sensible
//! let y_exact = x.sin().exp();
//! assert_eq!(x, x_target);
//! // note: this only holds approximately!
//! assert!(solver.conf().toler.is_eq(y[0], y_exact));
//! ```

pub mod ffi {
    use std::os::raw;

    pub use super::super::vector_driver::c::ffi::{Vector, VectorDriver};

    pub type SgDerivFn =
        unsafe extern "C" fn(*mut raw::c_void, raw::c_double, *const Vector, *mut Vector);

    pub enum SgOde {}

    #[link(name = "sgode")]
    extern "C" {
        pub fn sg_ode_try_new(drv: VectorDriver) -> *mut SgOde;

        pub fn sg_ode_del(this: *mut SgOde);

        pub fn sg_ode_de(
            this: *mut SgOde,
            f: SgDerivFn,
            f_ctx: *mut raw::c_void,
            y: *mut Vector,
            t: *mut raw::c_double,
            tout: raw::c_double,
            relerr: *mut raw::c_double,
            abserr: *mut raw::c_double,
            iflag: *mut raw::c_int,
            maxnum: raw::c_uint,
        );
    }
}

use super::utils::{abort_on_unwind, Toler};
use super::vector_driver::c::CVectorDriver;
use super::vector_driver::VectorDriver;
use conv::ValueInto;
use std::marker::PhantomData;
use std::os::raw;

quick_error! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Error {
        /// Integration did not reach target because the error tolerances were
        /// too small to be feasible.  The tolerances have been re-adjusted to
        /// be more reasonable.  The user may re-attempt.
        ToleranceTooLow {}
        /// Integration did not reach target because too many (> maxnum) steps
        /// were taken.
        TooManySteps {}
        /// Integration did not reach target because the equations appear to
        /// be stiff.
        TooStiff {}
        InvalidArgument {}
        Unknown(err: i32) {}
    }
}

/// Can be constructed using `Conf::default()` or directly.
#[derive(Clone, Copy, Debug)]
pub struct Conf {
    pub toler: Toler,
    /// Normally, the integrator may overshoot the target for interpolation
    /// purposes.  To prevent this, set `strict` to `true`.
    pub strict: bool,
    pub maxnum: u32,
}

impl Default for Conf {
    fn default() -> Self {
        Conf {
            toler: Default::default(),
            strict: false,
            maxnum: 500,
        }
    }
}

impl Conf {
    pub fn make_solver<D>(self, driver: D) -> Option<Solver<D>>
    where
        D: VectorDriver<Item = f64> + Sized,
    {
        let driver = Box::new(CVectorDriver::new(driver));
        unsafe { ffi::sg_ode_try_new(driver.as_raw()).as_mut() }.map(|state| Solver {
            conf: self,
            driver: driver,
            state,
            iflag: if self.strict { -1 } else { 1 },
        })
    }
}

/// Can be constructed using `conf.make_solver()`.
#[derive(Debug)]
pub struct Solver<D> {
    conf: Conf,
    driver: Box<CVectorDriver<D>>,
    state: *mut ffi::SgOde,
    iflag: i32,
}

impl<D> Drop for Solver<D> {
    fn drop(&mut self) {
        unsafe {
            ffi::sg_ode_del(self.state);
        }
    }
}

fn parse_status(status: raw::c_int) -> Result<(), Error> {
    match status {
        2 => Ok(()),
        3 => Err(Error::ToleranceTooLow),
        4 => Err(Error::TooManySteps),
        5 => Err(Error::TooStiff),
        6 => Err(Error::InvalidArgument),
        e => Err(Error::Unknown(e.value_into().unwrap_or(0))),
    }
}

impl<D> Solver<D>
where
    D: VectorDriver<Item = f64> + Sized,
{
    pub fn conf(&self) -> &Conf {
        &self.conf
    }

    pub fn step<F>(
        &mut self,
        f: F,
        target_x: f64,
        x: &mut f64,
        y: &mut D::Vector,
    ) -> Result<(), Error>
    where
        F: FnMut(f64, &D::Vector, &mut D::Vector),
    {
        struct Ctx<D, F> {
            f: F,
            phantom: PhantomData<(D, fn(D))>,
        }

        impl<D, F> Ctx<D, F>
        where
            F: FnMut(f64, &D::Vector, &mut D::Vector),
            D: VectorDriver<Item = f64> + Sized,
        {
            fn new(f: F) -> (Self, ffi::SgDerivFn) {
                (
                    Self {
                        f,
                        phantom: PhantomData,
                    },
                    Self::call,
                )
            }

            unsafe extern "C" fn call(
                ctx: *mut raw::c_void,
                x: raw::c_double,
                y: *const ffi::Vector,
                dydx: *mut ffi::Vector,
            ) {
                let ctx = &mut *(ctx as *mut Self);
                let y = &*(y as *const D::Vector);
                let dydx = &mut *(dydx as *mut D::Vector);
                abort_on_unwind(|| {
                    (ctx.f)(x, y, dydx);
                });
            }
        }

        let (mut ctx, call) = Ctx::<D, F>::new(f);
        unsafe {
            ffi::sg_ode_de(
                self.state,
                call,
                &mut ctx as *mut _ as _,
                y as *mut _ as _,
                x,
                target_x,
                &mut self.conf.toler.relerr,
                &mut self.conf.toler.abserr,
                &mut self.iflag,
                self.conf.maxnum,
            );
            parse_status(self.iflag.abs())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::vector_driver::basic::BasicVectorDriver;
    use super::*;

    #[test]
    fn test_simple_harmonic_oscillator() {
        use std::f64::consts::PI;
        fn f(_: f64, y: &Vec<f64>, dydx: &mut Vec<f64>) {
            dydx[0] = y[1];
            dydx[1] = -y[0];
        }
        let drv = BasicVectorDriver::new(2);
        for direction in &[-1.0, 1.0] {
            let mut x = 0.0;
            let mut y = drv.create_vector_from(&[1.0, 0.0]);
            let mut solver = Conf::default().make_solver(&drv).unwrap();
            let n = 12;
            for i in 1..n + 1 {
                let n = n as f64;
                let target = 2.0 * PI / n * direction * i as f64;
                solver.step(f, target, &mut x, &mut y).unwrap();
                assert_eq!(x, target);
                toler_assert_eq!(
                    Toler {
                        abserr: solver.conf().toler.abserr * n,
                        relerr: 0.0,
                    },
                    y[0],
                    x.cos()
                );
            }
        }
    }

    #[test]
    fn test_enright_pryce_a3() {
        use std::f64::consts::PI;
        fn f(x: f64, y: &Vec<f64>, dydx: &mut Vec<f64>) {
            dydx[0] = x.cos() * y[0];
        }
        let drv = BasicVectorDriver::new(1);
        for direction in &[-1.0, 1.0] {
            let mut x = 0.0;
            let mut y = drv.create_vector_from(&[1.0]);
            let mut solver = Conf {
                toler: Toler {
                    abserr: 1e-8,
                    relerr: 1e-8,
                },
                ..Default::default()
            }
            .make_solver(&drv)
            .unwrap();
            let n = 12;
            for i in 1..n + 1 {
                let n = n as f64;
                let target = 2.0 * PI / n * direction * i as f64;
                solver.step(f, target, &mut x, &mut y).unwrap();
                assert_eq!(x, target);
                toler_assert_eq!(
                    Toler {
                        abserr: solver.conf().toler.abserr * n,
                        relerr: 0.0,
                    },
                    y[0],
                    x.sin().exp()
                );
            }
        }
    }

    #[test]
    fn test_stiff_springs() {
        // source: https://en.wikipedia.org/wiki/Stiff_equation#math_10
        fn f(_: f64, y: &Vec<f64>, dydx: &mut Vec<f64>) {
            dydx[0] = y[1];
            dydx[1] = -1000.0 * y[0] - 1001.0 * y[1];
        }
        let drv = BasicVectorDriver::new(2);
        let mut x = 0.0;
        let mut y = drv.create_vector_from(&[1.0, 0.0]);
        let mut solver = Conf {
            toler: Toler {
                abserr: 1e-8,
                relerr: 1e-8,
            },
            ..Default::default()
        }
        .make_solver(&drv)
        .unwrap();
        assert_eq!(solver.step(f, 1.0, &mut x, &mut y), Err(Error::TooStiff));
    }
}
