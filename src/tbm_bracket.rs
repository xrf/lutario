//! Talmi–Brody–Moshinsky bracket for 3D harmonic oscillators.
use rug::{Integer, Rational};
use wigner_symbols::{SignedSqrt, Wigner6j};
use wigner_symbols::internal::{factorial, falling_factorial, phase};

pub fn a_coeff(
    l1: i32,
    lr: i32,
    l2: i32,
    ls: i32,
    x: i32,
) -> Rational {
    panic!()
}

pub fn base_case(
    lambda: i32,
    nr: i32,
    lr: i32,
    ns: i32,
    ls: i32,
    l1: i32,
    l2: i32,
) -> SignedSqrt {
    let denom =
        (Integer::from(1) << (lr + ls))
        * falling_factorial(2 * l1, l1)
        * falling_factorial(2 * l2, l2)
        * falling_factorial(2 * nr + 2 * lr + 1, nr + lr)
        * falling_factorial(2 * ns + 2 * ls + 1, ns + ls)
        * factorial(nr)
        * factorial(ns);
    let x_start = 0; //FIXME
    let x_stop = 0; //FIXME
    let numer =
        Integer::from(phase(nr + lr + ls - lambda))
        * (2 * lr + 1)
        * (2 * ls + 1)
        * (x_start .. x_stop + 1).map(|x| {
            Rational::from((2 * x + 1) * phase(lr + ls + lambda + l2))
                * a_coeff(l1, lr, l2, ls, x)
                // PROBLEM: 6j -> SignedSqrt, can't add those
                * Wigner6j {
                    tj1: 2 * lr,
                    tj2: 2 * ls,
                    tj3: 2 * lambda,
                    tj4: 2 * l2,
                    tj5: 2 * l1,
                    tj6: 2 * x,
                }.value()
        }).sum();
    panic!()
}
