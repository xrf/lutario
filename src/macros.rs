/// Force a mutable reference to be moved instead of reborrowed.
#[macro_export]
macro_rules! move_ref {
    ($e:expr) => {
        {
            let x = $e; x
        }
    }
}

/// Check whether two `f64` numbers are equal within the given
/// [`Toler`](utils/struct.Toler.html).
///
/// ```
/// #[macro_use]
/// extern crate lutario;
///
/// use lutario::utils::Toler;
///
/// fn main() {
///     toler_assert_eq!(Toler { abserr: 1e-2, relerr: 1e-3 }, 10.0, 10.02);
/// }
/// ```
#[macro_export]
macro_rules! toler_assert_eq {
    ($toler:expr, $left:expr, $right:expr) => {
        let toler = &$toler;
        let left = $left;
        let right = $right;
        assert!(toler.is_eq(left, right),
                "{} does not equal to {} within {:?}",
                left, right, toler)
    }
}

/// Declare a regular expression (`Regex`) cached via `lazy_static!`.
/// This macro is mainly for internal use.
///
/// ```
/// #[macro_use]
/// extern crate lazy_static;
/// #[macro_use]
/// extern crate lutario;
/// extern crate regex;
///
/// use regex::Regex;
///
/// fn main() {
///     let r: &Regex = re!(r"hello (\w+)");
/// }
/// ```
#[macro_export]
macro_rules! re {
    ($e:expr) => {
        {
            lazy_static! {
                static ref REGEX: Regex = Regex::new($e).unwrap();
            }
            &REGEX
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Vector driver

#[doc(hidden)]
#[macro_export]
macro_rules! _unsafe_vec_apply_bind {
    ($index:expr, $i:expr, $j:expr, $slices:expr, $mut_slices:expr, []) => {};
    ($index:expr, $i:expr, $j:expr, $slices:expr, $mut_slices:expr,
     [, $($vars:tt)*]) => {
        _unsafe_vec_apply_bind!($index, $i, $j, $slices, $mut_slices,
                                [$($vars)*])
    };
    ($index:expr, $i:expr, $j:expr, $slices:expr, $mut_slices:expr,
     [mut $var:ident $($vars:tt)*]) => {
        debug_assert!($j < $mut_slices.len());
        let $var = unsafe { &mut *($mut_slices
                                   .get_unchecked_mut($j)
                                   .get_unchecked_mut($index) as *mut _) };
        _unsafe_vec_apply_bind!($index, $i, $j + 1usize, $slices, $mut_slices,
                                [$($vars)*])
    };
    ($index:expr, $i:expr, $j:expr, $slices:expr, $mut_slices:expr,
     [$var:ident $($vars:tt)*]) => {
        debug_assert!($i < $slices.len());
        let $var = unsafe { $slices
                             .get_unchecked($i)
                             .get_unchecked($index) };
        _unsafe_vec_apply_bind!($index, $i + 1usize, $j, $slices, $mut_slices,
                                [$($vars)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vec_apply_reverse {
    ([ ] -> [$($body:tt)*]) => { [$($body)*] };
    ([$var:ident, $($vars:tt)*] -> [$($body:tt)*]) => {
        _vec_apply_reverse!([$($vars)*] -> [$var, $($body)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vec_apply_vectors {
    (( ) -> [$($body:tt)*]) => {
        _vec_apply_reverse!([$($body)*] -> [])
    };
    ((, $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors!(($($vars)*) -> [$($body)*])
    };
    ((mut $var:ident $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors!(($($vars)*) -> [$($body)*])
    };
    (($var:ident $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors!(($($vars)*) -> [$var, $($body)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vec_apply_vectors_mut {
    (( ) -> [$($body:tt)*]) => {
        _vec_apply_reverse!([$($body)*] -> [])
    };
    ((, $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors_mut!(($($vars)*) -> [$($body)*])
    };
    ((mut $var:ident $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors_mut!(($($vars)*) -> [$var, $($body)*])
    };
    (($var:ident $($vars:tt)*) -> [$($body:tt)*]) => {
        _vec_apply_vectors_mut!(($($vars)*) -> [$($body)*])
    };
}

/// Apply an applicative, multi-vector operation.
/// See [`vector_driver`](vector_driver/index.html).
///
/// ```
/// #[macro_use]
/// extern crate lutario;
///
/// use lutario::vector_driver::VectorDriver;
/// use lutario::vector_driver::basic::BasicVectorDriver;
///
/// fn main() {
///     let d = BasicVectorDriver::new(5);
///     let ref v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
///     let ref w = vec![2.0, 4.0, 1.0, 3.0, 5.0];
///     let ref mut u = d.create_vector(0.0).unwrap();
///     vec_apply! { for (v, w, mut u) in d { *u = *v - *w; } };
///     assert_eq!(u, &vec![-1.0, -2.0, 2.0, 1.0, 0.0]);
/// }
/// ```
#[macro_export]
macro_rules! vec_apply {
    { for ($($vars:tt)*) in $driver:tt $body:block } => {
        $driver.operate_on(
            &mut (),
            0,
            &_vec_apply_vectors!(($($vars)*) -> []),
            &mut _vec_apply_vectors_mut!(($($vars)*) -> []),
            |_: &mut _, _, _, slices: &[&[_]], mut_slices: &mut [&mut [_]]| {
                let n = $crate::vector_driver::assert_all_eq(
                    slices.iter()
                        .map(|s| s.len())
                        .chain(mut_slices.iter().map(|s| s.len())),
                    "all slices must have equal lengths").unwrap_or(0);
                for i in 0 .. n {
                    _unsafe_vec_apply_bind!(i, 0, 0, slices, mut_slices,
                                            [$($vars)*]);
                    $body
                }
            });
    };
    { for mut $var:ident in $driver:tt $body:block } => {
        vec_apply! { for (mut $var) in $driver $body }
    };
    { for $var:ident in $driver:tt $body:block } => {
        vec_apply! { for ($var) in $driver $body }
    };
}
