/// Force a mutable reference to be moved instead of reborrowed.
#[macro_export]
macro_rules! move_ref {
    ($e:expr) => {
        {
            let x = $e; x
        }
    }
}

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

/// Declare a regular expression cached via `lazy_static!`.
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
        debug_assert!($i < $mut_slices.len());
        let $var = unsafe { $slices
                             .get_unchecked($i)
                             .get_unchecked($index) };
        _unsafe_vec_apply_bind!($index, $i + 1usize, $j, $slices, $mut_slices,
                                [$($vars)*])
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _vec_apply_vectors {
    (( ) -> [$($body:tt)*]) => { [$($body)*] };
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
    (( ) -> [$($body:tt)*]) => { [$($body)*] };
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
