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
